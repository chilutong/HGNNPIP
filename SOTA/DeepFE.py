# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 20:07:38 2018

@author: yaoyu
"""

from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from sklearn.metrics import roc_curve, auc, roc_auc_score,average_precision_score
import numpy as np
from keras.layers.core import Dense, Dropout, Merge
import utils.tools as utils
from keras.regularizers import l2
import pandas as pd
from gensim.models.word2vec import Word2Vec
import copy
import h5py
from sklearn.model_selection import StratifiedKFold
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
from keras import backend as K
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.optimizers import SGD
import psutil
import os
from time import time


def averagenum(num):
    nsum = 0
    for i in range(len(num)):
        nsum += num[i]
    return nsum / len(num)
def plot(length):
    reversed_length =  sorted(length,reverse=True)
    x = np.linspace(0, len(length), len(length))
    plt.plot(x, reversed_length)
     
    plt.title('line chart')
    plt.xlabel('x')
    plt.ylabel('reversed_length')
    
    plt.show()  
    
   
def max_min_avg_length(seq):
    length = []
    for string in seq:
        length.append(len(string))
    plot(length)    
    maxNum = max(length) #maxNum = 5
    minNum = min(length) #minNum = 1
    
    avg = averagenum(length)
    
    print('The longest length of protein is: '+str(maxNum))
    print('The shortest length of protein is: '+str(minNum))
    print('The avgest length of protein is: '+str(avg))
    


def token(dataset):
    token_dataset = []
    for i in range(len(dataset)):
        seq = []
        for j in range(len(dataset[i])):
            seq.append(dataset[i][j])
        token_dataset.append(seq)  
        
    return  token_dataset

def connect(protein_A,protein_B):
    # contect protein A and B
    protein_AB = []
    for i in range(len(protein_A)):
        con = protein_A[i] + protein_B[i] 
        protein_AB.append(con)
        
    return np.array(protein_AB)
    
def read_pos_protein_pair(file_protein_A,file_protein_B):
    pos_protein_A = []
    with open(file_protein_A, 'r') as fp:
        i = 0
        for line in fp:
            if i%2==1:
                pos_protein_A.append(line.split('\n')[0])
            i = i+1            
            
    pos_protein_B = []
    with open(file_protein_B, 'r') as fp:
        i = 0
        for line in fp:
            if i%2==1:
                pos_protein_B.append(line.split('\n')[0])
            i = i+1       
    
    # contect protein A and B
    pos_protein_AB = connect(pos_protein_A,pos_protein_B )
          
    return pos_protein_AB
 

def read_human_and_hpylori_seq(file_name_human,posnum,negnum):
    protein = pd.read_csv(file_name_human)  
    
    pos_protein_A=protein.iloc[0:posnum,:] 
    pos_protein_B=protein.iloc[posnum:posnum*2,:]  
    seq_pos_protein_A = pos_protein_A['seq'].tolist()
    seq_pos_protein_B = pos_protein_B['seq'].tolist()
    neg_protein_A=protein.iloc[posnum*2:posnum*2+negnum,:]
    neg_protein_B=protein.iloc[posnum*2+negnum:posnum*2+negnum*2,:]
    seq_neg_protein_A = neg_protein_A['seq'].tolist()
    seq_neg_protein_B = neg_protein_B['seq'].tolist()
    seq = []
    seq.extend(seq_pos_protein_A)
    seq.extend(seq_pos_protein_B)
    seq.extend(seq_neg_protein_A)
    seq.extend(seq_neg_protein_B)
    max_min_avg_length(seq)
    return   seq_pos_protein_A, seq_pos_protein_B, seq_neg_protein_A, seq_neg_protein_B
    

#%% 
def merged_DBN(sequence_len):
    # left model
    model_left = Sequential()
    model_left.add(Dense(2048, input_dim=sequence_len ,activation='relu',W_regularizer=l2(0.01)))
    model_left.add(BatchNormalization())
    model_left.add(Dropout(0.5))
    model_left.add(Dense(1024, activation='relu',W_regularizer=l2(0.01)))
    model_left.add(BatchNormalization())
    model_left.add(Dropout(0.5))
    model_left.add(Dense(512, activation='relu',W_regularizer=l2(0.01)))
    model_left.add(BatchNormalization())
    model_left.add(Dropout(0.5))
    model_left.add(Dense(128, activation='relu',W_regularizer=l2(0.01)))
    model_left.add(BatchNormalization())
   
    # right model
    model_right = Sequential()
    model_right.add(Dense(2048,input_dim=sequence_len,activation='relu',W_regularizer=l2(0.01)))
    model_right.add(BatchNormalization())
    model_right.add(Dropout(0.5))   
    model_right.add(Dense(1024, activation='relu',W_regularizer=l2(0.01)))
    model_right.add(BatchNormalization())
    model_right.add(Dropout(0.5))
    model_right.add(Dense(512, activation='relu',W_regularizer=l2(0.01)))
    model_right.add(BatchNormalization())
    model_right.add(Dropout(0.5))
    model_right.add(Dense(128, activation='relu',W_regularizer=l2(0.01)))
    model_right.add(BatchNormalization())
    # together
    merged = Merge([model_left, model_right])
      
    model = Sequential()
    model.add(merged)
    model.add(Dense(8, activation='relu',W_regularizer=l2(0.01)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))
    #model.summary()
    
    return model
#%%

def pandding_J(protein,maxlen):           
    padded_protein = copy.deepcopy(protein)   
    for i in range(len(padded_protein)):
        if len(padded_protein[i])<maxlen:
            for j in range(len(padded_protein[i]),maxlen):
                padded_protein[i]=padded_protein[i]+'J'
    return padded_protein  
       

def residue_representation(wv,tokened_seq_protein,maxlen,size):  
    represented_protein  = []
    for i in range(len(tokened_seq_protein)):
        temp_sentence = []
        for j in range(maxlen):
            if tokened_seq_protein[i][j]=='J':
                temp_sentence.extend(np.zeros(size))
            else:
                temp_sentence.extend(wv[tokened_seq_protein[i][j]])
        represented_protein.append(np.array(temp_sentence))    
   
    return np.array(represented_protein)
    

        
def protein_reprsentation(wv,pos_protein_A,pos_protein_B,neg_protein_A,neg_protein_B,maxlen,size):
    # put positive and negative samples together
    pos_neg_protein_A = copy.deepcopy(pos_protein_A)   
    pos_neg_protein_A.extend(neg_protein_A)
    pos_neg_protein_B = copy.deepcopy(pos_protein_B)   
    pos_neg_protein_B.extend(neg_protein_B)
    
    # padding
    padded_pos_neg_protein_A = pandding_J(pos_neg_protein_A,maxlen)   
    padded_pos_neg_protein_B = pandding_J(pos_neg_protein_B,maxlen)   
                
    # token 
    token_padded_pos_neg_protein_A = token(padded_pos_neg_protein_A) 
    token_padded_pos_neg_protein_B = token(padded_pos_neg_protein_B)
                   
    # generate feature of pair A
    feature_protein_A = residue_representation(wv,token_padded_pos_neg_protein_A,maxlen,size )
    feature_protein_B = residue_representation(wv,token_padded_pos_neg_protein_B,maxlen,size )
    
    feature_protein_AB = np.hstack((np.array(feature_protein_A),np.array(feature_protein_B)))
    
    return feature_protein_AB


    
def human_data_processing(wv,maxlen,size):
    # get hpylori sequences
    file_name_human = 'dataset/human/human_protein.csv'
    pos_human_pair_A,pos_human_pair_B,neg_human_pair_A,neg_human_pair_B = read_human_and_hpylori_seq(file_name_human,3899,4262)
    
    feature_protein_AB = protein_reprsentation(wv, pos_human_pair_A,pos_human_pair_B,neg_human_pair_A,neg_human_pair_B,maxlen,size)
  
    # creat label
    label = np.ones(len(pos_human_pair_A)+len(neg_human_pair_A))
    
    label[len(pos_human_pair_A):] = 0

    
    return feature_protein_AB,label
    
# define the function
def training_vis(hist,i,plot_dir,swm,be):
    loss = hist.history['loss']
    #val_loss = hist.history['val_loss']
    precision = hist.history['precision']
    #val_acc = hist.history['val_acc']

    # make a figure
    fig = plt.figure(figsize=(8,4))
    # subplot loss
    ax1 = fig.add_subplot(121)
    ax1.plot(loss,label='train_loss')
    #ax1.plot(val_loss,label='val_loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss on Traingng Data')
    ax1.legend()
    # subplot acc
    ax2 = fig.add_subplot(122)
    ax2.plot(precision,label='train_precision')
    #ax2.plot(val_acc,label='val_acc')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision on Traingng Data')
    ax2.legend()
    plt.tight_layout()
    plt.savefig(plot_dir + swm+be+'/round_'+str(i)+'.png')
    
    
def mkdir(path):
    folder = os.path.exists(path)
    if not folder:                 
        os.makedirs(path)           
        print("---  new folder...  ---")
        print("---  OK  ---")
 
    else:
        print("---  There is this folder!  ---")
        
def getMemorystate():   
    phymem = psutil.virtual_memory()   
    line = "Memory: %5s%% %6s/%s"%(phymem.percent,
                                   str(int(phymem.used/1024/1024))+"M",
                                   str(int(phymem.total/1024/1024))+"M") 
    return line        
        
#%%    
if __name__ == "__main__":  
    # load dictionary
    model_wv = Word2Vec.load('model/word2vec/wv_swissProt_size_20_window_4.model')
#    runInfo_dir= 'runInfo/human/'
#    mkdir(runInfo_dir)  
    plot_dir = 'plot/human/'
       
    sizes = [20]
    windows = [4]
    maxlens = [700,800,900]
    batch_sizes = [32,64,128,256]
    nb_epoches = [35,45]
    for size in sizes:
        for window in windows:
            for maxlen in maxlens:
                for batch_size in batch_sizes:
                    for nb_epoch in nb_epoches:
                         
                        sequence_len = size*maxlen   
                        # get training data 
                        
                        train_fea_protein_AB,train_label = human_data_processing(model_wv.wv, maxlen,size)
                     
                        print('dataset is represented')
                        swm = 'swissProt_size_'+str(size)+'_window_'+str(window)+'_maxlen_'+str(maxlen) 
                                               
                        # StandardScaler
                        scaler = StandardScaler().fit(train_fea_protein_AB)
                        train_fea_protein_AB = scaler.transform(train_fea_protein_AB)   
                        
                        db_dir= 'dataset/human/different size represented data/size_'+str(size)
                        mkdir(db_dir)            
                        # creat HDF5 file
                        h5_file = h5py.File(db_dir + '/'+swm+'.h5','w')
                        h5_file.create_dataset('trainset_x', data = train_fea_protein_AB)
                        h5_file.create_dataset('trainset_y', data = train_label)
                        h5_file.close()
                                                 
                        fea_protein_A = train_fea_protein_AB[:,0:sequence_len]
                        fea_protein_B = train_fea_protein_AB[:,sequence_len:sequence_len*2]
                                              
                        i = 0 
                        scores = []  
                       
                        be = '_batch_size_'+str(batch_size)+'_nb_epoch_'+str(nb_epoch)
                        model_dir = 'model/dl/human/'
                        result_dir = 'result/5cv/human/'
                        mkdir(result_dir) 
                        
                     
                        # 5cv
                        skf = StratifiedKFold(n_splits = 5,random_state= 20181106,shuffle= True)
                        Y = utils.to_categorical(train_label)  
                       
                        for (train_index, test_index) in skf.split(train_fea_protein_AB,train_label):
                            print("================")
                            X_train_left = fea_protein_A[train_index]
                            X_train_right = fea_protein_B[train_index]
                            X_test_left = fea_protein_A[test_index]
                            X_test_right = fea_protein_B[test_index]
                    
                            X_train_left  = np.array(X_train_left)
                            X_train_right  = np.array(X_train_right)
                            
                            X_test_left  = np.array(X_test_left)
                            X_test_right  = np.array(X_test_right)
                          
                            y_train = Y[train_index]
                            y_test = Y[test_index]
                           
                            # print("================")
                            model =  merged_DBN(sequence_len)    
                            sgd = SGD(lr=0.01, momentum=0.9, decay=0.001)
                           
                            model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['precision'])
                            #model.compile(loss='categorical_crossentropy', optimizer='rmsprop',metrics=['accuracy'])
                            hist = model.fit([X_train_left, X_train_right], y_train,
                                      batch_size = batch_size,
                                      nb_epoch = nb_epoch,
                                      verbose = 1) 
                           
                            
                            print('******   model created!  ******')
                            mkdir(model_dir + swm+be+'/')
                            mkdir(plot_dir + swm+be+'/')
                            training_vis(hist,i,plot_dir,swm,be)
                            model.save(model_dir + swm+be+'/round_'+str(i)+'.h5')
                            predictions_test = model.predict([X_test_left, X_test_right])  
                            
                            auc_test = roc_auc_score(y_test[:,1], predictions_test[:,1])
                            pr_test = average_precision_score(y_test[:,1], predictions_test[:,1])
                         
                            label_predict_test = utils.categorical_probas_to_classes(predictions_test)  
                            tp_test,fp_test,tn_test,fn_test,accuracy_test, precision_test, sensitivity_test,recall_test, specificity_test, MCC_test, f1_score_test,_,_,_= utils.calculate_performace(len(label_predict_test), label_predict_test, y_test[:,1])
                            print('test:'+str(i))
                            print('\ttp=%0.0f,fp=%0.0f,tn=%0.0f,fn=%0.0f'%(tp_test,fp_test,tn_test,fn_test))
                            print('\tacc=%0.4f,pre=%0.4f,rec=%0.4f,sp=%0.4f,mcc=%0.4f,f1=%0.4f'
                                  % (accuracy_test, precision_test, recall_test, specificity_test, MCC_test, f1_score_test))
                            print('\tauc=%0.4f,pr=%0.4f'%(auc_test,pr_test))
                            scores.append([accuracy_test,precision_test, recall_test,specificity_test, MCC_test, f1_score_test, auc_test,pr_test]) 
                    
                            i=i+1
                            K.clear_session()
                            tf.reset_default_graph()
                            
                            
                        sc= pd.DataFrame(scores)   
                        sc.to_csv(result_dir+swm+be+'.csv')   
                        scores_array = np.array(scores)
                        print (swm+be+'_5cv:')
                        print(("accuracy=%.2f%% (+/- %.2f%%)" % (np.mean(scores_array, axis=0)[0]*100,np.std(scores_array, axis=0)[0]*100)))
                        print(("precision=%.2f%% (+/- %.2f%%)" % (np.mean(scores_array, axis=0)[1]*100,np.std(scores_array, axis=0)[1]*100)))
                        print("recall=%.2f%% (+/- %.2f%%)" % (np.mean(scores_array, axis=0)[2]*100,np.std(scores_array, axis=0)[2]*100))
                        print("specificity=%.2f%% (+/- %.2f%%)" % (np.mean(scores_array, axis=0)[3]*100,np.std(scores_array, axis=0)[3]*100))
                        print("MCC=%.2f%% (+/- %.2f%%)" % (np.mean(scores_array, axis=0)[4]*100,np.std(scores_array, axis=0)[4]*100))
                        print("f1_score=%.2f%% (+/- %.2f%%)" % (np.mean(scores_array, axis=0)[5]*100,np.std(scores_array, axis=0)[5]*100))
                        print("roc_auc=%.2f%% (+/- %.2f%%)" % (np.mean(scores_array, axis=0)[6]*100,np.std(scores_array, axis=0)[6]*100))
                        print("roc_pr=%.2f%% (+/- %.2f%%)" % (np.mean(scores_array, axis=0)[7]*100,np.std(scores_array, axis=0)[7]*100))
                       
                      
                        # memory and time for classify
                      
                       
                        with open(result_dir+'5cv_'+swm+be+'.txt','w') as f:
                            f.write('accuracy=%.2f%% (+/- %.2f%%)' % (np.mean(scores_array, axis=0)[0]*100,np.std(scores_array, axis=0)[0]*100))
                            f.write('\n')
                            f.write("precision=%.2f%% (+/- %.2f%%)" % (np.mean(scores_array, axis=0)[1]*100,np.std(scores_array, axis=0)[1]*100))
                            f.write('\n')
                            f.write("recall=%.2f%% (+/- %.2f%%)" % (np.mean(scores_array, axis=0)[2]*100,np.std(scores_array, axis=0)[2]*100))
                            f.write('\n')
                            f.write("specificity=%.2f%% (+/- %.2f%%)" % (np.mean(scores_array, axis=0)[3]*100,np.std(scores_array, axis=0)[3]*100))
                            f.write('\n')
                            f.write("MCC=%.2f%% (+/- %.2f%%)" % (np.mean(scores_array, axis=0)[4]*100,np.std(scores_array, axis=0)[4]*100))
                            f.write('\n')
                            f.write("f1_score=%.2f%% (+/- %.2f%%)" % (np.mean(scores_array, axis=0)[5]*100,np.std(scores_array, axis=0)[5]*100))
                            f.write('\n')
                            f.write("roc_auc=%.2f%% (+/- %.2f%%)" % (np.mean(scores_array, axis=0)[6]*100,np.std(scores_array, axis=0)[6]*100))
                            f.write('\n')
                            f.write("roc_pr=%.2f%% (+/- %.2f%%)" % (np.mean(scores_array, axis=0)[7]*100,np.std(scores_array, axis=0)[7]*100))
                            
                            f.write('\n')
                            f.write('\n')
                        
                        