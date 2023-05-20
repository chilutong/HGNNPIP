# import re
import numpy as np
from sklearn.decomposition import PCA
# import gensim
from gensim.models import Word2Vec
from sklearn.manifold import  TSNE
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd




def sequenceEmbedding(dataset, r_dim, length):
    print('Embedding sequence')
    f = np.loadtxt(dataset + 'sequenceList.txt', dtype='str')
    lines = []
    for line in f:  # 分别对每段分词
        words = []

        for aa in line:
            words.append(aa)
        lines.append(words)

    model = Word2Vec(lines, vector_size=20, window=4, min_count=0, epochs=20, negative=5, sg=1)
    # model = Word2Vec.load('wv_swissProt_size_20_window_4.model')

    # TSNE_model = TSNE(perplexity=1,n_components=3,init='pca',random_state=23,n_iter=2500)
    # print("F：\n",model.wv.get_vector('F'))
    # print(model.wv.most_similar('F', topn = 3))

    rawWordVec = []
    word2ind = {}
    for i, w in enumerate(model.wv.index_to_key):  # index_to_key 序号,词语
        
        rawWordVec.append(model.wv[w])  # 词向量
        word2ind[w] = i  # {词语:序号}
    rawWordVec = np.array(rawWordVec)
    X_reduced = PCA(n_components=r_dim).fit_transform(rawWordVec)
    # X_reduced = TSNE_model.fit_transform(rawWordVec)
    # print(1)
    features = []

    # print(1)
    for line in lines:
        pre = line[0:length]
        post = line[::-1][0:length]
        count = 0
        feature = []
        for aa in pre:
            if count  < length:
                aa_vector = X_reduced[word2ind[aa]]
                feature.extend(list(aa_vector))
                count = count + 1
            else:
                break
        if (count < length):
            feature += [0] * (length - count)*r_dim
        count=0
        for aa in post:
            if count  < length:
                aa_vector = X_reduced[word2ind[aa]]
                feature.extend(list(aa_vector))
                count = count + 1
            else:
                break
        if (count < length):
            feature += [0] * (length - count)*r_dim
        aa_vector=[]
        features.append(feature)
    features = pd.DataFrame(features)
    features.to_csv(dataset + 'word2vec_features.csv', index=True, header=False)
    print('finish Embedding')


