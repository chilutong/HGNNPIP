from __future__ import division
from __future__ import print_function
import pandas as pd
import os
import glob
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from MLP import MLPmodel
from torch.autograd import Variable
from DNN import DNNmodel
from utils_fp import load_data, accuracy, metric
from GNN import GNNmodel

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--predictList',default='/data/predict/prediclist.csv', help='predictList.')
parser.add_argument('--dataset', type=str, default='Oryza', help='the dataset')
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=20230101, help='Random seed.')
parser.add_argument('--epochs', type=int, default=1500, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=128, help='Number of hidden units.')
parser.add_argument('--nb_heads', type=int, default=1, help='Number of head attentions.')
parser.add_argument('--dropout', type=float, default=0.4, help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=100, help='Patience')
parser.add_argument('--threshold', type=float, default=0.1)
parser.add_argument('--outdim', type=int, default=32)
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
adj, features, seq_features, labels, idx_train, idx_test, net, prelist = load_data(dataset=args.dataset)

# Model and optimizer

gnn = GNNmodel(nfeat=features.shape[1],
            nproteins=features.shape[0],
            nhid=args.hidden,
            nclass=args.outdim,
            dropout=args.dropout,
            nheads=args.nb_heads,
            alpha=args.alpha)

layers = [seq_features.shape[1], 512, 256, 128, 64, 32]
dnn = DNNmodel(layers)

num_proteins = adj.shape[0]
mlp_indim = args.outdim * 2 + 32 * 2
mlp_outdim = 16
layers = []
temp = mlp_indim
layers.append(temp)

while temp > mlp_outdim:
    temp = temp / 2
    layers.append(int(temp))
mlp = MLPmodel(num_proteins, num_proteins, layers)

optimizer = torch.optim.Adam([{'params': gnn.parameters(), 'weight_decay': args.weight_decay},
                              {'params': mlp.parameters(), 'weight_decay': args.weight_decay},
                              {'params': dnn.parameters(), 'weight_decay': args.weight_decay}],
                             lr=args.lr)

if args.cuda:
    gnn.cuda()
    dnn.cuda()
    prelist = prelist.cuda()
    features = features.cuda()
    seq_features = seq_features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_test = idx_test.cuda()
    mlp.cuda()
# features, adj, labels,seq_features = Variable(features), Variable(adj), Variable(labels), Variable(seq_features)


def train(epoch):
    t = time.time()
    gnn.train()
    dnn.train()
    mlp.train()

    seq_features1 = dnn(seq_features)
    output = gnn(features, adj)
    final_emb = torch.cat((seq_features1, output), dim=-1)
    loss_func = nn.BCELoss(size_average=False, reduce=True)

    predictions = mlp(final_emb[net[idx_train][:, 0]], final_emb[net[idx_train][:, 1]])
    optimizer.zero_grad()
    loss_train = loss_func(predictions.reshape(predictions.shape[0], ).to(torch.float32),
                           labels[idx_train].to(torch.float32))
    acc_train = accuracy(predictions.reshape(predictions.shape[0], ), labels[idx_train],
                         args.threshold)
    loss_train.backward()
    optimizer.step()

    print('Epoch: {:04d}'.format(epoch),
          'loss_train: {:.4f}'.format(loss_train.data.item()),
          'acc_train: {:.4f}'.format(acc_train.data.item()),

          'time: {:.4f}s'.format(time.time() - t))

    # return loss_val.data.item()
    return loss_train.data.item()


def ealy_train():
    # # Train model
    t_total = time.time()
    loss_values = []
    bad_counter = 0
    # best = args.epochs + 1
    best = 10000000
    best_epoch = 0
    for epoch in range(args.epochs):
        loss_values.append(train(epoch))

        torch.save(gnn.state_dict(), '{}.gnn.pkl'.format(epoch))
        torch.save(mlp.state_dict(), '{}.mlp.pkl'.format(epoch))
        torch.save(dnn.state_dict(), '{}.dnn.pkl'.format(epoch))
        if loss_values[-1] < best - 0.001:
            best = loss_values[-1]
            best_epoch = epoch
            bad_counter = 0
            print(best_epoch)
        else:
            bad_counter += 1

        if bad_counter == args.patience:
            break

        files = glob.glob('*.pkl')
        for file in files:
            epoch_nb = int(file.split('.')[0])
            if epoch_nb < best_epoch:
                os.remove(file)

    files = glob.glob('*.pkl')
    for file in files:
        epoch_nb = int(file.split('.')[0])
        if epoch_nb > best_epoch:
            os.remove(file)

    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    # Restore best model
    print('Loading {}th epoch'.format(best_epoch))
    gnn.load_state_dict(torch.load('{}.gnn.pkl'.format(best_epoch)))
    mlp.load_state_dict(torch.load('{}.mlp.pkl'.format(best_epoch)))
    dnn.load_state_dict(torch.load('{}.dnn.pkl'.format(best_epoch)))
    #
    gnn.eval()
    mlp.eval()
    dnn.eval()
    with torch.no_grad():
        seq_features1 = dnn(seq_features)
        output = gnn(features, adj)
        final_emb = torch.cat((seq_features1, output), dim=-1)

        #测试模型精度
        predictions = mlp(final_emb[net[idx_test][:, 0]],final_emb[net[idx_test][:, 1]])
        loss_func = nn.BCELoss(size_average=False, reduce=True)
        loss_test = loss_func(predictions.reshape(predictions.shape[0], ).to(torch.float32),
                              labels[idx_test].to(torch.float32))
        acc_test = accuracy(predictions.reshape(predictions.shape[0], ), labels[idx_test],
                            args.threshold)
        precision, recall, f1, mcc, roc, ap = metric(predictions.reshape(predictions.shape[0], ),
                                                     labels[idx_test], args.threshold)
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.data.item()),
              "accuracy= {:.4f}".format(acc_test.data.item()),
              "precision= {:.4f}".format(precision),
              "recall= {:.4f}".format(recall),
              "F1-score= {:.4f}".format(f1),
              "mcc= {:.4f}".format(mcc),
              "AUC-ROC= {:.4f}".format(roc),
              "AP= {:.4f}".format(ap),
              )



        #预测
        dataset = './data/'+args.dataset
        predictPath = args.predictList
        prelist = pd.read_csv(predictPath, sep=',', header=None).values
        # dataset='./data/S.cere'
        idx2uniprot = np.loadtxt(dataset+'/uniprot', dtype='str')[:,1]
        predictions = mlp(final_emb[prelist[:, 0]], final_emb[prelist[:, 1]]).cpu().detach().numpy()

        list=prelist.cpu().detach().numpy()

        result = np.column_stack((list,idx2uniprot[list[:,0]],idx2uniprot[list[:,1]],predictions))
        # result = np.append(list, predictions, axis=1)

        result = result[np.argsort(result[:, -1])[::-1]]
        result = pd.DataFrame(result)
        result.to_csv(dataset+'/predict_result.csv', index=False, header=False)


ealy_train()
