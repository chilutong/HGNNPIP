from __future__ import division
from __future__ import print_function

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
from DNN import DNNmodel
from GNN import GNNmodel
from torch.autograd import Variable

from cv_load import parse_data,load_data, accuracy, metric


# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='S.cere', help='the dataset')
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=20221231, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=128, help='Number of hidden units.')
parser.add_argument('--nb_heads', type=int, default=1, help='Number of head attentions.')
parser.add_argument('--dropout', type=float, default=0.4, help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=150, help='Patience')
parser.add_argument('--threshold', type=float, default=0.5)
parser.add_argument('--outdim', type=int, default=32)
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Train the model
def train(epoch):
    t = time.time()

    gnn.train()
    mlp.train()
    dnn.train()
    # Extract the sequence features by dnn
    seq_features1 = dnn(seq_features)
    # Extract the topological features by gnn
    output = gnn(features, adj)
    # Merge the sequence and topological features
    final_emb=torch.cat((seq_features1, output), dim=-1)
    # Set the loss function
    loss_func = nn.BCELoss(size_average=False, reduce=True)
    # predict by MLP classifier
    predictions = mlp(final_emb[net[idx_train][:, 0]],final_emb[net[idx_train][:, 1]])

    # train
    optimizer.zero_grad()
    loss_train = loss_func(predictions.reshape(predictions.shape[0], ).to(torch.float32), labels[idx_train].to(torch.float32))
    acc_train = accuracy(predictions.reshape(predictions.shape[0], ), labels[idx_train],
                         args.threshold)
    loss_train.backward()
    optimizer.step()
    # Output the loss and acc of each epoch
    print('Epoch: {:04d}'.format(epoch),
          'loss_train: {:.4f}'.format(loss_train.data.item()),
          'acc_train: {:.4f}'.format(acc_train.data.item()),
          'time: {:.4f}s'.format(time.time() - t))


    return loss_train.data.item()

# Test the model
def compute_test():
    gnn.eval()
    mlp.eval()
    dnn.eval()
    # Extract the sequence features by dnn
    seq_features1 = dnn(seq_features)
    # Extract the topological features by gnn
    output = gnn(features, adj)
    # Merge the sequence and topological features
    final_emb=torch.cat((seq_features1, output), dim=-1)
    # predict by MLP classifier
    predictions = mlp(final_emb[net[idx_test][:, 0]],final_emb[net[idx_test][:, 1]])
    # Set the loss function
    loss_func = nn.BCELoss(size_average=False, reduce=True)
    # Test
    loss_test = loss_func(predictions.reshape(predictions.shape[0], ).to(torch.float32), labels[idx_test].to(torch.float32))
    acc_test = accuracy(predictions.reshape(predictions.shape[0], ), labels[idx_test],
                        args.threshold)
    precision, recall,spec, f1, mcc, roc, ap = metric(predictions.reshape(predictions.shape[0], ),
                                                 labels[idx_test], args.threshold)

    # Output the result
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.data.item()),
          "accuracy= {:.4f}".format(acc_test.data.item()),
          "precision= {:.4f}".format(precision),
          "recall= {:.4f}".format(recall),
          "specific= {:.4f}".format(spec),
          "F1-score= {:.4f}".format(f1),
          "mcc= {:.4f}".format(mcc),
          "AUC-ROC= {:.4f}".format(roc),
          "AP= {:.4f}".format(ap),
          )
    return np.array([round(acc_test.data.item(), 3), round(precision, 3), round(recall, 3),round(spec, 3), round(f1, 3), round(mcc, 3),
                     round(roc, 3), round(ap, 3)])

# The early stop function to avoid overfit
def early_train():
    # Train model
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
        if loss_values[-1] < best:
            best = loss_values[-1]-0.001
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

    # # Restore best model
    print('Loading {}th epoch'.format(best_epoch))
    gnn.load_state_dict(torch.load('{}.gnn.pkl'.format(best_epoch)))
    mlp.load_state_dict(torch.load('{}.mlp.pkl'.format(best_epoch)))
    dnn.load_state_dict(torch.load('{}.dnn.pkl'.format(best_epoch)))

# Set the fold of Cross-Validation
k = 5

all_metric = np.zeros(8, dtype=float)
all_acc = []
metrics = []
accuracys = []
# Parse the PPI dataset
idx_features,net_labels=parse_data(args.dataset,6,150)
# Each fold of the Cross-Validation
for i in range(k):
    #Load the PPI dataset
    adj, features, seq_features,labels, idx_train, idx_test, net = load_data(i,idx_features,net_labels)
    print('第i折', i)
    # Initial the gnn
    gnn = GNNmodel(nfeat=features.shape[1],
                nproteins=features.shape[0],
                nhid=args.hidden,
                nclass=args.outdim,
                dropout=args.dropout,
                nheads=args.nb_heads,
                alpha=args.alpha)
    # Initial the dnn
    layers=[seq_features.shape[1],512,256,128,64,32]
    dnn =DNNmodel(layers)
    num_proteins = adj.shape[0]
    # Initial the mlp
    mlp_indim=args.outdim*2+32*2
    mlp_outdim=16
    layers = []
    temp=mlp_indim
    layers.append(temp)
    while temp>mlp_outdim :
        temp=temp/2
        layers.append(int(temp))
    mlp = MLPmodel(layers)
    # Set the optimizer
    optimizer = torch.optim.Adam([{'params': gnn.parameters(), 'weight_decay': args.weight_decay},
                                   {'params': mlp.parameters(),'weight_decay': args.weight_decay},
                                 {'params': dnn.parameters(),'weight_decay': args.weight_decay}],
                                  lr=args.lr)

    if args.cuda:
        gnn.cuda()
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        seq_features = seq_features.cuda()
        idx_test = idx_test.cuda()
        mlp.cuda()
        dnn.cuda()
    # features, adj, labels = Variable(features), Variable(adj), Variable(labels)
    early_train()
    res = compute_test()
    files = glob.glob('*.pkl')
    for file in files:
        os.remove(file)

    all_metric = all_metric + res
    mean_metric = all_metric / 5
    metrics.append(res)

# output the result
print(all_metric)
print(metrics)
print(mean_metric)
