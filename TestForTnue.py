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
from torch.autograd import Variable

from cv_load import parse_data, load_data, accuracy, metric
from GNN import GNNmodel


def test(length, r_dim, dropout, DNN_outdim, GAT_hiddim, GAT_outdim,mlp_outdim, threshold=0.5, nheads=1):
    lr = 0.005
    seed = 20221231
    epochs = 500
    patience = 150
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    print('length: ' + str(length),
          'r_dim: ' + str(r_dim),
          'dropout: ' + str(dropout),
          'lr: ' + str(lr),
          'GAT_hiddim: ' + str(GAT_hiddim),
          'GAT_outdim: ' + str(GAT_outdim),
          'mlp_outdim: ' + str(mlp_outdim)
          )


    def train(epoch):
        t = time.time()
        dnn.train()
        gnn.train()
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
                             threshold)
        loss_train.backward()
        optimizer.step()

        print('Epoch: {:04d}'.format(epoch),
              'loss_train: {:.4f}'.format(loss_train.data.item()),
              'acc_train: {:.4f}'.format(acc_train.data.item()),

              'time: {:.4f}s'.format(time.time() - t))

        return loss_train.data.item()

    def compute_test():
        gnn.eval()
        mlp.eval()
        dnn.eval()
        seq_features1 = dnn(seq_features)
        output = gnn(features, adj)
        final_emb = torch.cat((seq_features1, output), dim=-1)
        predictions = mlp(final_emb[net[idx_test][:, 0]], final_emb[net[idx_test][:, 1]])
        loss_func = nn.BCELoss(size_average=False, reduce=True)
        loss_test = loss_func(predictions.reshape(predictions.shape[0], ).to(torch.float32),
                              labels[idx_test].to(torch.float32))
        acc_test = accuracy(predictions.reshape(predictions.shape[0], ), labels[idx_test],
                            threshold)
        precision, recall, f1, mcc, roc, ap = metric(predictions.reshape(predictions.shape[0], ),
                                                     labels[idx_test], threshold)

        return np.array(
            [round(acc_test.data.item(), 4), round(precision, 4), round(recall, 4), round(f1, 4), round(mcc, 4),
             round(roc, 4), round(ap, 4)])

    def early_train():
        # Train model
        t_total = time.time()
        loss_values = []
        bad_counter = 0
        # best = args.epochs + 1
        best = 10000000
        best_epoch = 0
        for epoch in range(epochs):
            loss_values.append(train(epoch))

            torch.save(gnn.state_dict(), '{}.gnn.pkl'.format(epoch))
            torch.save(mlp.state_dict(), '{}.mlp.pkl'.format(epoch))
            torch.save(dnn.state_dict(), '{}.dnn.pkl'.format(epoch))
            if loss_values[-1] < best - 1:
                best = loss_values[-1]
                best_epoch = epoch
                bad_counter = 0
            else:
                bad_counter += 1

            if bad_counter == patience:
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

        gnn.load_state_dict(torch.load('{}.gnn.pkl'.format(best_epoch)))
        mlp.load_state_dict(torch.load('{}.mlp.pkl'.format(best_epoch)))
        dnn.load_state_dict(torch.load('{}.dnn.pkl'.format(best_epoch)))

    k = 5
    all_metric = np.zeros(7, dtype=float)

    all_acc = []
    metrics = []

    accuracys = []
    idx_features, net_labels = parse_data(r_dim, length)
    for i in range(k):
        adj, features, seq_features, labels, idx_train, idx_test, net = load_data(i, idx_features, net_labels)
        # print('第i折', i)
        gnn = GNNmodel(nfeat=features.shape[1],
                    nproteins=features.shape[0],
                    nhid=GAT_hiddim,
                    nclass=GAT_outdim,
                    dropout=dropout,
                    nheads=nheads,
                    alpha=0.2)
        num_proteins = adj.shape[0]
        layers = [seq_features.shape[1], 512, 256, 128, DNN_outdim]
        dnn = DNNmodel(layers)
        layers = []
        mlp_indim = GAT_outdim * 2 + DNN_outdim * 2

        temp = mlp_indim
        layers.append(temp)
        while temp > mlp_outdim:
            temp = temp / 2
            layers.append(int(temp))

        mlp = MLPmodel(num_proteins, num_proteins, layers)

        optimizer = torch.optim.Adam([{'params': gnn.parameters(), 'weight_decay': 5e-4},
                                      {'params': mlp.parameters(), 'weight_decay': 5e-4},
                                      {'params': dnn.parameters(), 'weight_decay': 5e-4}],
                                     lr=lr)

        gnn.cuda()
        dnn.cuda()
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        seq_features = seq_features.cuda()
        idx_test = idx_test.cuda()
        mlp.cuda()

        # features, adj, labels = Variable(features), Variable(adj), Variable(labels)

        early_train()
        res = compute_test()
        files = glob.glob('*.pkl')
        for file in files:
            os.remove(file)

        all_metric = all_metric + res
        mean_metric = all_metric / 5
        metrics.append(res)

    print(mean_metric)
    acc = mean_metric[0]
    precision = mean_metric[1]
    recall = mean_metric[2]
    f1 = mean_metric[3]
    mcc = mean_metric[4]
    roc = mean_metric[5]
    ap = mean_metric[6]

    with open('search_log.txt', 'a') as log_text:
        log_text.write(str(length) + ' ' +
                       str(r_dim) + ' ' +
                       str(dropout) + ' ' +
                       str(DNN_outdim) + ' ' +
                       str(GAT_hiddim) + ' ' +
                       str(GAT_outdim) + ' ' +
                       str(mlp_outdim) + ' ' +
                       # str(threshold)+' '+
                       str(round(acc, 3)) + ' ' +
                       str(round(precision, 3)) + ' ' +
                       str(round(recall, 3)) + ' ' +
                       str(round(f1, 3)) + ' ' +
                       str(round(mcc, 3)) + ' ' +
                       str(round(roc, 3)) + ' ' +
                       str(round(ap, 3)) + ' ' + '\n')

    return acc

