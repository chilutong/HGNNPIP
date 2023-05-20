import numpy as np
import scipy.sparse as sp
import torch
import math
from functools import reduce
import word2vec
import pandas as pd
from sklearn.metrics import roc_auc_score, classification_report, average_precision_score, precision_recall_curve, auc
from sklearn.preprocessing import StandardScaler

def load_data(length=150, r_dim=6, path="./data", dataset="/S.cere/"):

    print('Loading {} dataset...'.format(dataset))
    # word2vec.sequenceEmbedding(path+dataset,r_dim,length)
    idx_features = pd.read_csv(path + dataset + 'word2vec_features.csv', header=None).values
    uniprot = np.loadtxt(path + dataset + 'uniprot', dtype='str')
    features = idx_features[:, 1:]
    prelist = pd.read_csv(path + dataset + 'B_negative', sep='\t', header=None).values
    prelist = uniprot2id(prelist,uniprot)

    net_labels = np.genfromtxt("{}{}Random_sample2".format(path, dataset), dtype=np.int)
    np.random.seed()
    np.random.shuffle(net_labels)
    net = net_labels[:, :2]
    labels = net_labels[:, -1]
    idx_train = range(0, int(0.8 * (len(net_labels) - 1)))
    idx_test = range(int(0.8 * (len(net_labels) - 1)), len(net_labels) - 1)
    train = net_labels[idx_train]
    # build graph
    idx = np.array(idx_features[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = train[np.where(train[:, 2] == 1)][:, 0:2]
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(features.shape[0], features.shape[0]), dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    degree = adj.todense().sum(axis=1)
    features = np.append(features, degree, axis=1)
    features = np.append(features, features, axis=1)
    # adj = normalize_adj(adj + sp.eye(adj.shape[0]))
    scaler = StandardScaler().fit(features)
    features = scaler.transform(features)
    seq_features = features[:, :-1]

    adj = torch.FloatTensor(np.array(adj.todense()))
    features = torch.FloatTensor(features)
    seq_features = torch.FloatTensor(seq_features)
    labels = torch.tensor(labels)
    idx_train = torch.LongTensor(idx_train)
    idx_test = torch.LongTensor(idx_test)
    prelist = torch.LongTensor(prelist)
    net = torch.LongTensor(net)
    return adj, features, seq_features, labels, idx_train, idx_test, net, prelist


def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels, threshold):
    one = torch.ones_like(output)
    zero = torch.zeros_like(output)
    preds = torch.where(output >= threshold, one, zero)
    correct = preds.eq(labels)
    correct = correct.sum()
    return correct / len(labels)


def metric(output, labels, threshold):
    one = torch.ones_like(output)
    zero = torch.zeros_like(output)

    preds = torch.where(output >= threshold, one, zero)
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    allp = 0
    for i in range(len(preds)):
        if labels[i] == 1 and preds[i] == 1:
            tp = tp + 1
        elif labels[i] == 1 and preds[i] == 0:
            fn = fn + 1
        elif labels[i] == 0 and preds[i] == 1:
            fp = fp + 1
        elif labels[i] == 0 and preds[i] == 0:
            tn = tn + 1
    acc = (tp + tn) / (tp + fn + fp + tn)

    precision = float(tp) / (tp + fp + 1e-06)
    recall = float(tp) / (tp + fn + 1e-06)
    f1 = float(2 * tp) / (2 * tp + fp + fn + 1e-06)
    mcc = (tp * tn - fp * fn) / (math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) + 1e-06)
    roc_auc = roc_auc_score(labels.cpu().detach().numpy(), output.cpu().detach().numpy())
    ap = average_precision_score(labels.cpu().detach().numpy(), output.cpu().detach().numpy(), average='macro')
    return precision, recall, f1, mcc, roc_auc, ap


def uniprot2id(network,uniprot):
    uniprot = dict(np.flip(uniprot))
    list=[]
    for pair in network:
        A_id = int(uniprot[pair[0]])
        B_id = int(uniprot[pair[1]])
        list.append([A_id,B_id,pair[2]])
    return list