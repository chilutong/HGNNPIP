import numpy as np
import scipy.sparse as sp
import torch
import math
import word2vec
import pandas as pd
from sklearn.metrics import roc_auc_score, classification_report, average_precision_score, precision_recall_curve, auc
from sklearn.preprocessing import StandardScaler

def parse_data(dataset,r_dim, length,path="./data/"):

    print('Loading {} dataset...'.format(dataset))
    # Sequence vectorization
    word2vec.sequenceEmbedding(path + dataset, r_dim, length)
    # load the Sequence vector
    idx_features = pd.read_csv(path + dataset + '/word2vec_features.csv', header=None).values
    # load the PPI network
    net_labels = np.genfromtxt("{}{}/re_network".format(path, dataset), dtype=np.int)
    # np.random.seed(20221231)
    np.random.shuffle(net_labels)
    return idx_features, net_labels


def load_data(i, idx_features, net_labels):

    features = idx_features[:, 1:]
    net = net_labels[:, :2]
    labels = net_labels[:, -1]
    total = len(net)
    fold = int(total / 5)
    idx_test = range(i * fold, (i + 1) * fold)
    idx_train = list(range(0, i * fold)) + list(range((i + 1) * fold, total))
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
    # get the degree feature
    degree = adj.todense().sum(axis=1)
    features = np.append(features, degree, axis=1)
    features = np.append(features, features, axis=1)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))
    scaler = StandardScaler().fit(features)
    features = scaler.transform(features)
    seq_features = features[:, :-1]

    adj = torch.FloatTensor(np.array(adj.todense()))
    features = torch.FloatTensor(features)
    seq_features = torch.FloatTensor(seq_features)
    labels = torch.tensor(labels)
    idx_train = torch.LongTensor(idx_train)
    idx_test = torch.LongTensor(idx_test)
    net = torch.LongTensor(net)
    return adj, features, seq_features, labels, idx_train, idx_test, net


def normalize_adj(mx):
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)

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

    precision = float(tp) / (tp + fp + 1e-06)
    recall = float(tp) / (tp + fn + 1e-06)
    spec=float(tn) / (tn + fp + 1e-06)
    f1 = float(2 * tp) / (2 * tp + fp + fn + 1e-06)
    mcc = (tp * tn - fp * fn) / (math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) + 1e-06)
    roc_auc = roc_auc_score(labels.cuda().data.cpu().numpy(), output.cuda().data.cpu().numpy())
    ap = average_precision_score(labels.cuda().data.cpu().numpy(), output.cuda().data.cpu().numpy(), average='macro')

    return precision, recall,spec, f1, mcc, roc_auc, ap


