import numpy as np
import scipy.sparse as sp
import torch
import math
import word2vec
# import word2vec_noPCA
import pandas as pd
from sklearn.metrics import roc_auc_score, classification_report, average_precision_score, precision_recall_curve, auc
from sklearn.preprocessing import StandardScaler


def encode_onehot(labels):
    # The classes must be sorted before encoding to enable static class encoding.
    # In other words, make sure the first class always maps to index 0.
    classes = sorted(list(set(labels)))
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


def parse_data(r_dim, length, path="../data/", dataset="HPRD1/"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))
    # word2vec.sequenceEmbedding(path + dataset, r_dim, length)
    idx_features = pd.read_csv(path + dataset + 'word2vec_features.csv', header=None).values
    net_labels = np.genfromtxt("{}{}Random_sample2".format(path, dataset), dtype=np.int)
    np.random.seed(20230101)
    np.random.shuffle(net_labels)
    return idx_features, net_labels


def load_data(i, idx_features, net_labels):
    # features = sp.csr_matrix(idx_features[:, 1:], dtype=np.float32)
    features = idx_features[:, 1:]

    net = net_labels[:, :2]
    labels = net_labels[:, -1]
    total = len(net)
    fold = int(total / 5)
    idx_test = range(i * fold, (i + 1) * fold)
    # idx_test = torch.LongTensor(idx_test)
    # idx_train = list(range(0,i*fold))+list(range((i+1)*fold,total))
    idx_train = list(range(0, i * fold)) + list(range((i + 1) * fold, total))

    # train_net=net[idx_train]
    # test_net = net[idx_test]
    train = net_labels[idx_train]
    # idx_train = torch.LongTensor(idx_train)
    # build graph
    idx = np.array(idx_features[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}

    # edges_unordered = np.genfromtxt("{}{}network".format(path, dataset), dtype=np.int32)[:, :2][0:4030]

    # edges_unordered = train[np.where(train[:, 2] >0)][:, 0:2]
    edges_unordered = train[np.where(train[:, 2] == 1)][:, 0:2]
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
    # edges = np.row_stack((edges, global_edges))
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(features.shape[0], features.shape[0]), dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    degree = adj.todense().sum(axis=1)
    features = np.append(features, degree, axis=1)
    features = np.append(features, features, axis=1)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))
    scaler = StandardScaler().fit(features)
    features = scaler.transform(features)
    seq_features = features[:, :-1]



    a = net_labels[idx_train][:, -1].sum()

    adj = torch.FloatTensor(np.array(adj.todense()))

    features = torch.FloatTensor(features)
    seq_features = torch.FloatTensor(seq_features)

    labels = torch.tensor(labels)

    idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    net = torch.LongTensor(net)
    return adj, features, seq_features, labels, idx_train, idx_test, net


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
    # precision = tp/(tp+fp)
    # recall = tp/(tp+fn)
    precision = float(tp) / (tp + fp + 1e-06)
    recall = float(tp) / (tp + fn + 1e-06)
    spec = float(tn) / (tn + fp + 1e-06)
    # f1=2*precision*recall/(precision+recall)
    f1 = float(2 * tp) / (2 * tp + fp + fn + 1e-06)
    mcc = (tp * tn - tp * fn) / (math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) + 1e-06)
    roc_auc = roc_auc_score(labels.cuda().data.cpu().numpy(), output.cuda().data.cpu().numpy())
    # precision1, recall1, thre = precision_recall_curve(labels.cuda().data.cpu().numpy(), output.cuda().data.cpu().numpy())
    # ap2=auc(recall1,precision1)
    ap = average_precision_score(labels.cuda().data.cpu().numpy(), output.cuda().data.cpu().numpy(), average='macro')

    return precision, recall,spec, f1, mcc, roc_auc, ap


