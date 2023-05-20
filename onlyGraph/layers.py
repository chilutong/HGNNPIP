import numpy as np
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import math


class AttentionLayer(nn.Module):
    def __init__(self, in_features, nproteins, out_features, dropout, alpha, concat=True):
        super(AttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.W1 = nn.Parameter(torch.empty(size=(int(in_features/2), out_features)))
        self.W2 = nn.Parameter(torch.empty(size=(int(in_features/2), out_features)))
        nn.init.xavier_uniform_(self.W1.data, gain=1.414)
        nn.init.xavier_uniform_(self.W2.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        self.bn1 = nn.BatchNorm1d(out_features)
        self.bn2 = nn.BatchNorm1d(2 * out_features)
        self.k1 = nn.Parameter(torch.rand(1))
        self.k2 = nn.Parameter(torch.rand(1))

        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        d=int(h.shape[1]/2)
        h1=h[:,:d]
        h2=h[:,d:]
        # Wh = torch.mm(h, self.W)  # h.shape: (N, in_features), Wh.shape: (N, out_features)
        Wh1 = torch.mm(h1, self.W1)  # h.shape: (N, in_features), Wh.shape: (N, out_features)
        Wh2 = torch.mm(h2, self.W2)  # h.shape: (N, in_features), Wh.shape: (N, out_features)
        e1 = self._prepare_attentional_mechanism_input(Wh1)
        e2 = self._prepare_attentional_mechanism_input(Wh2)
        adj_2hop = torch.mm(adj, adj)
        diag = torch.diag(adj_2hop)
        adj_2hop_diag = torch.diag_embed(diag)
        adj_2hop = adj_2hop - adj_2hop_diag

        zero_vec1 = -9e15 * torch.ones_like(e1)
        zero_vec2 = -9e15 * torch.ones_like(e2)
        # zero1 = torch.zeros_like(e)
        # e_abs = torch.abs(e)
        # ones = torch.ones_like(e)
        # e = torch.where(e_abs > 1, e, zero1)
        attention = torch.where(adj > 0, e1, zero_vec1)

        attention = F.softmax(attention, dim=1)

        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh1)
        # h_prime = self.bn1(h_prime)
        attention2 = torch.where(adj_2hop > 0, e2, zero_vec2)
        attention2 = F.softmax(attention2, dim=1)
        attention2 = F.dropout(attention2, self.dropout, training=self.training)
        h_prime2 = torch.matmul(attention2, Wh2)

        h_prime = torch.cat([h_prime, h_prime2], dim=-1)
        h_prime = self.bn2(h_prime)
        if self.concat:
            return self.leakyrelu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)

        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'