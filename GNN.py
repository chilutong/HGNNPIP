import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import AttentionLayer


class GNNmodel(nn.Module):
    def __init__(self, nfeat, nproteins, nhid, nclass, dropout, alpha, nheads):

        super(GNNmodel, self).__init__()

        self.dropout = dropout
        self.bn = nn.BatchNorm1d(nfeat)
        self.bn2 = nn.BatchNorm1d(nclass)
        # self.feature = nn.Parameter(torch.randn(nproteins,nfeat))
        self.attentions = [AttentionLayer(nfeat, nproteins, nhid, dropout=dropout, alpha=alpha, concat=True) for _
                           in range(nheads)]
        self.leakyrelu = nn.LeakyReLU(alpha)
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.linear = nn.Linear(nhid*nheads*2, nclass)
        self.out_att = AttentionLayer(nhid *nheads*2, nproteins, nclass, dropout=dropout, alpha=alpha,
                                           concat=False)
        # self.out_att = GraphAttentionLayer(nfeat * nheads, nproteins, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):

        x = F.dropout(x, self.dropout, training=self.training)

        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)

        x=self.linear(x)
        x=self.bn2(x)

        return x

