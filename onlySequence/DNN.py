import torch
import torch.nn as nn
import torch.nn.functional as F


class DNNmodel(nn.Module):
    def __init__(self, layers):
        super(DNNmodel, self).__init__()

        self.dropout = 0.5


        # 全连接网络
        self.dnn_network = nn.ModuleList()
        self.bn_start = nn.BatchNorm1d(layers[0])
        self.leakyrelu = nn.LeakyReLU(0.2)

        # self.dnn_network = nn.ModuleList(
        #     [nn.Linear(layer[0], layer[1]) for layer in list(zip(layers[:-1], layers[1:]))])

        i = 0
        for layer in list(zip(layers[:-1], layers[1:])):
            self.dnn_network.add_module('ln' + str(i), nn.Linear(layer[0], layer[1],bias=False))
            self.dnn_network.add_module('bn' + str(i), nn.BatchNorm1d(layer[1]))
            i = i + 1

        self.linear = nn.Linear(layers[-1], 1)


        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x = F.dropout(x, self.dropout, training=self.training)
        # 全连接网络
        for linear in self.dnn_network:

            if type(linear) == type(nn.Linear(2, 1)):
                x = linear(x)
            if type(linear) == type(nn.BatchNorm1d(1)):
                x = linear(x)
                x = self.leakyrelu(x)
                x = F.dropout(x, self.dropout, training=self.training)

        return x


