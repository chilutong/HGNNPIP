# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 21:24:08 2019
@author: zju
"""
import os
from math import log
import numpy as np

from sklearn.model_selection import KFold, ShuffleSplit
# from build_my_layer import MyMaskCompute, MySpatialDropout1D
# from utility import random_arr, array_split
# from input_preprocess import preprocess
# import warnings
# import argparse
from train100 import test

# result = test(length=150,r_dim=3,dropout=0.5,lr=0.001,opti_switch=0)



bounds = [
    # Discrete
    {'name': 'length', 'type': 'discrete', 'domain': (100, 150)},
    {'name': 'r_dim', 'type': 'discrete', 'domain': (3, 6)},
    {'name': 'dropout', 'type': 'discrete', 'domain': (0.4, 0.5)},
    {'name': 'DNN_outdim', 'type': 'discrete', 'domain': (32, 16)},
    {'name': 'GAT_hiddim', 'type': 'discrete', 'domain': (256, 128)},
    {'name': 'GAT_outdim', 'type': 'discrete', 'domain': (32, 16)},
    {'name': 'mlp_outdim', 'type': 'discrete', 'domain': (16,8)}
    # {'name': 'threshold', 'type': 'discrete', 'domain': (0.5)}
    # {'name': 'nheads', 'type': 'discrete', 'domain': (1)},

    # Categorical
    # {'name': 'opti_switch', 'type': 'categorical', 'domain': (0, 1)}
]

from dotmap import DotMap


def search_param(x):
    opt = DotMap()

    opt.length = int(x[:, 0])
    opt.r_dim = int(x[:, 1])
    opt.dropout = float(x[:, 2])
    opt.DNN_outdim = int(x[:, 3])
    opt.GAT_hiddim = int(x[:, 4])
    opt.GAT_outdim = int(x[:, 5])
    opt.mlp_outdim = int(x[:, 6])
    # opt.threshold=float(x[:,7])
    # opt.nheads=int(x[:,8])
    # opt.opti_switch = float(x[:, 6])
    return opt


yy = 0


def f(x):
    global yy

    local_yy = yy
    local_yy += 1
    yy = local_yy

    opt = search_param(x)
    param = {
        'length': opt.length,
        'r_dim': opt.r_dim,
        'dropout': opt.dropout,
        'DNN_outdim': opt.DNN_outdim,
        'GAT_hiddim': opt.GAT_hiddim,
        'GAT_outdim': opt.GAT_outdim,
        'mlp_outdim': opt.mlp_outdim
        # 'threshold':opt.threshold
        # 'nheads':opt.nheads

        # 'opti_switch': opt.opti_switch
    }

    result = test(**param)

    # evaluation = 1 - result
    evaluation = 1 - result
    # with open('search_log.txt', 'a') as log_text:
    #     log_text.write('evaluation: ' + str(evaluation) + '\n')
    #     log_text.write('---------\n')

    print('cycle: ' + str(yy))
    print('evaluation: ' + str(evaluation))

    return evaluation


import GPy
import GPyOpt

opt_model = GPyOpt.methods.BayesianOptimization(f=f, domain=bounds, initial_design_numdata=10)
new_x =[150,6,0.4,32,128,32,16]
new_y = f(new_x)
opt_model.set_XY(new_x,new_y)

opt_model.run_optimization(max_iter=1)

with open('search_log2.txt', 'a') as log_text:
    log_text.write('result: \n')

    for i, v in enumerate(bounds):
        name = v['name']
        log_text.write('parameter {}: {}\n'.format(name, opt_model.x_opt[i]))
    log_text.write('evaluation: ' + str(1 - opt_model.fx_opt) + '\n')

print('Congratulations, the training is complete')
