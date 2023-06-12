import numpy as np
import pandas as pd
import random
from itertools import combinations
import argparse
# get the argument
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='S.cere', help='the dataset')
args = parser.parse_args()


# load the PPI data
dataset = args.dataset
net = np.loadtxt(dataset+'/network',dtype=int)
# k=int(len(net)/2)
# net = net[0:k]
net = net[np.where(train[:, 2] == 1)]
l=len(net)
proteins=[]

# negative sampling
for pair in net:
    if pair[0] not in proteins:
        proteins.append(pair[0])
    if pair[1] not in proteins:
        proteins.append(pair[1])

i=0

# set the ratio of negative to positive(k)
k=2
while i <k*l:
    proA,proB = random.sample(proteins, 2)
    newPair = [proA, proB, 0]
    newPair_ = [proB, proA, 0]

    if not np.any(np.all(newPair[0:2]==net[:,0:2],axis=1)) and not np.any(np.all(newPair_[0:2]==net[:,0:2],axis=1)):
        net=np.row_stack((net,np.array([proA,proB,0])))
        i=i+1

# output
net = pd.DataFrame(net)
net.to_csv(dataset+'/re_network',sep='\t',index=False,header=False)


