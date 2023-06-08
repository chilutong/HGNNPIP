# HGNNPIP
HGNNPIP: A Hybrid Graph Neural Network framework for Protein-protein Interaction Prediction

HGNNPIP is a novel computational model for PPI predictions which comprehensively characterizes the intrinsic relationship between two proteins by integrating protein sequence and PPI network connection properties.

HGNNPIP Version: V1.0 Date: 2023-05-19 Platform: Pytorch 1.11.0 and Python 3.9.0

[1] The folders in the HGNNPIP package:

data: This folder contains nine benchmark datasets.
SOTA: This folder contains 4 state-of-the-art algorithms.
onlyGraph:This folder contains the model only use the PPI network information.
onlySequence:This folder contains the model only use the Protein sequence information.
[2] Scripts:

The source code files of HGNNPIP model

word2vec.py - This script is used to encode the protein sequences.

DNN.py - This script is the sequence encoding model of the HGNNPIP.

GNN.py - This script is the network embedding model of the HGNNPIP.

layers.py- This script is the layer of the GNN model.

utils.py - This script is used to prepare data and features for the model.

TuneHyparam.py - This script is used to optimal the hyperparam.

Sample.py - This script is used to generate negative examples.

5cv.py – This script is used to evaluate the performance of HNSPPI with MLP classifier.

5cv_cls.py – This script is used to evaluate the performance of HNSPPI with other classifiers.

5cv_LR.py – This script is used to evaluate the performance of HNSPPI with LR classifier.

5cv_SVM.py – This script is used to evaluate the performance of HNSPPI with SVM classifier.

prediction.py - This script is used to predict the new PPI.

The source code files for SOTA algorithms
DeepFE.py – This script is to evaluate the performance of DeepFE-PPI. The complete source code can be downloaded from https://github.com/xal2019/DeepFE-PPI.

DeepTrio.py- This script is to evaluate the performance of DCONV. The source code can be downloaded from https://github.com/huxiaoti/deeptrio.git.

PIPR.py- This script is to evaluate the performance of DFC. The source code can be downloaded from https://github.com/muhaochen/seq_ppi.git.

GAT.py- This script is to evaluate the performance of DeepPur(AAC). The code is downloaded from https://github.com/Diego999/pyGAT.


[3] Datasets:

(1) In this study, we provided nine testing datasets. The whole package of data can be downloaded from our official website: http://cdsic.njau.edu.cn/data/PPIDataBankV1.0.

(2) The data profiles for each dataset also can be downloaded from the folder ‘data’ in the Github.

-------Sharing/access Information-------

S.cerevisiae: PMID: 20500905

C.elegan: PMID: 20500905

E.coli: PMID: 20500905

D.melanogaster: PMID: 20500905

Human: PMID:20698572

Oryza: https://cn.string-db.org.
H.pylori : PMID: 11196647

Fly: PMID: 34536380

[4] Running:

#### Negative sampling
In HGNNPIP, we propose an effective negative sampling method to construct a dataset. You can use Sample.py to obtain negative examples based on the collected PPI data, in order to better train the model.
+ Firstly, you need to prepare a PPI data file and place it in the dataset folder, just like "/dataset/S.cere/network" in the project directory.
+ Secondly , execute the command line with arguments in shell:
```
python Sample.py [--dataset]
```
for example:
```buildoutcfg
python Sample.py --S.cere
```
#### 5cv
You can run the 5cv.py to train the HGNNPIP model and test the performance in five-fold Cross-Validation.We provide nine datasets, and you can choose one to train the model.You need to execute the command line with arguments in shell: 
```
python 5cv.py [--dataset]
```
for example:
```buildoutcfg
python 5cv.py --S.cere
```

#### predict
In HGNNPIP, we propose an effective negative sampling method to construct a dataset. You can use Sample.py to obtain negative examples based on the collected PPI data, in order to better train the model.
+ Firstly, you need to prepare a predict list and place it in the dataset folder, just like the "/dataset/predict/list" in the project directory.
+ Secondly , execute the command line with arguments in shell:
```
python prediction.py [--dataset] [--predictList]
```
for example:
```buildoutcfg
python prediction.py --Oryza --'/data/predict/prediclist.csv'
```
[5] Installation:

git clone https://github.com/chilutong/HGNNPIP