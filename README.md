# HGNNPIP
HGNNPIP: A Hybrid Graph Neural Network framework for Protein-protein Interaction Prediction

HGNNPIP is a novel computational model for PPI predictions which comprehensively characterizes the intrinsic relationship between two proteins by integrating protein sequence and PPI network connection properties.

HGNNPIP Version: V1.0 Date: 2023-05-19 Platform: Pytorch 1.11.0 and Python 3.9.0

[1] The folders in the HGNNPIP package:

data: This folder contains nine benchmark datasets, including two PPI information files (positive and negative cases) and sequence information file.
embeddings: This folder stores the embedding features generated in the training.
OpenNE: This folder contains the node2vec method source code.
SOTA: This folder contains 5 state-of-the-art algorithms.
[2] Scripts:

The source code files of HGNNPIP model

word2vec.py - This script is used to encode the protein sequences.

utils.py - This script is used to prepare data and features for the model.

TuneHyparam.py - This script is used to optimal the hyperparam.

5cv.py – This script is used to evaluate the performance of HNSPPI with MLP classifier.

5cv_cls.py – This script is used to evaluate the performance of HNSPPI with other classifiers.

5cv_LR.py – This script is used to evaluate the performance of HNSPPI with LR classifier.

5cv_SVM.py – This script is used to evaluate the performance of HNSPPI with SVM classifier.

prediction.py - This script is used to predict the new PPI.

The source code files for SOTA algorithms
DeepFE.py – This script is to evaluate the performance of DeepFE-PPI. The complete source code can be downloaded from https://github.com/xal2019/DeepFE-PPI.

DCONV.py- This script is to evaluate the performance of DCONV. The source code can be downloaded from https://gitlab.univnantes.fr/richoux-f/DeepPPI/tree/v1.tcbb.

DFC.py- This script is to evaluate the performance of DFC. The source code can be downloaded from https://gitlab.univnantes.fr/richoux-f/DeepPPI/tree/v1.tcbb.

DeepPur(AAC).py- This script is to evaluate the performance of DeepPur(AAC). The code is downloaded from https://github.com/kexinhuang12345/DeepPurpose.

DeepPur(CNN).py- This script is to evaluate the performance of DeepPur(CNN). The code is downloaded from https://github.com/kexinhuang12345/DeepPurpose.

[3] Datasets:

(1) In this study, we provided seven testing datasets. The whole package of data can be downloaded from our official website: http://cdsic.njau.edu.cn/data/PPIDataBankV1.0.

(2) The data profiles for each dataset also can be downloaded from the folder ‘data’ in the Github.

-------Sharing/access Information-------

S.cerevisiae: PMID: 25657331

M.musculus: PMID: 34536380

H.pylori : PMID: 11196647

D.melanogaster: PMID: 19171120

Fly: PMID: 34536380

Human: csbio.sjtu.edu.cn/bioinf/LR_PPI/Data.htm.

[4] Running:

--Running the HGNNPIP model requires two edgelist files (one is for positive samples, and the other is for negative samples) and a csv file for the amino acid sequences of proteins.

--Command line:

run main.py script with --input1 --input2 --output --species --seed

--Model output: will generate a file called results.csv

--For example:

python 5cv.py --dataset S.cere

[5] Installation:

git clone https://github.com/chilutong/HGNNPIP