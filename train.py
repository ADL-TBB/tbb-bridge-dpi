# Initialize connection with rest of lib
from utils import *
from DL_ClassifierModel import *
import os
from pathlib import Path

"""
model_name:

DTI_Bridge          # baseline model
ST_Bridge           # baseline model with ST_fingerprint and pSeq + kmers
p_Embedding_Bridge  # pEmbeddings with just morgan fingerprint
p_Emb_ST_Bridge     # pEmbeddings with just st_fingerprint

p_Embedding_Bridge  # pEmbeddings with kmers and dgraph features, useFeatures= {'dSeq': True, 'kmers': True, 
'pEmbeddings': True, 'FP': True, 'ST_fingerprint': False, "pSeq": False}
use p_Embedding_Seq_Bridge as model_name
"""

# Choose dataset and modelfile name
data = "chembl"  # folder name in /data
model_name = "p_Embedding_Bridge"
save_path = "chembl_model"

data_path = Path(os.path.join("data", data))
assert data_path.exists(), "Download the necessary data from the following link: " \
                           "https://raw.githubusercontent.com/masashitsubaki/CPI_prediction/master/dataset/celegans/original/data.txt"

if data=="celegans" or data=="human":
    data_class = LoadCelegansHuman(dataPath=data_path, model_name=model_name)
elif data == "bindingdb":
    data_class = LoadBindingDB(dataPath=data_path, model_name=model_name)
else: #"chembl"
    data_class = LoadChembl(dataPath=data_path, model_name=model_name)


"""
model = DTI_Bridge(outSize=128,
                  cHiddenSizeList=[1024],
                  fHiddenSizeList=[1024, 256],
                  fSize=1024, cSize=8424, # change if kmer size changes, or use data_class.pContFeat.shape[1]
                  gcnHiddenSizeList=[128, 128], fcHiddenSizeList=[128], nodeNum=64,
                  hdnDropout=0.5, fcDropout=0.5, device=torch.device('cuda'))

"""
# model class name correspond to model_name variable name, like:

model = p_Embedding_Bridge(outSize=128,
                  cHiddenSizeList=[1024],
                  fHiddenSizeList=[1024, 256],
                  fSize=1024, cSize=8424, # change if kmer size changes, or use data_class.pContFeat.shape[1]
                  gcnHiddenSizeList=[128, 128], fcHiddenSizeList=[128], nodeNum=64,
                  hdnDropout=0.5, fcDropout=0.5, device=torch.device('cuda'))

model.train(data_class, trainSize=512, batchSize=512, epoch=128,
            stopRounds=-1, earlyStop=30,
            savePath=save_path, metrics="AUC", report=["ACC", "AUC", "LOSS"],
            preheat=0)
