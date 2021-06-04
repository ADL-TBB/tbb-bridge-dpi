# Initialize connection with rest of lib
from utils import *
from DL_ClassifierModel import *
import os
from pathlib import Path

"""
model_name:

DTI_Bridge
ST_Bridge
p_Embedding_Bridge
p_Emb_ST_Bridge
"""

# Choose dataset and modelfile name
data = "celegans"
model_name = "DTI_Bridge"
save_path = "test_celegans"

data_path = Path(os.path.join("data", data))
assert data_path.exists(), "Download the necessary data from the following link: " \
                           "https://raw.githubusercontent.com/masashitsubaki/CPI_prediction/master/dataset/celegans/original/data.txt"

if data=="celegans" or data=="human":
    data_class = LoadCelegansHuman(dataPath=data_path, model_name=model_name)
elif data == "bindingdb":
    data_class = LoadBindingDB(dataPath=data_path, model_name=model_name)
else: #"chembl"
    data_class = LoadChembl(dataPath=data_path, model_name=model_name)


model = DTI_Bridge(outSize=128,
                  cHiddenSizeList=[1024],
                  fHiddenSizeList=[1024, 256],
                  fSize=1024, cSize=8424, # change if kmer size changes, or use data_class.pContFeat.shape[1]
                  gcnHiddenSizeList=[128, 128], fcHiddenSizeList=[128], nodeNum=64,
                  hdnDropout=0.5, fcDropout=0.5, device=torch.device('cuda'))

"""
model class name correspond to model_name, like:

model = p_Embedding_Bridge(outSize=128,
                  cHiddenSizeList=[1024],
                  fHiddenSizeList=[1024, 256],
                  fSize=1024, cSize=8424, # change if kmer size changes, or use data_class.pContFeat.shape[1]
                  gcnHiddenSizeList=[128, 128], fcHiddenSizeList=[128], nodeNum=64,
                  hdnDropout=0.5, fcDropout=0.5, device=torch.device('cuda'))
"""



model.train(data_class, trainSize=512, batchSize=512, epoch=128,
            stopRounds=-1, earlyStop=30,
            savePath=save_path, metrics="AUC", report=["ACC", "AUC", "LOSS"],
            preheat=0)
'''
model.cv_train(data_class, trainSize=512, batchSize=512, epoch=128,
            stopRounds=-1, earlyStop=30,
            savePath=save_path, metrics="AUC", report=["ACC", "AUC", "LOSS"],
            preheat=0)
'''