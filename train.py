# Initialize connection with rest of lib
from utils import *
from DL_ClassifierModel import *
import os
from pathlib import Path

# Choose dataset and modelfile name
data = "bindingdb"
save_path = "TEST_bindingdb"

data_path = Path(os.path.join("data", data))
assert data_path.exists(), "Download the necessary data from the following link: " \
                           "https://raw.githubusercontent.com/masashitsubaki/CPI_prediction/master/dataset/celegans/original/data.txt"

if data=="celegans" or data=="human":
    data_class = LoadCelegansHuman(dataPath=data_path)
else: #bindingdb
    data_class = LoadBindingDB(dataPath=data_path)

model = DTI_Bridge(outSize=128,
                  cHiddenSizeList=[1024],
                  fHiddenSizeList=[1024, 256],
                  fSize=1024, cSize=data_class.pContFeat.shape[1],
                  gcnHiddenSizeList=[128, 128], fcHiddenSizeList=[128], nodeNum=64,
                  hdnDropout=0.5, fcDropout=0.5, device=torch.device('cuda'))

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