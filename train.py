# Initialize connection with rest of lib
from new_utils import *
from DL_ClassifierModel import *

# Own imports
from pathlib import Path

data_path = Path("data/bindingdb")
# data_path = Path("data/bindingdb")
assert data_path.exists(), "Download the necessary data from the following link: " \
                           "https://raw.githubusercontent.com/masashitsubaki/CPI_prediction/master/dataset/celegans/original/data.txt"

# For celegans/human
# data_class = LoadCelegans(dataPath=data_path)

# For bindingdb
data_class = LoadBindingDB(dataPath=data_path)

model = DTI_Bridge(outSize=128,
                  cHiddenSizeList=[1024],
                  fHiddenSizeList=[1024, 256],
                  fSize=1024, cSize=data_class.pContFeat.shape[1],
                  gcnHiddenSizeList=[128, 128], fcHiddenSizeList=[128], nodeNum=64,
                  hdnDropout=0.5, fcDropout=0.5, device=torch.device('cuda'))


model.train(data_class, trainSize=512, batchSize=512, epoch=128,
            stopRounds=-1, earlyStop=30,
            savePath='new_utils_test_bindingdb', metrics="AUC", report=["ACC", "AUC", "LOSS"],
            preheat=0)  # Used to have learning rate (lr=0.001) as param ???
'''
model.cv_train(data_class, trainSize=512, batchSize=512, epoch=128,
            stopRounds=-1, earlyStop=30,
            savePath='human', metrics="AUC", report=["ACC", "AUC", "LOSS"],
            preheat=0)  # Used to have learning rate (lr=0.001) as param ???
'''