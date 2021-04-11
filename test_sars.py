from metrics import Metrictor
from utils import *
from DL_ClassifierModel import *
import numpy as np
import os
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt

avg_results = np.zeros((12,12))

# Choose data and model
dataset = "sarscov2" # celegans / human / bindingdb

# Could also average results over multiple models
model_path = "Last_BindingDB_958.pkl"

data_path = Path(os.path.join("data", dataset))

data_class = LoadSarscov2_with_BindingDB(data_path)
# This shape is dependent on the data that's loaded
# And should be the same as it was with training as with testing
# Which is why the 'original' training dat has to be loaded
print(data_class.pContFeat.shape)

model = DTI_Bridge(outSize=128,
                cHiddenSizeList=[1024],
                fHiddenSizeList=[1024, 256],
                fSize=1024, cSize=data_class.pContFeat.shape[1],
                gcnHiddenSizeList=[128, 128], fcHiddenSizeList=[128], nodeNum=64,
                hdnDropout=0.5, fcDropout=0.5, device=torch.device('cuda'))

model.load(path=model_path, map_location="cuda", dataClass=data_class)
model.to_eval_mode()

stream = data_class.unshuffled_data_stream(batchSize=12, type='test', device=torch.device('cuda'))

YArr, Y_preArr = [], []
# results = {}
i = 1
while True:
    try:
        X, Y = next(stream)
        # print(X['pSeqLen']) # Check if they're in correct order
    except:
        break
    Y_pre = model.calculate_y_prob(X, mode='predict').cpu().data.numpy()
    Y_preArr.append(Y_pre)
    i += 1
results = np.array(Y_preArr).T # Transpose to get rows=drugs, columns=proteins

drugnames = ["Hydroxychloroquine", "Chloroquine", "Dexamethasone", "Remdesivir", "Nafamostat", "Camostat", 
            "Pepcid", "Arbidol", "Nitazoxanide", "Ivermectin", "Fluvoxamine", "EIDD-2801"]

proteinnames = ["p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10", "p11", "p12"]

plt.figure(figsize=(10,10))
ax = sns.heatmap(results, xticklabels=proteinnames, yticklabels=drugnames)
plt.show()