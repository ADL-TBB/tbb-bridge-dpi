# Initialize connection with rest of lib
from metrics import Metrictor
from utils import *
from DL_ClassifierModel import *
from sklearn import metrics
import numpy as np
from time import time

# Own imports
from pathlib import Path

dataset = 'celegans' # celegans / human / bindingdb

if dataset=='celegans':
    model_path = Path("TEST_celegans_996.pkl")
    data_path = Path("data/celegans/data.txt")
    data_class = DataClass_normal(dataPath=data_path)
elif dataset=='human':
    model_path = Path("TEST_human_991.pkl")
    data_path = Path("data/human/data.txt")
    data_class = DataClass_normal(dataPath=data_path)
else: #bindingdb
    model_path = Path("TEST_bindingdb_962.pkl")
    data_path = Path("data/bindingdb")
    data_class = DataClass(dataPath=data_path)

model = DTI_Bridge(outSize=128,
                   cHiddenSizeList=[1024],
                   fHiddenSizeList=[1024, 256],
                   fSize=1024, cSize=data_class.pContFeat.shape[1],
                   gcnHiddenSizeList=[128, 128], fcHiddenSizeList=[128], nodeNum=64,
                   hdnDropout=0.5, fcDropout=0.5, device=torch.device('cuda'))
model.load(path=model_path, map_location="cuda", dataClass=data_class)
model.to_eval_mode()

def seen_stats(data, seen_data, Y_heavi, Y):
    n_seen = np.sum(seen_data)
    n_unseen = data.shape[0] - n_seen
    seen_correct = 0
    unseen_correct = 0
    for i in range(data.shape[0]):
        if Y_heavi[i] == Y[i]:
            if seen_data[i]:
                seen_correct += 1
            else:
                unseen_correct += 1
    return n_seen, n_unseen, seen_correct, unseen_correct

def get_metrics(database):
    if database=='celegans' or database=='human':
        valid = np.array(data_class.eSeqData[data_class.trainSampleNum:])
        # Don't give type='test' to one_epoch_data_stream, because there is no test set for CElegans/Human
        Ypre, Y, seenbool = model.calculate_y_with_seenbool(data_class.one_epoch_batch_data_stream(batchSize=128, device=torch.device('cuda')))
        Y_heavi = np.where(Ypre>=0.5, 1, 0)
        n_seen, n_unseen, seen_correct, unseen_correct = seen_stats(valid, seenbool, Y_heavi, Y)
        print("\nNumber of SEEN proteins in validation set:", n_seen)
        print("Number of UNSEEN proteins in validation set:", n_unseen)
        print("Accuracy on SEEN proteins in validation set:", seen_correct/n_seen)
        print("Accuracy on UNSEEN proteins in validation set:", unseen_correct/n_unseen)
        print("\nAccuracy and AUC on total validation set:")
    else: # bindingdb
        test = np.array(data_class.eSeqData['test'])
        Ypre, Y, seenbool = model.calculate_y_with_seenbool(data_class.one_epoch_batch_data_stream(batchSize=128, type='test', device=torch.device('cuda')))
        Y_heavi = np.where(Ypre>=0.5, 1, 0)
        n_seen, n_unseen, seen_correct, unseen_correct = seen_stats(test, seenbool, Y_heavi, Y)
        print("\nNumber of SEEN proteins in test set:", n_seen)
        print("Number of UNSEEN proteins in test set:", n_unseen)
        print("Accuracy on SEEN proteins in test set:", seen_correct/n_seen)
        print("Accuracy on UNSEEN proteins in test set:", unseen_correct/n_unseen)
        print("\nAccuracy and AUC on total test set:")
    
    metrictor = Metrictor()
    metrictor.set_data(Ypre, Y)
    report = ["ACC", "AUC"]
    metrictor(report)

print("\nNumber of training examples:", data_class.trainSampleNum)
print("Number of valid examples:", data_class.validSampleNum)
print("Number of test examples:", data_class.testSampleNum)
get_metrics(dataset)

''' To test whether there are differences between new_utils test set and utils test set: NO DIFFERENCE
dif = 0
nodif = 0
a = np.array(([[1,2,3],[2,3,4],[3,4,5]]))
b = np.array(([[1,1,1],[2,2,2],[1,2,3]]))
for i in range(len(a)):
    if not(np.any(a[i]==b)):
        dif += 1
    else:
        nodif += 1
print(dif)
'''