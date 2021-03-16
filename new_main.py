# Initialize connection with rest of lib
from metrics import Metrictor
from new_utils import *
from DL_ClassifierModel import *
from sklearn import metrics
import numpy as np
from time import time

# Own imports
from pathlib import Path

dataset = 'bindingdb' # celegans / human / bindingdb


if dataset=='celegans':
    model_path = Path("../results/new_utils_test_992.pkl")
    data_path = Path("data/celegans")
    data_class = LoadCelegans(dataPath=data_path)


elif dataset=='human':
    model_path = Path("../results/human_cv1_985.pkl")
    data_path = Path("data/human/data.txt")
    data_class = LoadCelegans(dataPath=data_path)

else: #bindingdb
    model_path = Path("../results/new_utils_test_bindingdb_962.pkl")
    data_path = Path("data/bindingdb")
    data_class = LoadBindingDB(dataPath=data_path)


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
    test = np.array(data_class.eSeqData['test'])
    
    print("TEST1:",test[0])
    print("TEST2:",test[1])
    print("TEST3:",test[2])

    ones = 0
    for i in range(len(test)):
        if test[i,2]==1:
            ones += 1
    print("NUMBER OF 1 LABELS:", ones)

    print(len(test))
    Ypre, Y, seenbool = model.calculate_y_with_seenbool(data_class.one_epoch_batch_data_stream(batchSize=128, type='test', device=torch.device('cuda')))
    # print(Ypre)
    print(Ypre.shape)

    Y_heavi = np.where(Ypre>=0.5, 1, 0)
    n_seen, n_unseen, seen_correct, unseen_correct = seen_stats(test, seenbool, Y_heavi, Y)
    print("\nNumber of SEEN proteins in test set:", n_seen)
    print("Number of UNSEEN proteins in test set:", n_unseen)
    print("\nAccuracy on SEEN proteins in test set:", seen_correct/n_seen)
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

