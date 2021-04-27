# Initialize connection with rest of lib
import logging
import rdkit
from utils import *
from DL_ClassifierModel import *
import os
from pathlib import Path
import torch

dataset = "bindingdb" # celegans / human / bindingdb
model_path = "TEST_binding_db"
model = 'DTI_bridge'
dump_file = "misclass_pEmbeddings"

data_path = Path(os.path.join("data", dataset))

if dataset=='celegans' or dataset=='human':
    data_class = LoadCelegansHuman(dataPath=data_path)
else: #bindingdb
    data_class = LoadBindingDB(dataPath=data_path)

data_class = LoadBindingDB(dataPath=data_path)

if model == 'DTI_bridge':
    model = DTI_Bridge(outSize=128,
                   cHiddenSizeList=[1024],
                   fHiddenSizeList=[1024, 256],
                   fSize=1024, cSize=data_class.pContFeat.shape[1],
                   gcnHiddenSizeList=[128, 128], fcHiddenSizeList=[128], nodeNum=64,
                   hdnDropout=0.5, fcDropout=0.5, device=torch.device('cuda'))

else:
    model = p_Embedding_Bridge(outSize=128,
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

def get_miscl():
    test = np.array(data_class.eSeqData['test'])
    Ypre, Y, _ = model.calculate_y_with_seenbool(data_class.one_epoch_batch_data_stream(batchSize=128, type='test', device=torch.device('cuda')))
    Y_heavi = np.where(Ypre>=0.5, 1, 0)
    pred_bool = np.where(Y_heavi != Y)
    examples_misclass = test[pred_bool]
    misclass_couples = [(data_class.id2p[miscl[0]], data_class.id2p[miscl[1]]) for miscl in examples_misclass]

    misclass_file = open(dump_file, 'wb')
    pkl.dump(misclass_couples, misclass_file)
    misclass_file.close()

    length_proteins = []
    length_drugs = []
    for i in misclass_couples:
        length_proteins.append(len(i[0]))
        length_drugs.append(len(i[1]))
    print(f'Average misclassified protein length: {np.mean(length_proteins)}')
    print(f'Average misclassified drug length: {np.mean(length_drugs)}')

    return misclass_couples

get_miscl()


    





