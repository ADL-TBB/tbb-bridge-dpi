# Initialize connection with rest of lib
import logging
import rdkit
from utils import *
from DL_ClassifierModel import *
import os
from pathlib import Path
import torch

#The path with the pre-saved model parameters
model_path = Path(os.path.join("data", "chembl", "chembl_model_0_996.pkl"))
method = 'p_Embedding_Bridge'

# Define dataset paths
data_bdb = "bindingdb"
data_human = "human"

#Get paths for the datasets
data_path_bdb = Path(os.path.join("data", data_bdb))
assert data_path_bdb.exists()
data_path_human = Path(os.path.join("data", data_human))
assert data_path_human.exists()

# Define log file
log_file = 'results_crossdatasets_chEMBL'

#Function to log 
def write_to_file(dataset_name, stats, save_path):
    file = open(f"{save_path}.txt","a") 
    file.write(f'Prediction from chEMBL on: {dataset_name}\n')
    file.write(f'ACC = {stats[0]}, AUC = {stats[1]}, Precision = {stats[2]}, Recall = {stats[3]}, F1 = {stats[4]}')
    file.write('\n')
    file.close()

experiments = ['bindingdb', 'human']
for experiment in experiments:
    if experiment == "bindingdb":
        data_class = LoadBindingDB(dataPath = data_path_bdb, model_name=method, save_d_names=False)
    
    elif experiment == "human":
        data_class = LoadCelegansHuman(dataPath = data_path_human, model_name=method, save_d_names=False)
        
    #Load the p_Embedding_Bridge model and parametrize it
    model = p_Embedding_Bridge(outSize=128,
                       cHiddenSizeList=[1024],
                       fHiddenSizeList=[1024, 256],
                       fSize=1024, cSize=8424,
                       gcnHiddenSizeList=[128, 128], fcHiddenSizeList=[128], nodeNum=64,
                       hdnDropout=0.5, fcDropout=0.5, device=torch.device('cuda'))
    
    model.load(path=model_path, map_location="cuda", dataClass=data_class)
    model.to_eval_mode()
    metrictor = Metrictor()

    Ypre, Y, seenbool = model.calculate_y_with_seenbool(data_class.one_epoch_batch_data_stream(batchSize=128, type='test', device=torch.device('cuda')))
    metrictor.set_data(Ypre, Y)
    test_stats = [metrictor.ACC(), metrictor.AUC(), metrictor.Precision(), metrictor.Recall(), metrictor.F1()]
    write_to_file(experiment, test_stats, log_file)


