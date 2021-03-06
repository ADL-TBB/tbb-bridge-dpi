# Initialize connection with rest of lib

from utils import *
from DL_ClassifierModel import *
import os
from pathlib import Path
import torch
import matplotlib.pyplot as plt
import seaborn as sns

# Experiment repetitions
repetitions = 3

# Define dataset paths
data_bdb = "bindingdb"
data_human = "human"
data_celegans = "celegans"
data_path_bdb = Path(os.path.join("data", data_bdb))
assert data_path_bdb.exists()
data_path_human = Path(os.path.join("data", data_human))
assert data_path_human.exists()
data_path_celegans = Path(os.path.join("data", data_celegans))
assert data_path_celegans.exists()

testset = "sarscov2"
data_path_sarscov = Path(os.path.join("data", testset))
assert data_path_sarscov.exists()

# Define drug- and proteinnames that are tested
drugnames = ["Hydroxychloroquine", "Chloroquine", "Dexamethasone", "Remdesivir", "Nafamostat", "Camostat", 
            "Pepcid", "Arbidol", "Nitazoxanide", "Ivermectin", "Fluvoxamine", "EIDD-2801"]
proteinnames = ["p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10", "p11", "p12"]

# Train model
def train(method, data_class, save_path):
    if method == 'DTI_Bridge':
        print(f'Training DTI_Bridge with pEmbeddings\n')
        model = DTI_Bridge(outSize=128,
            cHiddenSizeList=[1024],
            fHiddenSizeList=[1024, 256],
            fSize=1024, cSize=data_class.pContFeat.shape[1],
            gcnHiddenSizeList=[128,128], fcHiddenSizeList=[128], nodeNum=64,
            hdnDropout=0.5, fcDropout=0.5, device=torch.device('cuda'))
    else: # 'p_Embedding_Bridge'
        print(f'Training p_Embedding_Bridge with pEmbeddings\n')
        model = p_Embedding_Bridge(outSize=128,
            cHiddenSizeList=[1024],
            fHiddenSizeList=[1024, 256],
            fSize=1024, cSize=8424,
            gcnHiddenSizeList=[128,128], fcHiddenSizeList=[128], nodeNum=64,
            hdnDropout=0.5, fcDropout=0.5, device=torch.device('cuda'))

    model.train(data_class, trainSize=512, batchSize=512, epoch=128,
        stopRounds=-1, earlyStop=30,
        savePath=save_path, metrics="AUC", report=["ACC", "AUC", "LOSS"],
        preheat=0)

    return model


experiments = ["celegans", "human", "bindingdb"]

for experiment in experiments:
    for method in ['DTI_Bridge', 'p_Embedding_Bridge']:
        avg_results = np.zeros((12,12))
        for r in range(repetitions):
            save_path = f"sarscov_experiment_{str(r+1)}_{method}_{experiment}"
            
            if experiment=="celegans":
                data_class = LoadSarscov2_with_Celegans([data_path_celegans, data_path_sarscov], model_name=method, save_d_names=False)
            elif experiment=="human":
                data_class = LoadSarscov2_with_Human([data_path_human, data_path_sarscov], model_name=method, save_d_names=False)
            else: # "bindingdb"
                data_class = LoadSarscov2_with_BindingDB([data_path_bdb, data_path_sarscov], model_name=method, save_d_names=False)

            model = train(method, data_class, save_path)
            model.to_eval_mode()

            stream = data_class.random_batch_data_stream(batchSize=12, type='test', device=torch.device('cuda'), shuffle=False)

            YArr = []
            i = 0
            while i<12:
                try:
                    X, Y = next(stream)
                except:
                    break
                Y_pre = model.calculate_y_prob(X, mode='predict').cpu().data.numpy()
                
                avg_results[i] += Y_pre
                i += 1

        avg_results /= repetitions
        results = np.array(avg_results).T # Transpose to get rows=drugs, columns=proteins

        plt.figure(figsize=(12,12))
        ax = sns.heatmap(results, xticklabels=proteinnames, yticklabels=drugnames)
        save_file = f"SARSCOV_{method}_{experiment}"
        plt.savefig(save_file)