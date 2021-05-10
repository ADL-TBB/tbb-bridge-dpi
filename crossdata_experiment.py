# Initialize connection with rest of lib
import logging
import rdkit
from utils import *
from DL_ClassifierModel import *
import os
from pathlib import Path
import torch

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

# Define logger
log_file = 'results_crossdatasets.log'
log_format = '%(asctime)s : %(levelname)s : %(message)s'
logging.basicConfig(format=log_format, filename = log_file, level=logging.DEBUG, force=True)
logger = logging.getLogger()

def log(model_type, trainset, testset, training_data, valid_data, test_data):
    logger.debug(f'Conditions --> Model: {model_type}, Trained on: {trainset}, Tested on: {testset}\n')
    logger.info(f'Training --> ACC = {training_data[0]}, AUC = {training_data[1]}\n')
    logger.info(f'Validation --> ACC = {valid_data[0]}, AUC = {valid_data[1]}\n')
    logger.info(f'Test --> ACC = {test_data[0]}, AUC = {test_data[1]}\n')


# Train model
def train(method, data_class):
    if method == 'DTI_Bridge':
        kmers, pSeq, FP, dSeq = True, True, True, True
        pEmbeddings, ST_fingerprint = False, False
        useFeatures={"pEmbeddings": pEmbeddings, "kmers": kmers, "pSeq": pSeq,
                    "FP": FP, "dSeq": dSeq, "ST_fingerprint": ST_fingerprint}
        print(f'Training DTI_Bridge with pEmbeddings: {useFeatures["pEmbeddings"]}, kmers: {useFeatures["kmers"]}, pSeq: {useFeatures["pSeq"]}, FP: {useFeatures["FP"]}, dSeq: {useFeatures["dSeq"]}, ST_fingerprint: {useFeatures["ST_fingerprint"]}\n')
        model = DTI_Bridge(outSize=128,
            cHiddenSizeList=[1024],
            fHiddenSizeList=[1024, 256],
            fSize=1024, cSize=data_class.pContFeat.shape[1],
            gcnHiddenSizeList=[128,128], fcHiddenSizeList=[128], nodeNum=64,
            hdnDropout=0.5, fcDropout=0.5, device=torch.device('cuda'), 
            useFeatures=useFeatures)
    else: # 'p_Embedding_Bridge'
        pEmbeddings, dSeq = True, True
        ST_fingerprint, FP, kmers, pSeq = False, False, False, False
        useFeatures={"pEmbeddings": pEmbeddings, "kmers": kmers, "pSeq": pSeq,
                    "FP": FP, "dSeq": dSeq, "ST_fingerprint": ST_fingerprint}
        print(f'Training p_Embedding_Bridge with pEmbeddings: {useFeatures["pEmbeddings"]}, kmers: {useFeatures["kmers"]}, pSeq: {useFeatures["pSeq"]}, FP: {useFeatures["FP"]}, dSeq: {useFeatures["dSeq"]}, ST_fingerprint: {useFeatures["ST_fingerprint"]}\n')
        model = p_Embedding_Bridge(outSize=128,
            cHiddenSizeList=[1024],
            fHiddenSizeList=[1024, 256],
            fSize=1024, cSize=data_class.pContFeat.shape[1],
            gcnHiddenSizeList=[128,128], fcHiddenSizeList=[128], nodeNum=64,
            hdnDropout=0.5, fcDropout=0.5, device=torch.device('cuda'), 
            useFeatures=useFeatures)

    model.train(data_class, trainSize=512, batchSize=512, epoch=128,
        stopRounds=-1, earlyStop=30,
        savePath=save_path, metrics="AUC", report=["ACC", "AUC", "LOSS"],
        preheat=0)

    return model, useFeatures

# Test model
def test(model, data_class):
    train_stats = model.final_res['training']
    valid_stats = model.final_res['valid']
    #Get test results
    model.to_eval_mode()
    Ypre, Y, seenbool = model.calculate_y_with_seenbool(data_class.one_epoch_batch_data_stream(batchSize=128, type='test', device=torch.device('cuda')))
    metrictor = Metrictor()
    metrictor.set_data(Ypre, Y)
    test_stats = [metrictor.ACC(), metrictor.AUC()]
    return train_stats, valid_stats, test_stats


experiments = [ ["bindingdb", "celegans"],
                ["bindingdb", "human"],
                ["human", "celegans"],
                ["celegans", "human"],
                ["human", "bindingdb"],
                ["celegans", "bindingdb"] ]

for experiment in experiments:

    for method in ['DTI_Bridge', 'p_Embedding_Bridge']:
        
        train_stats, valid_stats, test_stats = [], [], []

        for i in range(repetitions):
            save_path = f"experiment_{i}_{method}_{experiment}"
            if experiment == ["bindingdb", "celegans"]:
                data_class = Load_trainBDB_testCElegans(dataPath = [data_path_bdb, data_path_celegans])
            
            elif experiment == ["bindingdb", "human"]:
                data_class = Load_trainBDB_testHuman(dataPath = [data_path_bdb, data_path_human])
            
            elif experiment == ["human", "celegans"]:
                data_class = Load_trainHuman_testCElegans(dataPath = [data_path_human, data_path_celegans])
            
            elif experiment == ["celegans", "human"]:
                data_class = Load_trainCElegans_testHuman(dataPath = [data_path_celegans, data_path_human])

            elif experiment == ["human", "bindingdb"]:
                data_class = Load_trainHuman_testBDB(dataPath = [data_path_human, data_path_bdb])

            else: # ["celegans", "bindingdb"]
                data_class = Load_trainCElegans_testBDB(dataPath = [data_path_celegans, data_path_bdb])

            model, useFeatures = train(method, data_class)
            train_res, valid_res, test_res = test(model, data_class)

            train_stats.append(train_res)
            valid_stats.append(valid_res)
            test_stats.append(test_res)
        
        train_mean = np.mean(np.array(train_stats), axis = 0)
        valid_mean = np.mean(np.array(valid_stats), axis = 0)
        test_mean = np.mean(np.array(test_stats), axis = 0)

        log(method, experiment[0], experiment[1], train_mean, valid_mean, test_mean)
