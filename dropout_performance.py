# Initialize connection with rest of lib
import logging
import rdkit
from utils import *
from DL_ClassifierModel import *
import os
from pathlib import Path
import torch

log_file = 'dropout.log'
log_format = '%(asctime)s : %(levelname)s : %(message)s'
logging.basicConfig(format=log_format, filename = log_file, level=logging.DEBUG)
logger = logging.getLogger()

def log(fc_dropout, setting, training_data, valid_data, test_data):
    logger.debug(f'Conditions --> fc_dropout: {fc_dropout}, pEmbeddings: {setting["pEmbeddings"]}, kmers: {setting["kmers"]}, pSeq: {setting["pSeq"]}, FP: {setting["FP"]}, dSeq: {setting["dSeq"]}, ST_fingerprint: {setting["ST_fingerprint"]}\n')
    logger.info(f'Training --> ACC = {training_data[0]}, AUC = {training_data[1]}\n')
    logger.info(f'Validation --> ACC = {valid_data[0]}, AUC = {valid_data[1]}\n')
    logger.info(f'Test --> ACC = {test_data[0]}, AUC = {test_data[1]}\n')
    logger.info('\n')

def write_to_file(fc_dropout, setting, training_data, valid_data, test_data, save_path):
    file = open(f"{save_path}.txt","a") 
    file.write(f'Conditions --> fc_dropout: {fc_dropout}, pEmbeddings: {setting["pEmbeddings"]}, kmers: {setting["kmers"]}, pSeq: {setting["pSeq"]}, FP: {setting["FP"]}, dSeq: {setting["dSeq"]}, ST_fingerprint: {setting["ST_fingerprint"]}\n')
    file.write(f'Training --> ACC = {training_data[0]}, AUC = {training_data[1]}\n')
    file.write(f'Validation --> ACC = {valid_data[0]}, AUC = {valid_data[1]}\n')
    file.write(f'Test --> ACC = {test_data[0]}, AUC = {test_data[1]}\n')
    file.write('\n')
    file.close()

#Training data binding DB
data = "bindingdb"


data_path = Path(os.path.join("data", data))
assert data_path.exists()

#Get the bindingDB class
data_class = LoadBindingDB(dataPath=data_path)

#Set up the test set
test = np.array(data_class.eSeqData['test'])

#Iterate on the separate possible methods
for dropout in [0.0,0.1,0.2,0.3,0.4]:
    for method in ['DTI_Bridge','p_Embedding_Bridge']:
        save_path = f"bindingdb_DO_{dropout}_model_{method}"
        if method == 'DTI_Bridge':
            for (kmers, pSeq) in [(True, True)]:
                for (FP, dSeq) in [(True, True)]:
                    if (kmers, pSeq, FP, dSeq) == (False, False, False, False):
                        break
                    pEmbeddings = False
                    ST_fingerprint = False
                    train_stats = []
                    valid_stats = []
                    test_stats = []
                    useFeatures={"pEmbeddings": pEmbeddings, "kmers": kmers, "pSeq": pSeq,
                                "FP": FP, "dSeq": dSeq, "ST_fingerprint": ST_fingerprint}
                    print(f'Training model with pEmbeddings: {useFeatures["pEmbeddings"]}, kmers: {useFeatures["kmers"]}, pSeq: {useFeatures["pSeq"]}, FP: {useFeatures["FP"]}, dSeq: {useFeatures["dSeq"]}, ST_fingerprint: {useFeatures["ST_fingerprint"]}\n')
                    for iter in range(3):
                        model = DTI_Bridge(outSize=128,
                            cHiddenSizeList=[1024],
                            fHiddenSizeList=[1024, 256],
                            fSize=1024, cSize=data_class.pContFeat.shape[1],
                            gcnHiddenSizeList=[128,128], fcHiddenSizeList=[128], nodeNum=64,
                            dropout=dropout, hdnDropout=0.5, fcDropout=0.5, device=torch.device('cuda'), 
                            useFeatures=useFeatures)
                        model.train(data_class, trainSize=512, batchSize=512, epoch=128,
                            stopRounds=-1, earlyStop=30,
                            savePath=save_path, metrics="AUC", report=["ACC", "AUC", "LOSS"],
                            preheat=0)
                        train_stats.append(model.final_res['training'])
                        valid_stats.append(model.final_res['valid'])
                        #Get test results
                        model.to_eval_mode()
                        Ypre, Y, seenbool = model.calculate_y_with_seenbool(data_class.one_epoch_batch_data_stream(batchSize=128, type='test', device=torch.device('cuda')))
                        metrictor = Metrictor()
                        metrictor.set_data(Ypre, Y)
                        test_stats.append([metrictor.ACC(), metrictor.AUC()])
                    train_mean = np.mean(np.array(train_stats), axis = 0)
                    valid_mean = np.mean(np.array(valid_stats), axis = 0)
                    test_mean = np.mean(np.array(test_stats), axis = 0)
                    log(dropout, useFeatures, train_mean, valid_mean, test_mean)
                    write_to_file(dropout, useFeatures, train_mean, valid_mean, test_mean, save_path)

        elif method == 'p_Embedding_Bridge':
            for (FP, dSeq) in [(True, False)]:
                    pEmbeddings = True
                    ST_fingerprint = False
                    kmers = False 
                    pSeq = False
                    train_stats = []
                    valid_stats = []
                    test_stats = []
                    useFeatures={"pEmbeddings": pEmbeddings, "kmers": kmers, "pSeq": pSeq,
                                "FP": FP, "dSeq": dSeq, "ST_fingerprint": ST_fingerprint}
                    print(f'Training model with pEmbeddings: {useFeatures["pEmbeddings"]}, kmers: {useFeatures["kmers"]}, pSeq: {useFeatures["pSeq"]}, FP: {useFeatures["FP"]}, dSeq: {useFeatures["dSeq"]}, ST_fingerprint: {useFeatures["ST_fingerprint"]}\n')
                    for iter in range(3):
                        model = p_Embedding_Bridge(outSize=128,
                            cHiddenSizeList=[1024],
                            fHiddenSizeList=[1024, 256],
                            fSize=1024, cSize=data_class.pContFeat.shape[1],
                            gcnHiddenSizeList=[128,128], fcHiddenSizeList=[128], nodeNum=64,
                            dropout=dropout, hdnDropout=0.5, fcDropout=0.5, device=torch.device('cuda'), 
                            useFeatures=useFeatures)
                        model.train(data_class, trainSize=512, batchSize=512, epoch=128,
                            stopRounds=-1, earlyStop=30,
                            savePath=save_path, metrics="AUC", report=["ACC", "AUC", "LOSS"],
                            preheat=0)
                        train_stats.append(model.final_res['training'])
                        valid_stats.append(model.final_res['valid'])
                        #Get test results
                        model.to_eval_mode()
                        Ypre, Y, seenbool = model.calculate_y_with_seenbool(data_class.one_epoch_batch_data_stream(batchSize=128, type='test', device=torch.device('cuda')))
                        metrictor = Metrictor()
                        metrictor.set_data(Ypre, Y)
                        test_stats.append([metrictor.ACC(), metrictor.AUC()])
                    train_mean = np.mean(np.array(train_stats), axis = 0)
                    valid_mean = np.mean(np.array(valid_stats), axis = 0)
                    test_mean = np.mean(np.array(test_stats), axis = 0)
                    log(dropout, useFeatures, train_mean, valid_mean, test_mean)
                    write_to_file(dropout, useFeatures, train_mean, valid_mean, test_mean, save_path)



