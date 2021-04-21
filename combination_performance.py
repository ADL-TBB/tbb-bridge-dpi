# Initialize connection with rest of lib
import logging
import rdkit
from utils import *
from DL_ClassifierModel import *
import os
from pathlib import Path
import torch

log_file = 'results_combination_of_models.log'
log_format = '%(asctime)s : %(levelname)s : %(message)s'
logging.basicConfig(format=log_format, filename = log_file, level=logging.DEBUG)
logger = logging.getLogger()

def log(setting, training_data, valid_data, test_data):
    logger.debug(f'Conditions --> pEmbeddings: {setting["pEmbeddings"]}, kmers: {setting["kmers"]}, pSeq: {setting["pSeq"]}, FP: {setting["FP"]}, dSeq: {setting["dSeq"]}, ST_fingerprint: {setting["ST_fingerprint"]}\n')
    logger.info(f'Training --> AUC = {training_data[0]}, ACC = {training_data[1]}\n')
    logger.info(f'Validation --> AUC = {valid_data[0]}, ACC = {valid_data[1]}\n')
    logger.info(f'Test --> AUC = {test_data[0]}, ACC = {test_data[1]}\n')
    logger.info('\n')



#Training data binding DB
data = "bindingdb"
save_path = "TEST_bindingdb"


data_path = Path(os.path.join("data", data))
assert data_path.exists()

#Get the bindingDB class
data_class = LoadBindingDB(dataPath=data_path)

#Set up the test set
test = np.array(data_class.eSeqData['test'])

#Iterate on the separate possible methods
for method in ['DTI_Bridge', 'ST_Bridge', 'p_Embedding_Bridge', 'p_Emb_ST_Bridge']:
    if method == 'DTI_Bridge':
        for (kmers, pSeq) in [(True, True), (True, False), (False, True), (False, False)]:
            for (FP, dSeq) in [(True, True), (True, False), (False, True), (False, False)]:
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
                        gcnHiddenSizeList=[128, 128], fcHiddenSizeList=[128], nodeNum=64,
                        hdnDropout=0.5, fcDropout=0.5, device=torch.device('cuda'), 
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
                log(useFeatures, train_mean, valid_mean, test_mean)

    elif method == 'ST_Bridge':
        for (kmers, pSeq) in [(True, True), (True, False), (False, True), (False, False)]:
                pEmbeddings = False
                ST_fingerprint = True
                dSeq = False
                FP = False
                train_stats = []
                valid_stats = []
                test_stats = []
                useFeatures={"pEmbeddings": pEmbeddings, "kmers": kmers, "pSeq": pSeq,
                              "FP": FP, "dSeq": dSeq, "ST_fingerprint": ST_fingerprint}
                print(f'Training model with pEmbeddings: {useFeatures["pEmbeddings"]}, kmers: {useFeatures["kmers"]}, pSeq: {useFeatures["pSeq"]}, FP: {useFeatures["FP"]}, dSeq: {useFeatures["dSeq"]}, ST_fingerprint: {useFeatures["ST_fingerprint"]}\n')
                for iter in range(3):
                    model = ST_Bridge(outSize=128,
                        cHiddenSizeList=[1024],
                        fHiddenSizeList=[1024, 256],
                        fSize=1024, cSize=data_class.pContFeat.shape[1],
                        gcnHiddenSizeList=[128, 128], fcHiddenSizeList=[128], nodeNum=64,
                        hdnDropout=0.5, fcDropout=0.5, device=torch.device('cuda'), 
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
                log(useFeatures, train_mean, valid_mean, test_mean)
    
    elif method == 'p_Embedding_Bridge':
        for (FP, dSeq) in [(True, True), (True, False), (False, True), (False, False)]:
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
                        gcnHiddenSizeList=[128, 128], fcHiddenSizeList=[128], nodeNum=64,
                        hdnDropout=0.5, fcDropout=0.5, device=torch.device('cuda'), 
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
                log(useFeatures, train_mean, valid_mean, test_mean)
    

    elif method == 'p_Emb_ST_Bridge':    
        pEmbeddings = True
        ST_fingerprint = True
        FP = False
        dSeq = False
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
                gcnHiddenSizeList=[128, 128], fcHiddenSizeList=[128], nodeNum=64,
                hdnDropout=0.5, fcDropout=0.5, device=torch.device('cuda'), 
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
        log(useFeatures, train_mean, valid_mean, test_mean)    

                




    





