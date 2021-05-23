# Initialize connection with rest of lib
import logging
import rdkit
from utils import *
from DL_ClassifierModel import *
import os
from pathlib import Path
import torch

log_file = 'cele_hum_baseline.log'
log_format = '%(asctime)s : %(levelname)s : %(message)s'
fhandler = logging.FileHandler(filename=log_file, mode='a')
logging.basicConfig(format=log_format, filename=log_file,  level=logging.DEBUG)

logger = logging.getLogger()
logger.addHandler(fhandler)

def log(data, setting, training_data, valid_data, test_data):
    logger.info(f'Averaged results')
    logger.info(f'Data: {data}, Conditions --> pEmbeddings: {setting["pEmbeddings"]}, kmers: {setting["kmers"]}, '
                f'pSeq: {setting["pSeq"]}, FP: {setting["FP"]}, dSeq: {setting["dSeq"]}, ST_fingerprint: {setting["ST_fingerprint"]}\n')
    logger.info(f'Training --> AUC = {format(training_data[1], ".3f")}, ACC = {format(training_data[0], ".3f")}, '
                f'Precision = {format(training_data[2], ".3f")}, Recall = {format(training_data[3], ".3f")}, F1 = {format(training_data[4], ".3f")}\n')
    logger.info(f'Validation --> AUC = {format(valid_data[1], ".3f")}, ACC = {format(valid_data[0], ".3f")}, '
                f'Precision = {format(valid_data[2], ".3f")}, Recall = {format(valid_data[3], ".3f")}, '
                f'F1 = {format(valid_data[4], ".3f")}\n')
    logger.info(f'Test --> AUC = {format(test_data[1], ".3f")}, ACC = {format(test_data[0], ".3f")}, '
                f'Precision = {format(test_data[2], ".3f")}, Recall = {format(test_data[3], ".3f")}, F1 = {format(test_data[4], ".3f")}\n')
    logger.info('\n')

def log_per_iteration(data, setting, train, valid, test):
    logger.info(f'Single iteration')
    logger.info(f'Data: {data}, Conditions --> pEmbeddings: {setting["pEmbeddings"]}, kmers: {setting["kmers"]}, '
                f'pSeq: {setting["pSeq"]}, FP: {setting["FP"]}, dSeq: {setting["dSeq"]}, ST_fingerprint: {setting["ST_fingerprint"]}\n')
    logger.info(f'Training --> AUC = {train[1]}, ACC = {train[0]}, Precision = {train[2]}, Recall = {train[3]}, F1 = {train[4]}\n')
    logger.info(f'Validation --> AUC = {valid[1]}, ACC = {valid[0]}, Precision = {valid[2]}, Recall = {valid[3]}, F1 = {valid[4]}\n')
    logger.info(f'Test --> AUC = {test[1]}, ACC = {test[0]}, Precision = {test[2]}, Recall = {test[3]}, F1 = {test[4]}\n')
    logger.info('\n')


report = ["ACC", "AUC", "LOSS", "Precision", "Recall", "F1"]

for data in ["celegans", "human"]:

    data_path = Path(os.path.join("data", data))
    assert data_path.exists()

    # Get the human/celegans class
    data_class = LoadCelegansHuman(dataPath=data_path)

    # Set up the test set
    test = np.array(data_class.eSeqData['test'])

    train_stats = []
    valid_stats = []
    test_stats = []
    useFeatures = {"pEmbeddings": False, "kmers": True, "pSeq": True,
                   "FP": True, "dSeq": True, "ST_fingerprint": False}
    print(
        f'Training model with pEmbeddings: {useFeatures["pEmbeddings"]}, kmers: {useFeatures["kmers"]}, pSeq: {useFeatures["pSeq"]}, FP: {useFeatures["FP"]}, dSeq: {useFeatures["dSeq"]}, ST_fingerprint: {useFeatures["ST_fingerprint"]}\n')
    for iter in range(3):
        save_path = "baseline_" + data + f"_{iter}"

        model = DTI_Bridge(outSize=128,
                           cHiddenSizeList=[1024],
                           fHiddenSizeList=[1024, 256],
                           fSize=1024, cSize=data_class.pContFeat.shape[1],
                           gcnHiddenSizeList=[128, 128], fcHiddenSizeList=[128], nodeNum=64,
                           hdnDropout=0.5, fcDropout=0.5, device=torch.device('cuda'),
                           useFeatures=useFeatures)
        avg_results = model.cv_train(data_class, trainSize=512, batchSize=512, epoch=128,
                    stopRounds=-1, earlyStop=30,
                    savePath=save_path, metrics="AUC", report=report,
                    preheat=0, kFold=5)

        iteration_train = []
        iteration_valid = []

        for met in report:
            iteration_train.append(avg_results['train'][met])
            iteration_valid.append(avg_results['valid'][met])
        train_stats.append(iteration_train)
        valid_stats.append(iteration_valid)

        # # Get test results
        model.to_eval_mode()
        Ypre, Y, seenbool = model.calculate_y_with_seenbool(
            data_class.one_epoch_batch_data_stream(batchSize=128, type='test', device=torch.device('cuda')))
        metrictor = Metrictor()
        metrictor.set_data(Ypre, Y)
        test_stats.append([metrictor.ACC(), metrictor.AUC(), metrictor.Precision(), metrictor.Recall(), metrictor.F1()])

        log_per_iteration(data, useFeatures, iteration_train, iteration_valid,
                          [metrictor.ACC(), metrictor.AUC(), metrictor.Precision(), metrictor.Recall(), metrictor.F1()])
        print(f'done iteration {iter} on test {data}')

    train_mean = np.mean(np.array(train_stats), axis=0)
    valid_mean = np.mean(np.array(valid_stats), axis=0)
    test_mean = np.mean(np.array(test_stats), axis=0)
    log(data, useFeatures, train_mean, valid_mean, test_mean)
