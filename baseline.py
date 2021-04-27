# Initialize connection with rest of lib
import logging
import rdkit
from utils import *
from DL_ClassifierModel import *
import os
from pathlib import Path
import torch

log_file = 'baseline.log'
log_format = '%(asctime)s : %(levelname)s : %(message)s'
fhandler = logging.FileHandler(filename='mylog.log', mode='a')
logging.basicConfig(format=log_format, filename=log_file,  level=logging.DEBUG)

logger = logging.getLogger()
logger.addHandler(fhandler)

def log(setting, training_data, valid_data, test_data):
    logger.debug(f'Conditions --> pEmbeddings: {setting["pEmbeddings"]}, kmers: {setting["kmers"]}, pSeq: {setting["pSeq"]}, FP: {setting["FP"]}, dSeq: {setting["dSeq"]}, ST_fingerprint: {setting["ST_fingerprint"]}\n')
    logger.info(f'Training --> AUC = {training_data[0]}, ACC = {training_data[1]}\n')
    logger.info(f'Validation --> AUC = {valid_data[0]}, ACC = {valid_data[1]}\n')
    logger.info(f'Test --> AUC = {test_data[0]}, ACC = {test_data[1]}\n')
    logger.info('\n')


#Training data binding DB
data = "celegans"
# data = "human"



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
        model.train(data_class, trainSize=512, batchSize=512, epoch=128,
                    stopRounds=-1, earlyStop=30,
                    savePath=save_path, metrics="AUC", report=["ACC", "AUC", "LOSS"],
                    preheat=0)
        train_stats.append(model.final_res['training'])
        valid_stats.append(model.final_res['valid'])
        # Get test results
        model.to_eval_mode()
        Ypre, Y, seenbool = model.calculate_y_with_seenbool(
            data_class.one_epoch_batch_data_stream(batchSize=128, type='test', device=torch.device('cuda')))
        metrictor = Metrictor()
        metrictor.set_data(Ypre, Y)
        test_stats.append([metrictor.ACC(), metrictor.AUC()])
        print(f'done iteration {iter} on test {data}')

    train_mean = np.mean(np.array(train_stats), axis=0)
    valid_mean = np.mean(np.array(valid_stats), axis=0)
    test_mean = np.mean(np.array(test_stats), axis=0)
    log(useFeatures, train_mean, valid_mean, test_mean)

