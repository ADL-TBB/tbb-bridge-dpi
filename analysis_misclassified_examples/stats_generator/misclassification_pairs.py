# Initialize connection with rest of lib
import logging
import rdkit
from utils import *
from DL_ClassifierModel import *
import os
from pathlib import Path
import torch



#Select the dataset and the path/pickle file where the data will be saved
model_path = "TEST_binding_db"
model = 'DTI_bridge' #Change it with pEmbeddings for when you run the test with those
dump_file = "misclass_pEmbeddings"

data_path = Path(os.path.join("data", dataset))

#Create the data class
data_class = LoadBindingDB(dataPath=data_path)

#Create the model
if model == 'DTI_bridge':
    model = DTI_Bridge(outSize=128,
                   cHiddenSizeList=[1024],
                   fHiddenSizeList=[1024, 256],
                   fSize=1024, cSize=data_class.pContFeat.shape[1],
                   gcnHiddenSizeList=[128, 128], fcHiddenSizeList=[128], nodeNum=64,
                   hdnDropout=0.5, fcDropout=0.5, device=torch.device('cuda'), save_d_names = True)

else:
    model = p_Embedding_Bridge(outSize=128,
                   cHiddenSizeList=[1024],
                   fHiddenSizeList=[1024, 256],
                   fSize=1024, cSize=data_class.pContFeat.shape[1],
                   gcnHiddenSizeList=[128, 128], fcHiddenSizeList=[128], nodeNum=64,
                   hdnDropout=0.5, fcDropout=0.5, device=torch.device('cuda'), save_d_names = True)

#Perform the test
model.load(path=model_path, map_location="cuda", dataClass=data_class)
model.to_eval_mode()


def get_miscl():
    #Get test set
    test = np.array(data_class.eSeqData['test'])
    
    #Predict on it
    print(len(test))
    Ypre, Y, _ = model.calculate_y_with_seenbool(data_class.one_epoch_batch_data_stream(batchSize=128, type='test', device=torch.device('cuda')))
    
    #Transform the probability predictions to 0 and 1
    Y_heavi = np.where(Ypre>=0.5, 1, 0)
    print(len(Y_heavi))
    pred_bool = np.where(Y_heavi != Y)
    examples_misclass = test[pred_bool]
    classes_misclass = Y[pred_bool]
    
    test_names = [(data_class.id2p[example[0]], data_class.id2d[example[1]]) for example in test] #all examples from test set
    misclass_couples = [(data_class.id2p[miscl[0]], data_class.id2d[miscl[1]]) for miscl in examples_misclass] #all examples from misclassified test
    dNames_misclass = [data_class.drug_names[miscl[1]] for miscl in examples_misclass] #drug names misclassified example
    drugs_test = [data_class.drug_names[ex[1]] for ex in test] #drug names of the whole test set 
    
    #Save test set and drug names 
    test_set_file = open('test.pkl', 'wb')
    pkl.dump([test_names, dNames_misclass, drugs_test] , test_set_file)
    test_set_file.close()
    
    #Save misclassified examples
    misclass_file = open(dump_file, 'wb')
    pkl.dump([misclass_couples, classes_misclass], misclass_file)
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


    





