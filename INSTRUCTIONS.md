# Instructions - The Bridge Builders

In the *dev* branch, all material necessary for the reproduction of the results can be found. 

**Setup files**

The files to set the Python working environment up are env.yml, poetry.lock and pyproject.toml. For more information on how to install the anaconda environment used for the experiments, read the README.md file included in the repository.

**Basic implementation files**

The implementation files are the scripts used to configure the input data, build the model, train and test it. 

- nnLayer.py

  Contains a series of stand-alone Pytorch implementations of neural network architectures to integrate directly into models.

- Others.py

  Contains a basic implementation for optimal learning rate scheduling used in DL_CassifierModel.py

- utils.py

  Contains the entire configuration of the data object. The mode of creation depends on both the input dataset used and the variant of the model trained. 

- DL_ClassifierModel.py

  Contains the implementation necessary to train the neural network architecture of interest over a user-defined number of epochs.

- train.py

  The basic script for training a single instance of the model. The input data and model type to train shall be specified within the script.

- main.py

  The basic testing script. After training, a .pkl file with the model parameters will be dropped in a pre-defined directory. The model can be imported by main.py to test the performance on the test and validation set. 

- metrics.py

  Contains the basic implementation of train and test performance evaluation metrics (AUC, ACC, Precision, Recall, F1, Loss) and plotting functions for the loss and AUC during training.

**Model combination and parameter exploration**

A series of scripts to test different combinations of datasets, drug and protein embedding creation methods and hyperparameters.

- combination_performance.py

  Train and test all combinations of modules and sub-modules for protein and drug embedding creation by averaging their performance over three independent runs.

- dropout_performance.py 

  Experiments with the dropout value of the MLP networks involved in feature creation of proteins and drugs.

- hypernodes_performance.py

  Experiment with the number of hypernodes in the graph neural network architecture.

- crossdata_experiment.py

  Experiment where the neural network model is trained on one specific dataset and its performance is tested on another.
  
- crossdata_experiment_chembl.py

  Experiment using the model trained on our chEMBL dataset on BindingDB and human datasets.

**Case studies**

Two case studies concerning SARS-CoV-2 and Mycobacterium tuberculosis are implemented in the following scripts:

- predict_mtb.py
- sarscov_experiment.py

**Folder 1:  analysis_misclassified_examples**

Comparison of the characteristics of misclassified examples between a run with the original model and one with a modified version using *SeqVec* [1] and *Morgan fingerprints* to embed proteins and drugs, respectively.

- analysis folder

  Contains the notebook with the inspection of the misclassified examples and all files necessary to yield it. An additional descriptive file called *file_description.txt* is included in such folder.

- stats_generator folder

  Contains the script used to generate the analyzed data and the trained models (.pkl files) employed to perform the prediction. 

**Folder 2: data**

Contains all data employed in the project. 

- binindgdb

  Refined dataset from https://github.com/IBM/InterpretableDTIP with protein/drug examples divided into train, test and validation sub-folders.

- chembl

  Personally created interaction dataset with active and inactive drug-protein compounds from the chEMBL database.

- celegans / human

  Balanced datasets of protein-compound interactions from https://github.com/masashitsubaki/CPI_prediction/tree/master/dataset with human and C.elegans examples.

- embedding_files

  Pre-embedded protein datasets using *SeqVec* as in https://github.com/Rostlab/SeqVec

- smiles_trfm_model

  Pre-trained model for embedding drugs as described in [2].

- sarscov2

  The dataset of the protein-drug interaction examples studied in SARS-CoV-2

- mtb

  The dataset of the protein-drug interaction examples studied in tuberculosis. 

  

**Folder 3: smiles_transformer**

Contains the necessary scripts to embed drug SMILES using a pre-trained transformer model from [2].








1. M. Heinzinger, A. Elnaggar, Y. Wang, C. Dallago, D. Nechaev, F. Matthes and B. Rost, 2019, Modeling aspects of the language of life through transfer-learning protein sequences

2. S. Honda, S. Shi,, H.R. Ueda, 2019, SMILES Transformer: Pre-trained Molecular Fingerprint
   for Low Data Drug Discovery









