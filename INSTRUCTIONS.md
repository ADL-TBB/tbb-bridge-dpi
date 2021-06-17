# Instructions - The Bridge Builders

On the *dev* branch, all material necessary for the reproduction of the results can be found. 

**Setup files**

Files to set the Python working environment up are env.yml, poetry.lock and pyproject.toml. For more information on how to install the anaconda environment used for the experiments, read the README.md file included in the repository.

**Basic implementation files**

The implementation files are the scripts used to configure the input data, build the model, train and test it. 

-  nnLayer.py

  Contains a series of stand-alone Pytorch implementations of neural network architectures to integrate directly into models.

- Others.py

  Contains a basic implementation for learning rate schedule optimization used in DL_CassifierModel.py

- utils.py

  Contains the entire configuration of the data object. The mode of creation depends on both the input dataset used and the variant of the model trained. 

- DL_ClassifierModel.py

  Contains the implementation necessary to train the neural network architecture of interest over a user-defined number of epochs.

- train.py

  The basic script for training a single instance of the model. The input data and model type to train shall be specified within the script.

- main.py

  The basic testing script. After training, a pkl file with the model parameters will be dropped in a pre-defined directory. The model can be imported by main.py to test the performance on the test and validation set. 

- metrics.py

  Contains the besic implementation of train and test performance evaluation metrics (AUC, ACC, Precision, Recall, F1, Loss) and plotting functions for the loss and AUC during training.

**Model combination and parameter exploration**

A series of scripts to test different combinations of datasets, drug and protein embedding creation methods and hyperparameters thereof.

- combination_performance.py

  Train and test all combinations of modules and sub-modules for protein and drug embedding creation by averaging their performance over three independent runs.

- dropout_performance.py 

  Experiment with the dropout value of the MLP networks involved in feature creation for proteins and drugs.

- hypernodes_performance.py

  Experiment with the number of hypernodes in the graph neural network architecture.

- crossdata_experiment.py

  Experiment where the neural network model is trained on one specific dataset and its test performance is tested on another.

**Case studies**

Two case studies concerning SARS-CoV-2 and Mycobacterium tuberculosis are  implemented in the following scripts:

- predict_mtb.py
- sarscov_experiment.py









