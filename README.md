# The Bridge Builders
This repo houses an academic project for the seminar course at Leiden University, Advances in Deep Learning (2021).
### Members

- Alessandro Palma
- Heleen Severin
- Julius Cathalina
- Laurens Engwegen
- Stijn Oudshoorn*


## Getting started
- Make sure you have conda installed on your machine. You can download it [here](https://conda.io/projects/conda/en/latest/user-guide/install/index.html)
- Install the conda environment by running `conda env create -f env.yml`
- Install the rest of the packages by running `poetry install`
- Install torch with cudatoolkit for version 11 using poe get-torch-cuda
- Get gxx for linux (for rdkit support) in conda by running poe get-gxx
- Install seaborn via `pip install seaborn`

## ReferencesInstall torch with cudatoolkit for version 11 using poe get-torch-cuda
- The original repo that our project is based on can be found [here](https://github.com/DeepAAI/BridgeDPI) along with its [paper](https://arxiv.org/abs/2101.12547) [1].

1) Wu, Y., Gao, M., Zeng, M., Chen, F., Li, M. and Zhang, J., 2021. BridgeDPI: A Novel Graph Neural Network for Predicting Drug-Protein Interactions. arXiv preprint arXiv:2101.12547.
