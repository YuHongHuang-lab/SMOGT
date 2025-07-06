# SMOGT
## Introduction
SMOGT（Single-cell Multo-Omics Graph Transformer） constructs a heterogeneous graph learning architecture guided by epigenetic mechanisms. It is a novel method for building high-precision gene regulatory networks from single-cell multi-omics data.
## downstream module
- BioStreamNet: Predict target gene expression
- RWR_Pert: Identify TF perturbation targets
- RWR_CTS: Discover driver regulatory factors
- Louvain_Co-CRE: Analyze disease-related co-regulatory modules
## Contact ##
1395214203@qq.com
## Basic Usage ##
### 1. Dependencies
You can use the following code to create a conda environment named SMOGT and download the dependencies.
```sh
conda create -n SMOGT python==3.10.0
conda activate SMOGT
conda install scikit-learn
conda install joblib
conda install yaml
conda install matplotlib
conda install pandas
pip install torch
pip install torch-geometric
pip install numpy
pip install networkx
pip install cvxpy
pip install tqdm
pip install rich
pip install dask
pip install distributed  
```
### 2. Installation
The source code of SMOGT is freely available at https://github.com/YuHongHuang-lab/SMOGT  you can use the code .
```sh
cd /your working path/ 
git clone https://github.com/YuHongHuang-lab/SMOGT
```
### 3. prepare the configuration file
Prepare a configuration file similar to the provided config.yaml, including data paths, model parameters, and training settings.
### 4.Data preprocessing
You can obtain our project's data through this URL: https://zenodo.org/records/15816796
```sh
Rscript data_prepare.R
```
### 5. Model Training
```sh
python main.py
```

