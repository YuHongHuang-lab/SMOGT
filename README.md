# SMOGT
**Single‑cell Multi‑Omics Graph Transformer**

A tool for constructing hierarchical regulatory networks (HRNet) from single-cell multi-omics data and deciphering the mechanisms of cell fate decisions.

---

## Core Workflow
The entire pipeline is divided into three core scripts:
1. Model training: `SMOGT_Model.py`
2. Driver regulator identification: `SMOGT_MRWR_Driver_regulators.py`
3. Gene expression prediction and perturbation simulation: `SMOGT_BioStreamNet_TG_Regression.py`

All scripts include detailed comments for review and modification.

---

## Script Function Details

### 1. SMOGT_Model.py
**Primary Function**  
Complete pipeline from data preprocessing to obtaining low-dimensional embeddings for TFs, CREs, and Targets.

**Included Modules**
- Data preprocessing: `data_preprocess.R`
- Configuration: `config.yaml`
- Dataset generation
- Negative sampling
- Model training

**Output**  
Node embedding file for downstream analysis.

---

### 2. SMOGT_MRWR_Driver_regulators.py
**Primary Function**  
Construct HRNet using trained embeddings and perform multi-layer random walk with restart (MRWR).

**Key Biological Tasks**
1. Identify driver transcription factors (Driver TFs)
2. Identify driver enhancers (Driver CREs)
3. Predict TF perturbation target genes (candidates for perturbation simulation)

**Output**  
Regulatory score tables for TF‑Target and DEG‑CRE relationships.

---

### 3. SMOGT_BioStreamNet_TG_Regression.py
**Primary Function**  
Train gene-specific neural networks for each target gene based on HRNet.

**Key Functions**
1. Gene expression prediction
2. Genetic perturbation simulation (TF/CRE knockout/overexpression)
3. Genome-wide expression change prediction
4. Perturbation effect vector field plot generation
5. Gene-specific HRNet subnetwork generation

**Output**
- Gene prediction accuracy
- Trained models
- Perturbation simulation results
- Visualization files

---

## Usage
1. Modify corresponding paths and parameters according to your data
2. Run scripts directly
3. Example data is available on Zenodo for testing：https://zenodo.org/records/19111535

---

## Notes
Each script contains detailed comments explaining input/output formats, parameter meanings, and function usage.
