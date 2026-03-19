# The SMOGT_BioStreamNet_TG_Regression.py script fully implements two core functions:
# gene expression prediction and cell fate perturbation simulation based on Hierarchical Regulatory Network (HRNet).
# It uses the hierarchical regulatory network learned by SMOGT to quantify the regulatory strength of TFs and CREs on target genes
# through a neural network model.
# Based on this, it simulates genetic perturbations and predicts changes in cell states.
# This script serves as a key bridge connecting the SMOGT model with biological mechanism analysis.
# Its outputs can be directly used to generate vector field plots and gene regulatory subnetworks shown in the paper,
# providing a computational tool for understanding the causal logic of cell fate decisions.

import os

import numpy as np

from model.BioStreamNet import *
import pandas as pd
import gc
import scanpy as sc

import multiprocessing
from functools import partial

# The sample version of this module file (with consistent naming) has been uploaded to Zenodo (https://zenodo.org/records/19111535)

# File directory configuration
output_dir = '/mnt/data/home/tycloud/workspace/algorithms_raw_paper_4/data/HSPC_preturb'

# BioStreamNet fits a model for each target gene (TG), and the model parameters are saved in the models_and_networks_p directory
models_and_networks_dir = os.path.join(output_dir, "models_and_networks_p")
os.makedirs(models_and_networks_dir, exist_ok=True)

# Read the expression matrix — graph_0_nodes
all_nodes_path = '/mnt/data/home/tycloud/workspace/algorithms_raw_paper_4/data/HSPC_preturb/raw_600/graph_0_nodes.csv'
# Read the nodes used for training; the rest are used for testing
train_nodes_path = '/mnt/data/home/tycloud/workspace/algorithms_raw_paper_4/data/HSPC_preturb/train_id.csv'
node_names_path = '/mnt/data/home/tycloud/workspace/algorithms_raw_paper_4/data/HSPC_preturb/raw_600/graph_0_node_names.csv'

# Load HRNet (see details in SMOGT_MRWR_Driver_regulators.py)
edges_path = '/mnt/data/home/tycloud/workspace/algorithms_raw_paper_4/data/HSPC_preturb/GRN_df_p.csv'

# According to the number of CREs linked to each target gene (TG) in the prior network,
# only TGs with CRE counts above a specified threshold (typically 3 or 4) are retained for training and testing.
gene_names_path = '/mnt/data/home/tycloud/workspace/algorithms_raw_paper_4/data/HSPC_preturb/predicted_genes.txt'

common_seacells_mat_train = pd.read_csv(
    train_nodes_path,
    index_col=0)
train_id = common_seacells_mat_train['x'].tolist()
del common_seacells_mat_train

common_seacells_mat = pd.read_csv(all_nodes_path,
                                  index_col=0)

# Generate training file
train_df = pd.DataFrame({'id' : common_seacells_mat.columns,
                        'train' : [id in train_id for id in common_seacells_mat.columns]})
train_df['train'] = train_df['train'].astype(int)

Genes_Peaks_df = pd.read_csv(node_names_path,
                             index_col=0)
TF = Genes_Peaks_df.loc[Genes_Peaks_df['type']=='TF', 'name'].to_list()
Target = Genes_Peaks_df.loc[Genes_Peaks_df['type']=='Target', 'name'].to_list()
CRE = Genes_Peaks_df.loc[Genes_Peaks_df['type']=='CRE', 'name'].to_list()

GRN_df = pd.read_csv(edges_path)

# Prepare expression matrix
common_seacells_mat_TF = common_seacells_mat.loc[TF, ]
common_seacells_mat_Target = common_seacells_mat.loc[Target, ]
common_seacells_mat_CRE = common_seacells_mat.loc[CRE, ]

TF_TF_network = GRN_df.loc[GRN_df['edge_id_type']=='TF_TF', ['from', 'to']]
TF_TF_network['score'] = 1

CRE_Target_network = GRN_df.loc[GRN_df['edge_id_type']=='CRE_Target', ['from', 'to']]
CRE_Target_network['score'] = 1

CRE_TF_network = GRN_df.loc[GRN_df['edge_id_type']=='CRE_TF', ['from', 'to']]
CRE_TF_network['score'] = 1

TF_CRE_network = GRN_df.loc[GRN_df['edge_id_type']=='TF_CRE', ['from', 'to']]
TF_CRE_network['score'] = 1

CRE_CRE_network = GRN_df.loc[GRN_df['edge_id_type']=='CRE_CRE', ['from', 'to']]
CRE_CRE_network['score'] = 1

# Convert HRNet to GlobalNetworkManager object
global_network = GlobalNetworkManager(
    TF_TF_network=TF_TF_network,
    CRE_CRE_network=CRE_CRE_network,
    TF_CRE_network=TF_CRE_network,
    CRE_TF_network=CRE_TF_network,
    CRE_Target_network=CRE_Target_network,
    TF_names=TF,
    CRE_names=CRE,
    Target_names=Target
)

# Create predictor
predictor = GeneExpressionPredictor(global_network)

# Load the required data into the predictor
predictor.prepare_data(tf_expr=common_seacells_mat_TF,
                       cre_expr=common_seacells_mat_CRE,
                       target_expr=common_seacells_mat_Target,
                       train_df=train_df)

gene_names = pd.read_csv(gene_names_path,
                         header=None, names=['gene'])
gene_names = gene_names['gene'].tolist()

# Start training for each gene
results = []
for i, gene_name in enumerate(tqdm(gene_names, desc="Training genes")):
    print(f"\n[{i + 1}/{len(gene_names)}] 正在训练基因: {gene_name}")

    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        model, evaluation = predictor.train_and_evaluate(
            gene_name=gene_name,
            batch_size=256,
            lr=0.001,
            epochs=250,
            save_dir="/mnt/data/home/tycloud/workspace/algorithms_raw_paper_4/data/HSPC_predict/models"
        )

        correlation = evaluation['correlation']
        p_value = evaluation.get('p_value', np.nan)

        # Release model memory
        del model, evaluation
        gc.collect()

    except Exception as e:
        print(f"训练基因 {gene_name} 时发生错误: {str(e)}")
        correlation = 0.0
        p_value = np.nan

    # Save results
    results.append({
        'gene': gene_name,
        'correlation': correlation,
        'p_value': p_value
    })

# Create final results
results_df = pd.DataFrame(results)
results_df = results_df.dropna(subset=['correlation'])
np.mean(results_df['correlation'])

results_df.to_csv(os.path.join(output_dir, 'gene_prediction_our.csv'), index=False)


gene_prediction = pd.read_csv(os.path.join(output_dir, 'gene_prediction_our_parallel.csv'))
gene_list = gene_prediction.loc[gene_prediction['correlation']>0, 'gene']
gene_list = gene_list.tolist()

# Read the single-cell dataset (scATAC+scRNA) in AnnData format
adata_RNA_BM_seacells_600_ad = sc.read_h5ad('/mnt/data/home/tycloud/workspace/algorithms_raw_paper_4/data/seob_RNA_BM_seacells_600_ad_reduce.h5ad')

# Perform perturbation analysis. For detailed parameter settings, refer to the usage of the simulate_perturbation function
gene_predict_df = simulate_perturbation(perturbations={'NFIA':2},
                                        predictor=predictor,
                                        original_tf_expr_df=common_seacells_mat_TF,
                                        original_cre_expr_df=common_seacells_mat_CRE,
                                        models_dir=models_and_networks_dir,
                                        gene_list=gene_list,
                                        ncores=1)

# Perform multiple iterations to simulate long-range regulatory effects of the network
# common_seacells_mat_TF_new = common_seacells_mat_TF.copy()
# common_seacells_mat_TF_new.update(gene_predict_df)
#
# gene_predict_df = simulate_perturbation(perturbations={'chr16-85882146-85883060':2},
#                                         predictor=predictor,
#                                         original_tf_expr_df=common_seacells_mat_TF_new,
#                                         original_cre_expr_df=common_seacells_mat_CRE,
#                                         models_dir=models_and_networks_dir,
#                                         gene_list=gene_list,
#                                         ncores=1)

gene_predict_df.to_csv(os.path.join(output_dir,'NFIA_2.csv'))

# gene_predict_df = pd.read_csv(os.path.join(output_dir, 'gene_preturb_our_parallel_KLF1.csv'))


# Generate perturbation plots.
# For detailed parameter settings, refer to the usage of the run_perturbation_analysis function
run_perturbation_analysis(
    perturbations={'KLF1':0},
    predictor=predictor,
    original_tf_expr_df=common_seacells_mat_TF,
    original_cre_expr_df=common_seacells_mat_CRE,
    original_target_expr_df=common_seacells_mat_Target,
    models_dir=models_and_networks_dir,
    adata=adata_RNA_BM_seacells_600_ad,
    output_dir=output_dir,
    ko_results_path=os.path.join(output_dir, 'NFIA_2.csv'),
    gene_list=gene_names,
    embedding_key='X_umap',
    metadata_key='seurat_clusters',
    n_cpu=1,
    tf_name='NFIA_2')


# Extract HRNet for a specific gene
# KLF1
learned_network_df = predictor.analyze_weights(
    gene_name='KLF1',
    model_path=os.path.join(output_dir, 'models_and_networks_p/KLF1_model.pth')
)

learned_network_df.to_csv(os.path.join(output_dir, 'GRN_KLF1.csv'))


# IRF8
learned_network_df = predictor.analyze_weights(
    gene_name='IRF8',
    model_path=os.path.join(output_dir, 'models_and_networks_p/IRF8_model.pth')
)

learned_network_df.to_csv(os.path.join(output_dir, 'GRN_IRF8.csv'))
# endregion
