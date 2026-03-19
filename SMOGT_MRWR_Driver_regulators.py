# SMOGT_MRWR_Driver_regulators.py is the core downstream analysis script of the SMOGT framework.
# It aims to construct a Hierarchical Regulatory Network (HRNet) using trained node embeddings,
# and accomplishes three key biological tasks through Multi‑Layer Random Walk with Restart (MRWR):
#
# 1. Identify driver transcription factors (Driver TFs): Based on cell type-specific differentially expressed genes (DEGs),
#    perform reverse random walk from the target gene layer to the TF layer to identify TFs that play core regulatory roles
#    in cell fate transitions.
#
# 2. Identify driver enhancers (Driver CREs): Similarly, perform random walk from DEGs to the CRE layer to identify
#    potential key regulatory elements in non-coding regions.
#
# 3. Predict TF perturbation target genes: Using TFs of interest as seeds, perform forward random walk from the TF layer
#    to the target gene layer to simulate the set of genes potentially affected by TF overexpression or knockout,
#    providing candidate targets for subsequent perturbation modeling (e.g., BioStreamNet).
#
# The script converts SMOGT-output node embeddings (e.g., avg_df_42_BM_preturb.csv) into a multilayer network format
# readable by HuMMuS/MultiXrank, and calls core_grn.get_output_from_dicts to perform random walks.
# It returns a dataframe containing regulatory relationship scores, which can be directly used for downstream analysis
# or experimental validation.

import numpy as np
import pandas as pd

from hummuspy import *
import torch.optim as optim
from dataset.bio_dataset import *
from model.GraphAutoencoder import *
from train import *
from utils import *
from hummuspy import *

config_path = "../config.yaml"
config = load_config(config_path)


# The sample version of this module file (with consistent naming) has been uploaded to Zenodo (https://zenodo.org/records/19111535)

# Create EvaluationBuider object to evaluate whether node latent representations reflect true regulatory relationships
# For detailed parameter settings, refer to the usage of the simulate_perturbation function
EBuider = EvaluationBuider(config=config,
                           embedding_path='/mnt/data/home/tycloud/workspace/algorithms_raw_paper_4/data/avg_df_42_BM_preturb.csv',
                           negative_ratio=1)
# Load node embeddings
EBuider.load_embedding()

# Calculate AUC and AUPR
result = EBuider.analysis(edge_types=['TF-CRE', 'CRE-CRE'])


# Create NetworkBuider object to construct HRNet based on node low-dimensional embeddings
buider = NetworkBuider(config=config,
                       edge_types=[['TF','TF'], ['TF','CRE'], ['CRE', 'CRE'],
                                   ['CRE', 'Target'], ['CRE', 'TF'], ['Target', 'Target']],
                       threshold={'TF_TF':0.999,
                         'TF_CRE':0.8,
                         'CRE_CRE':0.9,
                         'CRE_Target':0.2,
                         'CRE_TF':0.2,
                         'Target_Target':0.999},
                       method='dot',
                       topp_CRE=0.01)

# Load node low-dimensional representations
buider.load_data(file_path='/mnt/data/home/tycloud/workspace/algorithms_raw_paper_4/data/avg_df_42_BM_preturb.csv')

# Start construction
buider.buide_networks()

# Save networks
buider.save_networks(base_dir='/mnt/data/home/tycloud/workspace/algorithms_raw_paper_4/data/HSPC_preturb/processed/hummus_BM')


# Multiplex Networks configuration
# Source: HuMMuS — Paper: 10.1093/bioinformatics/btae143; Code: https://github.com/cantinilab/HuMMuS

# Value: a dictionary describing each layer within the multiplex network.
#      - Key: layer name (usually the same as the multiplex name, as each multiplex here has one layer).
#      - Value: graph type string consisting of two characters:
#           First digit: 0 = undirected graph, 1 = directed graph;
#           Second digit: 0 = unweighted graph, 1 = weighted graph.
#           Examples:
#             '00' = undirected unweighted graph
#             '01' = undirected weighted graph
#             '10' = directed unweighted graph
#             '11' = directed weighted graph
multiplexes_dictionary = {
    'TF': {'TF': '00'}, # TF multiplex, containing a layer named 'TF', as an undirected unweighted graph
    'Target': {'Target': '00'}, # Target multiplex, containing a layer named 'Target', as an undirected unweighted graph
    'CRE': {'CRE': '01'} # CRE multiplex, containing a layer named 'CRE', as an undirected weighted graph (edges have weights)
}

# Bipartite Networks configuration dictionary.
# Key: name of the bipartite edge file (e.g., 'TF_CRE.csv'), which should be located in the specified bipartite network folder.
# Value: a dictionary specifying the two multiplex networks connected by this bipartite edge.
#      - 'multiplex_right': name of the right multiplex network (i.e., target node type in the edge file).
#      - 'multiplex_left' : name of the left multiplex network (i.e., source node type in the edge file).
# Note: left/right here do not indicate directionality, but rather distinguish which multiplex the two endpoints belong to.
#       The actual direction of edges is determined by the column order in the bipartite edge file (typically first column as source, second as target).
bipartites_dictionary = {
    'TF_CRE.csv': { # Bipartite edge file connecting TF and CRE
        'multiplex_right': 'TF', # Right multiplex is 'TF', meaning target nodes in the edge file belong to the TF multiplex
        'multiplex_left': 'CRE' # Left multiplex is 'CRE', meaning source nodes in the edge file belong to the CRE multiplex
    },
    'CRE_Target.csv': { # Bipartite edge file connecting CRE and Target
        'multiplex_right': 'CRE', # Right multiplex is 'CRE'
        'multiplex_left': 'Target' # Left multiplex is 'Target'
    }
}


# Call functions from the core_grn module to perform random walk analysis based on the multilayer network configuration,
# predicting target genes or target enhancers associated with a given TF list.
# This function calculates the probability score of reaching each target gene for each TF,
# returning a DataFrame containing these prediction results.
tf_file = '/mnt/data/home/tycloud/workspace/algorithms_raw/data/BM_DEGs_6.txt'
with open(tf_file, 'r') as f:
    tf_list = [line.strip() for line in f.readlines()]

#Target gene
output = core_grn.get_output_from_dicts(
        output_request='target_genes',
        multilayer_f='/mnt/data/home/tycloud/workspace/algorithms_raw/data/HSPC/processed/hummus_BM_reverse',
        multiplexes_list=multiplexes_dictionary,
        bipartites_list=bipartites_dictionary,
        bipartites_type=('00','01'),
        gene_list=None,
        tf_list=tf_list,
        config_filename='target_genes_config.yml',
        config_folder='config',
        output_f=None,
        tf_multiplex='TF',
        peak_multiplex='CRE',
        rna_multiplex='Target',
        update_config=True,
        save=False,
        return_df=True,
        njobs=18)

output.to_csv('/mnt/data/home/tycloud/workspace/algorithms_raw/data/BM_DEGs_6_preturb.txt',
                   sep='\t', index=False)

#Target enhancer
output = core_grn.get_output_from_dicts(
        output_request='binding_regions',
        multilayer_f='/mnt/data/home/tycloud/workspace/algorithms_raw/data/HSPC/processed/hummus_BM_reverse',
        multiplexes_list=multiplexes_dictionary,
        bipartites_list=bipartites_dictionary,
        bipartites_type=('00','01'),
        gene_list=None,
        tf_list=tf_list,
        config_filename='target_genes_config.yml',
        config_folder='config',
        output_f=None,
        tf_multiplex='TF',
        peak_multiplex='CRE',
        rna_multiplex='Target',
        update_config=True,
        save=False,
        return_df=True,
        njobs=12)
output.to_csv('/mnt/data/home/tycloud/workspace/algorithms_raw/data/BM_DEGs_6_preturb_CRE.txt',
                   sep='\t', index=False)

