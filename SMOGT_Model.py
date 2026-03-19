# SMOGT_Model includes the entire workflow from data preprocessing to obtaining low-dimensional representations of TF, TG, and CRE,
# serving as the foundation for downstream tasks.
# 
# 1. Data Preprocessing ------ data_preprocess.R
#
# Readers should perform customized data preprocessing according to data_preprocess.R.
# The data_preprocess.R script uses the hematopoietic stem cell (HSPC) dataset
# (GEO accession: GSE200046, original paper: https://www.nature.com/articles/s41587-023-01716-9)
# as an example to complete dimensionality reduction, clustering, and annotation within the Seurat,
# as well as metacell conversion, along with the extraction and construction of prior networks.
# Required example files and intermediate processing files have been uploaded to Zenodo.
#
# Finally, three files are output:
#
# graph_0_node_names.csv (node names, including TF, TG, and CRE)
#
# graph_0_nodes.csv (node types, expression profiles of TF and TG, chromatin accessibility of CRE)
#
# graph_0_edges.csv (prior regulatory network)
#
# Example files for the above three outputs have been uploaded to Zenodo (SMOGT_Model).


# 
# 2. Configure the config file
#
# Specific parameter settings are explained in detail in config.yaml.
# Readers should configure the file according to their needs.
#
# 3. Run generate_dataset.py
#
# Convert graph_0_node_names.csv, graph_0_nodes.csv, and graph_0_edges.csv
# into an InMemoryDataset object.
#
# bash
# python generate_dataset.py
#
# 4. Run generate_negative_data.py
#
# To reduce runtime, SMOGT performs negative sampling before training.
# Multiple negative sampling sets are generated,
# such as data_list_hetero_1.pkl, data_list_hetero_2.pkl, etc.,
# and stored on processed_dir to be used as input during training.
#
# Example files have been uploaded to Zenodo (data_list_hetero_1.pkl, data_list_hetero_2.pkl, data_list_hetero_3.pkl, data_list_hetero_4.pkl).
# 
# 5. Run main.py
#
# Obtain low-dimensional representations of TF, TG, and CRE for downstream analysis.
# Example files of the model's output low-dimensional representations have been uploaded to Zenodo (avg_df_42_HSPC_predicted)
# 
# 
# 
# 
