import os

import numpy as np

from model.BioStreamNet import *
import pandas as pd
import gc
import scanpy as sc

import multiprocessing
from functools import partial


# region predict other
# BM
# output_dir = '/mnt/data/home/tycloud/workspace/algorithms_raw_paper_4/data/HSPC_predict'
# common_seacells_mat_train = pd.read_csv('/mnt/data/home/tycloud/workspace/algorithms_raw_paper_4/data/HSPC_predict/raw_600/graph_0_nodes.csv',
#                   index_col=0)
# train_id = common_seacells_mat_train.columns
# del common_seacells_mat_train
#
#
# common_seacells_mat = pd.read_csv('/mnt/data/home/tycloud/workspace/algorithms_raw_paper_4/data/HSPC/raw_600/graph_0_nodes.csv',
#                                   index_col=0)
#
# #生成训练文件
# train_df = pd.DataFrame({'id' : common_seacells_mat.columns,
#                         'train' : [id in train_id for id in common_seacells_mat.columns]})
# train_df['train'] = train_df['train'].astype(int)
#
# Genes_Peaks_df = pd.read_csv('/mnt/data/home/tycloud/workspace/algorithms_raw_paper_4/data/HSPC_predict/raw_600/graph_0_node_names.csv',
#                              index_col=0)
# TF = Genes_Peaks_df.loc[Genes_Peaks_df['type']=='TF', 'name'].to_list()
# Target = Genes_Peaks_df.loc[Genes_Peaks_df['type']=='Target', 'name'].to_list()
# CRE = Genes_Peaks_df.loc[Genes_Peaks_df['type']=='CRE', 'name'].to_list()
#
# GRN_df = pd.read_csv('/mnt/data/home/tycloud/workspace/algorithms_raw_paper_4/data/HSPC_predict/raw_600/graph_0_edges.csv')
#
# #表达矩阵准备
# common_seacells_mat_TF = common_seacells_mat.loc[TF, ]
# common_seacells_mat_Target = common_seacells_mat.loc[Target, ]
# common_seacells_mat_CRE = common_seacells_mat.loc[CRE, ]
#
# #网络准备
# TF_TF_network = GRN_df.loc[GRN_df['edge_id_type']=='TF_TF', ['from', 'to']]
# TF_TF_network['score'] = 1
#
# CRE_Target_network = GRN_df.loc[GRN_df['edge_id_type']=='CRE_Target', ['from', 'to']]
# CRE_Target_network['score'] = 1
#
# CRE_TF_network = GRN_df.loc[GRN_df['edge_id_type']=='CRE_TF', ['from', 'to']]
# CRE_TF_network['score'] = 1
#
# TF_CRE_network = pd.read_csv('/mnt/data/home/tycloud/workspace/algorithms_raw_paper_4/data/HSPC_predict/processed/hummus_BM/bipartite/TF_CRE.csv',
#                      sep='\t', header=None, names=['from', 'to', 'score'])
#
# CRE_CRE_network = pd.read_csv('/mnt/data/home/tycloud/workspace/algorithms_raw_paper_4/data/HSPC_predict/processed/hummus_BM/multiplex/CRE/CRE.csv',
#                      sep='\t', header=None, names=['from', 'to', 'score'])
#
# global_network = GlobalNetworkManager(
#     TF_TF_network=TF_TF_network,
#     CRE_CRE_network=CRE_CRE_network,
#     TF_CRE_network=TF_CRE_network,
#     CRE_TF_network=CRE_TF_network,
#     CRE_Target_network=CRE_Target_network,
#     TF_names=TF,
#     CRE_names=CRE,
#     Target_names=Target
# )
#
# predictor = GeneExpressionPredictor(global_network)
#
# predictor.prepare_data(tf_expr=common_seacells_mat_TF,
#                        cre_expr=common_seacells_mat_CRE,
#                        target_expr=common_seacells_mat_Target,
#                        train_df=train_df)
#
# gene_names = pd.read_csv('/mnt/data/home/tycloud/workspace/algorithms_raw_paper_4/data/HSPC_predict/predicted_genes.txt',
#                          header=None, names=['gene'])
# gene_names = gene_names['gene'].tolist()
#
# results = []
# # 为每个基因训练模型
# for i, gene_name in enumerate(tqdm(gene_names, desc="Training genes")):
#     print(f"\n[{i + 1}/{len(gene_names)}] 正在训练基因: {gene_name}")
#
#     try:
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()
#
#         model, evaluation = predictor.train_and_evaluate(
#             gene_name=gene_name,
#             batch_size=256,
#             lr=0.001,
#             epochs=250,
#             save_dir="/mnt/data/home/tycloud/workspace/algorithms_raw_paper_4/data/HSPC_predict/models"
#         )
#
#         correlation = evaluation['correlation']
#         p_value = evaluation.get('p_value', np.nan)
#
#         # 释放模型内存
#         del model, evaluation
#         gc.collect()
#
#     except Exception as e:
#         print(f"训练基因 {gene_name} 时发生错误: {str(e)}")
#         correlation = 0.0
#         p_value = np.nan
#
#     # 保存结果
#     results.append({
#         'gene': gene_name,
#         'correlation': correlation,
#         'p_value': p_value
#     })
#
# # 创建最终结果DataFrame
# results_df = pd.DataFrame(results)
# results_df = results_df.dropna(subset=['correlation'])
# np.mean(results_df['correlation'])
#
# results_df.to_csv(os.path.join(output_dir, 'gene_prediction_our.csv'), index=False)

# # PBMC
# output_dir = '/mnt/data/home/tycloud/workspace/algorithms_raw_paper_4/data/PBMC_predict'
# common_seacells_mat_train = pd.read_csv('/mnt/data/home/tycloud/workspace/algorithms_raw_paper_4/data/PBMC_predict/raw_600/graph_0_nodes.csv',
#                   index_col=0)
# train_id = common_seacells_mat_train.columns
# del common_seacells_mat_train
#
#
# common_seacells_mat = pd.read_csv('/mnt/data/home/tycloud/workspace/algorithms_raw_paper_4/data/PBMC/raw_600/graph_0_nodes.csv',
#                                   index_col=0)
#
# #生成训练文件
# train_df = pd.DataFrame({'id' : common_seacells_mat.columns,
#                         'train' : [id in train_id for id in common_seacells_mat.columns]})
# train_df['train'] = train_df['train'].astype(int)
#
# Genes_Peaks_df = pd.read_csv('/mnt/data/home/tycloud/workspace/algorithms_raw_paper_4/data/PBMC_predict/raw_600/graph_0_node_names.csv',
#                              index_col=0)
# TF = Genes_Peaks_df.loc[Genes_Peaks_df['type']=='TF', 'name'].to_list()
# Target = Genes_Peaks_df.loc[Genes_Peaks_df['type']=='Target', 'name'].to_list()
# CRE = Genes_Peaks_df.loc[Genes_Peaks_df['type']=='CRE', 'name'].to_list()
#
# GRN_df = pd.read_csv('/mnt/data/home/tycloud/workspace/algorithms_raw_paper_4/data/PBMC_predict/raw_600/graph_0_edges.csv')
#
# #表达矩阵准备
# common_seacells_mat_TF = common_seacells_mat.loc[TF, ]
# common_seacells_mat_Target = common_seacells_mat.loc[Target, ]
# common_seacells_mat_CRE = common_seacells_mat.loc[CRE, ]
#
# #网络准备
# TF_TF_network = GRN_df.loc[GRN_df['edge_id_type']=='TF_TF', ['from', 'to']]
# TF_TF_network['score'] = 1
#
# CRE_Target_network = GRN_df.loc[GRN_df['edge_id_type']=='CRE_Target', ['from', 'to']]
# CRE_Target_network['score'] = 1
#
# CRE_TF_network = GRN_df.loc[GRN_df['edge_id_type']=='CRE_TF', ['from', 'to']]
# CRE_TF_network['score'] = 1

#TF_CRE:0.8;CRE_CRE:0.01
# TF_CRE_network = pd.read_csv('/mnt/data/home/tycloud/workspace/algorithms_raw_paper_4/data/PBMC_predict/processed/hummus_PBMC/bipartite/TF_CRE.csv',
#                      sep='\t', header=None, names=['from', 'to', 'score'])
#
# CRE_CRE_network = pd.read_csv('/mnt/data/home/tycloud/workspace/algorithms_raw_paper_4/data/PBMC_predict/processed/hummus_PBMC/multiplex/CRE/CRE.csv',
#                      sep='\t', header=None, names=['from', 'to', 'score'])
#
# global_network = GlobalNetworkManager(
#     TF_TF_network=TF_TF_network,
#     CRE_CRE_network=CRE_CRE_network,
#     TF_CRE_network=TF_CRE_network,
#     CRE_TF_network=CRE_TF_network,
#     CRE_Target_network=CRE_Target_network,
#     TF_names=TF,
#     CRE_names=CRE,
#     Target_names=Target
# )
#
# predictor = GeneExpressionPredictor(global_network)
#
# predictor.prepare_data(tf_expr=common_seacells_mat_TF,
#                        cre_expr=common_seacells_mat_CRE,
#                        target_expr=common_seacells_mat_Target,
#                        train_df=train_df)
#
# gene_names = pd.read_csv('/mnt/data/home/tycloud/workspace/algorithms_raw_paper_4/data/PBMC_predict/predicted_genes.txt',
#                          header=None, names=['gene'])
# gene_names = gene_names['gene'].tolist()
#
# # 为每个基因训练模型
# results = []
# for i, gene_name in enumerate(tqdm(gene_names, desc="Training genes")):
#     print(f"\n[{i + 1}/{len(gene_names)}] 正在训练基因: {gene_name}")
#     try:
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()
#
#         model, evaluation = predictor.train_and_evaluate(
#             gene_name=gene_name,
#             batch_size=256,
#             lr=0.001,
#             epochs=250,
#             save_dir="/mnt/data/home/tycloud/workspace/algorithms_raw_paper_4/data/PBMC_predict/models"
#         )
#
#         correlation = evaluation['correlation']
#         p_value = evaluation.get('p_value', np.nan)
#
#         # 释放模型内存
#         del model, evaluation
#         gc.collect()
#
#     except Exception as e:
#         print(f"训练基因 {gene_name} 时发生错误: {str(e)}")
#         correlation = 0.0
#         p_value = np.nan
#
#     # 保存结果
#     results.append({
#         'gene': gene_name,
#         'correlation': correlation,
#         'p_value': p_value
#     })
#
# # 创建最终结果DataFrame
# results_df = pd.DataFrame(results)
# results_df = results_df.dropna(subset=['correlation'])
# np.mean(results_df['correlation'])
#
# results_df.to_csv(os.path.join(output_dir, 'gene_prediction_our.csv'), index=False)


# PBMC TF_CRE,CRE_CRE 参数敏感性 并行计算
# TF_CRE:0.8, CRE_CRE:0.1

# num_processes = 4  # 使用更少的进程
#
# TF_CRE_network = pd.read_csv('/mnt/data/home/tycloud/workspace/algorithms_raw_paper_4/data/PBMC_predict/processed/hummus_PBMC_TF_CRE_0.8_CRE_CRE_0.1/bipartite/TF_CRE.csv',
#                      sep='\t', header=None, names=['from', 'to', 'score'])
#
# CRE_CRE_network = pd.read_csv('/mnt/data/home/tycloud/workspace/algorithms_raw_paper_4/data/PBMC_predict/processed/hummus_PBMC_TF_CRE_0.8_CRE_CRE_0.1/multiplex/CRE/CRE.csv',
#                      sep='\t', header=None, names=['from', 'to', 'score'])
#
# gene_names = pd.read_csv('/mnt/data/home/tycloud/workspace/algorithms_raw_paper_4/data/PBMC_predict/predicted_genes_compare.txt',
#                          header=None, names=['gene'])
# gene_names = gene_names['gene'].tolist()
#
# # gene_names = gene_names[:24]
#
# # 串行
# global_network = GlobalNetworkManager(
#     TF_TF_network=TF_TF_network,
#     CRE_CRE_network=CRE_CRE_network,
#     TF_CRE_network=TF_CRE_network,
#     CRE_TF_network=CRE_TF_network,
#     CRE_Target_network=CRE_Target_network,
#     TF_names=TF,
#     CRE_names=CRE,
#     Target_names=Target
# )
#
# predictor = GeneExpressionPredictor(global_network)
#
# predictor.prepare_data(tf_expr=common_seacells_mat_TF,
#                        cre_expr=common_seacells_mat_CRE,
#                        target_expr=common_seacells_mat_Target,
#                        train_df=train_df)
#
# #
# #
# #
# # 为每个基因训练模型
# results = []
# for i, gene_name in enumerate(tqdm(gene_names, desc="Training genes")):
#     print(f"\n[{i + 1}/{len(gene_names)}] 正在训练基因: {gene_name}")
#     try:
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()
#
#         model, evaluation = predictor.train_and_evaluate(
#             gene_name=gene_name,
#             batch_size=256,
#             lr=0.001,
#             epochs=250,
#             save_dir=None
#         )
#
#         correlation = evaluation['correlation']
#         p_value = evaluation.get('p_value', np.nan)
#
#         # 释放模型内存
#         del model, evaluation
#         gc.collect()
#
#     except Exception as e:
#         print(f"训练基因 {gene_name} 时发生错误: {str(e)}")
#         correlation = 0.0
#         p_value = np.nan
#
#     # 保存结果
#     results.append({
#         'gene': gene_name,
#         'correlation': correlation,
#         'p_value': p_value
#     })
#
# # 创建最终结果DataFrame
# results_df = pd.DataFrame(results)
# results_df = results_df.dropna(subset=['correlation'])
# np.mean(results_df['correlation'])
#
# results_df.to_csv(os.path.join(output_dir, 'gene_prediction_our_TF_CRE_0.8_CRE_CRE_0.1.csv'), index=False)



#A549
# output_dir = '/mnt/data/home/tycloud/workspace/algorithms_raw_paper_4/data/A549_predict'
# common_seacells_mat_train = pd.read_csv('/mnt/data/home/tycloud/workspace/algorithms_raw_paper_4/data/A549_predict/raw_500/graph_0_nodes.csv',
#                   index_col=0)
# train_id = common_seacells_mat_train.columns
# del common_seacells_mat_train
#
#
# common_seacells_mat = pd.read_csv('/mnt/data/home/tycloud/workspace/algorithms_raw_paper_4/data/A549/raw_500/graph_0_nodes.csv',
#                                   index_col=0)
#
# #生成训练文件
# train_df = pd.DataFrame({'id' : common_seacells_mat.columns,
#                         'train' : [id in train_id for id in common_seacells_mat.columns]})
# train_df['train'] = train_df['train'].astype(int)
#
# Genes_Peaks_df = pd.read_csv('/mnt/data/home/tycloud/workspace/algorithms_raw_paper_4/data/A549_predict/raw_500/graph_0_node_names.csv',
#                              index_col=0)
# TF = Genes_Peaks_df.loc[Genes_Peaks_df['type']=='TF', 'name'].to_list()
# Target = Genes_Peaks_df.loc[Genes_Peaks_df['type']=='Target', 'name'].to_list()
# CRE = Genes_Peaks_df.loc[Genes_Peaks_df['type']=='CRE', 'name'].to_list()
#
# GRN_df = pd.read_csv('/mnt/data/home/tycloud/workspace/algorithms_raw_paper_4/data/A549_predict/raw_500/graph_0_edges.csv')
#
# #表达矩阵准备
# common_seacells_mat_TF = common_seacells_mat.loc[TF, ]
# common_seacells_mat_Target = common_seacells_mat.loc[Target, ]
# common_seacells_mat_CRE = common_seacells_mat.loc[CRE, ]
#
# #网络准备
# TF_TF_network = GRN_df.loc[GRN_df['edge_id_type']=='TF_TF', ['from', 'to']]
# TF_TF_network['score'] = 1
#
# CRE_Target_network = GRN_df.loc[GRN_df['edge_id_type']=='CRE_Target', ['from', 'to']]
# CRE_Target_network['score'] = 1
#
# CRE_TF_network = GRN_df.loc[GRN_df['edge_id_type']=='CRE_TF', ['from', 'to']]
# CRE_TF_network['score'] = 1
#
# TF_CRE_network = pd.read_csv('/mnt/data/home/tycloud/workspace/algorithms_raw_paper_4/data/A549_predict/processed/hummus_A549/bipartite/TF_CRE.csv',
#                      sep='\t', header=None, names=['from', 'to', 'score'])
#
# CRE_CRE_network = pd.read_csv('/mnt/data/home/tycloud/workspace/algorithms_raw_paper_4/data/A549_predict/processed/hummus_A549/multiplex/CRE/CRE.csv',
#                      sep='\t', header=None, names=['from', 'to', 'score'])
#
# global_network = GlobalNetworkManager(
#     TF_TF_network=TF_TF_network,
#     CRE_CRE_network=CRE_CRE_network,
#     TF_CRE_network=TF_CRE_network,
#     CRE_TF_network=CRE_TF_network,
#     CRE_Target_network=CRE_Target_network,
#     TF_names=TF,
#     CRE_names=CRE,
#     Target_names=Target
# )
#
# predictor = GeneExpressionPredictor(global_network)
#
# predictor.prepare_data(tf_expr=common_seacells_mat_TF,
#                        cre_expr=common_seacells_mat_CRE,
#                        target_expr=common_seacells_mat_Target,
#                        train_df=train_df)
#
# gene_names = pd.read_csv('/mnt/data/home/tycloud/workspace/algorithms_raw_paper_4/data/A549_predict/predicted_genes.txt',
#                          header=None, names=['gene'])
# gene_names = gene_names['gene'].tolist()
#
# # 为每个基因训练模型
# results = []
# for i, gene_name in enumerate(tqdm(gene_names, desc="Training genes")):
#     print(f"\n[{i + 1}/{len(gene_names)}] 正在训练基因: {gene_name}")
#
#     try:
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()
#
#         model, evaluation = predictor.train_and_evaluate(
#             gene_name=gene_name,
#             batch_size=256,
#             lr=0.001,
#             epochs=250,
#             save_dir="/mnt/data/home/tycloud/workspace/algorithms_raw_paper_4/data/A549_predict/models"
#         )
#
#         correlation = evaluation['correlation']
#         p_value = evaluation.get('p_value', np.nan)
#
#         # 释放模型内存
#         del model, evaluation
#         gc.collect()
#
#     except Exception as e:
#         print(f"训练基因 {gene_name} 时发生错误: {str(e)}")
#         correlation = 0.0
#         p_value = np.nan
#
#     # 保存结果
#     results.append({
#         'gene': gene_name,
#         'correlation': correlation,
#         'p_value': p_value
#     })
#
# # 创建最终结果DataFrame
# results_df = pd.DataFrame(results)
# results_df = results_df.dropna(subset=['correlation'])
# np.mean(results_df['correlation'])
#
# results_df.to_csv(os.path.join(output_dir, 'gene_prediction_our.csv'), index=False)

# Melanoma
# endregion other


# region HPSC preturb
# HPSC preturb
output_dir = '/mnt/data/home/tycloud/workspace/algorithms_raw_paper_4/data/HSPC_preturb'
models_and_networks_dir = os.path.join(output_dir, "models_and_networks_p")
os.makedirs(models_and_networks_dir, exist_ok=True)
all_nodes_path = '/mnt/data/home/tycloud/workspace/algorithms_raw_paper_4/data/HSPC_preturb/raw_600/graph_0_nodes.csv'
train_nodes_path = '/mnt/data/home/tycloud/workspace/algorithms_raw_paper_4/data/HSPC_preturb/train_id.csv'
node_names_path = '/mnt/data/home/tycloud/workspace/algorithms_raw_paper_4/data/HSPC_preturb/raw_600/graph_0_node_names.csv'
edges_path = '/mnt/data/home/tycloud/workspace/algorithms_raw_paper_4/data/HSPC_preturb/GRN_df_p.csv'
gene_names_path = '/mnt/data/home/tycloud/workspace/algorithms_raw_paper_4/data/HSPC_preturb/predicted_genes.txt'

common_seacells_mat_train = pd.read_csv(
    train_nodes_path,
    index_col=0)
train_id = common_seacells_mat_train['x'].tolist()
del common_seacells_mat_train

common_seacells_mat = pd.read_csv(all_nodes_path,
                                  index_col=0)

#生成训练文件
train_df = pd.DataFrame({'id' : common_seacells_mat.columns,
                        'train' : [id in train_id for id in common_seacells_mat.columns]})
train_df['train'] = train_df['train'].astype(int)

Genes_Peaks_df = pd.read_csv(node_names_path,
                             index_col=0)
TF = Genes_Peaks_df.loc[Genes_Peaks_df['type']=='TF', 'name'].to_list()
Target = Genes_Peaks_df.loc[Genes_Peaks_df['type']=='Target', 'name'].to_list()
CRE = Genes_Peaks_df.loc[Genes_Peaks_df['type']=='CRE', 'name'].to_list()

GRN_df = pd.read_csv(edges_path)

#表达矩阵准备
common_seacells_mat_TF = common_seacells_mat.loc[TF, ]
common_seacells_mat_Target = common_seacells_mat.loc[Target, ]
common_seacells_mat_CRE = common_seacells_mat.loc[CRE, ]

# #网络准备
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

predictor = GeneExpressionPredictor(global_network)

predictor.prepare_data(tf_expr=common_seacells_mat_TF,
                       cre_expr=common_seacells_mat_CRE,
                       target_expr=common_seacells_mat_Target,
                       train_df=train_df)

gene_names = pd.read_csv(gene_names_path,
                         header=None, names=['gene'])
gene_names = gene_names['gene'].tolist()


gene_prediction = pd.read_csv(os.path.join(output_dir, 'gene_prediction_our_parallel.csv'))
gene_list = gene_prediction.loc[gene_prediction['correlation']>0, 'gene']
gene_list = gene_list.tolist()

adata_RNA_BM_seacells_600_ad = sc.read_h5ad('/mnt/data/home/tycloud/workspace/algorithms_raw_paper_4/data/seob_RNA_BM_seacells_600_ad_reduce.h5ad')

gene_predict_df = simulate_perturbation(perturbations={'NFIA':2},
                                        predictor=predictor,
                                        original_tf_expr_df=common_seacells_mat_TF,
                                        original_cre_expr_df=common_seacells_mat_CRE,
                                        models_dir=models_and_networks_dir,
                                        gene_list=gene_list,
                                        ncores=1)

# 进行多次迭代，模拟网络远距离调控
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


# region Melanoma preturb
# Melanoma preturb
output_dir = '/mnt/data/home/tycloud/workspace/algorithms_raw_paper_4/data/Melanoma'
models_and_networks_dir = os.path.join(output_dir, "models_and_networks_p")
os.makedirs(models_and_networks_dir, exist_ok=True)
all_nodes_path = '/mnt/data/home/tycloud/workspace/algorithms_raw_paper_4/data/Melanoma/raw_500/graph_0_nodes.csv'
train_nodes_path = '/mnt/data/home/tycloud/workspace/algorithms_raw_paper_4/data/Melanoma/train_id.csv'
node_names_path = '/mnt/data/home/tycloud/workspace/algorithms_raw_paper_4/data/Melanoma/raw_500/graph_0_node_names.csv'
edges_path = '/mnt/data/home/tycloud/workspace/algorithms_raw_paper_4/data/Melanoma/GRN_df_p.csv'
gene_names_path = '/mnt/data/home/tycloud/workspace/algorithms_raw_paper_4/data/Melanoma/predicted_genes.txt'

common_seacells_mat_train = pd.read_csv(
    train_nodes_path,
    index_col=0)
train_id = common_seacells_mat_train['x'].tolist()
del common_seacells_mat_train

common_seacells_mat = pd.read_csv(all_nodes_path,
                                  index_col=0)

#生成训练文件
train_df = pd.DataFrame({'id' : common_seacells_mat.columns,
                        'train' : [id in train_id for id in common_seacells_mat.columns]})
train_df['train'] = train_df['train'].astype(int)

Genes_Peaks_df = pd.read_csv(node_names_path,
                             index_col=0)
TF = Genes_Peaks_df.loc[Genes_Peaks_df['type']=='TF', 'name'].to_list()
Target = Genes_Peaks_df.loc[Genes_Peaks_df['type']=='Target', 'name'].to_list()
CRE = Genes_Peaks_df.loc[Genes_Peaks_df['type']=='CRE', 'name'].to_list()

GRN_df = pd.read_csv(edges_path)

#表达矩阵准备
common_seacells_mat_TF = common_seacells_mat.loc[TF, ]
common_seacells_mat_Target = common_seacells_mat.loc[Target, ]
common_seacells_mat_CRE = common_seacells_mat.loc[CRE, ]

# #网络准备
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

predictor = GeneExpressionPredictor(global_network)

predictor.prepare_data(tf_expr=common_seacells_mat_TF,
                       cre_expr=common_seacells_mat_CRE,
                       target_expr=common_seacells_mat_Target,
                       train_df=train_df)

gene_names = pd.read_csv(gene_names_path,
                         header=None, names=['gene'])
gene_names = gene_names['gene'].tolist()


gene_prediction = pd.read_csv(os.path.join(output_dir, 'gene_prediction_our_parallel.csv'))
gene_list = gene_prediction.loc[gene_prediction['correlation']>0, 'gene']
gene_list = gene_list.tolist()

adata_RNA_MM_seacells_500_ad = sc.read_h5ad('/mnt/data/home/tycloud/workspace/algorithms_raw_paper_4/data/seob_MM_RNA_seacells_500_ad_pac.h5ad')
adata_RNA_MM_seacells_500_ad.obs_names = adata_RNA_MM_seacells_500_ad.obs_names.str.replace('-', '.')

gene_predict_df = simulate_perturbation(perturbations={'SOX10':2},
                                        predictor=predictor,
                                        original_tf_expr_df=common_seacells_mat_TF,
                                        original_cre_expr_df=common_seacells_mat_CRE,
                                        models_dir=models_and_networks_dir,
                                        gene_list=gene_list,
                                        ncores=1)

# region 进行多次迭代，模拟网络远距离调控
# 进行多次迭代，模拟网络远距离调控
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
# endregion

gene_predict_df.to_csv(os.path.join(output_dir,'SOX10_2.csv'))

# gene_predict_df = pd.read_csv(os.path.join(output_dir, 'gene_preturb_our_parallel_KLF1.csv'))


run_perturbation_analysis(
    perturbations={'KLF1':0},
    predictor=predictor,
    original_tf_expr_df=common_seacells_mat_TF,
    original_cre_expr_df=common_seacells_mat_CRE,
    original_target_expr_df=common_seacells_mat_Target,
    models_dir=models_and_networks_dir,
    adata=adata_RNA_MM_seacells_500_ad,
    output_dir=output_dir,
    ko_results_path=os.path.join(output_dir, 'SOX10_2.csv'),
    gene_list=gene_names,
    embedding_key='X_pca',
    metadata_key='seurat_clusters',
    n_cpu=1,
    tf_name='SOX10_2')
# endregion

