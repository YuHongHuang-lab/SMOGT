import numpy as np
import pandas as pd

from hummuspy import *
import torch.optim as optim
from dataset.bio_dataset import *
from model.GraphAutoencoder import *
from train import *
from utils import *
from hummuspy import *

#K562
config_path = "config.yaml"
config = load_config(config_path)

EBuider = EvaluationBuider(config=config,
                           embedding_path='/mnt/data/home/tycloud/workspace/algorithms_raw/data/avg_df_42_K562_new.csv',
                           negative_ratio=1)
EBuider.load_embedding()
result = EBuider.analysis(edge_types=['TF-CRE', 'CRE-CRE'])

# buider = NetworkBuider(config=config,
#                        edge_types=[['TF','TF'], ['TF','CRE'], ['CRE', 'CRE'],
#                                    ['CRE', 'Target'], ['CRE', 'TF'], ['Target', 'Target']],
#                        threshold={'TF_TF':0.999,
#                                   'TF_CRE':0.5,
#                                   'CRE_CRE':0.9,
#                                   'CRE_Target':0.2,
#                                   'CRE_TF':0.2,
#                                   'Target_Target':0.999},
#                        method='dot',
#                        topp_CRE=0.05)
# buider.load_data(file_path='/mnt/data/home/tycloud/workspace/algorithms_raw/data/avg_df_42_K562_new.csv')
# buider.buide_networks()
# buider.save_networks(base_dir='/mnt/data/home/tycloud/workspace/algorithms_raw/data/K562/processed/hummus')
#
# multiplexes_dictionary = {
#     'TF': {'TF': '00'},
#     'Target': {'Target': '00'},
#     'CRE': {'CRE': '10'}
# }
#
# bipartites_dictionary = {
#     'TF_CRE.csv': {
#         'multiplex_right': 'TF',
#         'multiplex_left': 'CRE'
#     },
#     'CRE_Target.csv': {
#         'multiplex_right': 'CRE',
#         'multiplex_left': 'Target'
#     }
# }
#
#
# tf_file = '/mnt/data/home/tycloud/workspace/algorithms_raw/data/K562_com_TF_all.txt'
# with open(tf_file, 'r') as f:
#     tf_list = [line.strip() for line in f.readlines()]
#
# # our
# output = core_grn.get_output_from_dicts(
#         output_request='target_genes',
#         multilayer_f='/mnt/data/home/tycloud/workspace/algorithms_raw/data/K562/processed/hummus',
#         multiplexes_list=multiplexes_dictionary,
#         bipartites_list=bipartites_dictionary,
#         gene_list=None,
#         tf_list=tf_list,
#         config_filename='target_genes_config.yml',
#         config_folder='config',
#         output_f=None,
#         tf_multiplex='TF',
#         peak_multiplex='CRE',
#         rna_multiplex='Target',
#         update_config=True,
#         save=False,
#         return_df=True,
#         njobs=24)
#
# output.to_csv('/mnt/data/home/tycloud/workspace/algorithms_raw/data/K562_com_TF_all_preturb.txt',
#                    sep='\t', index=False)
#
# #motif cicero
# output_motif_cicero = core_grn.get_output_from_dicts(
#         output_request='target_genes',
#         multilayer_f='/mnt/data/home/tycloud/workspace/algorithms_raw/data/K562/processed/hummus_motif_cicero',
#         multiplexes_list=multiplexes_dictionary,
#         bipartites_list=bipartites_dictionary,
#         gene_list=None,
#         tf_list=tf_list,
#         config_filename='target_genes_config.yml',
#         config_folder='config',
#         output_f=None,
#         tf_multiplex='TF',
#         peak_multiplex='CRE',
#         rna_multiplex='Target',
#         update_config=True,
#         save=False,
#         return_df=True,
#         njobs=24)
#
# output_motif_cicero.to_csv('/mnt/data/home/tycloud/workspace/algorithms_raw/data/K562_com_TF_all_preturb_motif_cicero.txt',
#                    sep='\t', index=False)
#
#
# #BM
config_path = "config.yaml"
config = load_config(config_path)

# EBuider = EvaluationBuider(config=config,
#                            embedding_path='/mnt/data/home/tycloud/workspace/algorithms_raw/data/avg_df_42_BM.csv',
#                            negative_ratio=1)
# EBuider.load_embedding()
# result = EBuider.analysis(edge_types=['TF-CRE', 'CRE-CRE'])
#
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
                       topp_CRE=0.02)
buider.load_data(file_path='/mnt/data/home/tycloud/workspace/algorithms_raw/data/avg_df_42_BM.csv')
buider.buide_networks()
buider.save_networks(base_dir='/mnt/data/home/tycloud/workspace/algorithms_raw/data/HSPC/processed/hummus_BM')

multiplexes_dictionary = {
    'TF': {'TF': '00'},
    'Target': {'Target': '00'},
    'CRE': {'CRE': '01'}
}

bipartites_dictionary = {
    'TF_CRE.csv': {
        'multiplex_right': 'TF',
        'multiplex_left': 'CRE'
    },
    'CRE_Target.csv': {
        'multiplex_right': 'CRE',
        'multiplex_left': 'Target'
    }
}

#DEGs_6
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

#DEGs_1
tf_file = '/mnt/data/home/tycloud/workspace/algorithms_raw/data/BM_DEGs_1.txt'
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
      njobs=12)

output.to_csv('/mnt/data/home/tycloud/workspace/algorithms_raw/data/BM_DEGs_1_preturb.txt',
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
output.to_csv('/mnt/data/home/tycloud/workspace/algorithms_raw/data/BM_DEGs_1_preturb_CRE.txt',
              sep='\t', index=False)


#DEGs_0
tf_file = '/mnt/data/home/tycloud/workspace/algorithms_raw/data/BM_DEGs_0.txt'
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
  njobs=12)

output.to_csv('/mnt/data/home/tycloud/workspace/algorithms_raw/data/BM_DEGs_0_preturb.txt',
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
output.to_csv('/mnt/data/home/tycloud/workspace/algorithms_raw/data/BM_DEGs_0_preturb_CRE.txt',
              sep='\t', index=False)


#DEGs_MPO
tf_file = '/mnt/data/home/tycloud/workspace/algorithms_raw/data/BM_DEGs_MPO.txt'
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

output.to_csv('/mnt/data/home/tycloud/workspace/algorithms_raw/data/BM_DEGs_MPO_preturb.txt',
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
output.to_csv('/mnt/data/home/tycloud/workspace/algorithms_raw/data/BM_DEGs_MPO_preturb_CRE.txt',
              sep='\t', index=False)


#DEGs_CLP
tf_file = '/mnt/data/home/tycloud/workspace/algorithms_raw/data/BM_DEGs_CLP.txt'
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

output.to_csv('/mnt/data/home/tycloud/workspace/algorithms_raw/data/BM_DEGs_CLP_preturb.txt',
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
output.to_csv('/mnt/data/home/tycloud/workspace/algorithms_raw/data/BM_DEGs_CLP_preturb_CRE.txt',
              sep='\t', index=False)


#BM predicted
config_path = "config.yaml"
config = load_config(config_path)

EBuider = EvaluationBuider(config=config,
                           embedding_path='/mnt/data/home/tycloud/workspace/algorithms_raw/data/avg_df_42_BM_predicted.csv',
                           negative_ratio=1)
EBuider.load_embedding()
result = EBuider.analysis(edge_types=['TF-CRE', 'CRE-CRE'])

#buide network
buider = NetworkBuider(config=config,
                       edge_types=[['TF','TF'], ['TF','CRE'], ['CRE', 'CRE'],
                                   ['CRE', 'Target'], ['CRE', 'TF'], ['Target', 'Target']],
                       threshold={'TF_TF':0.999,
                                  'TF_CRE':0.9,
                                  'CRE_CRE':0.9,
                                  'CRE_Target':0.2,
                                  'CRE_TF':0.2,
                                  'Target_Target':0.999},
                       method='dot',
                       topp_CRE=0.01)
buider.load_data(file_path='/mnt/data/home/tycloud/workspace/algorithms_raw/data/avg_df_42_BM_predicted.csv')
buider.buide_networks()
buider.save_networks(base_dir='/mnt/data/home/tycloud/workspace/algorithms_raw/data/HSPC_predict/processed/hummus_BM')



#PBMC predicted
config_path = "config.yaml"
config = load_config(config_path)

EBuider = EvaluationBuider(config=config,
                           embedding_path='/mnt/data/home/tycloud/workspace/algorithms_raw/data/avg_df_42_PBMC_predicted.csv',
                           negative_ratio=1)
EBuider.load_embedding()
result = EBuider.analysis(edge_types=['TF-CRE', 'CRE-CRE'])

#buide network
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
buider.load_data(file_path='/mnt/data/home/tycloud/workspace/algorithms_raw/data/avg_df_42_PBMC_predicted.csv')
buider.buide_networks()
buider.save_networks(base_dir='/mnt/data/home/tycloud/workspace/algorithms_raw/data/PBMC_predict/processed/hummus_PBMC')

# TF_CRE,CRE_CRE 参数敏感性 for predict
# TF_CRE:0.8, CRE_CRE:0.02
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
                       topp_CRE=0.02)
buider.load_data(file_path='/mnt/data/home/tycloud/workspace/algorithms_raw/data/avg_df_42_PBMC_predicted.csv')
buider.buide_networks()
buider.save_networks(base_dir='/mnt/data/home/tycloud/workspace/algorithms_raw/data/PBMC_predict/processed/hummus_PBMC_TF_CRE_0.8_CRE_CRE_0.02')

# TF_CRE:0.8, CRE_CRE:0.03
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
                       topp_CRE=0.03)
buider.load_data(file_path='/mnt/data/home/tycloud/workspace/algorithms_raw/data/avg_df_42_PBMC_predicted.csv')
buider.buide_networks()
buider.save_networks(base_dir='/mnt/data/home/tycloud/workspace/algorithms_raw/data/PBMC_predict/processed/hummus_PBMC_TF_CRE_0.8_CRE_CRE_0.03')

# TF_CRE:0.8, CRE_CRE:0.04
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
                       topp_CRE=0.04)
buider.load_data(file_path='/mnt/data/home/tycloud/workspace/algorithms_raw/data/avg_df_42_PBMC_predicted.csv')
buider.buide_networks()
buider.save_networks(base_dir='/mnt/data/home/tycloud/workspace/algorithms_raw/data/PBMC_predict/processed/hummus_PBMC_TF_CRE_0.8_CRE_CRE_0.04')

# TF_CRE:0.8, CRE_CRE:0.08
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
                       topp_CRE=0.08)
buider.load_data(file_path='/mnt/data/home/tycloud/workspace/algorithms_raw/data/avg_df_42_PBMC_predicted.csv')
buider.buide_networks()
buider.save_networks(base_dir='/mnt/data/home/tycloud/workspace/algorithms_raw/data/PBMC_predict/processed/hummus_PBMC_TF_CRE_0.8_CRE_CRE_0.08')

# TF_CRE:0.8, CRE_CRE:0.1
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
                       topp_CRE=0.1)
buider.load_data(file_path='/mnt/data/home/tycloud/workspace/algorithms_raw/data/avg_df_42_PBMC_predicted.csv')
buider.buide_networks()
buider.save_networks(base_dir='/mnt/data/home/tycloud/workspace/algorithms_raw/data/PBMC_predict/processed/hummus_PBMC_TF_CRE_0.8_CRE_CRE_0.1')

# TF_CRE:0.75, CRE_CRE:0.02
buider = NetworkBuider(config=config,
                       edge_types=[['TF','TF'], ['TF','CRE'], ['CRE', 'CRE'],
                                   ['CRE', 'Target'], ['CRE', 'TF'], ['Target', 'Target']],
                       threshold={'TF_TF':0.999,
                         'TF_CRE':0.75,
                         'CRE_CRE':0.9,
                         'CRE_Target':0.2,
                         'CRE_TF':0.2,
                         'Target_Target':0.999},
                       method='dot',
                       topp_CRE=0.02)
buider.load_data(file_path='/mnt/data/home/tycloud/workspace/algorithms_raw/data/avg_df_42_PBMC_predicted.csv')
buider.buide_networks()
buider.save_networks(base_dir='/mnt/data/home/tycloud/workspace/algorithms_raw/data/PBMC_predict/processed/hummus_PBMC_TF_CRE_0.75_CRE_CRE_0.02')

# TF_CRE:0.85, CRE_CRE:0.02
buider = NetworkBuider(config=config,
                       edge_types=[['TF','TF'], ['TF','CRE'], ['CRE', 'CRE'],
                                   ['CRE', 'Target'], ['CRE', 'TF'], ['Target', 'Target']],
                       threshold={'TF_TF':0.999,
                         'TF_CRE':0.85,
                         'CRE_CRE':0.85,
                         'CRE_Target':0.2,
                         'CRE_TF':0.2,
                         'Target_Target':0.999},
                       method='dot',
                       topp_CRE=0.02)
buider.load_data(file_path='/mnt/data/home/tycloud/workspace/algorithms_raw/data/avg_df_42_PBMC_predicted.csv')
buider.buide_networks()
buider.save_networks(base_dir='/mnt/data/home/tycloud/workspace/algorithms_raw/data/PBMC_predict/processed/hummus_PBMC_TF_CRE_0.85_CRE_CRE_0.02')

# TF_CRE:0.9, CRE_CRE:0.02
buider = NetworkBuider(config=config,
                       edge_types=[['TF','TF'], ['TF','CRE'], ['CRE', 'CRE'],
                                   ['CRE', 'Target'], ['CRE', 'TF'], ['Target', 'Target']],
                       threshold={'TF_TF':0.999,
                         'TF_CRE':0.9,
                         'CRE_CRE':0.85,
                         'CRE_Target':0.2,
                         'CRE_TF':0.2,
                         'Target_Target':0.999},
                       method='dot',
                       topp_CRE=0.02)
buider.load_data(file_path='/mnt/data/home/tycloud/workspace/algorithms_raw/data/avg_df_42_PBMC_predicted.csv')
buider.buide_networks()
buider.save_networks(base_dir='/mnt/data/home/tycloud/workspace/algorithms_raw/data/PBMC_predict/processed/hummus_PBMC_TF_CRE_0.9_CRE_CRE_0.02')


#A549 predicted
config_path = "config.yaml"
config = load_config(config_path)

EBuider = EvaluationBuider(config=config,
                           embedding_path='/mnt/data/home/tycloud/workspace/algorithms_raw/data/avg_df_42_A549_predicted.csv',
                           negative_ratio=1)
EBuider.load_embedding()
result = EBuider.analysis(edge_types=['TF-CRE', 'CRE-CRE'])

#buide network
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
buider.load_data(file_path='/mnt/data/home/tycloud/workspace/algorithms_raw/data/avg_df_42_A549_predicted.csv')
buider.buide_networks()
buider.save_networks(base_dir='/mnt/data/home/tycloud/workspace/algorithms_raw/data/A549_predict/processed/hummus_A549')


#A549
config_path = "config.yaml"
config = load_config(config_path)

EBuider = EvaluationBuider(config=config,
                           embedding_path='/mnt/data/home/tycloud/workspace/algorithms_raw/data/avg_df_42_A549.csv',
                           negative_ratio=1)
EBuider.load_embedding()
result = EBuider.analysis(edge_types=['TF-CRE', 'CRE-CRE'])

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
                       topp_CRE=0.03)
buider.load_data(file_path='/mnt/data/home/tycloud/workspace/algorithms_raw/data/avg_df_42_A549_2.csv')
buider.buide_networks()
buider.save_networks(base_dir='/mnt/data/home/tycloud/workspace/algorithms_raw/data/A549/processed/hummus_2')


#GM12878
config_path = "config.yaml"
config = load_config(config_path)

EBuider = EvaluationBuider(config=config,
                           embedding_path='/mnt/data/home/tycloud/workspace/algorithms_raw/data/avg_df_42_GM12878_2.csv',
                           negative_ratio=1)
EBuider.load_embedding()
result = EBuider.analysis(edge_types=['TF-CRE', 'CRE-CRE'])

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
                       topp_CRE=0.03)
buider.load_data(file_path='/mnt/data/home/tycloud/workspace/algorithms_raw/data/avg_df_42_GM12878_2.csv')
buider.buide_networks()
buider.save_networks(base_dir='/mnt/data/home/tycloud/workspace/algorithms_raw/data/GM12878/processed/hummus_2')


#HCT116
config_path = "config.yaml"
config = load_config(config_path)

EBuider = EvaluationBuider(config=config,
                           embedding_path='/mnt/data/home/tycloud/workspace/algorithms_raw/data/avg_df_42_HCT116_2.csv',
                           negative_ratio=1)
EBuider.load_embedding()
result = EBuider.analysis(edge_types=['TF-CRE', 'CRE-CRE'])

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
                       topp_CRE=0.03)
buider.load_data(file_path='/mnt/data/home/tycloud/workspace/algorithms_raw/data/avg_df_42_HCT116_2.csv')
buider.buide_networks()
buider.save_networks(base_dir='/mnt/data/home/tycloud/workspace/algorithms_raw/data/HCT116/processed/hummus_2')


