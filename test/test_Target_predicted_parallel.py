import os
# Must be set before importing numpy/torch/pandas to take effect when spawn subprocesses import modules
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['TORCH_NUM_THREADS'] = '1'
import numpy as np
import pandas as pd
import gc
import torch
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing
from functools import partial

from model.BioStreamNet import *

# Process-level cache (only used in process pool mode)
_PROC_STATE = {}

def _worker_init():
    # Subprocess initialization: Limit the number of underlying threads in each process to avoid crashes caused by overlapping with multiprocessing
    import os
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    os.environ['TORCH_NUM_THREADS'] = '1'
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass

def _worker_init_with_data(all_nodes_path,
                           train_nodes_path,
                           node_names_path,
                           edges_path,
                           models_and_networks_dir):
    _worker_init()
    # Load large matrices and networks into local cache within each process to avoid copying via pickle
    global _PROC_STATE
    # Read training and full node matrices
    common_seacells_mat_train = pd.read_csv(
        train_nodes_path,
        index_col=0)
    train_id = common_seacells_mat_train['x'].tolist()
    del common_seacells_mat_train


    common_seacells_mat = pd.read_csv(all_nodes_path, index_col=0)

    train_df = pd.DataFrame({'id': common_seacells_mat.columns,
                             'train': [id in train_id for id in common_seacells_mat.columns]})
    train_df['train'] = train_df['train'].astype(int)

    Genes_Peaks_df = pd.read_csv(node_names_path, index_col=0)
    TF = Genes_Peaks_df.loc[Genes_Peaks_df['type'] == 'TF', 'name'].to_list()
    Target = Genes_Peaks_df.loc[Genes_Peaks_df['type'] == 'Target', 'name'].to_list()
    CRE = Genes_Peaks_df.loc[Genes_Peaks_df['type'] == 'CRE', 'name'].to_list()

    GRN_df = pd.read_csv(edges_path)

    common_seacells_mat_TF = common_seacells_mat.loc[TF,]
    common_seacells_mat_Target = common_seacells_mat.loc[Target,]
    common_seacells_mat_CRE = common_seacells_mat.loc[CRE,]

    TF_TF_network = GRN_df.loc[GRN_df['edge_id_type'] == 'TF_TF', ['from', 'to']]
    TF_TF_network['score'] = 1

    CRE_Target_network = GRN_df.loc[GRN_df['edge_id_type'] == 'CRE_Target', ['from', 'to']]
    CRE_Target_network['score'] = 1

    CRE_TF_network = GRN_df.loc[GRN_df['edge_id_type'] == 'CRE_TF', ['from', 'to']]
    CRE_TF_network['score'] = 1

    TF_CRE_network = GRN_df.loc[GRN_df['edge_id_type'] == 'TF_CRE', ['from', 'to']]
    TF_CRE_network['score'] = 1

    CRE_CRE_network = GRN_df.loc[GRN_df['edge_id_type'] == 'CRE_CRE', ['from', 'to']]
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

    _PROC_STATE = {
        'global_network': global_network,
        'common_seacells_mat_TF': common_seacells_mat_TF,
        'common_seacells_mat_Target': common_seacells_mat_Target,
        'common_seacells_mat_CRE': common_seacells_mat_CRE,
        'train_df': train_df,
        'models_and_networks_dir': models_and_networks_dir,
    }

def _train_single_gene_proc(gene_name):
    # Train using in-process cache
    try:
        global _PROC_STATE
        global_network = _PROC_STATE['global_network']
        common_seacells_mat_TF = _PROC_STATE['common_seacells_mat_TF']
        common_seacells_mat_CRE = _PROC_STATE['common_seacells_mat_CRE']
        common_seacells_mat_Target = _PROC_STATE['common_seacells_mat_Target']
        train_df = _PROC_STATE['train_df']
        models_and_networks_dir = _PROC_STATE['models_and_networks_dir']

        gene_type = 'Target' if gene_name in global_network.Target_names else 'TF'
        _, sub_network_dfs = global_network.get_gene_specific_network(gene_name, gene_type=gene_type)

        if sub_network_dfs:
            network_df_path = os.path.join(models_and_networks_dir, f"{gene_name}_network.csv")
            combined_df = pd.concat(sub_network_dfs.values(), ignore_index=True)
            combined_df.to_csv(network_df_path, index=False)
        else:
            print(f"Cannot generate network for {gene_name}. Skipping training.")
            return {'gene': gene_name, 'correlation': 0.0, 'p_value': np.nan}

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        local_predictor = GeneExpressionPredictor(global_network)
        local_predictor.prepare_data(tf_expr=common_seacells_mat_TF,
                                     cre_expr=common_seacells_mat_CRE,
                                     target_expr=common_seacells_mat_Target,
                                     train_df=train_df)

        model, evaluation = local_predictor.train_and_evaluate(
            gene_name=gene_name,
            batch_size=256,
            lr=0.001,
            epochs=250,
            save_dir=models_and_networks_dir
        )

        correlation = evaluation['correlation']
        p_value = evaluation.get('p_value', np.nan)

        del model, evaluation
        gc.collect()

        return {
            'gene': gene_name,
            'correlation': correlation,
            'p_value': p_value
        }
    except Exception as e:
        print(f"Error training gene {gene_name}: {str(e)}")
        return {
            'gene': gene_name,
            'correlation': 0.0,
            'p_value': np.nan
        }

def train_single_gene(args):
    """
    Train model for a single gene (for parallel processing)
    """
    (gene_name, global_network, models_and_networks_dir,
     common_seacells_mat_TF, common_seacells_mat_CRE, common_seacells_mat_Target, train_df) = args

    try:
        # Save network file first
        gene_type = 'Target' if gene_name in global_network.Target_names else 'TF'
        _, sub_network_dfs = global_network.get_gene_specific_network(gene_name, gene_type=gene_type)

        if sub_network_dfs:
            network_df_path = os.path.join(models_and_networks_dir, f"{gene_name}_network.csv")
            combined_df = pd.concat(sub_network_dfs.values(), ignore_index=True)
            combined_df.to_csv(network_df_path, index=False)
        else:
            print(f"Cannot generate network for {gene_name}. Skipping training.")
            return {'gene': gene_name, 'correlation': 0.0, 'p_value': np.nan}

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Create an independent predictor for each task to avoid shared state between threads
        local_predictor = GeneExpressionPredictor(global_network)
        local_predictor.prepare_data(tf_expr=common_seacells_mat_TF,
                                     cre_expr=common_seacells_mat_CRE,
                                     target_expr=common_seacells_mat_Target,
                                     train_df=train_df)

        model, evaluation = local_predictor.train_and_evaluate(
            gene_name=gene_name,
            batch_size=256,
            lr=0.001,
            epochs=250,
            save_dir=models_and_networks_dir
        )

        correlation = evaluation['correlation']
        p_value = evaluation.get('p_value', np.nan)

        del model, evaluation
        gc.collect()

        return {
            'gene': gene_name,
            'correlation': correlation,
            'p_value': p_value
        }

    except Exception as e:
        print(f"Error training gene {gene_name}: {str(e)}")
        return {
            'gene': gene_name,
            'correlation': 0.0,
            'p_value': np.nan
        }


def train_genes_parallel(gene_names, global_network, models_and_networks_dir,
                         common_seacells_mat_TF, common_seacells_mat_CRE,
                         common_seacells_mat_Target, train_df, n_cores=4, use_threads=True,
                         data_paths=None):
    """
    Train models for multiple genes in parallel
    """
    args_list = []
    for gene_name in gene_names:
        args = (gene_name, global_network, models_and_networks_dir,
                common_seacells_mat_TF, common_seacells_mat_CRE,
                common_seacells_mat_Target, train_df)
        args_list.append(args)

    results = []

    if use_threads:
        with ThreadPoolExecutor(max_workers=n_cores) as executor:
            future_to_gene = {
                executor.submit(train_single_gene, args): args[0]
                for args in args_list
            }

            for future in tqdm(future_to_gene, desc="Training genes in parallel"):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    gene_name = future_to_gene[future]
                    print(f"Error processing gene {gene_name}: {e}")
                    results.append({
                        'gene': gene_name,
                        'correlation': 0.0,
                        'p_value': np.nan
                    })
    else:
        # Process pool mode: Load data within processes to avoid copying large objects across processes
        if not data_paths:
            raise ValueError('Process mode requires data_paths with keys: all_nodes_path, train_nodes_path, node_names_path, edges_path')
        ctx = multiprocessing.get_context('spawn')
        with ProcessPoolExecutor(
                max_workers=n_cores,
                mp_context=ctx,
                initializer=_worker_init_with_data,
                initargs=(
                    data_paths['all_nodes_path'],
                    data_paths['train_nodes_path'],
                    data_paths['node_names_path'],
                    data_paths['edges_path'],
                    models_and_networks_dir,
                )
        ) as executor:
            future_to_gene = {
                executor.submit(_train_single_gene_proc, gene_name): gene_name
                for gene_name in gene_names
            }

            for future in tqdm(future_to_gene, desc="Training genes in parallel"):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    gene_name = future_to_gene[future]
                    print(f"Error processing gene {gene_name}: {e}")
                    results.append({
                        'gene': gene_name,
                        'correlation': 0.0,
                        'p_value': np.nan
                    })

    return results
