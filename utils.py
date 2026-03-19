import os.path

import joblib
import yaml
import logging
import multiprocessing as mp
from functools import partial
import glob

import numpy as np
from collections import Counter
import logging
from copy import deepcopy
from collections import defaultdict

import pandas as pd
from typing import Dict, Any, Tuple
from torch_geometric.utils import negative_sampling

import torch

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def split_edges(data, rate=0.2, seed=42, mask_prior=False, prior_mask_rate=0.2, use_last_col=True):
    torch.manual_seed(seed)

    # homogeneous
    if hasattr(data, 'num_node_types'):
        edge_index = data.edge_index
        edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None

        if edge_attr is not None:
            if use_last_col:
                # Original splitting method
                mask = edge_attr[:, -1] == 1
                true_indices = torch.where(mask)[0]

                if len(true_indices) == 0:
                    data.train_edge_index = edge_index
                    data.train_edge_attr = edge_attr
                    return data

                print(f"Homogeneous graph - true edges: {mask.sum().item()}, prior edges: {(~mask).sum().item()}")

                perm = torch.randperm(len(true_indices))
                split_point = int(len(true_indices) * (1 - rate))

                train_mask = torch.zeros_like(mask)
                test_mask = torch.zeros_like(mask)

                train_mask[true_indices[perm[:split_point]]] = True
                test_mask[true_indices[perm[split_point:]]] = True

            else:
                # New splitting method based on last two columns
                train_mask = edge_attr[:, -2] == 1
                test_mask = edge_attr[:, -1] == 1

                print(
                    f"Homogeneous graph - train edges: {train_mask.sum().item()}, test edges: {test_mask.sum().item()}")

            if train_mask.any():
                data.train_edge_index = edge_index[:, train_mask]
                data.train_edge_attr = edge_attr[train_mask]

            if test_mask.any():
                data.test_edge_index = edge_index[:, test_mask]
                data.test_edge_attr = edge_attr[test_mask]

            if mask_prior:
                if train_mask.any():
                    # Remove test edges from edge_index
                    test_edge_set = {tuple(edge.tolist()) for edge in data.test_edge_index.t()}
                    keep_mask = torch.tensor([tuple(edge.tolist()) not in test_edge_set
                                              for edge in edge_index.t()], dtype=torch.bool)

                    edge_index = edge_index[:, keep_mask]
                    edge_attr = edge_attr[keep_mask]
                    print(f"After removing test edges - remaining prior edges: {edge_index.size(1)}")

            # Random masking of remaining prior edges
            num_edges = edge_index.size(1)
            num_edges_to_keep = int(num_edges * (1 - prior_mask_rate))
            keep_indices = torch.randperm(num_edges)[:num_edges_to_keep]

            data.edge_index = edge_index[:, keep_indices]
            data.edge_attr = edge_attr[keep_indices]

            print(f"After random masking - masked {num_edges - num_edges_to_keep} prior edges, "
                  f"keeping {num_edges_to_keep} edges ({(1 - prior_mask_rate) * 100:.1f}%)")

        else:
            data.train_edge_index = edge_index

        return data

    else:
        for edge_type in data.edge_types:
            if not hasattr(data[edge_type], 'edge_index'):
                continue

            edge_index = data[edge_type].edge_index
            edge_attr = data[edge_type].edge_attr if hasattr(data[edge_type], 'edge_attr') else None

            if edge_attr is not None:
                if use_last_col:
                    # Original splitting method
                    mask = edge_attr[:, -1] == 1
                    true_indices = torch.where(mask)[0]

                    if len(true_indices) == 0:
                        data[edge_type].train_edge_index = edge_index
                        data[edge_type].train_edge_attr = edge_attr
                        continue

                    print(f"{edge_type}: true edges: {mask.sum().item()}, prior edges: {(~mask).sum().item()}")

                    perm = torch.randperm(len(true_indices))
                    split_point = int(len(true_indices) * (1 - rate))

                    train_mask = torch.zeros_like(mask)
                    test_mask = torch.zeros_like(mask)

                    train_mask[true_indices[perm[:split_point]]] = True
                    test_mask[true_indices[perm[split_point:]]] = True

                else:
                    # New splitting method based on last two columns
                    train_mask = edge_attr[:, -2] == 1
                    test_mask = edge_attr[:, -1] == 1

                    print(f"{edge_type}: train edges: {train_mask.sum().item()}, test edges: {test_mask.sum().item()}")

                if train_mask.any():
                    data[edge_type].train_edge_index = edge_index[:, train_mask]
                    data[edge_type].train_edge_attr = edge_attr[train_mask]

                if test_mask.any():
                    data[edge_type].test_edge_index = edge_index[:, test_mask]
                    data[edge_type].test_edge_attr = edge_attr[test_mask]

                if mask_prior:
                    if test_mask.any():
                        test_edge_set = {tuple(edge.tolist()) for edge in data[edge_type].test_edge_index.t()}
                        keep_mask = torch.tensor([tuple(edge.tolist()) not in test_edge_set
                                                  for edge in edge_index.t()], dtype=torch.bool)

                        edge_index = edge_index[:, keep_mask]
                        edge_attr = edge_attr[keep_mask]
                        print(f"{edge_type}: After removing test edges - remaining prior edges: {edge_index.size(1)}")

                # Random masking of remaining prior edges
                num_edges = edge_index.size(1)
                num_edges_to_keep = int(num_edges * (1 - prior_mask_rate))
                keep_indices = torch.randperm(num_edges)[:num_edges_to_keep]

                data[edge_type].edge_index = edge_index[:, keep_indices]
                data[edge_type].edge_attr = edge_attr[keep_indices]

                print(f"{edge_type}: After random masking - masked {num_edges - num_edges_to_keep} prior edges, "
                      f"keeping {num_edges_to_keep} edges ({(1 - prior_mask_rate) * 100:.1f}%)")

            else:
                data[edge_type].train_edge_index = edge_index

        return data


def split_nodes(data, rate=0.2, seed=42, k_fold=None):
    torch.manual_seed(seed)

    k_fold_splits = {}

    if k_fold is None:
        for node_type in data.node_types:
            # Only process nodes with cluster attribute
            if hasattr(data[node_type], 'cluster'):
                # Get indices where cluster == 1
                regression_indices = torch.where(data[node_type].cluster == 1)[0]

                if len(regression_indices) > 0:

                    k_fold_splits[node_type] = []

                    # Shuffle indices
                    perm = torch.randperm(len(regression_indices))
                    split_idx = int(len(regression_indices) * (1 - rate))

                    # Create masks
                    train_mask = torch.zeros(data[node_type].x.size(0), dtype=torch.bool)
                    test_mask = torch.zeros(data[node_type].x.size(0), dtype=torch.bool)

                    train_mask[regression_indices[perm[:split_idx]]] = True
                    test_mask[regression_indices[perm[split_idx:]]] = True

                    # Add masks to data
                    data[node_type].node_train_mask = train_mask
                    data[node_type].node_test_mask = test_mask
                    k_fold_splits[node_type].append({
                        'node_train_mask': train_mask,
                        'node_test_mask': test_mask
                    })
    else:
        for node_type in data.node_types:
            if hasattr(data[node_type], 'cluster'):
                regression_indices = torch.where(data[node_type].cluster == 1)[0]

                if len(regression_indices) > 0:
                    # Shuffle indices
                    perm = torch.randperm(len(regression_indices))
                    shuffled_indices = regression_indices[perm]

                    # Calculate fold size
                    fold_size = len(shuffled_indices) // k_fold

                    # Initialize dictionary for this node type
                    k_fold_splits[node_type] = []

                    # Create k folds
                    for fold in range(k_fold):
                        start_idx = fold * fold_size
                        end_idx = start_idx + fold_size if fold < k_fold - 1 else len(shuffled_indices)

                        # Create masks for this fold
                        train_mask = torch.zeros(data[node_type].x.size(0), dtype=torch.bool)
                        test_mask = torch.zeros(data[node_type].x.size(0), dtype=torch.bool)

                        # Test indices for current fold
                        test_indices = shuffled_indices[start_idx:end_idx]
                        # Train indices are all other indices
                        train_indices = torch.cat([
                            shuffled_indices[:start_idx],
                            shuffled_indices[end_idx:]
                        ])

                        train_mask[train_indices] = True
                        test_mask[test_indices] = True

                        k_fold_splits[node_type].append({
                            'train_mask': train_mask,
                            'test_mask': test_mask
                        })

    return k_fold_splits


def sample_negative_edges(data, negative_ratio=1,name_id=None,node_label_dict=None):
    neg_edge_dict = {}
    pos_edge_dict = getattr(data, name_id)
    # device = next(iter(pos_edge_dict.values())).device

    for edge_type, pos_edge_index in pos_edge_dict.items():
        src_type, _, dst_type = edge_type if isinstance(edge_type, tuple) else eval(edge_type)

        if src_type=='CRE' and dst_type=='CRE':

            negative_ratio_CRE_l, negative_ratio_CRE_g = negative_ratio, negative_ratio*0.2

            # 对每个染色体单独处理
            cre_labels = node_label_dict[src_type]

            chromosomes = np.array([label.split('-')[0] for label in cre_labels.values()])
            unique_chromosomes = np.unique(chromosomes)

            all_neg_edges = []

            # 对每个染色体单独处理
            for chrom in unique_chromosomes:
                chrom_indices = np.where(chromosomes == chrom)[0]
                if len(chrom_indices) > 1:

                    #生成局部索引
                    idx_to_local = {idx: local_idx for local_idx, idx in enumerate(chrom_indices)}
                    local_to_idx = {local_idx: idx for idx, local_idx in idx_to_local.items()}

                    pos_edge_index_np = pos_edge_index.cpu().numpy()
                    src_matches = chromosomes[pos_edge_index_np[0]] == chrom
                    dst_matches = chromosomes[pos_edge_index_np[1]] == chrom
                    mask = src_matches & dst_matches

                    chrom_pos_edges = pos_edge_index[:, torch.from_numpy(mask)]

                    if chrom_pos_edges.size(1) > 0:
                        local_pos_edges = torch.tensor([
                            [idx_to_local[idx.item()] for idx in chrom_pos_edges[0]],
                            [idx_to_local[idx.item()] for idx in chrom_pos_edges[1]]
                        ])

                        num_neg_edges = int(chrom_pos_edges.size(1) * negative_ratio_CRE_l)

                        # 在染色体内进行负采样（使用局部索引）
                        local_neg_edges = negative_sampling(
                            edge_index=local_pos_edges,
                            num_nodes=(len(chrom_indices), len(chrom_indices)),
                            num_neg_samples=num_neg_edges
                        )

                        # 将局部索引转回全局索引
                        global_neg_edges = torch.tensor([
                            [local_to_idx[idx.item()] for idx in local_neg_edges[0]],
                            [local_to_idx[idx.item()] for idx in local_neg_edges[1]]
                        ])

                        all_neg_edges.append(global_neg_edges)

            if all_neg_edges:
                neg_edge_index = torch.cat(all_neg_edges, dim=1)

            #对所有染色体处理
            num_src_nodes = data.x_dict[src_type].size(0)
            num_dst_nodes = data.x_dict[dst_type].size(0)
            num_neg_edges = int(pos_edge_index.size(1) * negative_ratio_CRE_g)

            neg_edge_index_ = negative_sampling(
                edge_index=pos_edge_index,
                num_nodes=(num_src_nodes, num_dst_nodes),
                num_neg_samples=num_neg_edges
            )

            neg_edge_index = torch.cat([neg_edge_index, neg_edge_index_], dim=1)
        elif src_type=='TF' and dst_type=='CRE':
            unique_TFs = torch.unique(pos_edge_index[0]).cpu().numpy()

            idx_to_local = {idx:local_idx for local_idx, idx in enumerate(unique_TFs)}
            local_to_idx = {local_idx:idx for local_idx, idx in enumerate(unique_TFs)}

            local_pos_edges = torch.tensor([
                [idx_to_local[idx.item()] for idx in pos_edge_index[0]],
                pos_edge_index[1].tolist()
            ])

            num_neg_edges = int(pos_edge_index.size(1) * negative_ratio)

            local_neg_edges = negative_sampling(edge_index=local_pos_edges,
                                                num_nodes=(len(unique_TFs), data.x_dict[dst_type].size(0)),
                                                num_neg_samples=num_neg_edges)

            neg_edge_index = torch.tensor([
                [local_to_idx[local_idx.item()] for local_idx in local_neg_edges[0]],
                local_neg_edges[1].tolist()
            ])

        else:
            num_src_nodes = data.x_dict[src_type].size(0)
            num_dst_nodes = data.x_dict[dst_type].size(0)
            num_neg_edges = int(pos_edge_index.size(1) * negative_ratio)

            neg_edge_index = negative_sampling(
                edge_index=pos_edge_index,
                num_nodes=(num_src_nodes, num_dst_nodes),
                num_neg_samples=num_neg_edges
            )

        neg_edge_np = neg_edge_index.t().cpu().numpy()

        # 收集所有存在的边到集合中
        # existing_edges = set()

        # 添加先验边
        # existing_edges.update(
        #     map(tuple, data.prior_edge_index_dict[edge_type].t().tolist())
        # )

        # 添加全局边
        # if hasattr(data, 'global_edge_index_dict') and edge_type in data.global_edge_index_dict:
        #     existing_edges.update(
        #         map(tuple, data.global_edge_index_dict[edge_type].t().tolist())
        #     )

        # 添加额外边池
        if hasattr(data, 'add_edge_pool_dict') and edge_type in data.add_edge_pool_dict:
            pool_edges = data.add_edge_pool_dict[edge_type].t().cpu().numpy()

            # 由claude生成的加速算法
            pool_dtype = np.dtype([('src', pool_edges.dtype), ('dst', pool_edges.dtype)])
            pool_struct = np.zeros(len(pool_edges), dtype=pool_dtype)
            pool_struct['src'] = pool_edges[:, 0]
            pool_struct['dst'] = pool_edges[:, 1]

            neg_dtype = np.dtype([('src', neg_edge_np.dtype), ('dst', neg_edge_np.dtype)])
            neg_struct = np.zeros(len(neg_edge_np), dtype=neg_dtype)
            neg_struct['src'] = neg_edge_np[:, 0]
            neg_struct['dst'] = neg_edge_np[:, 1]

            valid_mask = ~np.in1d(neg_struct, pool_struct)
            valid_edges = neg_edge_np[valid_mask]

            if len(valid_edges) > 0:
                neg_edge_dict[str(edge_type)] = torch.tensor(valid_edges).t()

    return neg_edge_dict


def generate_samples_range(data_base, negative_rate, node_label_dict,range_tuple):
    start, end = range_tuple
    samples = []
    pid = mp.current_process().pid

    for i in range(start, end):
        data = deepcopy(data_base)
        print(f"pid{pid} generate {i} sample")
        train_neg = sample_negative_edges(data, negative_rate, 'train_edge_index_dict', node_label_dict)
        test_neg = sample_negative_edges(data, negative_rate, 'test_edge_index_dict', node_label_dict)
        samples.append((train_neg, test_neg))

        del train_neg, test_neg, data
        torch.cuda.empty_cache()

    return samples


def create_data_with_neg_samples_parallel(data_base,
                                          device,
                                          negative_rate: float,
                                          num_samples: int,
                                          num_workers: int = None,
                                          node_label_dict: Dict=None) -> list:
    """
    并行版本，由claude生成
    """
    import resource
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (4096, hard))
    data_base = data_base.cpu()
    if num_workers is None:
        num_workers = max(1, mp.cpu_count() - 1)

    logging.info(f"Generating {num_samples} versions of data using {num_workers} workers...")

    mp.set_start_method('spawn', force=True)
    samples_per_worker = num_samples // num_workers
    remaining_samples = num_samples % num_workers

    worker_ranges = []
    start = 0
    for i in range(num_workers):
        count = samples_per_worker + (1 if i < remaining_samples else 0)
        worker_ranges.append((start, start + count))
        start += count

    # 并行计算
    with mp.Pool(processes=num_workers) as pool:
        results = pool.map(
            partial(generate_samples_range,
                    data_base,
                    negative_rate,
                    node_label_dict),
            worker_ranges)
    pool.close()
    pool.join()

    return [item for sublist in results for item in sublist]


def create_data_with_neg_samples_in_batches(data_base=None,
                                            device=None,
                                            negative_rate=None,
                                            total_epochs=None,
                                            batch_size=None,
                                            num_workers=None,
                                            config=None):
    """
    由claude3生成的测试版本
    """
    import math
    import logging
    from tqdm import tqdm

    if config is None:
        # Calculate number of batches needed
        num_batches = math.ceil(total_epochs / batch_size)
        remaining_epochs = total_epochs
        combined_data_list = []

        logging.info(f"Processing {total_epochs} epochs in {num_batches} batches of {batch_size}")

        # Process each batch
        for batch_num in tqdm(range(num_batches), desc="Processing batches"):
            # Calculate epochs for this batch
            current_batch_size = min(batch_size, remaining_epochs)

            logging.info(f"Generating batch {batch_num + 1}/{num_batches} with {current_batch_size} epochs")

            # Generate data for current batch
            batch_data_list = create_data_with_neg_samples_parallel(
                data_base=data_base,
                device=device,
                negative_rate=negative_rate,
                num_samples=current_batch_size,
                num_workers=num_workers
            )

            # Extend the combined list with current batch
            combined_data_list.extend(batch_data_list)

            # Update remaining epochs
            remaining_epochs -= current_batch_size

            logging.info(f"Completed batch {batch_num + 1}, total samples so far: {len(combined_data_list)}")

            # Clear batch data to free memory
            del batch_data_list
            import gc
            gc.collect()
            torch.cuda.empty_cache()

        logging.info(f"Completed all batches. Total samples generated: {len(combined_data_list)}")
        return combined_data_list

    else:
        processed_dir = config['dataset']['processed_dir']
        prefix = config['dataset']['edge_neg']
        pattern = os.path.join(processed_dir, f"{prefix}_*.pkl")

        neg_sample_paths = glob.glob(pattern)

        return neg_sample_paths


def prepare_batch_edges(pos_edge_dict, neg_edge_dict: Dict)->Tuple[Dict, Dict]:
    """
    由claude3生成并行边处理函数
    """
    edge_dict = {}
    labels_dict = {}

    for edge_type in pos_edge_dict.keys():
        if str(edge_type) in neg_edge_dict:
            combined_edges = torch.cat([
                pos_edge_dict[edge_type],
                neg_edge_dict[str(edge_type)]
            ], dim=1)

            combined_labels = torch.cat([
                torch.ones(pos_edge_dict[edge_type].size(1)),
                torch.zeros(neg_edge_dict[str(edge_type)].size(1))
            ])

            # Generate random permutation
            num_edges = combined_edges.size(1)
            perm = torch.randperm(num_edges)

            # Shuffle both edges and labels using the same permutation
            edge_dict[edge_type] = combined_edges[:, perm]
            labels_dict[str(edge_type)] = combined_labels[perm]
            #
            # edge_dict[edge_type] = combined_edges
            # labels_dict[str(edge_type)] = combined_labels


    return edge_dict, labels_dict


def compute_edge_metrics(model, data, edge_index_dict, neg_edge_dict, device, criterion, edge_type_weights=None):
    """
    Returns:
        tuple: (total_loss, all_predictions, all_labels)
    """
    edge_dict, labels_dict = prepare_batch_edges(edge_index_dict, neg_edge_dict)
    labels_dict = {k: v.to(device) for k, v in labels_dict.items()}
    edge_preds = model.edge_decoder(data['z_dict'], edge_dict)

    total_loss = 0
    all_preds = []
    all_labels = []
    test_auc_dict = {}
    test_aupr_dict = {}

    if edge_type_weights is None:
        edge_type_weights = {}

    if model.training:
        total_loss = 0

        for edge_type_str, edge_pred in edge_preds.items():
            edge_label = labels_dict[edge_type_str]
            edge_pred = edge_pred.to(device)

            edge_loss = criterion(edge_pred, edge_label)

            src_type, _, dst_type = eval(edge_type_str)

            if f"{src_type}-{dst_type}" in edge_type_weights:
                weight = edge_type_weights[f"{src_type}-{dst_type}"]
            else:
                weight = 1.0  # 其他边类型的默认权重

            total_loss += weight * edge_loss

            all_preds.append(edge_pred.detach())
            all_labels.append(edge_label)

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        return total_loss, all_preds, all_labels
    else:
        print("\nEdge Type Metrics:")
        for edge_type_str, edge_pred in edge_preds.items():
            edge_label = labels_dict[edge_type_str]
            edge_pred = edge_pred.to(device)
            total_loss += criterion(edge_pred, edge_label)

            # Calculate and print AUC and AUPR for the current edge type
            type_auc = calculate_auc(edge_pred.cpu(), edge_label.cpu())
            type_aupr = calculate_aupr(edge_pred.cpu(), edge_label.cpu())
            test_auc_dict[edge_type_str] = type_auc
            test_aupr_dict[edge_type_str] = type_aupr
            print(f"{edge_type_str} - AUC: {type_auc:.4f}, AUPR: {type_aupr:.4f}")

            all_preds.append(edge_pred.detach())
            all_labels.append(edge_label)

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        return total_loss, all_preds, all_labels, test_auc_dict, test_aupr_dict


def prepare_tf_cre_evaluation_data(data, min_positive_edges=100):
    # Get TF-CRE edge type
    tf_cre_edge_type = None
    for edge_type in data.edge_types:
        src_type, edge_relation, dst_type = edge_type
        if src_type == 'TF' and dst_type == 'CRE':
            tf_cre_edge_type = edge_type
            break

    if tf_cre_edge_type is None:
        logging.warning("TF-CRE edge type not found in the graph")
        return {}

    edge_index = data[tf_cre_edge_type].edge_index
    edge_attr = data[tf_cre_edge_type].edge_attr

    # Get the real column index for edge_id_T
    edge_id_t_idx = -1

    pos_mask = edge_attr[:, edge_id_t_idx] == 1
    pos_edge_index = edge_index[:, pos_mask]

    tf_to_edges = defaultdict(list)
    tf_to_cres = defaultdict(set)

    for i in range(pos_edge_index.size(1)):
        tf_id = pos_edge_index[0, i].item()
        cre_id = pos_edge_index[1, i].item()
        tf_to_edges[tf_id].append((tf_id, cre_id))
        tf_to_cres[tf_id].add(cre_id)

    # Get unique TFs and CREs
    tf_nodes = list(tf_to_edges.keys())
    all_cre_nodes = set()
    for cre_set in tf_to_cres.values():
        all_cre_nodes.update(cre_set)
    all_cre_nodes = list(all_cre_nodes)

    tf_evaluation_data = {}

    all_cre_nodes_array = np.array(list(all_cre_nodes))

    eligible_tfs = [tf_id for tf_id in tf_nodes if len(tf_to_edges[tf_id]) >= min_positive_edges]

    for tf_id in eligible_tfs:
        # Get positive CREs for this TF (convert to numpy array)
        pos_cres = np.array(list(tf_to_cres[tf_id]))

        # Create a mask of CREs that are not in the positive set
        neg_cre_mask = np.isin(all_cre_nodes_array, pos_cres, invert=True)
        neg_cres = all_cre_nodes_array[neg_cre_mask]

        if len(neg_cres) == 0:
            continue

        pos_edges = torch.tensor(tf_to_edges[tf_id], dtype=torch.long).t()

        tf_ids = np.repeat(tf_id, len(neg_cres))
        neg_edges = torch.tensor(np.vstack([tf_ids, neg_cres]), dtype=torch.long)

        all_edges = torch.cat([pos_edges, neg_edges], dim=1)

        labels = torch.cat([
            torch.ones(pos_edges.size(1)),
            torch.zeros(neg_edges.size(1))
        ])

        tf_evaluation_data[tf_id] = {
            "edge_type": tf_cre_edge_type,
            "edges": all_edges,
            "labels": labels,
            "num_positive": pos_edges.size(1),
            "num_negative": neg_edges.size(1)
        }

    return tf_evaluation_data


def evaluate_tf_cre_metrics(model, data, tf_evaluation_data, device):
    """
    Returns:
        dict: Dictionary containing AUPR and F1 scores for each TF and average metrics
    """
    results = {}
    tf_aupr_values = []
    tf_f1_values = []

    outputs = model(data)

    for tf_id, eval_data in tf_evaluation_data.items():
        edge_type = eval_data["edge_type"]
        edges = eval_data["edges"].to(device)
        labels = eval_data["labels"].to(device)

        edge_preds = model.edge_decoder(outputs['z_dict'], {edge_type: edges})

        preds = edge_preds[str(edge_type)]

        aupr = calculate_aupr(preds.cpu(), labels.cpu())

        f1 = calculate_max_f1(preds.cpu(), labels.cpu())

        results[f"TF_{tf_id}"] = {
            "aupr": aupr,
            "f1": f1,
            "num_positive": eval_data["num_positive"],
            "num_negative": eval_data["num_negative"]
        }

        tf_aupr_values.append(aupr)
        tf_f1_values.append(f1)

    if tf_aupr_values:
        results["average"] = {
            "aupr": np.mean(tf_aupr_values),
            "f1": np.mean(tf_f1_values),
            "num_tfs": len(tf_aupr_values)
        }

    return results


def calculate_max_f1(predictions, labels, beta=1.0):
    try:
        from sklearn.metrics import precision_recall_curve
        precision, recall, thresholds = precision_recall_curve(labels.numpy(), predictions.numpy())

        # Calculate F1 for each threshold
        f1_scores = []
        for p, r in zip(precision, recall):
            if p + r == 0:
                f1_scores.append(0)
            else:
                f_beta = (1 + beta ** 2) * (p * r) / ((beta ** 2 * p) + r)
                f1_scores.append(f_beta)

        return max(f1_scores)
    except:
        return 0.0

def calculate_auc(predictions, labels):
    try:
        from sklearn.metrics import roc_auc_score
        return roc_auc_score(labels.numpy(), predictions.numpy())
    except:
        return 0.0

def calculate_aupr(predictions, labels):
    try:
        from sklearn.metrics import average_precision_score
        return average_precision_score(labels.numpy(), predictions.numpy())
    except:
        return 0.0


def calculate_grn_consistency(model, data, top_k_ratio=0.1, device='cuda'):
    model.eval()
    with torch.no_grad():
        # Get embeddings from the model
        outputs = model(data)
        z_dict = outputs['z_dict']

        # Get all possible edge combinations for each edge type
        all_edges_dict = {}
        edge_scores_dict = {}

        for edge_type in data.train_edge_index_dict.keys():
            src_type, _, dst_type = edge_type if isinstance(edge_type, tuple) else eval(edge_type)

            # Get number of nodes for source and destination types
            num_src_nodes = data.x_dict[src_type].size(0)
            num_dst_nodes = data.x_dict[dst_type].size(0)

            # Generate all possible edges
            src_nodes = torch.arange(num_src_nodes, device=device)
            dst_nodes = torch.arange(num_dst_nodes, device=device)
            all_edges = torch.cartesian_prod(src_nodes, dst_nodes).t()

            # Remove self-loops if source and destination types are the same
            if src_type == dst_type:
                mask = all_edges[0] != all_edges[1]
                all_edges = all_edges[:, mask]

            all_edges_dict[edge_type] = all_edges

            # Calculate scores for all possible edges
            edge_scores = model.edge_decoder(z_dict, {edge_type: all_edges})
            edge_scores_dict[str(edge_type)] = edge_scores[str(edge_type)]

        # Calculate top K edges for each edge type
        consistency_scores = []
        for edge_type in data.train_edge_index_dict.keys():
            # Get real edges for this type
            real_edges = data.train_edge_index_dict[edge_type]
            # real_edges_set = {(int(src), int(dst)) for src, dst in real_edges.t()}
            real_edges_set = set((int(edge[0]), int(edge[1])) for edge in real_edges.t())

            # Get predictions and sort them
            edge_scores = edge_scores_dict[str(edge_type)]
            all_edges = all_edges_dict[edge_type]

            # Calculate number of top edges to consider
            num_real_edges = all_edges.size(1)
            top_k = int(num_real_edges * top_k_ratio)

            if top_k == 0:
                continue

            # Get top K predicted edges
            _, top_indices = torch.topk(edge_scores, k=top_k)
            top_edges = all_edges[:, top_indices]

            # Convert top predicted edges to set
            # pred_edges_set = {(int(src), int(dst)) for src, dst in top_edges.t()}
            pred_edges_set = set((int(edge[0]), int(edge[1])) for edge in top_edges.t())

            # Calculate overlap
            overlap = len(pred_edges_set.intersection(real_edges_set))

            # Calculate consistency for this edge type
            consistency = overlap / top_k
            consistency_scores.append(consistency)

        # Calculate average consistency across all edge types
        overall_consistency = sum(consistency_scores) / len(consistency_scores)

    return overall_consistency


def pearson_loss(pred, target):
    if pred.dim() == 1:
        pred = pred.unsqueeze(0)
    if target.dim() == 1:
        target = target.unsqueeze(0)


    # Calculate mean and standard deviation for each row
    pred_mean = pred.mean(dim=1, keepdim=True)
    target_mean = target.mean(dim=1, keepdim=True)

    pred_std = pred.std(dim=1, keepdim=True, unbiased=False)
    target_std = target.std(dim=1, keepdim=True, unbiased=False)

    # Handle zero standard deviation cases
    epsilon = 1e-8
    pred_std = torch.clamp(pred_std, min=epsilon)
    target_std = torch.clamp(target_std, min=epsilon)

    # Normalize each row
    pred_normalized = (pred - pred_mean) / pred_std
    target_normalized = (target - target_mean) / target_std

    # Calculate correlation for each row
    correlations = (pred_normalized * target_normalized).mean(dim=1)

    # Clip correlations to [-1, 1] range for numerical stability
    # correlations = torch.clamp(correlations, min=0, max=1.0)
    # correlations = correlations[correlations > 0]
    # Calculate mean correlation across all rows
    # top_50_correlations = torch.topk(correlations, 150, largest=True, sorted=False).values

    print(f"pos: {(correlations > 0).sum()} | neg: {(correlations < 0).sum()}")

    mean_correlation = correlations.mean()

    # Return loss (1 - correlation)
    return 1 - mean_correlation, correlations


def merge_fold_results(old_results, new_results, fold):
    new_correlation_dict, new_embedding_dict = new_results

    if old_results is None:
        old_correlation_dict, old_embedding_dict = {}, {}
    else:
        old_correlation_dict, old_embedding_dict = old_results


    merged_correlation_dict = {}
    for node_type in set(old_correlation_dict.keys()) | set(new_correlation_dict):
        correlation_dfs = []

        if node_type in old_correlation_dict:
            old_df = old_correlation_dict[node_type].copy()
            correlation_dfs.append(old_df)

        if node_type in new_correlation_dict:
            new_df = new_correlation_dict[node_type].copy()
            new_df['fold'] = fold
            correlation_dfs.append(new_df)

        if correlation_dfs:
            merged_correlation_dict[node_type] = pd.concat(correlation_dfs, ignore_index=True)

    merged_embedding_dict = {}
    for node_type in set(old_embedding_dict.keys()) | set(new_embedding_dict.keys()):
        embedding_dfs = []

        if node_type in old_embedding_dict:
            old_df = old_embedding_dict[node_type].copy()
            embedding_dfs.append(old_df)

        if node_type in new_embedding_dict:
            new_df = new_embedding_dict[node_type].copy()
            new_df['fold'] = fold
            embedding_dfs.append(new_df)

        if embedding_dfs:
            all_embeddings = pd.concat(embedding_dfs, ignore_index=True)

            dim_cols = [col for col in all_embeddings.columns if col.startswith('dim_')]

            # 计算平均嵌入
            avg_embeddings = all_embeddings.groupby('node_id')[dim_cols].mean().reset_index()
            avg_embeddings['fold'] = fold

            merged_embedding_dict[node_type] = avg_embeddings

    return merged_correlation_dict, merged_embedding_dict










