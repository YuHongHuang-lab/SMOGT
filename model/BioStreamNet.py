import os
import torch
import numpy as np
import pandas as pd
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
from tqdm import tqdm
from typing import Optional
import sys

class ExpressionDataset(Dataset):
    def __init__(self, tf_expr, cre_expr, target_expr, gene_network):
        self.gene_network = gene_network

        tf_expr = tf_expr.astype(np.float64)
        cre_expr = cre_expr.astype(np.float64)
        target_expr = target_expr.astype(np.float64)

        self.tf_indices = gene_network['related_TFs'].numpy()
        self.tf_expr = torch.FloatTensor(tf_expr[:, self.tf_indices])

        if 'all_CREs' in gene_network and len(gene_network['all_CREs']) > 0:
            self.cre_indices = gene_network['all_CREs'].numpy()
            self.cre_expr = torch.FloatTensor(cre_expr[:, self.cre_indices])
        else:
            self.cre_indices = []
            self.cre_expr = None

        # Indexing and expression of the target gene
        if 'gene_type' in gene_network and gene_network['gene_type'] == 'TF':
            self.target_idx = gene_network['gene_idx']
            target_column = self.target_idx
            source = tf_expr
        else:
            self.target_idx = gene_network['gene_idx']
            target_column = self.target_idx
            source = target_expr

        self.target_expr = torch.FloatTensor(source[:, target_column].reshape(-1, 1))

        self.n_samples = tf_expr.shape[0]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        if self.cre_expr is not None:
            return self.tf_expr[idx], self.cre_expr[idx], self.target_expr[idx]
        else:
            return self.tf_expr[idx], None, self.target_expr[idx]


class GlobalNetworkManager:
    """
    Global network object
    """
    def __init__(self,
                 TF_TF_network=None,
                 CRE_CRE_network=None,
                 TF_CRE_network=None,
                 CRE_TF_network=None,
                 CRE_Target_network=None,
                 TF_names=None,
                 CRE_names=None,
                 Target_names=None,
                 use_symbol=True):

        self.TF_names = TF_names
        self.CRE_names = CRE_names
        self.Target_names = Target_names

        self.TF_symbol_to_idx = {name: i for i, name in enumerate(self.TF_names)}
        self.CRE_symbol_to_idx = {name: i for i, name in enumerate(self.CRE_names)}
        self.Target_symbol_to_idx = {name: i for i, name in enumerate(self.Target_names)}

        self.TF_idx_to_symbol = {i: name for i, name in enumerate(self.TF_names)}
        self.CRE_idx_to_symbol = {i: name for i, name in enumerate(self.CRE_names)}
        self.Target_idx_to_symbol = {i: name for i, name in enumerate(self.Target_names)}

        if use_symbol:
            if TF_TF_network is not None:
                TF_TF_network = self._convert_network_to_idx(TF_TF_network, 'TF', 'TF')

            if CRE_CRE_network is not None:
                CRE_CRE_network = self._convert_network_to_idx(CRE_CRE_network, 'CRE', 'CRE')

            if TF_CRE_network is not None:
                TF_CRE_network = self._convert_network_to_idx(TF_CRE_network, 'TF', 'CRE')

            if CRE_TF_network is not None:
                CRE_TF_network = self._convert_network_to_idx(CRE_TF_network, 'CRE', 'TF')

            if CRE_Target_network is not None:
                CRE_Target_network = self._convert_network_to_idx(CRE_Target_network, 'CRE', 'Target')

        self.TF_TF_network = TF_TF_network
        self.CRE_CRE_network = CRE_CRE_network
        self.TF_CRE_network = TF_CRE_network
        self.CRE_TF_network = CRE_TF_network
        self.CRE_Target_network = CRE_Target_network

        self.TF_idx_map = self.TF_symbol_to_idx
        self.CRE_idx_map = self.CRE_symbol_to_idx
        self.Target_idx_map = self.Target_symbol_to_idx

        self._build_global_networks()

    def _convert_network_to_idx(self, network_df, from_type, to_type):
        network_df = network_df.copy()

        from_map = getattr(self, f"{from_type}_symbol_to_idx")
        to_map = getattr(self, f"{to_type}_symbol_to_idx")

        missing_from = [x for x in network_df['from'] if x not in from_map and isinstance(x, str)]
        missing_to = [x for x in network_df['to'] if x not in to_map and isinstance(x, str)]

        if missing_from:
            print(f"警告: 在网络文件中发现{len(missing_from)}个未知的{from_type}源节点")
            if len(missing_from) < 10:
                print(f"未知节点示例: {missing_from}")

        if missing_to:
            print(f"警告: 在网络文件中发现{len(missing_to)}个未知的{to_type}目标节点")
            if len(missing_to) < 10:
                print(f"未知节点示例: {missing_to}")

        from_is_str = network_df['from'].apply(lambda x: isinstance(x, str))
        to_is_str = network_df['to'].apply(lambda x: isinstance(x, str))

        network_df.loc[from_is_str, 'from'] = network_df.loc[from_is_str, 'from'].map(
            lambda x: from_map.get(x, -1))
        network_df.loc[to_is_str, 'to'] = network_df.loc[to_is_str, 'to'].map(
            lambda x: to_map.get(x, -1))

        valid_edges = (network_df['from'] != -1) & (network_df['to'] != -1)
        network_df = network_df[valid_edges]

        return network_df

    def _build_global_networks(self):
        """
        Global network
        """
        n_TFs = len(self.TF_names)
        n_CREs = len(self.CRE_names)
        n_Targets = len(self.Target_names)

        # TF-TF
        self.TF_TF_matrix = torch.zeros((n_TFs, n_TFs))
        if self.TF_TF_network is not None:
            for _, row in self.TF_TF_network.iterrows():
                src, tgt = int(row['from']), int(row['to'])
                if src < n_TFs and tgt < n_TFs:
                    self.TF_TF_matrix[src, tgt] = float(row['score'])

        # CRE-CRE
        self.CRE_CRE_matrix = torch.zeros((n_CREs, n_CREs))
        if self.CRE_CRE_network is not None:
            for _, row in self.CRE_CRE_network.iterrows():
                src, tgt = int(row['from']), int(row['to'])
                if src < n_CREs and tgt < n_CREs:
                    self.CRE_CRE_matrix[src, tgt] = float(row['score'])

        # TF-CRE
        self.TF_CRE_matrix = torch.zeros((n_TFs, n_CREs))
        if self.TF_CRE_network is not None:
            for _, row in self.TF_CRE_network.iterrows():
                src, tgt = int(row['from']), int(row['to'])
                if src < n_TFs and tgt < n_CREs:
                    self.TF_CRE_matrix[src, tgt] = float(row['score'])

        # CRE-TF
        self.CRE_TF_matrix = torch.zeros((n_CREs, n_TFs))
        if self.CRE_TF_network is not None:
            for _, row in self.CRE_TF_network.iterrows():
                src, tgt = int(row['from']), int(row['to'])
                if src < n_CREs and tgt < n_TFs:
                    self.CRE_TF_matrix[src, tgt] = float(row['score'])

        # CRE-Target
        self.CRE_Target_matrix = torch.zeros((n_CREs, n_Targets))
        if self.CRE_Target_network is not None:
            for _, row in self.CRE_Target_network.iterrows():
                src, tgt = int(row['from']), int(row['to'])
                if src < n_CREs and tgt < n_Targets:
                    self.CRE_Target_matrix[src, tgt] = float(row['score'])

    def get_gene_specific_network(self, gene_name, gene_type='Target'):
        """
        Obtain the subnetworks of specific genes and add self-loops and extended representations to all network layers
        """
        if gene_type == 'Target':
            if gene_name not in self.Target_idx_map:
                print(f"Warning: Target gene {gene_name} not found in network")
                return None, None

            Target_idx = self.Target_idx_map[gene_name]

            # direct CRE
            direct_CREs = torch.where(self.CRE_Target_matrix[:, Target_idx] > 0)[0]

            if len(direct_CREs) == 0:
                print(f"Warning: No direct CREs found for Target gene {gene_name}")
                return None, None

            sub_network_dfs = {}
            cre_output_edges_list = []
            direct_cre_names = [self.CRE_idx_to_symbol[i.item()] for i in direct_CREs]
            for name in direct_cre_names:
                cre_output_edges_list.append({'from': name, 'to': gene_name, 'type': 'CRE-Target'})
            if cre_output_edges_list:
                sub_network_dfs['CRE-Target'] = pd.DataFrame(cre_output_edges_list)

            # First-order CRE of direct CRE
            adjacent_CREs = []
            for cre_idx in direct_CREs:
                incoming_cres = torch.where(self.CRE_CRE_matrix[:, cre_idx] > 0)[0]
                adjacent_CREs.extend(incoming_cres.tolist())
            adjacent_CREs = torch.tensor(list(set(adjacent_CREs) - set(direct_CREs.tolist())))

            if len(adjacent_CREs) == 0:
                print(f"Warning: No adjacent CREs found for Target gene {gene_name}, adding a dummy adjacent CRE")
                adjacent_CREs = torch.tensor([])
                dummy_adjacent = True
            else:
                dummy_adjacent = False
                print(f"找到与{gene_name}相关的CRE: 直接相关{len(direct_CREs)}个, 上游邻接{len(adjacent_CREs)}个")

            all_CREs_tensor = torch.cat([direct_CREs])  if dummy_adjacent else torch.cat([adjacent_CREs, direct_CREs])

            # CRE connects to TF
            tfs_from_adj = []
            if not dummy_adjacent:
                for cre_idx in adjacent_CREs:
                    tfs = torch.where(self.TF_CRE_matrix[:, cre_idx] > 0)[0]
                    tfs_from_adj.extend(tfs.tolist())
            tfs_from_dir = []
            for cre_idx in direct_CREs:
                tfs = torch.where(self.TF_CRE_matrix[:, cre_idx] > 0)[0]
                tfs_from_dir.extend(tfs.tolist())
            related_TFs = torch.tensor(list(set(tfs_from_adj + tfs_from_dir)))

            if len(related_TFs) == 0:
                print(f"Warning: No TFs found related to CREs for Target gene {gene_name}")
                related_TFs = torch.tensor([])

            n_TFs = len(related_TFs)
            n_adjacent = len(adjacent_CREs)
            n_direct = len(direct_CREs)
            n_all_CREs = n_direct if dummy_adjacent else n_adjacent + n_direct

            tf_names = [self.TF_idx_to_symbol[i.item()] for i in related_TFs]
            all_cre_names = [self.CRE_idx_to_symbol[i.item()] for i in all_CREs_tensor]

            tf_tf_edges_list = []
            tf_cre_edges_list = []
            cre_cre_edges_list = []

            # Add diagonal self-loops to the TF-TF transformation matrix
            TF_TF_subnetwork = self.TF_TF_matrix[related_TFs][:, related_TFs].clone()
            active_tf_tf_indices = TF_TF_subnetwork.nonzero(as_tuple=False)
            for row, col in active_tf_tf_indices:
                tf_tf_edges_list.append({'from': tf_names[row.item()], 'to': tf_names[col.item()], 'type': 'TF-TF'})
            if tf_tf_edges_list:
                sub_network_dfs['TF_TF'] = pd.DataFrame(tf_tf_edges_list)
            for i in range(n_TFs):
                if TF_TF_subnetwork[i, i] == 0:
                    TF_TF_subnetwork[i, i] = 1.0

            # CRE-CRE transformation matrix (n_adjacent + n_direct) × n_direct
            CRE_CRE_subnetwork = torch.zeros((n_all_CREs, n_direct))
            if not dummy_adjacent:
                CRE_CRE_subnetwork[:n_adjacent, :] = self.CRE_CRE_matrix[adjacent_CREs][:, direct_CREs]
            CRE_CRE_subnetwork[n_adjacent:, :] = self.CRE_CRE_matrix[direct_CREs][:, direct_CREs]

            active_cre_cre_indices = CRE_CRE_subnetwork.nonzero(as_tuple=False)
            for row, col in active_cre_cre_indices:
                cre_cre_edges_list.append(
                    {'from': all_cre_names[row.item()], 'to': direct_cre_names[col.item()], 'type': 'CRE-CRE'})
            if cre_cre_edges_list:
                sub_network_dfs['CRE_CRE'] = pd.DataFrame(cre_cre_edges_list)
            for i in range(n_direct):  # Self-loop
                CRE_CRE_subnetwork[n_adjacent + i, i] = 1.0

            # Build an extended TF-CRE transformation matrix(n_TFs + n_all_CREs) × n_all_CREs
            TF_CRE_extended = torch.zeros((n_TFs + n_all_CREs, n_all_CREs))
            if n_TFs > 0:
                TF_CRE_extended[:n_TFs, :] = self.TF_CRE_matrix[related_TFs][:, all_CREs_tensor]

            active_tf_cre_indices = (TF_CRE_extended[:n_TFs, :] > 0).nonzero(as_tuple=False)
            for row, col in active_tf_cre_indices:
                tf_cre_edges_list.append(
                    {'from': tf_names[row.item()], 'to': all_cre_names[col.item()], 'type': 'TF-CRE'})
            if tf_cre_edges_list:
                sub_network_dfs['TF_CRE'] = pd.DataFrame(tf_cre_edges_list)
            for i in range(n_all_CREs):  # 自环
                TF_CRE_extended[n_TFs + i, i] = 1.0

            # Cre-target connection matrix (n_direct × 1)
            CRE_Target_subnetwork = self.CRE_Target_matrix[direct_CREs, Target_idx].view(-1, 1)

            cre_types = torch.zeros(n_all_CREs)
            cre_types[n_adjacent:] = 1

            # Return to the dictionary
            gene_network = {
                'gene_name': gene_name,
                'gene_idx': Target_idx,
                'gene_type': 'Target',
                'related_TFs': related_TFs,
                'all_CREs': all_CREs_tensor,
                'adjacent_CREs': adjacent_CREs,
                'direct_CREs': direct_CREs,
                'dummy_adjacent': dummy_adjacent,
                'cre_types': cre_types,
                'n_TFs': n_TFs,
                'n_adjacent': n_adjacent,
                'n_direct': n_direct,
                'n_all_CREs': n_all_CREs,
                'TF_TF_network': TF_TF_subnetwork,  # 网络
                'CRE_CRE_network': CRE_CRE_subnetwork,
                'TF_CRE_network': TF_CRE_extended,
                'CRE_Target_network': CRE_Target_subnetwork,
                'TF_TF_mask': (TF_TF_subnetwork > 0).float(),  # 网络掩码
                'CRE_CRE_mask': (CRE_CRE_subnetwork > 0).float(),
                'TF_CRE_mask': (TF_CRE_extended > 0).float(),
                'CRE_Target_mask': (CRE_Target_subnetwork > 0).float()
            }
            return gene_network, sub_network_dfs

        elif gene_type == 'TF':
            if gene_name not in self.TF_idx_map:
                print(f"Warning: TF gene {gene_name} not found in network")
                return None, None

            TF_idx = self.TF_idx_map[gene_name]

            # Direct CRE
            direct_CREs = torch.where(self.CRE_TF_matrix[:, TF_idx] > 0)[0]

            if len(direct_CREs) == 0:
                print(f"Warning: No direct CREs found for TF gene {gene_name}")
                return None, None

            sub_network_dfs = {}
            cre_output_edges_list = []
            direct_cre_names = [self.CRE_idx_to_symbol[i.item()] for i in direct_CREs]
            for name in direct_cre_names:
                cre_output_edges_list.append({'from': name, 'to': gene_name, 'type': 'CRE-TF'})
            if cre_output_edges_list:
                sub_network_dfs['CRE-TF'] = pd.DataFrame(cre_output_edges_list)

            # First-order CRE of direct CRE
            adjacent_CREs = []
            for cre_idx in direct_CREs:
                incoming_cres = torch.where(self.CRE_CRE_matrix[:, cre_idx] > 0)[0]
                adjacent_CREs.extend(incoming_cres.tolist())
            adjacent_CREs = torch.tensor(list(set(adjacent_CREs) - set(direct_CREs.tolist())))

            if len(adjacent_CREs) == 0:
                print(f"Warning: No adjacent CREs found for TF gene {gene_name}, adding a dummy adjacent CRE")
                adjacent_CREs = torch.tensor([])
                dummy_adjacent = True
            else:
                dummy_adjacent = False
                print(f"找到与TF {gene_name}相关的CRE: 直接相关{len(direct_CREs)}个, 上游邻接{len(adjacent_CREs)}个")

            all_CREs_tensor = torch.cat([direct_CREs])  if dummy_adjacent else torch.cat([adjacent_CREs, direct_CREs])

            # CRE connects to TF
            tfs_from_adj = []
            if not dummy_adjacent:
                for cre_idx in adjacent_CREs:
                    tfs = torch.where(self.TF_CRE_matrix[:, cre_idx] > 0)[0]
                    tfs_from_adj.extend(tfs.tolist())
            tfs_from_dir = []
            for cre_idx in direct_CREs:
                tfs = torch.where(self.TF_CRE_matrix[:, cre_idx] > 0)[0]
                tfs_from_dir.extend(tfs.tolist())
            related_TFs = torch.tensor(list(set(tfs_from_adj + tfs_from_dir)))

            if len(related_TFs) == 0:
                print(f"Warning: No TFs found related to CREs for TF gene {gene_name}")
                related_TFs = torch.tensor([])

            n_TFs = len(related_TFs)
            n_adjacent = len(adjacent_CREs)
            n_direct = len(direct_CREs)
            n_all_CREs = n_direct if dummy_adjacent else n_adjacent + n_direct

            tf_names = [self.TF_idx_to_symbol[i.item()] for i in related_TFs]
            all_cre_names = [self.CRE_idx_to_symbol[i.item()] for i in all_CREs_tensor]

            tf_tf_edges_list = []
            tf_cre_edges_list = []
            cre_cre_edges_list = []

            # Add diagonal self-loops to the TF-TF transformation matrix
            TF_TF_subnetwork = self.TF_TF_matrix[related_TFs][:, related_TFs].clone()
            active_tf_tf_indices = TF_TF_subnetwork.nonzero(as_tuple=False)
            for row, col in active_tf_tf_indices:
                tf_tf_edges_list.append({'from': tf_names[row.item()], 'to': tf_names[col.item()], 'type': 'TF-TF'})
            if tf_tf_edges_list:
                sub_network_dfs['TF_TF'] = pd.DataFrame(tf_tf_edges_list)
            for i in range(n_TFs):
                if TF_TF_subnetwork[i, i] == 0:
                    TF_TF_subnetwork[i, i] = 1.0

            # CRE-CRE transformation matrix (n_adjacent + n_direct) × n_direct
            CRE_CRE_subnetwork = torch.zeros((n_all_CREs, n_direct))
            if not dummy_adjacent:
                CRE_CRE_subnetwork[:n_adjacent, :] = self.CRE_CRE_matrix[adjacent_CREs][:, direct_CREs]
            CRE_CRE_subnetwork[n_adjacent:, :] = self.CRE_CRE_matrix[direct_CREs][:, direct_CREs]

            active_cre_cre_indices = CRE_CRE_subnetwork.nonzero(as_tuple=False)
            for row, col in active_cre_cre_indices:
                cre_cre_edges_list.append(
                    {'from': all_cre_names[row.item()], 'to': direct_cre_names[col.item()], 'type': 'CRE-CRE'})
            if cre_cre_edges_list:
                sub_network_dfs['CRE_CRE'] = pd.DataFrame(cre_cre_edges_list)
            for i in range(n_direct):  # Set the diagonal to 1 (self-loop)
                CRE_CRE_subnetwork[n_adjacent + i, i] = 1.0

            # Build an extended TF-CRE transformation matrix (n_TFs + n_all_CREs) × n_all_CREs
            TF_CRE_extended = torch.zeros((n_TFs + n_all_CREs, n_all_CREs))
            if n_TFs > 0:
                TF_CRE_extended[:n_TFs, :] = self.TF_CRE_matrix[related_TFs][:, all_CREs_tensor]

            active_tf_cre_indices = (TF_CRE_extended[:n_TFs, :] > 0).nonzero(as_tuple=False)
            for row, col in active_tf_cre_indices:
                tf_cre_edges_list.append(
                    {'from': tf_names[row.item()], 'to': all_cre_names[col.item()], 'type': 'TF-CRE'})
            if tf_cre_edges_list:
                sub_network_dfs['TF_CRE'] = pd.DataFrame(tf_cre_edges_list)
            for i in range(n_all_CREs):
                TF_CRE_extended[n_TFs + i, i] = 1.0

            # Cr-target connection matrix (n_direct × 1)
            CRE_TF_subnetwork = self.CRE_TF_matrix[direct_CREs, TF_idx].view(-1, 1)

            # Build a full CRE list and type tags
            all_CREs = direct_CREs if dummy_adjacent else torch.cat([adjacent_CREs, direct_CREs])
            cre_types = torch.zeros(n_all_CREs)
            cre_types[n_adjacent:] = 1

            gene_network = {
                'gene_name': gene_name,
                'gene_idx': TF_idx,
                'gene_type': 'TF',
                'related_TFs': related_TFs,
                'all_CREs': all_CREs,
                'adjacent_CREs': adjacent_CREs,
                'direct_CREs': direct_CREs,
                'dummy_adjacent': dummy_adjacent,
                'cre_types': cre_types,
                'n_TFs': n_TFs,
                'n_adjacent': n_adjacent,
                'n_direct': n_direct,
                'n_all_CREs': n_all_CREs,
                'TF_TF_network': TF_TF_subnetwork,  # Network connection
                'CRE_CRE_network': CRE_CRE_subnetwork,
                'TF_CRE_network': TF_CRE_extended,
                'CRE_Target_network': CRE_TF_subnetwork,
                'TF_TF_mask': (TF_TF_subnetwork > 0).float(),  # Network mask
                'CRE_CRE_mask': (CRE_CRE_subnetwork > 0).float(),
                'TF_CRE_mask': (TF_CRE_extended > 0).float(),
                'CRE_Target_mask': (CRE_TF_subnetwork > 0).float()  # Reuse variable names
            }

            return gene_network, sub_network_dfs

        else:
            raise ValueError(f"Unknown gene type: {gene_type}")



class BioStreamLinear(nn.Module):
    def __init__(self, in_features, out_features, mask, bias=True):
        super(BioStreamLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        if mask.shape[1] == out_features and mask.shape[0] == in_features:
            mask = mask.t()

        self.register_buffer('mask', mask)

        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

        with torch.no_grad():
            self.weight.data.mul_(self.mask)

        self.weight.register_hook(self.mask_gradient)

    def mask_gradient(self, grad):
        """Gradient hook: Ensure that only the connections allowed by the mask are updated"""
        return grad * self.mask

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / np.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        weight = self.weight * self.mask
        return F.linear(input, weight, self.bias)


class GeneSpecificModel(nn.Module):
    """
   Gene-specific network model
    """

    def __init__(self, gene_network):
        super(GeneSpecificModel, self).__init__()
        self.gene_network = gene_network

        self.n_TFs = gene_network['n_TFs']
        self.n_all_CREs = gene_network['n_all_CREs']
        self.n_direct = gene_network['n_direct']

        # TF-TF layer (n_TFs × n_TFs)
        self.TF_TF_layer = BioStreamLinear(
            self.n_TFs,
            self.n_TFs,
            gene_network['TF_TF_mask']
        )

        # TF+CRE-CRE layer((n_TFs + n_all_CREs) × n_all_CREs)
        self.TF_CRE_layer = BioStreamLinear(
            self.n_TFs + self.n_all_CREs,
            self.n_all_CREs,
            gene_network['TF_CRE_mask']
        )

        # CRE-CRE layer(n_all_CREs × n_direct)
        self.CRE_CRE_layer = BioStreamLinear(
            self.n_all_CREs,
            self.n_direct,
            gene_network['CRE_CRE_mask']
        )

        # CRE-Target(or TF) layer(n_direct × 1)
        self.CRE_Target_layer = BioStreamLinear(
            self.n_direct,
            1,
            gene_network['CRE_Target_mask']
        )

    def forward(self, tf_expr, cre_expr=None):
        batch_size = tf_expr.shape[0]

        # TF-TF propagation
        TF_exp_emb = F.relu(self.TF_TF_layer(tf_expr))

        if cre_expr is None:
            cre_expr = torch.zeros(batch_size, self.n_all_CREs, device=tf_expr.device)

        # Spliced TF_exp_emb and CRE_exp
        TF_emb_CRE_exp = torch.cat([TF_exp_emb, cre_expr], dim=1)

        # TF+CRE → CRE propagation: TF_emb_CRE_exp → CRE_exp_emb
        CRE_exp_emb = F.relu(self.TF_CRE_layer(TF_emb_CRE_exp))

        # CRE_exp_emb → CRE_exp_emb_direct
        CRE_exp_emb_direct = F.relu(self.CRE_CRE_layer(CRE_exp_emb))

        #CRE-Target propagation: CRE_exp_emb_direct → Target_predict
        Target_predict = self.CRE_Target_layer(CRE_exp_emb_direct)

        return Target_predict


class GeneExpressionPredictor:
    def __init__(self, global_network, device=None):
        self.global_network = global_network
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.tf_expr = None
        self.cre_expr = None
        self.target_expr = None
        self.train_indices = None
        self.test_indices = None

    def prepare_data(self, tf_expr, cre_expr, target_expr, train_df=None, test_size=0.2, random_state=42):
        """
        Split the training set and the test set
        """
        if isinstance(tf_expr, pd.DataFrame):
            self.cell_ids = tf_expr.columns.tolist()
        else:
            self.cell_ids = None

        # Gene sequence and global_"network consistency
        if isinstance(tf_expr, pd.DataFrame):
            tf_genes = tf_expr.index.tolist()
            network_tf_genes = self.global_network.TF_names

            missing_tfs = set(network_tf_genes) - set(tf_genes)
            if missing_tfs:
                print(f"警告: {len(missing_tfs)}个TF在表达矩阵中缺失")
                if len(missing_tfs) < 10:
                    print(f"缺失的TF示例: {list(missing_tfs)[:10]}")

            aligned_tf_expr = pd.DataFrame(index=network_tf_genes, columns=tf_expr.columns)
            for gene in network_tf_genes:
                if gene in tf_genes:
                    aligned_tf_expr.loc[gene] = tf_expr.loc[gene].values
                else:
                    aligned_tf_expr.loc[gene] = 0

            tf_expr = aligned_tf_expr.values.T
        else:
            # If it is already a numpy array, assume that the order has been aligned
            tf_expr = np.asarray(tf_expr)

        if isinstance(cre_expr, pd.DataFrame):
            cre_genes = cre_expr.index.tolist()
            network_cre_genes = self.global_network.CRE_names

            missing_cres = set(network_cre_genes) - set(cre_genes)
            if missing_cres:
                print(f"警告: {len(missing_cres)}个CRE在表达矩阵中缺失")
                if len(missing_cres) < 10:
                    print(f"缺失的CRE示例: {list(missing_cres)[:10]}")

            aligned_cre_expr = pd.DataFrame(index=network_cre_genes, columns=cre_expr.columns)
            for gene in network_cre_genes:
                if gene in cre_genes:
                    aligned_cre_expr.loc[gene] = cre_expr.loc[gene].values
                else:
                    aligned_cre_expr.loc[gene] = 0

            cre_expr = aligned_cre_expr.values.T
        else:
            cre_expr = np.asarray(cre_expr)

        if isinstance(target_expr, pd.DataFrame):
            target_genes = target_expr.index.tolist()
            network_target_genes = self.global_network.Target_names

            missing_targets = set(network_target_genes) - set(target_genes)
            if missing_targets:
                print(f"警告: {len(missing_targets)}个Target在表达矩阵中缺失")
                if len(missing_targets) < 10:
                    print(f"缺失的Target示例: {list(missing_targets)[:10]}")

            aligned_target_expr = pd.DataFrame(index=network_target_genes, columns=target_expr.columns)
            for gene in network_target_genes:
                if gene in target_genes:
                    aligned_target_expr.loc[gene] = target_expr.loc[gene].values
                else:
                    aligned_target_expr.loc[gene] = 0

            target_expr = aligned_target_expr.values.T
        else:
            target_expr = np.asarray(target_expr)

        self.tf_expr = tf_expr
        self.cre_expr = cre_expr
        self.target_expr = target_expr

        # Use train_df divides the training set and the test set
        if train_df is not None:
            if self.cell_ids is None:
                raise ValueError("当使用train_df时，tf_expr必须是DataFrame以提供细胞ID")

            common_cells = set(train_df['id']).intersection(set(self.cell_ids))
            if len(common_cells) == 0:
                raise ValueError("train_df中的id与细胞ID没有交集")

            cell_to_idx = {cell: idx for idx, cell in enumerate(self.cell_ids)}

            train_indices = []
            test_indices = []

            for _, row in train_df.iterrows():
                cell_id = row['id']
                if cell_id in cell_to_idx:
                    idx = cell_to_idx[cell_id]
                    if row['train'] == 1:
                        train_indices.append(idx)
                    else:
                        test_indices.append(idx)

            if not train_indices or not test_indices:
                raise ValueError("训练集或测试集为空，请检查train_df")

            self.train_indices = np.array(train_indices)
            self.test_indices = np.array(test_indices)

            print(f"数据准备完成: 训练样本数={len(self.train_indices)}, 测试样本数={len(self.test_indices)}")
        else:
            n_samples = tf_expr.shape[0]
            indices = np.arange(n_samples)
            self.train_indices, self.test_indices = train_test_split(
                indices, test_size=test_size, random_state=random_state
            )
            print(f"随机划分数据: 训练样本数={len(self.train_indices)}, 测试样本数={len(self.test_indices)}")

        print(f"表达矩阵形状:")
        print(f"  TF表达矩阵: {self.tf_expr.shape} (样本数 × TF数)")
        print(f"  CRE表达矩阵: {self.cre_expr.shape} (样本数 × CRE数)")
        print(f"  Target表达矩阵: {self.target_expr.shape} (样本数 × Target数)")

    def train_and_evaluate(self,
                           gene_name,
                           gene_type='auto',
                           batch_size=32,
                           lr=0.001, epochs=30,
                           save_dir=None,
                           verbose=False):
        """
        Train and evaluate simultaneously to find each gene to the optimal model
        """
        if self.tf_expr is None or self.target_expr is None:
            raise ValueError("Data not prepared. Call prepare_data first.")

        if gene_type == 'auto':
            gene_type = 'Target' if gene_name in self.global_network.Target_names else 'TF'

        gene_network, sub_network_dfs = self.global_network.get_gene_specific_network(gene_name, gene_type=gene_type)
        gene_network['gene_type'] = gene_type  # Add gene type information for the Dataset to use

        if save_dir and sub_network_dfs:
            pass

        if gene_network is None:
            raise ValueError(f"No network found for gene {gene_name}")

        dataset = ExpressionDataset(
            self.tf_expr,
            self.cre_expr,
            self.target_expr,
            gene_network,
        )

        train_dataset = torch.utils.data.Subset(dataset, self.train_indices)
        test_dataset = torch.utils.data.Subset(dataset, self.test_indices)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        model = GeneSpecificModel(gene_network).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        model.train()
        for epoch in range(epochs):
            train_loss = 0.0
            for batch in train_loader:
                if len(batch) == 3:
                    tf_batch, cre_batch, target_batch = batch
                    tf_batch = tf_batch.to(self.device)
                    cre_batch = cre_batch.to(self.device) if cre_batch is not None else None
                    target_batch = target_batch.to(self.device)

                    outputs = model(tf_batch, cre_batch)
                else:
                    tf_batch, _, target_batch = batch
                    tf_batch = tf_batch.to(self.device)
                    target_batch = target_batch.to(self.device)

                    outputs = model(tf_batch)

                loss = criterion(outputs, target_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            if (epoch + 1) % 1 == 0 or epoch == epochs - 1:
                model.eval()
                test_loss = 0.0
                all_preds = []
                all_targets = []

                with torch.no_grad():
                    for batch in test_loader:
                        if len(batch) == 3:
                            tf_batch, cre_batch, target_batch = batch
                            tf_batch = tf_batch.to(self.device)
                            cre_batch = cre_batch.to(self.device) if cre_batch is not None else None
                            target_batch = target_batch.to(self.device)

                            outputs = model(tf_batch, cre_batch)
                        else:
                            tf_batch, _, target_batch = batch
                            tf_batch = tf_batch.to(self.device)
                            target_batch = target_batch.to(self.device)

                            outputs = model(tf_batch)

                        loss = criterion(outputs, target_batch)
                        test_loss += loss.item()

                        all_preds.extend(outputs.cpu().numpy())
                        all_targets.extend(target_batch.cpu().numpy())

                # Calculate the Pearson correlation coefficient
                all_preds = np.array(all_preds).flatten()
                all_targets = np.array(all_targets).flatten()
                correlation, _ = pearsonr(all_preds, all_targets)

                if verbose:
                    print(
                        f"Epoch {epoch + 1}/{epochs}, Test Loss: {test_loss / len(test_loader):.6f}, Correlation: {correlation:.6f}")
                model.train()

        # Final assessment
        model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in test_loader:
                if len(batch) == 3:
                    tf_batch, cre_batch, target_batch = batch
                    tf_batch = tf_batch.to(self.device)
                    cre_batch = cre_batch.to(self.device) if cre_batch is not None else None
                    target_batch = target_batch.to(self.device)

                    outputs = model(tf_batch, cre_batch)
                else:
                    tf_batch, _, target_batch = batch
                    tf_batch = tf_batch.to(self.device)
                    target_batch = target_batch.to(self.device)

                    outputs = model(tf_batch)

                all_preds.extend(outputs.cpu().numpy())
                all_targets.extend(target_batch.cpu().numpy())

        # Calculate the evaluation indicators
        all_preds = np.array(all_preds).flatten()
        all_targets = np.array(all_targets).flatten()
        correlation, p_value = pearsonr(all_preds, all_targets)

        test_loss = np.mean((all_preds - all_targets) ** 2)
        print(f"Final Test Loss: {test_loss:.6f}, Correlation: {correlation:.6f}, p-value: {p_value:.6f}")

        # Save the model
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            model_path = os.path.join(save_dir, f"{gene_name}_model.pth")
            torch.save(model.state_dict(), model_path)
            print(f"Model saved to {model_path}")

        # Save the model
        evaluation = {
            'test_loss': test_loss,
            'correlation': correlation,
            'p_value': p_value,
            'predictions': all_preds,
            'targets': all_targets
        }

        return model, evaluation

    def predict(self, model, tf_expr=None, cre_expr=None):
        """
        """
        model.eval()
        gene_network = model.gene_network

        # If no expression data is provided, use the test set
        if tf_expr is None:
            # Use the existing test set
            gene_network['gene_type'] = gene_network.get('gene_type', 'Target')  # 确保有基因类型信息
            test_dataset = ExpressionDataset(
                self.tf_expr,
                self.cre_expr,
                self.target_expr,
                gene_network
            )
            test_dataset = torch.utils.data.Subset(test_dataset, self.test_indices)
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

            all_preds = []
            with torch.no_grad():
                for batch in test_loader:
                    if len(batch) == 3:
                        tf_batch, cre_batch, _ = batch
                        tf_batch = tf_batch.to(self.device)
                        cre_batch = cre_batch.to(self.device) if cre_batch is not None else None

                        outputs = model(tf_batch, cre_batch)
                    else:
                        tf_batch, _, _ = batch
                        tf_batch = tf_batch.to(self.device)

                        outputs = model(tf_batch)

                    all_preds.extend(outputs.cpu().numpy())

            return np.array(all_preds)
        else:
            tf_indices = gene_network['related_TFs'].numpy()
            tf_data = torch.FloatTensor(tf_expr[:, tf_indices]).to(self.device)

            cre_data = None
            if cre_expr is not None and 'all_CREs' in gene_network:
                cre_indices = gene_network['all_CREs'].numpy()
                cre_data = torch.FloatTensor(cre_expr[:, cre_indices]).to(self.device)

            with torch.no_grad():
                outputs = model(tf_data, cre_data)

            return outputs.cpu().numpy()

    def predict_with_structural_knockout(self, model, tf_expr, cre_expr, cre_knockout_list: list):
        """
        Make predictions after dynamically masking the model weights (structural knockout).
        """
        if not cre_knockout_list:
            return self.predict(model, tf_expr, cre_expr)

        model.eval()
        gene_network = model.gene_network

        original_weights = {
            'TF_CRE': model.TF_CRE_layer.weight.data.clone(),
            'CRE_CRE': model.CRE_CRE_layer.weight.data.clone(),
            'CRE_Target': model.CRE_Target_layer.weight.data.clone()
        }

        try:
            all_cre_names_in_model = [self.global_network.CRE_idx_to_symbol[i.item()] for i in gene_network['all_CREs']]
            direct_cre_names_in_model = [self.global_network.CRE_idx_to_symbol[i.item()] for i in
                                         gene_network['direct_CREs']]

            for cre_name in cre_knockout_list:
                if cre_name in all_cre_names_in_model:
                    cre_idx_local = all_cre_names_in_model.index(cre_name)
                    model.TF_CRE_layer.weight.data[cre_idx_local, :] = 0
                    model.CRE_CRE_layer.weight.data[:, cre_idx_local] = 0
                    if cre_name in direct_cre_names_in_model:
                        direct_cre_idx_local = direct_cre_names_in_model.index(cre_name)
                        model.CRE_Target_layer.weight.data[:, direct_cre_idx_local] = 0

            return self.predict(model, tf_expr, cre_expr)

        finally:
            # Whether it succeeds or fails, the original weights of the model are restored
            model.TF_CRE_layer.weight.data = original_weights['TF_CRE']
            model.CRE_CRE_layer.weight.data = original_weights['CRE_CRE']
            model.CRE_Target_layer.weight.data = original_weights['CRE_Target']

    def load_model(self, gene_name, model_path):
        gene_type = 'Target' if gene_name in self.global_network.Target_names else 'TF'
        gene_network,_ = self.global_network.get_gene_specific_network(gene_name, gene_type=gene_type)

        if gene_network is None:
            raise ValueError(f"No network found for gene {gene_name}")

        model = GeneSpecificModel(gene_network)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model = model.to(self.device)

        return model

    def analyze_weights(self, gene_name: str, model_path: str):
        """
        Load a trained gene-specific model and parse its parameters into a weighted multi-layer network.
        """
        print(f"--- 正在解析基因 {gene_name} 的模型权重 ---")

        # Load the model and the network structure saved inside it
        model = self.load_model(gene_name, model_path)
        gene_network = model.gene_network

        related_tf_names = [self.global_network.TF_idx_to_symbol[i.item()] for i in gene_network['related_TFs']]
        all_cre_names = [self.global_network.CRE_idx_to_symbol[i.item()] for i in gene_network['all_CREs']]

        direct_cre_names = []
        if 'direct_CREs' in gene_network and len(gene_network['direct_CREs']) > 0:
            direct_cre_names = [self.global_network.CRE_idx_to_symbol[i.item()] for i in gene_network['direct_CREs']]

        tf_tf_weights = model.TF_TF_layer.weight.data.cpu().numpy()
        tf_cre_weights = model.TF_CRE_layer.weight.data.cpu().numpy()
        cre_cre_weights = model.CRE_CRE_layer.weight.data.cpu().numpy()
        cre_target_weights = model.CRE_Target_layer.weight.data.cpu().numpy()

        all_networks_list = []

        # TF-TF layer
        rows, cols = np.nonzero(tf_tf_weights)
        if len(rows) > 0:
            df = pd.DataFrame({
                'from': [related_tf_names[j] for j in cols],
                'to': [related_tf_names[i] for i in rows],
                'weight': tf_tf_weights[rows, cols]
            })
            df['edge_id_type'] = 'TF_TF'
            all_networks_list.append(df)

        # TF-CRE layer
        rows, cols = np.nonzero(tf_cre_weights)
        tf_source_mask = cols < gene_network['n_TFs']
        if np.any(tf_source_mask):
            rows, cols = rows[tf_source_mask], cols[tf_source_mask]
            df = pd.DataFrame({
                'from': [related_tf_names[j] for j in cols],
                'to': [all_cre_names[i] for i in rows],
                'weight': tf_cre_weights[rows, cols]
            })
            df['edge_id_type'] = 'TF_CRE'
            all_networks_list.append(df)

        # CRE-CRE layer
        rows, cols = np.nonzero(cre_cre_weights)
        if len(rows) > 0 and direct_cre_names:
            df = pd.DataFrame({
                'from': [all_cre_names[j] for j in cols],
                'to': [direct_cre_names[i] for i in rows],
                'weight': cre_cre_weights[rows, cols]
            })
            df['edge_id_type'] = 'CRE_CRE'
            all_networks_list.append(df)

        # CRE-Target/TF layer
        rows, cols = np.nonzero(cre_target_weights)
        if len(rows) > 0 and direct_cre_names:
            output_layer_name = 'CRE-Target' if gene_network['gene_type'] == 'Target' else 'CRE-TF'
            df = pd.DataFrame({
                'from': [direct_cre_names[j] for j in cols],
                'to': gene_name,
                'weight': cre_target_weights[rows, cols]
            })
            df['edge_id_type'] = output_layer_name
            all_networks_list.append(df)

        print(f"--- 模型权重解析完成 ---")

        if not all_networks_list:
            return pd.DataFrame(columns=['from', 'to', 'weight', 'edge_id_type'])

        final_network_df = pd.concat(all_networks_list, ignore_index=True)

        return final_network_df




from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing


def simulate_perturbation(perturbations: dict,
                          predictor,
                          original_tf_expr_df,
                          original_cre_expr_df,
                          models_dir,
                          gene_list=None,
                          ncores=1):
    """
    Simulate TF and CRE Perturbations
        Args:
        perturbations (dict):
        Dictionary specifying perturbation settings,
        where keys are gene identifiers and values are fold changes or target expression levels.
        Example: {'GATA1': 2.0, 'MYC': 0.5}
        means upregulating GATA1 expression by 2-fold and downregulating MYC expression by half.

        predictor:
        A trained GeneExpressionPredictor object.
        It contains global regulatory network (HRNet) information and
        provides methods for loading gene-specific models and making predictions.
        This object has already loaded the original expression data via prepare_data.

        original_tf_expr_df (pd.DataFrame):
        Original TF expression dataframe, with genes as rows and samples as columns.

        original_cre_expr_df (pd.DataFrame):
        Original CRE chromatin accessibility dataframe, with elements as rows and samples as columns.

        models_dir (str or Path):
        Directory path storing trained gene-specific model parameters (e.g., models_and_networks_p).
        Each target gene corresponds to a .pth file (e.g., NFIA_model.pth) used for loading models and making predictions.

        gene_list (list, optional):
        List of target genes to predict.
        This list is typically pre-filtered to genes potentially relevant to the perturbation (e.g., genes with prediction accuracy > 0 during training).
        If None, genes involved in perturbations are used by default; however, explicitly passing a list here limits the prediction scope.

        ncores (int, optional):
        Number of CPU cores for parallel computation. Default is 1.

    Returns:
        The function returns a DataFrame
        containing the predicted expression values for each target gene
        in each cell after perturbation,
        which can be used for downstream cell fate trajectory analysis (e.g., generating vector field plots).
    """
    perturbed_tf_expr_df = original_tf_expr_df.copy()
    perturbed_cre_expr_df = original_cre_expr_df.copy()

    cre_knockout_list = []
    for name_to_perturb, value in perturbations.items():
        entity_type = None
        if name_to_perturb in original_tf_expr_df.index:
            target_df, source_df, entity_type = perturbed_tf_expr_df, original_tf_expr_df, "TF"
        elif name_to_perturb in original_cre_expr_df.index:
            target_df, source_df, entity_type = perturbed_cre_expr_df, original_cre_expr_df, "CRE"
        else:
            print(f"警告: 扰动对象 '{name_to_perturb}' 在TF和CRE列表中均未找到，已跳过。")
            continue

        if value == 0:
            target_df.loc[name_to_perturb, :] = 0
            if entity_type == "CRE":
                cre_knockout_list.append(name_to_perturb)
        elif value > 0:
            original_vector = source_df.loc[name_to_perturb, :]
            target_df.loc[name_to_perturb, :] = original_vector * value
        else:
            print(f"警告: 扰动值 '{value}' 无效，已跳过 {entity_type} {name_to_perturb}。")

    if cre_knockout_list:
        print(f"将对以下CRE进行模型结构性敲除: {cre_knockout_list}")


    # The numpy array format (cell x gene) required for conversion to the prediction function
    perturbed_tf_expr_np = perturbed_tf_expr_df.values.T
    perturbed_cre_expr_np = perturbed_cre_expr_df.values.T

    # Search for genes regulated by this TF and CRE
    all_perturbed_entities = list(perturbations.keys())
    affected_genes = set()
    for filename in os.listdir(models_dir):
        if filename.endswith("_network.csv"):
            gene_name = filename.replace("_network.csv", "")
            network_df = pd.read_csv(os.path.join(models_dir, filename))
            if any(entity in network_df['from'].values or entity in network_df['to'].values for entity in
                   all_perturbed_entities):
                affected_genes.add(gene_name)

    affected_genes = list(affected_genes)

    if not affected_genes:
        print(f"未找到受 {all_perturbed_entities} 调控的模型。退出。")
        return pd.DataFrame()

    print(f"找到 {len(affected_genes)} 个可能受扰动影响的靶基因。")

    if gene_list is not None:
        affected_genes = list(set(gene_list) & set(affected_genes))

    all_predictions = {}
    num_samples = original_tf_expr_df.shape[1]

    if ncores == 1:
        for gene_name in tqdm(affected_genes, desc=f"靶基因"):
            model_path = os.path.join(models_dir, f"{gene_name}_model.pth")
            if not os.path.exists(model_path):
                print(f"警告: {gene_name} 的网络文件存在，但模型文件缺失。跳过。")
                all_predictions[gene_name] = np.full(num_samples, np.nan)
                continue

            try:
                model = predictor.load_model(gene_name, model_path)

                if cre_knockout_list:
                    predictions = predictor.predict_with_structural_knockout(
                        model,
                        tf_expr=perturbed_tf_expr_np,
                        cre_expr=perturbed_cre_expr_np,
                        cre_knockout_list=cre_knockout_list
                    )
                else:
                    predictions = predictor.predict(model, tf_expr=perturbed_tf_expr_np, cre_expr=perturbed_cre_expr_np)

                all_predictions[gene_name] = predictions.flatten()

                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"为基因 {gene_name} 进行预测时发生错误: {e}")
                all_predictions[gene_name] = np.full(num_samples, np.nan)

    else:
        # Parallel processing uses a process pool
        print(f"使用 {ncores} 个核心进行并行处理...")

        # Prepare global_The network information is used for the reconstruction of child processes
        global_network_info = {
            'TF_names': predictor.global_network.TF_names,
            'CRE_names': predictor.global_network.CRE_names,
            'Target_names': predictor.global_network.Target_names,
            'TF_TF_network': predictor.global_network.TF_TF_network,
            'CRE_CRE_network': predictor.global_network.CRE_CRE_network,
            'TF_CRE_network': predictor.global_network.TF_CRE_network,
            'CRE_TF_network': predictor.global_network.CRE_TF_network,
            'CRE_Target_network': predictor.global_network.CRE_Target_network,
            'use_symbol': True
        }

        # Prepare the parameters for all tasks
        args_list = []
        for gene_name in affected_genes:
            model_path = os.path.join(models_dir, f"{gene_name}_model.pth")
            args_list.append((gene_name, model_path, perturbed_tf_expr_np, perturbed_cre_expr_np,
                              global_network_info, models_dir))

        # Use ProcessPoolExecutor for parallel processing
        with ProcessPoolExecutor(max_workers=ncores) as executor:
            # Submit all tasks
            future_to_gene = {
                executor.submit(_predict_single_gene_knockout_standalone, args): args[0]
                for args in args_list
            }

            # Collect results
            for future in tqdm(as_completed(future_to_gene),
                               total=len(affected_genes),
                               desc=f"靶基因"):
                try:
                    gene_name, predictions = future.result()
                    all_predictions[gene_name] = predictions
                except Exception as e:
                    gene_name = future_to_gene[future]
                    print(f"为基因 {gene_name} 进行预测时发生错误: {e}")
                    all_predictions[gene_name] = np.full(num_samples, np.nan)

    if not all_predictions:
        print("没有生成任何预测结果。")
        return pd.DataFrame()

    gene_predict_df = pd.DataFrame(all_predictions, index=original_tf_expr_df.columns).T
    gene_predict_df.index.name = 'gene'

    return gene_predict_df

# simulate_knockout Multiprocessing Component
def _predict_single_gene_knockout_standalone(args):
    """
    An independent gene prediction function runs in the child process
    """
    (gene_name, model_path, perturbed_tf_expr_np, cre_expr_np,
     global_network_info, models_dir) = args

    if not os.path.exists(model_path):
        return gene_name, np.full(perturbed_tf_expr_np.shape[0], np.nan)

    try:
        # Recreate all necessary objects in the child process
        global_network = GlobalNetworkManager(**global_network_info)
        predictor = GeneExpressionPredictor(global_network)

        model = predictor.load_model(gene_name, model_path)
        predictions = predictor.predict(model, tf_expr=perturbed_tf_expr_np, cre_expr=cre_expr_np)

        # Clear the memory
        del model, predictor, global_network
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return gene_name, predictions.flatten()

    except Exception as e:
        print(f"为基因 {gene_name} 进行预测时发生错误: {e}")
        return gene_name, np.full(perturbed_tf_expr_np.shape[0], np.nan)


# Rewrite SCENIC+ code for perturbation modeling
from scipy import sparse
from sklearn.neighbors import NearestNeighbors
from scipy.stats import norm as normal
from velocyto.estimation import colDeltaCorpartial
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import matplotlib.colors as mcolors
import anndata as ad

def _project_perturbation_in_embedding(
        original_matrix_np,
        perturbed_matrix_np,
        embedding,
        sigma_corr=0.05, n_cpu=1):
    """
    Calculate the projection of the perturbation in the embedding space (delta embedding)
    """
    delta_matrix = perturbed_matrix_np.astype('double') - original_matrix_np.astype('double')

    n_neighbors = int(perturbed_matrix_np.shape[0] / 5)
    nn = NearestNeighbors(n_neighbors=n_neighbors + 1, n_jobs=n_cpu)
    nn.fit(embedding)
    embedding_knn = nn.kneighbors_graph(mode='connectivity')

    neigh_ixs = embedding_knn.indices.reshape((-1, n_neighbors + 1))
    p = np.linspace(0.5, 0.1, neigh_ixs.shape[1])
    p = p / p.sum()

    # Randomly select neighbors for calculation
    sampling_ixs = np.stack([np.random.choice(neigh_ixs.shape[1],
                                              size=(int(0.3 * (n_neighbors + 1)),),
                                              replace=False,
                                              p=p) for i in range(neigh_ixs.shape[0])], 0)

    neigh_ixs = neigh_ixs[np.arange(neigh_ixs.shape[0])[:, None], sampling_ixs]

    nonzero = neigh_ixs.shape[0] * neigh_ixs.shape[1]
    embedding_knn = sparse.csr_matrix((np.ones(nonzero),
                                       neigh_ixs.ravel(),
                                       np.arange(0, nonzero + 1, neigh_ixs.shape[1])),
                                      shape=(neigh_ixs.shape[0],
                                             neigh_ixs.shape[0]))

    # Calculate the correlation
    perturbed_matrix_T = np.ascontiguousarray(perturbed_matrix_np.T)
    delta_matrix_T = np.ascontiguousarray(delta_matrix.T)

    corrcoef = colDeltaCorpartial(perturbed_matrix_T, delta_matrix_T, neigh_ixs, threads=n_cpu)
    corrcoef[np.isnan(corrcoef)] = 1
    np.fill_diagonal(corrcoef, 0)

    # Calculate the transition probability
    transition_prob = np.exp(corrcoef / sigma_corr) * embedding_knn.toarray()
    transition_prob /= transition_prob.sum(1)[:, None]

    # Calculate the direction vector
    unitary_vectors = embedding.T[:, None, :] - embedding.T[:, :, None]
    unitary_vectors /= np.linalg.norm(unitary_vectors, ord=2, axis=0)
    np.fill_diagonal(unitary_vectors[0, ...], 0)
    np.fill_diagonal(unitary_vectors[1, ...], 0)

    delta_embedding = (transition_prob * unitary_vectors).sum(2)
    delta_embedding = delta_embedding - ((embedding_knn.toarray() * unitary_vectors).sum(2) / embedding_knn.sum(1).A.T)
    delta_embedding = delta_embedding.T
    return delta_embedding


# File: BioStreamNet.py (or test_Target_predicted.py)

def _calculate_grid_arrows(embedding, delta_embedding, offset_frac, n_grid_cols, n_grid_rows, n_neighbors, n_cpu):
    """
    Prepare the arrow grid
    """
    min_x, max_x = np.min(embedding[:, 0]), np.max(embedding[:, 0])
    min_y, max_y = np.min(embedding[:, 1]), np.max(embedding[:, 1])

    offset_x = (max_x - min_x) * offset_frac
    offset_y = (max_y - min_y) * offset_frac

    x_dist_between_points = (max_x - min_x) / n_grid_cols
    y_dist_between_points = (max_y - min_y) / n_grid_rows
    minimal_distance = np.mean([y_dist_between_points, x_dist_between_points])

    grid_x, grid_y = np.meshgrid(
        np.linspace(min_x + offset_x, max_x - offset_x, n_grid_cols),
        np.linspace(min_y + offset_y, max_y - offset_y, n_grid_rows)
    )
    grid_xy = np.array([np.hstack(grid_x), np.hstack(grid_y)]).T

    nn = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=n_cpu)
    nn.fit(embedding)
    dists, neighs = nn.kneighbors(grid_xy)

    std = np.mean([abs(g[1] - g[0]) for g in grid_xy])
    gaussian_w = normal.pdf(loc=0, scale=0.5 * std, x=dists)
    total_p_mass = gaussian_w.sum(1)

    uv = (delta_embedding[neighs] * gaussian_w[:, :, None]).sum(1) / np.maximum(1, total_p_mass)[:, None]

    mask = dists.min(1) < minimal_distance

    return grid_xy, uv, mask


def plot_perturbation_effect_in_embedding(
        original_matrix_df: pd.DataFrame,
        perturbed_matrix_df: pd.DataFrame,
        embedding_df: pd.DataFrame,
        cell_metadata_df: pd.DataFrame,
        variable_to_color: str,
        grid_n_cols: Optional[int] = 25,
        grid_n_rows: Optional[int] = 25,
        n_cpu: Optional[int] = 1,
        n_neighbors=25,
        figsize: Optional[tuple] = (8, 8),
        save: Optional[str] = None,
        **kwargs):
    """
   Draw the arrow grid of the perturbation effect on the two-dimensional embedding graph
    """
    delta_embedding = _project_perturbation_in_embedding(
        original_matrix_np=original_matrix_df.values,
        perturbed_matrix_np=perturbed_matrix_df.values,
        embedding=embedding_df[['X', 'Y']].values,
        n_cpu=n_cpu)

    grid_xy, uv, mask = _calculate_grid_arrows(
        embedding=embedding_df[['X', 'Y']].values,
        delta_embedding=delta_embedding,
        offset_frac=0.05,
        n_grid_cols=grid_n_cols,
        n_grid_rows=grid_n_rows,
        n_neighbors=n_neighbors,
        n_cpu=n_cpu)

    def scale_array(X):
        min_val = np.min(X)
        max_val = np.max(X)
        range_val = max_val - min_val
        if range_val == 0:
            return np.zeros_like(X)
        return (X - min_val) / range_val

    distances = np.sqrt((uv ** 2).sum(1))
    norm = mcolors.Normalize(vmin=0.15, vmax=0.5, clip=True)
    uv[~mask] = np.nan

    fig, ax = plt.subplots(figsize=figsize)

    #Draw a scatter plot of cells
    groups = cell_metadata_df[variable_to_color].unique()
    for group in groups:
        idx_to_plot = cell_metadata_df[variable_to_color] == group
        ax.scatter(
            embedding_df.loc[idx_to_plot, 'X'],
            embedding_df.loc[idx_to_plot, 'Y'],
            label=group,
            s=10, alpha=0.6
        )

    # Draw the flow graph
    ax.streamplot(
        grid_xy.reshape(grid_n_cols, grid_n_rows, 2)[:, :, 0],
        grid_xy.reshape(grid_n_cols, grid_n_rows, 2)[:, :, 1],
        uv.reshape(grid_n_cols, grid_n_rows, 2)[:, :, 0],
        uv.reshape(grid_n_cols, grid_n_rows, 2)[:, :, 1],
        density=1.5,
        color=scale_array(distances).reshape(grid_n_cols, grid_n_rows),  # 核心修改：使用归一化后的箭头长度作为颜色
        cmap='Greys',
        norm=norm,
        zorder=10,
        linewidth=0.5
    )

    ax.legend(markerscale=3)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"Perturbation effect of knocking out {kwargs.get('tf_knockout', '')}")

    if save is not None:
        fig.savefig(save)
        print(f"图像已保存至: {save}")
    else:
        plt.show(fig)

    return ax


def run_perturbation_analysis(
        perturbations,
        predictor,
        original_tf_expr_df,
        original_cre_expr_df,
        original_target_expr_df,
        models_dir,
        output_dir=None,
        adata = None,
        gene_list = None,
        ko_results_path = None,
        embedding_key = 'X_umap',
        metadata_key = 'celltype',
        n_cpu: int = 1,
        tf_name='A'
):
    """
    Perform an End-to-End Analysis Pipeline from Perturbation Simulation to Visualization
        Args:

        perturbations (dict):
        Same as in simulate_perturbation.

        predictor:
        Same as in simulate_perturbation.

        original_tf_expr_df (pd.DataFrame):
        Same as in simulate_perturbation.

        original_cre_expr_df (pd.DataFrame):
        Same as in simulate_perturbation.

        models_dir (str or Path):
        Same as in simulate_perturbation.

        output_dir:
        Directory where output files (e.g., plots) will be saved.

        adata:
        An AnnData object containing dimensionality reduction coordinates (e.g., UMAP) and
        cell metadata (e.g., cell types) for visualization mapping.

        gene_list (list, optional):
        Same as in simulate_perturbation.

        ko_results_path:
        Cache file path for saving/loading results from simulate_perturbation.
        If the path exists and the file is valid, results will be loaded directly to avoid redundant computation.

        embedding_key (str):
        String specifying the key in adata.obsm where dimensionality reduction coordinates are stored (e.g., 'X_umap').
        The function extracts the first two dimensions from this key for plotting.

        metadata_key (str):
        String specifying the column name in adata.obs used for cell coloring,
        such as the cell type column 'celltype' or 'seurat_clusters'.

        n_cpu (int, optional):
        Number of CPU cores for parallel computation. Default is 1.

        tf_name (str):
        String identifier for the perturbed factor,
        used in plot titles and output filenames (e.g., 'NFIA_2') to distinguish results from different perturbation experiments.
    """
    print(f"\n--- 开始端到端扰动分析 ---")

    gene_predict_df = None
    if ko_results_path and os.path.exists(ko_results_path):
        try:
            gene_predict_df = pd.read_csv(ko_results_path, index_col='gene')
        except Exception as e:
            print(f"读取缓存文件失败: {e}。将重新进行计算。")
            gene_predict_df = None

    if gene_predict_df is None:
        gene_predict_df = simulate_perturbation(perturbations=perturbations,
                                                predictor=predictor,
                                                original_tf_expr_df=original_tf_expr_df,
                                                original_cre_expr_df=original_cre_expr_df,
                                                models_dir=models_dir,
                                                gene_list=gene_list,
                                                ncores=1)

        if ko_results_path and not gene_predict_df.empty:
            print(f"将新的扰动模拟结果保存至缓存路径: {ko_results_path}")
            os.makedirs(os.path.dirname(ko_results_path), exist_ok=True)
            gene_predict_df.to_csv(ko_results_path)

    if gene_predict_df.empty:
        print(f"扰动模拟未能生成任何预测结果，分析终止。")
        return

    print("\n--- 开始准备可视化数据 ---")

    original_full_matrix_df = pd.concat([original_tf_expr_df, original_target_expr_df])
    original_matrix_aligned_df = original_full_matrix_df.loc[gene_predict_df.index]

    # Prepare the expression matrix after perturbation
    perturbed_matrix_df = original_matrix_aligned_df.copy()
    perturbed_matrix_df.update(gene_predict_df)

    # adata = ad.read_h5ad(embedding_adata_path)
    # adata.obs_names = adata.obs_names.str.replace('-', '.')

    embedding_coords = adata.obsm[embedding_key][:, :2]
    embedding_df = pd.DataFrame(embedding_coords, index=adata.obs_names, columns=['X', 'Y'])

    if metadata_key not in adata.obs.columns:
        raise KeyError(f"在 anndata.obs 中未找到元数据列 '{metadata_key}'，请检查 .h5ad 文件。")
    cell_metadata_for_plot = adata.obs[[metadata_key]].copy()

    common_cells = cell_metadata_for_plot.index.intersection(embedding_df.index).intersection(
        original_matrix_aligned_df.columns)

    cell_metadata = adata.obs

    cell_metadata_for_plot = cell_metadata_for_plot.loc[common_cells]
    cell_metadata = cell_metadata.loc[common_cells]
    embedding_df = embedding_df.loc[common_cells]
    original_matrix_aligned_df = original_matrix_aligned_df[common_cells]
    perturbed_matrix_df = perturbed_matrix_df[common_cells]

    plot_perturbation_effect_in_embedding(
        original_matrix_df=original_matrix_aligned_df.T,
        perturbed_matrix_df=perturbed_matrix_df.T,
        embedding_df=embedding_df,
        cell_metadata_for_plot=cell_metadata_for_plot,
        cell_metadata_df=cell_metadata,
        variable_to_color=metadata_key,
        n_cpu=n_cpu,
        tf_knockout=tf_name,
        save=os.path.join(output_dir, f'perturbation_effect_{tf_name}_by_{metadata_key}.pdf')
    )





