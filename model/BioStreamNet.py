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

        # 目标基因的索引和表达
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
    全局网络对象
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
        全局网络
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
        获取特定基因的子网络，为所有网络层添加自环和扩展表示
        """
        if gene_type == 'Target':
            if gene_name not in self.Target_idx_map:
                print(f"Warning: Target gene {gene_name} not found in network")
                return None

            Target_idx = self.Target_idx_map[gene_name]

            # 直接CRE
            direct_CREs = torch.where(self.CRE_Target_matrix[:, Target_idx] > 0)[0]

            if len(direct_CREs) == 0:
                print(f"Warning: No direct CREs found for Target gene {gene_name}")
                return None

            # 直接CRE的一阶CRE
            adjacent_CREs = []
            for cre_idx in direct_CREs:
                incoming_cres = torch.where(self.CRE_CRE_matrix[:, cre_idx] > 0)[0]
                adjacent_CREs.extend(incoming_cres.tolist())

            adjacent_CREs = list(set(adjacent_CREs) - set(direct_CREs.tolist()))
            adjacent_CREs = torch.tensor(adjacent_CREs)

            if len(adjacent_CREs) == 0:
                print(f"Warning: No adjacent CREs found for Target gene {gene_name}, adding a dummy adjacent CRE")
                adjacent_CREs = torch.tensor([0])
                dummy_adjacent = True
            else:
                dummy_adjacent = False
                print(f"找到与{gene_name}相关的CRE: 直接相关{len(direct_CREs)}个, 上游邻接{len(adjacent_CREs)}个")

            # CRE连接TF
            related_TFs = []

            if not dummy_adjacent:
                for cre_idx in adjacent_CREs:
                    tfs = torch.where(self.TF_CRE_matrix[:, cre_idx] > 0)[0]
                    related_TFs.extend(tfs.tolist())

            for cre_idx in direct_CREs:
                tfs = torch.where(self.TF_CRE_matrix[:, cre_idx] > 0)[0]
                related_TFs.extend(tfs.tolist())

            related_TFs = torch.tensor(list(set(related_TFs)))

            if len(related_TFs) == 0:
                print(f"Warning: No TFs found related to CREs for Target gene {gene_name}")
                return None

            n_TFs = len(related_TFs)
            n_adjacent = len(adjacent_CREs)
            n_direct = len(direct_CREs)
            n_all_CREs = n_adjacent + n_direct

            # TF-TF转换矩阵 - 添加对角线自环
            TF_TF_subnetwork = self.TF_TF_matrix[related_TFs][:, related_TFs].clone()

            for i in range(n_TFs):
                if TF_TF_subnetwork[i, i] == 0:
                    TF_TF_subnetwork[i, i] = 1.0

            # CRE-CRE转换矩阵 (n_adjacent + n_direct) × n_direct
            CRE_CRE_subnetwork = torch.zeros((n_all_CREs, n_direct))

            for i, adj_cre in enumerate(adjacent_CREs):
                for j, dir_cre in enumerate(direct_CREs):
                    if dummy_adjacent:
                        CRE_CRE_subnetwork[i, j] = 0
                    else:
                        CRE_CRE_subnetwork[i, j] = self.CRE_CRE_matrix[adj_cre, dir_cre]

            for i, dir_cre_i in enumerate(direct_CREs):
                for j, dir_cre_j in enumerate(direct_CREs):
                    if i == j:
                        CRE_CRE_subnetwork[n_adjacent + i, j] = 1.0
                    else:
                        CRE_CRE_subnetwork[n_adjacent + i, j] = self.CRE_CRE_matrix[dir_cre_i, dir_cre_j]

            # 构建扩展的TF-CRE转换矩阵 (n_TFs + n_all_CREs) × n_all_CREs
            TF_CRE_extended = torch.zeros((n_TFs + n_all_CREs, n_all_CREs))

            for i, tf_idx in enumerate(related_TFs):
                if not dummy_adjacent:
                    for j, cre_idx in enumerate(adjacent_CREs):
                        TF_CRE_extended[i, j] = self.TF_CRE_matrix[tf_idx, cre_idx]

                for j, cre_idx in enumerate(direct_CREs):
                    TF_CRE_extended[i, n_adjacent + j] = self.TF_CRE_matrix[tf_idx, cre_idx]

            for i in range(n_all_CREs):
                TF_CRE_extended[n_TFs + i, i] = 1.0

            # CRE-Target连接矩阵 (n_direct × 1)
            CRE_Target_subnetwork = torch.zeros(n_direct, 1)
            for i, cre_idx in enumerate(direct_CREs):
                CRE_Target_subnetwork[i, 0] = self.CRE_Target_matrix[cre_idx, Target_idx]

            TF_TF_mask = (TF_TF_subnetwork > 0).float()
            CRE_CRE_mask = (CRE_CRE_subnetwork > 0).float()
            TF_CRE_mask = (TF_CRE_extended > 0).float()
            CRE_Target_mask = (CRE_Target_subnetwork > 0).float()

            if dummy_adjacent:
                all_CREs = direct_CREs
            else:
                all_CREs = torch.cat([adjacent_CREs, direct_CREs])

            cre_types = torch.zeros(n_all_CREs)
            cre_types[n_adjacent:] = 1

            return {
                'gene_name': gene_name,
                'gene_idx': Target_idx,
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

                # 网络
                'TF_TF_network': TF_TF_subnetwork,
                'CRE_CRE_network': CRE_CRE_subnetwork,
                'TF_CRE_network': TF_CRE_extended,
                'CRE_Target_network': CRE_Target_subnetwork,

                # 网络掩码
                'TF_TF_mask': TF_TF_mask,
                'CRE_CRE_mask': CRE_CRE_mask,
                'TF_CRE_mask': TF_CRE_mask,
                'CRE_Target_mask': CRE_Target_mask
            }
        elif gene_type == 'TF':
            if gene_name not in self.TF_idx_map:
                print(f"Warning: TF gene {gene_name} not found in network")
                return None

            TF_idx = self.TF_idx_map[gene_name]

            # 直接CRE
            direct_CREs = torch.where(self.CRE_TF_matrix[:, TF_idx] > 0)[0]

            if len(direct_CREs) == 0:
                print(f"Warning: No direct CREs found for TF gene {gene_name}")
                return None

            # 直接CRE的一阶CRE
            adjacent_CREs = []
            for cre_idx in direct_CREs:
                incoming_cres = torch.where(self.CRE_CRE_matrix[:, cre_idx] > 0)[0]
                adjacent_CREs.extend(incoming_cres.tolist())

            adjacent_CREs = list(set(adjacent_CREs) - set(direct_CREs.tolist()))
            adjacent_CREs = torch.tensor(adjacent_CREs)

            if len(adjacent_CREs) == 0:
                print(f"Warning: No adjacent CREs found for TF gene {gene_name}, adding a dummy adjacent CRE")
                adjacent_CREs = torch.tensor([0])
                dummy_adjacent = True
            else:
                dummy_adjacent = False
                print(f"找到与TF {gene_name}相关的CRE: 直接相关{len(direct_CREs)}个, 上游邻接{len(adjacent_CREs)}个")

            # CRE连接TF
            related_TFs = []

            if not dummy_adjacent:
                for cre_idx in adjacent_CREs:
                    tfs = torch.where(self.TF_CRE_matrix[:, cre_idx] > 0)[0]
                    related_TFs.extend(tfs.tolist())

            for cre_idx in direct_CREs:
                tfs = torch.where(self.TF_CRE_matrix[:, cre_idx] > 0)[0]
                related_TFs.extend(tfs.tolist())

            # # 确保目标TF在列表中
            # if int(TF_idx) not in related_TFs:
            #     related_TFs.append(int(TF_idx))

            related_TFs = torch.tensor(list(set(related_TFs)))

            if len(related_TFs) == 0:
                print(f"Warning: No TFs found related to CREs for TF gene {gene_name}")
                return None

            n_TFs = len(related_TFs)
            n_adjacent = len(adjacent_CREs)
            n_direct = len(direct_CREs)
            n_all_CREs = n_adjacent + n_direct

            # TF-TF转换矩阵 - 添加对角线自环
            TF_TF_subnetwork = self.TF_TF_matrix[related_TFs][:, related_TFs].clone()

            for i in range(n_TFs):
                if TF_TF_subnetwork[i, i] == 0:
                    TF_TF_subnetwork[i, i] = 1.0

            # CRE-CRE转换矩阵 (n_adjacent + n_direct) × n_direct
            CRE_CRE_subnetwork = torch.zeros((n_all_CREs, n_direct))

            for i, adj_cre in enumerate(adjacent_CREs):
                for j, dir_cre in enumerate(direct_CREs):
                    if dummy_adjacent:
                        CRE_CRE_subnetwork[i, j] = 0
                    else:
                        CRE_CRE_subnetwork[i, j] = self.CRE_CRE_matrix[adj_cre, dir_cre]

            for i, dir_cre_i in enumerate(direct_CREs):
                for j, dir_cre_j in enumerate(direct_CREs):
                    # 对角线设置为1（自环）
                    if i == j:
                        CRE_CRE_subnetwork[n_adjacent + i, j] = 1.0
                    else:
                        CRE_CRE_subnetwork[n_adjacent + i, j] = self.CRE_CRE_matrix[dir_cre_i, dir_cre_j]

            # 构建扩展的TF-CRE转换矩阵 (n_TFs + n_all_CREs) × n_all_CREs
            TF_CRE_extended = torch.zeros((n_TFs + n_all_CREs, n_all_CREs))

            for i, tf_idx in enumerate(related_TFs):

                if not dummy_adjacent:
                    for j, cre_idx in enumerate(adjacent_CREs):
                        TF_CRE_extended[i, j] = self.TF_CRE_matrix[tf_idx, cre_idx]

                for j, cre_idx in enumerate(direct_CREs):
                    TF_CRE_extended[i, n_adjacent + j] = self.TF_CRE_matrix[tf_idx, cre_idx]

            for i in range(n_all_CREs):
                TF_CRE_extended[n_TFs + i, i] = 1.0

            # CRE-Target连接矩阵 (n_direct × 1)
            CRE_TF_subnetwork = torch.zeros(n_direct, 1)
            for i, cre_idx in enumerate(direct_CREs):
                CRE_TF_subnetwork[i, 0] = self.CRE_TF_matrix[cre_idx, TF_idx]

            TF_TF_mask = (TF_TF_subnetwork > 0).float()
            CRE_CRE_mask = (CRE_CRE_subnetwork > 0).float()
            TF_CRE_mask = (TF_CRE_extended > 0).float()
            CRE_TF_mask = (CRE_TF_subnetwork > 0).float()

            # 10. 构建全CRE列表和类型标记
            if dummy_adjacent:
                all_CREs = direct_CREs
            else:
                all_CREs = torch.cat([adjacent_CREs, direct_CREs])

            cre_types = torch.zeros(n_all_CREs)
            cre_types[n_adjacent:] = 1

            return {
                'gene_name': gene_name,
                'gene_idx': TF_idx,
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

                # 网络连接
                'TF_TF_network': TF_TF_subnetwork,
                'CRE_CRE_network': CRE_CRE_subnetwork,
                'TF_CRE_network': TF_CRE_extended,
                'CRE_Target_network': CRE_TF_subnetwork,

                # 网络掩码
                'TF_TF_mask': TF_TF_mask,
                'CRE_CRE_mask': CRE_CRE_mask,
                'TF_CRE_mask': TF_CRE_mask,
                'CRE_Target_mask': CRE_TF_mask  # 复用变量名
            }
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
        """梯度钩子：确保只更新掩码允许的连接"""
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
    基因特异性网络模型
    """

    def __init__(self, gene_network):
        super(GeneSpecificModel, self).__init__()
        self.gene_network = gene_network

        self.n_TFs = gene_network['n_TFs']
        self.n_all_CREs = gene_network['n_all_CREs']
        self.n_direct = gene_network['n_direct']

        # TF-TF层 (n_TFs × n_TFs)
        self.TF_TF_layer = BioStreamLinear(
            self.n_TFs,
            self.n_TFs,
            gene_network['TF_TF_mask']
        )

        # TF+CRE-CRE层 ((n_TFs + n_all_CREs) × n_all_CREs)
        self.TF_CRE_layer = BioStreamLinear(
            self.n_TFs + self.n_all_CREs,
            self.n_all_CREs,
            gene_network['TF_CRE_mask']
        )

        # CRE-CRE层 (n_all_CREs × n_direct)
        self.CRE_CRE_layer = BioStreamLinear(
            self.n_all_CREs,
            self.n_direct,
            gene_network['CRE_CRE_mask']
        )

        # CRE-Target(或TF)层 (n_direct × 1)
        self.CRE_Target_layer = BioStreamLinear(
            self.n_direct,
            1,
            gene_network['CRE_Target_mask']
        )

    def forward(self, tf_expr, cre_expr=None):
        batch_size = tf_expr.shape[0]

        # TF-TF传播
        TF_exp_emb = F.relu(self.TF_TF_layer(tf_expr))

        if cre_expr is None:
            cre_expr = torch.zeros(batch_size, self.n_all_CREs, device=tf_expr.device)

        # 拼接TF_exp_emb和CRE_exp
        TF_emb_CRE_exp = torch.cat([TF_exp_emb, cre_expr], dim=1)

        # TF+CRE → CRE传播: TF_emb_CRE_exp → CRE_exp_emb
        CRE_exp_emb = F.relu(self.TF_CRE_layer(TF_emb_CRE_exp))

        # CRE_exp_emb → CRE_exp_emb_direct
        CRE_exp_emb_direct = F.relu(self.CRE_CRE_layer(CRE_exp_emb))

        # CRE-Target传播: CRE_exp_emb_direct → Target_predict
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
        切分训练集和测试集
        """
        if isinstance(tf_expr, pd.DataFrame):
            self.cell_ids = tf_expr.columns.tolist()
        else:
            self.cell_ids = None

        # 基因顺序与global_network一致
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
            # 如果已经是numpy数组，假设顺序已经对齐
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

        # 使用train_df划分训练集和测试集
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
        同时训练和评估以找每个基因到最优模型
        """
        if self.tf_expr is None or self.target_expr is None:
            raise ValueError("Data not prepared. Call prepare_data first.")

        if gene_type == 'auto':
            gene_type = 'Target' if gene_name in self.global_network.Target_names else 'TF'

        gene_network = self.global_network.get_gene_specific_network(gene_name, gene_type=gene_type)
        gene_network['gene_type'] = gene_type  # 添加基因类型信息供Dataset使用

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

                # 计算Pearson相关系数
                all_preds = np.array(all_preds).flatten()
                all_targets = np.array(all_targets).flatten()
                correlation, _ = pearsonr(all_preds, all_targets)

                if verbose:
                    print(
                        f"Epoch {epoch + 1}/{epochs}, Test Loss: {test_loss / len(test_loader):.6f}, Correlation: {correlation:.6f}")
                model.train()

        # 最终评估
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

        # 计算评估指标
        all_preds = np.array(all_preds).flatten()
        all_targets = np.array(all_targets).flatten()
        correlation, p_value = pearsonr(all_preds, all_targets)

        test_loss = np.mean((all_preds - all_targets) ** 2)
        print(f"Final Test Loss: {test_loss:.6f}, Correlation: {correlation:.6f}, p-value: {p_value:.6f}")

        # 保存模型
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            model_path = os.path.join(save_dir, f"{gene_name}_model.pth")
            torch.save(model.state_dict(), model_path)
            print(f"Model saved to {model_path}")

        # 构建评估结果
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

        # 如果未提供表达数据，使用测试集
        if tf_expr is None:
            # 使用已有的测试集
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

    def load_model(self, gene_name, model_path):
        gene_network = self.global_network.get_gene_specific_network(gene_name, gene_type='target')

        if gene_network is None:
            raise ValueError(f"No network found for gene {gene_name}")

        model = GeneSpecificModel(gene_network)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model = model.to(self.device)

        return model

    def analyze_weights(self, model, gene_name):
        model.eval()
        gene_network = model.gene_network

        tf_tf_weights = model.tf_tf_layer.weight.detach().cpu().numpy()
        cre_cre_weights = model.cre_cre_layer.weight.detach().cpu().numpy()
        tf_cre_weights = model.tf_cre_layer.weight.detach().cpu().numpy()

        if hasattr(model, 'cre_target_layer'):
            output_weights = model.cre_target_layer.weight.detach().cpu().numpy()
        else:
            output_weights = model.cre_tf_layer.weight.detach().cpu().numpy()

        tf_influence = np.zeros(len(gene_network['related_tfs']))
        for i in range(len(gene_network['related_tfs'])):
            tf_tf_influence = np.abs(tf_tf_weights[i, :]).sum()

            tf_cre_influence = np.abs(tf_cre_weights[i, :]).sum()

            tf_influence[i] = tf_tf_influence + tf_cre_influence

        cre_influence = np.zeros(len(gene_network['related_cres']))
        for i in range(len(gene_network['related_cres'])):
            cre_cre_influence = np.abs(cre_cre_weights[i, :]).sum()

            cre_output_influence = np.abs(output_weights[i, 0]) if i < output_weights.shape[0] else 0

            cre_influence[i] = cre_cre_influence + cre_output_influence

        tf_indices = gene_network['related_tfs'].cpu().numpy()
        tf_names = [self.global_network.tf_names[idx] if idx < len(self.global_network.tf_names) else f"Unknown_{idx}"
                    for idx in tf_indices]

        cre_indices = gene_network['related_cres'].cpu().numpy()
        cre_names = [
            self.global_network.cre_names[idx] if idx < len(self.global_network.cre_names) else f"Unknown_{idx}"
            for idx in cre_indices]

        # 创建结果数据框
        tf_df = pd.DataFrame({
            'TF_Index': tf_indices,
            'TF_Name': tf_names,
            'Influence': tf_influence
        }).sort_values('Influence', ascending=False)

        cre_df = pd.DataFrame({
            'CRE_Index': cre_indices,
            'CRE_Name': cre_names,'Influence': cre_influence
        }).sort_values('Influence', ascending=False)

        # 返回分析结果
        return {
            'gene_name': gene_name,
            'tf_influence': tf_df,
            'cre_influence': cre_df
        }