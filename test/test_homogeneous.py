import torch
import torch.nn.functional as F
import torch.optim as optim
import logging
from torch_geometric.nn import GATConv
from torch_geometric.utils import negative_sampling
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
import pandas as pd
import os
from scipy.stats import pearsonr
import numpy as np

from dataset.bio_dataset import BioDataset
from utils import *


class HomogeneousGATMultiTask(torch.nn.Module):
    """
    基于GAT的SMOGT
    """

    def __init__(self, in_channels, hidden_channels, out_channels, expression_out_channels,
                 heads=4, dropout=0.2, layer_nums=2):
        super().__init__()

        self.layer_nums = layer_nums
        self.dropout = dropout
        self.heads = heads

        # GAT层列表
        self.gat_layers = torch.nn.ModuleList()

        # 根据原异构逻辑构建GAT层
        for i in range(layer_nums):
            if layer_nums == 1:
                # 单层情况
                in_dim = in_channels
                out_dim = hidden_channels
                self.gat_layers.append(
                    GATConv(in_dim, out_dim, heads=1, concat=False, dropout=dropout)
                )
            else:
                # 多层情况
                if i == 0:
                    # 第0层：raw_dim -> hidden_dim * 2
                    in_dim = in_channels
                    out_dim = hidden_channels * 2
                    self.gat_layers.append(
                        GATConv(in_dim, out_dim, heads=heads, concat=False, dropout=dropout)
                    )
                elif i == 1:
                    # 第1层：hidden_dim * 2 -> hidden_dim
                    in_dim = hidden_channels * 2
                    out_dim = hidden_channels
                    self.gat_layers.append(
                        GATConv(in_dim, out_dim, heads=heads, concat=False, dropout=dropout)
                    )
                else:
                    # 第2层及以后：hidden_dim -> hidden_dim
                    in_dim = hidden_channels
                    out_dim = hidden_channels
                    self.gat_layers.append(
                        GATConv(in_dim, out_dim, heads=heads, concat=False, dropout=dropout)
                    )

        # 最终线性层：hidden_dim -> embedding_dim（与原异构逻辑一致）
        self.lin = torch.nn.Linear(hidden_channels, out_channels)

        # 基因表达重构解码器（与原异构逻辑结构一致）
        self.expression_decoder = torch.nn.Sequential(
            torch.nn.Linear(out_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, hidden_channels * 2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels * 2, hidden_channels * 3),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels * 3, expression_out_channels)
        )

    def encode(self, x, edge_index):
        """编码器：将节点特征编码为嵌入，遵循原异构逻辑的层数处理"""
        current_x = x

        # 应用GAT层
        for i, gat_layer in enumerate(self.gat_layers):
            current_x = F.dropout(current_x, p=self.dropout, training=self.training)
            current_x = gat_layer(current_x, edge_index)
            current_x = F.relu(current_x)

            # 在第1层后应用dropout（与原异构逻辑一致）
            if i == 1:
                current_x = F.dropout(current_x, p=self.dropout, training=self.training)

        # 最终线性变换
        z = self.lin(current_x)
        return z

    def decode_link(self, z, edge_label_index):
        """链接预测解码器：使用内积预测边的存在概率"""
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

    def decode_expression(self, z):
        """基因表达重构解码器：重构基因表达值"""
        return self.expression_decoder(z)


def evaluate_link_prediction(model, data, device):
    model.eval()
    with torch.no_grad():
        z = model.encode(data.x.to(device), data.train_edge_index.to(device))
        pos_edge_index = data.test_edge_index.to(device)
        neg_edge_index = negative_sampling(
            edge_index=pos_edge_index, num_nodes=data.num_nodes,
            num_neg_samples=pos_edge_index.size(1)).to(device)
        edge_label_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
        edge_label = torch.cat([torch.ones(pos_edge_index.size(1)), torch.zeros(neg_edge_index.size(1))], dim=0)
        out = model.decode_link(z, edge_label_index).sigmoid()
        preds, labels = out.cpu().numpy(), edge_label.cpu().numpy()
        return roc_auc_score(labels, preds), average_precision_score(labels, preds)

def evaluate_expression_recon(model, data, device):
    model.eval()
    with torch.no_grad():
        z = model.encode(data.x.to(device), data.train_edge_index.to(device))
        eval_mask = data.regression_node_mask.to(device) & data.expression_test_mask.to(device)
        if eval_mask.sum() == 0: return 0.0
        pred_y = model.decode_expression(z)[eval_mask]
        # 注意：这里也需要与训练保持一致，使用data.x作为真实值
        true_y = data.x.to(device)[eval_mask]
        pred_y_flat, true_y_flat = pred_y.cpu().numpy().flatten(), true_y.cpu().numpy().flatten()
        if len(pred_y_flat) < 2: return 0.0
        corr, _ = pearsonr(pred_y_flat, true_y_flat)
        return corr


def evaluate_tf_cre_metrics_homogeneous(model, data, device, tf_eval_data):
    """
    同构版本的 TF-CRE 评估函数。
    """
    model.eval()
    results = {}
    tf_aupr_values = []
    tf_f1_values = []

    with torch.no_grad():
        z = model.encode(data.x.to(device), data.train_edge_index.to(device))

        for tf_global_id, eval_data in tf_eval_data.items():
            edges = eval_data["edges"].to(device)
            labels = eval_data["labels"]
            preds = model.decode_link(z, edges).sigmoid()
            aupr = average_precision_score(labels.cpu().numpy(), preds.cpu().numpy())
            f1 = calculate_max_f1(preds, labels)
            results[f"TF_{tf_global_id}"] = {"aupr": aupr, "f1": f1}
            tf_aupr_values.append(aupr)
            tf_f1_values.append(f1)

    if tf_aupr_values:
        results["average"] = {
            "aupr": np.mean(tf_aupr_values),
            "f1": np.mean(tf_f1_values),
            "num_tfs": len(tf_aupr_values)
        }
    return results


def prepare_homogeneous_tf_cre_evaluation_data(hetero_data, data, offsets, min_positive_edges=100):
    """为同构图评估准备 TF-CRE 数据"""
    tf_eval_data = {}

    # 获取节点类型和偏移量
    node_type_map = {node_type: i for i, node_type in enumerate(hetero_data.node_types)}
    tf_type_idx = node_type_map['TF']
    cre_type_idx = node_type_map['CRE']

    # 找到所有 TF 节点的全局索引
    all_tf_indices = (data.node_type == tf_type_idx).nonzero(as_tuple=True)[0]

    # 获取异构图中的 TF-CRE 边 (局部索引)
    tf_cre_edge_type = ('TF', 'edge', 'CRE')
    if tf_cre_edge_type not in hetero_data.edge_types:
        logging.warning("未在异构图中找到 TF-CRE 边类型")
        return {}

    pos_edges_local = hetero_data[tf_cre_edge_type].edge_index

    # 将局部索引转换为全局索引
    pos_edges_global = torch.stack([
        pos_edges_local[0] + offsets['TF'],
        pos_edges_local[1] + offsets['CRE']
    ])

    # 按TF对正样本边进行分组
    tf_to_pos_cres = {}
    for i in range(pos_edges_global.size(1)):
        tf_id = pos_edges_global[0, i].item()
        cre_id = pos_edges_global[1, i].item()
        if tf_id not in tf_to_pos_cres:
            tf_to_pos_cres[tf_id] = set()
        tf_to_pos_cres[tf_id].add(cre_id)

    # MODIFIED: 收集所有在正边中出现的CRE节点，而不是所有CRE节点
    all_cre_indices_in_pos_edges = set()
    for cre_set in tf_to_pos_cres.values():
        all_cre_indices_in_pos_edges.update(cre_set)

    # 为每个符合条件的TF创建评估数据
    for tf_id in all_tf_indices:
        tf_id = tf_id.item()
        if tf_id in tf_to_pos_cres and len(tf_to_pos_cres[tf_id]) >= min_positive_edges:
            pos_cres_set = tf_to_pos_cres[tf_id]
            # MODIFIED: 负样本CRE只从在正边中出现的CRE中选择
            neg_cres_set = all_cre_indices_in_pos_edges - pos_cres_set

            if not neg_cres_set:
                continue

            pos_edges = torch.tensor([[tf_id] * len(pos_cres_set), list(pos_cres_set)], dtype=torch.long)
            neg_edges = torch.tensor([[tf_id] * len(neg_cres_set), list(neg_cres_set)], dtype=torch.long)

            all_edges = torch.cat([pos_edges, neg_edges], dim=1)
            labels = torch.cat([torch.ones(pos_edges.size(1)), torch.zeros(neg_edges.size(1))])

            tf_eval_data[tf_id] = {
                "edges": all_edges,
                "labels": labels
            }
    return tf_eval_data


def train_epoch(model, data, optimizer, device, link_loss_weight):
    model.train()
    optimizer.zero_grad()
    z = model.encode(data.x.to(device), data.train_edge_index.to(device))

    # 链接预测损失
    pos_edge_index = data.train_edge_index.to(device)
    neg_edge_index = negative_sampling(
        edge_index=pos_edge_index, num_nodes=data.num_nodes,
        num_neg_samples=pos_edge_index.size(1)).to(device)
    edge_label_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
    edge_label = torch.cat([torch.ones(pos_edge_index.size(1)), torch.zeros(neg_edge_index.size(1))], dim=0).to(device)
    link_out = model.decode_link(z, edge_label_index)
    loss_link = F.binary_cross_entropy_with_logits(link_out, edge_label)

    # 基因表达重构损失
    recon_mask = data.regression_node_mask.to(device) & data.expression_train_mask.to(device)
    loss_expr = 0
    if recon_mask.sum() > 0:
        pred_y = model.decode_expression(z)[recon_mask]
        # MODIFIED: 使用 data.x 作为真实值
        true_y = data.x.to(device)[recon_mask]
        loss_expr = F.mse_loss(pred_y, true_y)

    total_loss = loss_link * link_loss_weight + loss_expr * (1 - link_loss_weight)
    total_loss.backward()
    optimizer.step()

    return total_loss.item(), loss_link.item(), loss_expr.item() if isinstance(loss_expr, torch.Tensor) else loss_expr


config = load_config("../config.yaml")

logging.basicConfig(
    level=logging.INFO if config["log"]["level"] == "INFO" else logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logging.info("--- 开始同构图版本（多任务）训练流程 ---")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"使用设备: {device}")

hetero_dataset = BioDataset(config=config["dataset"])
hetero_data = hetero_dataset.data

# --- START: 新增 - 手动计算偏移量 ---
offsets = {}
current_offset = 0
for node_type in hetero_data.node_types:
    offsets[node_type] = current_offset
    current_offset += hetero_data[node_type].num_nodes
# --- END: 新增 ---

logging.info("将异构图数据转换为同构图...")
data = hetero_dataset.to_homegeneous()

# --- 创建回归节点掩码 ---
data.regression_node_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
for ntype in hetero_data.node_types:
    if hasattr(hetero_data[ntype], 'cluster'):
        local_regression_indices = (hetero_data[ntype].cluster == 1).nonzero(as_tuple=True)[0]
        if len(local_regression_indices) > 0:
            # MODIFIED: 使用正确的 offsets 字典
            offset = offsets[ntype]
            global_indices = local_regression_indices + offset
            data.regression_node_mask[global_indices] = True
logging.info(f"共找到 {data.regression_node_mask.sum().item()} 个节点用于表达重构 (cluster=1)。")

# --- 创建训练/测试掩码 ---
k_fold_splits = split_nodes(hetero_data, k_fold=config['dataset']['k_fold'], seed=config['dataset']['seed'])
fold_splits = {ntype: masks[0] for ntype, masks in k_fold_splits.items()}

expression_train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
expression_test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
for ntype, masks in fold_splits.items():
    # MODIFIED: 使用正确的 offsets 字典
    if ntype in offsets:
        offset = offsets[ntype]
        num_nodes_in_type = len(masks['train_mask'])
        expression_train_mask[offset:offset + num_nodes_in_type] = masks['train_mask']
        expression_test_mask[offset:offset + num_nodes_in_type] = masks['test_mask']
data.expression_train_mask = expression_train_mask
data.expression_test_mask = expression_test_mask

data = split_edges(data, rate=config['dataset'].get('rate', 0.2), seed=config['dataset']['seed'], use_last_col=False)

# --- 准备TF-CRE评估数据 ---
logging.info("正在准备TF-CRE评估数据...")
# MODIFIED: 传递 offsets 字典
tf_eval_data = prepare_homogeneous_tf_cre_evaluation_data(hetero_data, data, offsets)
logging.info(f"已为 {len(tf_eval_data)} 个TFs准备好评估数据。")

# --- 模型初始化 ---
model = HomogeneousGATMultiTask(
    in_channels=data.num_features,
    hidden_channels=config['model']['hidden_dim'],
    out_channels=config['model']['embedding_dim'],
    expression_out_channels=data.num_features,
    heads=config['attention'].get('heads', 4),
    layer_nums=config['model']['num_layers']
).to(device)

optimizer = optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
link_loss_weight = config['model'].get('edge_loss_rate', 0.5)

logging.info(f"模型已初始化:\n{model}")

# --- 训练循环 ---
num_epochs = config["training"]["epochs"]
eval_interval = config.get('tf_metrics', {}).get('interval', 10)

for epoch in range(1, num_epochs + 1):
    loss, loss_l, loss_e = train_epoch(model, data, optimizer, device, link_loss_weight)

    if epoch % eval_interval == 0:
        auc, aupr = evaluate_link_prediction(model, data, device)
        corr = evaluate_expression_recon(model, data, device)
        tf_cre_results = evaluate_tf_cre_metrics_homogeneous(model, data, device, tf_eval_data)
        avg_tf_aupr = tf_cre_results.get("average", {}).get("aupr", 0.0)
        avg_tf_f1 = tf_cre_results.get("average", {}).get("f1", 0.0)

        logging.info(f"Epoch: {epoch:03d}, Loss: {loss:.4f} | Link AUC: {auc:.4f}, Link AUPR: {aupr:.4f} | "
                     f"Expr Recon Corr: {corr:.4f} | TF-CRE AUPR: {avg_tf_aupr:.4f}, TF-CRE F1: {avg_tf_f1:.4f}")

logging.info("--- 同构图版本（多任务）训练流程结束 ---")