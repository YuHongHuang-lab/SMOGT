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
    SMOGT based on GAT
    """

    def __init__(self, in_channels, hidden_channels, out_channels, expression_out_channels,
                 heads=4, dropout=0.2, layer_nums=2):
        super().__init__()

        self.layer_nums = layer_nums
        self.dropout = dropout
        self.heads = heads

        # GAT layer list
        self.gat_layers = torch.nn.ModuleList()

        # Build GAT layers according to original heterogeneous logic
        for i in range(layer_nums):
            if layer_nums == 1:
                # Single layer case
                in_dim = in_channels
                out_dim = hidden_channels
                self.gat_layers.append(
                    GATConv(in_dim, out_dim, heads=1, concat=False, dropout=dropout)
                )
            else:
                # Multi-layer case
                if i == 0:
                    # Layer 0: raw_dim -> hidden_dim * 2
                    in_dim = in_channels
                    out_dim = hidden_channels * 2
                    self.gat_layers.append(
                        GATConv(in_dim, out_dim, heads=heads, concat=False, dropout=dropout)
                    )
                elif i == 1:
                    # Layer 1: hidden_dim * 2 -> hidden_dim
                    in_dim = hidden_channels * 2
                    out_dim = hidden_channels
                    self.gat_layers.append(
                        GATConv(in_dim, out_dim, heads=heads, concat=False, dropout=dropout)
                    )
                else:
                    # Layer 2 and above: hidden_dim -> hidden_dim
                    in_dim = hidden_channels
                    out_dim = hidden_channels
                    self.gat_layers.append(
                        GATConv(in_dim, out_dim, heads=heads, concat=False, dropout=dropout)
                    )

        # Final linear layer: hidden_dim -> embedding_dim (consistent with original heterogeneous logic)
        self.lin = torch.nn.Linear(hidden_channels, out_channels)

        # Gene expression reconstruction decoder (consistent with original heterogeneous logic structure)
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
        """Encoder: Encode node features into embeddings, following the layer processing logic of the original heterogeneous model"""
        current_x = x

        # Apply GAT layers
        for i, gat_layer in enumerate(self.gat_layers):
            current_x = F.dropout(current_x, p=self.dropout, training=self.training)
            current_x = gat_layer(current_x, edge_index)
            current_x = F.relu(current_x)

            # Apply dropout after layer 1 (consistent with original heterogeneous logic)
            if i == 1:
                current_x = F.dropout(current_x, p=self.dropout, training=self.training)

        # Final linear transformation
        z = self.lin(current_x)
        return z

    def decode_link(self, z, edge_label_index):
        """Link prediction decoder: Predict edge existence probability using inner product"""
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

    def decode_expression(self, z):
        """Gene expression reconstruction decoder: Reconstruct gene expression values"""
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
        # Note: Consistent with training, use data.x as ground truth
        true_y = data.x.to(device)[eval_mask]
        pred_y_flat, true_y_flat = pred_y.cpu().numpy().flatten(), true_y.cpu().numpy().flatten()
        if len(pred_y_flat) < 2: return 0.0
        corr, _ = pearsonr(pred_y_flat, true_y_flat)
        return corr


def evaluate_tf_cre_metrics_homogeneous(model, data, device, tf_eval_data):
    """
    TF-CRE evaluation function for homogeneous graph version.
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
    """Prepare TF-CRE data for homogeneous graph evaluation"""
    tf_eval_data = {}

    # Get node types and offsets
    node_type_map = {node_type: i for i, node_type in enumerate(hetero_data.node_types)}
    tf_type_idx = node_type_map['TF']
    cre_type_idx = node_type_map['CRE']

    # Find global indices of all TF nodes
    all_tf_indices = (data.node_type == tf_type_idx).nonzero(as_tuple=True)[0]

    # Get TF-CRE edges in heterogeneous graph (local indices)
    tf_cre_edge_type = ('TF', 'edge', 'CRE')
    if tf_cre_edge_type not in hetero_data.edge_types:
        logging.warning("TF-CRE edge type not found in heterogeneous graph")
        return {}

    pos_edges_local = hetero_data[tf_cre_edge_type].edge_index

    # Convert local indices to global indices
    pos_edges_global = torch.stack([
        pos_edges_local[0] + offsets['TF'],
        pos_edges_local[1] + offsets['CRE']
    ])

    # Group positive edges by TF
    tf_to_pos_cres = {}
    for i in range(pos_edges_global.size(1)):
        tf_id = pos_edges_global[0, i].item()
        cre_id = pos_edges_global[1, i].item()
        if tf_id not in tf_to_pos_cres:
            tf_to_pos_cres[tf_id] = set()
        tf_to_pos_cres[tf_id].add(cre_id)

    # MODIFIED: Collect all CRE nodes that appear in positive edges instead of all CRE nodes
    all_cre_indices_in_pos_edges = set()
    for cre_set in tf_to_pos_cres.values():
        all_cre_indices_in_pos_edges.update(cre_set)

    # Create evaluation data for each eligible TF
    for tf_id in all_tf_indices:
        tf_id = tf_id.item()
        if tf_id in tf_to_pos_cres and len(tf_to_pos_cres[tf_id]) >= min_positive_edges:
            pos_cres_set = tf_to_pos_cres[tf_id]
            # MODIFIED: Negative sample CREs are only selected from CREs appearing in positive edges
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

    # Link prediction loss
    pos_edge_index = data.train_edge_index.to(device)
    neg_edge_index = negative_sampling(
        edge_index=pos_edge_index, num_nodes=data.num_nodes,
        num_neg_samples=pos_edge_index.size(1)).to(device)
    edge_label_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
    edge_label = torch.cat([torch.ones(pos_edge_index.size(1)), torch.zeros(neg_edge_index.size(1))], dim=0).to(device)
    link_out = model.decode_link(z, edge_label_index)
    loss_link = F.binary_cross_entropy_with_logits(link_out, edge_label)

    # Gene expression reconstruction loss
    recon_mask = data.regression_node_mask.to(device) & data.expression_train_mask.to(device)
    loss_expr = 0
    if recon_mask.sum() > 0:
        pred_y = model.decode_expression(z)[recon_mask]
        # MODIFIED: Use data.x as ground truth
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
logging.info("--- Starting Homogeneous Graph Version (Multi-Task) Training Pipeline ---")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

hetero_dataset = BioDataset(config=config["dataset"])
hetero_data = hetero_dataset.data

# --- START: New - Manually calculate offsets ---
offsets = {}
current_offset = 0
for node_type in hetero_data.node_types:
    offsets[node_type] = current_offset
    current_offset += hetero_data[node_type].num_nodes
# --- END: New ---

logging.info("Converting heterogeneous graph data to homogeneous graph...")
data = hetero_dataset.to_homegeneous()

# --- Create regression node mask ---
data.regression_node_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
for ntype in hetero_data.node_types:
    if hasattr(hetero_data[ntype], 'cluster'):
        local_regression_indices = (hetero_data[ntype].cluster == 1).nonzero(as_tuple=True)[0]
        if len(local_regression_indices) > 0:
            # MODIFIED: Use correct offsets dictionary
            offset = offsets[ntype]
            global_indices = local_regression_indices + offset
            data.regression_node_mask[global_indices] = True
logging.info(f"Found {data.regression_node_mask.sum().item()} nodes for expression reconstruction (cluster=1).")

# --- Create train/test masks ---
k_fold_splits = split_nodes(hetero_data, k_fold=config['dataset']['k_fold'], seed=config['dataset']['seed'])
fold_splits = {ntype: masks[0] for ntype, masks in k_fold_splits.items()}

expression_train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
expression_test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
for ntype, masks in fold_splits.items():
    # MODIFIED: Use correct offsets dictionary
    if ntype in offsets:
        offset = offsets[ntype]
        num_nodes_in_type = len(masks['train_mask'])
        expression_train_mask[offset:offset + num_nodes_in_type] = masks['train_mask']
        expression_test_mask[offset:offset + num_nodes_in_type] = masks['test_mask']
data.expression_train_mask = expression_train_mask
data.expression_test_mask = expression_test_mask

data = split_edges(data, rate=config['dataset'].get('rate', 0.2), seed=config['dataset']['seed'], use_last_col=False)

# --- Prepare TF-CRE evaluation data ---
logging.info("Preparing TF-CRE evaluation data...")
# MODIFIED: Pass offsets dictionary
tf_eval_data = prepare_homogeneous_tf_cre_evaluation_data(hetero_data, data, offsets)
logging.info(f"Prepared evaluation data for {len(tf_eval_data)} TFs.")

# --- Model initialization ---
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

logging.info(f"Model initialized:\n{model}")

# --- Training loop ---
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

logging.info("--- Homogeneous Graph Version (Multi-Task) Training Pipeline Completed ---")
