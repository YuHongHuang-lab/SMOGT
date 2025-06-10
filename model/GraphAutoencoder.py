import torch
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv, GCNConv
from model.GraphAttention import GraphAttentionLayer

#GAT+regression
class SimplifiedGraphAtten_regression(torch.nn.Module):
    """
    Graph attention network for regression on heterogeneous graphs.
    Predicts expression values for Target genes and selected TFs.
    """

    def __init__(self, metadata,
                 hidden_dim,
                 embedding_dim,
                 edge_order=None,
                 heads=4,
                 dropout=0.2,
                 raw_dim=None,
                 layer_nums=2,
                 top_k_type='globel',
                 top_k=10):
        super(SimplifiedGraphAtten_regression, self).__init__()

        self.gat_conv_dict = torch.nn.ModuleList()
        self.dropout = dropout
        self.heads = heads

        # Default edge order if none provided
        if edge_order is None:
            self.edge_order = [
                ('TF', 'edge', 'TF'),
                ('TF', 'edge', 'CRE'),
                ('CRE', 'edge', 'CRE'),
                ('CRE', 'edge', 'TF'),
                ('CRE', 'edge', 'Target')
            ]
        else:
            self.edge_order = edge_order

        # Initialize attention convolution layers
        for i in range(layer_nums):
            gat_conv_dict = {}

            # Add self loops
            for node_type in ['TF', 'CRE']:
                self_loop_type = (node_type, 'self_loop', node_type)

                if layer_nums==1:
                    in_channels = (raw_dim, raw_dim)
                    out_channels = hidden_dim
                else:
                    if i == 0:
                        in_channels = (raw_dim, raw_dim)
                        out_channels = hidden_dim * 2
                    elif i==1:
                        in_channels = (hidden_dim * 2, hidden_dim * 2)
                        out_channels = hidden_dim
                    else:
                        in_channels = (hidden_dim, hidden_dim)
                        out_channels = hidden_dim

                gat_conv_dict[self_loop_type] = SAGEConv(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    aggr='mean'
                )

            # Add edges in specified order
            for edge_type in self.edge_order:
                if edge_type not in metadata['edge_types']:
                    continue

                src_type, _, dst_type = edge_type

                if layer_nums==1:
                    in_channels = (raw_dim, raw_dim)
                    out_channels = hidden_dim
                else:
                    if i == 0:
                        in_channels = (raw_dim, raw_dim)
                        out_channels = hidden_dim * 2
                    elif i==1:
                        in_channels = (hidden_dim * 2, hidden_dim * 2)
                        out_channels = hidden_dim
                    else:
                        in_channels = (hidden_dim, hidden_dim)
                        out_channels = hidden_dim


                if src_type=='TF' and dst_type=='TF':
                    if isinstance(in_channels, tuple):
                        in_channels = in_channels[0]

                    gat_conv_dict[edge_type] = GCNConv(
                        in_channels=in_channels,
                        out_channels=out_channels
                    )
                else:
                    gat_conv_dict[edge_type] = GraphAttentionLayer(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        heads=heads,
                        dropout=dropout,
                        top_k_type=top_k_type,
                        top_k=top_k
                    )

            self.gat_conv_dict.append(HeteroConv(gat_conv_dict, aggr='sum'))

        # Final layers for regression
        self.lin = torch.nn.Linear(hidden_dim, embedding_dim)

        # Regression head for predicting expression values
        self.regression_head = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim*2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim*2, hidden_dim*3),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim*3, raw_dim)
        )

        self.edge_decoder = EdgePredictionDecoder()

    def forward(self, data):
        device = next(self.parameters()).device
        current_x_dict = {node_type: x.to(device) for node_type, x in data.x_dict.items()}

        for node_type in ['TF', 'CRE']:
            x = current_x_dict[node_type]
            # 如果该节点类型的自环边类型还未在 edge_index_dict 中，则添加
            self_loop_edge_type = (node_type, 'self_loop', node_type)

            if self_loop_edge_type not in data.prior_edge_index_dict:
                # 创建自环边索引，使每个节点都连到自身
                self_loop_index = torch.arange(x.size(0)).unsqueeze(0).repeat(2, 1).to(device)
                data.prior_edge_index_dict[self_loop_edge_type] = self_loop_index

        # Apply GAT layers
        for i, gat_conv in enumerate(self.gat_conv_dict):
            conv_dict = {}

            for edge_type, conv_op in gat_conv.convs.items():
                src_type, type, dst_type = edge_type
                if type!='self_loop':
                    if src_type=='TF' and dst_type=='TF':
                        edge_out = conv_op(
                            current_x_dict[src_type],
                            data.prior_edge_index_dict[edge_type]
                        )
                    else:
                        edge_out = conv_op(
                            (current_x_dict[src_type], current_x_dict[dst_type]),
                            data.prior_edge_index_dict[edge_type],
                            return_attention_weights=False
                        )
                else:
                    edge_out = conv_op(
                        (current_x_dict[src_type], current_x_dict[dst_type]),
                        data.prior_edge_index_dict[edge_type]
                    )

                edge_out = F.relu(edge_out)

                if dst_type not in conv_dict:
                    conv_dict[dst_type] = []
                conv_dict[dst_type].append(edge_out)

            # Combine outputs for each node type
            current_x_dict = {
                node_type: torch.stack(outs).sum(dim=0)
                for node_type, outs in conv_dict.items()
            }

            if i==1:
                current_x_dict = {
                    k: F.dropout(v, self.dropout, training=self.training) for k, v in current_x_dict.items()
                }

        # Get embeddings
        z_dict = {key: self.lin(x) for key, x in current_x_dict.items()}

        # Generate expression predictions for Target genes and selected TFs
        expression_dict = {}
        for node_type in ['Target', 'TF']:
            if node_type in z_dict:
                expression_dict[node_type] = self.regression_head(z_dict[node_type])

        return {
            'z_dict': z_dict,
            'expression_dict': expression_dict
        }


class EdgePredictionDecoder(torch.nn.Module):
    """
    边预测解码器
    基于节点的嵌入向量 z 和填充的 edge_index 来预测边的存在性
    """
    def forward(self, z_dict, edge_index_dict):
        edge_logits_dict = {}
        for edge_type, edge_index in edge_index_dict.items():
            if edge_index.size(1)>0:
                src_type, _, dst_type = edge_type
                edge_logits = (z_dict[src_type][edge_index[0]] * z_dict[dst_type][edge_index[1]]).sum(dim=-1)
                edge_logits_dict[str(edge_type)] = torch.sigmoid(edge_logits)
        return edge_logits_dict


