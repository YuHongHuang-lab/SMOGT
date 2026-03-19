import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import MessagePassing, Linear
from torch_geometric.utils import softmax
from typing import Optional, Tuple, Union, Dict
from torch import Tensor
from torch_scatter import scatter
import math


class GraphAttentionLayer(MessagePassing):
    def __init__(self,
                in_channels: Union[int, Tuple[int, int]],
                out_channels: int,
                heads: int = 4,
                dropout: float = 0.1,
                top_k: int = -1,
                top_k_type: str = 'globel',
                scaling=False,
                **kwargs):
        super(GraphAttentionLayer, self).__init__(aggr='add', node_dim=0, **kwargs)

        if isinstance(in_channels, (list, tuple)):
            in_channels = in_channels[0]

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.dropout = dropout
        self.top_k = top_k
        self.top_k_type = top_k_type
        self.dim_per_head = out_channels // heads
        self._alpha = None
        self._edge_index_for_message = None


        self.k_proj = Linear(in_channels, self.dim_per_head * heads, bias=False, weight_initializer='uniform')
        self.q_proj = Linear(in_channels, self.dim_per_head * heads, bias=False, weight_initializer='uniform')
        self.v_proj = Linear(in_channels, self.dim_per_head * heads, bias=False, weight_initializer='uniform')


        self.o_proj = Linear(self.dim_per_head * heads, out_channels, weight_initializer='uniform')


        self.scaling = self.out_channels ** -0.5 if scaling else 1

        self.ffn = FeedForward(out_channels, dropout)

        self.first_norm = nn.LayerNorm(in_channels)
        self.final_norm = nn.LayerNorm(out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        self.k_proj.reset_parameters()
        self.q_proj.reset_parameters()
        self.v_proj.reset_parameters()
        self.o_proj.reset_parameters()

    def _calculate_auxiliary(self, edge_index: Tensor, num_nodes: int) -> Tensor:
        row, col = edge_index
        deg = torch.bincount(row, minlength=num_nodes) + torch.bincount(col, minlength=num_nodes)
        min_deg = deg.min()
        max_deg = deg.max()

        if max_deg > min_deg:
            normalized_deg = (deg - min_deg) / (max_deg - min_deg)
        else:
            normalized_deg = torch.ones_like(deg)

        return normalized_deg

    def topk_attention_local(self, attn_scores: Tensor, edge_index: Tensor, size_i: Optional[int]) -> Tensor:
        row, col = edge_index[0], edge_index[1]
        attn_scores = attn_scores.mean(dim=1)

        mask = torch.zeros_like(attn_scores)
        for node in torch.unique(col):
            print(node)
            node_mask = (col == node)
            node_scores = attn_scores[node_mask]

            if len(node_scores) > self.top_k:
                kth_value = torch.topk(torch.abs(node_scores),
                                       k=min(self.top_k, len(node_scores)))[0][-1]
                node_scores[torch.abs(node_scores) < kth_value] = 0

            mask[node_mask] = node_scores

        return mask.unsqueeze(-1)

    def topk_attention_globel(self, attn_scores: Tensor, edge_index: Tensor, size_i: Optional[int]) -> Tensor:
        row, col = edge_index[0], edge_index[1]
        attn_scores = attn_scores.mean(dim=1)

        if len(attn_scores) > self.top_k:
            abs_score = torch.abs(attn_scores)
            topk_values, topk_indices = torch.topk(abs_score, k=self.top_k)
            mask = torch.zeros_like(attn_scores)
            mask[topk_indices] = 1
            attn_scores = attn_scores * mask

        return attn_scores.unsqueeze(-1)

    def forward(self, x: Union[Tensor, Tuple[Tensor, Tensor]],
                edge_index: Tensor,
                x_auxiliary: Optional[Union[str, Tensor, None]] = None,
                return_attention_weights: bool = False) -> Tensor:

        if isinstance(x, tuple):
            x_src, x_dst = x
        else:
            x_src = x_dst = x

        if isinstance(x_auxiliary, str) and x_auxiliary.lower() == 'auto':
            x_auxiliary = self._calculate_auxiliary(edge_index, x_src.size(0))

        k = self.k_proj(x_src).view(-1, self.heads, self.dim_per_head)
        q = self.q_proj(x_dst).view(-1, self.heads, self.dim_per_head)
        v = self.v_proj(x_src).view(-1, self.heads, self.dim_per_head)

        self._edge_index_for_message = edge_index

        out = self.propagate(edge_index,
                             k=k,
                             q=q,
                             v=v,
                             x_auxiliary=x_auxiliary,
                             size=(x_src.size(0), x_dst.size(0)))

        out = out.view(-1, self.heads * self.dim_per_head)
        out = self.o_proj(out)

        out = self.ffn(out)

        del self._edge_index_for_message

        if return_attention_weights:
            alpha = self._alpha
            self._alpha = None
            return out, (edge_index, alpha)
        else:
            return out

    def message(self,
                k_j: Tensor,
                q_i: Tensor,
                v_j: Tensor,
                x_auxiliary_j: Optional[Tensor],
                index: Tensor,
                size_i: Optional[int] = None) -> Tensor:

        q_i_norm = F.normalize(q_i, p=2, dim=-1)
        k_j_norm = F.normalize(k_j, p=2, dim=-1)

        alpha = torch.abs((q_i_norm * k_j_norm).sum(dim=-1))  # [E, H]
        alpha = alpha * self.scaling

        if x_auxiliary_j is not None:
            alpha = alpha * x_auxiliary_j.view(-1, 1)

        if self.top_k > 0:
            if self.top_k_type == 'local':
                alpha = self.topk_attention_local(alpha, self._edge_index_for_message, size_i)
            else:
                alpha = self.topk_attention_globel(alpha, self._edge_index_for_message, size_i)

        # alpha = softmax(alpha, index, num_nodes=size_i)

        self._alpha = alpha

        # Dropout
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        return v_j * alpha.unsqueeze(-1)

    def __repr__(self):
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')


class FeedForward(nn.Module):
    def __init__(self, d_model: int, dropout: float=0.1):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model*4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model*4, d_model)
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor) -> Tensor:
        return x+self.net(self.norm(x))


class AttentionDiffusion(nn.Module):

    def __init__(self,
                 hidden_dim: int,
                 hop_num: int,
                 dropout: float = 0.1,
                 top_k: int = 10,
                 eps: float = 1e-6):
        super(AttentionDiffusion, self).__init__()

        self.alpha_calculation = AlphaCalculation(hidden_dim, dropout)
        self.hop_num = hop_num
        self.dropout = dropout
        self.top_k = top_k
        self.eps = eps

        # 预分配存储空间用于中间计算
        self.register_buffer('empty_cache', torch.tensor([]))

    def _compute_global_embedding(self, feat_dict: Dict[str, Tensor]) -> Tensor:
        return torch.stack([feat.mean(dim=0, keepdim=True) for feat in feat_dict.values()]).mean(dim=0)

    def _aggregate_messages(self,
                            edge_index: Tensor,
                            src_feat: Tensor,
                            attention: Tensor,
                            num_nodes: int) -> Tensor:
        msg = src_feat * attention.view(-1, 1)
        return scatter(msg,
                       edge_index[1],
                       dim=0,
                       dim_size=num_nodes,
                       reduce='sum')

    def _update_features(self,
                         messages: Tensor,
                         feat_0: Tensor,
                         alpha: float) -> Tensor:
        return feat_0.mul(alpha) + messages.mul(1 - alpha)

    def forward(self,
                x_dict: Dict[str, Tensor],
                attention_scores_dict: Dict[str, Tensor],
                edge_index_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:

        feat_0_dict = {node_type: feat.clone() for node_type, feat in x_dict.items()}
        current_feat_dict = feat_0_dict

        global_embedding = self._compute_global_embedding(current_feat_dict)
        alpha = self.alpha_calculation(global_embedding)
        alpha = alpha.clamp(min=self.eps, max=1 - self.eps)

        for _ in range(self.hop_num):
            new_feat_dict = {}

            for dst_type, dst_feat in current_feat_dict.items():
                accumulated_messages = []

                for edge_type, edge_index in edge_index_dict.items():
                    src_type, edge_name, curr_dst_type = edge_type

                    if curr_dst_type != dst_type or edge_name == 'self_loop':
                        continue

                    attention = attention_scores_dict[str(edge_type)]
                    src_feat = current_feat_dict[src_type][edge_index[0]]

                    msg = self._aggregate_messages(
                        edge_index,
                        src_feat,
                        attention,
                        dst_feat.size(0)
                    )
                    accumulated_messages.append(msg)

                if accumulated_messages:
                    combined_msg = torch.stack(accumulated_messages).mean(dim=0)
                    new_feat = self._update_features(combined_msg, feat_0_dict[dst_type], alpha)
                else:
                    new_feat = feat_0_dict[dst_type]

                new_feat.clamp_(min=self.eps, max=1 / self.eps)

                if self.training:
                    new_feat = F.dropout(new_feat, p=self.dropout, training=True, inplace=True)

                new_feat_dict[dst_type] = new_feat

            current_feat_dict = new_feat_dict

            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()

        return current_feat_dict

    def __repr__(self):
        return (f'{self.__class__.__name__}(hidden_dim={self.hidden_dim}, '
                f'hop_num={self.hop_num}, dropout={self.dropout})')

class AlphaCalculation(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1):
        super(AlphaCalculation, self).__init__()
        self.fc_1 = nn.Linear(dim, dim)
        self.fc_2 = nn.Linear(dim, 1)
        self.dropout = nn.Dropout(p=dropout)
        self.reset_parameters()

    def forward(self, x: Tensor) -> Tensor:
        return torch.sigmoid(self.fc_2(torch.tanh(self.fc_1(x)))).squeeze(-1)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc_1.weight, gain=0.02)
        nn.init.xavier_uniform_(self.fc_2.weight, gain=0.02)
        if self.fc_2.bias is not None:
            nn.init.zeros_(self.fc_2.bias)
