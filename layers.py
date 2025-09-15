#!/usr/bin/env python3
"""
layers.py
---------
Self-Attention Graph Pooling (SAGPool) layer used in MoAGNN.

- Scores nodes via a 1-dim GCNConv: score = GCN(x, edge_index)
- Select top-k (ratio) per-graph using torch_geometric.nn.topk
- Filters edges with filter_adj
- Applies gating with non_linearity(score)

Reference: Your original implementation kept the same behavior.
"""

from typing import Optional, Tuple
import torch
from torch import Tensor
from torch_geometric.nn import GCNConv
from torch_geometric.nn.pool.topk_pool import topk, filter_adj


class SAGPool(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        ratio: float = 0.8,
        Conv=GCNConv,
        non_linearity=torch.tanh
    ) -> None:
        """
        Args:
            in_channels: Input node feature dimension.
            ratio: Keep ratio in (0,1] or integer K for top-k.
            Conv: Scoring GNN constructor, default GCNConv.
            non_linearity: Gating function on kept node scores.
        """
        super().__init__()
        self.in_channels = in_channels
        self.ratio = ratio
        self.score_layer = Conv(in_channels, 1)
        self.non_linearity = non_linearity

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
        batch: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Optional[Tensor], Tensor, Tensor]:
        """
        Args:
            x: Node features [N, F]
            edge_index: COO edges [2, E]
            edge_attr: (optional) edge attributes
            batch: (optional) batch vector [N], graph ids

        Returns:
            x_pool:    pooled node features
            edge_index:filtered edges
            edge_attr: filtered edge attributes
            batch:     pooled batch vector
            perm:      indices of kept nodes in original indexing
        """
        if batch is None:
            # All nodes belong to a single graph
            batch = edge_index.new_zeros(x.size(0))

        # Node scoring via a 1-dim GCN
        score = self.score_layer(x, edge_index).squeeze(-1)  # [N]

        # Select nodes per-graph (ratio∈(0,1] or top K as int)
        perm = topk(score, self.ratio, batch)  # [N_kept]

        # Gate features with non-linearity(score) on kept nodes
        x = x[perm] * self.non_linearity(score[perm]).view(-1, 1)
        batch = batch[perm]

        # Filter edges to the kept node set
        edge_index, edge_attr = filter_adj(
            edge_index, edge_attr, perm, num_nodes=score.size(0)
        )

        return x, edge_index, edge_attr, batch, perm
