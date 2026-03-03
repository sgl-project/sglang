# Copyright 2023-2025 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
Utility functions for EPLB metrics and analysis.
"""

from typing import List

import torch


def _rank_loads_from_list(
    act_weights: List[torch.Tensor],
) -> torch.Tensor:
    """Compute per-rank total load from a list of per-rank expert counts.

    Args:
        act_weights: list of R tensors, each [L, P].

    Returns:
        rank_loads: [L, R] total token count per (layer, rank).
    """
    R = len(act_weights)
    L = act_weights[0].shape[0]
    device = act_weights[0].device
    rank_loads = torch.zeros(L, R, device=device, dtype=torch.float32)
    for r in range(R):
        rank_loads[:, r] = act_weights[r].to(torch.float32).sum(dim=1)
    return rank_loads


def inter_rank_imbalance_ratio(
    act_weights: List[torch.Tensor],
) -> float:
    """Inter-rank load imbalance, averaged over layers.

    For each layer: max(rank_load) / mean(rank_load).
    Returns the mean ratio across layers.

    Args:
        act_weights: list of R tensors, each [L, P] (per-rank physical
            expert counts).
    """
    rank_loads = _rank_loads_from_list(act_weights)  # [L, R]
    layer_max = rank_loads.max(dim=1).values  # [L]
    layer_mean = rank_loads.mean(dim=1)  # [L]
    ratio = torch.where(
        layer_mean > 0,
        layer_max / layer_mean,
        torch.ones_like(layer_mean),
    )
    return float(ratio.mean().item())


def inter_node_imbalance_ratio(
    act_weights: List[torch.Tensor],
    ranks_per_node: int,
) -> float:
    """Inter-node load imbalance, averaged over layers.

    For each layer: max(node_load) / mean(node_load).
    Returns the mean ratio across layers.

    Args:
        act_weights: list of R tensors, each [L, P].
        ranks_per_node: number of ranks per node.
    """
    rank_loads = _rank_loads_from_list(act_weights)  # [L, R]
    R = rank_loads.shape[1]
    assert R % ranks_per_node == 0
    N = R // ranks_per_node
    node_loads = rank_loads.view(rank_loads.shape[0], N, ranks_per_node).sum(
        dim=2
    )  # [L, N]
    layer_max = node_loads.max(dim=1).values  # [L]
    layer_mean = node_loads.mean(dim=1)  # [L]
    ratio = torch.where(
        layer_mean > 0,
        layer_max / layer_mean,
        torch.ones_like(layer_mean),
    )
    return float(ratio.mean().item())


def inter_node_traffic_ratio(
    act_weights: List[torch.Tensor],
    ranks_per_node: int,
) -> float:
    """Fraction of tokens that cross node boundaries (RDMA traffic ratio).

    For each layer l:
        ratio_l = inter_node_tokens[l] / total_tokens[l]

    Returns mean_l(ratio_l), with 0 for empty layers.

    Args:
        act_weights: list of R tensors, each [L, P] (per-rank physical
            expert counts, where P = R * experts_per_rank).
        ranks_per_node: number of ranks per node.
    """
    if not act_weights:
        return 0.0
    R = len(act_weights)
    first = act_weights[0]
    L, P = first.shape
    device = first.device
    experts_per_rank = P // R

    p = torch.arange(P, device=device, dtype=torch.long)
    dest_rank_p = p // experts_per_rank
    dest_node_p = dest_rank_p // ranks_per_node

    total_tokens_l = torch.zeros(L, device=device, dtype=torch.float32)
    rdma_tokens_l = torch.zeros(L, device=device, dtype=torch.float32)

    for src_rank in range(R):
        src_node = src_rank // ranks_per_node
        is_rdma_p = (dest_node_p != src_node).to(torch.float32)
        lp = act_weights[src_rank].to(torch.float32)
        total_tokens_l += lp.sum(dim=1)
        rdma_tokens_l += (lp * is_rdma_p).sum(dim=1)

    ratio_l = torch.where(
        total_tokens_l > 0,
        rdma_tokens_l / total_tokens_l.clamp(min=1e-30),
        torch.zeros_like(total_tokens_l),
    )
    return float(ratio_l.mean().item())


__all__ = [
    "inter_rank_imbalance_ratio",
    "inter_node_imbalance_ratio",
    "inter_node_traffic_ratio",
]
