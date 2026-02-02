from typing import Tuple

import torch

from sglang.srt.eplb.eplb_algorithms.deepseek import rebalance_experts_hierarchical


def rebalance_experts(
    weight: torch.Tensor,
    num_replicas: int,
    num_groups: int,
    num_nodes: int,
    num_gpus: int,
    enable_hierarchical: bool,
    active_ranks: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Entry point for expert-parallelism load balancer.

    Parameters:
        weight: [layers, num_logical_experts], the load statistics for all logical experts
        num_replicas: number of physical experts, must be a multiple of `num_gpus`
        num_groups: number of expert groups
        num_nodes: number of server nodes, where the intra-node network (e.g, NVLink) is faster
        num_gpus: number of GPUs, must be a multiple of `num_nodes`

    Returns:
        physical_to_logical_map: [layers, num_replicas], the expert index of each replica
        logical_to_physical_map: [layers, num_logical_experts, X], the replica indices for each expert
        expert_count: [layers, num_logical_experts], number of physical replicas for each logical expert
    """

    num_layers, num_logical_experts = weight.shape
    weight = weight.float().cpu()
    num_active_ranks = active_ranks.sum().item()
    num_local_experts = num_replicas // num_gpus
    if num_active_ranks < num_gpus:
        # Must fall back to global load-balance policy
        # and fix some params
        phy2log, phyrank, logcnt = rebalance_experts_hierarchical(
            weight,
            num_local_experts * num_active_ranks,
            1,
            1,
            num_active_ranks,
        )
    elif enable_hierarchical:
        # use hierarchical load-balance policy
        phy2log, phyrank, logcnt = rebalance_experts_hierarchical(
            weight, num_replicas, num_groups, num_nodes, num_gpus
        )
    else:
        # use global load-balance policy
        phy2log, phyrank, logcnt = rebalance_experts_hierarchical(
            weight, num_replicas, 1, 1, num_gpus
        )
    maxlogcnt = logcnt.max().item()
    log2phy: torch.Tensor = torch.full(
        (num_layers, num_logical_experts, maxlogcnt),
        -1,
        dtype=torch.int64,
        device=logcnt.device,
    )
    log2phy.view(num_layers, -1).scatter_(
        -1,
        phy2log * maxlogcnt + phyrank,
        torch.arange(
            num_local_experts * num_active_ranks,
            dtype=torch.int64,
            device=log2phy.device,
        ).expand(num_layers, -1),
    )
    if num_active_ranks < num_gpus:
        phy2log_slices = list(
            phy2log.view(num_layers, num_active_ranks, -1).unbind(dim=1)
        )
        active_ranks_list = active_ranks.tolist()
        for idx, active_rank in enumerate(active_ranks_list):
            if not active_rank:
                phy2log_slices.insert(idx, torch.zeros_like(phy2log_slices[0]))
                log2phy = torch.where(
                    log2phy >= idx * num_local_experts,
                    log2phy + num_local_experts,
                    log2phy,
                )
        phy2log = torch.stack(phy2log_slices, dim=1).contiguous().view(num_layers, -1)
    return phy2log, log2phy, logcnt
