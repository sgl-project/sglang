# This file is copied from https://github.com/deepseek-ai/EPLB/blob/main/eplb.py since that one is not a pypi package
from typing import Tuple

import torch


def balanced_packing(
    weight: torch.Tensor, num_packs: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pack n weighted objects to m packs, such that each bin contains exactly n/m objects and the weights of all packs
    are as balanced as possible.

    Parameters:
        weight: [X, n], the weight of each item
        num_packs: number of packs

    Returns:
        pack_index: [X, n], the pack index of each item
        rank_in_pack: [X, n], the rank of the item in the pack
    """
    num_layers, num_groups = weight.shape
    assert num_groups % num_packs == 0
    groups_per_pack = num_groups // num_packs

    if groups_per_pack == 1:
        pack_index = torch.arange(
            weight.size(-1), dtype=torch.int64, device=weight.device
        ).expand(weight.shape)
        rank_in_pack = torch.zeros_like(weight, dtype=torch.int64)
        return pack_index, rank_in_pack

    indices = weight.float().sort(-1, descending=True).indices.cpu()
    pack_index = torch.full_like(weight, fill_value=-1, dtype=torch.int64, device="cpu")
    rank_in_pack = torch.full_like(pack_index, fill_value=-1)
    for i in range(num_layers):
        pack_weights = [0] * num_packs
        pack_items = [0] * num_packs
        for group in indices[i]:
            pack = min(
                (i for i in range(num_packs) if pack_items[i] < groups_per_pack),
                key=pack_weights.__getitem__,
            )
            assert pack_items[pack] < groups_per_pack
            pack_index[i, group] = pack
            rank_in_pack[i, group] = pack_items[pack]
            pack_weights[pack] += weight[i, group]
            pack_items[pack] += 1
    return pack_index, rank_in_pack


def replicate_experts(
    weight: torch.Tensor, num_phy: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Replicate `num_log` experts to `num_phy` replicas, such that the maximum load of all replicas is minimized.

    Parameters:
        weight: [X, num_log]
        num_phy: total number of experts after replication

    Returns:
        phy2log: [X, num_phy], logical expert id of each physical expert
        rank: [X, num_phy], the replica rank
        logcnt: [X, num_log], number of replicas for each logical expert
    """
    n, num_log = weight.shape
    num_redundant = num_phy - num_log
    assert num_redundant >= 0
    device = weight.device
    phy2log = torch.arange(num_phy, dtype=torch.int64, device=device).repeat(n, 1)
    rank = torch.zeros(n, num_phy, dtype=torch.int64, device=device)
    logcnt = torch.ones(n, num_log, dtype=torch.int64, device=device)
    arangen = torch.arange(n, dtype=torch.int64, device=device)
    for i in range(num_log, num_phy):
        redundant_indices = (weight / logcnt).max(dim=-1).indices
        phy2log[:, i] = redundant_indices
        rank[:, i] = logcnt[arangen, redundant_indices]
        logcnt[arangen, redundant_indices] += 1
    return phy2log, rank, logcnt


def replicate_experts_topology_aware(
    weight: torch.Tensor, num_phy: int, num_nodes: int, base_node: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Topology-aware expert replication with cross-node coverage priority.

    Redundant copies are preferentially placed on nodes where the expert
    doesn't yet have a copy, maximizing node coverage to reduce RDMA traffic.
    When no uncovered node has capacity, falls back to the highest-load-per-
    replica heuristic (same as replicate_experts).

    Output phy2log is in **node-contiguous** layout:
      [0, phy_per_node) → node 0,  [phy_per_node, 2*phy_per_node) → node 1, …

    Parameters:
        weight:    [X, num_log], load statistics per expert
        num_phy:   total number of physical expert slots
        num_nodes: number of server nodes
        base_node: [X, num_log], node id of each expert's base (primary) copy

    Returns:
        phy2log: [X, num_phy], logical expert id  (node-contiguous)
        phyrank: [X, num_phy], replica rank
        logcnt:  [X, num_log], total replica count per expert
    """
    n_layers, num_log = weight.shape
    num_redundant = num_phy - num_log
    assert num_redundant >= 0
    assert num_phy % num_nodes == 0
    phy_per_node = num_phy // num_nodes
    device = weight.device

    phy2log = torch.zeros(n_layers, num_phy, dtype=torch.int64, device=device)
    phyrank = torch.zeros(n_layers, num_phy, dtype=torch.int64, device=device)
    logcnt = torch.ones(n_layers, num_log, dtype=torch.int64, device=device)

    for layer_idx in range(n_layers):
        w = weight[layer_idx].float()
        cnt = torch.ones(num_log, dtype=torch.int64, device=device)

        expert_on_node = [[False] * num_nodes for _ in range(num_log)]
        # list of (expert_id, replica_rank)
        node_buckets = [[] for _ in range(num_nodes)]
        node_count = [0] * num_nodes

        # Base assignment: one copy per expert on its primary node
        for e in range(num_log):
            node_id = int(base_node[layer_idx, e].item())
            node_buckets[node_id].append((e, 0))
            node_count[node_id] += 1
            expert_on_node[e][node_id] = True

        # Greedy redundant placement
        for _ in range(num_redundant):
            scores = w / cnt.float()
            order = torch.argsort(scores, descending=True)
            placed = False

            # Priority 1: place on an UNCOVERED node (maximises node coverage)
            for e_t in order:
                e = int(e_t.item())
                uncovered = [
                    node_id
                    for node_id in range(num_nodes)
                    if not expert_on_node[e][node_id]
                    and node_count[node_id] < phy_per_node
                ]
                if uncovered:
                    node_id = min(uncovered, key=lambda x: node_count[x])
                    rk = int(cnt[e].item())
                    node_buckets[node_id].append((e, rk))
                    node_count[node_id] += 1
                    expert_on_node[e][node_id] = True
                    cnt[e] += 1
                    placed = True
                    break

            # Priority 2: no uncovered placement, place on any node with space
            if not placed:
                for e_t in order:
                    e = int(e_t.item())
                    available = [
                        node_id
                        for node_id in range(num_nodes)
                        if node_count[node_id] < phy_per_node
                    ]
                    if available:
                        node_id = min(available, key=lambda x: node_count[x])
                        rk = int(cnt[e].item())
                        node_buckets[node_id].append((e, rk))
                        node_count[node_id] += 1
                        placed = True
                        break

            if not placed:
                break

        # Write node-contiguous layout
        idx = 0
        for node_id in range(num_nodes):
            for e, rk in node_buckets[node_id]:
                phy2log[layer_idx, idx] = e
                phyrank[layer_idx, idx] = rk
                idx += 1
        logcnt[layer_idx] = cnt

    return phy2log, phyrank, logcnt


def rebalance_experts_hierarchical(
    weight: torch.Tensor,
    num_physical_experts: int,
    num_groups: int,
    num_nodes: int,
    num_gpus: int,
):
    """
    Parameters:
        weight: [num_moe_layers, num_logical_experts]
        num_physical_experts: number of physical experts after replication
        num_groups: number of expert groups
        num_nodes: number of server nodes, where the intra-node network (e.g, NVLink) is faster
        num_gpus: number of GPUs, must be a multiple of `num_nodes`

    Returns:
        physical_to_logical_map: [num_moe_layers, num_physical_experts]
        logical_to_physical_map: [num_moe_layers, num_logical_experts, X]
        logical_count: [num_moe_layers, num_logical_experts]
    """
    num_layers, num_logical_experts = weight.shape
    assert num_logical_experts % num_groups == 0
    group_size = num_logical_experts // num_groups
    assert num_groups % num_nodes == 0
    groups_per_node = num_groups // num_nodes
    assert num_gpus % num_nodes == 0
    assert num_physical_experts % num_gpus == 0
    phy_experts_per_gpu = num_physical_experts // num_gpus

    def inverse(perm: torch.Tensor) -> torch.Tensor:
        inv = torch.empty_like(perm)
        inv.scatter_(
            1,
            perm,
            torch.arange(perm.size(1), dtype=torch.int64, device=perm.device).expand(
                perm.shape
            ),
        )
        return inv

    # Step 1: pack groups to nodes
    tokens_per_group = weight.unflatten(-1, (num_groups, group_size)).sum(-1)
    group_pack_index, group_rank_in_pack = balanced_packing(tokens_per_group, num_nodes)
    log2mlog = (
        (
            (group_pack_index * groups_per_node + group_rank_in_pack) * group_size
        ).unsqueeze(-1)
        + torch.arange(group_size, dtype=torch.int64, device=group_pack_index.device)
    ).flatten(-2)
    mlog2log = inverse(log2mlog)

    # Step 2: construct redundant experts within nodes
    # [num_layers * num_nodes, num_logical_experts // num_nodes]
    tokens_per_mlog = weight.gather(-1, mlog2log).view(
        -1, num_logical_experts // num_nodes
    )
    phy2mlog, phyrank, mlogcnt = replicate_experts(
        tokens_per_mlog, num_physical_experts // num_nodes
    )

    # Step 3: pack physical_experts to GPUs
    # [num_layers * num_nodes, num_physical_experts // num_nodes]
    tokens_per_phy = (tokens_per_mlog / mlogcnt).gather(-1, phy2mlog)
    pack_index, rank_in_pack = balanced_packing(tokens_per_phy, num_gpus // num_nodes)
    phy2pphy = pack_index * phy_experts_per_gpu + rank_in_pack
    pphy2phy = inverse(phy2pphy)

    pphy2mlog = phy2mlog.gather(
        -1, pphy2phy
    )  # [num_layers * num_nodes, num_log_per_nodes]
    pphy2mlog = (
        pphy2mlog.view(num_layers, num_nodes, -1)
        + torch.arange(
            0,
            num_logical_experts,
            num_logical_experts // num_nodes,
            device=group_pack_index.device,
        ).view(1, -1, 1)
    ).flatten(-2)
    pphy2log = mlog2log.gather(-1, pphy2mlog)
    pphyrank = phyrank.gather(-1, pphy2phy).view(num_layers, -1)
    logcnt = mlogcnt.view(num_layers, -1).gather(-1, log2mlog)
    return pphy2log, pphyrank, logcnt


def rebalance_experts_hierarchical_topology_aware(
    weight: torch.Tensor,
    num_physical_experts: int,
    num_groups: int,
    num_nodes: int,
    num_gpus: int,
):
    """
    Topology-aware hierarchical expert rebalancing.

    Differs from ``rebalance_experts_hierarchical`` only in Step 2:
    redundant copies are placed on *different* nodes to maximise node
    coverage, instead of being confined within the home node.  This
    reduces inter-node (RDMA) traffic by increasing NVLink-local
    expert availability.

    Parameters / Returns: same as ``rebalance_experts_hierarchical``.
    """
    num_layers, num_logical_experts = weight.shape
    assert num_logical_experts % num_groups == 0
    group_size = num_logical_experts // num_groups
    assert num_groups % num_nodes == 0
    groups_per_node = num_groups // num_nodes
    assert num_gpus % num_nodes == 0
    assert num_physical_experts % num_gpus == 0
    phy_experts_per_gpu = num_physical_experts // num_gpus
    phy_per_node = num_physical_experts // num_nodes
    log_per_node = num_logical_experts // num_nodes

    def inverse(perm: torch.Tensor) -> torch.Tensor:
        inv = torch.empty_like(perm)
        inv.scatter_(
            1,
            perm,
            torch.arange(perm.size(1), dtype=torch.int64, device=perm.device).expand(
                perm.shape
            ),
        )
        return inv

    # ---- Step 1: pack expert groups to nodes (same as original) ----
    tokens_per_group = weight.unflatten(-1, (num_groups, group_size)).sum(-1)
    group_pack_index, group_rank_in_pack = balanced_packing(tokens_per_group, num_nodes)
    log2mlog = (
        (
            (group_pack_index * groups_per_node + group_rank_in_pack) * group_size
        ).unsqueeze(-1)
        + torch.arange(group_size, dtype=torch.int64, device=group_pack_index.device)
    ).flatten(-2)

    # Derive base node for each logical expert
    # In mlog space, node n owns experts [n*log_per_node, (n+1)*log_per_node)
    base_node = log2mlog // log_per_node  # [L, num_logical_experts]

    # ---- Step 2: topology-aware replication (cross-node coverage) ----
    phy2log, phyrank, logcnt = replicate_experts_topology_aware(
        weight, num_physical_experts, num_nodes, base_node
    )

    # ---- Step 3: pack physical experts to GPUs within each node ----
    tokens_per_phy = (weight.float().gather(-1, phy2log)) / logcnt.float().gather(
        -1, phy2log
    )
    tokens_per_phy_nodes = tokens_per_phy.view(-1, phy_per_node)

    pack_index, rank_in_pack = balanced_packing(
        tokens_per_phy_nodes, num_gpus // num_nodes
    )
    phy2pphy = pack_index * phy_experts_per_gpu + rank_in_pack
    pphy2phy = inverse(phy2pphy)

    phy2log_nodes = phy2log.view(-1, phy_per_node)
    phyrank_nodes = phyrank.view(-1, phy_per_node)

    pphy2log = phy2log_nodes.gather(-1, pphy2phy).view(num_layers, -1)
    pphyrank = phyrank_nodes.gather(-1, pphy2phy).view(num_layers, -1)

    return pphy2log, pphyrank, logcnt


def rebalance_experts(
    weight: torch.Tensor,
    num_replicas: int,
    num_groups: int,
    num_nodes: int,
    num_gpus: int,
    enable_hierarchical: bool,
    topology_aware: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Entry point for expert-parallelism load balancer.

    Parameters:
        weight: [layers, num_logical_experts], the load statistics for all logical experts
        num_replicas: number of physical experts, must be a multiple of `num_gpus`
        num_groups: number of expert groups
        num_nodes: number of server nodes, where the intra-node network (e.g, NVLink) is faster
        num_gpus: number of GPUs, must be a multiple of `num_nodes`
        topology_aware: if True, use topology-aware placement that spreads
            redundant copies across different nodes to maximise NVLink-local
            expert availability and reduce inter-node (RDMA) traffic.
            Default False keeps the original intra-node replication behaviour.

    Returns:
        physical_to_logical_map: [layers, num_replicas], the expert index of each replica
        logical_to_physical_map: [layers, num_logical_experts, X], the replica indices for each expert
        expert_count: [layers, num_logical_experts], number of physical replicas for each logical expert
    """

    num_layers, num_logical_experts = weight.shape
    weight = weight.float().cpu()

    if num_groups % num_nodes == 0:
        if topology_aware:
            # use topology-aware hierarchical policy (cross-node replication)
            phy2log, phyrank, logcnt = rebalance_experts_hierarchical_topology_aware(
                weight, num_replicas, num_groups, num_nodes, num_gpus
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
        torch.arange(num_replicas, dtype=torch.int64, device=log2phy.device).expand(
            num_layers, -1
        ),
    )
    return phy2log, log2phy, logcnt


__all__ = ["rebalance_experts"]
