import heapq
from typing import Dict, List, Tuple

import numpy as np
import torch


def build_ddtree_tree(
    draft_logits: torch.Tensor,  # [bs, L, vocab_size]
    tree_budget: int,            # 鑺傜偣棰勭畻 B
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[Dict[int, Dict[int, int]]], torch.Tensor]:
    bs, L, V = draft_logits.shape
    topk = min(tree_budget, V)

    logits = draft_logits.float()
    top_logits, top_token_ids = torch.topk(logits, k=topk, dim=-1)
    log_z = torch.logsumexp(logits, dim=-1, keepdim=True)
    top_log_probs = (top_logits - log_z).to(device="cpu", dtype=torch.float32)
    top_token_ids_cpu = top_token_ids.to(device="cpu", dtype=torch.long)

    all_node_token_ids = []
    all_node_depths = []
    all_parents = []
    all_child_maps = []
    all_visibility = []
    actual_sizes = []

    for b in range(bs):
        node_ids, depths, parents, child_map, vis, actual = _build_single_tree(
            top_log_probs[b], top_token_ids_cpu[b], topk, L, tree_budget
        )
        all_node_token_ids.append(node_ids)
        all_node_depths.append(depths)
        all_parents.append(parents)
        all_child_maps.append(child_map)
        all_visibility.append(vis)
        actual_sizes.append(actual)

    max_nodes = tree_budget + 1
    padded_node_token_ids = torch.zeros(bs, tree_budget, dtype=torch.long, device=device)
    padded_node_depths = torch.zeros(bs, tree_budget, dtype=torch.long, device=device)
    padded_parents = torch.full((bs, max_nodes), -1, dtype=torch.long, device=device)
    padded_visibility = torch.zeros(bs, max_nodes, max_nodes, dtype=torch.bool, device=device)

    for b in range(bs):
        n = actual_sizes[b] - 1
        if n > 0:
            padded_node_token_ids[b, :n] = torch.from_numpy(all_node_token_ids[b]).to(device)
            padded_node_depths[b, :n] = torch.from_numpy(all_node_depths[b]).to(device)
        padded_parents[b, :actual_sizes[b]] = torch.from_numpy(all_parents[b]).to(device)
        vis = all_visibility[b]
        padded_visibility[b, :actual_sizes[b], :actual_sizes[b]] = torch.from_numpy(vis).to(device)

    return (
        padded_node_token_ids,
        padded_node_depths,
        padded_parents,
        all_child_maps,
        padded_visibility,
    )


def _build_single_tree(
    top_log_probs: torch.Tensor,
    top_token_ids: torch.Tensor,
    topk: int,
    depth_limit: int,
    budget: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[int, Dict[int, int]], np.ndarray, int]:
    log_probs_np = top_log_probs.numpy().astype(np.float64)
    token_ids_np = top_token_ids.numpy().astype(np.int64)

    node_token_ids = np.zeros(budget, dtype=np.int64)
    node_depths = np.zeros(budget, dtype=np.int32)
    parents = np.full(budget + 1, -1, dtype=np.int32)
    child_maps: Dict[int, Dict[int, int]] = {0: {}}

    first_logw = float(log_probs_np[0, 0])
    heap = [(-first_logw, (0,), 0, 1, 0, first_logw)]

    node_count = 0
    while heap and node_count < budget:
        _, ranks, parent_idx, depth, rank, logw = heapq.heappop(heap)

        token_id = int(token_ids_np[depth - 1, rank])
        current_idx = node_count + 1

        node_token_ids[node_count] = token_id
        node_depths[node_count] = depth
        parents[current_idx] = parent_idx
        child_maps.setdefault(parent_idx, {})[token_id] = current_idx
        child_maps.setdefault(current_idx, {})
        node_count += 1

        if rank + 1 < topk:
            sibling_logw = (
                logw
                - float(log_probs_np[depth - 1, rank])
                + float(log_probs_np[depth - 1, rank + 1])
            )
            sibling_ranks = ranks[:-1] + (rank + 1,)
            heapq.heappush(
                heap,
                (-sibling_logw, sibling_ranks, parent_idx, depth, rank + 1, sibling_logw),
            )

        if depth < depth_limit:
            child_logw = logw + float(log_probs_np[depth, 0])
            child_ranks = ranks + (0,)
            heapq.heappush(
                heap,
                (-child_logw, child_ranks, current_idx, depth + 1, 0, child_logw),
            )

    current_length = node_count + 1
    visibility = np.zeros((current_length, current_length), dtype=bool)
    visibility[0, 0] = True
    for idx in range(1, current_length):
        p = int(parents[idx])
        visibility[idx, :idx] = visibility[p, :idx]
        visibility[idx, idx] = True

    return (
        node_token_ids[:node_count],
        node_depths[:node_count],
        parents[:current_length],
        child_maps,
        visibility,
        current_length,
    )


def compile_ddtree_tree(
    root_token_ids: torch.Tensor,
    node_token_ids: torch.Tensor,
    node_depths: torch.Tensor,
    visibility: torch.Tensor,
    start_positions: torch.Tensor,
    past_lengths: torch.Tensor,
    tree_budget: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    bs = root_token_ids.shape[0]
    max_nodes = tree_budget + 1

    actual_tree_sizes = torch.zeros(bs, dtype=torch.long, device=device)
    for b in range(bs):
        non_zero = (node_token_ids[b] != 0).sum().item()
        actual_tree_sizes[b] = non_zero + 1

    verify_input_ids = torch.zeros(bs, max_nodes, dtype=torch.long, device=device)
    verify_input_ids[:, 0] = root_token_ids
    verify_input_ids[:, 1:] = node_token_ids

    verify_position_ids = torch.zeros(bs, max_nodes, dtype=torch.long, device=device)
    verify_position_ids[:, 0] = start_positions
    verify_position_ids[:, 1:] = start_positions.unsqueeze(1) + node_depths

    max_past = int(past_lengths.max().item())
    total_kv_len = max_past + max_nodes

    tree_attention_mask = torch.zeros(bs, max_nodes, total_kv_len, dtype=dtype, device=device)

    for b in range(bs):
        past_len = int(past_lengths[b].item())
        actual_size = int(actual_tree_sizes[b].item())

        tree_attention_mask[b, :actual_size, :past_len] = 0.0

        vis = visibility[b, :actual_size, :actual_size]
        neg_inf = torch.finfo(dtype).min
        tree_attention_mask[b, :actual_size, past_len:past_len + actual_size] = torch.where(
            vis,
            torch.tensor(0.0, dtype=dtype, device=device),
            torch.tensor(neg_inf, dtype=dtype, device=device),
        )

        if actual_size < max_nodes:
            tree_attention_mask[b, actual_size:, :] = neg_inf

    return verify_input_ids, verify_position_ids, tree_attention_mask, actual_tree_sizes


def follow_verified_tree(
    child_maps: List[Dict[int, Dict[int, int]]],
    posterior_tokens: torch.Tensor,
) -> Tuple[List[List[int]], torch.Tensor]:
    bs = len(child_maps)
    accepted_indices = []
    next_tokens_list = []

    for b in range(bs):
        posterior = posterior_tokens[b].tolist()
        accepted = [0]
        current_idx = 0
        next_token = posterior[0]

        cmap = child_maps[b]
        while next_token in cmap.get(current_idx, {}):
            current_idx = cmap[current_idx][next_token]
            accepted.append(current_idx)
            next_token = posterior[current_idx]

        accepted_indices.append(accepted)
        next_tokens_list.append(next_token)

    next_tokens = torch.tensor(next_tokens_list, dtype=torch.long, device=posterior_tokens.device)
    return accepted_indices, next_tokens


def compact_ddtree_kv_cache(
    kv_cache_pool,
    layer,
    cache_locs: torch.Tensor,
    keep_indices: List[List[int]],
    past_lengths: torch.Tensor,
    actual_tree_sizes: torch.Tensor,
):
    for b in range(len(keep_indices)):
        keep = keep_indices[b]
        actual = int(actual_tree_sizes[b].item())

        if len(keep) == actual:
            continue

        all_locs = cache_locs[b, :actual]
        keep_locs = all_locs[keep]

        for i, (src_loc, tgt_loc) in enumerate(zip(keep_locs, all_locs[:len(keep)])):
            if src_loc == tgt_loc:
                continue
            k_buffer, v_buffer = kv_cache_pool.get_kv_buffer(layer)
            kv_cache_pool.set_kv_buffer(
                layer,
                tgt_loc.unsqueeze(0),
                k_buffer[src_loc:src_loc + 1],
                v_buffer[src_loc:src_loc + 1],
            )
