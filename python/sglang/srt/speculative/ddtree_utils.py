"""DDTree (Diffusion Draft Tree) tree-construction utilities.

DDTree extends DFlash speculative decoding: instead of verifying a single
linear chain of draft tokens, it builds a *tree* over the draft model's
per-position distributions (best-first beam search) and verifies the whole
tree in one target forward pass, following the longest accepted path. This
recovers branching information the diffusion drafter produces but a linear
chain discards.

This module builds the tree and converts it to the layout the (reused) EAGLE
tree-verify kernel + tree-aware Mamba scan expect:
- build_ddtree_tree / _build_single_tree: beam-search node selection, then a
  BFS / parent-grouped relabel so the flat node order satisfies the backend
  invariants (parent index < child index, depth-major, siblings contiguous).
- build_eagle_tree_format: parents -> EAGLE LCRS encoding
  (retrive_next_token / retrive_next_sibling / retrive_index).
- compile_ddtree_tree / follow_verified_tree / compact_ddtree_kv_cache:
  legacy helpers from the upstream PR (the active full-tree path uses the
  EAGLE-format conversion above, not these).
"""

import heapq
import math
from typing import Dict, List, Tuple

import numpy as np
import torch


# Dynamic early-stop margin for tree building.
# When the heap top's log-prob falls below (root_log_prob - log(budget) - margin),
# we stop expanding the tree — remaining candidates are too unlikely to contribute.
_DDTREE_EARLY_STOP_MARGIN: float = 1.0

# Depth bonus: added to child log-probs to encourage deeper tree exploration
# over clustering siblings at shallow depths.  Kept small (0.2) to allow
# some sibling exploration when budget exceeds draft depth.
_DDTREE_DEPTH_BONUS: float = 0.2


def build_ddtree_tree(
    draft_logits: torch.Tensor,  # [bs, L, vocab_size]
    tree_budget: int,            # 节点预算 B
    device: torch.device,
    _out_node_token_ids: torch.Tensor | None = None,
    _out_node_depths: torch.Tensor | None = None,
    _out_parents: torch.Tensor | None = None,
    _out_visibility: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[Dict[int, Dict[int, int]]], torch.Tensor]:
    bs, L, V = draft_logits.shape
    max_nodes = tree_budget + 1

    logits = draft_logits.float()

    # --- Spine fast path: when budget equals L, the beam search only ever
    # picks spine nodes (one per depth).  No siblings can be chosen because
    # the spine consumes the entire budget.  This holds for any depth_bonus >= 0
    # since child_push always happens before sibling_push in the loop.
    if tree_budget == L:
        _, top_token_ids = torch.topk(logits, k=1, dim=-1)  # [bs, L, 1]
        top_token_ids = top_token_ids.squeeze(-1)  # [bs, L]
        spine_nodes = tree_budget  # all non-root nodes

        if _out_node_token_ids is None or _out_node_token_ids.shape[0] < bs:
            padded_node_token_ids = torch.zeros(bs, tree_budget, dtype=torch.long, device=device)
        else:
            padded_node_token_ids = _out_node_token_ids[:bs]
            padded_node_token_ids.zero_()
        padded_node_token_ids[:, :spine_nodes] = top_token_ids[:, :spine_nodes]

        if _out_node_depths is None or _out_node_depths.shape[0] < bs:
            padded_node_depths = torch.zeros(bs, tree_budget, dtype=torch.long, device=device)
        else:
            padded_node_depths = _out_node_depths[:bs]
            padded_node_depths.zero_()
        padded_node_depths[:, :spine_nodes] = torch.arange(
            1, spine_nodes + 1, device=device
        ).unsqueeze(0)

        if _out_parents is None or _out_parents.shape[0] < bs:
            padded_parents = torch.full((bs, max_nodes), -1, dtype=torch.long, device=device)
        else:
            padded_parents = _out_parents[:bs]
            padded_parents.fill_(-1)
        padded_parents[:, 1 : spine_nodes + 1] = torch.arange(
            spine_nodes, device=device
        ).unsqueeze(0)

        # Lower-triangular visibility (ancestral: node i sees nodes 0..i)
        if _out_visibility is None or _out_visibility.shape[0] < bs:
            padded_visibility = torch.zeros(bs, max_nodes, max_nodes, dtype=torch.bool, device=device)
        else:
            padded_visibility = _out_visibility[:bs]
            padded_visibility.zero_()
        actual = spine_nodes + 1
        padded_visibility[:, :actual, :actual] = torch.tril(
            torch.ones(actual, actual, dtype=torch.bool, device=device)
        ).unsqueeze(0)

        actual_tree_sizes_t = torch.full((bs,), actual, dtype=torch.long, device=device)

        # Build per-batch child_maps (chain: node i → {token: i+1})
        top_ids_cpu = top_token_ids[:, :spine_nodes].to(device="cpu", dtype=torch.long)
        all_child_maps = []
        for b in range(bs):
            cm: Dict[int, Dict[int, int]] = {0: {}}
            for i in range(spine_nodes):
                tok = int(top_ids_cpu[b, i])
                cm[i][tok] = i + 1
                cm[i + 1] = {}
            all_child_maps.append(cm)

        return (
            padded_node_token_ids,
            padded_node_depths,
            padded_parents,
            all_child_maps,
            padded_visibility,
            actual_tree_sizes_t,
        )

    # --- Full tree path (budget > L): beam search with branching ---
    # topk = max candidates per position (tree width). The rank-probe shows the
    # target token is often top2-8 in the draft distribution (not top1), so a
    # wider tree raises per-layer hit rate. Tunable via SGLANG_DDTREE_MAX_TOPK
    # (default 6). NOTE: effective width is also bounded by budget//L+1, so
    # raising MAX_TOPK only helps when ddtree_budget is large enough.
    from sglang.srt.environ import envs

    _max_topk = int(envs.SGLANG_DDTREE_MAX_TOPK.get())
    topk = min(max(tree_budget // max(L, 1) + 1, 2), V, _max_topk)
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

    # Reuse or allocate padded output buffers.
    if _out_node_token_ids is None or _out_node_token_ids.shape[0] < bs:
        padded_node_token_ids = torch.zeros(bs, tree_budget, dtype=torch.long, device=device)
    else:
        padded_node_token_ids = _out_node_token_ids[:bs]
        padded_node_token_ids.zero_()

    if _out_node_depths is None or _out_node_depths.shape[0] < bs:
        padded_node_depths = torch.zeros(bs, tree_budget, dtype=torch.long, device=device)
    else:
        padded_node_depths = _out_node_depths[:bs]
        padded_node_depths.zero_()

    if _out_parents is None or _out_parents.shape[0] < bs:
        padded_parents = torch.full((bs, max_nodes), -1, dtype=torch.long, device=device)
    else:
        padded_parents = _out_parents[:bs]
        padded_parents.fill_(-1)

    if _out_visibility is None or _out_visibility.shape[0] < bs:
        padded_visibility = torch.zeros(bs, max_nodes, max_nodes, dtype=torch.bool, device=device)
    else:
        padded_visibility = _out_visibility[:bs]
        padded_visibility.zero_()

    for b in range(bs):
        n = actual_sizes[b] - 1
        if n > 0:
            padded_node_token_ids[b, :n] = torch.from_numpy(all_node_token_ids[b]).to(device)
            padded_node_depths[b, :n] = torch.from_numpy(all_node_depths[b]).to(device)
        padded_parents[b, :actual_sizes[b]] = torch.from_numpy(all_parents[b]).to(device)
        vis = all_visibility[b]
        padded_visibility[b, :actual_sizes[b], :actual_sizes[b]] = (
            torch.from_numpy(vis).to(device=device, dtype=torch.bool)
        )

    actual_tree_sizes_t = torch.tensor(actual_sizes, dtype=torch.long, device=device)
    return (
        padded_node_token_ids,
        padded_node_depths,
        padded_parents,
        all_child_maps,
        padded_visibility,
        actual_tree_sizes_t,
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

    # Depth bonus: higher → tree grows deeper; lower/negative → wider. Tunable
    # via SGLANG_DDTREE_DEPTH_BONUS to trade depth vs width empirically.
    from sglang.srt.environ import envs

    depth_bonus = float(envs.SGLANG_DDTREE_DEPTH_BONUS.get())

    node_token_ids = np.zeros(budget, dtype=np.int64)
    node_depths = np.zeros(budget, dtype=np.int32)
    parents = np.full(budget + 1, -1, dtype=np.int32)
    child_maps: Dict[int, Dict[int, int]] = {0: {}}

    first_logw = float(log_probs_np[0, 0])
    heap = [(-first_logw, (0,), 0, 1, 0, first_logw)]

    # Dynamic early-stop: skip candidates whose log-prob is more than
    # log(budget) + margin below the root log-prob.
    _stop_threshold = first_logw - math.log(max(budget, 1)) - _DDTREE_EARLY_STOP_MARGIN

    node_count = 0
    while heap and node_count < budget:
        _, ranks, parent_idx, depth, rank, logw = heapq.heappop(heap)

        # Early stop applies to BRANCHES (siblings) only, never to the main
        # top-1 spine. The spine is the chain where every level picked rank 0;
        # it must always be allowed to grow to depth_limit, matching DFLASH /
        # EAGLE which verify the full top-1 chain unconditionally. Pruning the
        # spine on cumulative log-prob (as a plain `break` did) decapitates the
        # chain after a few levels on weak drafts (low per-step top-1 prob),
        # collapsing AL below spine. Use `continue` (not `break`) so a low-prob
        # branch is skipped without killing spine nodes still queued behind it.
        is_spine = all(r == 0 for r in ranks)
        if (not is_spine) and logw < _stop_threshold:
            continue

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
            child_logw = logw + float(log_probs_np[depth, 0]) + depth_bonus
            child_ranks = ranks + (0,)
            heapq.heappush(
                heap,
                (-child_logw, child_ranks, current_idx, depth + 1, 0, child_logw),
            )

    current_length = node_count + 1

    # --- Reorder nodes into BFS / level order (non-decreasing depth) ---
    # The heap pops nodes by probability, so the flat node order is NOT
    # level-ordered (depths jump around). EAGLE guarantees BFS order
    # (eagle_utils.py: `torch.sort(top_scores_index)`), and the tree-aware
    # attention + Mamba scan rely on it: each node must appear after its parent
    # AND nodes must be grouped by depth so the flat sequence is a valid
    # processing order. Without this, branching trees corrupt the target KV /
    # recurrent state (a pure chain is already monotonic so it was unaffected).
    # node indices 1..node_count are the drafted nodes (0 is the root).
    # BFS layer by layer; within each layer order nodes by their parent's NEW
    # index (so siblings are contiguous and grouped by parent), matching EAGLE's
    # sorted candidate layout (depth-major, parent-grouped). A simple (depth,
    # parent_old_idx) sort is insufficient because parent grouping must be by the
    # parent's *new* position; we resolve this by processing layers in order.
    remap = np.full(current_length, -1, dtype=np.int64)
    remap[0] = 0
    max_depth = int(node_depths[:node_count].max()) if node_count > 0 else 0
    next_pos = 1
    for d in range(1, max_depth + 1):
        # nodes (old idx) at this depth
        layer = [i for i in range(1, current_length) if int(node_depths[i - 1]) == d]
        # order by parent's NEW index (parents are shallower -> already remapped),
        # tie-break by old index for determinism.
        layer.sort(key=lambda i: (int(remap[int(parents[i])]), i))
        for old_idx in layer:
            remap[old_idx] = next_pos
            next_pos += 1
    order = [None] * node_count
    for old_idx in range(1, current_length):
        order[int(remap[old_idx]) - 1] = old_idx

    new_node_token_ids = np.zeros(node_count, dtype=np.int64)
    new_node_depths = np.zeros(node_count, dtype=np.int32)
    new_parents = np.full(current_length, -1, dtype=np.int32)
    for new_pos, old_idx in enumerate(order, start=1):
        new_node_token_ids[new_pos - 1] = node_token_ids[old_idx - 1]
        new_node_depths[new_pos - 1] = node_depths[old_idx - 1]
        new_parents[new_pos] = remap[int(parents[old_idx])]

    # Rebuild child_maps with remapped indices.
    new_child_maps: Dict[int, Dict[int, int]] = {
        int(remap[k]): {} for k in child_maps
    }
    for old_par, kids in child_maps.items():
        np_par = int(remap[old_par])
        for tok, old_child in kids.items():
            new_child_maps[np_par][tok] = int(remap[old_child])

    node_token_ids = new_node_token_ids
    node_depths = new_node_depths
    parents = new_parents
    child_maps = new_child_maps

    # Visibility from the remapped parents (ancestor closure), now in BFS order.
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
    actual_tree_sizes: torch.Tensor,
    dtype: torch.dtype,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    bs = root_token_ids.shape[0]
    max_nodes = tree_budget + 1

    verify_input_ids = torch.zeros(bs, max_nodes, dtype=torch.long, device=device)
    verify_input_ids[:, 0] = root_token_ids
    verify_input_ids[:, 1:] = node_token_ids

    verify_position_ids = torch.zeros(bs, max_nodes, dtype=torch.long, device=device)
    verify_position_ids[:, 0] = start_positions
    verify_position_ids[:, 1:] = start_positions.unsqueeze(1) + node_depths

    max_past = int(past_lengths.max().item())
    total_kv_len = max_past + max_nodes

    # The Triton attention kernel interprets `custom_mask` values as:
    #   0.0 (→ tl.int1: False)  → masked  (Q-K pair SKIPPED)
    #   ≠0.0 (→ tl.int1: True)  → allowed (Q-K pair COMPUTED)
    #
    # We initialise the whole mask to 0.0 (= all masked) and then set allowed
    # positions to 1.0.
    tree_attention_mask = torch.zeros(bs, max_nodes, total_kv_len, dtype=dtype, device=device)

    # --- Prefix portion (vectorized broadcast) ---
    # All actual tree nodes may attend to their full prefix.
    # Build a mask of shape [bs, max_nodes, total_kv_len] where
    # mask[b, q, k] = 1.0 if k < past_lengths[b] and q < actual_tree_sizes[b]
    past_lens_t = past_lengths.to(device=device)  # [bs]
    actual_t = actual_tree_sizes.to(device=device)  # [bs]
    kv_range = torch.arange(total_kv_len, device=device)  # [total_kv_len]
    q_range = torch.arange(max_nodes, device=device)  # [max_nodes]
    # [bs, 1, total_kv_len]: True where k < past_len
    prefix_visible = kv_range.unsqueeze(0).unsqueeze(0) < past_lens_t.unsqueeze(1).unsqueeze(2)
    # [bs, max_nodes, 1]: True where q < actual_size
    actual_query_mask = q_range.unsqueeze(0).unsqueeze(2) < actual_t.unsqueeze(1).unsqueeze(2)
    tree_attention_mask.copy_(prefix_visible.logical_and(actual_query_mask).to(dtype=dtype))

    # --- Tree portion (per-batch, visibility shapes differ) ---
    # Within the tree portion, only ancestor-visible pairs are allowed.
    for b in range(bs):
        past_len_i = int(past_lengths[b].item())
        actual_size = int(actual_tree_sizes[b].item())

        vis = visibility[b, :actual_size, :actual_size]
        tree_attention_mask[b, :actual_size, past_len_i:past_len_i + actual_size] = (
            vis.to(dtype=dtype, device=device)
        )

        # Padding nodes that are not part of the actual tree stay at 0.0
        # (fully masked).

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
    """Compact KV cache by moving kept slots to the front.

    Uses batched index_select + index_copy_ to replace per-element
    set_kv_buffer kernel launches with 2 launches per layer.
    """
    k_buffer, v_buffer = kv_cache_pool.get_kv_buffer(layer.layer_id)
    device = cache_locs.device

    src_list: List[torch.Tensor] = []
    tgt_list: List[torch.Tensor] = []

    for b in range(len(keep_indices)):
        keep = keep_indices[b]
        actual = int(actual_tree_sizes[b].item())

        if len(keep) == actual:
            continue

        # Safety: clamp keep indices to valid range [0, actual).
        keep = [idx for idx in keep if 0 <= idx < actual]
        if not keep or len(keep) == actual:
            continue

        # Fast path: if kept indices are contiguous from 0, no compaction needed.
        if keep == list(range(len(keep))):
            continue

        all_locs = cache_locs[b, :actual]
        keep_t = torch.tensor(keep, dtype=torch.long, device=device)
        keep_locs = all_locs[keep_t]
        tgt_locs = all_locs[: len(keep)]

        mask = keep_locs != tgt_locs
        if mask.any():
            src_list.append(keep_locs[mask])
            tgt_list.append(tgt_locs[mask])

    if not src_list:
        return

    src_idx = torch.cat(src_list)
    tgt_idx = torch.cat(tgt_list)

    k_selected = k_buffer.index_select(0, src_idx)
    v_selected = v_buffer.index_select(0, src_idx)
    k_buffer.index_copy_(0, tgt_idx, k_selected)
    v_buffer.index_copy_(0, tgt_idx, v_selected)


def build_eagle_tree_format(
    parents: torch.Tensor,  # [bs, max_nodes], -1 = no parent / padding
    actual_tree_sizes: torch.Tensor,  # [bs]
    draft_token_num: int,  # D (= max_nodes), per-req fixed width
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert DDTree's parent encoding to EAGLE's LCRS tree format.

    EAGLE's tree verify kernel walks the tree via a left-child/right-sibling
    (LCRS) encoding over per-request *local* node indices 0..D-1 (node 0 = root):
      retrive_next_token[b, k]   = local index of node k's FIRST child   (-1 if none)
      retrive_next_sibling[b, k] = local index of node k's NEXT sibling  (-1 if none)
      retrive_index[b, k]        = global flat slot of node k = b*D + k   (identity)

    Children of a node are the nodes whose parent == that node, taken in
    ascending index order (matches DDTree's build order, where a child always
    gets a higher index than its parent). Padding nodes (index >= actual size)
    get all -1 and are never reachable from the root.

    Returns (retrive_index, retrive_next_token, retrive_next_sibling), each
    [bs, draft_token_num] int64 on `device`.
    """
    bs = parents.shape[0]
    D = int(draft_token_num)

    parents_cpu = parents.to(device="cpu", dtype=torch.long)
    sizes_cpu = actual_tree_sizes.to(device="cpu", dtype=torch.long)

    next_token = torch.full((bs, D), -1, dtype=torch.long)
    next_sibling = torch.full((bs, D), -1, dtype=torch.long)

    for b in range(bs):
        actual = int(sizes_cpu[b].item())
        # Collect children per parent in ascending index order.
        children: List[List[int]] = [[] for _ in range(actual)]
        for node in range(1, actual):
            p = int(parents_cpu[b, node].item())
            if 0 <= p < actual:
                children[p].append(node)
        for node in range(actual):
            kids = children[node]
            if kids:
                next_token[b, node] = kids[0]
                for i in range(len(kids) - 1):
                    next_sibling[b, kids[i]] = kids[i + 1]

    # retrive_index is the identity map local k -> global flat slot b*D + k.
    base = (torch.arange(bs, dtype=torch.long).unsqueeze(1) * D)
    retrive_index = base + torch.arange(D, dtype=torch.long).unsqueeze(0)

    return (
        retrive_index.to(device),
        next_token.to(device),
        next_sibling.to(device),
    )
