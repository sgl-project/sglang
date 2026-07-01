"""Host-side tree construction for DFlash tree drafting (SGLang #29524, JetSpec).

Pure-Python/torch; no GPU dependency and no sglang-internal imports. The draft
head's per-depth top-W tokens and log-probs are pulled to host, the tree is built
here, and the resulting verify buffers + tree-causal mask are consumed by the
existing tree-general verify kernels (``verify_tree_greedy`` /
``tree_speculative_sampling_target_only``).

See ``docs`` plan ``2026-06-27-sglang-29524-jetspec-tree-core``. Algorithms ported
from the reference vLLM integration ``Jet-Flow/vllm-jetspec`` ``dflash_tree.py``.
"""

from __future__ import annotations

import heapq

import msgspec
import torch


def compute_tree_budget(
    block_size: int, tree_width: int, max_budget: int | None = None
) -> int:
    """Total node budget B. Chain (W<=1) == block_size; else the full W-ary tree of
    depth block_size-1, i.e. (W^block_size - 1)/(W-1), capped by max_budget."""
    if tree_width <= 1:
        return block_size
    full = (tree_width**block_size - 1) // (tree_width - 1)
    if max_budget is not None and max_budget > 0:
        return min(full, max_budget)
    return full


class DFlashDraftTree(msgspec.Struct):
    """Host tree. Node 0 is the root (= bonus / last verified token).

    Invariants: parent_indices[0] == -1; parent_indices[i] < i; depths[0] == 0.
    """

    tokens: list[int]
    parent_indices: list[int]
    depths: list[int]
    num_nodes: int

    def longest_chain_len(self) -> int:
        return (max(self.depths) + 1) if self.num_nodes else 0


def build_tree_opt_prefix(root_token, topk_tokens, topk_logprobs, budget):
    """Greedy best-first: enumerate the top-`budget` highest-accumulated-logprob
    root->node prefixes via a max-heap of rank-tuples. One node per pop; each pop
    pushes the node's next sibling (same parent, next rank) and first child (next
    depth, rank 0). Port of vllm-jetspec dflash_tree.py:467-541 (paper Algorithm 1)."""
    depth_count = len(topk_tokens)
    width = len(topk_tokens[0]) if depth_count else 0
    K = min(width, budget)
    tokens = [root_token]
    parent_indices = [-1]
    depths = [0]
    rank_to_idx = {(): 0}
    n = 1
    if K == 0 or budget <= 1 or depth_count == 0:
        return DFlashDraftTree(tokens, parent_indices, depths, n)
    counter = 0
    heap = [(-topk_logprobs[0][0], counter, (0,))]
    while heap and n < budget:
        neg_score, _, ranks = heapq.heappop(heap)
        score = -neg_score
        d = len(ranks)  # 1-based tree depth of this node
        parent = rank_to_idx[ranks[:-1]]
        r = ranks[-1]
        tokens.append(topk_tokens[d - 1][r])
        parent_indices.append(parent)
        depths.append(d)
        rank_to_idx[ranks] = n
        n += 1
        if r + 1 < K:  # next sibling: swap last logprob term
            counter += 1
            sib = score - topk_logprobs[d - 1][r] + topk_logprobs[d - 1][r + 1]
            heapq.heappush(heap, (-sib, counter, ranks[:-1] + (r + 1,)))
        if d < depth_count:  # first child: extend with best at next depth
            counter += 1
            child = score + topk_logprobs[d][0]
            heapq.heappush(heap, (-child, counter, ranks + (0,)))
    return DFlashDraftTree(tokens, parent_indices, depths, n)


def build_tree_depth_first(root_token, topk_tokens, topk_logprobs, budget):
    """Phase 1: lay down the greedy top-1 spine (guarantees tree acceptance >= chain
    acceptance). Phase 2: add side branches by accumulated logprob, spine nodes
    skipping their already-present rank-0 child. Port of dflash_tree.py:365-457."""
    depth_count = len(topk_tokens)
    width = len(topk_tokens[0]) if depth_count else 0
    tokens = [root_token]
    parent_indices = [-1]
    depths = [0]
    cum_logprob = [0.0]
    is_spine = [True]
    n = 1
    if depth_count == 0 or budget <= 1 or width == 0:
        return DFlashDraftTree(tokens, parent_indices, depths, n)
    # Phase 1: greedy top-1 spine
    parent = 0
    for d in range(depth_count):
        if n >= budget:
            break
        tokens.append(topk_tokens[d][0])
        parent_indices.append(parent)
        depths.append(d + 1)
        cum_logprob.append(cum_logprob[parent] + topk_logprobs[d][0])
        is_spine.append(True)
        parent = n
        n += 1
    # Phase 2: side branches, best accumulated-logprob first
    counter = 0
    heap = []

    def _push_children(node_idx):
        nonlocal counter
        d = depths[node_idx]  # parent tree-depth; child uses draft-depth d
        if d >= depth_count:
            return
        start = 1 if is_spine[node_idx] else 0  # spine already owns its rank-0 child
        for r in range(start, width):
            counter += 1
            score = cum_logprob[node_idx] + topk_logprobs[d][r]
            heapq.heappush(heap, (-score, counter, node_idx, d, r))

    for idx in range(n):
        _push_children(idx)
    while heap and n < budget:
        neg_score, _, parent_idx, d, r = heapq.heappop(heap)
        score = -neg_score
        tokens.append(topk_tokens[d][r])
        parent_indices.append(parent_idx)
        depths.append(d + 1)
        cum_logprob.append(score)
        is_spine.append(False)
        new_idx = n
        n += 1
        _push_children(new_idx)
    return DFlashDraftTree(tokens, parent_indices, depths, n)


def _rows_to_lists(rows):
    """Normalize per-depth rows (tensor / list-of-tensors / list-of-lists) to nested
    Python lists so the host-side builders avoid torch scalar-dispatch overhead."""
    if torch.is_tensor(rows):
        return rows.tolist()
    return [row.tolist() if torch.is_tensor(row) else list(row) for row in rows]


def build_tree(
    root_token,
    topk_tokens,
    topk_logprobs,
    *,
    tree_width,
    budget,
    construction="depth_first",
):
    """Dispatch to a builder using at most ``tree_width`` candidates per depth.

    Rows may carry more top-k columns than wanted, so cap the per-depth fan-out to
    ``tree_width`` here. ``tree_width == 1`` degenerates to a chain in either builder.

    Assumes ``topk_logprobs[d]`` is sorted in descending order (standard
    torch.topk(largest=True, sorted=True) contract).

    Inputs may arrive as tensors (drafted on-device); normalize to host lists up
    front so the per-node Python loops in the builders avoid torch scalar-dispatch
    overhead. This is the single production entry point for both builders.
    """
    topk_tokens = _rows_to_lists(topk_tokens)
    topk_logprobs = _rows_to_lists(topk_logprobs)
    topk_tokens = [row[:tree_width] for row in topk_tokens]
    topk_logprobs = [row[:tree_width] for row in topk_logprobs]
    if construction == "opt_prefix":
        return build_tree_opt_prefix(root_token, topk_tokens, topk_logprobs, budget)
    if construction == "depth_first":
        return build_tree_depth_first(root_token, topk_tokens, topk_logprobs, budget)
    raise ValueError(f"unknown tree construction: {construction!r}")


def tree_to_verify_buffers(trees, num_verify_tokens, base_positions, device="cpu"):
    """Convert host trees into SGLang verify buffers. Emits left-child
    (`retrieve_next_token`) / right-sibling (`retrieve_next_sibling`) pointers with
    the -1 sentinel, matching the chain builder's contract (dflash_utils.py:199-255).
    Short trees are padded to num_verify_tokens with isolated nodes (never accepted)."""
    bs = len(trees)
    n = num_verify_tokens
    draft_token = torch.zeros((bs, n), dtype=torch.int64)
    retrieve_next_token = torch.full((bs, n), -1, dtype=torch.int64)
    retrieve_next_sibling = torch.full((bs, n), -1, dtype=torch.int64)
    positions = torch.zeros((bs, n), dtype=torch.int64)
    for b, tree in enumerate(trees):
        m = tree.num_nodes
        assert m <= n, f"tree has {m} nodes > num_verify_tokens {n}"
        assert tree.parent_indices[0] == -1
        draft_token[b, :m] = torch.tensor(tree.tokens, dtype=torch.int64)
        positions[b, :m] = base_positions[b] + torch.tensor(
            tree.depths, dtype=torch.int64
        )
        last_child = {}
        for i in range(m):
            p = tree.parent_indices[i]
            if p == -1:
                continue
            assert p < i, "parent must precede child"
            if p in last_child:
                retrieve_next_sibling[b, last_child[p]] = i
            else:
                retrieve_next_token[b, p] = i
            last_child[p] = i
    retrieve_index = torch.arange(bs * n, dtype=torch.int64).view(bs, n)
    out = {
        "draft_token": draft_token,
        "retrieve_index": retrieve_index,
        "retrieve_next_token": retrieve_next_token,
        "retrieve_next_sibling": retrieve_next_sibling,
        "positions": positions,
        # Output buffers filled by the verify kernel; uninitialized on purpose.
        "predicts": torch.empty((bs * n,), dtype=torch.int32),
        "accept_index": torch.empty((bs, n), dtype=torch.int32),
        "accept_token_num": torch.empty((bs,), dtype=torch.int32),
    }
    if device != "cpu":
        out = {k: v.to(device) for k, v in out.items()}
    return out


def build_ancestor_mask(parent_indices, num_nodes):
    """[N, N] bool allow-mask: m[i, j] True iff j == i, j == root(0), or j is a
    transitive ancestor of i. Port of build_ancestor_matrix_np semantics."""
    n = num_nodes
    m = torch.zeros((n, n), dtype=torch.bool)
    for i in range(n):
        m[i, i] = True
        m[i, 0] = True
        p = parent_indices[i]
        while p != -1:
            m[i, p] = True
            p = parent_indices[p]
    return m


def _padded_parents(tree, n):
    return list(tree.parent_indices) + [-1] * (n - tree.num_nodes)  # pad nodes isolated


def build_tree_custom_mask(trees, num_verify_tokens, kv_lens, device="cpu"):
    """Flattened DFlash custom_mask ("True == allowed"). Per request: each of the N
    draft-node query rows attends (a) the full committed prefix [kv_len keys, all
    allowed] then (b) the N intra-block keys gated by the ancestor mask. Total size
    matches generate_attn_arg_prefill (dflash_info.py:130-133):
        sum(kv_lens) * N + N*N * bs.
    NOTE: ancestor semantics + size are unit-tested here; exact byte interleaving vs
    each attention backend's reader is verified by the GPU e2e lossless gate."""
    n = num_verify_tokens
    parts = []
    for b, tree in enumerate(trees):
        kv = kv_lens[b]
        anc = build_ancestor_mask(_padded_parents(tree, n), n)  # [N, N]
        ctx = torch.ones((n, kv), dtype=torch.bool)  # [N, kv]
        parts.append(torch.cat([ctx, anc], dim=1).reshape(-1))  # row-major per req
    mask = torch.cat(parts, dim=0)
    return mask.to(device) if device != "cpu" else mask


def verify_tree_greedy_cpu(tree, target_predict):
    """CPU reference for greedy tree acceptance (test oracle + eager fallback).

    A child c is accepted iff tree.tokens[c] == target_predict[parent(c)]. Returns the
    longest root->leaf path of consecutively-accepted nodes; the bonus_token is the
    target's prediction at the last accepted node (the correction). Mirrors
    tree_accept_greedy (vllm-jetspec dflash_tree.py:1218)."""
    # target_predict may be a tensor (target argmax); use host values so the
    # comparison yields Python bools, not 0-D tensors, and avoids slow indexing.
    if torch.is_tensor(target_predict):
        target_predict = target_predict.tolist()
    children = {i: [] for i in range(tree.num_nodes)}
    for i in range(1, tree.num_nodes):
        children[tree.parent_indices[i]].append(i)

    best_path = [0]

    def walk(node, path):
        nonlocal best_path
        if len(path) > len(best_path):
            best_path = list(path)
        for c in children[node]:
            if tree.tokens[c] == target_predict[node]:
                walk(c, path + [c])

    walk(0, [0])
    last = best_path[-1]
    return {
        "accept_indices": best_path,
        "num_accept_tokens": len(best_path),
        "bonus_token": int(target_predict[last]),
    }
