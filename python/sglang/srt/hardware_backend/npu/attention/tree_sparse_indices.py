"""Tree-aware ``sparse_indices`` construction for DSA attention on Ascend.

GLM-5.2 runs DeepSeek Sparse Attention on every attention layer: instead of a
mask, ``npu_sparse_flash_attention`` receives ``sparse_indices[T, K]`` — an
explicit list of KV positions each query token may read. Tree verify therefore
encodes the topology *in the index list*:

    row_r = [ ancestors(r) | prefix top-(K-A) (score order) | -1 padding ]

Why this shape
--------------
* **Ancestors first.** ``sparse_mode=3`` gives each row a budget of
  ``min(K, Lkv - Lq + i + 1)`` slots (see
  ``sparse_flash_attention_kernel_mla.h``), so short contexts truncate the tail.
  Putting ancestors first guarantees they survive.
* **No sorting.** SFA does not require ascending indices (its own tests feed
  ``randperm`` order), and sorting would push ancestors (larger positions) into
  the truncated tail.
* **``-1`` only in the tail.** The kernel stops at the first ``-1``; the
  lightning indexer pads with ``-1`` when the prefix is shorter than the
  requested width, which lands exactly at the end of the prefix block.
* **Duplicates are avoided for ancestors** (self-padding only for shallow
  nodes): a duplicate consumes budget and double-counts in softmax.

Pure torch, no ``torch_npu`` import: unit-testable on CPU.
"""

from __future__ import annotations

from typing import Optional

import torch

__all__ = [
    "ancestor_width_for",
    "gather_tree_blocks",
    "draft_ancestor_ids_batched",
    "merge_tree_sparse_indices",
]


def ancestor_width_for(num_steps: int, num_verify_tokens: Optional[int] = None) -> int:
    """Slots reserved for "self + ancestors" in each sparse-index row.

    A node at depth ``d`` needs ``d + 1`` entries; the deepest node of an EAGLE
    tree built with ``num_steps`` draft steps sits at depth ``num_steps``.
    """
    width = int(num_steps) + 1
    if num_verify_tokens is not None:
        width = min(width, int(num_verify_tokens))
    if width <= 0:
        raise ValueError(f"invalid ancestor width {width}")
    return width


def gather_tree_blocks(
    full_mask: torch.Tensor,
    prefix_lens: torch.Tensor,
    num_verify_tokens: int,
) -> torch.Tensor:
    """Slice the draft-block part of the flat ``custom_mask`` into ``[bs, N, N]``.

    ``custom_mask`` is the FLAT buffer produced by
    ``build_tree_kernel_efficient``: for request ``i`` it holds ``N`` rows of
    width ``P_i + N``; columns ``[0, P_i)`` are the (always visible) prefix and
    columns ``[P_i, P_i + N)`` are the draft block, where ``True`` means
    "visible" (self + ancestors).

    Returns a bool tensor with the same convention (``True`` = visible).
    """
    n = int(num_verify_tokens)
    if n <= 0:
        raise ValueError(f"num_verify_tokens must be positive, got {n}")
    if prefix_lens.dim() != 1:
        raise ValueError(f"prefix_lens must be 1-D, got {prefix_lens.shape}")
    p = prefix_lens.to(torch.int64)
    bs = p.numel()
    flat = full_mask.reshape(-1)

    expected = int((p + n).sum().item()) * n
    if flat.numel() < expected:
        raise ValueError(
            f"custom_mask too small: need {expected} elements for N={n}, "
            f"prefix_lens={p.tolist()}, got {flat.numel()}"
        )

    device = flat.device
    p = p.to(device)
    widths = p + n
    offsets = torch.cumsum(widths * n, dim=0) - widths * n
    rows = torch.arange(n, device=device)
    # index[b, r, c] = offset[b] + r * (P_b + N) + P_b + c
    index = (
        offsets.view(bs, 1, 1)
        + rows.view(1, n, 1) * widths.view(bs, 1, 1)
        + p.view(bs, 1, 1)
        + rows.view(1, 1, n)
    )
    return flat[index].to(torch.bool)


def draft_ancestor_ids_batched(
    scores: torch.Tensor,
    allowed: torch.Tensor,
    ancestor_width: int,
) -> torch.Tensor:
    """Pick, per query row, the highest-scoring allowed draft nodes.

    Args:
        scores: ``[bs, N, N]`` draft-block indexer scores; disallowed columns
            must already be ``-inf`` (or will be masked here).
        allowed: ``[bs, N, N]`` bool, "self + ancestors" relation.
        ancestor_width: reserved width ``A``; clamped to ``N``.

    Returns:
        ``[bs, N, A]`` int32 request-local node ids, ordered by descending
        score. The node itself is always present: the allowed set never exceeds
        ``A``, and the last slot is forced to self if that ever changes.

    A stable sort is used instead of ``topk`` on purpose: when a row has fewer
    legal columns than ``A`` (or equal scores), ``topk``'s tie-breaking may drop
    a real ancestor.
    """
    if scores.shape != allowed.shape:
        raise ValueError(
            f"scores/allowed shape mismatch: {tuple(scores.shape)} vs "
            f"{tuple(allowed.shape)}"
        )
    if scores.dim() != 3:
        raise ValueError(f"expected 3-D scores, got {tuple(scores.shape)}")

    num_nodes = scores.shape[1]
    width = min(int(ancestor_width), num_nodes)
    allowed = allowed.to(torch.bool)

    masked = scores.masked_fill(~allowed, float("-inf"))
    order = torch.argsort(masked, dim=-1, descending=True, stable=True)[..., :width]
    legal = allowed.gather(-1, order)

    self_ids = torch.arange(num_nodes, device=scores.device).view(1, num_nodes, 1)
    ids = torch.where(legal, order, self_ids)

    # Guarantee self without clobbering a real ancestor.
    has_self = (ids == self_ids).any(dim=-1)
    ids[..., -1] = torch.where(has_self, ids[..., -1], self_ids[..., 0])
    return ids.to(torch.int32)


def merge_tree_sparse_indices(
    prefix_indices: torch.Tensor,
    draft_ids: torch.Tensor,
    seq_lens: torch.Tensor,
    num_verify_tokens: int,
    topk: int,
) -> torch.Tensor:
    """Concatenate ancestors and prefix top-k into one sparse-index row.

    Args:
        prefix_indices: ``[bs*N, K - A]`` absolute KV positions (or ``-1``
            padding produced by the indexer when the prefix is shorter).
        draft_ids: ``[bs*N, A]`` request-local node ids from
            :func:`draft_ancestor_ids_batched`.
        seq_lens: ``[bs]`` prefix length per request (excluding draft tokens).
        num_verify_tokens: ``N``, rows per request.
        topk: ``K``, the sparse width the attention op expects.

    Returns:
        ``[bs*N, K]`` int32, **ancestors first** then the prefix block, with the
        prefix block's own order preserved (score order) and no sorting.

    Raises:
        AssertionError: width/row-count mismatch.
        ValueError: an ancestor outside its request's draft block, or a prefix
            index that is not ``< P_i`` (and not ``-1``).
    """
    n = int(num_verify_tokens)
    topk = int(topk)
    bs = int(seq_lens.numel())
    rows = bs * n

    assert prefix_indices.dim() == 2 and draft_ids.dim() == 2, (
        f"prefix_indices/draft_ids must be 2-D, got "
        f"{tuple(prefix_indices.shape)} / {tuple(draft_ids.shape)}"
    )
    assert prefix_indices.shape[0] == rows, (
        f"prefix_indices has {prefix_indices.shape[0]} rows, expected bs*N={rows}"
    )
    assert draft_ids.shape[0] == rows, (
        f"draft_ids has {draft_ids.shape[0]} rows, expected bs*N={rows}"
    )
    assert prefix_indices.shape[1] + draft_ids.shape[1] == topk, (
        f"reserved width mismatch: prefix {prefix_indices.shape[1]} + "
        f"ancestors {draft_ids.shape[1]} != topk {topk}"
    )

    draft_ids = draft_ids.to(torch.int64)
    if draft_ids.numel() and (draft_ids.min() < 0 or draft_ids.max() >= n):
        raise ValueError(
            f"draft node ids must be in [0, {n}), got "
            f"[{int(draft_ids.min())}, {int(draft_ids.max())}]"
        )

    prefix_lens = seq_lens.to(torch.int64).repeat_interleave(n)
    absolute = prefix_lens.view(-1, 1) + draft_ids

    # Ancestors first: sparse_mode=3 truncates the tail on short contexts.
    merged = torch.cat([absolute, prefix_indices.to(torch.int64)], dim=-1)

    if absolute.numel():
        low = prefix_lens.view(-1, 1)
        high = (prefix_lens + n).view(-1, 1)
        if bool((absolute < low).any()) or bool((absolute >= high).any()):
            raise ValueError(
                f"draft ancestor index outside [{int(low.min())}, "
                f"{int(high.max()) - 1}]"
            )

    pref = prefix_indices.to(torch.int64)
    limit = prefix_lens.view(-1, 1).expand_as(pref)
    bad = (pref >= 0) & (pref >= limit)
    if bool(bad.any()):
        raise ValueError(
            f"prefix index {int(pref[bad].max())} is not < its prefix length"
        )

    return merged.to(torch.int32)
