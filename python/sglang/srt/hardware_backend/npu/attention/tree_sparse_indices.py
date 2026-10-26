"""Tree-aware ``sparse_indices`` construction for DSA attention on Ascend.

GLM-5.2 runs DeepSeek Sparse Attention on every attention layer: instead of a
mask, ``npu_sparse_flash_attention`` receives ``sparse_indices[T, K]`` — an
explicit list of KV positions each query token may read. Tree verify therefore
encodes the topology *in the index list*:

    row_r = [ ancestors(r) | prefix top-* (score order) | -1 padding ]

Why this shape
--------------
* **Ancestors first.** ``sparse_mode=3`` bounds the row by
  ``j <= (L_kv - L_q) + i`` and the kernel stops scanning after
  ``min(K, L_kv - L_q + i + 1)`` entries; a short context therefore truncates
  the tail. Putting ancestors first keeps them.
* **No sorting.** SFA does not require ascending indices (its own tests feed
  ``randperm``); sorting would push ancestors (larger positions) into the
  truncated tail.
* **No duplicate positions.** The kernel scans the list verbatim — there is no
  uniqueness check (``sparse_flash_attention_kernel_mla.h`` only stops at
  ``-1``), and ``sparse_block_size`` is 1, so a repeated position is read twice
  and double-counts in the softmax *and* eats the causal budget. Every row is
  therefore built by shifting the prefix block left by the row's own ancestor
  count instead of padding the ancestor block with the node itself.
* **``-1`` only in the tail.** The kernel stops at the first ``-1``; valid
  values must all precede the padding. The prefix block keeps its own (already
  tail-aligned) ``-1`` padding, and the slack after it is ``-1`` too, so the
  whole row stays "valid... valid, -1... -1".

Pure torch, no ``torch_npu`` import: unit-testable on CPU.
"""

from __future__ import annotations

from typing import Optional

import torch

__all__ = [
    "ancestor_width_for",
    "gather_tree_blocks",
    "draft_ancestors",
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


def draft_ancestors(allowed: torch.Tensor, ancestor_width: int) -> torch.Tensor:
    """Per row, the visible draft nodes in ascending node order, ``-1`` padded.

    Args:
        allowed: ``[bs, N, N]`` bool, "self + ancestors" relation (``True`` =
            visible), as returned by :func:`gather_tree_blocks`.
        ancestor_width: reserved width ``A``; clamped to ``N``.

    Returns:
        ``[bs, N, A]`` int32 request-local node ids, ascending, with ``-1`` in
        the unused tail slots.

    Ascending order is used instead of a score ranking on purpose: a row's
    visible set is exactly ``{self} | ancestors``, whose size is ``depth + 1``
    and therefore never exceeds ``A = num_steps + 1``. There is nothing to
    choose between, so ordering by position removes both the scoring matmul and
    any tie-breaking risk.
    """
    if allowed.dim() != 3:
        raise ValueError(f"expected 3-D allowed mask, got {tuple(allowed.shape)}")

    num_nodes = allowed.shape[1]
    if allowed.shape[2] != num_nodes:
        raise ValueError(
            f"allowed mask must be square in the draft dims, got "
            f"{tuple(allowed.shape)}"
        )
    width = min(int(ancestor_width), num_nodes)
    if width <= 0:
        raise ValueError(f"invalid ancestor width {width}")

    allowed = allowed.to(torch.bool)
    order = torch.arange(num_nodes, device=allowed.device)
    visible = allowed.sum(dim=-1)
    if bool((visible > width).any()):
        raise ValueError(
            f"a row sees {int(visible.max())} draft nodes but only {width} slots "
            f"are reserved (num_steps too small for N={num_nodes}); dropping an "
            "ancestor would silently change the attention"
        )
    # Allowed nodes keep their own index; hidden ones become the sentinel, so a
    # stable ascending sort puts the visible ids first, in ascending order.
    key = torch.where(allowed, order, torch.full_like(order, num_nodes))
    picked = torch.sort(key, dim=-1, stable=True).values[..., :width]
    return torch.where(picked < num_nodes, picked, -1).to(torch.int32)


def merge_tree_sparse_indices(
    prefix_indices: torch.Tensor,
    draft_ids: torch.Tensor,
    seq_lens: torch.Tensor,
    num_verify_tokens: int,
    topk: int,
) -> torch.Tensor:
    """Concatenate ancestors and the prefix top-k into one sparse-index row.

    The prefix block is shifted left per row by that row's ancestor count, so a
    shallow node simply carries more prefix entries and the row tail stays
    ``-1``:

        a_r = (draft_ids[r] >= 0).sum()
        row = [ ancestors(a_r) | prefix(0 .. K - a_r) | -1 (slack) ]

    Args:
        prefix_indices: ``[bs*N, W]`` absolute KV positions (or ``-1`` padding
            produced by the indexer when the prefix is shorter), score order.
        draft_ids: ``[bs*N, A]`` request-local node ids from
            :func:`draft_ancestors`, ``-1`` padded.
        seq_lens: ``[bs]`` prefix length per request (excluding draft tokens).
        num_verify_tokens: ``N``, rows per request.
        topk: ``K``, the sparse width the attention op expects.

    Returns:
        ``[bs*N, K]`` int32, **ancestors first** then the prefix block, with the
        prefix block's own order preserved (score order), no sorting, and
        ``-1`` only in the tail. No position appears twice.

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

    device = prefix_indices.device
    prefix_width = int(prefix_indices.shape[1])
    prefix_indices = prefix_indices.to(torch.int64).to(device)
    draft_ids = draft_ids.to(torch.int64).to(device)

    valid = draft_ids >= 0
    if draft_ids.numel() and bool((draft_ids[valid] >= n).any()):
        raise ValueError(
            f"draft node ids must be in [0, {n}) or -1, got "
            f"max {int(draft_ids.max())}"
        )
    if bool((valid.sum(dim=-1) == 0).any()):
        raise ValueError("every row needs at least its own node id")

    prefix_lens = seq_lens.to(device=device, dtype=torch.int64).repeat_interleave(n)
    prefix_lens = prefix_lens.view(-1, 1)
    absolute = prefix_lens + draft_ids.clamp(min=0)

    if prefix_width:
        pref = prefix_indices
        bad = (pref >= 0) & (pref >= prefix_lens)
        if bool(bad.any()):
            raise ValueError(
                f"prefix index {int(pref[bad].max())} is not < its prefix length"
            )
    if bool((absolute < prefix_lens).any()) or bool(
        (absolute >= prefix_lens + n).any()
    ):
        raise ValueError(
            "draft ancestor index outside its request's draft block "
            f"[P, P + {n})"
        )

    # Per-row shift: a_r ancestors occupy the front, the prefix follows.
    count = valid.sum(dim=-1, keepdim=True)  # [rows, 1]
    slots = torch.arange(topk, device=device).view(1, -1)
    rel = slots - count  # >= 0 marks the prefix region

    anc_col = slots.clamp(max=draft_ids.shape[1] - 1).expand(rows, -1)
    anc_val = torch.gather(absolute, 1, anc_col)
    if prefix_width:
        pref_col = rel.clamp(0, prefix_width - 1)
        pref_val = torch.gather(prefix_indices, 1, pref_col)
    else:
        pref_val = torch.full((rows, topk), -1, dtype=torch.int64, device=device)

    neg_one = torch.full((), -1, dtype=torch.int64, device=device)
    merged = torch.where(
        rel < 0, anc_val, torch.where(rel < prefix_width, pref_val, neg_one)
    )
    return merged.to(torch.int32)
