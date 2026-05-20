"""Page-score computation + top-K selection for Double Sparsity.

Two pipeline stages, both capture-safe (no host syncs, no dynamic shapes):

1. **Score**. ``compute_page_scores`` consumes the per-(layer, page)
   compressed signatures (``[L, P, H_local, label_dim]`` fp16) and a per-row
   query (``[bs, num_local_heads, head_dim]`` bf16/fp16) and returns
   ``page_scores[bs, max_pages]`` fp16 — max-over-heads of the channel-mask-
   projected dot product. For TP-correctness the caller all-reduces the
   resulting scalar scores across the attention TP group (DEC-9), so per-
   rank top-K agrees by construction.

2. **Select**. ``select_topk_sequence_order`` consumes the all-reduced
   scores plus the ``valid_mask`` for invalidated pages and an optional
   list of hot pages that must be forced into the selected set (active
   in-fill page + N most-recent pages, per CMT-14). Returns
   ``(selected_indices, valid_lengths)`` with ``selected_indices`` in
   **sequence-order ascending** (logical-page-ID order) with ``-1`` padding,
   per the selector ABI contract.

The torch-based reference implementation is correct, deterministic, and
capture-safe. A Triton kernel for the score step lives in
:func:`_triton_score_kernel` and is selected automatically when running on
CUDA; the torch path is kept as the documented reference and as the
fallback for capture-mode debugging and CPU unit tests. The top-K step
uses ``torch.topk`` + ``torch.sort`` (both CUDA-graph capture-safe with
static shapes).
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence, Tuple

import torch

logger = logging.getLogger(__name__)


SELECTED_PAD_VALUE = -1
_TRITON_AVAILABLE = False
try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False


def _next_pow2(n: int) -> int:
    """Smallest power of two >= ``n`` (n>=1). Triton's ``tl.arange`` extents
    must be powers of two; we pad up and mask the unused tail.
    """
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


if _TRITON_AVAILABLE:

    @triton.jit
    def _compute_page_scores_kernel(
        q_proj_ptr,  # [bs, H, label_dim] fp32
        sig_ptr,  # [P, H, label_dim] fp16 or fp32
        valid_ptr,  # [P] bool
        out_ptr,  # [bs, P] fp32
        bs: tl.constexpr,
        num_heads: tl.constexpr,
        max_pages: tl.constexpr,
        label_dim: tl.constexpr,
        q_stride_b: tl.constexpr,
        q_stride_h: tl.constexpr,
        sig_stride_p: tl.constexpr,
        sig_stride_h: tl.constexpr,
        out_stride_b: tl.constexpr,
        PAGE_BLOCK: tl.constexpr,  # power-of-two >= desired block
        LABEL_DIM_POW2: tl.constexpr,  # power-of-two >= label_dim
    ):
        batch_id = tl.program_id(0)
        page_block = tl.program_id(1)
        page_offsets = page_block * PAGE_BLOCK + tl.arange(0, PAGE_BLOCK)
        page_in_range = page_offsets < max_pages

        d_offsets = tl.arange(0, LABEL_DIM_POW2)
        d_mask = d_offsets < label_dim

        # Per-head max accumulator: start at -inf.
        max_score = tl.full((PAGE_BLOCK,), float("-inf"), dtype=tl.float32)

        for h in range(num_heads):
            q_offsets = (
                batch_id * q_stride_b + h * q_stride_h + d_offsets
            )
            q_block = tl.load(
                q_proj_ptr + q_offsets, mask=d_mask, other=0.0
            ).to(tl.float32)
            # [LABEL_DIM_POW2]; padded entries are 0 so they contribute 0 to the dot.

            # Page-block × label_dim signatures: shape [PAGE_BLOCK, LABEL_DIM_POW2].
            sig_offsets = (
                page_offsets[:, None] * sig_stride_p
                + h * sig_stride_h
                + d_offsets[None, :]
            )
            sig_block = tl.load(
                sig_ptr + sig_offsets,
                mask=page_in_range[:, None] & d_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            # [PAGE_BLOCK, LABEL_DIM_POW2]

            # Dot per page = sum over d of q[d] * sig[p, d]. Padded label-dim
            # entries are 0 on both sides and contribute 0.
            dot = tl.sum(q_block[None, :] * sig_block, axis=1)
            # [PAGE_BLOCK]

            max_score = tl.where(dot > max_score, dot, max_score)

        # Mask invalid pages to -inf.
        valid_block = tl.load(
            valid_ptr + page_offsets, mask=page_in_range, other=0
        ).to(tl.int1)
        out_score = tl.where(
            valid_block, max_score, tl.full(max_score.shape, float("-inf"), dtype=tl.float32)
        )

        out_offsets = batch_id * out_stride_b + page_offsets
        tl.store(out_ptr + out_offsets, out_score, mask=page_in_range)


def _compute_page_scores_triton(
    q_proj: torch.Tensor,
    sig_layer: torch.Tensor,
    valid_layer: torch.Tensor,
    *,
    page_block: int = 64,
) -> torch.Tensor:
    """Triton kernel-driven page scoring.

    Args:
        q_proj: ``[bs, H, label_dim]`` fp32 (caller has already applied
            ``project_query_onto_channels``).
        sig_layer: ``[max_pages, H, label_dim]`` fp16/fp32 — the active
            layer's slice of the page signature table.
        valid_layer: ``[max_pages]`` bool — the active layer's valid mask.

    Returns:
        ``[bs, max_pages]`` fp32 page scores. Invalid pages are ``-inf``.
    """

    assert q_proj.is_cuda and sig_layer.is_cuda and valid_layer.is_cuda
    bs, num_heads, label_dim = q_proj.shape
    max_pages = int(sig_layer.shape[0])
    out = torch.empty((bs, max_pages), dtype=torch.float32, device=q_proj.device)

    q_proj_c = q_proj.contiguous()
    sig_c = sig_layer.contiguous()
    valid_c = valid_layer.contiguous()

    # Triton's tl.arange extents must be powers of two; pad the per-tile
    # block sizes up and rely on masks to drop padded entries.
    desired_block = min(page_block, max(max_pages, 1))
    if desired_block <= 0:
        desired_block = max(1, max_pages)
    page_block_pow2 = _next_pow2(desired_block)
    label_dim_pow2 = _next_pow2(max(label_dim, 1))
    num_page_blocks = (max_pages + page_block_pow2 - 1) // page_block_pow2
    grid = (bs, num_page_blocks)

    _compute_page_scores_kernel[grid](
        q_proj_c,
        sig_c,
        valid_c,
        out,
        bs=bs,
        num_heads=num_heads,
        max_pages=max_pages,
        label_dim=label_dim,
        q_stride_b=q_proj_c.stride(0),
        q_stride_h=q_proj_c.stride(1),
        sig_stride_p=sig_c.stride(0),
        sig_stride_h=sig_c.stride(1),
        out_stride_b=out.stride(0),
        PAGE_BLOCK=page_block_pow2,
        LABEL_DIM_POW2=label_dim_pow2,
    )
    return out


def project_query_onto_channels(
    queries: torch.Tensor,
    channel_selection: torch.Tensor,
    channel_weights: torch.Tensor,
) -> torch.Tensor:
    """Project ``queries[bs, H, head_dim]`` onto each head's channel mask.

    Returns ``query_projected[bs, H, label_dim]`` =
        queries.gather(channel_selection[h]) * channel_weights[h].

    Caller is responsible for handing in the per-layer slice of
    ``channel_selection`` / ``channel_weights``, both of shape
    ``[H, label_dim]``.
    """

    if queries.dim() != 3:
        raise ValueError(
            f"queries must be 3-D [bs, H, head_dim], got shape {tuple(queries.shape)}."
        )
    if channel_selection.dim() != 2 or channel_weights.dim() != 2:
        raise ValueError(
            "channel_selection/channel_weights must be 2-D [H, label_dim]."
        )
    bs, num_heads, head_dim = queries.shape
    if channel_selection.shape[0] != num_heads:
        raise ValueError(
            f"channel_selection head dim {int(channel_selection.shape[0])} does not match "
            f"queries num_heads {num_heads}."
        )
    label_dim = int(channel_selection.shape[-1])

    # gather: queries[bs, h, channel_selection[h, d]] for each d in [0, label_dim)
    # → [bs, H, label_dim]
    selection_idx = channel_selection.long().unsqueeze(0).expand(bs, -1, -1)
    gathered = torch.gather(queries, dim=-1, index=selection_idx)
    return gathered * channel_weights.unsqueeze(0)


def compute_page_scores(
    queries: torch.Tensor,
    page_signatures: torch.Tensor,
    valid_mask: torch.Tensor,
    channel_selection: torch.Tensor,
    channel_weights: torch.Tensor,
    layer_id: int,
) -> torch.Tensor:
    """Compute per-(batch, page) scalar scores.

    queries:           [bs, num_local_heads, head_dim] (bf16 or fp16)
    page_signatures:   [num_layers_local, max_pages, num_heads_local, label_dim]
    valid_mask:        [num_layers_local, max_pages] bool
    channel_selection: [num_layers_local, num_heads_local, label_dim] int32
    channel_weights:   [num_layers_local, num_heads_local, label_dim] fp32

    Returns ``page_scores[bs, max_pages]`` fp32. Invalid pages get
    ``-inf`` so the top-K step ignores them deterministically.
    """

    if not (0 <= layer_id < page_signatures.shape[0]):
        raise IndexError(
            f"layer_id={layer_id} out of range [0, {page_signatures.shape[0]})."
        )

    sel_layer = channel_selection[layer_id]  # [H, label_dim]
    w_layer = channel_weights[layer_id]  # [H, label_dim]
    sig_layer = page_signatures[layer_id]  # [P, H, label_dim]
    valid_layer = valid_mask[layer_id]  # [P]

    # Project queries onto channel-mask channels per head.
    q_proj = project_query_onto_channels(queries, sel_layer, w_layer)  # [bs, H, D]

    # Triton fast path: fuses gather/projection/per-head max-reduce in one
    # kernel. Falls back to the torch einsum reference on CPU or when Triton
    # is unavailable (e.g. import-time errors in CI environments).
    if (
        _TRITON_AVAILABLE
        and q_proj.is_cuda
        and sig_layer.is_cuda
        and valid_layer.is_cuda
    ):
        return _compute_page_scores_triton(
            q_proj.to(torch.float32),
            sig_layer,
            valid_layer,
        )

    # Score per (batch, page, head) = sum over label_dim of (q_proj * signature).
    # Reduce over heads with max (per CMT-7 / DEC-9: the reduction is the
    # per-rank "best head wins" view; an all_reduce(SUM) across TP ranks
    # composes ranks).
    # Shapes: q_proj [bs, H, D]  sig [P, H, D] -> scores [bs, P, H] -> max over H.
    scores_full = torch.einsum(
        "bhd,phd->bph", q_proj.to(torch.float32), sig_layer.to(torch.float32)
    )
    scores = scores_full.amax(dim=-1)  # [bs, P]
    return scores.masked_fill(~valid_layer.unsqueeze(0), float("-inf"))


def all_reduce_page_scores(
    page_scores: torch.Tensor,
    *,
    process_group=None,
) -> torch.Tensor:
    """All-reduce per-rank scalar page scores across the attention TP group.

    Per DEC-9 the page signatures stay TP/head-sharded; only the scalar
    per-page scores are reduced (SUM). The reduction operates in-place on
    ``page_scores`` and returns the same tensor for convenience. When no
    process group is provided, this is a no-op (single-rank semantics).
    """

    if process_group is None:
        return page_scores
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return page_scores
    torch.distributed.all_reduce(
        page_scores,
        op=torch.distributed.ReduceOp.SUM,
        group=process_group,
    )
    return page_scores


def select_topk_sequence_order(
    page_scores: torch.Tensor,
    max_top_k: int,
    *,
    hot_pages: Optional[Sequence[Sequence[int]]] = None,
    per_request_valid: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Top-K selection returning sequence-order-ascending indices.

    page_scores: [bs, max_pages] fp32 (invalid pages = -inf).
    hot_pages:   optional [bs] list of int lists with pages that must be in
                 the selected set regardless of score (active in-fill page +
                 local-window).
    per_request_valid: optional [bs, max_pages] bool/int mask. When supplied,
                 hot-page forcing only overwrites the score for cells the
                 row's mask says are valid; foreign-page cells stay at the
                 caller-set -inf. Filtering happens device-side via
                 ``torch.where`` — no host sync.

    Returns:
        selected_indices: int32 [bs, max_top_k], ascending, -1 padded.
        valid_lengths:    int32 [bs].
    """

    if page_scores.dim() != 2:
        raise ValueError(
            f"page_scores must be 2-D [bs, max_pages], got {tuple(page_scores.shape)}."
        )
    if max_top_k <= 0:
        raise ValueError(f"max_top_k must be positive, got {max_top_k}.")
    bs, max_pages = page_scores.shape
    device = page_scores.device

    # Force hot pages to +inf so they always win the top-K. When
    # per_request_valid is supplied, gate the +inf write per cell with a
    # device-side ``torch.where`` so a hot page outside the row's owned set
    # stays at -inf (the caller's per-request mask).
    if hot_pages is not None:
        if len(hot_pages) != bs:
            raise ValueError(
                f"hot_pages length {len(hot_pages)} must match batch size {bs}."
            )
        inf_scalar = (
            page_scores.new_tensor(float("inf"))
            if per_request_valid is not None
            else None
        )
        for row, pages in enumerate(hot_pages):
            for page_id in pages:
                if not (0 <= page_id < max_pages):
                    raise IndexError(
                        f"hot page {page_id} out of range [0, {max_pages})."
                    )
                if per_request_valid is None:
                    page_scores[row, page_id] = float("inf")
                else:
                    page_scores[row, page_id] = torch.where(
                        per_request_valid[row, page_id].to(torch.bool),
                        inf_scalar,
                        page_scores[row, page_id],
                    )

    effective_top_k = min(max_top_k, max_pages)
    topk = torch.topk(
        page_scores, k=effective_top_k, dim=-1, largest=True, sorted=False
    )
    topk_indices = topk.indices  # [bs, effective_top_k]
    topk_scores = topk.values  # [bs, effective_top_k]

    # An entry whose underlying score is -inf was an invalid page (or
    # an unfilled row position when effective_top_k > num_valid_pages).
    # Mark it as padding so valid_lengths counts only real pages.
    # Note: +inf is reserved for hot-page overrides and must NOT be marked
    # invalid here.
    invalid_entries = torch.isneginf(topk_scores)
    topk_indices = torch.where(
        invalid_entries,
        torch.full_like(topk_indices, max_pages),  # sentinel beyond valid range
        topk_indices,
    )

    # Sort selected indices ascending (sequence order) along the row.
    sorted_indices, _ = torch.sort(topk_indices, dim=-1)

    selected = torch.full(
        (bs, max_top_k),
        SELECTED_PAD_VALUE,
        dtype=torch.int32,
        device=device,
    )
    valid_mask_real = sorted_indices < max_pages
    valid_lengths = valid_mask_real.to(torch.int32).sum(dim=-1)

    # Pack the real entries into the front of selected, -1 the rest.
    if effective_top_k > 0:
        position_grid = torch.arange(effective_top_k, device=device)
        keep_positions = position_grid.unsqueeze(0) < valid_lengths.unsqueeze(1)
        # Where keep_positions is True we copy sorted_indices, else leave -1.
        real_slice = torch.where(
            keep_positions,
            sorted_indices.to(torch.int32),
            torch.full_like(sorted_indices, SELECTED_PAD_VALUE, dtype=torch.int32),
        )
        selected[:, :effective_top_k] = real_slice

    return selected, valid_lengths.to(torch.int32)


def retrieve_topk_via_signatures(
    *,
    queries: torch.Tensor,
    page_signatures: torch.Tensor,
    valid_mask: torch.Tensor,
    channel_selection: torch.Tensor,
    channel_weights: torch.Tensor,
    layer_id: int,
    max_top_k: int,
    hot_pages: Optional[Sequence[Sequence[int]]] = None,
    process_group=None,
    per_request_valid: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """End-to-end real-selector flow: score → all-reduce → per-request mask → top-K → ascend.

    ``per_request_valid``: optional ``[bs, max_pages]`` bool/int tensor flagging
    the pages each request actually owns. The global ``valid_mask`` only says
    which pages are currently live across the whole table; without a per-row
    mask, a request would be free to pick globally-valid pages owned by other
    requests in a mixed batch. ``None`` disables the per-row gate (single-
    request / unit-test paths).
    """

    scores = compute_page_scores(
        queries=queries,
        page_signatures=page_signatures,
        valid_mask=valid_mask,
        channel_selection=channel_selection,
        channel_weights=channel_weights,
        layer_id=layer_id,
    )
    scores = all_reduce_page_scores(scores, process_group=process_group)
    pr_bool: Optional[torch.Tensor] = None
    if per_request_valid is not None:
        if per_request_valid.shape != scores.shape:
            raise ValueError(
                f"per_request_valid shape {tuple(per_request_valid.shape)} must "
                f"match page_scores shape {tuple(scores.shape)}."
            )
        pr_bool = per_request_valid.to(torch.bool)
        scores = scores.masked_fill(~pr_bool, float("-inf"))
    indices, valid_lengths = select_topk_sequence_order(
        scores,
        max_top_k,
        hot_pages=hot_pages,
        per_request_valid=pr_bool,
    )
    # Best-effort observability emit. Requires a .item() sync on
    # valid_lengths (and per_request_valid when supplied), so it must be
    # skipped while a CUDA graph is being captured — otherwise the host
    # sync fails the capture. Eager / non-capture replays keep the metric.
    # The page-table adapter milestone will replace this with a capture-
    # safe pattern (write to a pre-allocated CPU scratch then emit after
    # replay).
    if not torch.cuda.is_current_stream_capturing():
        from sglang.srt.layers.attention.double_sparsity import metrics as _metrics

        selected_pages = int(valid_lengths.sum().item())
        if pr_bool is not None:
            total_valid_pages = int(pr_bool.to(torch.int64).sum().item())
        else:
            total_valid_pages = int(valid_lengths.shape[0]) * int(scores.shape[-1])
        _metrics.record_selection(
            selected_pages=selected_pages,
            total_valid_pages=total_valid_pages,
        )
    return indices, valid_lengths
