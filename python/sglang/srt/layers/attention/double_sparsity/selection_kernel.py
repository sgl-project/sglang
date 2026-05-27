"""Token-score computation + top-K selection for Double Sparsity.

Two pipeline stages, both capture-safe (no host syncs, no dynamic shapes):

1. **Score**. ``compute_token_scores`` consumes the per-(layer, token)
   compressed label signatures (``[L, T, H_local, label_dim]`` fp16) and a
   per-row query (``[bs, num_local_heads, head_dim]`` bf16/fp16) and returns
   ``token_scores[bs, max_tokens]`` fp32 — max-over-heads of the channel-mask-
   projected dot product. For TP-correctness the caller all-reduces the
   resulting scores across the attention TP group, so per-rank top-K
   agrees by construction.

2. **Select**. ``select_topk_sequence_order`` consumes the all-reduced
   scores plus the ``written`` mask for unpopulated tokens. Returns
   ``(selected_indices, valid_lengths)`` with ``selected_indices`` in
   **sequence-order ascending** (logical token position order) with ``-1``
   padding, per the selector ABI contract.

The torch-based reference implementation is correct, deterministic, and
capture-safe. A Triton kernel for the score step lives in
:func:`_compute_token_scores_kernel` and is selected automatically when
running on CUDA; the torch path is kept as the documented reference and as
the fallback for capture-mode debugging and CPU unit tests. The top-K step
uses ``torch.topk`` + ``torch.sort`` (both CUDA-graph capture-safe with
static shapes).
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

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
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


if _TRITON_AVAILABLE:

    @triton.jit
    def _compute_token_scores_kernel(
        q_proj_ptr,  # [bs, H, label_dim] fp32
        sig_ptr,     # [T, H, label_dim] fp16 or fp32
        written_ptr, # [T] bool
        out_ptr,     # [bs, T] fp32
        bs: tl.constexpr,
        num_heads: tl.constexpr,
        max_tokens: tl.constexpr,
        label_dim: tl.constexpr,
        q_stride_b: tl.constexpr,
        q_stride_h: tl.constexpr,
        sig_stride_t: tl.constexpr,
        sig_stride_h: tl.constexpr,
        out_stride_b: tl.constexpr,
        TOKEN_BLOCK: tl.constexpr,
        LABEL_DIM_POW2: tl.constexpr,
    ):
        batch_id = tl.program_id(0)
        token_block = tl.program_id(1)
        token_offsets = token_block * TOKEN_BLOCK + tl.arange(0, TOKEN_BLOCK)
        token_in_range = token_offsets < max_tokens

        d_offsets = tl.arange(0, LABEL_DIM_POW2)
        d_mask = d_offsets < label_dim

        max_score = tl.full((TOKEN_BLOCK,), float("-inf"), dtype=tl.float32)

        for h in range(num_heads):
            q_offsets = batch_id * q_stride_b + h * q_stride_h + d_offsets
            q_block = tl.load(
                q_proj_ptr + q_offsets, mask=d_mask, other=0.0
            ).to(tl.float32)

            sig_offsets = (
                token_offsets[:, None] * sig_stride_t
                + h * sig_stride_h
                + d_offsets[None, :]
            )
            sig_block = tl.load(
                sig_ptr + sig_offsets,
                mask=token_in_range[:, None] & d_mask[None, :],
                other=0.0,
            ).to(tl.float32)

            dot = tl.sum(q_block[None, :] * sig_block, axis=1)
            max_score = tl.where(dot > max_score, dot, max_score)

        written_block = tl.load(
            written_ptr + token_offsets, mask=token_in_range, other=0
        ).to(tl.int1)
        out_score = tl.where(
            written_block,
            max_score,
            tl.full(max_score.shape, float("-inf"), dtype=tl.float32),
        )

        out_offsets = batch_id * out_stride_b + token_offsets
        tl.store(out_ptr + out_offsets, out_score, mask=token_in_range)


def _compute_token_scores_triton(
    q_proj: torch.Tensor,
    sig_layer: torch.Tensor,
    written_layer: torch.Tensor,
    *,
    token_block: int = 64,
) -> torch.Tensor:
    """Triton kernel-driven token scoring.

    Args:
        q_proj: ``[bs, H, label_dim]`` fp32.
        sig_layer: ``[max_tokens, H, label_dim]`` fp16/fp32.
        written_layer: ``[max_tokens]`` bool.

    Returns:
        ``[bs, max_tokens]`` fp32. Unwritten tokens are ``-inf``.
    """

    assert q_proj.is_cuda and sig_layer.is_cuda and written_layer.is_cuda
    bs, num_heads, label_dim = q_proj.shape
    max_tokens = int(sig_layer.shape[0])
    out = torch.empty((bs, max_tokens), dtype=torch.float32, device=q_proj.device)

    q_proj_c = q_proj.contiguous()
    sig_c = sig_layer.contiguous()
    written_c = written_layer.contiguous()

    desired_block = min(token_block, max(max_tokens, 1))
    if desired_block <= 0:
        desired_block = max(1, max_tokens)
    token_block_pow2 = _next_pow2(desired_block)
    label_dim_pow2 = _next_pow2(max(label_dim, 1))
    num_token_blocks = (max_tokens + token_block_pow2 - 1) // token_block_pow2
    grid = (bs, num_token_blocks)

    _compute_token_scores_kernel[grid](
        q_proj_c,
        sig_c,
        written_c,
        out,
        bs=bs,
        num_heads=num_heads,
        max_tokens=max_tokens,
        label_dim=label_dim,
        q_stride_b=q_proj_c.stride(0),
        q_stride_h=q_proj_c.stride(1),
        sig_stride_t=sig_c.stride(0),
        sig_stride_h=sig_c.stride(1),
        out_stride_b=out.stride(0),
        TOKEN_BLOCK=token_block_pow2,
        LABEL_DIM_POW2=label_dim_pow2,
    )
    return out


def project_query_onto_channels(
    queries: torch.Tensor,
    channel_selection: torch.Tensor,
    channel_weights: torch.Tensor,
) -> torch.Tensor:
    """Project ``queries[bs, H, head_dim]`` onto each head's channel mask.

    Returns ``query_projected[bs, H, label_dim]``.
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

    selection_idx = channel_selection.long().unsqueeze(0).expand(bs, -1, -1)
    gathered = torch.gather(queries, dim=-1, index=selection_idx)
    return gathered * channel_weights.unsqueeze(0)


def compute_token_scores(
    queries: torch.Tensor,
    token_signatures: torch.Tensor,
    written: torch.Tensor,
    channel_selection: torch.Tensor,
    channel_weights: torch.Tensor,
    layer_id: int,
) -> torch.Tensor:
    """Compute per-(batch, token) scalar scores.

    queries:           [bs, num_local_heads, head_dim] (bf16 or fp16)
    token_signatures:  [num_layers_local, max_tokens, num_heads_local, label_dim]
    written:           [num_layers_local, max_tokens] bool
    channel_selection: [num_layers_local, num_heads_local, label_dim] int32
    channel_weights:   [num_layers_local, num_heads_local, label_dim] fp32

    Returns ``token_scores[bs, max_tokens]`` fp32. Unwritten tokens get
    ``-inf`` so the top-K step ignores them deterministically.
    """

    if not (0 <= layer_id < token_signatures.shape[0]):
        raise IndexError(
            f"layer_id={layer_id} out of range [0, {token_signatures.shape[0]})."
        )

    sel_layer = channel_selection[layer_id]    # [H, label_dim]
    w_layer = channel_weights[layer_id]        # [H, label_dim]
    sig_layer = token_signatures[layer_id]     # [T, H, label_dim]
    written_layer = written[layer_id]          # [T]

    q_proj = project_query_onto_channels(queries, sel_layer, w_layer)  # [bs, H, D]

    if (
        _TRITON_AVAILABLE
        and q_proj.is_cuda
        and sig_layer.is_cuda
        and written_layer.is_cuda
    ):
        return _compute_token_scores_triton(
            q_proj.to(torch.float32),
            sig_layer,
            written_layer,
        )

    scores_full = torch.einsum(
        "bhd,thd->bth", q_proj.to(torch.float32), sig_layer.to(torch.float32)
    )
    scores = scores_full.amax(dim=-1)  # [bs, T]
    return scores.masked_fill(~written_layer.unsqueeze(0), float("-inf"))


def all_reduce_token_scores(
    token_scores: torch.Tensor,
    *,
    process_group=None,
) -> torch.Tensor:
    """All-reduce per-rank scalar token scores across the attention TP group.

    The token label signatures stay TP/head-sharded; only the scalar
    per-token scores are reduced (SUM). The reduction operates in-place on
    ``token_scores`` and returns the same tensor for convenience.
    """

    if process_group is None:
        return token_scores
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return token_scores
    torch.distributed.all_reduce(
        token_scores,
        op=torch.distributed.ReduceOp.SUM,
        group=process_group,
    )
    return token_scores


def select_topk_sequence_order(
    token_scores: torch.Tensor,
    max_top_k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Top-K selection returning sequence-order-ascending logical token positions.

    token_scores: [bs, max_tokens] fp32. Caller must have already applied any
                  per-request validity mask (unwritten / out-of-range tokens = -inf).

    Returns:
        selected_indices: int32 [bs, max_top_k], ascending, -1 padded.
        valid_lengths:    int32 [bs].
    """

    if token_scores.dim() != 2:
        raise ValueError(
            f"token_scores must be 2-D [bs, max_tokens], got {tuple(token_scores.shape)}."
        )
    if max_top_k <= 0:
        raise ValueError(f"max_top_k must be positive, got {max_top_k}.")
    bs, max_tokens = token_scores.shape
    device = token_scores.device

    effective_top_k = min(max_top_k, max_tokens)
    topk = torch.topk(
        token_scores, k=effective_top_k, dim=-1, largest=True, sorted=False
    )
    topk_indices = topk.indices  # [bs, effective_top_k]
    topk_scores = topk.values    # [bs, effective_top_k]

    invalid_entries = torch.isneginf(topk_scores)
    topk_indices = torch.where(
        invalid_entries,
        torch.full_like(topk_indices, max_tokens),
        topk_indices,
    )

    sorted_indices, _ = torch.sort(topk_indices, dim=-1)

    selected = torch.full(
        (bs, max_top_k),
        SELECTED_PAD_VALUE,
        dtype=torch.int32,
        device=device,
    )
    valid_mask_real = sorted_indices < max_tokens
    valid_lengths = valid_mask_real.to(torch.int32).sum(dim=-1)

    if effective_top_k > 0:
        position_grid = torch.arange(effective_top_k, device=device)
        keep_positions = position_grid.unsqueeze(0) < valid_lengths.unsqueeze(1)
        real_slice = torch.where(
            keep_positions,
            sorted_indices.to(torch.int32),
            torch.full_like(sorted_indices, SELECTED_PAD_VALUE, dtype=torch.int32),
        )
        selected[:, :effective_top_k] = real_slice

    return selected, valid_lengths.to(torch.int32)


def retrieve_topk_via_labels(
    *,
    queries: torch.Tensor,
    token_signatures: torch.Tensor,
    written: torch.Tensor,
    channel_selection: torch.Tensor,
    channel_weights: torch.Tensor,
    layer_id: int,
    max_top_k: int,
    process_group=None,
    per_request_valid: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """End-to-end selector flow: score → all-reduce → per-request mask → top-K → ascend.

    ``per_request_valid``: optional ``[bs, max_tokens]`` bool/int tensor flagging
    the tokens each request actually owns. Without a per-row mask a request could
    pick globally-written tokens owned by other requests in a mixed batch.
    ``None`` disables the per-row gate (single-request / unit-test paths).

    Returns ``(selected_indices, valid_lengths)`` where ``selected_indices``
    contains logical token positions (sequence-ascending, -1 padded).
    """

    scores = compute_token_scores(
        queries=queries,
        token_signatures=token_signatures,
        written=written,
        channel_selection=channel_selection,
        channel_weights=channel_weights,
        layer_id=layer_id,
    )
    scores = all_reduce_token_scores(scores, process_group=process_group)
    if per_request_valid is not None:
        if per_request_valid.shape != scores.shape:
            raise ValueError(
                f"per_request_valid shape {tuple(per_request_valid.shape)} must "
                f"match token_scores shape {tuple(scores.shape)}."
            )
        scores = scores.masked_fill(~per_request_valid.to(torch.bool), float("-inf"))
    indices, valid_lengths = select_topk_sequence_order(scores, max_top_k)
    if not torch.cuda.is_current_stream_capturing():
        from sglang.srt.layers.attention.double_sparsity import metrics as _metrics

        selected_tokens = int(valid_lengths.sum().item())
        if per_request_valid is not None:
            total_valid_tokens = int(per_request_valid.to(torch.int64).sum().item())
        else:
            total_valid_tokens = int(valid_lengths.shape[0]) * int(scores.shape[-1])
        _metrics.record_selection(
            selected_pages=selected_tokens,
            total_valid_pages=total_valid_tokens,
        )
    return indices, valid_lengths
