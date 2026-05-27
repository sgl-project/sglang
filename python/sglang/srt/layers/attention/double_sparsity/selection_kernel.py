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
    def _logical_score_kernel(
        q_ptr,          # [bs, H, head_dim] fp32
        ch_sel_ptr,     # [H, label_dim] int32 (per-layer slice)
        ch_w_ptr,       # [H, label_dim] fp32 (per-layer slice)
        sig_ptr,        # [T, H, label_dim] fp32 (per-layer slice)
        written_ptr,    # [T] bool (per-layer slice)
        rpi_ptr,        # [bs] int32
        rtt_ptr,        # [num_pools, max_pool_len] int32
        sl_ptr,         # [bs] int32
        out_ptr,        # [bs, max_seq_len] fp32 (pre-allocated)
        num_heads: tl.constexpr,
        max_seq_len: tl.constexpr,
        label_dim: tl.constexpr,
        max_pool_len: tl.constexpr,
        max_tokens: tl.constexpr,
        q_stride_b: tl.constexpr,
        q_stride_h: tl.constexpr,
        ch_sel_stride_h: tl.constexpr,
        ch_w_stride_h: tl.constexpr,
        sig_stride_t: tl.constexpr,
        sig_stride_h: tl.constexpr,
        rtt_stride_p: tl.constexpr,
        out_stride_b: tl.constexpr,
        TOKEN_BLOCK: tl.constexpr,
        LABEL_DIM_POW2: tl.constexpr,
    ):
        batch_id = tl.program_id(0)
        tok_blk = tl.program_id(1)
        tok_offs = tok_blk * TOKEN_BLOCK + tl.arange(0, TOKEN_BLOCK)
        in_range = tok_offs < max_seq_len

        seq_len_i = tl.load(sl_ptr + batch_id).to(tl.int32)
        pos_valid = in_range & (tok_offs < seq_len_i)

        pool_idx = tl.load(rpi_ptr + batch_id).to(tl.int64)
        safe_tok = tl.minimum(tok_offs, max_pool_len - 1)
        phys = tl.load(
            rtt_ptr + pool_idx * rtt_stride_p + safe_tok,
            mask=in_range,
            other=0,
        ).to(tl.int64)
        safe_phys = tl.minimum(tl.maximum(phys, 0), max_tokens - 1)

        written = tl.load(
            written_ptr + safe_phys, mask=in_range, other=0
        ).to(tl.int1)
        valid = pos_valid & written

        d_offs = tl.arange(0, LABEL_DIM_POW2)
        d_mask = d_offs < label_dim

        max_score = tl.full((TOKEN_BLOCK,), float("-inf"), dtype=tl.float32)
        for h in range(num_heads):
            sel_h = tl.load(
                ch_sel_ptr + h * ch_sel_stride_h + d_offs,
                mask=d_mask,
                other=0,
            ).to(tl.int64)
            w_h = tl.load(
                ch_w_ptr + h * ch_w_stride_h + d_offs,
                mask=d_mask,
                other=0.0,
            ).to(tl.float32)
            q_base = q_ptr + batch_id * q_stride_b + h * q_stride_h
            q_h = tl.load(q_base + sel_h, mask=d_mask, other=0.0).to(tl.float32)
            q_proj_h = q_h * w_h

            sig_offs = (
                safe_phys[:, None] * sig_stride_t
                + h * sig_stride_h
                + d_offs[None, :]
            )
            sig_block = tl.load(
                sig_ptr + sig_offs,
                mask=in_range[:, None] & d_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            dot = tl.sum(q_proj_h[None, :] * sig_block, axis=1)
            max_score = tl.where(dot > max_score, dot, max_score)

        out_score = tl.where(
            valid,
            max_score,
            tl.full(max_score.shape, float("-inf"), dtype=tl.float32),
        )
        tl.store(out_ptr + batch_id * out_stride_b + tok_offs, out_score, mask=in_range)


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


def _compute_logical_token_scores(
    queries: torch.Tensor,
    token_signatures: torch.Tensor,
    written: torch.Tensor,
    channel_selection: torch.Tensor,
    channel_weights: torch.Tensor,
    layer_id: int,
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    seq_lens: torch.Tensor,
    max_seq_len: int = 0,
) -> torch.Tensor:
    """Score tokens in logical-sequence-position space.

    Gathers physical KV-cache labels for each request's logical positions
    0..seq_len-1 via ``req_to_token``, then scores each logical position
    against the projected query. Returns ``[bs, max_seq_len]`` fp32 scores,
    masked to ``-inf`` for positions >= seq_len and for unwritten slots.

    This keeps the top-K output in logical-position domain so that
    ``logical_to_physical`` can convert it correctly.

    ``max_seq_len`` must be a static Python int when called inside a
    ``torch.cuda.graph`` capture region. Providing it skips the
    ``seq_lens.max().item()`` host sync that would raise
    ``CUDA error: operation not permitted when stream is capturing``.
    """
    bs = queries.shape[0]
    if max_seq_len <= 0:
        max_seq_len = int(seq_lens.max().item()) if bs > 0 else 0
    device = queries.device

    if max_seq_len == 0:
        return torch.full((bs, 1), float("-inf"), dtype=torch.float32, device=device)

    sel_layer = channel_selection[layer_id]  # [H, label_dim]
    w_layer = channel_weights[layer_id]      # [H, label_dim]
    q_proj = project_query_onto_channels(queries, sel_layer, w_layer)  # [bs, H, D]

    num_pools = req_to_token.shape[0]
    max_seqlen_in_pool = req_to_token.shape[1]
    safe_pool = req_pool_indices.clamp(0, max(num_pools - 1, 0)).long()  # [bs]

    # logical_positions[b, i] = i (0-indexed position within each request)
    logical_positions = torch.arange(max_seq_len, device=device).unsqueeze(0).expand(bs, -1)  # [bs, max_seq_len]
    safe_positions = logical_positions.clamp(0, max(max_seqlen_in_pool - 1, 0))  # [bs, max_seq_len]

    # physical_slots[b, i] = req_to_token[safe_pool[b], safe_positions[b, i]]
    pool_expanded = safe_pool.unsqueeze(1).expand(-1, max_seq_len)         # [bs, max_seq_len]
    physical_slots = req_to_token[pool_expanded, safe_positions.long()]    # [bs, max_seq_len] int32

    # Gather label signatures for each logical position's physical slot.
    sig_layer = token_signatures[layer_id]  # [max_tokens, H, label_dim]
    max_tokens = sig_layer.shape[0]
    safe_phys = physical_slots.long().clamp(0, max(max_tokens - 1, 0))     # [bs, max_seq_len]
    gathered_sig = sig_layer[safe_phys]                                     # [bs, max_seq_len, H, label_dim]

    # scores[b, i] = max_over_heads(q_proj[b] · sig[b, i])
    # q_proj: [bs, H, D] → [bs, 1, H, D]; gathered_sig: [bs, max_seq_len, H, D]
    dot = (q_proj.unsqueeze(1).to(torch.float32) * gathered_sig.to(torch.float32)).sum(-1)  # [bs, max_seq_len, H]
    scores = dot.amax(dim=-1)  # [bs, max_seq_len]

    # Mask: unwritten physical slots and positions >= seq_len
    written_layer = written[layer_id]  # [max_tokens] bool
    written_gathered = written_layer[safe_phys]  # [bs, max_seq_len] bool
    scores = scores.masked_fill(~written_gathered, float("-inf"))

    seq_len_mask = logical_positions < seq_lens.unsqueeze(1).to(device)  # [bs, max_seq_len]
    scores = scores.masked_fill(~seq_len_mask, float("-inf"))

    return scores


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
    req_pool_indices: Optional[torch.Tensor] = None,
    req_to_token: Optional[torch.Tensor] = None,
    seq_lens: Optional[torch.Tensor] = None,
    max_seq_len: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """End-to-end selector flow: score → all-reduce → per-request mask → top-K → ascend.

    **Logical-domain mode** (when ``req_pool_indices``, ``req_to_token``, and
    ``seq_lens`` are all provided): gathers physical labels per-request using
    ``req_to_token``, scores in the ``[bs, max_seq_len]`` logical-position
    domain, and returns logical positions (0-indexed). This is the correct
    production path — ``logical_to_physical`` can then map the returned
    positions to physical KV slots.

    **Physical-domain mode** (when those three are absent, default): scores
    over all physical slots in ``token_signatures`` directly. Used by the
    sanity probe and unit tests that construct labels at known physical slot
    indices without a per-request ``req_to_token`` mapping.

    ``per_request_valid``: optional ``[bs, max_tokens/max_seq_len]`` bool mask
    applied after scoring. ``None`` disables the gate.

    Returns ``(selected_indices, valid_lengths)`` — sequence-ascending, -1 padded.
    """

    use_logical = (
        req_pool_indices is not None
        and req_to_token is not None
        and seq_lens is not None
    )

    if use_logical:
        scores = _compute_logical_token_scores(
            queries=queries,
            token_signatures=token_signatures,
            written=written,
            channel_selection=channel_selection,
            channel_weights=channel_weights,
            layer_id=layer_id,
            req_pool_indices=req_pool_indices,
            req_to_token=req_to_token,
            seq_lens=seq_lens,
            max_seq_len=max_seq_len,
        )
        scores = all_reduce_token_scores(scores, process_group=process_group)
    else:
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
            selected_tokens=selected_tokens,
            total_valid_tokens=total_valid_tokens,
        )
    return indices, valid_lengths


def _logical_score_triton(
    q_proj_input: torch.Tensor,         # [bs, H, head_dim] fp32 (the raw queries; gather via ch_sel inside)
    channel_selection_layer: torch.Tensor,  # [H, label_dim] int32
    channel_weights_layer: torch.Tensor,    # [H, label_dim] fp32
    sig_layer: torch.Tensor,            # [T, H, label_dim] fp32 (cast outside if not fp32)
    written_layer: torch.Tensor,        # [T] bool
    req_pool_indices: torch.Tensor,     # [bs] int32
    req_to_token: torch.Tensor,         # [num_pools, max_pool_len] int32
    seq_lens: torch.Tensor,             # [bs] int32
    out: torch.Tensor,                  # [bs_buf, max_seq_len] fp32 (pre-allocated, slice [bs])
    max_seq_len: int,
    *,
    token_block: int = 64,
) -> None:
    """Fill ``out[:bs, :max_seq_len]`` with per-(batch, logical-position) scores.

    All allocations happen in the caller. No `.item()` host syncs.
    """
    bs, num_heads, _head_dim = q_proj_input.shape
    label_dim = int(channel_selection_layer.shape[1])
    max_pool_len = int(req_to_token.shape[1])
    max_tokens = int(sig_layer.shape[0])

    desired_block = min(token_block, max(max_seq_len, 1))
    token_block_pow2 = _next_pow2(desired_block)
    label_dim_pow2 = _next_pow2(max(label_dim, 1))
    num_token_blocks = (max_seq_len + token_block_pow2 - 1) // token_block_pow2
    grid = (bs, num_token_blocks)

    _logical_score_kernel[grid](
        q_proj_input,
        channel_selection_layer,
        channel_weights_layer,
        sig_layer,
        written_layer,
        req_pool_indices,
        req_to_token,
        seq_lens,
        out,
        num_heads=num_heads,
        max_seq_len=max_seq_len,
        label_dim=label_dim,
        max_pool_len=max_pool_len,
        max_tokens=max_tokens,
        q_stride_b=q_proj_input.stride(0),
        q_stride_h=q_proj_input.stride(1),
        ch_sel_stride_h=channel_selection_layer.stride(0),
        ch_w_stride_h=channel_weights_layer.stride(0),
        sig_stride_t=sig_layer.stride(0),
        sig_stride_h=sig_layer.stride(1),
        rtt_stride_p=req_to_token.stride(0),
        out_stride_b=out.stride(0),
        TOKEN_BLOCK=token_block_pow2,
        LABEL_DIM_POW2=label_dim_pow2,
    )


def retrieve_topk_graph_safe(
    *,
    queries: torch.Tensor,
    token_signatures: torch.Tensor,
    written: torch.Tensor,
    channel_selection: torch.Tensor,
    channel_weights: torch.Tensor,
    layer_id: int,
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    seq_lens: torch.Tensor,
    max_seq_len: int,
    max_top_k: int,
    out_indices: torch.Tensor,
    out_lengths: torch.Tensor,
    # Pre-allocated scratch tensors (required for CUDA / Triton fast path)
    scratch_scores: Optional[torch.Tensor] = None,         # fp32 [max_bs, max_seq_len]
    scratch_topk_values: Optional[torch.Tensor] = None,    # fp32 [max_bs, max_top_k]
    scratch_topk_indices: Optional[torch.Tensor] = None,   # int64 [max_bs, max_top_k]
    scratch_invalid_mask: Optional[torch.Tensor] = None,   # bool [max_bs, max_top_k]
    scratch_sorted_vals: Optional[torch.Tensor] = None,    # int64 [max_bs, max_top_k]
    scratch_boundary: Optional[torch.Tensor] = None,       # int64 [max_bs, 1] = max_seq_len
    scratch_valid_i64: Optional[torch.Tensor] = None,      # int64 [max_bs, 1]
    per_request_valid: Optional[torch.Tensor] = None,      # bool [bs, max_seq_len]
    scratch_pv_mask: Optional[torch.Tensor] = None,        # bool [max_bs, max_seq_len]
    scratch_throwaway_idx: Optional[torch.Tensor] = None,  # int64 [max_bs, max_top_k]
    process_group=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Capture-safe retrieve_topk that writes results into caller-owned buffers.

    On CUDA with Triton + all scratch buffers provided: uses an allocation-free
    pipeline.  After a single warmup call, subsequent calls perform zero new
    CUDA allocations:

        1. ``_logical_score_kernel`` fills ``scratch_scores`` directly.
        2. (optional) ``per_request_valid`` is applied via in-place masked_fill_.
        3. ``topk`` with ``out=(values, indices)`` (allocation-free after warmup).
        4. ``isneginf`` + ``masked_fill_`` to sentinel-out invalid entries.
        5. ``topk(largest=False, sorted=True)`` for an allocation-free ascending sort.
        6. ``ge`` + ``masked_fill_`` to convert sentinels to ``-1`` in output.
        7. ``searchsorted`` with ``out=`` for valid_lengths.

    Fallback path (CPU, or scratch tensors missing): calls the legacy
    :func:`retrieve_topk_via_labels`.  This branch is intended for unit tests;
    do NOT route production graph capture through it.
    """
    bs = req_pool_indices.shape[0]
    device = queries.device
    use_triton_fast = (
        _TRITON_AVAILABLE
        and device.type == "cuda"
        and scratch_scores is not None
        and scratch_topk_values is not None
        and scratch_topk_indices is not None
        and scratch_invalid_mask is not None
        and scratch_sorted_vals is not None
        and scratch_boundary is not None
        and scratch_valid_i64 is not None
        and scratch_throwaway_idx is not None
    )

    if not use_triton_fast:
        indices, valid = retrieve_topk_via_labels(
            queries=queries,
            token_signatures=token_signatures,
            written=written,
            channel_selection=channel_selection,
            channel_weights=channel_weights,
            layer_id=layer_id,
            max_top_k=max_top_k,
            process_group=process_group,
            per_request_valid=per_request_valid,
            req_pool_indices=req_pool_indices,
            req_to_token=req_to_token,
            seq_lens=seq_lens,
            max_seq_len=max_seq_len,
        )
        mtk = indices.shape[1]
        out_indices[:bs, :mtk].copy_(indices)
        out_lengths[:bs].copy_(valid)
        return out_indices, out_lengths

    # Triton fast path — zero-allocation after warmup.
    # Contract (caller responsibility — bind_runtime_data enforces it for the
    # channel-mask tensors): channel_selection int32, channel_weights fp32,
    # req_pool_indices / req_to_token / seq_lens int32. queries and sig_layer
    # may be fp32 / fp16 / bf16 — the kernel casts via tl.load(...).to(tl.float32).
    sel_layer = channel_selection[layer_id]
    w_layer = channel_weights[layer_id]
    sig_layer = token_signatures[layer_id]
    written_layer = written[layer_id]
    assert sel_layer.dtype == torch.int32, (
        f"channel_selection must be int32, got {sel_layer.dtype}"
    )
    assert w_layer.dtype == torch.float32, (
        f"channel_weights must be float32, got {w_layer.dtype}"
    )
    assert req_pool_indices.dtype == torch.int32, (
        f"req_pool_indices must be int32, got {req_pool_indices.dtype}"
    )
    assert req_to_token.dtype == torch.int32, (
        f"req_to_token must be int32, got {req_to_token.dtype}"
    )
    assert seq_lens.dtype == torch.int32, (
        f"seq_lens must be int32, got {seq_lens.dtype}"
    )

    scores_view = scratch_scores[:bs, :max_seq_len]
    _logical_score_triton(
        q_proj_input=queries,
        channel_selection_layer=sel_layer,
        channel_weights_layer=w_layer,
        sig_layer=sig_layer,
        written_layer=written_layer,
        req_pool_indices=req_pool_indices,
        req_to_token=req_to_token,
        seq_lens=seq_lens,
        out=scores_view,
        max_seq_len=max_seq_len,
    )

    if process_group is not None and torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(
            scores_view, op=torch.distributed.ReduceOp.SUM, group=process_group
        )

    if per_request_valid is not None:
        assert scratch_pv_mask is not None, (
            "per_request_valid requires scratch_pv_mask in graph-safe path"
        )
        pv_view = scratch_pv_mask[:bs, :max_seq_len]
        # copy_ handles dtype conversion in-place (no allocation when shapes match).
        pv_view.copy_(per_request_valid)
        # In-place flip: True = valid → True = invalid; then masked_fill_(invalid, -inf).
        torch.logical_not(pv_view, out=pv_view)
        scores_view.masked_fill_(pv_view, float("-inf"))

    effective_k = min(max_top_k, max_seq_len)
    topk_vals_view = scratch_topk_values[:bs, :effective_k]
    topk_idx_view = scratch_topk_indices[:bs, :effective_k]
    invalid_view = scratch_invalid_mask[:bs, :effective_k]
    sorted_vals_view = scratch_sorted_vals[:bs, :effective_k]
    boundary_view = scratch_boundary[:bs]
    valid_i64_view = scratch_valid_i64[:bs]

    # Step 1: top-K by score (unsorted, largest).
    torch.topk(
        scores_view,
        effective_k,
        dim=-1,
        largest=True,
        sorted=False,
        out=(topk_vals_view, topk_idx_view),
    )

    # Step 2: sentinel-out invalid (-inf) entries; replace their position with max_seq_len.
    torch.isneginf(topk_vals_view, out=invalid_view)
    topk_idx_view.masked_fill_(invalid_view, max_seq_len)

    # Step 3: ascending sort using topk(largest=False, sorted=True).
    # PyTorch's topk requires output indices NOT to alias input — aliasing
    # corrupts the read (observed: input [3, 1] → output values [0, 1]).
    # Route throwaway gather indices into a dedicated scratch.
    assert scratch_throwaway_idx is not None, (
        "scratch_throwaway_idx is required for the graph-safe topk pipeline"
    )
    throwaway_view = scratch_throwaway_idx[:bs, :effective_k]
    torch.topk(
        topk_idx_view,
        effective_k,
        dim=-1,
        largest=False,
        sorted=True,
        out=(sorted_vals_view, throwaway_view),
    )

    # Step 4: copy sorted positions to int32 output, then sentinel → -1.
    out_indices[:bs, :effective_k].copy_(sorted_vals_view)
    torch.ge(sorted_vals_view, max_seq_len, out=invalid_view)
    out_indices[:bs, :effective_k].masked_fill_(invalid_view, -1)
    if effective_k < out_indices.shape[1]:
        out_indices[:bs, effective_k:].fill_(-1)

    # Step 5: count valid (< max_seq_len) entries via searchsorted on the sorted vector.
    boundary_view.fill_(max_seq_len)
    torch.searchsorted(
        sorted_vals_view, boundary_view, right=False, out=valid_i64_view
    )
    out_lengths[:bs].copy_(valid_i64_view.squeeze(-1))

    return out_indices, out_lengths


# Public alias for the end-to-end selector pipeline.
retrieve_topk = retrieve_topk_via_labels
