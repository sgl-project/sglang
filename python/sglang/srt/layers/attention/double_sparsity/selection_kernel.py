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
        sig_ptr,        # [T, H, label_dim] fp16/fp32/int8 (per-layer slice)
        scale_ptr,      # [T, H] fp16 (compact int8 path) or unused when HAS_SCALE is False
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
        scale_stride_t: tl.constexpr,
        scale_stride_h: tl.constexpr,
        HAS_SCALE: tl.constexpr,
        rtt_stride_p: tl.constexpr,
        out_stride_b: tl.constexpr,
        TOKEN_BLOCK: tl.constexpr,
        LABEL_DIM_POW2: tl.constexpr,
        SCORER_NORM: tl.constexpr,       # 0=off(raw), 1=cosine, 2=hybrid
        HEAD_AGG_MEAN: tl.constexpr,     # bool: True=mean over heads, False=max
        HYBRID_THRESHOLD: tl.constexpr,  # int: hybrid uses cosine when seq_len > this
    ):
        batch_id = tl.program_id(0)
        tok_blk = tl.program_id(1)
        tok_offs = tok_blk * TOKEN_BLOCK + tl.arange(0, TOKEN_BLOCK)
        in_range = tok_offs < max_seq_len

        seq_len_i = tl.load(sl_ptr + batch_id).to(tl.int32)
        # Skip token-blocks entirely past this request's sequence length: every
        # position would be masked to -inf anyway, so store -inf and return
        # instead of running the per-head signature loads + dot products for the
        # unused tail. The score scratch / topk operate over the full KV-index
        # width (req_to_token width == model context length), so without this a
        # short request scores the entire context every layer every decode step.
        # Output is bit-identical to the masked full scan; the launch grid is
        # unchanged so it stays CUDA-graph capture/replay safe (no host sync,
        # no dynamic shape).
        if tok_blk * TOKEN_BLOCK >= seq_len_i:
            tl.store(
                out_ptr + batch_id * out_stride_b + tok_offs,
                tl.full((TOKEN_BLOCK,), float("-inf"), dtype=tl.float32),
                mask=in_range,
            )
            return
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
        eps = 1e-6
        # Per-request cosine decision for hybrid: cosine above the length
        # threshold, raw channel-dot at/below it (matches the eager
        # _compute_logical_token_scores length-conditional switch). Scalar.
        hybrid_cos = seq_len_i > HYBRID_THRESHOLD

        # Cross-head accumulator: 0 for mean (sum then divide), -inf for max.
        if HEAD_AGG_MEAN:
            acc = tl.zeros((TOKEN_BLOCK,), dtype=tl.float32)
        else:
            acc = tl.full((TOKEN_BLOCK,), float("-inf"), dtype=tl.float32)

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

            if SCORER_NORM == 0:
                # Raw channel-dot (production), dequant-scaled for the int8 path
                # (scale >= 0 preserves ordering).
                if HAS_SCALE:
                    scale_h = tl.load(
                        scale_ptr + safe_phys * scale_stride_t + h * scale_stride_h,
                        mask=in_range,
                        other=0.0,
                    ).to(tl.float32)
                    score_h = dot * scale_h
                else:
                    score_h = dot
            else:
                # Cosine (direction-only): unit-normalize the weighted query and
                # the token signature per head. Equal to the eager
                # ((qf/||qf||)*(sf/||sf||)).sum form (normalize-then-sum); scale
                # is intentionally ignored (it cancels under normalization).
                q_norm = tl.sqrt(tl.sum(q_proj_h * q_proj_h)) + eps
                sig_norm = tl.sqrt(tl.sum(sig_block * sig_block, axis=1)) + eps
                cos = tl.sum(
                    (q_proj_h[None, :] / q_norm) * (sig_block / sig_norm[:, None]),
                    axis=1,
                )
                if SCORER_NORM == 1:
                    score_h = cos
                else:
                    # Hybrid: cosine above the threshold, raw (scaled) below.
                    if HAS_SCALE:
                        scale_h = tl.load(
                            scale_ptr + safe_phys * scale_stride_t + h * scale_stride_h,
                            mask=in_range,
                            other=0.0,
                        ).to(tl.float32)
                        raw_h = dot * scale_h
                    else:
                        raw_h = dot
                    score_h = tl.where(hybrid_cos, cos, raw_h)

            if HEAD_AGG_MEAN:
                acc += score_h
            else:
                acc = tl.where(score_h > acc, score_h, acc)

        if HEAD_AGG_MEAN:
            acc = acc / num_heads

        out_score = tl.where(
            valid,
            acc,
            tl.full(acc.shape, float("-inf"), dtype=tl.float32),
        )
        tl.store(out_ptr + batch_id * out_stride_b + tok_offs, out_score, mask=in_range)


    @triton.jit
    def _compute_token_scores_kernel(
        q_proj_ptr,  # [bs, H, label_dim] fp32
        sig_ptr,     # [T, H, label_dim] fp16/fp32/int8
        written_ptr, # [T] bool
        out_ptr,     # [bs, T] fp32
        scale_ptr,   # [T, H] fp16 (compact int8 path) or unused when HAS_SCALE is False
        bs: tl.constexpr,
        num_heads: tl.constexpr,
        max_tokens: tl.constexpr,
        label_dim: tl.constexpr,
        q_stride_b: tl.constexpr,
        q_stride_h: tl.constexpr,
        sig_stride_t: tl.constexpr,
        sig_stride_h: tl.constexpr,
        out_stride_b: tl.constexpr,
        scale_stride_t: tl.constexpr,
        scale_stride_h: tl.constexpr,
        HAS_SCALE: tl.constexpr,
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
            if HAS_SCALE:
                # Dequant: multiply the int8 dot by the per-(token, head) scale
                # before the cross-head max (scale >= 0 preserves ordering).
                scale_h = tl.load(
                    scale_ptr + token_offsets * scale_stride_t + h * scale_stride_h,
                    mask=token_in_range,
                    other=0.0,
                ).to(tl.float32)
                dot = dot * scale_h
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
    scale_layer: Optional[torch.Tensor] = None,
    token_block: int = 64,
) -> torch.Tensor:
    """Triton kernel-driven token scoring.

    Args:
        q_proj: ``[bs, H, label_dim]`` fp32.
        sig_layer: ``[max_tokens, H, label_dim]`` fp16/fp32/int8.
        written_layer: ``[max_tokens]`` bool.
        scale_layer: optional ``[max_tokens, H]`` per-(slot, head) dequant scale
            for the int8 compact path; ``None`` keeps the fp16 path.

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
    has_scale = scale_layer is not None
    scale_c = scale_layer.contiguous() if has_scale else sig_c
    scale_stride_t = scale_c.stride(0) if has_scale else 0
    scale_stride_h = scale_c.stride(1) if has_scale else 0

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
        scale_c,
        bs=bs,
        num_heads=num_heads,
        max_tokens=max_tokens,
        label_dim=label_dim,
        q_stride_b=q_proj_c.stride(0),
        q_stride_h=q_proj_c.stride(1),
        sig_stride_t=sig_c.stride(0),
        sig_stride_h=sig_c.stride(1),
        out_stride_b=out.stride(0),
        scale_stride_t=scale_stride_t,
        scale_stride_h=scale_stride_h,
        HAS_SCALE=has_scale,
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


def _scorer_norm_mode() -> str:
    """Flag-gated DS scorer normalization (Loop-7 Tier-2.B candidate).

    ``SGLANG_DS_SCORER_NORM``:
      - ``"off"`` (default): the production raw channel-dot scorer, byte-identical.
      - ``"cosine"``: unit-normalize the query projection and each token signature
        per head before the dot, so the score is direction-only (magnitude
        invariant). Rationale (M0 oracle): at 16K the needle ranks ~= its
        position, i.e. per-token background magnitude dominates the raw dot — a
        cosine score removes that bias so a salient needle can outrank bulk
        filler. Scale-invariant, so the int8 dequant scale cancels (ignored).
    """
    import os as _os

    mode = _os.environ.get("SGLANG_DS_SCORER_NORM", "off").strip().lower()
    return mode if mode in ("off", "cosine", "hybrid") else "off"


def ds_scorer_is_default(config) -> bool:
    """``True`` iff the DS selector config uses the production raw channel-dot /
    cross-head-max scorer with no variant flags — i.e. the graph-safe Triton
    scorer path is valid. Any non-default variant (scorer_norm/head_agg/
    anchor_budget) must run the eager logical scorer instead, since the graph-
    safe Triton scorer only implements the production path.
    """
    if config is None:
        return True
    return (
        getattr(config, "scorer_norm", "off") == "off"
        and getattr(config, "head_agg", "max") == "max"
        and getattr(config, "anchor_mode", "off") == "off"
    )


def ds_scorer_is_graph_safe(config) -> bool:
    """``True`` iff the configured selector variants are all on the graph-safe
    path, so the selector can run under CUDA-graph capture.

    As of R9 ALL non-learned variants are graph-safe: ``scorer_norm``
    (cosine/hybrid) + ``head_agg`` (mean) live in ``_logical_score_kernel`` (R6),
    and ``anchor_mode`` (recency/global/strided) is a tensorized fixed-shape
    post-topK force-include in ``retrieve_topk_graph_safe`` (R9). None require
    ``--disable-cuda-graph``. (The ``recall_oracle`` diagnostic is gated
    separately by ``ds_recall_oracle_enabled``.) Retained as the single guard
    predicate so a future non-graph-safe variant can re-introduce a gate here.
    """
    return True


def ds_recall_oracle_enabled(config) -> bool:
    """``True`` iff the config-borne recall-oracle diagnostic is on.

    Config-borne (not env) so it reaches TP worker subprocesses
    (BL-20260602-ds-flag-must-be-config-borne-not-env). Like a non-default
    scorer it forces the eager selector path so the host-syncing oracle hook
    actually re-runs every decode step (under CUDA-graph replay the Python does
    not re-run); the validator additionally requires ``--disable-cuda-graph``.
    """
    if config is None:
        return False
    return bool(getattr(config, "recall_oracle", False))


def ds_lifted_budget_decode_available() -> bool:
    """``True`` iff the opt-in adjustable-budget (lifted) decode backend path is
    implemented and wired into selection/decode, so ``enable_lifted_budget_decode``
    can actually be honored.

    Returns ``False`` today: the config ABI (``enable_lifted_budget_decode`` +
    ``lifted_budget_top_k``) is recognized, but the lifted decode path
    (wider-than-``index_topk`` selection → request-local compact remap →
    ``flash_mla_sparse_fwd``) is not built yet. The validator uses this so a
    boot with the flag set fails closed instead of silently running the locked
    ``index_topk`` selector or routing a wider selection into the default
    ``flashmla_kv`` ``indices.shape[-1] == dsa_index_topk`` assert. This is the
    single capability seam the lifted-budget decode landing flips to ``True``
    once that path exists (mirroring :func:`ds_scorer_is_graph_safe`).
    """
    return False


def compute_token_scores(
    queries: torch.Tensor,
    token_signatures: torch.Tensor,
    written: torch.Tensor,
    channel_selection: torch.Tensor,
    channel_weights: torch.Tensor,
    layer_id: int,
    token_scales: Optional[torch.Tensor] = None,
    scorer_norm: Optional[str] = None,
    head_agg: str = "max",
) -> torch.Tensor:
    """Compute per-(batch, token) scalar scores.

    queries:           [bs, num_local_heads, head_dim] (bf16 or fp16)
    token_signatures:  [num_layers_local, max_tokens, num_heads_local, label_dim]
    written:           [num_layers_local, max_tokens] bool
    channel_selection: [num_layers_local, num_heads_local, label_dim] int32
    channel_weights:   [num_layers_local, num_heads_local, label_dim] fp32
    token_scales:      optional [num_layers_local, max_tokens, num_heads_local]
                       per-(slot, head) dequant scale for the int8 compact path.

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
    scale_layer = token_scales[layer_id] if token_scales is not None else None  # [T, H]

    q_proj = project_query_onto_channels(queries, sel_layer, w_layer)  # [bs, H, D]

    norm_mode = scorer_norm if scorer_norm is not None else _scorer_norm_mode()
    # The physical scorer has no per-request seq_len, so "hybrid" (which is
    # length-conditional) cannot be applied correctly here and must NOT silently
    # degrade to cosine (that is exactly the moderate-context regression hybrid
    # avoids). Reject it; the logical scorer is the hybrid-capable path.
    if norm_mode == "hybrid":
        raise ValueError(
            "Double Sparsity scorer_norm='hybrid' is length-conditional and "
            "requires per-request seq_len; it is only valid on the logical "
            "scoring path (_compute_logical_token_scores), not the physical "
            "compute_token_scores path."
        )
    cosine_like = norm_mode == "cosine"

    if (
        not cosine_like
        and head_agg == "max"
        and _TRITON_AVAILABLE
        and q_proj.is_cuda
        and sig_layer.is_cuda
        and written_layer.is_cuda
    ):
        return _compute_token_scores_triton(
            q_proj.to(torch.float32),
            sig_layer,
            written_layer,
            scale_layer=scale_layer,
        )

    qf = q_proj.to(torch.float32)
    sf = sig_layer.to(torch.float32)
    if cosine_like:
        # Direction-only score: unit-normalize per (head) channel vector. The
        # int8 dequant scale is a positive per-(token,head) magnitude factor and
        # cancels under normalization, so scale_layer is intentionally ignored.
        eps = 1e-6
        qf = qf / (qf.norm(dim=-1, keepdim=True) + eps)
        sf = sf / (sf.norm(dim=-1, keepdim=True) + eps)
        scores_full = torch.einsum("bhd,thd->bth", qf, sf)  # [bs, T, H]
    else:
        scores_full = torch.einsum("bhd,thd->bth", qf, sf)  # [bs, T, H]
        if scale_layer is not None:
            # Dequant the int8 dot per (token, head) before the cross-head agg.
            scores_full = scores_full * scale_layer.unsqueeze(0).to(torch.float32)
    scores = scores_full.mean(dim=-1) if head_agg == "mean" else scores_full.amax(dim=-1)
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


def _topk_by_score_then_pos(
    vals: torch.Tensor,
    pos: torch.Tensor,
    k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Top-``k`` candidates by (value DESCENDING, then position ASCENDING) — the one
    deterministic tie-break all DS top-k selectors share.

    ``vals``/``pos`` are ``[bs, m]`` (``pos`` are the distinct logical positions of the
    candidates). Done as two stable passes: (1) order candidates by position ascending,
    then (2) stable argsort by value descending — so equal values resolve toward the
    lower position. Returns ``(top_positions [bs, k] int64, top_vals [bs, k])``. Uses
    fresh argsort outputs (no in/out aliasing — BL-20260527-torch-topk-aliasing).
    """
    pos_order = torch.argsort(pos, dim=-1, stable=True)            # ascending position
    pos_a = torch.gather(pos, 1, pos_order)
    vals_a = torch.gather(vals, 1, pos_order)
    val_order = torch.argsort(vals_a, dim=-1, descending=True, stable=True)[:, :k]
    return torch.gather(pos_a, 1, val_order), torch.gather(vals_a, 1, val_order)


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
    # Deterministic tie-break: select by (score DESCENDING, then logical position
    # ASCENDING). token_scores columns are already in position order, so the shared
    # helper's stable score-descending argsort breaks score ties toward the lower
    # position -- the single ordering all DS top-k selectors honor (see
    # _topk_by_score_then_pos / blocked_topk_sequence_order).
    positions = torch.arange(max_tokens, device=device, dtype=torch.int64).unsqueeze(0).expand(bs, -1)
    topk_indices, topk_scores = _topk_by_score_then_pos(token_scores, positions, effective_top_k)

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


def blocked_topk_sequence_order(
    token_scores: torch.Tensor,
    max_top_k: int,
    block_width: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Exact blocked top-K — identical output to :func:`select_topk_sequence_order`.

    Partitions the ``max_tokens`` score axis into fixed ``block_width`` blocks,
    keeps each block's top ``min(max_top_k, block_width)`` candidates, then merges
    them and takes the global top-K. The result (ascending logical positions,
    ``-1`` padded, + valid_lengths) is identical to the monolithic selection.

    Exactness: a token in the global top-K has within-block rank <= its global
    rank <= K, so it is in its block's top-min(K, block_width); the union of the
    per-block candidates therefore contains the global top-K, and merging them under
    the SHARED deterministic tie-break (score descending, then logical position
    ascending) reproduces ``select_topk_sequence_order`` EXACTLY -- including on
    finite ties (equal scores resolve to the lower position in both). This is the
    exactness ORACLE and the
    eager fallback for the graph-safe blocked top-k, whose value is that it can
    SKIP blocks entirely past each request's ``seq_len`` (every such block is all
    ``-inf`` and contributes no candidate), shrinking the per-decode-step work
    versus a monolithic ``torch.topk`` over the full KV-index width.

    token_scores: [bs, max_tokens] fp32 (unwritten/out-of-range tokens = -inf).
    Returns: selected_indices int32 [bs, max_top_k] ascending -1-padded; valid_lengths int32 [bs].
    """
    if token_scores.dim() != 2:
        raise ValueError(f"token_scores must be 2-D, got {tuple(token_scores.shape)}.")
    if max_top_k <= 0:
        raise ValueError(f"max_top_k must be positive, got {max_top_k}.")
    if block_width <= 0:
        raise ValueError(f"block_width must be positive, got {block_width}.")
    bs, max_tokens = token_scores.shape
    device = token_scores.device
    K = min(max_top_k, max_tokens)
    bw = block_width
    nb = (max_tokens + bw - 1) // bw
    pad = nb * bw - max_tokens
    if pad:
        sc = token_scores.new_full((bs, nb * bw), float("-inf"))
        sc[:, :max_tokens] = token_scores
    else:
        sc = token_scores
    blk = sc.view(bs, nb, bw)
    kb = min(K, bw)
    # per-block top-kb candidate LOCAL positions by (score desc, local-pos asc) -- the
    # argsort indices ARE the local positions; stable keeps ascending pos on ties.
    blk_order = torch.argsort(blk, dim=-1, descending=True, stable=True)[:, :, :kb]
    block_base = (torch.arange(nb, device=device, dtype=torch.int64) * bw).view(1, nb, 1)
    cand_pos = (block_base + blk_order).reshape(bs, nb * kb)
    cand_vals = torch.gather(blk, 2, blk_order).reshape(bs, nb * kb)
    # global top-K over the union, by the SHARED (score desc, position asc) contract
    # -- identical to select_topk_sequence_order, including on finite ties.
    eff = min(K, nb * kb)
    sel_pos, merge_vals = _topk_by_score_then_pos(cand_vals, cand_pos, eff)
    invalid = torch.isneginf(merge_vals)
    sel_pos = torch.where(invalid, torch.full_like(sel_pos, max_tokens), sel_pos)
    sorted_pos, _ = torch.sort(sel_pos, dim=-1)
    selected = torch.full((bs, max_top_k), SELECTED_PAD_VALUE, dtype=torch.int32, device=device)
    valid_lengths = (sorted_pos < max_tokens).to(torch.int32).sum(dim=-1)
    grid = torch.arange(eff, device=device)
    keep = grid.unsqueeze(0) < valid_lengths.unsqueeze(1)
    real = torch.where(
        keep, sorted_pos.to(torch.int32),
        torch.full_like(sorted_pos, SELECTED_PAD_VALUE, dtype=torch.int32),
    )
    selected[:, :eff] = real
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
    token_scales: Optional[torch.Tensor] = None,
    scorer_norm: str = "off",
    head_agg: str = "max",
    hybrid_threshold: int = 8192,
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
    qf = q_proj.unsqueeze(1).to(torch.float32)          # [bs, 1, H, D]
    sf = gathered_sig.to(torch.float32)                  # [bs, max_seq_len, H, D]

    # Raw channel-dot (production), dequant-scaled for the int8 path.
    raw_dot = (qf * sf).sum(-1)  # [bs, max_seq_len, H]
    if token_scales is not None:
        scale_layer = token_scales[layer_id]                       # [max_tokens, H]
        scale_gathered = scale_layer[safe_phys].to(torch.float32)  # [bs, max_seq_len, H]
        raw_dot = raw_dot * scale_gathered

    if scorer_norm == "off":
        dot = raw_dot
    else:
        # Direction-only (cosine) score: unit-normalize per (head) channel vector
        # so per-token background magnitude can't dominate. Scale-invariant, so
        # token_scales is intentionally ignored for the cosine contribution.
        eps = 1e-6
        cos_dot = (
            (qf / (qf.norm(dim=-1, keepdim=True) + eps))
            * (sf / (sf.norm(dim=-1, keepdim=True) + eps))
        ).sum(-1)  # [bs, max_seq_len, H]
        if scorer_norm == "cosine":
            dot = cos_dot
        else:  # "hybrid": raw for short context, cosine for long (per request)
            use_cos = (seq_lens.to(device) > hybrid_threshold).view(-1, 1, 1)
            dot = torch.where(use_cos, cos_dot, raw_dot)

    # Cross-head aggregation.
    scores = dot.mean(dim=-1) if head_agg == "mean" else dot.amax(dim=-1)  # [bs, max_seq_len]

    # Mask: unwritten physical slots and positions >= seq_len
    written_layer = written[layer_id]  # [max_tokens] bool
    written_gathered = written_layer[safe_phys]  # [bs, max_seq_len] bool
    scores = scores.masked_fill(~written_gathered, float("-inf"))

    seq_len_mask = logical_positions < seq_lens.unsqueeze(1).to(device)  # [bs, max_seq_len]
    scores = scores.masked_fill(~seq_len_mask, float("-inf"))

    return scores


def _anchor_positions(n: int, budget: int, mode: str) -> list:
    """Deterministic anchor logical positions in ``[0, n)`` for one request.

    - ``recency``: the ``budget`` most-recent positions ``[n-budget, n)``.
    - ``global``: the ``budget`` earliest stable positions ``[0, budget)``.
    - ``strided``: ``budget`` distinct evenly-spaced positions over ``[0, n)``.
    Clamps ``budget`` to ``n``; returns ``[]`` for ``off`` / empty.
    """
    if budget <= 0 or n <= 0 or mode == "off":
        return []
    b = min(budget, n)
    if mode == "recency":
        return list(range(n - b, n))
    if mode == "global":
        return list(range(0, b))
    if mode == "strided":
        if b == 1:
            return [0]
        step = (n - 1) / (b - 1)
        return sorted({int(round(i * step)) for i in range(b)})
    return []


def _anchor_positions_tensor(
    seq_lens: torch.Tensor, eb: torch.Tensor, A: int, mode: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Tensorized ``_anchor_positions``: ``[bs, A]`` anchor logical positions +
    ``[bs, A]`` validity (slot ``i`` valid iff ``i < eb`` and, for strided, it is
    the first occurrence of its value). Graph-safe (no host sync, fixed shape)."""
    device = seq_lens.device
    bs = seq_lens.shape[0]
    i = torch.arange(A, device=device).view(1, A)
    n = seq_lens.view(bs, 1).to(torch.int64)
    ebv = eb.view(bs, 1)
    valid = i < ebv
    if mode == "recency":
        pos = n - ebv + i
    elif mode == "global":
        pos = i.expand(bs, A).clone()
    elif mode == "strided":
        denom = (ebv - 1).clamp(min=1).to(torch.float64)
        step = (n - 1).to(torch.float64) / denom
        pos = torch.round(i.to(torch.float64) * step).to(torch.int64)
        pos = torch.where(ebv == 1, torch.zeros_like(pos), pos)
        # strided's set-dedup: values are ascending in i, so a duplicate is == prev.
        prev = torch.cat(
            [torch.full((bs, 1), -1, dtype=torch.int64, device=device), pos[:, :-1]],
            dim=1,
        )
        valid = valid & (pos != prev)
    else:
        pos = torch.zeros(bs, A, dtype=torch.int64, device=device)
        valid = torch.zeros(bs, A, dtype=torch.bool, device=device)
    pos = torch.where(valid, pos, torch.full_like(pos, -1))
    return pos, valid


def _stable_argsort_ascending(
    key: torch.Tensor, tiebreak_pos: torch.Tensor
) -> torch.Tensor:
    """Argsort ``key`` ascending with ``tiebreak_pos`` ascending as the stable
    tie-break (two stable passes). Mirrors the eager list ``.sort(key=score)``,
    which keeps the original position-ascending order among equal scores."""
    order_p = torch.argsort(tiebreak_pos, dim=1, stable=True)
    key_p = torch.gather(key, 1, order_p)
    order_k = torch.argsort(key_p, dim=1, stable=True)
    return torch.gather(order_p, 1, order_k)


def _force_include_anchor(
    indices: torch.Tensor,
    scores: torch.Tensor,
    seq_lens: torch.Tensor,
    anchor_budget: int,
    anchor_mode: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Force each request's deterministic anchor positions (per ``anchor_mode``)
    into the selection, evicting the lowest-scoring non-anchor selected positions
    (stable position-ascending tie-break) and preserving the per-row selected
    count.

    Fully tensorized (no per-row Python loop, no ``.item()`` host sync, fixed
    shapes), so it is graph-safe and is used by BOTH the eager logical path and
    the graph-safe Triton path — guaranteeing identical selection. Bit-identical
    to the former per-row reference (fuzz-verified) including the R3 over-budget
    clamp (``effective_budget = min(anchor_budget, valid_count, seq_len)``) and
    strided set-dedup.
    """
    if anchor_mode == "off" or anchor_budget <= 0:
        return indices.to(torch.int32), (indices >= 0).to(torch.int32).sum(-1)
    device = indices.device
    bs, K = indices.shape
    max_seq = scores.shape[1]
    # Bound the [bs, A] temporaries: the effective budget is
    # min(anchor_budget, valid_count, seq_len) and valid_count <= K, so anchor
    # slots beyond K (or max_seq) can never be valid. Clamping A here is
    # bit-identical (clamped-out slots would be invalid anyway) but stops a
    # pathological opt-in anchor_budget from over-allocating scratch.
    A = min(int(anchor_budget), K, max_seq)
    pos = indices.to(torch.int64)
    real_mask = pos >= 0
    real_count = real_mask.sum(1)
    n = seq_lens.to(torch.int64)
    eb = torch.minimum(torch.full_like(real_count, A), real_count)
    eb = torch.minimum(eb, n)  # _anchor_positions further clamps the budget to n

    apos, avalid = _anchor_positions_tensor(n, eb, A, anchor_mode)
    psafe = pos.clamp(min=0)

    # max_seq-wide membership masks (an extra sentinel column absorbs -1 pads).
    sel_mask = torch.zeros(bs, max_seq + 1, dtype=torch.bool, device=device)
    sel_mask.scatter_(1, torch.where(real_mask, pos, torch.full_like(pos, max_seq)), True)
    sel_mask = sel_mask[:, :max_seq]
    anc_mask = torch.zeros(bs, max_seq + 1, dtype=torch.bool, device=device)
    anc_mask.scatter_(1, torch.where(avalid, apos, torch.full_like(apos, max_seq)), True)
    anc_mask = anc_mask[:, :max_seq]

    missing = avalid & ~torch.gather(sel_mask, 1, apos.clamp(min=0))   # [bs,A]
    evictable = real_mask & ~torch.gather(anc_mask, 1, psafe)          # [bs,K]
    k = torch.minimum(missing.sum(1), evictable.sum(1))               # [bs]

    # Evict the k lowest-score evictables (score asc, position asc tie-break).
    big_score = torch.finfo(torch.float32).max
    evict_score = torch.where(
        evictable, torch.gather(scores, 1, psafe),
        torch.full((bs, K), big_score, dtype=scores.dtype, device=device),
    )
    order = _stable_argsort_ascending(evict_score, pos)
    rank = torch.empty_like(order)
    rank.scatter_(1, order, torch.arange(K, device=device).view(1, K).expand(bs, K))
    drop = evictable & (rank < k.view(bs, 1))
    keep = real_mask & ~drop

    # Insert the first k missing anchors (ascending position).
    miss_rank = torch.cumsum(missing.to(torch.int64), dim=1) - 1
    insert = missing & (miss_rank < k.view(bs, 1))

    # Combine keep + inserted positions, sort ascending, pad to K with -1.
    big = max_seq + 10
    keep_pos = torch.where(keep, psafe, torch.full_like(psafe, big))
    ins_pos = torch.where(insert, apos.clamp(min=0), torch.full((bs, A), big, dtype=torch.int64, device=device))
    combined, _ = torch.sort(torch.cat([keep_pos, ins_pos], dim=1), dim=1)
    out = combined[:, :K]
    out = torch.where(out >= big, torch.full_like(out, -1), out).to(torch.int32)
    return out, (out >= 0).to(torch.int32).sum(-1)


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
    token_scales: Optional[torch.Tensor] = None,
    scorer_norm: Optional[str] = None,
    head_agg: str = "max",
    hybrid_threshold: int = 8192,
    anchor_mode: str = "off",
    anchor_budget: int = 0,
    recall_oracle: bool = False,
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

    _norm = scorer_norm if scorer_norm is not None else _scorer_norm_mode()
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
            token_scales=token_scales,
            scorer_norm=_norm,
            head_agg=head_agg,
            hybrid_threshold=hybrid_threshold,
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
            token_scales=token_scales,
            scorer_norm=_norm,
            head_agg=head_agg,
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
    if anchor_mode != "off" and anchor_budget > 0 and use_logical and seq_lens is not None:
        indices, valid_lengths = _force_include_anchor(
            indices, scores, seq_lens, int(anchor_budget), anchor_mode
        )
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
        # Flag-gated recall oracle (off by default). Records on the live
        # all-reduced ``scores`` (the authoritative tensor consumed by the top-K
        # above) for the harness-registered NIAH trial. Guarded by the same
        # capture check as the metrics call: the oracle does host syncs
        # (``.item()``/dict build) that are illegal during CUDA-graph capture.
        _maybe_record_recall_oracle(
            scores,
            indices,
            layer_id,
            max_top_k,
            process_group=process_group,
            recall_oracle=recall_oracle,
        )
    return indices, valid_lengths


def _maybe_record_recall_oracle(
    scores: torch.Tensor,
    selected_indices: torch.Tensor,
    layer_id: int,
    max_top_k: int,
    process_group=None,
    recall_oracle: bool = False,
) -> None:
    """Record one recall-oracle sample for the active NIAH trial, if enabled.

    Pure no-op (immediate return) when the oracle is off — so production
    selection is byte-for-byte unchanged. Enabled either by the config-borne
    ``recall_oracle`` flag (the path that reaches TP workers) or the env flag
    (harness / unit tests).

    **Fail-closed when enabled** — a diagnostic must never silently guess or
    silently drop a sample. With no active trial, an out-of-range harness needle
    position, or a payload-build exception, we emit an explicit ``failure``
    record keyed by ``(request_id, trial_id, layer_id, decode_step)`` instead of
    returning quietly; the sweep asserts on these + on missing successes. We do
    NOT filter out-of-range positions (that silently masked the absent 64K
    records) and we do NOT swallow exceptions.

    Records ONLY on the primary TP rank: the scores are identical across ranks
    after ``all_reduce_token_scores``, so rank-0-only recording avoids 8×
    duplicate writes + cross-process file contention on the sink.
    """
    from sglang.srt.layers.attention.double_sparsity import oracle_artifact_sink as _sink

    if recall_oracle:
        # Latch the config-borne enable so the sink + trial-file paths resolve
        # to the fixed cross-process defaults (env does not reach TP workers).
        _sink.enable_via_config()
    if not _sink.oracle_enabled():
        return
    # Primary-rank guard (scores are all-reduce-identical across TP ranks).
    try:
        _rk = -1
        if (
            process_group is not None
            and torch.distributed.is_available()
            and torch.distributed.is_initialized()
        ):
            _rk = torch.distributed.get_rank(group=process_group)
        if _rk not in (-1, 0):
            return  # non-primary rank (silent — every rank but 0)
    except Exception:
        pass

    sample_idx = _sink.next_sample_index()
    trial = _sink.get_active_trial()
    if trial is None:
        # Fail-closed: enabled but no trial registered for this decode. Emit a
        # marker (the harness clears the sink before measured trials, so warmup
        # markers do not pollute the measured run).
        _sink.record_oracle_failure(
            reason="no_active_trial",
            request_id=None,
            trial_id=None,
            layer_id=int(layer_id),
            decode_step=int(sample_idx),
        )
        return

    max_tokens = int(scores.shape[-1])
    out_of_range = [p for p in trial.needle_positions if not (0 <= p < max_tokens)]
    if out_of_range:
        # Fail-closed: reject (do NOT filter) — a partial/empty span would
        # silently mis-measure recall.
        _sink.record_oracle_failure(
            reason="span_out_of_range",
            request_id=trial.request_id,
            trial_id=trial.trial_id,
            layer_id=int(layer_id),
            decode_step=int(sample_idx),
            extra={
                "needle_positions": list(trial.needle_positions),
                "out_of_range": out_of_range,
                "max_tokens": max_tokens,
            },
        )
        return

    try:
        from sglang.srt.layers.attention.double_sparsity.selection_recall_oracle import (
            oracle_payload_for_row,
        )

        needle = torch.as_tensor(
            trial.needle_positions, dtype=torch.int64, device=scores.device
        )
        payload = oracle_payload_for_row(
            scores[0],
            needle,
            selected_indices_row=selected_indices[0],
            stride=1,
            index_topk=int(max_top_k),
        )
        _sink.record_oracle_sample(
            request_id=trial.request_id,
            trial_id=trial.trial_id,
            layer_id=int(layer_id),
            decode_step=int(sample_idx),
            payload=payload,
        )
    except Exception as _e:
        # Fail-closed: surface the failure as a record rather than swallowing it.
        _sink.record_oracle_failure(
            reason=f"exception:{type(_e).__name__}:{_e}",
            request_id=trial.request_id,
            trial_id=trial.trial_id,
            layer_id=int(layer_id),
            decode_step=int(sample_idx),
        )
        return


def _logical_score_triton(
    q_proj_input: torch.Tensor,         # [bs, H, head_dim] fp32 (the raw queries; gather via ch_sel inside)
    channel_selection_layer: torch.Tensor,  # [H, label_dim] int32
    channel_weights_layer: torch.Tensor,    # [H, label_dim] fp32
    sig_layer: torch.Tensor,            # [T, H, label_dim] fp16/fp32/int8
    written_layer: torch.Tensor,        # [T] bool
    req_pool_indices: torch.Tensor,     # [bs] int32
    req_to_token: torch.Tensor,         # [num_pools, max_pool_len] int32
    seq_lens: torch.Tensor,             # [bs] int32
    out: torch.Tensor,                  # [bs_buf, max_seq_len] fp32 (pre-allocated, slice [bs])
    max_seq_len: int,
    *,
    scale_layer: Optional[torch.Tensor] = None,  # [T, H] per-(slot, head) int8 dequant scale, else None
    token_block: int = 64,
    scorer_norm: str = "off",
    head_agg: str = "max",
    hybrid_threshold: int = 8192,
) -> None:
    """Fill ``out[:bs, :max_seq_len]`` with per-(batch, logical-position) scores.

    All allocations happen in the caller. No `.item()` host syncs.
    """
    bs, num_heads, _head_dim = q_proj_input.shape
    label_dim = int(channel_selection_layer.shape[1])
    max_pool_len = int(req_to_token.shape[1])
    max_tokens = int(sig_layer.shape[0])

    has_scale = scale_layer is not None
    scale_ptr = scale_layer if has_scale else sig_layer
    scale_stride_t = scale_layer.stride(0) if has_scale else 0
    scale_stride_h = scale_layer.stride(1) if has_scale else 0

    desired_block = min(token_block, max(max_seq_len, 1))
    token_block_pow2 = _next_pow2(desired_block)
    label_dim_pow2 = _next_pow2(max(label_dim, 1))
    num_token_blocks = (max_seq_len + token_block_pow2 - 1) // token_block_pow2
    grid = (bs, num_token_blocks)

    scorer_norm_code = {"off": 0, "cosine": 1, "hybrid": 2}.get(scorer_norm or "off", 0)
    head_agg_mean = head_agg == "mean"

    _logical_score_kernel[grid](
        q_proj_input,
        channel_selection_layer,
        channel_weights_layer,
        sig_layer,
        scale_ptr,
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
        scale_stride_t=scale_stride_t,
        scale_stride_h=scale_stride_h,
        HAS_SCALE=has_scale,
        rtt_stride_p=req_to_token.stride(0),
        out_stride_b=out.stride(0),
        TOKEN_BLOCK=token_block_pow2,
        LABEL_DIM_POW2=label_dim_pow2,
        SCORER_NORM=scorer_norm_code,
        HEAD_AGG_MEAN=head_agg_mean,
        HYBRID_THRESHOLD=int(hybrid_threshold),
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
    token_scales: Optional[torch.Tensor] = None,           # fp16 [L, T, H] int8 dequant scale, else None
    process_group=None,
    recall_oracle: bool = False,
    scorer_norm: str = "off",
    head_agg: str = "max",
    hybrid_threshold: int = 8192,
    anchor_mode: str = "off",
    anchor_budget: int = 0,
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
            token_scales=token_scales,
            recall_oracle=recall_oracle,
            scorer_norm=scorer_norm,
            head_agg=head_agg,
            hybrid_threshold=hybrid_threshold,
            anchor_mode=anchor_mode,
            anchor_budget=anchor_budget,
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
    scale_layer = token_scales[layer_id] if token_scales is not None else None
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
        scale_layer=scale_layer,
        scorer_norm=scorer_norm,
        head_agg=head_agg,
        hybrid_threshold=hybrid_threshold,
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

    # Graph-safe anchor-budget force-include (R9): tensorized, fixed-shape, no
    # host sync — bit-identical to the eager path (same _force_include_anchor).
    # Off by default; under CUDA-graph capture the extra ops are captured once and
    # replay reuses their memory (alloc-free on replay).
    if anchor_mode != "off" and anchor_budget > 0:
        a_idx, a_len = _force_include_anchor(
            out_indices[:bs, :max_top_k], scores_view, seq_lens,
            int(anchor_budget), anchor_mode,
        )
        out_indices[:bs, :max_top_k].copy_(a_idx)
        out_lengths[:bs].copy_(a_len)

    # Flag-gated recall oracle on the production GPU decode path. ``scores_view``
    # is the all-reduced + per-request-masked score tensor (after the all_reduce
    # above, the same tensor the top-K consumed); ``out_indices[:bs]`` is the
    # selection. Capture-guarded (host syncs illegal during graph capture) and
    # off by default, so production decode is unaffected. Records only in eager
    # decode (under graph replay this Python does not re-run).
    if not torch.cuda.is_current_stream_capturing():
        _maybe_record_recall_oracle(
            scores_view,
            out_indices[:bs],
            layer_id,
            max_top_k,
            process_group=process_group,
            recall_oracle=recall_oracle,
        )

    return out_indices, out_lengths


# Public alias for the end-to-end selector pipeline.
retrieve_topk = retrieve_topk_via_labels
