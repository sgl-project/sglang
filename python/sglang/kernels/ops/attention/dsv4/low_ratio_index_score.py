"""Triton kernel for DeepSeek V4.1 low-ratio indexer score + mask (A5).

A5 variant of the A3 ``low_ratio_index_score`` kernel: identical masking and
candidate-source epilogue, plus in-kernel dequantization of the quantized
A5 index-K pool (fp8_e4m3fn + fp32 scale on arch35; int8 + fp16 scale on
older NPUs), matching ``NPUDeepSeekV4IndexerPool.get_index_k_dequant``.

Replaces the per-request Python loop in
``ascend_dsv4_backend._low_ratio_index_topk_torch_a5`` with one fused kernel:

  score[t, j] = sum_h relu(q[t, h] . (k[slot(t, j)] * scale[slot(t, j)])) * w[t, h]

where ``slot(t, j) = req_to_token[req[t], j * ratio] // ratio``.

Masking (fused in-kernel):
  - causal: position j visible iff j < lens[t] = (pos[t] + 1) // ratio,
    invalid positions written as -inf;
  - consume (layers that use candidates): additionally -inf where the
    candidate mask published by the candidate-source layer is False.

NOT fused (kept in torch on the [T, max_lc] score matrix, per design):
  - the final token-level top-k;
  - the candidate-source layer's block-level top-k (argmax iterations over
    the block scores — itself a top-k, computed outside the kernel).

For the candidate-source layer the kernel additionally emits per-block amax
scores ``block_scores`` [T, num_blocks] (one program == one candidate block,
i.e. BLOCK_N is forced to the candidate block size): the block amax and the
"newest block forced to +inf" fill are fused in-kernel, so outside the kernel
only the block top-k and the block->position mask expansion remain.

Output: ``out`` [T, max_lc] float32 with -inf at every invalid position —
ready for ``s.topk(...)`` outside.

Set SGLANG_DSV4_INDEX_SCORE_BLOCK_N to override the position tiling
(ignored for the candidate-source layer, where BLOCK_N == block size).
"""

from __future__ import annotations

import os
from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["T", "max_lc", "lc_w", "stride_rtt", "H"])
def _index_score_kernel(
    q_ptr,  # [T, H, D] bf16
    weights_ptr,  # [T, H] bf16
    req_ptr,  # [T] int — request pool index per token
    req_to_token_ptr,  # [R, max_len] int32
    k_buf_ptr,  # [total_slots, D] fp8/int8 (quantized) or bf16 (unquantized)
    k_scale_ptr,  # [total_slots] fp32/fp16 (only when K_QUANTIZED)
    lens_ptr,  # [T] int64 — visible compressed length per token
    consume_ptr,  # [T, lc_w] bool — candidate mask (only when HAS_CONSUME)
    out_ptr,  # [T, max_lc] fp32
    block_scores_ptr,  # [T, num_blocks] fp32 (only when IS_CANDIDATE_SOURCE)
    T,
    max_lc,
    lc_w,
    stride_rtt,
    H,
    D,
    RATIO: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HAS_CONSUME: tl.constexpr,
    IS_CANDIDATE_SOURCE: tl.constexpr,
    K_QUANTIZED: tl.constexpr,
):
    t = tl.program_id(0)
    pid_n = tl.program_id(1)

    lens_t = tl.load(lens_ptr + t)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    valid = offs_n < lens_t  # causal: group end passed by this query

    r = tl.load(req_ptr + t)
    # Full-pool slot of each visible group's base token -> index-pool slot.
    rtt_off = r.to(tl.int64) * stride_rtt + offs_n * RATIO
    rtt_off = tl.where(valid, rtt_off, 0)
    full = tl.load(req_to_token_ptr + rtt_off, mask=valid, other=0)
    slots = (full // RATIO).to(tl.int64)
    slots = tl.where(valid, slots, 0)

    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < D
    k_ptrs = k_buf_ptr + slots[:, None] * D + offs_d[None, :]
    k_mask = valid[:, None] & d_mask[None, :]
    k_v = tl.load(k_ptrs, mask=k_mask, other=0.0).to(tl.float32)
    if K_QUANTIZED:
        # Per-slot dequant scale, broadcast over D — matches
        # get_index_k_dequant's (k * scale).to(bf16) up to the final rounding,
        # which the fp32 accumulation here skips (slightly more accurate).
        sc = tl.load(k_scale_ptr + slots, mask=valid, other=0.0).to(tl.float32)
        k_v = k_v * sc[:, None]

    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    for h in range(0, H):
        q_h = tl.load(
            q_ptr + (t.to(tl.int64) * H + h) * D + offs_d,
            mask=d_mask,
            other=0.0,
        ).to(tl.float32)
        # relu before the head weights, as the reference scores()
        s_h = tl.maximum(tl.sum(k_v * q_h[None, :], axis=1), 0.0)
        w_h = tl.load(weights_ptr + t.to(tl.int64) * H + h).to(tl.float32)
        acc += s_h * w_h

    out = tl.where(valid, acc, float("-inf"))
    if HAS_CONSUME:
        cm = tl.load(
            consume_ptr + t.to(tl.int64) * lc_w + offs_n,
            mask=offs_n < lc_w,
            other=0,
        )
        out = tl.where(valid & (cm != 0), out, float("-inf"))
    tl.store(
        out_ptr + t.to(tl.int64) * max_lc + offs_n,
        out,
        mask=offs_n < max_lc,
    )

    if IS_CANDIDATE_SOURCE:
        # BLOCK_N == candidate block size here, so this program's tile is
        # exactly one candidate block: emit its amax. Lanes beyond lens are
        # -inf (causal), so an all-unreachable block yields -inf, matching
        # the torch reference's padded amax. The block holding the query's
        # newest position is forced to +inf (always kept), as the reference
        # does with masked_fill before its block top-k.
        bmax = tl.max(out, axis=0)
        is_last = (lens_t > 0) & (pid_n == (lens_t - 1) // BLOCK_N)
        bmax = tl.where(is_last, float("inf"), bmax)
        tl.store(
            block_scores_ptr + t.to(tl.int64) * tl.num_programs(1) + pid_n, bmax
        )


def low_ratio_index_score_triton(
    q: torch.Tensor,  # [T, H, D] bf16 contiguous
    weights: torch.Tensor,  # [T, H] bf16/fp32 contiguous
    req: torch.Tensor,  # [T] int — request pool index per token
    req_to_token: torch.Tensor,  # [R, max_len] int32
    k_buf: torch.Tensor,  # [total_slots, D] fp8/int8 (quantized) or bf16
    lens: torch.Tensor,  # [T] int64
    max_lc: int,  # host-side upper bound of lens (>= max(lens)); an upper
    # bound is safe: columns beyond a row's lens are -inf-masked in-kernel,
    # so topk's `reach` maps them to -1 exactly like exact-width allocation.
    k_scale: Optional[torch.Tensor] = None,  # [total_slots] fp32/fp16 —
    # required iff k_buf is a quantized (fp8/int8) pool; per-slot dequant
    # scale applied in-kernel.
    consume: Optional[torch.Tensor] = None,  # [T, W] bool
    ratio: int = 2,
    candidate_block_size: Optional[int] = None,  # set on the candidate-source
    # layer: additionally emit [T, cdiv(max_lc, block_size)] fp32 block amax
    # scores (newest block already forced to +inf) for the outside block
    # top-k; BLOCK_N is forced to the block size so one program == one block.
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Fused indexer scores + dequant + causal (+consume) mask; topk outside.

    Returns ``(s, block_scores)``: ``s`` is [T, max_lc] fp32 with -inf at
    invalid positions; ``block_scores`` is [T, num_blocks] fp32 when
    ``candidate_block_size`` is given, else None. ``max_lc`` must be computed
    on the host (e.g. from seq_lens_cpu) — no device sync here.
    """
    T, H, D = q.shape
    if T == 0 or max_lc == 0:
        return q.new_empty(T, 0, dtype=torch.float32), None

    assert q.dtype == torch.bfloat16 and q.is_contiguous()
    assert weights.is_contiguous() and weights.shape == (T, H)
    assert k_buf.is_contiguous() and k_buf.shape[-1] == D
    assert lens.dtype in (torch.int32, torch.int64) and lens.is_contiguous()
    assert ratio in (1, 2)

    if k_scale is not None:
        assert k_buf.dtype in (torch.float8_e4m3fn, torch.int8), (
            "k_scale given but k_buf is not a quantized pool"
        )
        assert k_scale.is_contiguous() and k_scale.dim() == 1
        assert k_scale.shape[0] == k_buf.shape[0]
        k_quantized = True
    else:
        assert k_buf.dtype == torch.bfloat16, (
            "unquantized path expects the bf16 index-K pool"
        )
        k_quantized = False

    if consume is not None:
        # Coerce (not assert): the published mask may be non-bool or a view.
        if consume.dtype != torch.bool:
            consume = consume.to(torch.bool)
        if not consume.is_contiguous():
            consume = consume.contiguous()
        assert consume.shape[0] == T
        lc_w = consume.shape[1]
    else:
        lc_w = max_lc

    is_candidate_source = candidate_block_size is not None
    if is_candidate_source:
        # tl.arange needs a power of two; one program must cover exactly one
        # candidate block, so the env override is ignored on this layer.
        assert candidate_block_size & (candidate_block_size - 1) == 0, (
            "candidate_block_size must be a power of two for the fused kernel"
        )
        BLOCK_N = candidate_block_size
    else:
        env_bn = os.environ.get("SGLANG_DSV4_INDEX_SCORE_BLOCK_N", "")
        BLOCK_N = int(env_bn) if env_bn else 32
    BLOCK_D = triton.next_power_of_2(D)

    num_blocks = triton.cdiv(max_lc, BLOCK_N) if is_candidate_source else 0
    out = torch.empty((T, max_lc), dtype=torch.float32, device=q.device)
    block_scores = (
        torch.empty((T, num_blocks), dtype=torch.float32, device=q.device)
        if is_candidate_source
        else out  # dummy ptr when unused
    )
    grid = (T, triton.cdiv(max_lc, BLOCK_N))
    _index_score_kernel[grid](
        q,
        weights,
        req,
        req_to_token,
        k_buf,
        k_scale if k_scale is not None else k_buf,  # dummy ptr when unused
        lens,
        consume if consume is not None else q,  # dummy ptr when unused
        out,
        block_scores,
        T,
        max_lc,
        lc_w,
        req_to_token.stride(0),
        H,
        D,
        RATIO=ratio,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        HAS_CONSUME=consume is not None,
        IS_CANDIDATE_SOURCE=is_candidate_source,
        K_QUANTIZED=k_quantized,
    )
    return out, block_scores
