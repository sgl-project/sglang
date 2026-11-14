"""Fused target-probability front end for speculative verify sampling.

``eagle_sample`` builds ``target_probs`` from the ``[bs * rows_per_request, V]``
verify logits with a chain of full-vocab passes: per-request tensors expanded
with ``repeat_interleave``, then additive penalties, scaling (repetition)
penalties, logit bias, temperature and softmax. This op does all of that in two
launches, indexing per-request tensors by ``row // rows_per_request``.

Order (must match ``eagle_sample``):
    1. x = logits + additive_penalties[req]
    2. x = where(x < 0, x * s, x / s), s = scaling_penalties[req]
    3. x = x + logit_bias[req]
    4. probs = softmax(x / temperatures[req]) in fp32

Top-p / top-k renorm is not fused: it stays in ``sglang.srt.layers.sampling_renorm``,
which switches to deterministic kernels when TP ranks must agree bit for bit.
Grammar masks and watermarking run between steps 3 and 4, so the caller keeps
the unfused path when either is active.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

# Split each row across CTAs so small batches still fill the GPU.
_TILE = 2048
_TARGET_CTAS = 512


@triton.jit
def _penalized(
    logits_ptr,
    add_ptr,
    scale_ptr,
    bias_ptr,
    row,
    req,
    offs,
    mask,
    logits_row_stride,
    vocab_size,
    HAS_ADD: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    x = tl.load(logits_ptr + row * logits_row_stride + offs, mask=mask).to(tl.float32)
    if HAS_ADD:
        x += tl.load(add_ptr + req * vocab_size + offs, mask=mask, other=0.0)
    if HAS_SCALE:
        s = tl.load(scale_ptr + req * vocab_size + offs, mask=mask, other=1.0)
        x = tl.where(x < 0, x * s, x / s)
    if HAS_BIAS:
        x += tl.load(bias_ptr + req * vocab_size + offs, mask=mask, other=0.0)
    return x


@triton.jit
def _partial_stats_kernel(
    logits_ptr,
    temps_ptr,
    add_ptr,
    scale_ptr,
    bias_ptr,
    partial_max_ptr,
    partial_sum_ptr,
    logits_row_stride,
    vocab_size,
    chunk,
    num_splits,
    rows_per_request,
    HAS_ADD: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    WRITE_BACK: tl.constexpr,
    TILE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    split = tl.program_id(1)
    req = row // rows_per_request
    inv_t = 1.0 / tl.load(temps_ptr + req).to(tl.float32)

    start = split * chunk
    end = tl.minimum(start + chunk, vocab_size)
    m = tl.full([TILE], -float("inf"), tl.float32)
    s = tl.zeros([TILE], tl.float32)
    for base in range(start, end, TILE):
        offs = base + tl.arange(0, TILE)
        mask = offs < end
        x = _penalized(
            logits_ptr, add_ptr, scale_ptr, bias_ptr, row, req, offs, mask,
            logits_row_stride, vocab_size, HAS_ADD, HAS_SCALE, HAS_BIAS,
        )  # fmt: skip
        if WRITE_BACK:
            tl.store(
                logits_ptr + row * logits_row_stride + offs,
                x.to(logits_ptr.dtype.element_ty),
                mask=mask,
            )
        y = tl.where(mask, x * inv_t, -float("inf"))
        m_new = tl.maximum(m, y)
        # Lanes that have only seen -inf keep a zero sum instead of exp(nan).
        safe = tl.where(m_new == -float("inf"), 0.0, m_new)
        s = s * tl.exp(m - safe) + tl.exp(y - safe)
        m = m_new

    block_max = tl.max(m, axis=0)
    safe_max = tl.where(block_max == -float("inf"), 0.0, block_max)
    lane_m = tl.where(m == -float("inf"), safe_max, m)
    block_sum = tl.sum(
        tl.where(m == -float("inf"), 0.0, s * tl.exp(lane_m - safe_max)), axis=0
    )
    tl.store(partial_max_ptr + row * num_splits + split, block_max)
    tl.store(partial_sum_ptr + row * num_splits + split, block_sum)


@triton.jit
def _normalize_kernel(
    logits_ptr,
    temps_ptr,
    add_ptr,
    scale_ptr,
    bias_ptr,
    partial_max_ptr,
    partial_sum_ptr,
    probs_ptr,
    logits_row_stride,
    probs_row_stride,
    vocab_size,
    chunk,
    num_splits,
    rows_per_request,
    HAS_ADD: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    SPLITS_POW2: tl.constexpr,
    TILE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    split = tl.program_id(1)
    req = row // rows_per_request
    inv_t = 1.0 / tl.load(temps_ptr + req).to(tl.float32)

    # Each program combines its row's few partials itself, so no separate
    # combine launch is needed.
    so = tl.arange(0, SPLITS_POW2)
    smask = so < num_splits
    pm = tl.load(
        partial_max_ptr + row * num_splits + so, mask=smask, other=-float("inf")
    )
    ps = tl.load(partial_sum_ptr + row * num_splits + so, mask=smask, other=0.0)
    row_max = tl.max(pm, axis=0)
    row_sum = tl.sum(
        tl.where(pm == -float("inf"), 0.0, ps * tl.exp(pm - row_max)), axis=0
    )
    inv_sum = 1.0 / row_sum

    start = split * chunk
    end = tl.minimum(start + chunk, vocab_size)
    for base in range(start, end, TILE):
        offs = base + tl.arange(0, TILE)
        mask = offs < end
        x = _penalized(
            logits_ptr, add_ptr, scale_ptr, bias_ptr, row, req, offs, mask,
            logits_row_stride, vocab_size, HAS_ADD, HAS_SCALE, HAS_BIAS,
        )  # fmt: skip
        p = tl.exp(x * inv_t - row_max) * inv_sum
        tl.store(probs_ptr + row * probs_row_stride + offs, p, mask=mask)


def _split_geometry(rows: int, vocab_size: int):
    splits = max(
        1, min(triton.cdiv(_TARGET_CTAS, rows), triton.cdiv(vocab_size, _TILE))
    )
    chunk = triton.cdiv(triton.cdiv(vocab_size, splits), _TILE) * _TILE
    return chunk, triton.cdiv(vocab_size, chunk)


def verify_target_probs(
    logits: torch.Tensor,
    *,
    rows_per_request: int,
    temperatures: torch.Tensor,
    additive_penalties: Optional[torch.Tensor] = None,
    scaling_penalties: Optional[torch.Tensor] = None,
    logit_bias: Optional[torch.Tensor] = None,
    write_back_logits: bool = False,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Return fp32 ``target_probs`` of shape ``[R, V]`` for verify sampling.

    Args:
        logits: ``[R, V]`` target logits (fp32/bf16/fp16), ``R = bs * rows_per_request``.
        rows_per_request: verify rows per request (``draft_token_num``).
        temperatures: ``[bs]`` or ``[bs, 1]`` per-request temperatures.
        additive_penalties: ``[bs, V]`` fp32 frequency/presence penalties, or None.
        scaling_penalties: ``[bs, V]`` fp32 repetition penalties, or None.
        logit_bias: ``[bs, V]`` fp32 logit bias, or None.
        write_back_logits: also store the penalized logits into ``logits`` in
            place, as the unfused path does (spec logprobs read them later).
        out: optional ``[R, V]`` fp32 output buffer.
    """
    _check_inputs(
        logits,
        rows_per_request=rows_per_request,
        temperatures=temperatures,
        per_request_vocab=(additive_penalties, scaling_penalties, logit_bias),
    )
    rows, vocab = logits.shape
    if out is None:
        out = torch.empty((rows, vocab), dtype=torch.float32, device=logits.device)
    if rows == 0:
        return out

    chunk, num_splits = _split_geometry(rows, vocab)
    partial_max = torch.empty(
        (rows, num_splits), dtype=torch.float32, device=logits.device
    )
    partial_sum = torch.empty_like(partial_max)
    temps = temperatures.reshape(-1)
    has_add = additive_penalties is not None
    has_scale = scaling_penalties is not None
    has_bias = logit_bias is not None
    write_back = write_back_logits and (has_add or has_scale or has_bias)
    # Unused pointer slots get any valid tensor; the constexpr flags skip them.
    add = additive_penalties if has_add else temps
    scale = scaling_penalties if has_scale else temps
    bias = logit_bias if has_bias else temps

    grid = (rows, num_splits)
    _partial_stats_kernel[grid](
        logits, temps, add, scale, bias, partial_max, partial_sum,
        logits.stride(0), vocab, chunk, num_splits, rows_per_request,
        HAS_ADD=has_add, HAS_SCALE=has_scale, HAS_BIAS=has_bias,
        WRITE_BACK=write_back, TILE=_TILE, num_warps=8,
    )  # fmt: skip
    # After a write-back the logits already hold the penalized values.
    reapply = not write_back
    _normalize_kernel[grid](
        logits, temps, add, scale, bias, partial_max, partial_sum, out,
        logits.stride(0), out.stride(0), vocab, chunk, num_splits, rows_per_request,
        HAS_ADD=has_add and reapply, HAS_SCALE=has_scale and reapply,
        HAS_BIAS=has_bias and reapply,
        SPLITS_POW2=triton.next_power_of_2(num_splits), TILE=_TILE, num_warps=8,
    )  # fmt: skip
    return out


def _check_inputs(
    logits: torch.Tensor,
    *,
    rows_per_request: int,
    temperatures: torch.Tensor,
    per_request_vocab,
) -> None:
    if logits.dim() != 2 or logits.stride(1) != 1:
        raise ValueError(
            f"logits must be a row-contiguous [R, V], got {tuple(logits.shape)}"
        )
    rows, vocab = logits.shape
    if rows_per_request < 1 or rows % rows_per_request != 0:
        raise ValueError(
            f"rows ({rows}) must be a multiple of rows_per_request ({rows_per_request})"
        )
    bs = rows // rows_per_request
    if temperatures.numel() != bs:
        raise ValueError(
            f"temperatures must have {bs} values, got {temperatures.numel()}"
        )
    for t in per_request_vocab:
        if t is None:
            continue
        if tuple(t.shape) != (bs, vocab) or not t.is_contiguous():
            raise ValueError(
                f"per-request vocab tensor must be contiguous {(bs, vocab)}, "
                f"got {tuple(t.shape)}"
            )
        if t.dtype != torch.float32:
            raise ValueError(f"per-request vocab tensor must be fp32, got {t.dtype}")
