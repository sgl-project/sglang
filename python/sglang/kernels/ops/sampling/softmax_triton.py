"""Fused in-place Triton temperature softmax for CUDA and HIP GPUs."""

from __future__ import annotations

import functools
from typing import Optional

import torch
import triton
import triton.language as tl

_LOG2E = tl.constexpr(1.4426950408889634)
_SINGLE_PASS_MAX_VOCAB = 8192
_SPLIT_MIN_BLOCK = 2048
_MAX_SPLITS = 128
_INPUT_DTYPES = (torch.float32, torch.bfloat16, torch.float16)

# Kept for backwards compatibility with callers/tests referencing the low-row threshold.
ROW_LIMIT = 32


@functools.cache
def _device_num_sms(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


@triton.jit
def _row_softmax_kernel(
    logits_ptr,
    temps_ptr,
    out_ptr,
    logits_row_stride,
    out_row_stride,
    vocab_size,
    rows_per_request,
    HAS_TEMP: tl.constexpr,
    SINGLE_PASS: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    if HAS_TEMP:
        inv_t = 1.0 / tl.load(temps_ptr + row // rows_per_request).to(tl.float32)
        scale = inv_t * _LOG2E
    else:
        scale = _LOG2E

    in_row = logits_ptr + row * logits_row_stride
    out_row = out_ptr + row * out_row_stride

    if SINGLE_PASS:
        offs = tl.arange(0, BLOCK_V)
        mask = offs < vocab_size
        x = tl.load(in_row + offs, mask=mask, other=-float("inf")).to(tl.float32)
        m = tl.max(x, axis=0)
        safe_m = tl.where(m == -float("inf"), 0.0, m)
        e = tl.where(mask, tl.math.exp2((x - safe_m) * scale), 0.0)
        inv_s = 1.0 / tl.sum(e, axis=0)
        tl.store(out_row + offs, e * inv_s, mask=mask)
        return

    # Pass 1: online unscaled raw max + sum-exp across vocab tiles. Keeping the
    # max unscaled avoids overflowing +-1e30 sanitized logits on small temperatures.
    m = -float("inf")
    acc = 0.0
    for v0 in range(0, vocab_size, BLOCK_V):
        offs = v0 + tl.arange(0, BLOCK_V)
        mask = offs < vocab_size
        x = tl.load(in_row + offs, mask=mask, other=-float("inf")).to(tl.float32)
        tile_m = tl.max(x, axis=0)
        m_new = tl.maximum(m, tile_m)
        safe_m = tl.where(m_new == -float("inf"), 0.0, m_new)
        alpha = tl.where(m > -float("inf"), tl.math.exp2((m - safe_m) * scale), 0.0)
        tile_e = tl.where(mask, tl.math.exp2((x - safe_m) * scale), 0.0)
        acc = acc * alpha + tl.sum(tile_e, axis=0)
        m = m_new

    # Pass 2: normalize and write directly into out (safe when out aliases logits).
    safe_m = tl.where(m == -float("inf"), 0.0, m)
    inv_s = 1.0 / acc
    for v0 in range(0, vocab_size, BLOCK_V):
        offs = v0 + tl.arange(0, BLOCK_V)
        mask = offs < vocab_size
        x = tl.load(in_row + offs, mask=mask, other=-float("inf")).to(tl.float32)
        p = tl.math.exp2((x - safe_m) * scale) * inv_s
        tl.store(out_row + offs, p, mask=mask)


@triton.jit
def _split_stats_kernel(
    logits_ptr,
    temps_ptr,
    partial_max_ptr,
    partial_sum_ptr,
    logits_row_stride,
    vocab_size,
    chunk_size,
    num_splits,
    rows_per_request,
    HAS_TEMP: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    split = tl.program_id(1)
    if HAS_TEMP:
        inv_t = 1.0 / tl.load(temps_ptr + row // rows_per_request).to(tl.float32)
        scale = inv_t * _LOG2E
    else:
        scale = _LOG2E

    in_row = logits_ptr + row * logits_row_stride
    start = split * chunk_size
    end = tl.minimum(start + chunk_size, vocab_size)

    m = -float("inf")
    acc = 0.0
    for v0 in range(start, end, BLOCK_V):
        offs = v0 + tl.arange(0, BLOCK_V)
        mask = offs < end
        x = tl.load(in_row + offs, mask=mask, other=-float("inf")).to(tl.float32)
        tile_m = tl.max(x, axis=0)
        m_new = tl.maximum(m, tile_m)
        safe_m = tl.where(m_new == -float("inf"), 0.0, m_new)
        alpha = tl.where(m > -float("inf"), tl.math.exp2((m - safe_m) * scale), 0.0)
        tile_e = tl.where(mask, tl.math.exp2((x - safe_m) * scale), 0.0)
        acc = acc * alpha + tl.sum(tile_e, axis=0)
        m = m_new

    out_idx = row * num_splits + split
    tl.store(partial_max_ptr + out_idx, m)
    tl.store(partial_sum_ptr + out_idx, acc)


@triton.jit
def _split_write_kernel(
    logits_ptr,
    temps_ptr,
    partial_max_ptr,
    partial_sum_ptr,
    out_ptr,
    logits_row_stride,
    out_row_stride,
    vocab_size,
    chunk_size,
    num_splits,
    rows_per_request,
    HAS_TEMP: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    split = tl.program_id(1)
    if HAS_TEMP:
        inv_t = 1.0 / tl.load(temps_ptr + row // rows_per_request).to(tl.float32)
        scale = inv_t * _LOG2E
    else:
        scale = _LOG2E

    # Combine per-split statistics inline (<= 128 splits) to avoid a 3rd launch.
    s_offs = tl.arange(0, BLOCK_S)
    s_mask = s_offs < num_splits
    base_s = row * num_splits + s_offs
    pm = tl.load(partial_max_ptr + base_s, mask=s_mask, other=-float("inf"))
    ps = tl.load(partial_sum_ptr + base_s, mask=s_mask, other=0.0)
    row_max = tl.max(pm, axis=0)
    safe_m = tl.where(row_max == -float("inf"), 0.0, row_max)
    row_sum = tl.sum(
        tl.where(
            s_mask & (pm > -float("inf")),
            ps * tl.math.exp2((pm - safe_m) * scale),
            0.0,
        ),
        axis=0,
    )
    inv_s = 1.0 / row_sum

    in_row = logits_ptr + row * logits_row_stride
    out_row = out_ptr + row * out_row_stride
    start = split * chunk_size
    end = tl.minimum(start + chunk_size, vocab_size)
    for v0 in range(start, end, BLOCK_V):
        offs = v0 + tl.arange(0, BLOCK_V)
        mask = offs < end
        x = tl.load(in_row + offs, mask=mask, other=-float("inf")).to(tl.float32)
        p = tl.math.exp2((x - safe_m) * scale) * inv_s
        tl.store(out_row + offs, p, mask=mask)


def can_softmax(
    logits: torch.Tensor,
    temperatures: Optional[torch.Tensor] = None,
    *,
    rows_per_request: int = 1,
    out: Optional[torch.Tensor] = None,
) -> bool:
    """Whether `temperature_softmax` supports these inputs on the current device."""
    if (
        not isinstance(logits, torch.Tensor)
        or not logits.is_cuda
        or logits.ndim != 2
        or logits.dtype not in _INPUT_DTYPES
        or logits.stride(1) != 1
        or rows_per_request < 1
    ):
        return False
    rows, _ = logits.shape
    if rows % rows_per_request != 0:
        return False
    if out is not None:
        if (
            out.device != logits.device
            or out.dtype != torch.float32
            or out.shape != logits.shape
            or out.stride(1) != 1
        ):
            return False
    if temperatures is not None:
        t = temperatures.reshape(-1)
        if (
            t.device != logits.device
            or t.dtype not in _INPUT_DTYPES
            or t.stride(0) != 1
            or t.numel() != rows // rows_per_request
        ):
            return False
    return True


@functools.cache
def _split_geometry(rows: int, vocab_size: int, num_sms: int) -> tuple[int, int, int]:
    """Return ``(block_v, chunk_size, num_splits)`` for low-row split-CTA softmax."""
    target_ctas = num_sms * 2
    desired_splits = min(
        _MAX_SPLITS,
        triton.cdiv(target_ctas, rows),
        max(1, triton.cdiv(vocab_size, _SPLIT_MIN_BLOCK)),
    )
    block_v = (
        _SPLIT_MIN_BLOCK
        if triton.cdiv(vocab_size, desired_splits) <= _SPLIT_MIN_BLOCK
        else 4096
    )
    chunk_size = triton.cdiv(triton.cdiv(vocab_size, desired_splits), block_v) * block_v
    num_splits = triton.cdiv(vocab_size, chunk_size)
    return block_v, chunk_size, num_splits


def _temperature_softmax_unchecked(
    logits: torch.Tensor,
    temperatures: Optional[torch.Tensor] = None,
    *,
    rows_per_request: int = 1,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    rows, vocab_size = logits.shape
    if out is None:
        out = torch.empty((rows, vocab_size), dtype=torch.float32, device=logits.device)
    if rows == 0 or vocab_size == 0:
        return out

    has_temp = temperatures is not None
    t_flat = temperatures.reshape(-1) if has_temp else out

    if vocab_size <= _SINGLE_PASS_MAX_VOCAB:
        block_v = max(1024, triton.next_power_of_2(vocab_size))
        num_warps = 8 if block_v >= 4096 else 4
        _row_softmax_kernel[(rows,)](
            logits,
            t_flat,
            out,
            logits.stride(0),
            out.stride(0),
            vocab_size,
            rows_per_request,
            HAS_TEMP=has_temp,
            SINGLE_PASS=True,
            BLOCK_V=block_v,
            num_warps=num_warps,
        )
        return out

    num_sms = _device_num_sms(logits.device.index)
    if rows < max(1, num_sms // 2):
        block_v, chunk_size, num_splits = _split_geometry(rows, vocab_size, num_sms)
        if num_splits > 1:
            stats = torch.empty(
                (2, rows, num_splits), dtype=torch.float32, device=logits.device
            )
            partial_max, partial_sum = stats[0], stats[1]
            _split_stats_kernel[(rows, num_splits)](
                logits,
                t_flat,
                partial_max,
                partial_sum,
                logits.stride(0),
                vocab_size,
                chunk_size,
                num_splits,
                rows_per_request,
                HAS_TEMP=has_temp,
                BLOCK_V=block_v,
                num_warps=8,
            )
            _split_write_kernel[(rows, num_splits)](
                logits,
                t_flat,
                partial_max,
                partial_sum,
                out,
                logits.stride(0),
                out.stride(0),
                vocab_size,
                chunk_size,
                num_splits,
                rows_per_request,
                HAS_TEMP=has_temp,
                BLOCK_V=block_v,
                BLOCK_S=triton.next_power_of_2(num_splits),
                num_warps=8,
            )
            return out

    _row_softmax_kernel[(rows,)](
        logits,
        t_flat,
        out,
        logits.stride(0),
        out.stride(0),
        vocab_size,
        rows_per_request,
        HAS_TEMP=has_temp,
        SINGLE_PASS=False,
        BLOCK_V=8192,
        num_warps=8,
    )
    return out


def temperature_softmax(
    logits: torch.Tensor,
    temperatures: Optional[torch.Tensor] = None,
    *,
    rows_per_request: int = 1,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Fused ``softmax(logits / T)`` in fp32, in place when ``out is logits``.

    Covers 2D CUDA/HIP tensors with contiguous inner dim (fp32/bf16/fp16 logits,
    fp32 probs) across arbitrary vocab sizes, row strides, and ``rows_per_request``.
    Uses a 1-pass CTA kernel for small vocabularies (<= 8192), a 2-kernel split-row
    path when ``rows < num_sms // 2`` to keep SMs busy, and a zero-allocation
    single-launch 2-pass online kernel otherwise.
    """
    if not can_softmax(
        logits, temperatures, rows_per_request=rows_per_request, out=out
    ):
        if temperatures is not None:
            t = (
                temperatures.repeat_interleave(rows_per_request, dim=0)
                if rows_per_request > 1
                else temperatures
            )
            if out is logits:
                logits.div_(t)
            else:
                logits = logits / t
        probs = torch.softmax(logits.float(), dim=-1)
        return probs if out is None else out.copy_(probs)

    return _temperature_softmax_unchecked(
        logits, temperatures, rows_per_request=rows_per_request, out=out
    )
