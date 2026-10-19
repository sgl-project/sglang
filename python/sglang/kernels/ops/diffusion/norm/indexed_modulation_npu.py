# SPDX-License-Identifier: Apache-2.0
"""Triton-Ascend kernels for indexed MiniMax H3 modulation."""

from functools import lru_cache

import torch
import triton
import triton.language as tl
from triton.language.extra.cann import libdevice
from triton.runtime import driver


@lru_cache(maxsize=None)
def _vector_cores(device: torch.device) -> int:
    index = device.index if device.index is not None else torch.npu.current_device()
    return driver.active.utils.get_device_properties(index)["num_vectorcore"]


@triton.jit
def _rmsnorm_modulate_kernel(
    X,
    W,
    SHIFT,
    SCALE,
    INDICES,
    Y,
    Q,
    QS,
    ROWS,
    WIDTH: tl.constexpr,
    EPS: tl.constexpr,
    SHIFT_STRIDE: tl.constexpr,
    SCALE_STRIDE: tl.constexpr,
    INDEX_STRIDE: tl.constexpr,
    MODULATE: tl.constexpr,
    QUANTIZE: tl.constexpr,
    KEEP_FLOAT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    columns = tl.arange(0, BLOCK)
    mask = columns < WIDTH
    for row in range(tl.program_id(0), ROWS, tl.num_programs(0)):
        offset = tl.cast(row, tl.int64) * WIDTH + columns
        x = tl.load(X + offset, mask, other=0).to(tl.float32)
        weight = tl.load(W + columns, mask, other=0).to(tl.float32)
        inv_rms = tl.rsqrt(tl.sum(x * x, 0) / WIDTH + EPS)
        # Retain the eager dtype boundaries before quantizing the modulated row.
        y = (
            (x * inv_rms * weight)
            .to(X.dtype.element_ty, fp_downcast_rounding="rtne")
            .to(tl.float32)
        )
        if MODULATE:
            index = tl.load(INDICES + row * INDEX_STRIDE).to(tl.int64)
            scale = tl.load(SCALE + index * SCALE_STRIDE + columns, mask, other=0).to(
                tl.float32
            )
            shift = tl.load(SHIFT + index * SHIFT_STRIDE + columns, mask, other=0).to(
                tl.float32
            )
            factor = (
                (1.0 + scale)
                .to(X.dtype.element_ty, fp_downcast_rounding="rtne")
                .to(tl.float32)
            )
            y = (
                (y * factor)
                .to(X.dtype.element_ty, fp_downcast_rounding="rtne")
                .to(tl.float32)
            )
            y = (
                (y + shift)
                .to(X.dtype.element_ty, fp_downcast_rounding="rtne")
                .to(tl.float32)
            )
        if KEEP_FLOAT:
            tl.store(Y + offset, y, mask)
        if QUANTIZE:
            amax = tl.max(tl.where(mask, tl.abs(y), 0.0), 0)
            quant_scale = amax / 127.0
            divisor = tl.where(quant_scale > 0.0, quant_scale, 1.0)
            quantized = libdevice.nearbyint(y / divisor)
            quantized = tl.minimum(tl.maximum(quantized, -127.0), 127.0)
            tl.store(Q + offset, quantized.to(tl.int8), mask)
            tl.store(QS + row, quant_scale)


@triton.jit
def _indexed_gate_kernel(
    X,
    GATE,
    OTHER,
    INDICES,
    Y,
    ROWS,
    WIDTH: tl.constexpr,
    GATE_STRIDE: tl.constexpr,
    INDEX_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    columns = tl.arange(0, BLOCK)
    mask = columns < WIDTH
    for row in range(tl.program_id(0), ROWS, tl.num_programs(0)):
        offset = tl.cast(row, tl.int64) * WIDTH + columns
        index = tl.load(INDICES + row * INDEX_STRIDE).to(tl.int64)
        x = tl.load(X + offset, mask, other=0).to(tl.float32)
        other = tl.load(OTHER + offset, mask, other=0).to(tl.float32)
        gate = tl.load(GATE + index * GATE_STRIDE + columns, mask, other=0).to(
            tl.float32
        )
        delta = (
            (gate * other)
            .to(X.dtype.element_ty, fp_downcast_rounding="rtne")
            .to(tl.float32)
        )
        y = (x + delta).to(Y.dtype.element_ty, fp_downcast_rounding="rtne")
        tl.store(Y + offset, y, mask)


def rmsnorm_modulate(x, weight, shift, scale, indices, eps, quantize, keep_float):
    rows, width = x.shape
    out = torch.empty_like(x) if keep_float else None
    data = torch.empty(x.shape, device=x.device, dtype=torch.int8) if quantize else None
    scales = (
        torch.empty(rows, device=x.device, dtype=torch.float32) if quantize else None
    )
    modulate = shift is not None
    _rmsnorm_modulate_kernel[(min(rows, _vector_cores(x.device)),)](
        x,
        weight,
        shift if modulate else x,
        scale if modulate else x,
        indices if modulate else x,
        out if keep_float else x,
        data if quantize else x,
        scales if quantize else x,
        rows,
        width,
        eps,
        shift.stride(0) if modulate else 0,
        scale.stride(0) if modulate else 0,
        indices.stride(0) if modulate else 0,
        modulate,
        quantize,
        keep_float,
        triton.next_power_of_2(width),
        enable_fp_fusion=False,
    )
    return out, data, scales


def indexed_gate(x, gate, other, indices, allow_inplace):
    rows, width = x.shape
    out = x if allow_inplace else torch.empty_like(x)
    _indexed_gate_kernel[(min(rows, _vector_cores(x.device)),)](
        x,
        gate,
        other,
        indices,
        out,
        rows,
        width,
        gate.stride(0),
        indices.stride(0),
        triton.next_power_of_2(width),
        enable_fp_fusion=False,
    )
    return out
