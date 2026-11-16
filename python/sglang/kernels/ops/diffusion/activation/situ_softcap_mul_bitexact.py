# SPDX-License-Identifier: Apache-2.0
"""Fuse MOSSImage's bounded SiTU activation, preserving each BF16 rounding.

The eager formula is ``4*tanh(gate/4)*sigmoid(gate) * 25*tanh(hidden/25)``.
Each of its nine elementwise operations rounds to BF16. The operands may be
separate projections or strided halves of one packed FFN projection.
Large inputs use exact per-device tables for the two independent factors;
small inputs and cold CUDA graph captures retain the direct fused formula.

Verified on H200 for dense and packed rows with widths 128 and 8192,
including [1, 6620, 8192], [1, 4109, 8192], [1, 2524, 8192] and
[2, 65, 8192], plus every BF16 encoding of each operand.
"""

from __future__ import annotations

import threading

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

from sglang.kernels.numerics import (
    mul_rn_f32,
    round_bf16_to_fp32,
)
from sglang.kernels.ops.diffusion.common.platform import is_cuda
from sglang.srt.utils.custom_op import register_custom_op

_SITU_TABLES: dict[torch.device, tuple[torch.Tensor, torch.Tensor]] = {}
_SITU_TABLES_LOCK = threading.Lock()


def _get_situ_tables(device):
    tables = _SITU_TABLES.get(device)
    if tables is not None:
        return tables
    if torch.compiler.is_compiling() or torch.cuda.is_current_stream_capturing():
        return None
    with _SITU_TABLES_LOCK:
        tables = _SITU_TABLES.get(device)
        if tables is None:
            # Enumerate bit patterns, not evenly spaced floating-point values.
            # Generate with the live aten CUDA formula, including BF16 rounding.
            values = (
                torch.arange(65536, device=device, dtype=torch.int32)
                .to(torch.int16)
                .view(torch.bfloat16)
            )
            tables = (
                25.0 * torch.tanh(values / 25.0),
                (4.0 * torch.tanh(values / 4.0)) * torch.sigmoid(values),
            )
            # Publish only complete tables so subsequent streams need no wait.
            # Keep their 256 KiB of storage for the process lifetime: captured
            # graphs may retain these addresses after the originating request.
            torch.cuda.current_stream(device).synchronize()
            _SITU_TABLES[device] = tables
    return tables


@triton.jit
def _situ_lookup_kernel(
    out_ptr,
    hidden_ptr,
    gate_ptr,
    hidden_table,
    gate_table,
    numel,
    hidden_row_stride,
    gate_row_stride,
    DIM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < numel
    rows, cols = offsets // DIM, offsets % DIM
    hidden = tl.load(hidden_ptr + rows * hidden_row_stride + cols, mask, other=0).to(
        tl.uint16, bitcast=True
    )
    gate = tl.load(gate_ptr + rows * gate_row_stride + cols, mask, other=0).to(
        tl.uint16, bitcast=True
    )
    hidden_factor = tl.load(hidden_table + hidden.to(tl.int32)).to(tl.float32)
    gate_factor = tl.load(gate_table + gate.to(tl.int32)).to(tl.float32)
    tl.store(out_ptr + offsets, mul_rn_f32(gate_factor, hidden_factor), mask)


@triton.jit
def _tanh_bf16(value):
    # tanh(x) rounds back to x in this range. Avoid libdevice's FTZ path so
    # BF16 subnormals retain the same value as aten's tanh kernel.
    return round_bf16_to_fp32(
        tl.where(tl.abs(value) < 1.0e-4, value, libdevice.tanh(value))
    )


@triton.jit
def _situ_kernel(
    out_ptr,
    hidden_ptr,
    gate_ptr,
    numel,
    hidden_row_stride,
    gate_row_stride,
    DIM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < numel
    rows, cols = offsets // DIM, offsets % DIM
    hidden = tl.load(
        hidden_ptr + rows * hidden_row_stride + cols, mask=mask, other=0.0
    ).to(tl.float32)
    gate = tl.load(gate_ptr + rows * gate_row_stride + cols, mask=mask, other=0.0).to(
        tl.float32
    )

    # Explicit non-FTZ multiplies also preserve subnormal intermediate values.
    gate_quarter = round_bf16_to_fp32(mul_rn_f32(gate, 0.25))
    gate_tanh = _tanh_bf16(gate_quarter)
    gate_scaled = round_bf16_to_fp32(mul_rn_f32(gate_tanh, 4.0))
    gate_sigmoid = round_bf16_to_fp32(tl.sigmoid(gate))
    gate_factor = round_bf16_to_fp32(mul_rn_f32(gate_scaled, gate_sigmoid))
    # This reciprocal and BF16 boundary match the eager scalar division.
    hidden_scaled = round_bf16_to_fp32(mul_rn_f32(hidden, 1.0 / 25.0))
    hidden_factor = round_bf16_to_fp32(mul_rn_f32(25.0, _tanh_bf16(hidden_scaled)))
    result = mul_rn_f32(gate_factor, hidden_factor)
    # The integer BF16 rounding helper is for finite values; preserve NaNs.
    result = tl.where((hidden == hidden) & (gate == gate), result, float("nan"))
    tl.store(out_ptr + offsets, result, mask=mask)


def can_use_fused_situ_softcap_mul(*, device: torch.device, dtype: torch.dtype) -> bool:
    return is_cuda() and device.type == "cuda" and dtype == torch.bfloat16


def _valid_situ_inputs(hidden: torch.Tensor, gate: torch.Tensor) -> bool:
    return (
        is_cuda()
        and hidden.is_cuda
        and gate.is_cuda
        and hidden.device == gate.device
        and hidden.dtype == gate.dtype == torch.bfloat16
        and hidden.shape == gate.shape
        and hidden.ndim == 3
        and hidden.numel() > 0
        and hidden.stride(-1) == gate.stride(-1) == 1
        and hidden.stride(0) == hidden.shape[1] * hidden.stride(1)
        and gate.stride(0) == gate.shape[1] * gate.stride(1)
        and hidden.stride(1) >= hidden.shape[2]
        and gate.stride(1) >= gate.shape[2]
    )


def _fake_situ(hidden: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(hidden, memory_format=torch.contiguous_format)


@register_custom_op(
    op_name="triton_fused_situ_softcap_mul_bitexact",
    mutates_args=[],
    fake_impl=_fake_situ,
)
def fused_situ_softcap_mul_bitexact(
    hidden: torch.Tensor, gate: torch.Tensor
) -> torch.Tensor:
    """Evaluate SiTU for BF16 ``[B, S, D]`` operands with packed or dense rows."""
    if not _valid_situ_inputs(hidden, gate):
        raise ValueError("SiTU requires matching CUDA BF16 tensors with dense rows.")
    out = torch.empty_like(hidden, memory_format=torch.contiguous_format)
    with torch.cuda.device(hidden.device):
        tables = _get_situ_tables(hidden.device) if hidden.numel() >= 65536 else None
        if tables is not None:
            _situ_lookup_kernel[(triton.cdiv(hidden.numel(), 1024),)](
                out,
                hidden,
                gate,
                *tables,
                hidden.numel(),
                hidden.stride(1),
                gate.stride(1),
                DIM=hidden.shape[-1],
                BLOCK=1024,
                num_warps=4,
                enable_fp_fusion=False,
            )
            return out
        _situ_kernel[(triton.cdiv(hidden.numel(), 1024),)](
            out,
            hidden,
            gate,
            hidden.numel(),
            hidden.stride(1),
            gate.stride(1),
            DIM=hidden.shape[-1],
            BLOCK=1024,
            enable_fp_fusion=False,
        )
    return out
