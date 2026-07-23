"""Fused ``silu(gate + Δgate) * (up + Δup)`` for the experimental marlin MoE-LoRA path.

Replaces the stock sequence (in-place LoRA delta add over ``[T, 2N]`` +
``silu_and_mul``) with a single kernel: one read of the GEMM output, one read
of the delta, one write of the activation — and it lets the delta live in its
own buffer, which is what makes the side-stream gate_up overlap possible (the
side stream must not race the marlin GEMM writing the shared cache).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _silu_mul_add_delta_kernel(
    x_ptr,  # [T, 2N] gemm1 output, [gate || up]
    d_ptr,  # [T, 2N] LoRA gate_up delta, same layout
    out_ptr,  # [T, N]
    N,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    block = tl.program_id(1)
    cols = block * BLOCK + tl.arange(0, BLOCK)
    mask = cols < N

    x_row = x_ptr + row * 2 * N
    d_row = d_ptr + row * 2 * N
    gate = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(x_row + N + cols, mask=mask, other=0.0).to(tl.float32)
    gate += tl.load(d_row + cols, mask=mask, other=0.0).to(tl.float32)
    up += tl.load(d_row + N + cols, mask=mask, other=0.0).to(tl.float32)

    silu = gate * tl.sigmoid(gate)
    out = silu * up
    tl.store(out_ptr + row * N + cols, out.to(out_ptr.dtype.element_ty), mask=mask)


def silu_and_mul_add_delta(
    gemm1_out: torch.Tensor,
    gate_up_delta: torch.Tensor,
    out: torch.Tensor,
) -> None:
    """``out = silu(g + dg) * (u + du)`` over contiguous ``[T, 2N]`` inputs."""
    T, two_n = gemm1_out.shape
    N = two_n // 2
    assert gate_up_delta.shape == (T, two_n) and out.shape == (T, N)
    assert gemm1_out.is_contiguous() and gate_up_delta.is_contiguous()
    assert out.is_contiguous()
    if T == 0:
        return
    BLOCK = 512
    _silu_mul_add_delta_kernel[(T, triton.cdiv(N, BLOCK))](
        gemm1_out,
        gate_up_delta,
        out,
        N,
        BLOCK=BLOCK,
    )
