"""Fused sigmoid-gate-multiply Triton kernels.

Two variants:
- ``sigmoid_gate_mul``: element-wise ``x * sigmoid(gate)`` when x and gate
  have identical shapes.
- ``sigmoid_gate_mul_broadcast``: broadcast ``x * sigmoid(gate)`` when gate
  is ``(N, 1)`` and x is ``(N, D)``.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from sglang.srt.utils import is_hip

_is_hip = is_hip()


@triton.jit
def _sigmoid_gate_mul_kernel(
    x_ptr,
    gate_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)
    g = tl.load(gate_ptr + offsets, mask=mask).to(tl.float32)
    out = x * tl.sigmoid(g)
    tl.store(out_ptr + offsets, out.to(x_ptr.dtype.element_ty), mask=mask)


def sigmoid_gate_mul(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """Compute ``x * sigmoid(gate)`` in a single fused kernel (same-shape)."""
    out = torch.empty_like(x)
    n = x.numel()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    _sigmoid_gate_mul_kernel[grid](x, gate, out, n, BLOCK_SIZE=1024)
    return out


@triton.jit
def _sigmoid_gate_mul_broadcast_kernel(
    out_ptr,
    gate_ptr,
    x_ptr,
    hidden_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    g = tl.load(gate_ptr + row).to(tl.float32)
    g = tl.sigmoid(g)

    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < hidden_dim
    x = tl.load(x_ptr + row * hidden_dim + offs, mask=mask).to(tl.float32)
    out = x * g
    tl.store(
        out_ptr + row * hidden_dim + offs,
        out.to(x_ptr.dtype.element_ty),
        mask=mask,
    )


def sigmoid_gate_mul_broadcast(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """Compute ``x * sigmoid(gate)`` where gate is (N, 1) and x is (N, D)."""
    bs, hidden_dim = x.shape
    out = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(hidden_dim)
    max_warps = 16 if _is_hip else 32
    num_warps = max(
        min(triton.next_power_of_2(triton.cdiv(hidden_dim, 8 * 32)), max_warps), 4
    )
    _sigmoid_gate_mul_broadcast_kernel[(bs,)](
        out,
        gate,
        x,
        hidden_dim=hidden_dim,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return out


@triton.jit
def _sigmoid_gate_mul_broadcast_add_kernel(
    acc_ptr,
    gate_ptr,
    x_ptr,
    hidden_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    g = tl.load(gate_ptr + row).to(tl.float32)
    g = tl.sigmoid(g)

    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < hidden_dim
    idx = row * hidden_dim + offs
    x = tl.load(x_ptr + idx, mask=mask).to(tl.float32)
    acc = tl.load(acc_ptr + idx, mask=mask).to(tl.float32)
    out = acc + x * g
    tl.store(
        acc_ptr + idx,
        out.to(acc_ptr.dtype.element_ty),
        mask=mask,
    )


def sigmoid_gate_mul_broadcast_add(
    acc: torch.Tensor, x: torch.Tensor, gate: torch.Tensor
) -> torch.Tensor:
    """Fused ``acc += x * sigmoid(gate)`` (gate is ``(N, 1)``, x/acc are ``(N, D)``).

    Computes the shared-expert sigmoid gate, the broadcast multiply, and the
    residual add into ``acc`` in a single Triton kernel. ``acc`` is updated
    in-place and also returned.
    """
    bs, hidden_dim = x.shape
    # The kernel addresses both tensors row-major; ensure contiguity.
    if not acc.is_contiguous():
        acc = acc.contiguous()
    if not x.is_contiguous():
        x = x.contiguous()
    BLOCK_SIZE = triton.next_power_of_2(hidden_dim)
    max_warps = 16 if _is_hip else 32
    num_warps = max(
        min(triton.next_power_of_2(triton.cdiv(hidden_dim, 8 * 32)), max_warps), 4
    )
    _sigmoid_gate_mul_broadcast_add_kernel[(bs,)](
        acc,
        gate,
        x,
        hidden_dim=hidden_dim,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return acc
