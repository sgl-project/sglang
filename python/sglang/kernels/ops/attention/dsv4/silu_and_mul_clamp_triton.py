"""Triton implementation of the DeepSeek-V4 ``Expert`` activation (clamped SwiGLU).

Extracted from ``inference/model.py`` of
``deepseek-ai/DeepSeek-V4-Flash-0731`` (``Expert.forward``)::

    gate = self.w1(x).float()
    up = self.w3(x).float()
    if self.swiglu_limit > 0:
        up = torch.clamp(up, min=-self.swiglu_limit, max=self.swiglu_limit)
        gate = torch.clamp(gate, max=self.swiglu_limit)
    x = F.silu(gate) * up
    if weights is not None:
        x = weights * x

This module computes the same fused activation on the GEMM-fused ``[M, 2N]``
gate/up layout (gate in the leading half, up in the trailing half). All math
runs in fp32 with a single cast back to the output dtype, matching the
reference. The 2-D grid adapts to both serving phases:

- prefill (many tokens): wide tiles amortize per-program overhead;
- decode (few tokens): the hidden dim is split into more programs so the
  SMs stay busy even at ``M == 1``.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Launch configuration
# ---------------------------------------------------------------------------
# Heuristic block sizes instead of triton.autotune: the token count M changes
# every decode step, so autotuning keyed on M would re-bench continuously.

_DECODE_M_THRESHOLD = 16
_DECODE_BLOCK_M = 1
_DECODE_BLOCK_N = 256
_DECODE_NUM_WARPS = 4
_PREFILL_BLOCK_M = 8
_PREFILL_BLOCK_N = 1024
_PREFILL_NUM_WARPS = 8


@triton.jit
def _silu_and_mul_clamp_kernel(
    gate_up_ptr,
    out_ptr,
    weights_ptr,
    M,
    N,
    gate_up_row_stride,
    out_row_stride,
    swiglu_limit,
    HAS_SWIGLU_LIMIT: tl.constexpr,
    HAS_WEIGHTS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """out[row, col] = silu(clamp(gate, max=L)) * clamp(up, -L, L) (* w[row]).

    gate lives at ``gate_up[row, col]`` and up at ``gate_up[row, N + col]``;
    ``N`` is the half hidden size of the ``[M, 2N]`` gate_up tensor.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = rows < M
    col_mask = cols < N
    tile_mask = row_mask[:, None] & col_mask[None, :]

    gate_up_ptr += rows[:, None].to(tl.int64) * gate_up_row_stride
    gate = tl.load(gate_up_ptr + cols[None, :], mask=tile_mask, other=0.0)
    up = tl.load(gate_up_ptr + N + cols[None, :], mask=tile_mask, other=0.0)

    gate = gate.to(tl.float32)
    up = up.to(tl.float32)

    if HAS_SWIGLU_LIMIT:
        # gate clamps only from above (silu already saturates to 0 for -inf),
        # up clamps symmetrically -- mirrors Expert.forward.
        gate = tl.minimum(gate, swiglu_limit)
        up = tl.minimum(tl.maximum(up, -swiglu_limit), swiglu_limit)

    y = gate * tl.sigmoid(gate) * up

    if HAS_WEIGHTS:
        w = tl.load(weights_ptr + rows, mask=row_mask, other=1.0).to(tl.float32)
        y = w[:, None] * y

    out_ptr += rows[:, None].to(tl.int64) * out_row_stride
    tl.store(out_ptr + cols[None, :], y.to(out_ptr.dtype.element_ty), mask=tile_mask)


def silu_and_mul_clamp_triton(
    gate_up: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    swiglu_limit: Optional[float] = None,
    weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Clamped-SwiGLU activation from DeepSeek-V4 ``Expert.forward``.

    Args:
        gate_up: ``[..., 2N]`` tensor with the w1 (gate) output in the leading
            half and the w3 (up) output in the trailing half.
        out: optional ``[..., N]`` output tensor; allocated when ``None``.
        swiglu_limit: clamp limit ``L``; ``None`` or ``<= 0`` disables the
            clamp (``gate = min(gate, L)``, ``up = clamp(up, -L, L)``).
        weights: optional per-token routing weights broadcast over the hidden
            dim; must have ``gate_up.numel() // (2N)`` elements.

    Returns:
        The ``[..., N]`` tensor ``silu(clamp(gate, max=L)) * clamp(up, -L, L)``
        (optionally scaled by ``weights``), cast back to ``gate_up.dtype``.
    """
    if gate_up.stride(-1) != 1:
        gate_up = gate_up.contiguous()
    two_n = gate_up.shape[-1]
    assert two_n % 2 == 0, "gate_up last dim must be even ([..., 2N] layout)"
    n = two_n // 2
    m = gate_up.numel() // two_n

    if out is None:
        out = gate_up.new_empty(*gate_up.shape[:-1], n)
    out_2d = out.view(m, n)

    gate_up_2d = gate_up.reshape(m, two_n)

    if weights is not None:
        weights = weights.reshape(-1)
        if weights.stride(0) != 1:
            weights = weights.contiguous()
        assert weights.numel() == m, "weights must have one entry per row"

    if m <= _DECODE_M_THRESHOLD:
        block_m, block_n, num_warps = (
            _DECODE_BLOCK_M,
            _DECODE_BLOCK_N,
            _DECODE_NUM_WARPS,
        )
    else:
        block_m, block_n, num_warps = (
            _PREFILL_BLOCK_M,
            _PREFILL_BLOCK_N,
            _PREFILL_NUM_WARPS,
        )

    if m == 0:
        return out

    has_swiglu_limit = swiglu_limit is not None and swiglu_limit > 0
    grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n))
    _silu_and_mul_clamp_kernel[grid](
        gate_up_2d,
        out_2d,
        weights,
        m,
        n,
        gate_up_2d.stride(0),
        out_2d.stride(0),
        float(swiglu_limit) if has_swiglu_limit else 0.0,
        HAS_SWIGLU_LIMIT=has_swiglu_limit,
        HAS_WEIGHTS=weights is not None,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=num_warps,
    )
    return out
