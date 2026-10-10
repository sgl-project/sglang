"""Standalone KDA output finalization for Hopper's small-M FP8 GEMV."""

import torch
import triton
import triton.language as tl

_KDA_HEAD_DIM = 128
_FP8_E4M3_MAX = tl.constexpr(448.0)
_FP8_E4M3_MAX_INV = tl.constexpr(1.0 / 448.0)
_FP8_SCALE_EPS = tl.constexpr(1e-10)


@triton.jit
def _fused_kda_output_quant_kernel(
    x,
    gate,
    weight,
    output_q,
    output_scale,
    stride_x_t: tl.constexpr,
    stride_x_h: tl.constexpr,
    stride_x_d: tl.constexpr,
    stride_gate_t: tl.constexpr,
    stride_gate_h: tl.constexpr,
    stride_gate_d: tl.constexpr,
    eps,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    row = tl.program_id(0)
    token = row // NUM_HEADS
    head = row % NUM_HEADS
    offsets = tl.arange(0, HEAD_DIM)

    x_ptrs = x + token * stride_x_t + head * stride_x_h + offsets * stride_x_d
    gate_ptrs = (
        gate + token * stride_gate_t + head * stride_gate_h + offsets * stride_gate_d
    )
    values = tl.load(x_ptrs).to(tl.float32)
    gates = tl.load(gate_ptrs).to(tl.float32)
    weights = tl.load(weight + offsets).to(tl.float32)

    variance = tl.sum(values * values) / HEAD_DIM
    rstd = 1.0 / tl.sqrt(variance + eps)

    # Match FusedRMSNormGated's operation order and its BF16 output boundary.
    normalized = values * rstd
    finalized = normalized * weights
    finalized = finalized * tl.sigmoid(gates)
    finalized = finalized.to(tl.bfloat16).to(tl.float32)

    absmax = tl.maximum(tl.max(tl.abs(finalized)), _FP8_SCALE_EPS)
    # Match per_token_group_quant's two independently rounded FP32 paths.
    scale = absmax * _FP8_E4M3_MAX_INV
    quant_scale = _FP8_E4M3_MAX / absmax
    quantized = tl.clamp(finalized * quant_scale, -_FP8_E4M3_MAX, _FP8_E4M3_MAX).to(
        output_q.dtype.element_ty
    )

    tl.store(output_q + row * HEAD_DIM + offsets, quantized)
    tl.store(output_scale + row, scale)


def fused_kda_output_quant(
    x: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply sigmoid-gated RMSNorm and per-head FP8 quantization in one kernel.

    This is the exact standalone replacement for KDA's legacy
    ``FusedRMSNormGated`` followed by group-size-128 activation quantization.
    It intentionally retains the legacy BF16 rounding between the two steps.
    """
    assert x.ndim == 4 and x.shape[0] == 1
    _, num_tokens, num_heads, head_dim = x.shape
    assert 1 <= num_tokens <= 8
    assert head_dim == _KDA_HEAD_DIM
    assert x.dtype == torch.bfloat16 and x.is_cuda
    assert x.stride(-1) == 1

    assert gate.shape == (num_tokens, num_heads, head_dim)
    assert gate.dtype == torch.bfloat16 and gate.device == x.device
    assert gate.stride(-1) == 1

    assert weight.shape == (head_dim,)
    assert weight.dtype == torch.bfloat16 and weight.device == x.device
    assert weight.stride(0) == 1
    assert eps > 0

    output_q = torch.empty(
        (num_tokens, num_heads * head_dim),
        dtype=torch.float8_e4m3fn,
        device=x.device,
    )
    output_scale = torch.empty(
        (num_tokens, num_heads), dtype=torch.float32, device=x.device
    )

    _fused_kda_output_quant_kernel[(num_tokens * num_heads,)](
        x=x,
        gate=gate,
        weight=weight,
        output_q=output_q,
        output_scale=output_scale,
        stride_x_t=x.stride(1),
        stride_x_h=x.stride(2),
        stride_x_d=x.stride(3),
        stride_gate_t=gate.stride(0),
        stride_gate_h=gate.stride(1),
        stride_gate_d=gate.stride(2),
        eps=eps,
        NUM_HEADS=num_heads,
        HEAD_DIM=head_dim,
        num_warps=1,
        num_stages=1,
    )
    return output_q, output_scale
