# SPDX-License-Identifier: Apache-2.0
"""Residual gating with FP32 intermediates and a single final storage cast."""

import torch
import triton
import triton.language as tl

from sglang.srt.utils.custom_op import register_custom_op


@triton.jit
def _residual_gate_fp32_kernel(
    x,
    update,
    gate,
    out,
    count,
    TOKENS: tl.constexpr,
    HIDDEN: tl.constexpr,
    GATE_BATCH_STRIDE: tl.constexpr,
    GATE_CHANNEL_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    index = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    valid = index < count
    gate_index = (
        index // (TOKENS * HIDDEN) * GATE_BATCH_STRIDE
        + index % HIDDEN * GATE_CHANNEL_STRIDE
    )
    residual = tl.load(x + index, valid, other=0).to(tl.float32)
    value = tl.load(update + index, valid, other=0).to(tl.float32)
    weight = tl.load(gate + gate_index, valid, other=0).to(tl.float32)
    # eager rounds the multiply to FP32, not to the input dtype or an FMA
    result = residual + weight * value
    tl.store(out + index, result, valid)


def _fake_residual_gate_fp32(
    x: torch.Tensor, update: torch.Tensor, gate: torch.Tensor
) -> torch.Tensor:
    return torch.empty_like(x)


@register_custom_op(
    op_name="diffusion_residual_gate_fp32",
    mutates_args=[],
    fake_impl=_fake_residual_gate_fp32,
)
def _residual_gate_fp32(
    x: torch.Tensor, update: torch.Tensor, gate: torch.Tensor
) -> torch.Tensor:
    output = torch.empty_like(x)
    if x.numel():
        with torch.cuda.device(x.device):
            _residual_gate_fp32_kernel[(triton.cdiv(x.numel(), 1024),)](
                x,
                update,
                gate,
                output,
                x.numel(),
                TOKENS=x.shape[1],
                HIDDEN=x.shape[2],
                GATE_BATCH_STRIDE=gate.stride(0) if gate.shape[0] != 1 else 0,
                GATE_CHANNEL_STRIDE=gate.stride(2),
                BLOCK=1024,
                enable_fp_fusion=False,
            )
    return output


def residual_gate_fp32(
    x: torch.Tensor, update: torch.Tensor, gate: torch.Tensor
) -> torch.Tensor:
    """Compute ``(x.float() + gate.float() * update.float()).to(x.dtype)``.

    Unlike ``residual_gate_add``, the product is not rounded to BF16/FP16.
    The CUDA path accepts contiguous [B,S,D] and a [B or 1,1,D] gate.
    """
    if (
        x.is_cuda
        and torch.version.hip is None
        and not torch.is_grad_enabled()
        and x.ndim == 3
        and x.shape == update.shape
        and x.is_contiguous()
        and update.is_contiguous()
        and x.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and update.dtype == x.dtype
        and gate.dtype in (x.dtype, torch.float32)
        and gate.device == update.device == x.device
        and gate.ndim == 3
        and gate.shape[0] in (1, x.shape[0])
        and gate.shape[1:] == (1, x.shape[2])
    ):
        return _residual_gate_fp32(x, update, gate)
    return (x.float() + gate.float() * update.float()).to(x.dtype)
