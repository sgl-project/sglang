# SPDX-License-Identifier: Apache-2.0
"""Beam attention gate with activation-dtype bias and sigmoid rounding."""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from sglang.kernels.fused_op import BaseFusedOp
from sglang.kernels.spec import CapabilityRequirement, KernelBackend
from sglang.srt.utils.custom_op import register_custom_op


@triton.jit
def _fused_attention_gate_kernel(
    out_ptr,
    attn_ptr,
    gate_ptr,
    bias_ptr,
    hidden_dim: tl.constexpr,
    gate_row_stride: tl.constexpr,
    gate_col_stride: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    block = tl.program_id(1)
    offsets = block * BLOCK_H + tl.arange(0, BLOCK_H)
    mask = offsets < hidden_dim
    head = offsets // HEAD_DIM
    act_dtype = attn_ptr.dtype.element_ty

    off = row * hidden_dim + offsets
    attn = tl.load(attn_ptr + off, mask=mask, other=0.0).to(tl.float32)
    gate_off = row * gate_row_stride + offsets * gate_col_stride
    g = tl.load(gate_ptr + gate_off, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(bias_ptr + head, mask=mask, other=0.0).to(tl.float32)
    # As in eager, gate + bias, its sigmoid, and the product round to act_dtype.
    g = (g + b).to(act_dtype).to(tl.float32)
    if act_dtype == tl.float16:
        # Approximate exp/div can cross an FP16 rounding boundary near zero.
        m = tl.div_rn(1.0, 1.0 + libdevice.exp(-g))
    else:
        m = tl.sigmoid(g)
    m = m.to(act_dtype).to(tl.float32)
    tl.store(out_ptr + off, (attn * m).to(act_dtype), mask=mask)


def _fake_attention_gate(attention_output, gate, gate_bias, head_dim):
    return torch.empty_like(attention_output, memory_format=torch.contiguous_format)


@torch.compiler.disable
def _eager_sigmoid(x: torch.Tensor) -> torch.Tensor:
    # Inductor's sigmoid can round FP16 one ULP away from ATen's.
    return torch.sigmoid(x)


@register_custom_op(fake_impl=_fake_attention_gate)
def fused_attention_gate(
    attention_output: torch.Tensor,
    gate: torch.Tensor,
    gate_bias: torch.Tensor,
    head_dim: int,
) -> torch.Tensor:
    """`attention_output * sigmoid(gate + gate_bias[head])` for 2-D `[tokens, heads*head_dim]` inputs."""
    if attention_output.shape != gate.shape or attention_output.ndim != 2:
        raise ValueError("attention_output and gate must be matching 2-D tensors")
    if attention_output.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("attention_output must be fp16, bf16 or fp32")
    attention_output = attention_output.contiguous()
    hidden_dim = attention_output.shape[1]
    if gate_bias.numel() * head_dim != hidden_dim or not gate_bias.is_contiguous():
        raise ValueError("gate_bias must hold one contiguous entry per local head")
    out = torch.empty_like(attention_output)
    rows = attention_output.shape[0]
    if rows == 0:
        return out
    block_h = 1024
    _fused_attention_gate_kernel[(rows, triton.cdiv(hidden_dim, block_h))](
        out,
        attention_output,
        gate,
        gate_bias,
        hidden_dim=hidden_dim,
        gate_row_stride=gate.stride(0),
        gate_col_stride=gate.stride(1),
        HEAD_DIM=head_dim,
        BLOCK_H=block_h,
        num_warps=4,
    )
    return out


class BeamAttentionGateOp(BaseFusedOp):
    """Apply a per-head bias and sigmoid gate to flat attention output."""

    op = "elementwise.beam_attention_gate"
    capabilities = {KernelBackend.TRITON: {CapabilityRequirement.CUDA}}

    def _torch_compile_forward(self, num_tokens: int):
        # Native compilation can remove the required intermediate rounding.
        return None

    def forward_torch_compile(self, *args, **kwargs):
        if self._compiled_native is None:
            # Without emulated casts, Inductor drops eager's intermediate rounding.
            self._compiled_native = torch.compile(
                self.forward_native, options={"emulate_precision_casts": True}
            )
        return self._compiled_native(*args, **kwargs)

    def forward_native(
        self,
        attention_output: torch.Tensor,
        gate: torch.Tensor,
        gate_bias: torch.Tensor,
        head_dim: int,
    ) -> torch.Tensor:
        heads = gate_bias.numel()
        multiplier = _eager_sigmoid(
            gate.reshape(-1, heads, head_dim) + gate_bias[None, :, None]
        )
        return (attention_output.reshape(-1, heads, head_dim) * multiplier).reshape_as(
            attention_output
        )

    def forward_triton(
        self,
        attention_output: torch.Tensor,
        gate: torch.Tensor,
        gate_bias: torch.Tensor,
        head_dim: int,
    ) -> torch.Tensor:
        return fused_attention_gate(attention_output, gate, gate_bias, head_dim)
