# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Beam attention gate with activation-dtype bias and sigmoid rounding."""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from sglang.srt.utils.custom_op import register_custom_op


@triton.jit
def _fused_attention_gate_kernel(
    out_ptr,
    attn_ptr,
    gate_ptr,
    bias_ptr,
    hidden_dim: tl.constexpr,
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
    g = tl.load(gate_ptr + off, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(bias_ptr + head, mask=mask, other=0.0).to(tl.float32)
    # Eager: (gate + bias) rounds to bf16, sigmoid rounds to bf16, product rounds.
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
    attention_output = attention_output.contiguous()
    gate = gate.contiguous()
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
        HEAD_DIM=head_dim,
        BLOCK_H=block_h,
        num_warps=4,
    )
    return out
