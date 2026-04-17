"""Fused FP8 per-group quantization + scatter to 3D MoE input.

Replaces two separate operations in moe_ep_deepgemm_preprocess:
  1. per_token_group_quant_fp8(hidden_states) → fp8_hidden + scale
  2. fill_gateup_input(fp8_hidden, scale, src2dst → gateup_3d)

with a single kernel that reads BF16 hidden_states once, quantizes
on the fly, and scatters directly to the 3D output.

Saves one intermediate FP8 tensor allocation and one kernel launch.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch

from sglang.jit_kernel.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_fused_quant_scatter_module(dtype: torch.dtype) -> Module:
    args = make_cpp_args(dtype)
    return load_jit(
        "fused_quant_scatter",
        *args,
        cuda_files=["moe/fused_quant_scatter.cuh"],
        cuda_wrappers=[
            ("fused_quant_scatter", f"fused_quant_scatter<{args}>"),
        ],
    )


def fused_quant_scatter(
    hidden_states: torch.Tensor,       # [num_tokens, hidden_size] BF16
    src2dst: torch.Tensor,             # [num_tokens * topk] int32
    topk_ids: torch.Tensor,            # [num_tokens * topk] int32 (flat)
    gateup_input: torch.Tensor,        # [num_experts, m_max, hidden_size] FP8 (pre-allocated)
    gateup_input_scale: torch.Tensor,  # [num_experts, m_max, hidden_size//group_size] FP32
    topk: int,
    group_size: int = 128,
    eps: float = 1e-10,
) -> None:
    """
    Fused FP8 quant + scatter: reads BF16 hidden_states, writes FP8 + scale
    directly into 3D gateup_input at positions determined by src2dst.

    This replaces:
        hidden_fp8, scale = per_token_group_quant_fp8(hidden_states, group_size)
        fill_gateup_input(hidden_fp8, scale, src2dst, topk_ids, ...)
    """
    assert hidden_states.is_cuda and hidden_states.dtype == torch.bfloat16
    assert gateup_input.dtype == torch.float8_e4m3fn
    assert src2dst.dtype == torch.int32
    assert topk_ids.numel() == hidden_states.shape[0] * topk

    fp8_max = 448.0  # E4M3 max

    if hidden_states.shape[0] == 0:
        return

    module = _jit_fused_quant_scatter_module(hidden_states.dtype)
    module.fused_quant_scatter(
        hidden_states,
        gateup_input.view(-1, hidden_states.shape[1]),  # flatten 3D → 2D for kernel
        gateup_input_scale.view(-1, gateup_input_scale.shape[-1]),
        src2dst,
        topk_ids.flatten(),
        group_size,
        topk,
        eps,
        fp8_max,
    )
