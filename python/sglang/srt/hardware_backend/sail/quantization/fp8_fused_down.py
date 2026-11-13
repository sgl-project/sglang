# SPDX-License-Identifier: Apache-2.0

"""Fuse a dense MLP's activation and channelwise FP8 ``down_proj`` on SAIL.

The fused path quantizes ``silu(gate) * up`` per token and runs the DeepGEMM
FP8 GEMM against ``down_proj``'s weights directly, so ``down_proj.forward`` is
skipped. A layer is eligible only when skipping it loses nothing: no all-reduce,
no LoRA, and a dynamic channelwise FP8 weight that DeepGEMM accepts as stored.
"""

from typing import Optional

import torch

from sglang.kernels.ops.activation.silu_mul_quant import (
    silu_and_mul_post_per_token_quant_fp8,
)
from sglang.srt.environ import envs
from sglang.srt.layers import deep_gemm_wrapper
from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod
from sglang.srt.runtime_context import get_lora
from sglang.srt.utils import get_device_sm


def can_fuse_fp8_down_proj(down_proj: torch.nn.Module, gate_up: torch.Tensor) -> bool:
    if not (
        deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM
        and envs.SGLANG_SAIL_DEEPGEMM_DENSE.get()
        and get_device_sm() >= 89
    ):
        return False
    # Skipping down_proj.forward would drop its all-reduce and LoRA adapters.
    if down_proj.reduce_results or get_lora().enable_lora or get_lora().lora_paths:
        return False
    quant_method = down_proj.quant_method
    if (
        not isinstance(quant_method, Fp8LinearMethod)
        or quant_method.block_quant
        or quant_method.use_marlin
        or getattr(down_proj, "input_scale", None) is not None
    ):
        return False
    if gate_up.dtype != torch.bfloat16 or gate_up.shape[-1] % 16 != 0:
        return False
    # The weight is stored as [K, N] with one float32 scale per output channel.
    weight = getattr(down_proj, "weight", None)
    weight_scale = getattr(down_proj, "weight_scale", None)
    return (
        weight is not None
        and weight.dtype == torch.float8_e4m3fn
        and weight.shape == (gate_up.shape[-1] // 2, down_proj.output_size)
        and weight_scale is not None
        and weight_scale.dtype == torch.float32
        and weight_scale.numel() == weight.shape[1]
    )


def fused_fp8_down_proj(
    down_proj: torch.nn.Module, gate_up: torch.Tensor, swiglu_limit: Optional[float]
) -> torch.Tensor:
    M = gate_up.shape[0]
    down_input_fp8, down_input_scale = silu_and_mul_post_per_token_quant_fp8(
        gate_up, swiglu_limit=swiglu_limit
    )
    down_output = gate_up.new_empty((M, down_proj.output_size), dtype=torch.bfloat16)
    if M != 0:
        deep_gemm_wrapper.gemm_nt_f8f8bf16(
            (down_input_fp8, down_input_scale),
            (down_proj.weight.t(), down_proj.weight_scale),
            down_output,
        )
    return down_output
