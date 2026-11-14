# Copyright 2023-2024 SGLang Team
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
"""Kimi-K3 ROCm weight preparation for Quark checkpoints.

Quark serializes K3 as per-output-channel FP8, which is not the layout the
AITER kernels want. The conversions live here; ``kimi_k3.py`` calls every entry
point behind ``_is_hip`` and keeps the checkpoint's own layout otherwise.
"""

from typing import Optional

import torch
from torch import nn

from sglang.srt.utils import print_warning_once


def _k3_channel_fp8_to_bf16(module: nn.Module, weight: torch.Tensor) -> torch.Tensor:
    """Dequantize a per-output-channel FP8 weight to dense bf16.

    The aiter batched absorb GEMM takes ``w_scale`` as a single scalar (the
    Triton kernel documents it as "per-batch scale for WQ with shape (1,)"), so
    a per-channel vector cannot reach it. Requantizing to one per-tensor scale
    would fit, but the scale axis is the GEMM's *contraction* axis, and on real
    K3 weights that second quantization costs ~2.3% relative error against the
    checkpoint (worst channel ~3.8%). Dequantizing instead costs ~0.12% and
    lets the absorb run its bf16 path with w_scale left at the 1.0 default.

    Must run while dim 0 is still the channel axis weight_scale indexes, i.e.
    before the kv_b_proj head split.
    """
    weight_scale = module.weight_scale
    # Per-channel scale is 1D [out]; reshape so it broadcasts against [out, in].
    if weight_scale.dim() == 1:
        weight_scale = weight_scale.view(-1, 1)
    return (weight.to(torch.float32) * weight_scale.to(torch.float32)).to(
        torch.bfloat16
    )


def k3_fused_decode_f_b_weight(f_b_proj: nn.Module) -> Optional[torch.Tensor]:
    """Dense f_b for the fused KDA decode, which takes no weight scale;
    a Quark FP8 f_b becomes a BF16 copy, or None once it is post-processed."""
    from sglang.srt.layers.quantization.quark.quark import QuarkLinearMethod
    from sglang.srt.layers.quantization.quark.schemes import QuarkW8A8Fp8

    weight = f_b_proj.weight.data
    if not (
        isinstance(f_b_proj.quant_method, QuarkLinearMethod)
        and isinstance(f_b_proj.scheme, QuarkW8A8Fp8)
        and f_b_proj.scheme.weight_qscheme == "per_channel"
    ):
        return weight
    # ShardedStateLoader and RemoteModelLoader post-process it before post_load_weights;
    # its transposed (maybe shuffled) dim 0 no longer indexes weight_scale.
    if (
        weight.dtype not in (torch.float8_e4m3fn, torch.float8_e4m3fnuz)
        or weight.shape[0] != f_b_proj.weight_scale.shape[0]
    ):
        print_warning_once(
            "Kimi-K3 fused KDA decode not armed: the Quark f_b was already "
            "post-processed for its FP8 GEMM; decode runs the unfused chain"
        )
        return None
    return _k3_channel_fp8_to_bf16(f_b_proj, weight)
