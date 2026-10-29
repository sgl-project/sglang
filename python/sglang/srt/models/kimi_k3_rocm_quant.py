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

from sglang.srt.environ import envs


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


def k3_kv_b_for_absorb(
    *, module: nn.Module, weight: torch.Tensor
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """kv_b for the MLA absorb BMMs from a per-channel FP8 weight.

    On gfx950 with AITER, requantize to per-tensor FP8 and return its scalar
    scale, so the absorb runs AITER's a8w8 BMM. Otherwise (or with
    SGLANG_OPT_HIP_K3_MLA_ABSORB_FP8 off) return dense BF16 and no scale. Must
    run before the kv_b head split.
    """
    from sglang.srt.models.deepseek_common import utils as deepseek_utils

    # rocm_absorb_{q,v}_bmm only take the a8w8 BMM on gfx950 + AITER; elsewhere
    # an FP8 w_kc/w_vc would be dequantized back to BF16 after a lossy requant.
    if not (
        deepseek_utils._use_aiter_gfx95 and envs.SGLANG_OPT_HIP_K3_MLA_ABSORB_FP8.get()
    ):
        return _k3_channel_fp8_to_bf16(module, weight), None

    from sglang.srt.layers.quantization.fp8_utils import (
        channel_quant_to_tensor_quant,
    )

    weight_scale = module.weight_scale
    if weight_scale.dim() == 1:
        weight_scale = weight_scale.view(-1, 1)
    return channel_quant_to_tensor_quant(weight, weight_scale)


def _is_quark_channel_fp8(module: nn.Module) -> bool:
    """Quark per-channel FP8 with dynamic activations, before post-processing."""
    from sglang.srt.layers.quantization.quark.quark import QuarkLinearMethod
    from sglang.srt.layers.quantization.quark.schemes import QuarkW8A8Fp8

    if not (
        isinstance(module.quant_method, QuarkLinearMethod)
        and isinstance(module.scheme, QuarkW8A8Fp8)
    ):
        return False
    weight, scale = module.weight, module.weight_scale
    return (
        module.scheme.weight_qscheme == "per_channel"
        and not module.scheme.is_static_input_scheme
        and weight.dim() == 2
        and weight.dtype == torch.float8_e4m3fn
        and isinstance(scale, torch.Tensor)
        and scale.numel() == weight.shape[0]
    )


def k3_kda_inproj_fp8_tail_rows(
    *, wide_rows: int, f_a_proj: nn.Module, b_proj: nn.Module, f_b_proj: nn.Module
) -> int:
    """Rows fused_qkvg_proj reserves after [q,k,v,g] for [f_a | b | pad], or 0.

    Reserving them at init lets k3_merge_kda_inproj_fp8 fill them in place. A
    post-load row-cat would free the wide weight and allocate a slightly larger
    one per layer, which strands caching-allocator segments that empty_cache
    cannot return (~1 GB per GPU at Kimi-K3 TP8).
    """
    if not envs.SGLANG_ROCM_K3_FUSE_KDA_INPROJ.get():
        return 0
    if not all(_is_quark_channel_fp8(m) for m in (f_a_proj, b_proj, f_b_proj)):
        return 0
    rows = f_a_proj.weight.shape[0] + b_proj.weight.shape[0]
    # N % 64 == 0 keeps AITER's preshuffled FP8 GEMM (use_aiter_bpreshuffle_gemm).
    return rows + (-(wide_rows + rows)) % 64


def k3_merge_kda_inproj_fp8(
    *,
    qkvg_proj: nn.Module,
    qkvg_sizes: list[int],
    f_a_proj: nn.Module,
    b_proj: nn.Module,
    f_b_proj: nn.Module,
) -> Optional[tuple[list[int], torch.Tensor]]:
    """Write Quark FP8 [f_a | b | pad] rows and channel scales into qkvg_proj.

    Fills the tail reserved by k3_kda_inproj_fp8_tail_rows in place. Returns
    (split sizes of the merged output, BF16 f_b for the tiny GEMM), or None
    when no tail was reserved. Must run before ``process_weights_after_loading``,
    while weights are [out, in].
    """
    weight = qkvg_proj.weight.data
    wide = sum(qkvg_sizes)
    tail = weight.shape[0] - wide
    if tail == 0:
        return None
    scale = qkvg_proj.weight_scale.data
    mods = (qkvg_proj, f_a_proj, b_proj, f_b_proj)
    n_fa, n_b = f_a_proj.weight.shape[0], b_proj.weight.shape[0]
    if not (
        all(_is_quark_channel_fp8(m) for m in mods)
        and f_a_proj.weight.shape[1] == b_proj.weight.shape[1] == weight.shape[1]
        and tail >= n_fa + n_b
    ):
        raise RuntimeError(
            "fused_qkvg_proj reserved a [f_a | b] tail but the KDA in-proj "
            "weights are not the Quark per-channel FP8 layout it was sized for."
        )
    fa_end, b_end = wide + n_fa, wide + n_fa + n_b
    weight[wide:fa_end].copy_(f_a_proj.weight.data)
    weight[fa_end:b_end].copy_(b_proj.weight.data)
    weight[b_end:].zero_()
    scale[wide:fa_end].copy_(f_a_proj.weight_scale.data.reshape(-1))
    scale[fa_end:b_end].copy_(b_proj.weight_scale.data.reshape(-1))
    scale[b_end:].fill_(1.0)
    sizes = [*qkvg_sizes, n_fa, n_b, tail - n_fa - n_b]
    f_b = _k3_channel_fp8_to_bf16(f_b_proj, f_b_proj.weight.data).contiguous()
    return sizes, f_b
