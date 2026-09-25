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

from types import SimpleNamespace

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


def _k3_channel_fp8_to_tensor_fp8(
    module: nn.Module, weight: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Requantize a per-output-channel FP8 weight to (per-tensor FP8, scalar
    scale) for the aiter absorb GEMM. Must run before the kv_b head split."""
    from sglang.kernels.ops.quantization.fp8_kernel import is_fp8_fnuz
    from sglang.srt.layers.quantization.fp8_utils import (
        channel_quant_to_tensor_quant,
        normalize_e4m3fn_to_e4m3fnuz,
    )

    weight_scale = module.weight_scale
    if is_fp8_fnuz():
        weight, weight_scale, _ = normalize_e4m3fn_to_e4m3fnuz(
            weight=weight, weight_scale=weight_scale, input_scale=None
        )
    if weight_scale.dim() == 1:
        weight_scale = weight_scale.view(-1, 1)
    return channel_quant_to_tensor_quant(weight, weight_scale)


def _k3_is_raw_quark_channel_fp8(module: nn.Module) -> bool:
    """Quark per-channel FP8 with dynamic activations, not yet post-processed."""
    from sglang.srt.layers.quantization.quark.schemes import QuarkW8A8Fp8

    scheme = getattr(module, "scheme", None)
    weight = module.weight
    scale = getattr(module, "weight_scale", None)
    return (
        isinstance(scheme, QuarkW8A8Fp8)
        and scheme.weight_qscheme == "per_channel"
        and not scheme.is_static_input_scheme
        and weight.dim() == 2
        and weight.dtype == torch.float8_e4m3fn
        and isinstance(scale, torch.Tensor)
        and scale.numel() == weight.shape[0]
    )


def _k3_merge_kda_inproj_fp8(self_attn: nn.Module) -> bool:
    """Merge Quark FP8 [q,k,v,g | f_a | b] into one per-channel FP8 linear.

    Runs before process_weights_after_loading, so the weights are still [out, in]
    with an [out] scale. The merged copy serves decode up to the token limit;
    the original linears stay for larger batches.
    """
    if not (envs.SGLANG_ROCM_K3_FUSE_KDA_INPROJ.get() and self_attn.use_full_rank_gate):
        return False
    mods = (self_attn.fused_qkvg_proj, self_attn.f_a_proj, self_attn.b_proj)
    if not all(_k3_is_raw_quark_channel_fp8(m) for m in (*mods, self_attn.f_b_proj)):
        return False
    weights = [m.weight.data for m in mods]
    if len({w.shape[1] for w in weights}) != 1:
        return False
    scales = [m.weight_scale.data.reshape(-1).float() for m in mods]
    sizes = [w.shape[0] for w in weights]
    # N % 64 == 0 keeps aiter's preshuffled FP8 GEMM (use_aiter_bpreshuffle_gemm).
    pad = (-sum(sizes)) % 64
    if pad:
        weights.append(weights[0].new_zeros((pad, weights[0].shape[1])))
        scales.append(scales[0].new_ones(pad))
    scheme = self_attn.fused_qkvg_proj.scheme
    layer = SimpleNamespace(
        weight=torch.cat(weights).contiguous(),
        weight_scale=torch.cat(scales),
        input_scale=None,
        scheme=scheme,
    )
    scheme.process_weights_after_loading(layer)

    self_attn._qkvgbfa_layer = layer
    self_attn._bfa_fa_size, self_attn._bfa_b_size = sizes[1], sizes[2]
    self_attn._qkvgbfa_sizes = [*self_attn.split_sizes, sizes[1], sizes[2], pad]
    # f_b is [heads * head_dim, head_dim]; BF16 lets it use the tiny GEMM and
    # the fused KDA decode.
    self_attn._bfa_f_b_w = _k3_channel_fp8_to_bf16(
        self_attn.f_b_proj, self_attn.f_b_proj.weight.data
    ).contiguous()
    return True


def _k3_densify_quark_shared_experts(mlp: nn.Module) -> None:
    """Dequantize Quark MXFP4 shared experts to BF16 before the MoE front merge.

    With the default bf16 SGLANG_ROCM_QUARK_MXFP4_LINEAR_ACT, Quark dequantizes
    them anyway, but only after load_weights has merged the front. Doing it
    here lets them join the fused front; the later loader pass is a no-op.
    """
    from sglang.srt.layers.quantization.quark.schemes import quark_w4a4_mxfp4

    shared = getattr(mlp, "shared_experts", None)
    if shared is None or not quark_w4a4_mxfp4._dequant_linear_to_bf16:
        return
    for linear in (shared.gate_up_proj, shared.down_proj):
        scheme = getattr(linear, "scheme", None)
        if isinstance(scheme, quark_w4a4_mxfp4.QuarkW4A4MXFP4):
            scheme.process_weights_after_loading(linear)
