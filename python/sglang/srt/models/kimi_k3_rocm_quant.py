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
from sglang.srt.models.kimi_k3_rocm_fusion import _k3_log_once, _k3_ptpc_fp8


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


def _k3_kda_inproj_channel_fp8_scales(
    self_attn: nn.Module,
) -> Optional[list[torch.Tensor]]:
    """Per-channel FP8 scales of the three KDA input projections, if uniform.

    Quark stores self_attn.* as [out, in] e4m3 plus an [out] fp32 scale --
    already the PTPC layout -- so the merge can reuse them instead of
    re-quantizing a dequantized copy."""
    if not (
        _k3_ptpc_fp8
        and envs.SGLANG_ROCM_K3_FUSE_KDA_INPROJ.get()
        and self_attn.do_fuse_qkvbfg
        and self_attn.use_full_rank_gate
    ):
        return None
    scales = []
    for module in (self_attn.fused_qkvg_proj, self_attn.f_a_proj, self_attn.b_proj):
        weight = module.weight
        scale = getattr(module, "weight_scale", None)
        if (
            type(weight.data) is not torch.Tensor
            or weight.dim() != 2
            or weight.dtype != torch.float8_e4m3fn
            or not isinstance(scale, torch.Tensor)
            or scale.numel() != weight.shape[0]
        ):
            return None
        scales.append(scale.data.reshape(-1).float())
    return scales


def _k3_merge_kda_inproj_fp8(self_attn: nn.Module) -> bool:
    """Merge the per-channel FP8 KDA input projections into one PTPC GEMM."""
    scales = _k3_kda_inproj_channel_fp8_scales(self_attn)
    if scales is None:
        return False
    from sglang.kernels.ops.gemm import ptpc_fp8_aiter_hip

    if not ptpc_fp8_aiter_hip.available():
        return False
    mods = (self_attn.fused_qkvg_proj, self_attn.f_a_proj, self_attn.b_proj)
    weights = [module.weight.data for module in mods]
    if len({weight.shape[1] for weight in weights}) != 1:
        return False
    sizes = [weight.shape[0] for weight in weights]
    # Same 8-row pad as the BF16 merge: it keeps every fused-output row
    # 16-byte aligned for the vectorized consumers of the split slices.
    pad = (-sum(sizes)) % 8
    if pad:
        weights.append(weights[0].new_zeros((pad, weights[0].shape[1])))
        scales.append(scales[0].new_ones(pad))
    (
        self_attn._qkvgbfa_fp8_w,
        self_attn._qkvgbfa_fp8_s,
        self_attn._qkvgbfa_fp8_n,
    ) = ptpc_fp8_aiter_hip.pack_prequantized(
        torch.cat(weights, dim=0), torch.cat(scales)
    )
    self_attn._bfa_fa_size, self_attn._bfa_b_size = sizes[1], sizes[2]
    self_attn._qkvgbfa_sizes = [*self_attn.split_sizes, sizes[1], sizes[2], pad]
    ptpc_fp8_aiter_hip.warmup(
        self_attn._qkvgbfa_fp8_w,
        self_attn._qkvgbfa_fp8_s,
        self_attn._qkvgbfa_fp8_n,
        weights[0].shape[1],
    )
    _k3_prepare_f_b_tiny_gemm(self_attn)
    return True


def _k3_prepare_f_b_tiny_gemm(self_attn: nn.Module) -> None:
    """Dequant Quark PTPC ``f_b`` into the BF16 tiny-GEMM buffer.

    Decode ``f_b`` is ``[M, 128] @ [1536, 128].T``; tiny-GEMM covers it
    without the contiguous copy and group-quant the FP8 path needs.
    """
    if self_attn._bfa_f_b_w is not None:
        return
    weight = self_attn.f_b_proj.weight
    scale = getattr(self_attn.f_b_proj, "weight_scale", None)
    if (
        not isinstance(weight, torch.Tensor)
        or weight.dim() != 2
        or weight.dtype != torch.float8_e4m3fn
        or not isinstance(scale, torch.Tensor)
        or scale.numel() != weight.shape[0]
    ):
        return
    n, k = int(weight.shape[0]), int(weight.shape[1])
    from sglang.kernels.ops.gemm.kimi_k3 import _K3_TINY_GEMM_MAX_TOKENS

    if (n, k) not in _K3_TINY_GEMM_MAX_TOKENS:
        return
    self_attn._bfa_f_b_w = (
        (weight.data.float() * scale.data.reshape(-1, 1).float())
        .to(torch.bfloat16)
        .contiguous()
    )
    _k3_log_once(
        "kda_f_b_tiny_gemm", "K3 KDA f_b BF16 tiny-GEMM enabled (N=%d K=%d)", n, k
    )


def _k3_qkvgbfa_inproj(self_attn: nn.Module, hidden_states) -> Optional[torch.Tensor]:
    """Run the merged [q,k,v,g | f_a | b] projection, or None if uncovered."""
    if self_attn._use_qkvgbfa_ptpc_fp8(hidden_states):
        from sglang.kernels.ops.gemm import ptpc_fp8_aiter_hip

        if isinstance(hidden_states, tuple):
            return ptpc_fp8_aiter_hip.run(
                hidden_states[0],
                self_attn._qkvgbfa_fp8_w,
                self_attn._qkvgbfa_fp8_s,
                self_attn._qkvgbfa_fp8_n,
                x_scale=hidden_states[1],
            )
        return ptpc_fp8_aiter_hip.run(
            hidden_states,
            self_attn._qkvgbfa_fp8_w,
            self_attn._qkvgbfa_fp8_s,
            self_attn._qkvgbfa_fp8_n,
        )
    # A prequantized merge has no dense carrier to hand the linear method.
    if self_attn._qkvgbfa_layer is None:
        return None
    return self_attn.fused_qkvg_proj.quant_method.apply(
        self_attn._qkvgbfa_layer, hidden_states, None
    )


def _k3_apply_f_b(self_attn: nn.Module, f_a: torch.Tensor) -> torch.Tensor:
    if self_attn._bfa_f_b_w is None:
        # f_a is a column slice of the merged projection and the FP8
        # activation quantizer asserts on contiguity; the tiny GEMM below
        # takes the view as is.
        return self_attn.f_b_proj(f_a.contiguous())[0]
    from sglang.kernels.ops.gemm import kimi_k3_tiny_gemm

    if f_a.stride(-1) != 1:
        f_a = f_a.contiguous()
    return kimi_k3_tiny_gemm(f_a, self_attn._bfa_f_b_w)
