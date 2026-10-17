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
"""Kimi-K3 ROCm PTPC FP8 decode.

Producers fold the per-token activation quant into the kernel that already
reads the activation (the attention input norm, the KDA output norm), and the
per-channel FP8 linears take the ``(fp8, per-token scale)`` result as is.
"""

from typing import Optional

import msgspec
import torch
from torch import nn

from sglang.kernels.ops.gemm import ptpc_fp8_aiter_hip
from sglang.srt.environ import envs
from sglang.srt.layers.quantization.quark.quark import QuarkLinearMethod
from sglang.srt.layers.quantization.quark.schemes import QuarkW8A8Fp8

PTPC_FORMAT = "fp8_per_token"


class PtpcFp8Weight(msgspec.Struct, frozen=True):
    """Preshuffled FP8 weight, [N_pad, 1] FP32 channel scales, logical N."""

    weight: torch.Tensor
    scale: torch.Tensor
    out_features: int


class PtpcWindow(msgspec.Struct, frozen=True):
    """Decode batch sizes, in tokens, that run the PTPC producers."""

    min_tokens: int
    max_tokens: int

    def covers(self, num_tokens: int) -> bool:
        return self.min_tokens <= num_tokens <= self.max_tokens


def ptpc_window() -> Optional[PtpcWindow]:
    if not (envs.SGLANG_OPT_HIP_K3_PTPC_FP8.get() and ptpc_fp8_aiter_hip.available()):
        return None
    return PtpcWindow(
        min_tokens=envs.SGLANG_OPT_HIP_K3_PTPC_FP8_MIN_TOKENS.get(),
        max_tokens=envs.SGLANG_OPT_HIP_K3_PTPC_FP8_MAX_TOKENS.get(),
    )


def takes_ptpc_tuple(linear: nn.Module) -> bool:
    """Whether the linear runs (fp8, per-token scale) without quantizing again."""
    return (
        isinstance(linear.quant_method, QuarkLinearMethod)
        and isinstance(linear.scheme, QuarkW8A8Fp8)
        and linear.scheme.takes_per_token_fp8_input
    )


# ===== KDA =====


def init_kda_ptpc_state(self_attn: nn.Module) -> None:
    self_attn._k3_ptpc = ptpc_window()
    self_attn._qkvgbfa_ptpc: Optional[PtpcFp8Weight] = None
    self_attn._k3_o_norm_ptpc = (
        self_attn._k3_ptpc is not None
        and not self_attn.all_reduce_fusion
        and takes_ptpc_tuple(self_attn.o_proj)
    )


def merge_kda_inproj_ptpc(self_attn: nn.Module) -> bool:
    """Pack the Quark per-channel FP8 [q,k,v,g | f_a | b] in-projection into one
    PTPC GEMM, reusing the checkpoint's channel scales."""
    from sglang.srt.models.kimi_k3_rocm_quant import (
        _k3_channel_fp8_to_bf16,
        _k3_is_raw_quark_channel_fp8,
    )

    mods = (self_attn.fused_qkvg_proj, self_attn.f_a_proj, self_attn.b_proj)
    if not (
        self_attn._k3_ptpc is not None
        and envs.SGLANG_ROCM_K3_FUSE_KDA_INPROJ.get()
        and self_attn.use_full_rank_gate
        and all(_k3_is_raw_quark_channel_fp8(m) for m in (*mods, self_attn.f_b_proj))
    ):
        return False
    weights = [m.weight.data for m in mods]
    if len({w.shape[1] for w in weights}) != 1:
        return False
    sizes = [w.shape[0] for w in weights]
    # An 8-row pad keeps every split slice of the output 16-byte aligned.
    pad = (-sum(sizes)) % 8
    scales = [m.weight_scale.data.reshape(-1).float() for m in mods]
    if pad:
        weights.append(weights[0].new_zeros((pad, weights[0].shape[1])))
        scales.append(scales[0].new_ones(pad))
    weight, scale, out_features = ptpc_fp8_aiter_hip.pack_prequantized(
        torch.cat(weights), torch.cat(scales)
    )
    self_attn._qkvgbfa_ptpc = PtpcFp8Weight(
        weight=weight, scale=scale, out_features=out_features
    )
    self_attn._bfa_fa_size, self_attn._bfa_b_size = sizes[1], sizes[2]
    self_attn._qkvgbfa_sizes = [*self_attn.split_sizes, sizes[1], sizes[2], pad]
    # f_b is [heads * head_dim, head_dim]; BF16 lets it use the tiny GEMM and the
    # fused KDA decode.
    self_attn._bfa_f_b_w = _k3_channel_fp8_to_bf16(
        self_attn.f_b_proj, self_attn.f_b_proj.weight.data
    ).contiguous()
    ptpc_fp8_aiter_hip.warmup(
        weight=weight,
        scale=scale,
        out_features=out_features,
        in_features=weights[0].shape[1],
        max_tokens=self_attn._qkvgbfa_bs_limit,
    )
    return True


def run_kda_inproj_ptpc(self_attn: nn.Module, hidden_states) -> torch.Tensor:
    packed = self_attn._qkvgbfa_ptpc
    if isinstance(hidden_states, tuple):
        x, x_scale = hidden_states
        return ptpc_fp8_aiter_hip.run(
            x, packed.weight, packed.scale, packed.out_features, x_scale=x_scale
        )
    return ptpc_fp8_aiter_hip.run(
        hidden_states, packed.weight, packed.scale, packed.out_features
    )


def fuse_kda_o_norm_ptpc(
    self_attn: nn.Module, core_attn_out: torch.Tensor, g_proj_states: torch.Tensor
) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
    """Gated output RMSNorm fused with o_proj's per-token FP8 quant, or None."""
    rows = core_attn_out.shape[-3]
    if not (self_attn._k3_o_norm_ptpc and self_attn._k3_ptpc.covers(rows)):
        return None
    from aiter import dtypes as aiter_dtypes
    from aiter.ops.gated_rmsnorm_fp8_per_token_quant import (
        gated_rmsnorm_fp8_per_token_quant,
    )

    norm_input = core_attn_out.squeeze(0)
    quant_out = torch.empty(
        (rows, norm_input.shape[-2] * norm_input.shape[-1]),
        dtype=aiter_dtypes.fp8,
        device=norm_input.device,
    )
    quant_scale = torch.empty((rows, 1), dtype=torch.float32, device=norm_input.device)
    # Quantizes the FP32 gated value directly, skipping the BF16 round trip of
    # the split norm.
    gated_rmsnorm_fp8_per_token_quant(
        quant_out,
        quant_scale,
        norm_input,
        g_proj_states.unflatten(-1, (-1, self_attn.head_dim)),
        self_attn.o_norm.weight,
        self_attn.o_norm.eps,
        use_sigmoid=True,
    )
    return quant_out, quant_scale


# ===== MLA =====


def init_mla_ptpc_state(self_attn: nn.Module) -> None:
    window = ptpc_window()
    inproj = (
        self_attn.fused_qkv_a_proj_with_mqa
        if self_attn.q_lora_rank is not None
        else self_attn.kv_a_proj_with_mqa
    )
    self_attn._k3_ptpc = window
    self_attn._k3_inproj_ptpc = window is not None and takes_ptpc_tuple(inproj)
    self_attn._k3_q_b_ptpc = (
        window is not None
        and self_attn.q_lora_rank is not None
        and takes_ptpc_tuple(self_attn.q_b_proj)
    )
    self_attn._k3_gate_ptpc = (
        window is not None
        and self_attn.use_output_gate
        and takes_ptpc_tuple(self_attn.o_proj)
    )


def fuse_mla_gate_ptpc(x: torch.Tensor, gate: torch.Tensor):
    """x * sigmoid(gate) as (fp8, per-token scale), or the BF16 product when the
    fused kernel does not cover the shape."""
    from sglang.kernels.ops.attention import mla_output_gate_fp8_quant_hip

    if mla_output_gate_fp8_quant_hip.covered(x, gate):
        return mla_output_gate_fp8_quant_hip.kimi_k3_mla_output_gate_fp8_quant(x, gate)
    return x * torch.sigmoid(gate)


# ===== Attention input =====


def kda_attn_input_quant_format(self_attn: nn.Module, num_tokens: int) -> str:
    """PTPC when the merged in-projection consumes this decode batch, else ""."""
    window = self_attn._k3_ptpc
    if (
        window is not None
        and self_attn._qkvgbfa_ptpc is not None
        and window.covers(num_tokens)
        and num_tokens <= self_attn._qkvgbfa_bs_limit
    ):
        return PTPC_FORMAT
    return ""


def mla_attn_input_quant_format(self_attn: nn.Module, num_tokens: int) -> str:
    window = self_attn._k3_ptpc
    if self_attn._k3_inproj_ptpc and window.covers(num_tokens):
        return PTPC_FORMAT
    return ""
