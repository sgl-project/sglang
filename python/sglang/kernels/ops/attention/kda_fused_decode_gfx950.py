# SPDX-License-Identifier: Apache-2.0
"""HIP Kimi-K3 fused KDA decode for gfx950 (MI350X / MI355X).

Same fail-closed interface as ``kda_fused_decode_aiter_hip``: it serves
``SGLANG_ROCM_K3_KDA_FUSED_BACKEND=gfx950`` (the default on ROCm), and inputs it
does not cover keep the unfused KDA decode. The kernel is
``attention.kda_decode_gfx950`` (``KernelBackend.KDA``), implemented in
``sglang.kernels.kda_kernels.kda_decode_gfx950``.
"""

from __future__ import annotations

import os

import torch

from sglang.kernels.selector import get_kernel
from sglang.kernels.spec import KernelBackend
from sglang.srt.utils import is_hip

_HEADS = 12
_DIM = 128
_CHANNELS = 3 * _HEADS * _DIM


def enabled() -> bool:
    backend = os.environ.get("SGLANG_ROCM_K3_KDA_FUSED_BACKEND", "gfx950")
    return backend.lower() == "gfx950"


def available(device: torch.device | None = None) -> bool:
    if not is_hip() or not enabled() or not torch.cuda.is_available():
        return False
    arch = torch.cuda.get_device_properties(device).gcnArchName
    return arch.split(":", 1)[0] == "gfx950"


def warmup(**_: object) -> None:
    from sglang.kernels.kda_kernels.kda_decode_gfx950 import load_ext

    load_ext()


def covered(
    f_a: torch.Tensor,
    f_b_weight: torch.Tensor,
    mixed_qkv: torch.Tensor,
    raw_beta: torch.Tensor,
    conv_state: torch.Tensor,
    state: torch.Tensor,
    state_indices: torch.Tensor,
    output_gate: torch.Tensor,
    norm_weight: torch.Tensor,
) -> bool:
    if not available(f_a.device) or f_a.ndim != 2:
        return False
    batch = f_a.shape[0]
    return (
        batch > 0
        and f_a.shape == (batch, _DIM)
        and f_a.dtype == torch.bfloat16
        and f_a.stride(-1) == 1
        and f_b_weight.shape == (_HEADS, _DIM, _DIM)
        and f_b_weight.dtype == torch.bfloat16
        and f_b_weight.is_contiguous()
        and mixed_qkv.shape == (batch, _CHANNELS)
        and mixed_qkv.dtype == torch.bfloat16
        and mixed_qkv.stride(-1) == 1
        and raw_beta.shape == (1, batch, _HEADS)
        and raw_beta.dtype == torch.bfloat16
        and raw_beta.stride(-1) == 1
        and conv_state.ndim == 3
        and conv_state.shape[1:] == (_CHANNELS, 3)
        and conv_state.dtype == torch.bfloat16
        and state.ndim == 4
        and state.shape[1:] == (_HEADS, _DIM, _DIM)
        and state.dtype == torch.float32
        and state.stride()[-3:] == (_DIM * _DIM, _DIM, 1)
        and state_indices.shape == (batch,)
        and state_indices.dtype == torch.int32
        and state_indices.stride(0) == 1
        and output_gate.shape == (batch, _HEADS, _DIM)
        and output_gate.dtype == torch.bfloat16
        and output_gate.stride()[-2:] == (_DIM, 1)
        and norm_weight.shape == (_DIM,)
        and norm_weight.dtype == torch.bfloat16
        and norm_weight.is_contiguous()
    )


def run(
    *,
    f_a: torch.Tensor,
    f_b_weight: torch.Tensor,
    mixed_qkv: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_state: torch.Tensor,
    raw_beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float,
    state: torch.Tensor,
    state_indices: torch.Tensor,
    output_gate: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_eps: float,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    if out is None:
        out = f_a.new_empty((1, f_a.shape[0], _HEADS, _DIM))
    get_kernel("attention.kda_decode_gfx950", KernelBackend.KDA)(
        f_a,
        f_b_weight,
        mixed_qkv,
        conv_weight.contiguous(),
        conv_state,
        raw_beta,
        A_log,
        dt_bias,
        state,
        state_indices,
        output_gate,
        norm_weight,
        out,
        lower_bound,
        norm_eps,
    )
    return out
