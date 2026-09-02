"""Fail-closed adapter for AITER's gfx950 Kimi-K3 fused KDA decode."""

from __future__ import annotations

import os

import torch

from sglang.srt.utils import is_hip

_HEADS = 12
_DIM = 128
_CHANNELS = 3 * _HEADS * _DIM
_WARMED: set[tuple[int, float, float]] = set()


def enabled() -> bool:
    return os.environ.get("SGLANG_K3_KDA_FUSED_BACKEND", "").lower() == "aiter"


def _ops():
    try:
        from sglang.kernels.ops.kimi_k3.flydsl.source import load_module

        module = load_module(
            "sglang.kernels.ops.kimi_k3.flydsl.kimi_k3_kda_decode",
            "aiter.ops.flydsl.kimi_k3_kda_decode",
        )
    except (ImportError, ModuleNotFoundError):
        return None, None
    return (
        module.flydsl_kimi_k3_kda_decode_with_f_b,
        module.is_flydsl_kimi_k3_kda_decode_supported,
    )


def available(device: torch.device | None = None) -> bool:
    if not is_hip() or not enabled() or not torch.cuda.is_available():
        return False
    _, supported = _ops()
    if supported is None:
        return False
    return bool(supported(device))


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
        and f_b_weight.stride()[-2:] == (_DIM, 1)
        and mixed_qkv.shape == (batch, _CHANNELS)
        and mixed_qkv.dtype == torch.bfloat16
        and mixed_qkv.stride(-1) == 1
        and raw_beta.shape == (1, batch, _HEADS)
        and raw_beta.dtype == torch.bfloat16
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
        and norm_weight.shape == (_DIM,)
        and norm_weight.dtype == torch.bfloat16
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
    op, _ = _ops()
    if op is None:
        raise RuntimeError("AITER Kimi-K3 fused KDA decode is unavailable")
    return op(
        f_a=f_a,
        f_b_weight=f_b_weight,
        x=mixed_qkv,
        conv_weight=conv_weight,
        conv_bias=None,
        conv_state=conv_state,
        raw_beta=raw_beta,
        A_log=A_log,
        dt_bias=dt_bias,
        lower_bound=lower_bound,
        state=state,
        state_indices=state_indices,
        output_gate=output_gate,
        norm_weight=norm_weight,
        norm_eps=norm_eps,
        out=out,
    )


def warmup(
    *,
    f_b_weight: torch.Tensor,
    conv_weight: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float,
    norm_weight: torch.Tensor,
    norm_eps: float,
) -> None:
    if not available(f_b_weight.device):
        return
    device_index = -1 if f_b_weight.device.index is None else f_b_weight.device.index
    key = (device_index, float(norm_eps), float(lower_bound))
    if key in _WARMED:
        return

    device = f_b_weight.device
    run(
        f_a=torch.zeros(1, _DIM, dtype=torch.bfloat16, device=device),
        f_b_weight=f_b_weight,
        mixed_qkv=torch.zeros(1, _CHANNELS, dtype=torch.bfloat16, device=device),
        conv_weight=conv_weight,
        conv_state=torch.zeros(1, _CHANNELS, 3, dtype=torch.bfloat16, device=device),
        raw_beta=torch.zeros(1, 1, _HEADS, dtype=torch.bfloat16, device=device),
        A_log=A_log,
        dt_bias=dt_bias,
        lower_bound=lower_bound,
        state=torch.zeros(1, _HEADS, _DIM, _DIM, dtype=torch.float32, device=device),
        state_indices=torch.zeros(1, dtype=torch.int32, device=device),
        output_gate=torch.zeros(1, _HEADS, _DIM, dtype=torch.bfloat16, device=device),
        norm_weight=norm_weight,
        norm_eps=norm_eps,
    )
    torch.cuda.synchronize(device)
    _WARMED.add(key)
