"""Fail-closed adapter for AITER's gfx950 Kimi-K3 fused KDA decode."""

from __future__ import annotations

import logging
import os

import torch

from sglang.srt.utils import is_hip, print_warning_once

logger = logging.getLogger(__name__)

_HEADS = 12
_DIM = 128
_CHANNELS = 3 * _HEADS * _DIM
_SGLANG_MODULE = "sglang.kernels.ops.attention.kda_flydsl.kimi_k3_kda_decode"
# ROCm/aiter#6026 publishes the kernel here;
# older AITER builds ship this module with only the GDR decode ops.
_AITER_MODULE = "aiter.ops.flydsl.linear_attention_kernels"
# [cache, 4608, 3], or SGLang's [cache, 3, 4608] conv cache as transpose(-1, -2);
# AITER rejects any other slot layout.
_CONV_SLOT_STRIDES = ((3, 1), (1, _CHANNELS))
_STATE_SLOT_STRIDES = ((_DIM * _DIM, _DIM, 1),)
_WARMED: set[tuple[int, float, float, torch.dtype]] = set()


def enabled() -> bool:
    return os.environ.get("SGLANG_ROCM_K3_KDA_FUSED_BACKEND", "").lower() == "aiter"


def _ops():
    """(run, supported, recurrent-state dtypes) of the selected kernel."""
    try:
        from sglang.kernels.ops.attention.kda_flydsl.source import load_module

        module = load_module(_SGLANG_MODULE, _AITER_MODULE)
    except (ImportError, ModuleNotFoundError) as error:
        print_warning_once(
            f"Kimi-K3 fused KDA decode unavailable ({error}); "
            "decode runs the unfused chain"
        )
        return None, None, ()
    if module.__name__ == _SGLANG_MODULE:
        return (
            module.flydsl_kimi_k3_kda_decode_with_f_b,
            module.is_flydsl_kimi_k3_kda_decode_supported,
            (torch.float32,),
        )
    try:
        op = module.flydsl_kda_decode_with_f_b
        return op, module.is_flydsl_kda_decode_supported, tuple(op.state_dtypes)
    except AttributeError:
        print_warning_once(
            f"Kimi-K3 fused KDA decode unavailable: {module.__name__} lacks "
            "flydsl_kda_decode_with_f_b (ROCm/aiter#6026); "
            "decode runs the unfused chain"
        )
        return None, None, ()


def _state_dtypes(device: torch.device | None) -> tuple[torch.dtype, ...]:
    """State dtypes the selected kernel takes on ``device``; empty if unavailable."""
    if not is_hip() or not enabled() or not torch.cuda.is_available():
        return ()
    _, supported, state_dtypes = _ops()
    if supported is None or not supported(device):
        return ()
    return state_dtypes


def _op_name() -> str:
    op, _, _ = _ops()
    return f"{op.__module__}.{op.__name__}"


def available(device: torch.device | None, state_dtype: torch.dtype) -> bool:
    """Whether the selected kernel runs on ``device`` over a ``state_dtype`` pool."""
    state_dtypes = _state_dtypes(device)
    if state_dtypes and state_dtype not in state_dtypes:
        print_warning_once(
            f"Kimi-K3 fused KDA decode not armed: {_op_name()} takes a "
            f"{', '.join(map(str, state_dtypes))} recurrent state, the pool holds "
            f"{state_dtype}; decode runs the unfused chain"
        )
    return state_dtype in state_dtypes


def _dense_slots(tensor: torch.Tensor, layouts: tuple[tuple[int, ...], ...]) -> bool:
    return (
        tensor.stride()[1:] in layouts and tensor.stride(0) >= tensor.shape[1:].numel()
    )


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
    state_dtypes = _state_dtypes(f_a.device)
    if not state_dtypes or f_a.ndim != 2:
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
        and raw_beta.stride(-1) == 1
        and conv_state.ndim == 3
        and conv_state.shape[1:] == (_CHANNELS, 3)
        and conv_state.dtype == torch.bfloat16
        and _dense_slots(conv_state, _CONV_SLOT_STRIDES)
        and state.ndim == 4
        and state.shape[1:] == (_HEADS, _DIM, _DIM)
        and state.dtype in state_dtypes
        and _dense_slots(state, _STATE_SLOT_STRIDES)
        and state_indices.shape == (batch,)
        and state_indices.dtype == torch.int32
        and state_indices.stride(0) == 1
        and output_gate.shape == (batch, _HEADS, _DIM)
        and output_gate.dtype == torch.bfloat16
        and output_gate.stride(-1) == 1
        and norm_weight.shape == (_DIM,)
        and norm_weight.dtype == torch.bfloat16
        and norm_weight.stride(-1) == 1
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
    op, _, _ = _ops()
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
    state_dtype: torch.dtype,
) -> None:
    if state_dtype not in _state_dtypes(f_b_weight.device):
        return
    device_index = -1 if f_b_weight.device.index is None else f_b_weight.device.index
    key = (device_index, float(norm_eps), float(lower_bound), state_dtype)
    if key in _WARMED:
        return

    device = f_b_weight.device
    # Each state dtype is its own JIT build;
    # warm the pool's so it never compiles during graph capture.
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
        state=torch.zeros(1, _HEADS, _DIM, _DIM, dtype=state_dtype, device=device),
        state_indices=torch.zeros(1, dtype=torch.int32, device=device),
        output_gate=torch.zeros(1, _HEADS, _DIM, dtype=torch.bfloat16, device=device),
        norm_weight=norm_weight,
        norm_eps=norm_eps,
    )
    torch.cuda.synchronize(device)
    _WARMED.add(key)
    logger.info(
        "Kimi-K3 fused KDA decode armed: %s, %s recurrent state",
        _op_name(),
        state_dtype,
    )
