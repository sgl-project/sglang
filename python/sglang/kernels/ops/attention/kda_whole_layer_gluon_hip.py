"""Fail-closed adapter for the gfx950 Kimi-K3 whole-layer Gluon KDA kernel.

The selected kernel owns the rank-local path from input projections through
convolution, recurrence, gated RMSNorm, and the output projection. SGLang's
``RowParallelLinear`` still owns output allocation and the attention-TP
reduction. Unsupported shapes never enter the kernel and retain the native
SGLang path.
"""

from __future__ import annotations

import functools
import os
from contextvars import ContextVar
from typing import Callable, Optional

import torch

from sglang.srt.utils import is_hip

_BACKEND = "gluon"
_HEADS = 12
_DIM = 128
_HIDDEN = 7168
_QKVG = 4 * _HEADS * _DIM
_CONV = 3 * _HEADS * _DIM
_BETA_FORGET = _DIM + _HEADS + 4
_PENDING: ContextVar[Optional[dict]] = ContextVar(
    "kimi_k3_whole_kda_pending", default=None
)


def enabled() -> bool:
    """Whether the explicitly selected ROCm K3 backend is whole-layer Gluon."""
    return os.environ.get("SGLANG_ROCM_K3_KDA_FUSED_BACKEND", "").lower() == _BACKEND


@functools.cache
def _rocm_arch(device_index: int) -> Optional[str]:
    try:
        properties = torch.cuda.get_device_properties(device_index)
        arch = getattr(properties, "gcnArchName", None)
        return arch.split(":", 1)[0] if arch is not None else None
    except (AssertionError, RuntimeError, TypeError, ValueError):
        return None


@functools.cache
def _has_required_gluon_api() -> bool:
    try:
        from triton.experimental.gluon import language as gl

        return hasattr(gl.amd, "slice")
    except (AttributeError, ImportError, ModuleNotFoundError):
        return False


def available(device: torch.device | None = None) -> bool:
    if (
        not is_hip()
        or not enabled()
        or not torch.cuda.is_available()
        or not _has_required_gluon_api()
    ):
        return False
    if device is None:
        index = torch.cuda.current_device()
    else:
        resolved = torch.device(device)
        if resolved.type != "cuda":
            return False
        index = (
            torch.cuda.current_device() if resolved.index is None else resolved.index
        )
    return _rocm_arch(index) == "gfx950"


def _row_major(tensor: object, shape: tuple[int, ...], dtype: torch.dtype) -> bool:
    return (
        isinstance(tensor, torch.Tensor)
        and tuple(tensor.shape) == shape
        and tensor.dtype == dtype
        and tensor.layout == torch.strided
        and tensor.ndim == 2
        and tensor.stride(1) == 1
        and tensor.stride(0) >= shape[1]
    )


def can_prepare(layer) -> bool:
    """Check immutable model/weight requirements before binding the fast path."""
    if not available(layer.dt_bias.device):
        return False
    attn = layer.attn
    o_proj = layer.o_proj
    qkvg_weight = getattr(getattr(layer, "fused_qkvg_proj", None), "weight", None)
    beta_forget_weight = getattr(layer, "_bfa_w", None)
    output_weight = getattr(o_proj, "weight", None)
    forget_weight = getattr(layer, "_bfa_f_b_w", None)
    weights = (qkvg_weight, beta_forget_weight, output_weight, forget_weight)
    return (
        layer.use_full_rank_gate
        and not layer.all_reduce_fusion
        and layer.hidden_size == _HIDDEN
        and layer.local_num_heads == _HEADS
        and layer.head_dim == _DIM
        and layer.attn_tp_size == 8
        and (
            getattr(layer, "_bfa_fa_size", None),
            getattr(layer, "_bfa_b_size", None),
        )
        == (_DIM, _HEADS)
        and getattr(attn, "bias", None) is None
        and attn.lower_bound == -5.0
        and layer.o_norm.eps == 1e-5
        and _row_major(qkvg_weight, (_QKVG, _HIDDEN), torch.bfloat16)
        and _row_major(beta_forget_weight, (_BETA_FORGET, _HIDDEN), torch.bfloat16)
        and _row_major(output_weight, (_HIDDEN, _HEADS * _DIM), torch.bfloat16)
        and _row_major(forget_weight, (_HEADS * _DIM, _DIM), torch.bfloat16)
        and all(weight.device == layer.dt_bias.device for weight in weights)
        and isinstance(attn.conv_weights, torch.Tensor)
        and tuple(attn.conv_weights.shape) == (_CONV, 4)
        and attn.conv_weights.dtype == torch.float32
        and attn.conv_weights.is_contiguous()
        and isinstance(attn.A_log, torch.Tensor)
        and attn.A_log.numel() == _HEADS
        and attn.A_log.dtype == torch.float32
        and attn.A_log.is_contiguous()
        and isinstance(attn.dt_bias, torch.Tensor)
        and tuple(attn.dt_bias.shape) == (_HEADS * _DIM,)
        and attn.dt_bias.dtype == torch.float32
        and attn.dt_bias.is_contiguous()
        and isinstance(layer.o_norm.weight, torch.Tensor)
        and tuple(layer.o_norm.weight.shape) == (_DIM,)
        and layer.o_norm.weight.dtype == torch.bfloat16
        and o_proj.bias is None
        and o_proj.input_is_parallel
        and o_proj.reduce_results
        and getattr(getattr(o_proj, "quant_method", None), "apply_into", None)
        is not None
    )


def covered(
    hidden_states: torch.Tensor,
    conv_state: torch.Tensor,
    state: torch.Tensor,
    state_indices: torch.Tensor,
) -> bool:
    """Check mutable runtime inputs before any stateful kernel is launched."""
    if not available(hidden_states.device) or hidden_states.ndim != 2:
        return False
    rows = hidden_states.shape[0]
    return (
        1 <= rows <= 256
        and tuple(hidden_states.shape) == (rows, _HIDDEN)
        and hidden_states.dtype == torch.bfloat16
        and hidden_states.stride(1) == 1
        and hidden_states.layout == torch.strided
        and conv_state.ndim == 3
        and conv_state.shape[0] > 0
        and tuple(conv_state.shape[1:]) == (3, _CONV)
        and conv_state.dtype == torch.bfloat16
        and conv_state.is_contiguous()
        and state.ndim == 4
        and state.shape[0] == conv_state.shape[0]
        and tuple(state.shape[1:]) == (_HEADS, _DIM, _DIM)
        and state.dtype == torch.float32
        and state.is_contiguous()
        and tuple(state_indices.shape) == (rows,)
        and state_indices.dtype in (torch.int32, torch.int64)
        and state_indices.stride(0) == 1
        and hidden_states.device == conv_state.device
        and hidden_states.device == state.device
        and hidden_states.device == state_indices.device
    )


def _entrypoint_name(rows: int) -> str:
    exact = {
        1: "kda_layer_decode_m1",
        2: "kda_layer_decode_m2",
        4: "kda_layer_decode_m4",
        32: "kda_layer_decode_m32",
        64: "kda_layer_decode_m64",
        128: "kda_layer_decode_m128",
        256: "kda_layer_decode_m256",
    }
    return exact.get(rows, "kda_layer_decode_m1_256")


def _entrypoint(rows: int):
    from sglang.kernels.ops.attention.kda_gluon.kernels import (
        kimi_k3_kda_layer_decode as kernels,
    )

    return getattr(kernels, _entrypoint_name(rows))


def run(
    *,
    hidden_states: torch.Tensor,
    qkvg_weight: torch.Tensor,
    beta_forget_weight: torch.Tensor,
    output_weight: torch.Tensor,
    forget_weight: torch.Tensor,
    conv_weight: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    norm_weight: torch.Tensor,
    conv_state: torch.Tensor,
    state: torch.Tensor,
    state_indices: torch.Tensor,
    lower_bound: float,
    norm_eps: float,
    output_tensor: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _entrypoint(hidden_states.shape[0])(
        hidden_states,
        qkvg_weight,
        beta_forget_weight,
        output_weight,
        forget_weight,
        conv_weight,
        A_log,
        dt_bias,
        norm_weight,
        conv_state,
        state,
        state_indices,
        lower_bound=lower_bound,
        norm_eps=norm_eps,
        output_tensor=output_tensor,
    )


def bind_output_projection(projection) -> None:
    """Let RowParallelLinear retain allocation/reduction around a fused result."""
    if getattr(projection, "_k3_whole_kda_bound", False):
        return
    method = projection.quant_method
    original_apply = method.apply
    original_apply_into = getattr(method, "apply_into", None)
    if original_apply_into is None:
        raise RuntimeError("whole-layer KDA requires the output apply_into ABI")

    def execute(layer, x, bias, output_tensor):
        pending = _PENDING.get()
        if pending is None or pending["projection"] is not layer:
            return None
        if bias is not None or x is not pending["carrier"] or pending["calls"]:
            raise RuntimeError("unexpected whole-layer KDA output projection call")
        expected = (x.shape[0], _HIDDEN)
        if output_tensor is not None and (
            tuple(output_tensor.shape) != expected
            or output_tensor.dtype != torch.bfloat16
            or output_tensor.device != x.device
            or not output_tensor.is_contiguous()
        ):
            raise RuntimeError("invalid whole-layer KDA output buffer")
        pending["calls"] += 1
        output = pending["invoke"](output_tensor)
        if (
            tuple(output.shape) != expected
            or output.dtype != torch.bfloat16
            or output.device != x.device
            or not output.is_contiguous()
        ):
            raise RuntimeError("whole-layer KDA returned an invalid output")
        if output_tensor is not None and output.data_ptr() != output_tensor.data_ptr():
            raise RuntimeError("whole-layer KDA did not write the requested output")
        return output

    @functools.wraps(original_apply)
    def apply(layer, x, bias=None):
        output = execute(layer, x, bias, None)
        return original_apply(layer, x, bias) if output is None else output

    @functools.wraps(original_apply_into)
    def apply_into(layer, x, out, bias=None):
        output = execute(layer, x, bias, out)
        return (
            original_apply_into(layer, x, out, bias=bias) if output is None else output
        )

    method.apply = apply
    method.apply_into = apply_into
    projection._k3_whole_kda_bound = True


def project_output(projection, carrier: torch.Tensor, invoke: Callable) -> torch.Tensor:
    """Execute the fused kernel from inside RowParallelLinear's allocation scope."""
    if _PENDING.get() is not None:
        raise RuntimeError("nested whole-layer KDA invocation")
    pending = {
        "projection": projection,
        "carrier": carrier,
        "invoke": invoke,
        "calls": 0,
    }
    token = _PENDING.set(pending)
    try:
        output = projection(carrier)[0]
        if pending["calls"] != 1:
            raise RuntimeError("whole-layer KDA was not executed exactly once")
        return output
    finally:
        _PENDING.reset(token)
