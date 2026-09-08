# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""High-level API for the fused Kimi-K3 KDA decode specialization."""

from __future__ import annotations

import functools
from collections.abc import Iterable

import torch
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled

from .kernels.kimi_k3_kda_decode import (
    create_kimi_k3_kda_decode_kernel,
)
from .kernels.kimi_k3_kda_decode_fb import (
    create_kimi_k3_kda_decode_fb_kernel,
)

_HEADS = 12
_DIM = 128
_CONV_CHANNELS = 3 * _HEADS * _DIM
_CONV_WIDTH = 4


def _fb_build_options(batch: int) -> dict[str, int | bool]:
    """Use the validated gfx950 winner only for the exact-C2 bucket."""
    if batch != 2:
        return {}
    return {
        "waves_per_eu": 3,
        "cooperative_f_a": True,
        "parallel_front": True,
        "fused_norm_reduce": True,
        "projection_fdot2": True,
    }


@functools.cache
def _rocm_arch(device: torch.device) -> str | None:
    properties = torch.cuda.get_device_properties(device)
    arch = getattr(properties, "gcnArchName", None)
    return arch.split(":", 1)[0] if arch is not None else None


def is_flydsl_kimi_k3_kda_decode_supported(
    device: torch.device | str | int | None = None,
) -> bool:
    """Return whether ``device`` can run this gfx950-only specialization."""
    if not torch.cuda.is_available():
        return False
    try:
        resolved = torch.device(
            "cuda",
            torch.cuda.current_device(),
        )
        if device is not None:
            resolved = (
                torch.device("cuda", device)
                if isinstance(device, int)
                else torch.device(device)
            )
            if resolved.type != "cuda":
                return False
            if resolved.index is None:
                resolved = torch.device(
                    "cuda",
                    torch.cuda.current_device(),
                )
        return _rocm_arch(resolved) == "gfx950"
    except (AssertionError, RuntimeError, TypeError, ValueError):
        return False


def _check_tensor(
    name: str,
    tensor: torch.Tensor,
    *,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
    inner_strides: tuple[int, ...] = (),
) -> None:
    if tensor.shape != shape:
        raise ValueError(
            f"`{name}` must have shape {list(shape)}, got {list(tensor.shape)}."
        )
    if tensor.dtype != dtype:
        raise ValueError(f"`{name}` must have dtype {dtype}, got {tensor.dtype}.")
    if tensor.device != device:
        raise ValueError(f"`{name}` must be on {device}, got {tensor.device}.")
    if inner_strides and tensor.stride()[-len(inner_strides) :] != inner_strides:
        raise ValueError(
            f"`{name}` must have inner strides {inner_strides}, got {tensor.stride()}."
        )


def _check_same_device(
    tensors: Iterable[tuple[str, torch.Tensor]],
    device: torch.device,
) -> None:
    for name, tensor in tensors:
        if not tensor.is_cuda:
            raise ValueError(f"`{name}` must be a CUDA tensor.")
        if tensor.device != device:
            raise ValueError(f"`{name}` must be on {device}, got {tensor.device}.")


def _validate_kda_inputs(
    *,
    api_name: str,
    batch_source: str,
    device: torch.device,
    batch: int,
    x: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor | None,
    conv_state: torch.Tensor,
    raw_beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float | None,
    state: torch.Tensor,
    state_indices: torch.Tensor,
    output_gate: torch.Tensor,
    norm_weight: torch.Tensor,
    out: torch.Tensor | None,
) -> torch.Tensor:
    """Validate operands shared by both explicit KDA specializations."""
    if not is_flydsl_kimi_k3_kda_decode_supported(device):
        raise RuntimeError(f"`{api_name}` requires a gfx950 GPU.")
    if batch <= 0:
        raise ValueError(f"`{batch_source}` must have a non-empty batch dimension.")
    if conv_bias is not None:
        raise ValueError("This specialization requires `conv_bias=None`.")
    if lower_bound is None:
        raise ValueError("This specialization requires the KDA lower-bound gate.")

    _check_same_device(
        (
            ("x", x),
            ("conv_weight", conv_weight),
            ("conv_state", conv_state),
            ("raw_beta", raw_beta),
            ("A_log", A_log),
            ("dt_bias", dt_bias),
            ("state", state),
            ("state_indices", state_indices),
            ("output_gate", output_gate),
            ("norm_weight", norm_weight),
        ),
        device,
    )
    _check_tensor(
        "x",
        x,
        shape=(batch, _CONV_CHANNELS),
        dtype=torch.bfloat16,
        device=device,
        inner_strides=(1,),
    )
    _check_tensor(
        "conv_weight",
        conv_weight,
        shape=(_CONV_CHANNELS, _CONV_WIDTH),
        dtype=torch.float32,
        device=device,
    )
    if conv_state.ndim != 3 or conv_state.shape[1:] != (
        _CONV_CHANNELS,
        _CONV_WIDTH - 1,
    ):
        raise ValueError(
            "`conv_state` must have shape [cache, 4608, 3], "
            f"got {list(conv_state.shape)}."
        )
    if conv_state.dtype != torch.bfloat16:
        raise ValueError("`conv_state` must have dtype torch.bfloat16.")
    if state.ndim != 4 or state.shape[1:] != (
        _HEADS,
        _DIM,
        _DIM,
    ):
        raise ValueError(
            f"`state` must have shape [cache, 12, 128, 128], got {list(state.shape)}."
        )
    if state.dtype != torch.float32:
        raise ValueError("`state` must have dtype torch.float32.")
    if state.stride()[-3:] != (_DIM * _DIM, _DIM, 1):
        raise ValueError("`state` must be contiguous within each cache slot.")
    _check_tensor(
        "raw_beta",
        raw_beta,
        shape=(1, batch, _HEADS),
        dtype=torch.bfloat16,
        device=device,
        inner_strides=(1,),
    )
    _check_tensor(
        "A_log",
        A_log,
        shape=(_HEADS,),
        dtype=torch.float32,
        device=device,
        inner_strides=(1,),
    )
    _check_tensor(
        "dt_bias",
        dt_bias,
        shape=(_HEADS * _DIM,),
        dtype=torch.float32,
        device=device,
        inner_strides=(1,),
    )
    _check_tensor(
        "state_indices",
        state_indices,
        shape=(batch,),
        dtype=torch.int32,
        device=device,
        inner_strides=(1,),
    )
    _check_tensor(
        "output_gate",
        output_gate,
        shape=(batch, _HEADS, _DIM),
        dtype=torch.bfloat16,
        device=device,
        inner_strides=(1,),
    )
    _check_tensor(
        "norm_weight",
        norm_weight,
        shape=(_DIM,),
        dtype=torch.bfloat16,
        device=device,
        inner_strides=(1,),
    )

    if out is None:
        return torch.empty(
            (1, batch, _HEADS, _DIM),
            dtype=torch.bfloat16,
            device=device,
        )
    _check_same_device((("out", out),), device)
    _check_tensor(
        "out",
        out,
        shape=(1, batch, _HEADS, _DIM),
        dtype=torch.bfloat16,
        device=device,
        inner_strides=(1,),
    )
    return out


def flydsl_kimi_k3_kda_decode(
    x: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor | None,
    conv_state: torch.Tensor,
    raw_g: torch.Tensor,
    raw_beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float | None,
    state: torch.Tensor,
    state_indices: torch.Tensor,
    output_gate: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_eps: float,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run fused Kimi-K3 KDA decode on MI350-series GPUs.

    This pure-decode specialization fuses the packed width-4 Q/K/V causal
    convolution, the FP32 recurrent-state update, and the BF16
    RMSNorm/sigmoid output gate. Slot zero is reserved: non-positive
    ``state_indices`` produce zero output without modifying either cache.

    The layout is fixed to Kimi-K3 TP8: 12 local heads and 128-dimensional
    key/value state. Call
    :func:`is_flydsl_kimi_k3_kda_decode_supported` before dispatching from a
    model implementation.
    """
    if x.ndim != 2:
        raise ValueError(f"`x` must have rank 2, got rank {x.ndim}.")
    if not x.is_cuda:
        raise ValueError("`x` must be a CUDA tensor.")
    device = x.device
    batch = x.shape[0]
    out = _validate_kda_inputs(
        api_name="flydsl_kimi_k3_kda_decode",
        batch_source="x",
        device=device,
        batch=batch,
        x=x,
        conv_weight=conv_weight,
        conv_bias=conv_bias,
        conv_state=conv_state,
        raw_beta=raw_beta,
        A_log=A_log,
        dt_bias=dt_bias,
        lower_bound=lower_bound,
        state=state,
        state_indices=state_indices,
        output_gate=output_gate,
        norm_weight=norm_weight,
        out=out,
    )
    _check_same_device((("raw_g", raw_g),), device)
    _check_tensor(
        "raw_g",
        raw_g,
        shape=(1, batch, _HEADS, _DIM),
        dtype=torch.bfloat16,
        device=device,
        inner_strides=(_DIM, 1),
    )

    executable = create_kimi_k3_kda_decode_kernel(
        float(norm_eps),
        float(lower_bound),
    )
    with torch.cuda.device(device):
        stream = torch.cuda.current_stream(device)
        _run_compiled(
            executable,
            x,
            conv_weight,
            conv_state,
            raw_g,
            raw_beta,
            A_log,
            dt_bias,
            state,
            state_indices,
            output_gate,
            norm_weight,
            out,
            batch,
            x.stride(0),
            conv_weight.stride(0),
            conv_weight.stride(1),
            conv_state.stride(0),
            conv_state.stride(1),
            conv_state.stride(2),
            raw_g.stride(1),
            raw_beta.stride(1),
            state.stride(0),
            output_gate.stride(0),
            output_gate.stride(1),
            out.stride(1),
            out.stride(2),
            stream,
        )
    return out


def flydsl_kimi_k3_kda_decode_with_f_b(
    f_a: torch.Tensor,
    f_b_weight: torch.Tensor,
    x: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor | None,
    conv_state: torch.Tensor,
    raw_beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float | None,
    state: torch.Tensor,
    state_indices: torch.Tensor,
    output_gate: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_eps: float,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run the explicit gfx950 Kimi-K3 f_b plus KDA decode specialization.

    The kernel consumes ``f_a`` and the head-local ``f_b_weight`` directly,
    accumulates the projection in FP32, and rounds once to BF16 before the KDA
    lower-bound decay gate. It does not materialize the projected raw-g tensor
    in global memory.
    """
    if f_a.ndim != 2:
        raise ValueError(f"`f_a` must have rank 2, got rank {f_a.ndim}.")
    if not f_a.is_cuda:
        raise ValueError("`f_a` must be a CUDA tensor.")
    device = f_a.device
    batch = f_a.shape[0]
    _check_same_device((("f_b_weight", f_b_weight),), device)
    _check_tensor(
        "f_a",
        f_a,
        shape=(batch, _DIM),
        dtype=torch.bfloat16,
        device=device,
        inner_strides=(1,),
    )
    _check_tensor(
        "f_b_weight",
        f_b_weight,
        shape=(_HEADS, _DIM, _DIM),
        dtype=torch.bfloat16,
        device=device,
        inner_strides=(_DIM, 1),
    )
    out = _validate_kda_inputs(
        api_name="flydsl_kimi_k3_kda_decode_with_f_b",
        batch_source="f_a",
        device=device,
        batch=batch,
        x=x,
        conv_weight=conv_weight,
        conv_bias=conv_bias,
        conv_state=conv_state,
        raw_beta=raw_beta,
        A_log=A_log,
        dt_bias=dt_bias,
        lower_bound=lower_bound,
        state=state,
        state_indices=state_indices,
        output_gate=output_gate,
        norm_weight=norm_weight,
        out=out,
    )

    executable = create_kimi_k3_kda_decode_fb_kernel(
        float(norm_eps),
        float(lower_bound),
        **_fb_build_options(batch),
    )
    with torch.cuda.device(device):
        stream = torch.cuda.current_stream(device)
        _run_compiled(
            executable,
            f_a,
            f_b_weight,
            x,
            conv_weight,
            conv_state,
            raw_beta,
            A_log,
            dt_bias,
            state,
            state_indices,
            output_gate,
            norm_weight,
            out,
            batch,
            f_a.stride(0),
            f_b_weight.stride(0),
            f_b_weight.stride(1),
            x.stride(0),
            conv_weight.stride(0),
            conv_weight.stride(1),
            conv_state.stride(0),
            conv_state.stride(1),
            conv_state.stride(2),
            raw_beta.stride(1),
            state.stride(0),
            output_gate.stride(0),
            output_gate.stride(1),
            out.stride(1),
            out.stride(2),
            stream,
        )
    return out


__all__ = [
    "flydsl_kimi_k3_kda_decode",
    "flydsl_kimi_k3_kda_decode_with_f_b",
    "is_flydsl_kimi_k3_kda_decode_supported",
]
