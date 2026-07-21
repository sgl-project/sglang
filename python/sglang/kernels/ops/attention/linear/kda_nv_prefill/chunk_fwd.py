# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Serving adapter for NVIDIA's fused Blackwell KDA prefill kernel.

The SGLang wrapper repacks ordinary prefills into equal-length, 64-token
aligned batches before entering this module.  The former vendored entry point
also carried NVIDIA's split and variable-length experimental pipelines; those
paths were not reachable from serving and are intentionally not exposed here.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack

from .fuse_k1234_smem import make_host_fn as _make_host_fn

_CHUNK_SIZE = 64
_HEAD_DIM = 128

_compiled_kernels = {}
_buffers = {}


def _ct(tensor: torch.Tensor, element_type):
    """Wrap a PyTorch tensor as an explicitly typed CuTe tensor."""
    wrapped = from_dlpack(tensor, assumed_align=16)
    wrapped.element_type = element_type
    return wrapped


def _validate_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
    A_log: torch.Tensor | None,
    dt_bias: torch.Tensor | None,
) -> None:
    if q.ndim != 4:
        raise ValueError(f"NV KDA prefill expects q to be rank 4, got {q.ndim}")

    batch, seq_len, num_heads, key_dim = q.shape
    expected_q_shape = (batch, seq_len, num_heads, key_dim)
    if k.shape != expected_q_shape or g.shape != expected_q_shape:
        raise ValueError(
            "NV KDA prefill expects q, k, and g to have identical shapes; "
            f"got q={tuple(q.shape)}, k={tuple(k.shape)}, g={tuple(g.shape)}"
        )
    if v.ndim != 4 or v.shape[:3] != q.shape[:3]:
        raise ValueError(
            "NV KDA prefill expects v to match q on batch, sequence, and heads; "
            f"got q={tuple(q.shape)}, v={tuple(v.shape)}"
        )
    if beta.shape != q.shape[:3]:
        raise ValueError(
            f"NV KDA prefill expects beta shape {tuple(q.shape[:3])}, "
            f"got {tuple(beta.shape)}"
        )
    if key_dim != _HEAD_DIM or v.shape[-1] != _HEAD_DIM:
        raise ValueError(
            f"NV KDA prefill requires K=V={_HEAD_DIM}, "
            f"got K={key_dim}, V={v.shape[-1]}"
        )
    if seq_len == 0 or seq_len % _CHUNK_SIZE != 0:
        raise ValueError(
            f"NV KDA prefill requires a positive sequence length divisible by "
            f"{_CHUNK_SIZE}, got {seq_len}"
        )

    bf16_inputs = {"q": q, "k": k, "v": v, "g": g}
    bad_dtypes = {
        name: tensor.dtype
        for name, tensor in bf16_inputs.items()
        if tensor.dtype != torch.bfloat16
    }
    if bad_dtypes:
        raise TypeError(f"NV KDA prefill requires BF16 q/k/v/g, got {bad_dtypes}")
    if beta.dtype != torch.float32:
        raise TypeError(f"NV KDA prefill requires FP32 beta, got {beta.dtype}")

    tensors = [q, k, v, g, beta]
    if initial_state is not None:
        expected_state_shape = (batch, num_heads, key_dim, v.shape[-1])
        if initial_state.shape != expected_state_shape:
            raise ValueError(
                f"NV KDA prefill expects initial_state shape "
                f"{expected_state_shape}, got {tuple(initial_state.shape)}"
            )
        if initial_state.dtype != torch.float32:
            raise TypeError(
                f"NV KDA prefill requires an FP32 initial_state, "
                f"got {initial_state.dtype}"
            )
        tensors.append(initial_state)

    if A_log is None or A_log.numel() != num_heads:
        actual = None if A_log is None else A_log.numel()
        raise ValueError(
            f"NV KDA prefill requires one A_log value per head "
            f"({num_heads}), got {actual}"
        )
    tensors.append(A_log)

    if dt_bias is not None:
        expected_bias_values = num_heads * key_dim
        if dt_bias.numel() != expected_bias_values:
            raise ValueError(
                f"NV KDA prefill expects {expected_bias_values} dt_bias values, "
                f"got {dt_bias.numel()}"
            )
        tensors.append(dt_bias)

    if q.device.type != "cuda":
        raise ValueError(f"NV KDA prefill requires CUDA tensors, got {q.device}")
    mismatched_devices = [
        tensor.device for tensor in tensors if tensor.device != q.device
    ]
    if mismatched_devices:
        raise ValueError(
            f"NV KDA prefill requires all tensors on {q.device}, "
            f"got mismatches {mismatched_devices}"
        )


def chunk_kda_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
    safe_gate: bool = False,
    lower_bound: float | None = None,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
):
    """Run the fused K1-K4 kernel on an equal-length prefill batch.

    Returns the same 12-item tuple shape used by FLA's ``chunk_kda_fwd`` so
    the serving wrapper can consume the output and final state directly.
    """
    _validate_inputs(q, k, v, g, beta, initial_state, A_log, dt_bias)

    if safe_gate and lower_bound is None:
        lower_bound = -5.0

    batch, seq_len, num_heads, key_dim = q.shape
    value_dim = v.shape[-1]
    device = q.device
    device_index = device.index or 0
    num_chunks = seq_len // _CHUNK_SIZE
    has_bias = dt_bias is not None

    batch_heads = batch * num_heads
    state = (
        initial_state.reshape(batch_heads, key_dim, value_dim).contiguous()
        if initial_state is not None
        else torch.zeros(
            batch_heads,
            key_dim,
            value_dim,
            dtype=torch.float32,
            device=device,
        )
    )

    buffer_key = (
        device_index,
        batch,
        seq_len,
        num_heads,
        key_dim,
        value_dim,
        q.dtype,
    )
    buffers = _buffers.get(buffer_key)
    if buffers is None:
        buffers = (
            torch.empty(
                batch,
                seq_len,
                num_heads,
                value_dim,
                dtype=q.dtype,
                device=device,
            ),
            torch.empty(2, batch_heads, dtype=torch.int64, device=device),
        )
        _buffers[buffer_key] = buffers
    output, clocks = buffers

    bf16 = cutlass.BFloat16
    fp32 = cutlass.Float32
    if has_bias:
        bias = _ct(dt_bias.float().contiguous().view(num_heads, key_dim), fp32)
    else:
        bias = _ct(torch.empty(1, 1, dtype=torch.float32, device=device), fp32)

    kernel_args = (
        _ct(q.contiguous(), bf16),
        _ct(k.contiguous(), bf16),
        _ct(g.contiguous(), bf16),
        _ct(A_log.float().contiguous(), fp32),
        _ct(beta.contiguous(), fp32),
        float(scale),
        _ct(v.contiguous(), bf16),
        _ct(output, bf16),
        _ct(state, fp32),
        bias,
        float(lower_bound) if lower_bound is not None else 0.0,
        _ct(clocks, cutlass.Int64),
        num_chunks,
        num_heads,
        batch,
        cuda.CUstream(torch.cuda.current_stream(device).cuda_stream),
    )

    compile_key = (device_index, has_bias, safe_gate)
    if compile_key not in _compiled_kernels:
        host_fn = _make_host_fn(has_bias=has_bias, use_safe_gate=safe_gate)
        _compiled_kernels[compile_key] = cute.compile(host_fn, *kernel_args)
    _compiled_kernels[compile_key](*kernel_args)

    final_state = (
        state.reshape(batch, num_heads, key_dim, value_dim)
        if output_final_state
        else None
    )
    return (
        output,
        final_state,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        initial_state,
    )
