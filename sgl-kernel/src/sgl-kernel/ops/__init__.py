from typing import Optional

import torch
from sgl_kernel.ops._kernels import all_reduce as _all_reduce
from sgl_kernel.ops._kernels import dispose as _dispose
from sgl_kernel.ops._kernels import fused_add_rmsnorm as _fused_add_rmsnorm
from sgl_kernel.ops._kernels import gelu_and_mul as _gelu_and_mul
from sgl_kernel.ops._kernels import gelu_tanh_and_mul as _gelu_tanh_and_mul
from sgl_kernel.ops._kernels import gemma_fused_add_rmsnorm as _gemma_fused_add_rmsnorm
from sgl_kernel.ops._kernels import gemma_rmsnorm as _gemma_rmsnorm
from sgl_kernel.ops._kernels import (
    get_graph_buffer_ipc_meta as _get_graph_buffer_ipc_meta,
)
from sgl_kernel.ops._kernels import init_custom_ar as _init_custom_ar
from sgl_kernel.ops._kernels import int8_scaled_mm as _int8_scaled_mm
from sgl_kernel.ops._kernels import moe_align_block_size as _moe_align_block_size
from sgl_kernel.ops._kernels import register_graph_buffers as _register_graph_buffers
from sgl_kernel.ops._kernels import rmsnorm as _rmsnorm
from sgl_kernel.ops._kernels import rotary_embedding as _rotary_embedding
from sgl_kernel.ops._kernels import (
    sampling_scaling_penalties as _sampling_scaling_penalties,
)
from sgl_kernel.ops._kernels import silu_and_mul as _silu_and_mul


def get_cuda_stream(device: torch.device) -> int:
    return torch.cuda.current_stream(device).cuda_stream


def init_custom_reduce(
    rank_id, num_devices, rank_data, buffers, tmp_buffers, barrier_in, barrier_out
):
    return _init_custom_ar(
        rank_id, num_devices, rank_data, buffers, tmp_buffers, barrier_in, barrier_out
    )


def custom_dispose(fa):
    _dispose(fa)


def custom_reduce(fa, inp, out):
    _all_reduce(fa, inp, out)


def get_graph_buffer_ipc_meta(fa):
    return _get_graph_buffer_ipc_meta(fa)


def register_graph_buffers(fa, handles, offsets):
    _register_graph_buffers(fa, handles, offsets)


def moe_align_block_size(
    topk_ids,
    num_experts,
    block_size,
    sorted_token_ids,
    experts_ids,
    num_tokens_post_pad,
    token_cnts_buffer,
    cumsum_buffer,
):
    _moe_align_block_size(
        topk_ids,
        num_experts,
        block_size,
        sorted_token_ids,
        experts_ids,
        num_tokens_post_pad,
        token_cnts_buffer,
        cumsum_buffer,
    )


def sampling_scaling_penalties(logits, scaling_penalties):
    return _sampling_scaling_penalties(logits, scaling_penalties)


def int8_scaled_mm(mat_a, mat_b, scales_a, scales_b, out_dtype, bias=None):
    return _int8_scaled_mm(
        mat_a,
        mat_b,
        scales_a,
        scales_b,
        out_dtype,
        bias,
    )


def rotary_embedding(positions, query, key, head_size, cos_sin_cache, is_neox):
    return _rotary_embedding(positions, query, key, head_size, cos_sin_cache, is_neox)


def rmsnorm(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    with input.device as device:
        if out is None:
            out = torch.empty_like(input)
        _rmsnorm(out, input, weight, eps, get_cuda_stream(device))
        return out


def fused_add_rmsnorm(
    input: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
) -> None:
    with input.device as device:
        _fused_add_rmsnorm(input, residual, weight, eps, get_cuda_stream(device))


def gemma_rmsnorm(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    with input.device as device:
        if out is None:
            out = torch.empty_like(input)
        _gemma_rmsnorm(out, input, weight, eps, get_cuda_stream(device))
        return out


def gemma_fused_add_rmsnorm(
    input: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
) -> None:
    with input.device as device:
        _gemma_fused_add_rmsnorm(input, residual, weight, eps, get_cuda_stream(device))


def _check_shape(input: torch.Tensor, output: torch.Tensor) -> None:
    assert input.ndim == output.ndim, f"{input.ndim} != {output.ndim}"
    assert (
        input.shape[:-1] == output.shape[:-1]
    ), f"{input.shape[:-1]} != {output.shape[:-1]}"
    assert (
        input.shape[-1] == 2 * output.shape[-1]
    ), f"{input.shape[-1]} != {2 * output.shape[-1]}"


def silu_and_mul(input: torch.Tensor, out: torch.Tensor = None) -> torch.Tensor:
    if input.shape[-1] * input.dtype.itemsize % 16 != 0:
        raise ValueError("The pointers must be multiple of 16 bytes.")
    if out is not None:
        _check_shape(input, out)
    else:
        out = torch.empty(
            input.shape[:-1] + (input.shape[-1] // 2,),
            device=input.device,
            dtype=input.dtype,
        )
    with input.device as device:
        _silu_and_mul(out, input, get_cuda_stream(device))
        return out


def gelu_tanh_and_mul(input: torch.Tensor, out: torch.Tensor = None) -> torch.Tensor:
    if input.shape[-1] * input.dtype.itemsize % 16 != 0:
        raise ValueError("The pointers must be multiple of 16 bytes.")
    if out is not None:
        _check_shape(input, out)
    else:
        out = torch.empty(
            input.shape[:-1] + (input.shape[-1] // 2,),
            device=input.device,
            dtype=input.dtype,
        )
    with input.device as device:
        _gelu_tanh_and_mul(out, input, get_cuda_stream(device))
        return out


def gelu_and_mul(input: torch.Tensor, out: torch.Tensor = None) -> torch.Tensor:
    if input.shape[-1] * input.dtype.itemsize % 16 != 0:
        raise ValueError("The pointers must be multiple of 16 bytes.")
    if out is not None:
        _check_shape(input, out)
    else:
        out = torch.empty(
            input.shape[:-1] + (input.shape[-1] // 2,),
            device=input.device,
            dtype=input.dtype,
        )
    with input.device as device:
        _gelu_and_mul(out, input, get_cuda_stream(device))
        return out
