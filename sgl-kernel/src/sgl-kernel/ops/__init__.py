from typing import Optional, Tuple, Union

import torch
from torch.ops.sgl_kernels import (
    all_reduce as _all_reduce,
    bmm_fp8 as _bmm_fp8,
    dispose as _dispose,
    fused_add_rmsnorm as _fused_add_rmsnorm,
    gelu_and_mul as _gelu_and_mul,
    gelu_tanh_and_mul as _gelu_tanh_and_mul,
    gemma_fused_add_rmsnorm as _gemma_fused_add_rmsnorm,
    gemma_rmsnorm as _gemma_rmsnorm,
    get_graph_buffer_ipc_meta as _get_graph_buffer_ipc_meta,
    init_custom_ar as _init_custom_ar,
    int8_scaled_mm as _int8_scaled_mm,
    lightning_attention_decode as _lightning_attention_decode,
    min_p_sampling_from_probs as _min_p_sampling_from_probs,
    moe_align_block_size as _moe_align_block_size,
    register_graph_buffers as _register_graph_buffers,
    rmsnorm as _rmsnorm,
    rotary_embedding as _rotary_embedding,
    sampling_scaling_penalties as _sampling_scaling_penalties,
    silu_and_mul as _silu_and_mul,
    top_k_renorm_probs as _top_k_renorm_probs,
    top_k_top_p_sampling_from_probs as _top_k_top_p_sampling_from_probs,
    top_p_renorm_probs as _top_p_renorm_probs,
    top_p_sampling_from_probs as _top_p_sampling_from_probs,
)
from sgl_kernel.ops.utils import (
    _get_cache_buf,
    _get_cuda_stream,
    _to_tensor_scalar_tuple,
)


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


def lightning_attention_decode(q, k, v, past_kv, slope, output, new_kv):
    _lightning_attention_decode(q, k, v, past_kv, slope, output, new_kv)


def rotary_embedding(positions, query, key, head_size, cos_sin_cache, is_neox):
    return _rotary_embedding(positions, query, key, head_size, cos_sin_cache, is_neox)


# These implementations extensively draw from and build upon the FlashInfer project https://github.com/flashinfer-ai/flashinfer
# Kudos to @yzh119
def rmsnorm(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    with input.device as device:
        if out is None:
            out = torch.empty_like(input)
        _rmsnorm(out, input, weight, eps, _get_cuda_stream(device))
        return out


def fused_add_rmsnorm(
    input: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
) -> None:
    with input.device as device:
        _fused_add_rmsnorm(input, residual, weight, eps, _get_cuda_stream(device))


def gemma_rmsnorm(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    with input.device as device:
        if out is None:
            out = torch.empty_like(input)
        _gemma_rmsnorm(out, input, weight, eps, _get_cuda_stream(device))
        return out


def gemma_fused_add_rmsnorm(
    input: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
) -> None:
    with input.device as device:
        _gemma_fused_add_rmsnorm(input, residual, weight, eps, _get_cuda_stream(device))


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
        _silu_and_mul(out, input, _get_cuda_stream(device))
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
        _gelu_tanh_and_mul(out, input, _get_cuda_stream(device))
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
        _gelu_and_mul(out, input, _get_cuda_stream(device))
        return out


def _bmm_fp8_internal(
    workspace_buffer: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    D: torch.Tensor,
    A_scale: torch.Tensor,
    B_scale: torch.Tensor,
) -> None:
    with A.device as device:
        cublas_handle = torch.cuda.current_blas_handle()
        _bmm_fp8(
            A,
            B,
            D,
            A_scale,
            B_scale,
            workspace_buffer,
            cublas_handle,
            _get_cuda_stream(device),
        )


def bmm_fp8(
    A: torch.Tensor,
    B: torch.Tensor,
    A_scale: torch.Tensor,
    B_scale: torch.Tensor,
    dtype: torch.dtype,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if out is None:
        out = torch.empty(
            (A.shape[0], A.shape[1], B.shape[2]),
            device=A.device,
            dtype=dtype,
        )
    workspace_buffer = _get_cache_buf("bmm_fp8_workspace", 32 * 1024 * 1024, A.device)
    _bmm_fp8_internal(workspace_buffer, A, B, out, A_scale, B_scale)
    return out


def _top_k_renorm_probs_internal(
    probs: torch.Tensor,
    maybe_top_k_arr: Optional[torch.Tensor],
    top_k_val: int,
) -> torch.Tensor:
    with probs.device as device:
        probs = probs.float()
        maybe_top_k_arr = maybe_top_k_arr.int() if maybe_top_k_arr is not None else None
        renorm_probs = torch.empty_like(probs)
        _top_k_renorm_probs(
            probs,
            renorm_probs,
            maybe_top_k_arr,
            top_k_val,
            _get_cuda_stream(device),
        )
        return renorm_probs


def top_k_renorm_probs(
    probs: torch.Tensor,
    top_k: Union[torch.Tensor, int],
) -> torch.Tensor:
    return _top_k_renorm_probs_internal(probs, *_to_tensor_scalar_tuple(top_k))


top_k_renorm_prob = top_k_renorm_probs


def _top_p_renorm_probs_internal(
    probs: torch.Tensor,
    maybe_top_p_arr: Optional[torch.Tensor],
    top_p_val: float,
) -> torch.Tensor:
    with probs.device as device:
        probs = probs.float()
        maybe_top_p_arr = (
            maybe_top_p_arr.float() if maybe_top_p_arr is not None else None
        )
        renorm_probs = torch.empty_like(probs)
        _top_p_renorm_probs(
            probs,
            renorm_probs,
            maybe_top_p_arr,
            top_p_val,
            _get_cuda_stream(device),
        )
        return renorm_probs


def top_p_renorm_probs(
    probs: torch.Tensor,
    top_p: Union[torch.Tensor, float],
) -> torch.Tensor:
    return _top_p_renorm_probs_internal(probs, *_to_tensor_scalar_tuple(top_p))


top_p_renorm_prob = top_p_renorm_probs


def _top_p_sampling_from_probs_internal(
    probs: torch.Tensor,
    uniform_samples: torch.Tensor,
    maybe_top_p_arr: Optional[torch.Tensor],
    top_p_val: float,
    deterministic: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    with probs.device as device:
        probs = probs.float()
        uniform_samples = uniform_samples.float()
        maybe_top_p_arr = (
            maybe_top_p_arr.float() if maybe_top_p_arr is not None else None
        )
        samples = torch.empty(probs.size(0), dtype=torch.int32, device=device)
        success = torch.empty(probs.size(0), dtype=torch.bool, device=device)
        _top_p_sampling_from_probs(
            probs,
            uniform_samples,
            samples,
            success,
            maybe_top_p_arr,
            top_p_val,
            deterministic,
            _get_cuda_stream(device),
        )
        return samples, success


def top_p_sampling_from_probs(
    probs: torch.Tensor,
    uniform_samples: torch.Tensor,
    top_p: Union[torch.Tensor, float],
    deterministic: bool = True,
    check_nan: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if check_nan:
        if torch.any(torch.isnan(probs)):
            raise ValueError("Input probs contains NaN.")
    return _top_p_sampling_from_probs_internal(
        probs, uniform_samples, *_to_tensor_scalar_tuple(top_p), deterministic
    )


def _top_k_top_p_sampling_from_probs_internal(
    probs: torch.Tensor,
    uniform_samples: torch.Tensor,
    maybe_top_k_arr: Optional[torch.Tensor],
    top_k_val: int,
    maybe_top_p_arr: Optional[torch.Tensor],
    top_p_val: float,
    deterministic: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    with probs.device as device:
        probs = probs.float()
        uniform_samples = uniform_samples.float()
        maybe_top_k_arr = maybe_top_k_arr.int() if maybe_top_k_arr is not None else None
        maybe_top_p_arr = (
            maybe_top_p_arr.float() if maybe_top_p_arr is not None else None
        )
        samples = torch.empty(probs.size(0), dtype=torch.int32, device=device)
        success = torch.empty(probs.size(0), dtype=torch.bool, device=device)
        _top_k_top_p_sampling_from_probs(
            probs,
            uniform_samples,
            samples,
            success,
            maybe_top_k_arr,
            top_k_val,
            maybe_top_p_arr,
            top_p_val,
            deterministic,
            _get_cuda_stream(device),
        )
        return samples, success


def top_k_top_p_sampling_from_probs(
    probs: torch.Tensor,
    uniform_samples: torch.Tensor,
    top_k: Union[torch.Tensor, int],
    top_p: Union[torch.Tensor, float],
    filter_apply_order: str = "top_k_first",
    deterministic: bool = True,
    check_nan: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if filter_apply_order == "top_k_first":
        renorm_probs = top_k_renorm_probs(probs, top_k)
        return top_p_sampling_from_probs(
            renorm_probs, uniform_samples, top_p, deterministic, check_nan=check_nan
        )
    elif filter_apply_order == "joint":
        if check_nan:
            if torch.any(torch.isnan(probs)):
                raise ValueError("Input probs contains NaN.")
        return _top_k_top_p_sampling_from_probs_internal(
            probs,
            uniform_samples,
            *_to_tensor_scalar_tuple(top_k),
            *_to_tensor_scalar_tuple(top_p),
            deterministic,
        )
    else:
        raise ValueError(f"Invalid filter_apply_order: {filter_apply_order}")


def _min_p_sampling_from_probs_internal(
    probs: torch.Tensor,
    uniform_samples: torch.Tensor,
    maybe_min_p_arr: Optional[torch.Tensor],
    min_p_val: float,
    deterministic: bool,
) -> torch.Tensor:
    with probs.device as device:
        probs = probs.float()
        uniform_samples = uniform_samples.float()
        maybe_min_p_arr = (
            maybe_min_p_arr.float() if maybe_min_p_arr is not None else None
        )
        samples = torch.empty(probs.size(0), dtype=torch.int32, device=device)
        _min_p_sampling_from_probs(
            probs,
            uniform_samples,
            samples,
            maybe_min_p_arr,
            min_p_val,
            deterministic,
            _get_cuda_stream(device),
        )
        return samples


def min_p_sampling_from_probs(
    probs: torch.Tensor,
    uniform_samples: torch.Tensor,
    min_p: Union[torch.Tensor, float],
    deterministic: bool = True,
    check_nan: bool = False,
) -> torch.Tensor:
    if uniform_samples.dim() == 2:
        # Take the first row (round) of uniform_samples
        uniform_samples = uniform_samples[0]

    if check_nan:
        if torch.any(torch.isnan(probs)):
            raise ValueError("Input probs contains NaN.")
    return _min_p_sampling_from_probs_internal(
        probs, uniform_samples, *_to_tensor_scalar_tuple(min_p), deterministic
    )
