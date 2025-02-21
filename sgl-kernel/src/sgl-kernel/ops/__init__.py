import os
from typing import List, Optional, Tuple, Union

import sgl_kernel.ops._kernels
import torch
from sgl_kernel.ops.utils import (
    _get_cache_buf,
    _get_cuda_stream,
    _to_tensor_scalar_tuple,
)


def apply_rope_with_cos_sin_cache_inplace(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox: bool = True,
) -> None:
    r"""
    Apply rotary embedding to keys and queries with precomputed cos/sin values.
    This is designed to be compatible with the SGL/vLLM implementation.
    The result is inplace applied to the input tensors.

    Parameters
    ----------
    positions : torch.Tensor
        Position indices, shape: ``(nnz)``.
    query : torch.Tensor
        Query tensor, shape: ``(nnz, num_q_heads * head_size)``.
    key : torch.Tensor
        Key tensor, shape: ``(nnz, num_k_heads * head_size)``.
    cos_sin_cache : torch.Tensor
        Cosine and Sine cache tensor, shape: ``(max_seq_len, rotary_dim)``.
        Cosine is the first half and Sine is the second half on rotary_dim.
    is_neox : bool
        Whether to use Neox style RoPE, default: ``True``.

        * If ``True``, the last dimension of the query/key tensor is not interleaved, i.e.,
          we rorate the first half dimensions ``([..., :head_dim//2])`` and the second half
          dimensions ``([..., head_dim//2:])``.

        * If ``False``, the last dimension of the query/key tensor is interleaved, i.e.,
          we rotate the even dimensions ``([..., ::2])`` and odd dimensions ``([..., 1::2])``.
    Note
    ----
    The rotary dimension is determined by the cosine cache and sine cache.
    """
    if cos_sin_cache.dtype != torch.float32:
        raise ValueError("cos_sin_cache should be float32")

    with query.device as device:
        positions = positions.int()
        torch.ops.sgl_kernels.apply_rope_pos_ids_cos_sin_cache(
            q=query.view(query.shape[0], -1, head_size),
            k=key.view(key.shape[0], -1, head_size),
            q_rope=query.view(query.shape[0], -1, head_size),
            k_rope=key.view(key.shape[0], -1, head_size),
            cos_sin_cache=cos_sin_cache,
            pos_ids=positions,
            interleave=(not is_neox),
            cuda_stream=_get_cuda_stream(device),
        )


def init_custom_reduce(
    rank_id, num_devices, rank_data, buffers, tmp_buffers, barrier_in, barrier_out
):
    return torch.ops.sgl_kernels.init_custom_ar(
        rank_id, num_devices, rank_data, buffers, tmp_buffers, barrier_in, barrier_out
    )


def custom_dispose(fa):
    torch.ops.sgl_kernels.dispose(fa)


def custom_reduce(fa, inp, out):
    torch.ops.sgl_kernels.all_reduce(fa, inp, out)


def get_graph_buffer_ipc_meta(fa):
    return torch.ops.sgl_kernels.get_graph_buffer_ipc_meta(fa)


def register_graph_buffers(fa, handles, offsets):
    torch.ops.sgl_kernels.register_graph_buffers(fa, handles, offsets)


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
    torch.ops.sgl_kernels.moe_align_block_size(
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
    return torch.ops.sgl_kernels.sampling_scaling_penalties(logits, scaling_penalties)


def int8_scaled_mm(mat_a, mat_b, scales_a, scales_b, out_dtype, bias=None):
    return torch.ops.sgl_kernels.int8_scaled_mm(
        mat_a,
        mat_b,
        scales_a,
        scales_b,
        out_dtype,
        bias,
    )


def fp8_blockwise_scaled_mm(mat_a, mat_b, scales_a, scales_b, out_dtype):
    return torch.ops.sgl_kernels.fp8_blockwise_scaled_mm(
        mat_a,
        mat_b,
        scales_a,
        scales_b,
        out_dtype,
    )


def fp8_scaled_mm(mat_a, mat_b, scales_a, scales_b, out_dtype, bias=None):
    return torch.ops.sgl_kernels.fp8_scaled_mm(
        mat_a,
        mat_b,
        scales_a,
        scales_b,
        out_dtype,
        bias,
    )


def lightning_attention_decode(q, k, v, past_kv, slope, output, new_kv):
    torch.ops.sgl_kernels.lightning_attention_decode(
        q, k, v, past_kv, slope, output, new_kv
    )


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
        torch.ops.sgl_kernels.rmsnorm(out, input, weight, eps, _get_cuda_stream(device))
        return out


def fused_add_rmsnorm(
    input: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
) -> None:
    with input.device as device:
        torch.ops.sgl_kernels.fused_add_rmsnorm(input, residual, weight, eps)


def gemma_rmsnorm(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    with input.device as device:
        if out is None:
            out = torch.empty_like(input)
        torch.ops.sgl_kernels.gemma_rmsnorm(
            out, input, weight, eps, _get_cuda_stream(device)
        )
        return out


def gemma_fused_add_rmsnorm(
    input: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
) -> None:
    with input.device as device:
        torch.ops.sgl_kernels.gemma_fused_add_rmsnorm(
            input, residual, weight, eps, _get_cuda_stream(device)
        )


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
        torch.ops.sgl_kernels.silu_and_mul(out, input, _get_cuda_stream(device))
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
        torch.ops.sgl_kernels.gelu_tanh_and_mul(out, input, _get_cuda_stream(device))
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
        torch.ops.sgl_kernels.gelu_and_mul(out, input, _get_cuda_stream(device))
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
        torch.ops.sgl_kernels.bmm_fp8(
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
        torch.ops.sgl_kernels.top_k_renorm_probs_wrapper(
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
        torch.ops.sgl_kernels.top_p_renorm_probs(
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
        torch.ops.sgl_kernels.top_p_sampling_from_probs(
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
        torch.ops.sgl_kernels.top_k_top_p_sampling_from_probs(
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
        torch.ops.sgl_kernels.min_p_sampling_from_probs(
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


def tree_speculative_sampling_target_only(
    predicts: torch.Tensor,  # mutable
    accept_index: torch.Tensor,  # mutable
    accept_token_num: torch.Tensor,  # mutable
    candidates: torch.Tensor,
    retrive_index: torch.Tensor,
    retrive_next_token: torch.Tensor,
    retrive_next_sibling: torch.Tensor,
    uniform_samples: torch.Tensor,
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    deterministic: bool = True,
) -> None:
    with predicts.device as device:
        torch.ops.sgl_kernels.tree_speculative_sampling_target_only(
            predicts,
            accept_index,
            accept_token_num,
            candidates,
            retrive_index,
            retrive_next_token,
            retrive_next_sibling,
            uniform_samples,
            target_probs,
            draft_probs,
            deterministic,
            _get_cuda_stream(device),
        )


def build_tree_kernel_efficient(
    parent_list: torch.Tensor,
    selected_index: torch.Tensor,
    verified_seq_len: torch.Tensor,
    tree_mask: torch.Tensor,
    positions: torch.Tensor,
    retrive_index: torch.Tensor,
    retrive_next_token: torch.Tensor,
    retrive_next_sibling: torch.Tensor,
    topk: int,
    depth: int,
    draft_token_num: int,
) -> None:
    with parent_list.device as device:
        torch.ops.sgl_kernels.build_tree_kernel_efficient(
            parent_list,
            selected_index,
            verified_seq_len,
            tree_mask,
            positions,
            retrive_index,
            retrive_next_token,
            retrive_next_sibling,
            topk,
            depth,
            draft_token_num,
        )


def build_tree_kernel(
    parent_list: torch.Tensor,
    selected_index: torch.Tensor,
    verified_seq_len: torch.Tensor,
    tree_mask: torch.Tensor,
    positions: torch.Tensor,
    retrive_index: torch.Tensor,
    topk: int,
    depth: int,
    draft_token_num: int,
) -> None:
    with parent_list.device as device:
        torch.ops.sgl_kernels.build_tree_kernel(
            parent_list,
            selected_index,
            verified_seq_len,
            tree_mask,
            positions,
            retrive_index,
            topk,
            depth,
            draft_token_num,
        )


def sgl_per_token_group_quant_fp8(
    input: torch.Tensor,
    output_q: torch.Tensor,
    output_s: torch.Tensor,
    group_size: int,
    eps: float,
    fp8_min: float,
    fp8_max: float,
) -> None:
    torch.ops.sgl_kernels.sgl_per_token_group_quant_fp8(
        input, output_q, output_s, group_size, eps, fp8_min, fp8_max
    )


def cublas_grouped_gemm(
    inputs: List[torch.Tensor],
    weights: List[torch.Tensor],
    outputs: List[torch.Tensor],
    out_dtype: torch.dtype,
) -> None:
    with inputs[0].device as device:
        assert (
            len(inputs) > 0 and len(weights) > 0 and len(outputs) > 0
        ), "Inputs/weights/outputs should not be empty!"
        cublas_handle = torch.cuda.current_blas_handle()
        torch.ops.sgl_kernels.cublas_grouped_gemm(
            inputs,
            weights,
            outputs,
            out_dtype,
            cublas_handle,
            _get_cuda_stream(device),
        )
