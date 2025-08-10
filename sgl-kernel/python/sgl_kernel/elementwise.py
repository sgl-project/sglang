from typing import Any, Optional

import torch
from sgl_kernel.utils import get_cuda_stream, is_hopper_arch



# These implementations extensively draw from and build upon the FlashInfer project https://github.com/flashinfer-ai/flashinfer
# Kudos to @yzh119
def rmsnorm(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    out: Optional[torch.Tensor] = None,
    enable_pdl: Optional[bool] = None,
) -> torch.Tensor:
    r"""Root mean square normalization.

    ``out[i] = (input[i] / RMS(input)) * weight[i]``

    Parameters
    ----------
    input: torch.Tensor
        Input tensor, shape (batch_size, hidden_size).
    weight: torch.Tensor
        Weight tensor, shape (hidden_size,).
    eps: float
        Epsilon for numerical stability.
    out: Optional[torch.Tensor]
        The output tensor, if specified, the kernel will update this tensor inplace.
    enable_pdl: Optional[bool]
        Whether to enable `programmatic dependent launch
        <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programmatic-dependent-launch-and-synchronization>`_
        If None, will be automatically enabled on Hopper architecture.

    Returns
    -------
    output: torch.Tensor
        Normalized tensor, shape (batch_size, hidden_size).
    """
    if out is None:
        out = torch.empty_like(input)
    if enable_pdl is None:
        enable_pdl = is_hopper_arch()
    torch.ops.sgl_kernel.rmsnorm.default(out, input, weight, eps, enable_pdl)
    return out


def fused_add_rmsnorm(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    enable_pdl: Optional[bool] = None,
) -> None:
    r"""Fused add root mean square normalization.

    Step 1:
    ``residual[i] += input[i]``

    Step 2:
    ``input[i] = (residual[i] / RMS(residual)) * weight[i]``

    Parameters
    ----------
    input: torch.Tensor
        Input tensor, shape (batch_size, hidden_size).
    residual: torch.Tensor
        Residual tensor, shape (batch_size, hidden_size).
    weight: torch.Tensor
        Weight tensor, shape (hidden_size,).
    eps: float
        Epsilon for numerical stability.
    enable_pdl: Optional[bool]
        Whether to enable `programmatic dependent launch
        <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programmatic-dependent-launch-and-synchronization>`_
        If None, will be automatically enabled on Hopper architecture.
    """
    if enable_pdl is None:
        enable_pdl = is_hopper_arch()
    torch.ops.sgl_kernel.fused_add_rmsnorm.default(
        input, residual, weight, eps, enable_pdl
    )


def gemma_rmsnorm(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    out: Optional[torch.Tensor] = None,
    enable_pdl: Optional[bool] = None,
) -> torch.Tensor:
    r"""Gemma-style root mean square normalization.

    ``out[i] = (input[i] / RMS(input)) * (weight[i] + 1)``

    Parameters
    ----------
    input: torch.Tensor
        Input tensor, shape (batch_size, hidden_size).
    weight: torch.Tensor
        Weight tensor, shape (hidden_size,).
    eps: float
        Epsilon for numerical stability.
    out: Optional[torch.Tensor]
        The output tensor, if specified, the kernel will update this tensor inplace.
    enable_pdl: Optional[bool]
        Whether to enable `programmatic dependent launch
        <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programmatic-dependent-launch-and-synchronization>`_
        If None, will be automatically enabled on Hopper architecture.

    Returns
    -------
    output: torch.Tensor
        Gemma Normalized tensor, shape (batch_size, hidden_size).
    """
    if out is None:
        out = torch.empty_like(input)
    if enable_pdl is None:
        enable_pdl = is_hopper_arch()
    torch.ops.sgl_kernel.gemma_rmsnorm.default(out, input, weight, eps, enable_pdl)
    return out


def gemma_fused_add_rmsnorm(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    enable_pdl: Optional[bool] = None,
) -> None:
    r"""Gemma-style fused add root mean square normalization.

    Step 1:
    ``residual[i] += input[i]``

    Step 2:
    ``input[i] = (residual[i] / RMS(residual)) * (weight + 1)``

    Parameters
    ----------
    input: torch.Tensor
        Input tensor, shape (batch_size, hidden_size).
    residual: torch.Tensor
        Residual tensor, shape (batch_size, hidden_size).
    weight: torch.Tensor
        Weight tensor, shape (hidden_size,).
    eps: float
        Epsilon for numerical stability.
    enable_pdl: Optional[bool]
        Whether to enable `programmatic dependent launch
        <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programmatic-dependent-launch-and-synchronization>`_
        If None, will be automatically enabled on Hopper architecture.
    """
    if enable_pdl is None:
        enable_pdl = is_hopper_arch()
    torch.ops.sgl_kernel.gemma_fused_add_rmsnorm.default(
        input, residual, weight, eps, enable_pdl
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
    torch.ops.sgl_kernel.silu_and_mul.default(out, input)
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
    torch.ops.sgl_kernel.gelu_tanh_and_mul.default(out, input)
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
    torch.ops.sgl_kernel.gelu_and_mul.default(out, input)
    return out


if torch.version.hip is not None:

    def gelu_quick(input: torch.Tensor, out: torch.Tensor = None) -> torch.Tensor:
        """
        Quick-GELU:  y = x * sigmoid(1.702 * x)

        The CUDA/HIP kernel uses 128-bit (16-byte) vector loads & stores,
        so the last-dimension byte length must be a multiple of 16 bytes.
        """
        if input.shape[-1] * input.dtype.itemsize % 16 != 0:
            raise ValueError(
                f"The last dimension ({input.shape[-1]}) x itemsize "
                f"({input.dtype.itemsize}) must be a multiple of 16 bytes."
            )

        if out is not None:
            assert input.shape == out.shape, f"{input.shape} != {out.shape}"
        else:
            out = torch.empty_like(input)

        torch.ops.sgl_kernel.gelu_quick(out, input)
        return out


def apply_rope_with_cos_sin_cache_inplace(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox: bool = True,
    layer: Any = None,  # RadixAttention
    forward_batch = None,
    save_kv_cache: bool = False,
    value: Optional[torch.Tensor] = None,
    start_layer: Optional[int] = None,
    is_capture_mode: bool = False,
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
          we rotate the first half dimensions ``([..., :head_dim//2])`` and the second half
          dimensions ``([..., head_dim//2:])``.

        * If ``False``, the last dimension of the query/key tensor is interleaved, i.e.,
          we rotate the even dimensions ``([..., ::2])`` and odd dimensions ``([..., 1::2])``.
    Note
    ----
    The rotary dimension is determined by the cosine cache and sine cache.
    """
    if cos_sin_cache.dtype != torch.float32:
        raise ValueError("cos_sin_cache should be float32")

    ## fused from memory_pool set_kv_buffer
    """
    if layer_id_override is not None:
        layer_id = layer_id_override
    else:
        layer_id = layer.layer_id
    if cache_k.dtype != self.dtype:
        if k_scale is not None:
            cache_k.div_(k_scale)
        if v_scale is not None:
            cache_v.div_(v_scale)
        cache_k = cache_k.to(self.dtype)
        cache_v = cache_v.to(self.dtype)

    if self.store_dtype != self.dtype:
        cache_k = cache_k.view(self.store_dtype)
        cache_v = cache_v.view(self.store_dtype)

    if get_is_capture_mode() and self.alt_stream is not None:
        # Overlap the copy of K and V cache for small batch size
        current_stream = self.device_module.current_stream()
        self.alt_stream.wait_stream(current_stream)
        self.k_buffer[layer_id - self.start_layer][loc] = cache_k
        with self.device_module.stream(self.alt_stream):
            self.v_buffer[layer_id - self.start_layer][loc] = cache_v
        current_stream.wait_stream(self.alt_stream)
    else:
        self.k_buffer[layer_id - self.start_layer][loc] = cache_k
        self.v_buffer[layer_id - self.start_layer][loc] = cache_v
    """
    if save_kv_cache:
        layer_id = layer.layer_id
        token_to_kv_pool = forward_batch.token_to_kv_pool
        start_layer = token_to_kv_pool.start_layer
        k_buffer = token_to_kv_pool.k_buffer
        v_buffer = token_to_kv_pool.v_buffer
        alt_stream = token_to_kv_pool.alt_stream
        cache_loc = forward_batch.out_cache_loc
        k_buffer_ptr = k_buffer[layer_id - start_layer][cache_loc].contiguous()
        v_buffer_ptr = v_buffer[layer_id - start_layer][cache_loc].contiguous()

        k_scale, v_scale = layer.k_scale, layer.v_scale


        torch.ops.sgl_kernel.apply_rope_pos_ids_cos_sin_cache_with_set_kv_buffer.default(
            query.view(query.shape[0], -1, head_size),
            key.view(key.shape[0], -1, head_size),
            query.view(query.shape[0], -1, head_size),
            key.view(key.shape[0], -1, head_size),
            cos_sin_cache,
            positions.long(),
            (not is_neox),
            get_cuda_stream(),
            k_buffer_ptr,
            v_buffer_ptr,
            1.0 if k_scale is None else k_scale,
            1.0 if v_scale is None else v_scale,
            value.view(value.shape[0], -1, head_size),
            is_capture_mode,
            0 if alt_stream is None else alt_stream,
        )
    else:
        torch.ops.sgl_kernel.apply_rope_pos_ids_cos_sin_cache.default(
            query.view(query.shape[0], -1, head_size),
            key.view(key.shape[0], -1, head_size),
            query.view(query.shape[0], -1, head_size),
            key.view(key.shape[0], -1, head_size),
            cos_sin_cache,
            positions.long(),
            (not is_neox),
            get_cuda_stream(),
        )
