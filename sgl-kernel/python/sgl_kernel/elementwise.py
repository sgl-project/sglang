from typing import Optional

import torch
from sgl_kernel.utils import get_cuda_stream


# These implementations extensively draw from and build upon the FlashInfer project https://github.com/flashinfer-ai/flashinfer
# Kudos to @yzh119
def rmsnorm(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    out: Optional[torch.Tensor] = None,
    enable_pdl: bool = False,
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
    enable_pdl: bool
        Whether to enable `programmatic dependent launch
        <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programmatic-dependent-launch-and-synchronization>`_

    Returns
    -------
    output: torch.Tensor
        Normalized tensor, shape (batch_size, hidden_size).
    """
    if out is None:
        out = torch.empty_like(input)
    torch.ops.sgl_kernel.rmsnorm.default(out, input, weight, eps, enable_pdl)
    return out


def fused_add_rmsnorm(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    enable_pdl: bool = False,
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
    enable_pdl: bool
        Whether to enable `programmatic dependent launch
        <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programmatic-dependent-launch-and-synchronization>`_
    """
    torch.ops.sgl_kernel.fused_add_rmsnorm.default(
        input, residual, weight, eps, enable_pdl
    )


def gemma_rmsnorm(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    out: Optional[torch.Tensor] = None,
    enable_pdl: bool = False,
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
    enable_pdl: bool
        Whether to enable `programmatic dependent launch
        <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programmatic-dependent-launch-and-synchronization>`_

    Returns
    -------
    output: torch.Tensor
        Gemma Normalized tensor, shape (batch_size, hidden_size).
    """
    if out is None:
        out = torch.empty_like(input)
    torch.ops.sgl_kernel.gemma_rmsnorm.default(out, input, weight, eps, enable_pdl)
    return out


def gemma_fused_add_rmsnorm(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    enable_pdl: bool = False,
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
    enable_pdl: bool
        Whether to enable `programmatic dependent launch
        <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programmatic-dependent-launch-and-synchronization>`_
    """
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
    r"""
    Apply the SiLU (Sigmoid Linear Unit, also known as the swish activation) function
    to the input tensor, and multiply the result element-wise by the original input.

    The operation performed is: `output = input * SiLU(input)`, where
    `SiLU(x) = x * sigmoid(x)`.

    Parameters
    ----------
    input : torch.Tensor
        The input tensor to which the SiLU activation and elementwise multiplication
        are applied. Can be of any shape and should be a floating-point tensor.
    out : torch.Tensor, optional
        Optional output tensor to store the result. If provided, the result is
        written in-place to this tensor. Otherwise, a new tensor is allocated
        and returned.

    Returns
    -------
    output : torch.Tensor
        The result tensor after applying `input * SiLU(input)`. Has the same shape
        and dtype as the input.

    Note
    ----
    This function is commonly used in neural network layers where the "silu and mul"
    (sometimes called "gated SiLU" or "swish-gated") pattern is required. Using the
    optional `out` parameter can save memory and computation in performance-critical
    scenarios.
    """
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
    torch.ops.sgl_kernel.silu_and_mul.default(out, input, get_cuda_stream())
    return out


def gelu_tanh_and_mul(input: torch.Tensor, out: torch.Tensor = None) -> torch.Tensor:
    r"""
    Apply the GELU activation function (using the tanh approximation) to the input tensor,
    and multiply the result element-wise by the original input.

    The operation performed is: `output = input * GELU(input)`, where
    GELU (with tanh approximation) is defined as:
      GELU(x) = 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x^3)))

    Parameters
    ----------
    input : torch.Tensor
        The input tensor on which the GELU activation (tanh approximation) and
        elementwise multiplication are applied. Should be a floating-point tensor
        of any shape.
    out : torch.Tensor, optional
        Optional output tensor to store the result. If provided, the result is
        written in-place to this tensor. Otherwise, a new tensor is created
        and returned.

    Returns
    -------
    output : torch.Tensor
        The result tensor after applying `input * GELU(input)`. Shape and dtype
        are the same as the input.

    Note
    ----
    This function is useful in advanced neural network architectures where
    "gated GELU" or similar element-wise gated activations are required.
    Using the optional `out` parameter can help optimize memory usage and
    computation in performance-sensitive scenarios.
    """
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
    torch.ops.sgl_kernel.gelu_tanh_and_mul.default(out, input, get_cuda_stream())
    return out


def gelu_and_mul(input: torch.Tensor, out: torch.Tensor = None) -> torch.Tensor:
    r"""
    Apply the GELU (Gaussian Error Linear Unit) activation function to the input tensor,
    and multiply the result element-wise by the original input tensor.

    The operation performed is: `output = input * GELU(input)`, where
    GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
    or may use an approximation such as the tanh-based version.

    Parameters
    ----------
    input : torch.Tensor
        The input tensor. Should be a floating-point tensor of any shape.
    out : torch.Tensor, optional
        Optional tensor to store the output. If provided, the operation is performed
        in-place and the result is written to this tensor. If not provided, a new
        tensor is allocated and returned.

    Returns
    -------
    output : torch.Tensor
        The result tensor after applying `input * GELU(input)`. Shape and dtype
        match the input tensor.

    Note
    ----
    This function is typically used in neural network architectures that require
    "gated GELU" or element-wise gated activation patterns. Providing the `out`
    parameter allows for more memory-efficient and potentially faster computation
    in performance-critical contexts.
    """
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
    torch.ops.sgl_kernel.gelu_and_mul.default(out, input, get_cuda_stream())
    return out


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
