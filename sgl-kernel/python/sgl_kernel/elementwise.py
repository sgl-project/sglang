from dataclasses import dataclass
from typing import List, Optional

import torch
from sgl_kernel.utils import is_arch_support_pdl


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
        enable_pdl = is_arch_support_pdl()
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
        enable_pdl = is_arch_support_pdl()
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
        enable_pdl = is_arch_support_pdl()
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
        enable_pdl = is_arch_support_pdl()
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


@dataclass
class FusedSetKVBufferArg:
    """
    value : Optional[torch.Tensor]
        Value tensor, shape: ``(nnz, num_v_heads * head_size)``.
    k_buffer : Optional[torch.Tensor]
        Buffer for keys, shape: ``(nnz, num_k_heads * head_size)``.
    v_buffer : Optional[torch.Tensor]
        Buffer for values, shape: ``(nnz, num_v_heads * head_size)``.
    k_scale : Optional[float]
        Scale factor for keys.
    v_scale : Optional[float]
        Scale factor for values.
    cache_loc : Optional[torch.Tensor]
        Cache location tensor, used for indexing kv cache.
    """

    value: torch.Tensor
    k_buffer: torch.Tensor
    v_buffer: torch.Tensor
    k_scale: Optional[float]
    v_scale: Optional[float]
    cache_loc: torch.Tensor


def _view_3d(x, head_size):
    return x.view(x.shape[0], -1, head_size)


def apply_rope_with_cos_sin_cache_inplace(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox: bool = True,
    fused_set_kv_buffer_arg: Optional[FusedSetKVBufferArg] = None,
    enable_pdl: Optional[bool] = None,
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
    fused_set_kv_buffer_arg : FusedSetKVBufferArg
        Fuse the set-kv-buffer operation into this kernel

    Note
    ----
    The rotary dimension is determined by the cosine cache and sine cache.
    """
    if cos_sin_cache.dtype != torch.float32:
        raise ValueError("cos_sin_cache should be float32")

    if enable_pdl is None:
        # the non-fused branch does not yet support PDL, but after we switch to our impl for that branch it will
        enable_pdl = is_arch_support_pdl() and (fused_set_kv_buffer_arg is not None)

    if (a := fused_set_kv_buffer_arg) is not None:
        assert a.k_scale is None, "k_scale is not yet supported"
        assert a.v_scale is None, "v_scale is not yet supported"
        assert a.cache_loc.dtype == torch.int64, f"{a.cache_loc.dtype=}"

    torch.ops.sgl_kernel.apply_rope_pos_ids_cos_sin_cache.default(
        _view_3d(query, head_size),
        _view_3d(key, head_size),
        _view_3d(query, head_size),
        _view_3d(key, head_size),
        cos_sin_cache,
        positions.long(),
        (not is_neox),
        enable_pdl,
        (
            _view_3d(fused_set_kv_buffer_arg.value, head_size)
            if fused_set_kv_buffer_arg is not None
            else None
        ),
        (
            _view_3d(fused_set_kv_buffer_arg.k_buffer, head_size)
            if fused_set_kv_buffer_arg is not None
            else None
        ),
        (
            _view_3d(fused_set_kv_buffer_arg.v_buffer, head_size)
            if fused_set_kv_buffer_arg is not None
            else None
        ),
        (
            fused_set_kv_buffer_arg.cache_loc
            if fused_set_kv_buffer_arg is not None
            else None
        ),
    )


def rotary_embedding(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox: bool = True,
):
    torch.ops.sgl_kernel.rotary_embedding.default(
        positions, query, key, head_size, cos_sin_cache, is_neox
    )


def downcast_fp8(
    k: torch.Tensor,
    v: torch.Tensor,
    k_out: torch.Tensor,
    v_out: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    loc: torch.Tensor,
    mult: int = 1,
    offset: int = 0,
) -> None:
    torch.ops.sgl_kernel.downcast_fp8(
        k, v, k_out, v_out, k_scale, v_scale, loc, mult, offset
    )


def copy_to_gpu_no_ce(input: torch.Tensor, output: torch.Tensor):
    torch.ops.sgl_kernel.copy_to_gpu_no_ce(input, output)


def concat_mla_k(
    k: torch.Tensor,
    k_nope: torch.Tensor,
    k_rope: torch.Tensor,
):
    torch.ops.sgl_kernel.concat_mla_k(k, k_nope, k_rope)


def concat_mla_absorb_q(
    a: torch.Tensor,
    b: torch.Tensor,
):
    *batch_dims, _ = a.shape
    out = torch.empty(
        (*batch_dims, a.shape[-1] + b.shape[-1]), device=a.device, dtype=a.dtype
    )
    torch.ops.sgl_kernel.concat_mla_absorb_q(a, b, out)
    return out
