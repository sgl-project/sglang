from typing import Optional

import torch
from sgl_kernel.utils import is_arch_support_pdl

try:
    import flashinfer.norm as _flashinfer_norm

    _has_flashinfer = True
except ImportError:
    _has_flashinfer = False

_FLASHINFER_NORM_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16}


def _rmsnorm_internal(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    out: Optional[torch.Tensor],
    enable_pdl: Optional[bool],
) -> torch.Tensor:
    if out is None:
        out = torch.empty_like(input)
    if enable_pdl is None:
        enable_pdl = is_arch_support_pdl()
    torch.ops.sgl_kernel.rmsnorm.default(out, input, weight, eps, enable_pdl)
    return out


def _fused_add_rmsnorm_internal(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    enable_pdl: Optional[bool],
) -> None:
    if enable_pdl is None:
        enable_pdl = is_arch_support_pdl()
    torch.ops.sgl_kernel.fused_add_rmsnorm.default(
        input, residual, weight, eps, enable_pdl
    )


def _gemma_rmsnorm_internal(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    out: Optional[torch.Tensor],
    enable_pdl: Optional[bool],
) -> torch.Tensor:
    if out is None:
        out = torch.empty_like(input)
    if enable_pdl is None:
        enable_pdl = is_arch_support_pdl()
    torch.ops.sgl_kernel.gemma_rmsnorm.default(out, input, weight, eps, enable_pdl)
    return out


def _gemma_fused_add_rmsnorm_internal(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    enable_pdl: Optional[bool],
) -> None:
    if enable_pdl is None:
        enable_pdl = is_arch_support_pdl()
    torch.ops.sgl_kernel.gemma_fused_add_rmsnorm.default(
        input, residual, weight, eps, enable_pdl
    )


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
    if _has_flashinfer and input.dtype in _FLASHINFER_NORM_SUPPORTED_DTYPES:
        return _flashinfer_norm.rmsnorm(input, weight, eps, out, enable_pdl)
    else:
        return _rmsnorm_internal(input, weight, eps, out, enable_pdl)


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
    if _has_flashinfer and input.dtype in _FLASHINFER_NORM_SUPPORTED_DTYPES:
        _flashinfer_norm.fused_add_rmsnorm(input, residual, weight, eps, enable_pdl)
    else:
        _fused_add_rmsnorm_internal(input, residual, weight, eps, enable_pdl)


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
    if _has_flashinfer and input.dtype in _FLASHINFER_NORM_SUPPORTED_DTYPES:
        return _flashinfer_norm.gemma_rmsnorm(input, weight, eps, out, enable_pdl)
    else:
        return _gemma_rmsnorm_internal(input, weight, eps, out, enable_pdl)


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
    if _has_flashinfer and input.dtype in _FLASHINFER_NORM_SUPPORTED_DTYPES:
        _flashinfer_norm.gemma_fused_add_rmsnorm(
            input, residual, weight, eps, enable_pdl
        )
    else:
        _gemma_fused_add_rmsnorm_internal(input, residual, weight, eps, enable_pdl)


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


def dsv4_fused_q_norm_rope(
    q_input: torch.Tensor,
    freqs_cis: torch.Tensor,
    positions: torch.Tensor,
    eps: float = 1e-6,
    q_output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """DeepSeek-V4 fused Q RMSNorm (no weight) + RoPE.

    Parameters
    ----------
    q_input  : (B, num_q_heads, head_dim) bfloat16
    freqs_cis: (max_pos, rope_dim) float32, re/im interleaved
    positions: (B,) int32
    eps      : RMSNorm epsilon
    q_output : optional pre-allocated output tensor
    """
    if q_output is None:
        q_output = torch.empty_like(q_input)
    torch.ops.sgl_kernel.dsv4_fused_q_norm_rope.default(
        q_input, q_output, freqs_cis, positions, eps
    )
    return q_output


def dsv4_fused_k_norm_rope_flashmla(
    kv: torch.Tensor,
    kv_weight: torch.Tensor,
    freqs_cis: torch.Tensor,
    positions: torch.Tensor,
    out_loc: torch.Tensor,
    kvcache: torch.Tensor,
    eps: float = 1e-6,
    page_size: int = 1,
) -> None:
    """DeepSeek-V4 fused K RMSNorm + RoPE + FlashMLA FP8 store.

    Parameters
    ----------
    kv       : (B, 512) bfloat16
    kv_weight: (512,) bfloat16
    freqs_cis: (max_pos, 64) float32
    positions: (B,) int32
    out_loc  : (B,) int32  cache slot ids
    kvcache  : (npages, page_bytes) uint8
    eps      : RMSNorm epsilon
    page_size: page size (power of 2)
    """
    torch.ops.sgl_kernel.dsv4_fused_k_norm_rope_flashmla.default(
        kv, kv_weight, freqs_cis, positions, out_loc, kvcache, eps, page_size
    )


def dsv4_fused_q_indexer_rope_hadamard_quant(
    q_input: torch.Tensor,
    q_fp8: torch.Tensor,
    weight: torch.Tensor,
    weights_out: torch.Tensor,
    weight_scale: float,
    freqs_cis: torch.Tensor,
    positions: torch.Tensor,
) -> None:
    """DeepSeek-V4 fused Q indexer: RoPE + Hadamard + FP8 quant.

    Parameters
    ----------
    q_input    : (B, num_heads, 128) bfloat16
    q_fp8      : (B, num_heads, 128) fp8_e4m3 output
    weight     : (B, num_heads) bfloat16
    weights_out: (B, num_heads, 1) float32 output
    weight_scale: scalar
    freqs_cis  : (max_pos, 64) float32
    positions  : (B,) int32
    """
    torch.ops.sgl_kernel.dsv4_fused_q_indexer_rope_hadamard_quant.default(
        q_input, q_fp8, weight, weights_out, weight_scale, freqs_cis, positions
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
