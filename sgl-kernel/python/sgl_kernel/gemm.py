from typing import List, Optional, Tuple

import torch
from sgl_kernel.utils import _get_cache_buf, get_cuda_stream


def awq_dequantize(
    qweight: torch.Tensor, scales: torch.Tensor, qzeros: torch.Tensor
) -> torch.ByteTensor:
    r"""
    Dequantize quantized weights using the AWQ scheme.

    This function reconstructs the original (or approximate) weight values from quantized
    weights, scaling factors, and zero-point offsets.

    Parameters
    ----------
    qweight : torch.Tensor
        Quantized weight tensor. Typically of type `torch.uint8` or other integer type,
        and shape compatible with the underlying model layer (e.g., (out_features, in_features)).
    scales : torch.Tensor
        Scaling factors used for dequantization. The shape should be broadcastable to `qweight`,
        commonly per-channel or per-group.
    qzeros : torch.Tensor
        Zero-point offsets for dequantization. The shape should be broadcastable to `qweight`,
        commonly per-channel or per-group.

    Returns
    -------
    dequantized_weight : torch.ByteTensor
        Dequantized weight tensor, typically of type `torch.float32` or `torch.float16`
        (note: if a byte tensor is desired, clarify the use case). The shape matches
        `qweight` after dequantization.

    Notes
    -----
    - The dequantization formula is generally: `dequantized_weight = scales * (qweight - qzeros)`
    - Make sure that `scales` and `qzeros` are properly aligned (broadcastable) with `qweight`.
    - This function is commonly used in quantization-aware inference or model compression pipelines.
    """
    return torch.ops.sgl_kernel.awq_dequantize.default(qweight, scales, qzeros)


def int8_scaled_mm(mat_a, mat_b, scales_a, scales_b, out_dtype, bias=None):
    r"""
    Perform matrix multiplication of two int8 quantized matrices with scaling factors,
    optional bias, and output type casting.

    This function computes the result of `mat_a @ mat_b` where both inputs are int8 tensors
    that have been quantized with per-tensor or per-channel scales. The results are appropriately
    rescaled to the target output dtype, and an optional bias can be added.

    Parameters
    ----------
    mat_a : torch.Tensor
        Left matrix operand of shape (M, K), dtype `torch.int8`.
    mat_b : torch.Tensor
        Right matrix operand of shape (K, N), dtype `torch.int8`.
    scales_a : torch.Tensor or float
        Scaling factor(s) for `mat_a`. Can be a scalar, 1D tensor (per-row or per-channel), or
        broadcastable to `mat_a`.
    scales_b : torch.Tensor or float
        Scaling factor(s) for `mat_b`. Can be a scalar, 1D tensor (per-column or per-channel), or
        broadcastable to `mat_b`.
    out_dtype : torch.dtype
        Desired output data type (e.g., `torch.float16`, `torch.float32`).
    bias : torch.Tensor or None, optional
        Optional bias tensor of shape (N,) or broadcastable to output, added after
        matrix multiplication and scaling.

    Returns
    -------
    output : torch.Tensor
        Output matrix of shape (M, N), with type specified by `out_dtype`.

    Notes
    -----
    - The computation is typically:
      `output = (mat_a.float() * scales_a) @ (mat_b.float() * scales_b)`
      The result is then cast to `out_dtype`. If `bias` is provided, it is added after scaling.
    - This function is commonly used in quantization-aware inference, where fast int8 computation
      is combined with floating-point rescaling for accuracy.
    - Ensure that the provided scales are properly aligned (broadcastable) with their respective matrices.
    """
    return torch.ops.sgl_kernel.int8_scaled_mm.default(
        mat_a,
        mat_b,
        scales_a,
        scales_b,
        out_dtype,
        bias,
    )


def fp8_blockwise_scaled_mm(mat_a, mat_b, scales_a, scales_b, out_dtype):
    r"""
    Perform matrix multiplication of two FP8 blockwise-quantized matrices with
    per-block scaling factors and output type casting.

    This function computes the product of two matrices, where both inputs are
    in FP8 (8-bit floating point) format and quantized in blocks, each with its
    own scaling factor. The inputs are dequantized using the provided scaling
    factors, matrix multiplication is performed, and the result is cast to the
    specified output dtype.

    Parameters
    ----------
    mat_a : torch.Tensor
        Left matrix operand, shape (M, K), dtype `torch.uint8` or custom FP8 dtype.
        Each block of this matrix is quantized with its corresponding scale in `scales_a`.
    mat_b : torch.Tensor
        Right matrix operand, shape (K, N), dtype `torch.uint8` or custom FP8 dtype.
        Each block of this matrix is quantized with its corresponding scale in `scales_b`.
    scales_a : torch.Tensor
        Scaling factors for dequantizing `mat_a`. The shape depends on the block size
        and quantization scheme, e.g., (num_blocks_a,) or broadcastable to `mat_a`.
    scales_b : torch.Tensor
        Scaling factors for dequantizing `mat_b`. The shape depends on the block size
        and quantization scheme, e.g., (num_blocks_b,) or broadcastable to `mat_b`.
    out_dtype : torch.dtype
        Desired output data type (e.g., `torch.float16`, `torch.float32`).

    Returns
    -------
    output : torch.Tensor
        Output matrix of shape (M, N), with type specified by `out_dtype`.

    Notes
    -----
    - The typical computation is:
      1. Dequantize `mat_a` and `mat_b` blockwise:
         `dequant_a = fp8_dequantize(mat_a, scales_a)`
         `dequant_b = fp8_dequantize(mat_b, scales_b)`
      2. Perform matrix multiplication:
         `output = dequant_a @ dequant_b`
      3. Cast result to `out_dtype`.
    - FP8 blockwise quantization enables efficient memory usage and fast computation,
      useful in large language models and transformer inference.
    - The implementation of `fp8_dequantize` should handle the specifics of the FP8 format used.
    - Ensure that scaling factors and block partitioning are compatible with the quantized matrices.
    """
    return torch.ops.sgl_kernel.fp8_blockwise_scaled_mm.default(
        mat_a,
        mat_b,
        scales_a,
        scales_b,
        out_dtype,
    )


def fp8_scaled_mm(mat_a, mat_b, scales_a, scales_b, out_dtype, bias=None):
    r"""
    Perform matrix multiplication of two FP8-quantized matrices with scaling factors,
    optional bias addition, and output type casting.

    This function computes the product of two matrices, where both inputs are in FP8
    (8-bit floating point) format and quantized with their respective scaling factors.
    The inputs are dequantized using the provided scales, matrix multiplication is performed,
    bias is optionally added, and the result is cast to the specified output dtype.

    Parameters
    ----------
    mat_a : torch.Tensor
        Left matrix operand, shape (M, K), dtype `torch.uint8` or custom FP8 dtype.
        Quantized with `scales_a`.
    mat_b : torch.Tensor
        Right matrix operand, shape (K, N), dtype `torch.uint8` or custom FP8 dtype.
        Quantized with `scales_b`.
    scales_a : torch.Tensor or float
        Scaling factor(s) for dequantizing `mat_a`. Can be scalar, 1D, or broadcastable to `mat_a`.
    scales_b : torch.Tensor or float
        Scaling factor(s) for dequantizing `mat_b`. Can be scalar, 1D, or broadcastable to `mat_b`.
    out_dtype : torch.dtype
        Desired output data type (e.g., `torch.float16`, `torch.float32`).
    bias : torch.Tensor or None, optional
        Optional bias tensor of shape (N,) or broadcastable to output. Added after matrix multiplication.

    Returns
    -------
    output : torch.Tensor
        Output matrix of shape (M, N), with type specified by `out_dtype`.

    Notes
    -----
    - The typical computation is:
        1. Dequantize both matrices:
            `dequant_a = fp8_dequantize(mat_a, scales_a)`
            `dequant_b = fp8_dequantize(mat_b, scales_b)`
        2. Matrix multiplication:
            `output = dequant_a @ dequant_b`
        3. Add bias if provided:
            `output += bias`
        4. Cast result to `out_dtype`.
    - Ensure that scaling factors are compatible (broadcastable) with their respective matrices.
    - The implementation of `fp8_dequantize` should properly interpret the FP8 format.
    - This function is useful in quantization-aware inference for models employing FP8 quantization.

    """
    return torch.ops.sgl_kernel.fp8_scaled_mm.default(
        mat_a,
        mat_b,
        scales_a,
        scales_b,
        out_dtype,
        bias,
    )


def _bmm_fp8_internal(
    workspace_buffer: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    D: torch.Tensor,
    A_scale: torch.Tensor,
    B_scale: torch.Tensor,
) -> None:
    cublas_handle = torch.cuda.current_blas_handle()
    torch.ops.sgl_kernel.bmm_fp8.default(
        A,
        B,
        D,
        A_scale,
        B_scale,
        workspace_buffer,
        cublas_handle,
        get_cuda_stream(),
    )


def bmm_fp8(
    A: torch.Tensor,
    B: torch.Tensor,
    A_scale: torch.Tensor,
    B_scale: torch.Tensor,
    dtype: torch.dtype,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""
    Perform batch matrix multiplication (BMM) of two FP8-quantized tensors with scaling factors.

    This function computes batch-wise products of two FP8-quantized tensors by first dequantizing
    them using their respective scaling factors, performing batch matrix multiplication, and
    casting the result to the specified floating-point dtype. If an output tensor is provided,
    the result is written in-place to `out`.

    Parameters
    ----------
    A : torch.Tensor
        Left batch matrix operand, shape (batch_size, M, K), dtype `torch.uint8` or custom FP8 dtype.
        Quantized with scaling factors in `A_scale`.
    B : torch.Tensor
        Right batch matrix operand, shape (batch_size, K, N), dtype `torch.uint8` or custom FP8 dtype.
        Quantized with scaling factors in `B_scale`.
    A_scale : torch.Tensor or float
        Scaling factor(s) for dequantizing `A`. Can be scalar, 1D, or broadcastable to `A`.
    B_scale : torch.Tensor or float
        Scaling factor(s) for dequantizing `B`. Can be scalar, 1D, or broadcastable to `B`.
    dtype : torch.dtype
        Desired output data type (e.g., `torch.float16`, `torch.float32`).
    out : torch.Tensor, optional
        Optional output tensor for writing results in-place. Must have shape (batch_size, M, N) and dtype `dtype`.

    Returns
    -------
    output : torch.Tensor
        Output tensor of shape (batch_size, M, N), with type specified by `dtype`.

    Notes
    -----
    - Dequantization is performed using the provided scaling factors:
        `A_dequant = fp8_dequantize(A, A_scale)`
        `B_dequant = fp8_dequantize(B, B_scale)`
    - Batch matrix multiplication is then performed:
        `output = torch.bmm(A_dequant, B_dequant)`
    - If `out` is provided, the result is written in-place to `out`.
    - Ensure that scaling factors are correctly broadcastable to their respective tensors.
    - This function is useful for quantization-aware training or inference with FP8 quantized models.

    """
    if out is None:
        out = torch.empty(
            (A.shape[0], A.shape[1], B.shape[2]),
            device=A.device,
            dtype=dtype,
        )
    workspace_buffer = _get_cache_buf("bmm_fp8_workspace", 32 * 1024 * 1024, A.device)
    _bmm_fp8_internal(workspace_buffer, A, B, out, A_scale, B_scale)
    return out


def sgl_per_token_group_quant_fp8(
    input: torch.Tensor,
    output_q: torch.Tensor,
    output_s: torch.Tensor,
    group_size: int,
    eps: float,
    fp8_min: float,
    fp8_max: float,
) -> None:
    r"""
    Perform single-pass per-token group-wise quantization of a tensor into FP8 format.

    This function partitions the input tensor into groups of the specified size,
    computes a scaling factor for each group, and quantizes the values into the
    FP8 range [`fp8_min`, `fp8_max`]. The quantized values and their corresponding
    scaling factors are written to the provided output tensors.

    Parameters
    ----------
    input : torch.Tensor
        Input tensor to be quantized. Can be of any floating-point type and shape.
    output_q : torch.Tensor
        Output tensor for quantized values, typically of dtype `torch.uint8` or custom FP8.
        Must have the same shape as `input`.
    output_s : torch.Tensor
        Output tensor for scaling factors per group. Shape should match the number of groups
        along the quantized dimension (e.g., (num_groups,)).
    group_size : int
        Number of tokens or elements in each group for quantization.
    eps : float
        Small value added for numerical stability during scaling calculation.
    fp8_min : float
        Minimum representable value for FP8 quantization.
    fp8_max : float
        Maximum representable value for FP8 quantization.

    Returns
    -------
    None

    Notes
    -----
    - The input tensor is divided into groups of size `group_size` (the last group may be smaller).
    - For each group:
        1. Calculate a scaling factor based on the maximum absolute value in the group and `eps`.
        2. Quantize each value in the group to the FP8 range, clamping to [`fp8_min`, `fp8_max`].
        3. Store the quantized values in `output_q` and the scaling factor in `output_s`.
    - This method is often used for efficient group-wise quantization in large transformer models.
    - The quantization is performed in-place on the provided output tensors.

    """
    torch.ops.sgl_kernel.sgl_per_token_group_quant_fp8.default(
        input, output_q, output_s, group_size, eps, fp8_min, fp8_max
    )


def sgl_per_token_group_quant_int8(
    input: torch.Tensor,
    output_q: torch.Tensor,
    output_s: torch.Tensor,
    group_size: int,
    eps: float,
    int8_min: float,
    int8_max: float,
) -> None:
    r"""
    Perform single-pass per-token group-wise quantization of a tensor into int8 format.

    This function divides the input tensor into groups of the specified size,
    computes a scaling factor for each group, and quantizes each value in the group
    into the int8 range [`int8_min`, `int8_max`]. The quantized values and their
    corresponding scaling factors are written to the provided output tensors.

    Parameters
    ----------
    input : torch.Tensor
        Input tensor to be quantized. Should be of a floating-point dtype. Shape is arbitrary.
    output_q : torch.Tensor
        Output tensor for quantized values, typically of dtype `torch.int8`.
        Must have the same shape as `input`.
    output_s : torch.Tensor
        Output tensor for scaling factors per group. Shape should match the number of groups
        along the quantized dimension (e.g., (num_groups,)).
    group_size : int
        Number of elements in each group for quantization. The last group may be smaller if
        the tensor size is not divisible by `group_size`.
    eps : float
        Small value added for numerical stability when calculating the scaling factor.
    int8_min : float
        Minimum representable value for int8 quantization (typically -128).
    int8_max : float
        Maximum representable value for int8 quantization (typically 127).

    Returns
    -------
    None

    Notes
    -----
    - The input tensor is partitioned into groups of size `group_size`.
    - For each group:
        1. Calculate a scaling factor, usually based on the maximum absolute value in the group plus `eps`.
        2. Quantize each value in the group using this scaling factor and clamp to [`int8_min`, `int8_max`].
        3. Store quantized values in `output_q` and the scaling factor in `output_s`.
    - This method is commonly used for group-wise quantization in efficient neural network inference.
    - Output tensors must be pre-allocated and will be written in-place.
    """
    torch.ops.sgl_kernel.sgl_per_token_group_quant_int8.default(
        input, output_q, output_s, group_size, eps, int8_min, int8_max
    )


def sgl_per_tensor_quant_fp8(
    input: torch.Tensor,
    output_q: torch.Tensor,
    output_s: torch.Tensor,
    is_static: bool,
) -> None:
    r"""
    Perform single-pass per-tensor quantization of a tensor into FP8 format.

    This function computes a scaling factor for the entire input tensor (per-tensor quantization),
    then quantizes all values into FP8 representation using this shared scale. The quantized values
    and scaling factor are written to the provided output tensors.

    Parameters
    ----------
    input : torch.Tensor
        The input tensor to be quantized. Should be of floating-point dtype and any shape.
    output_q : torch.Tensor
        Output tensor for storing quantized values, typically of dtype `torch.uint8` or custom FP8.
        Must have the same shape as `input`.
    output_s : torch.Tensor
        Output tensor for storing the scaling factor. Should be a scalar tensor.
    is_static : bool
        If True, use a static (precomputed or fixed) scale for quantization;
        if False, compute scale dynamically (e.g., from the max absolute value in `input`).

    Returns
    -------
    None

    Notes
    -----
    - This function performs per-tensor quantization, i.e., the entire tensor shares a single scaling factor.
    - Typical quantization procedure:
        1. Determine the scaling factor, either statically or dynamically.
        2. Quantize each input value using this scale and clamp to the FP8 representable range.
        3. Store quantized values in `output_q` and the scaling factor in `output_s`.
    - Output tensors must be pre-allocated and will be written in-place.
    - This method is often used for efficient quantization in transformer models or other large-scale networks
      where per-tensor quantization is sufficient.
    """
    torch.ops.sgl_kernel.sgl_per_tensor_quant_fp8.default(
        input, output_q, output_s, is_static
    )


def sgl_per_token_quant_fp8(
    input: torch.Tensor,
    output_q: torch.Tensor,
    output_s: torch.Tensor,
) -> None:
    r"""
    Perform single-pass per-token quantization of a tensor into FP8 format.

    This function computes a scaling factor for each token (e.g., each row or element along a specified axis)
    and quantizes the corresponding values into FP8 format using its unique scale.
    The quantized values and their scaling factors are written to the provided output tensors.

    Parameters
    ----------
    input : torch.Tensor
        The input tensor to be quantized. Should be of floating-point dtype.
        Shape is typically (num_tokens, feature_dim) or similar, but can be arbitrary.
    output_q : torch.Tensor
        Output tensor for quantized values, typically of dtype `torch.uint8` or custom FP8.
        Must have the same shape as `input`.
    output_s : torch.Tensor
        Output tensor for scaling factors per token. Shape should match the first dimension of `input` (num_tokens,).

    Returns
    -------
    None

    Notes
    -----
    - The function performs per-token quantization, often meaning "per row" in a 2D tensor.
    - For each token:
        1. Compute a scaling factor, usually based on the maximum absolute value of that token plus a small epsilon.
        2. Quantize all values of the token using this scale and clamp to the FP8 representable range.
        3. Store quantized values in `output_q` and the scaling factor in `output_s`.
    - Output tensors must be pre-allocated and will be written in-place.
    - This method is useful for cases where per-token dynamic range is important, such as in sequence models
      or transformer inference, helping preserve accuracy with aggressive quantization.
    """
    torch.ops.sgl_kernel.sgl_per_token_quant_fp8.default(input, output_q, output_s)


def cutlass_scaled_fp4_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    block_scale_a: torch.Tensor,
    block_scale_b: torch.Tensor,
    alpha: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    r"""
    Perform matrix multiplication of two block-wise scaled FP4 quantized matrices using CUTLASS.

    This function multiplies two input matrices that are quantized in FP4 (4-bit floating point) format
    with per-block scaling. Each block of both matrices has its own scaling factor. The quantized inputs
    are dequantized using their respective block scales, multiplied as matrices, optionally scaled by `alpha`,
    and the result is cast to the specified output dtype.

    Parameters
    ----------
    a : torch.Tensor
        Left matrix operand, shape (M, K), dtype typically `torch.uint8` (to pack two FP4 values per byte).
        Quantized block-wise with scaling factors provided in `block_scale_a`.
    b : torch.Tensor
        Right matrix operand, shape (K, N), dtype typically `torch.uint8`.
        Quantized block-wise with scaling factors provided in `block_scale_b`.
    block_scale_a : torch.Tensor
        Per-block scaling factors for dequantizing `a`. Shape and broadcasting depend on block partitioning.
    block_scale_b : torch.Tensor
        Per-block scaling factors for dequantizing `b`. Shape and broadcasting depend on block partitioning.
    alpha : torch.Tensor
        Optional scaling factor (scalar or tensor broadcastable to the output shape) applied after multiplication.
    out_dtype : torch.dtype
        Desired PyTorch dtype for the output (e.g., `torch.float16`, `torch.float32`).

    Returns
    -------
    output : torch.Tensor
        Output matrix of shape (M, N) with type specified by `out_dtype`.

    Notes
    -----
    - Typical computation steps:
        1. Dequantize `a` and `b` block-wise using their respective scaling factors:
             `dequant_a = fp4_dequantize(a, block_scale_a)`
             `dequant_b = fp4_dequantize(b, block_scale_b)`
        2. Perform matrix multiplication:
             `out = dequant_a @ dequant_b`
        3. Apply scaling factor `alpha`:
             `out = out * alpha`
        4. Cast the result to `out_dtype`.
    - CUTLASS refers to NVIDIA's CUDA Templates for Linear Algebra Subroutines and Solvers,
      which provides optimized GPU kernels for quantized GEMM.
    - Ensure correct handling of FP4 format: two FP4 values per byte, proper unpacking/packing, and correct scaling.
    - Block partitioning and scaling must be consistent between quantization and dequantization.

    """
    assert a.ndim == 2 and b.ndim == 2
    m, n = a.shape[0], b.shape[0]
    out = torch.empty((m, n), dtype=out_dtype, device=a.device)
    torch.ops.sgl_kernel.cutlass_scaled_fp4_mm.default(
        out, a, b, block_scale_a, block_scale_b, alpha
    )
    return out


def scaled_fp4_quant(
    input: torch.Tensor, input_global_scale: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Quantize input tensor to FP4 and return quantized tensor and scale.

    This function quantizes the last dimension of the given tensor `input`. For every 16 consecutive elements
    along the last dimension, a single dynamically computed scaling factor is shared. This scaling factor is
    further quantized using the provided `input_global_scale` and is stored in a swizzled layout (see
    https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-scale-factor-b-layout-4x).

    Parameters
    ----------
    input : torch.Tensor
        The input tensor to be quantized to FP4. Can be of any floating-point dtype and any shape.
    input_global_scale : torch.Tensor or float
        A scalar scaling factor applied globally to the entire tensor before group-wise scale calculation.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        - The quantized tensor in FP4, where every two FP4 values are packed into a single uint8.
        - The scaling factors for each group of 16 elements in the last dimension, represented in float8_e4m3
          and stored in a swizzled layout.

    Notes
    -----
    - The quantization is performed along the last dimension of the input tensor.
    - For each group of 16 elements:
        1. Compute a dynamic scaling factor for the group.
        2. Quantize the scaling factor using `input_global_scale`.
        3. Quantize the group of 16 values to FP4, sharing the same scale.
        4. Store the scales in a swizzled layout for hardware compatibility (see NVIDIA MMA documentation).
    - FP4 means 4-bit floating point, typically packed as two per uint8.
    - The function returns two tensors: the packed FP4 quantized tensor, and the groupwise quantized scaling factors.

    """
    assert input.ndim >= 1, f"input.ndim needs to be >= 1, but got {input.ndim}."
    other_dims = 1 if input.ndim == 1 else -1
    input = input.reshape(other_dims, input.shape[-1])
    m, n = input.shape
    block_size = 16
    device = input.device

    assert n % block_size == 0, f"last dim has to be multiple of 16, but got {n}."
    assert input.dtype in (
        torch.float16,
        torch.bfloat16,
    ), f"input.dtype needs to be fp16 or bf16 but got {input.dtype}."

    # Two fp4 values will be packed into an uint8.
    output = torch.empty((m, n // 2), device=device, dtype=torch.uint8)

    # We use the rounded values to store the swizzled values. Then, the scaling
    # factors in float8_e4m3fn are packed into an int32 for every 4 values.
    rounded_m = ((m + 128 - 1) // 128) * 128
    scale_n = n // block_size
    rounded_n = ((scale_n + 4 - 1) // 4) * 4
    output_scale = torch.empty(
        (rounded_m, rounded_n // 4), device=device, dtype=torch.int32
    )

    torch.ops.sgl_kernel.scaled_fp4_quant.default(
        output, input, output_scale, input_global_scale
    )
    output_scale = output_scale.view(torch.float8_e4m3fn)
    return output, output_scale
