from typing import List, Optional

import torch
from sgl_kernel.utils import _get_cache_buf, get_cuda_stream


def awq_dequantize(
    qweight: torch.Tensor, scales: torch.Tensor, qzeros: torch.Tensor
) -> torch.ByteTensor:
    return torch.ops.sgl_kernel.awq_dequantize(qweight, scales, qzeros)


def int8_scaled_mm(mat_a, mat_b, scales_a, scales_b, out_dtype, bias=None):
    return torch.ops.sgl_kernel.int8_scaled_mm(
        mat_a,
        mat_b,
        scales_a,
        scales_b,
        out_dtype,
        bias,
    )


def fp8_blockwise_scaled_mm(mat_a, mat_b, scales_a, scales_b, out_dtype):
    return torch.ops.sgl_kernel.fp8_blockwise_scaled_mm(
        mat_a,
        mat_b,
        scales_a,
        scales_b,
        out_dtype,
    )


def fp8_scaled_mm(mat_a, mat_b, scales_a, scales_b, out_dtype, bias=None):
    return torch.ops.sgl_kernel.fp8_scaled_mm(
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
    torch.ops.sgl_kernel.bmm_fp8(
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
    torch.ops.sgl_kernel.sgl_per_token_group_quant_fp8(
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
    torch.ops.sgl_kernel.sgl_per_token_group_quant_int8(
        input, output_q, output_s, group_size, eps, int8_min, int8_max
    )


def sgl_per_tensor_quant_fp8(
    input: torch.Tensor,
    output_q: torch.Tensor,
    output_s: torch.Tensor,
    is_static: bool,
) -> None:
    torch.ops.sgl_kernel.sgl_per_tensor_quant_fp8(input, output_q, output_s, is_static)


def cublas_grouped_gemm(
    inputs: List[torch.Tensor],
    weights: List[torch.Tensor],
    outputs: List[torch.Tensor],
    out_dtype: torch.dtype,
) -> None:
    assert (
        len(inputs) > 0 and len(weights) > 0 and len(outputs) > 0
    ), "Inputs/weights/outputs should not be empty!"
    cublas_handle = torch.cuda.current_blas_handle()
    torch.ops.sgl_kernel.cublas_grouped_gemm(
        inputs,
        weights,
        outputs,
        out_dtype,
        cublas_handle,
        get_cuda_stream(),
    )


def sgl_per_token_quant_fp8(
    input: torch.Tensor,
    output_q: torch.Tensor,
    output_s: torch.Tensor,
) -> None:
    torch.ops.sgl_kernel.sgl_per_token_quant_fp8(input, output_q, output_s)
