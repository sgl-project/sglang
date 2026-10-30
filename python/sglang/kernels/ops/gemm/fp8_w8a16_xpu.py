"""Weight-only block-FP8 linear (W8A16) for Intel Xe2 GPUs.

Decode-sized inputs run sgl-kernel-xpu's fused dequant GEMM, which streams the fp8 weight once.
Larger inputs are compute-bound, so they dequantize the weight to bf16 and use the bf16 GEMM.
"""

from functools import lru_cache
from typing import Optional

import torch
import triton
import triton.language as tl

# Measured on Arc Pro B70 at Qwen3-32B TP4 shapes: the two paths tie at 512 rows, and above
# that the weight dequant plus bf16 GEMM wins.
_FUSED_MAX_ROWS = 512


@lru_cache(maxsize=1)
def has_xpu_w8a16_gemm() -> bool:
    try:
        import sgl_kernel  # noqa: F401
    except ImportError:
        return False
    return hasattr(torch.ops.sgl_kernel, "fp8_w8a16_mm_xe20")


@triton.jit
def _dequant_block_fp8_kernel(
    w_ptr,
    s_ptr,
    out_ptr,
    N,
    K,
    stride_wn,
    stride_sn,
    stride_sk,
    stride_on,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    mask = (offs_n[:, None] < N) & (offs_k[None, :] < K)
    w = tl.load(
        w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :], mask=mask, other=0.0
    )
    # One program covers exactly one 128x128 scale block.
    s = tl.load(s_ptr + pid_n * stride_sn + pid_k * stride_sk)
    out = (w.to(tl.float32) * s).to(out_ptr.dtype.element_ty)
    tl.store(out_ptr + offs_n[:, None] * stride_on + offs_k[None, :], out, mask=mask)


def dequant_block_fp8_weight(
    weight: torch.Tensor, weight_scale: torch.Tensor, dtype: torch.dtype
) -> torch.Tensor:
    """fp8 e4m3 [N, K] with fp32 [ceil(N / 128), ceil(K / 128)] block scales -> dtype [N, K]."""
    n, k = weight.shape
    out = torch.empty((n, k), dtype=dtype, device=weight.device)
    grid = (triton.cdiv(n, 128), triton.cdiv(k, 128))
    _dequant_block_fp8_kernel[grid](
        weight,
        weight_scale,
        out,
        n,
        k,
        weight.stride(0),
        weight_scale.stride(0),
        weight_scale.stride(1),
        out.stride(0),
        BLOCK_N=128,
        BLOCK_K=128,
    )
    return out


def xpu_w8a16_block_fp8_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """bf16 input [..., K] @ (fp8 e4m3 weight [N, K] * fp32 [N / 128, K / 128] scales)^T."""
    input_2d = input.reshape(-1, input.shape[-1])
    if input_2d.shape[0] > _FUSED_MAX_ROWS:
        weight_bf16 = dequant_block_fp8_weight(weight, weight_scale, input.dtype)
        output = torch.nn.functional.linear(input_2d, weight_bf16, bias)
    else:
        if input_2d.stride(-1) != 1:
            input_2d = input_2d.contiguous()
        # The fused op's layout contract is external; fail loudly rather than
        # feed it a layout sgl-kernel-xpu#587 was not validated against.
        assert weight.stride(-1) == 1, (
            "fp8_w8a16_mm_xe20 requires row-contiguous weight"
        )
        assert weight_scale.stride(1) == 1, (
            "fp8_w8a16_mm_xe20 requires k-contiguous weight_scale "
            "(prepare_xpu_block_scale_for_scaled_mm's transposed layout is unsupported)"
        )
        output = torch.empty(
            (input_2d.shape[0], weight.shape[0]),
            dtype=input.dtype,
            device=input.device,
        )
        torch.ops.sgl_kernel.fp8_w8a16_mm_xe20(output, input_2d, weight, weight_scale)
        if bias is not None:
            output += bias
    return output.view(*input.shape[:-1], weight.shape[0])
