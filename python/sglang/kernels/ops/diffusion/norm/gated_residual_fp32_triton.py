import torch
import triton  # type: ignore
import triton.language as tl  # type: ignore


@triton.jit
def _gated_residual_fp32_kernel(
    residual_ptr,
    x_ptr,
    gate_ptr,
    sum_ptr,
    rounded_ptr,
    L,
    C,
    BLOCK_L: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    offs_l = tl.program_id(0) * BLOCK_L + tl.arange(0, BLOCK_L)
    offs_c = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_c = offs_c < C
    mask = (offs_l[:, None] < L) & mask_c[None, :]
    offs = offs_l[:, None].to(tl.int64) * C + offs_c[None, :]
    residual = tl.load(residual_ptr + offs, mask=mask).to(tl.float32)
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    gate = tl.load(gate_ptr + offs_c, mask=mask_c)
    out = residual + x * gate[None, :]
    tl.store(sum_ptr + offs, out, mask=mask)
    tl.store(rounded_ptr + offs, out.to(rounded_ptr.dtype.element_ty), mask=mask)


def gated_residual_fp32(
    *, residual: torch.Tensor, x: torch.Tensor, gate: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """``residual + x * gate`` for half-precision ``residual``/``x`` and an fp32
    [1, 1, C] ``gate``, as (the fp32 sum, the sum rounded to ``x.dtype``).

    The same fp32 ops as the eager expression, without FMA contraction, so both
    outputs carry its bits. Returns None for any other operands.
    """
    if (
        not x.is_cuda
        or x.dtype not in (torch.bfloat16, torch.float16)
        or x.dim() != 3
        or x.numel() == 0
        or not x.is_contiguous()
        or residual.shape != x.shape
        or residual.dtype != x.dtype
        or residual.device != x.device
        or not residual.is_contiguous()
        or gate.dtype != torch.float32
        or gate.shape != (1, 1, x.shape[-1])
        or gate.device != x.device
        or not gate.is_contiguous()
    ):
        return None
    rows, channels = x.shape[0] * x.shape[1], x.shape[2]
    summed = torch.empty_like(x, dtype=torch.float32)
    rounded = torch.empty_like(x)
    # the tile of _scale_shift_fp32_rows_kernel, the other fp32 elementwise
    # pass over these shapes
    block_l, block_c = 16, 256
    _gated_residual_fp32_kernel[
        (triton.cdiv(rows, block_l), triton.cdiv(channels, block_c))
    ](
        residual,
        x,
        gate,
        summed,
        rounded,
        rows,
        channels,
        BLOCK_L=block_l,
        BLOCK_C=block_c,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return summed, rounded
