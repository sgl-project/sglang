"""Out-of-place RMSNorm that rounds the normalized activation before the weight multiply."""

from typing import Optional, Union

import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_cast_before_mul_kernel(
    X,
    W,
    Y,
    RESIDUAL,
    POST_RESIDUAL,
    RESIDUAL_OUT,
    rows,
    x_row_stride,
    D: tl.constexpr,
    EPS: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    HAS_POST_RESIDUAL: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = (tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS))[:, None]
    col = tl.arange(0, BLOCK_D)[None, :]
    mask = (row < rows) & (col < D)
    row = row.to(tl.int64)
    x = tl.load(X + row * x_row_stride + col, mask, other=0.0).to(tl.float32)
    if HAS_RESIDUAL:
        # Same fp32 sum order as RMSNorm.forward_native: (x + residual) + post_residual.
        x += tl.load(RESIDUAL + row * D + col, mask, other=0.0).to(tl.float32)
        if HAS_POST_RESIDUAL:
            x += tl.load(POST_RESIDUAL + row * D + col, mask, other=0.0).to(tl.float32)
        tl.store(RESIDUAL_OUT + row * D + col, x.to(Y.dtype.element_ty), mask)
    inv_rms = tl.rsqrt(tl.sum(x * x, axis=1) / D + EPS)
    normalized = (x * inv_rms[:, None]).to(Y.dtype.element_ty)
    weight = tl.load(W + col, col < D, other=0.0).to(tl.float32)
    output = normalized.to(tl.float32) * weight
    tl.store(Y + row * D + col, output.to(Y.dtype.element_ty), mask)


def rmsnorm_cast_before_mul(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    residual: Optional[torch.Tensor] = None,
    post_residual_addition: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """Match RMSNorm(cast_x_before_out_mul=True).forward_native, with or without a residual.

    Without a residual returns weight * (x * rsqrt(mean(x**2) + eps)).to(x.dtype).
    With one, normalizes x + residual (+ post_residual_addition) summed in fp32 and
    also returns that sum rounded to x.dtype as the new residual. Inputs are not modified.
    """
    assert x.device.type in ("cuda", "xpu") and x.stride(-1) == 1
    assert x.dtype in (torch.bfloat16, torch.float16) and weight.dtype == x.dtype
    assert x.device == weight.device and weight.shape == (x.shape[-1],)
    extra_inputs = [t for t in (residual, post_residual_addition) if t is not None]
    assert residual is not None or post_residual_addition is None
    assert all(
        t.shape == x.shape and t.is_contiguous() and t.device == x.device
        for t in extra_inputs
    )
    dim = x.shape[-1]
    # A view for evenly strided rows, such as the q/k halves from chunk(2, dim=-1).
    rows_2d = x.reshape(-1, dim)
    rows = rows_2d.shape[0]
    output = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    residual_out = torch.empty_like(output) if residual is not None else None
    if rows:
        block_d = triton.next_power_of_2(dim)
        # Measured on Arc Pro B60: one row per program for hidden-size norms,
        # several rows per program for 128-wide per-head norms.
        block_rows = max(1, min(16, 2048 // block_d))
        if x.dtype is torch.float16:
            # IGC 2.24.8 drops the fp16 rounding with several rows per program (IGC 2.36.3
            # does not); drop this once XPU drivers older than IGC 2.36 are unsupported.
            block_rows = 1
        # Triton launches on the current device; a split model has tensors on several.
        with torch.get_device_module(x.device).device(x.device):
            _rmsnorm_cast_before_mul_kernel[(triton.cdiv(rows, block_rows),)](
                rows_2d,
                weight.contiguous(),
                output,
                residual,
                post_residual_addition,
                residual_out,
                rows,
                rows_2d.stride(0),
                dim,
                eps,
                residual is not None,
                post_residual_addition is not None,
                block_rows,
                block_d,
                num_warps=8 if block_d >= 2048 else 4,
                enable_fp_fusion=False,
            )
    if residual is None:
        return output
    return output, residual_out
