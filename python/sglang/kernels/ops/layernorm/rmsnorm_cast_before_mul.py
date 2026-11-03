"""Out-of-place RMSNorm that rounds the normalized activation before the weight multiply."""

import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_cast_before_mul_kernel(
    X,
    W,
    Y,
    rows,
    x_row_stride,
    D: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = (tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS))[:, None]
    col = tl.arange(0, BLOCK_D)[None, :]
    mask = (row < rows) & (col < D)
    row = row.to(tl.int64)
    x = tl.load(X + row * x_row_stride + col, mask, other=0.0).to(tl.float32)
    inv_rms = tl.rsqrt(tl.sum(x * x, axis=1) / D + EPS)
    normalized = (x * inv_rms[:, None]).to(Y.dtype.element_ty)
    weight = tl.load(W + col, col < D, other=0.0).to(tl.float32)
    output = normalized.to(tl.float32) * weight
    tl.store(Y + row * D + col, output.to(Y.dtype.element_ty), mask)


def rmsnorm_cast_before_mul(
    x: torch.Tensor, weight: torch.Tensor, eps: float
) -> torch.Tensor:
    """Match weight * (x.float() * rsqrt(mean(x.float()**2) + eps)).to(x.dtype)."""
    assert x.device.type in ("cuda", "xpu") and x.stride(-1) == 1
    assert x.dtype in (torch.bfloat16, torch.float16) and weight.dtype == x.dtype
    assert x.device == weight.device and weight.shape == (x.shape[-1],)
    dim = x.shape[-1]
    # A view for evenly strided rows, such as the q/k halves from chunk(2, dim=-1).
    rows_2d = x.reshape(-1, dim)
    rows = rows_2d.shape[0]
    output = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    if rows:
        block_d = triton.next_power_of_2(dim)
        # Measured on Arc Pro B60: one row per program for hidden-size norms,
        # several rows per program for 128-wide per-head norms.
        block_rows = max(1, min(16, 2048 // block_d))
        # Triton launches on the current device; a split model has tensors on several.
        with torch.get_device_module(x.device).device(x.device):
            _rmsnorm_cast_before_mul_kernel[(triton.cdiv(rows, block_rows),)](
                rows_2d,
                weight.contiguous(),
                output,
                rows,
                rows_2d.stride(0),
                dim,
                eps,
                block_rows,
                block_d,
                num_warps=8 if block_d >= 2048 else 4,
                enable_fp_fusion=False,
            )
    else:
        pass
    return output
