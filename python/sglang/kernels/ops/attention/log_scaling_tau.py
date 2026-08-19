import torch
import triton
import triton.language as tl


@triton.jit
def _apply_log_scaling_tau_kernel(
    x_ptr,
    tau_ptr,  # [rows] fp32 (flattened per-row scale)
    out_ptr,  # [rows, inner] contiguous, same dtype as x
    x_row_stride,
    inner,
    total,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total
    row = offs // inner
    col = offs % inner
    x = tl.load(x_ptr + row * x_row_stride + col, mask=mask).to(tl.float32)
    tau = tl.load(tau_ptr + row, mask=mask)
    y = x * tau
    tl.store(out_ptr + offs, y.to(out_ptr.dtype.element_ty), mask=mask)


def apply_log_scaling_tau(x: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    """out = (x.float() * tau).to(x.dtype) with tau broadcast per leading row,
    fused into one launch. x may carry a leading-dim stride (the q slice of the
    fused qkvr output); its trailing dims must be contiguous. No dynamo: the
    torch.compile'd predecessor's call sites spanned enough rank /
    dispatch-key / 0-1 specialization variants (target + de-tied MTP heads) to
    exceed the recompile limit, which crashed (fullgraph) or wedged capture
    (raised limit)."""
    rows = x.shape[0]
    inner = x.numel() // rows if rows else 0
    inner_contiguous = x.stride(-1) == 1 and (
        x.dim() == 2 or x.stride(-2) == x.shape[-1] * x.stride(-1)
    )
    if rows == 0 or inner == 0 or not inner_contiguous:
        return (x.float() * tau).to(x.dtype)

    if (
        x.is_cuda
        and x.dtype == torch.bfloat16
        and inner % 8 == 0
        and x.data_ptr() % 16 == 0
        and (x.stride(0) * 2) % 16 == 0
    ):
        # Vectorized JIT kernel (16B loads, one row divide per vector) --
        # bit-identical output (same fp32-mul + bf16-round), ~2-3x the
        # scalar triton kernel below at every size.
        from sglang.kernels.ops.attention.inkling_row_scale import row_scale_bf16

        x2d = torch.as_strided(x, (rows, inner), (x.stride(0), 1))
        return row_scale_bf16(x2d, tau.reshape(rows).float()).view(x.shape)

    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    total = rows * inner
    BLOCK = 1024
    _apply_log_scaling_tau_kernel[(triton.cdiv(total, BLOCK),)](
        x,
        tau.reshape(rows).to(torch.float32),
        out,
        x.stride(0),
        inner,
        total,
        BLOCK=BLOCK,
    )
    return out
