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
