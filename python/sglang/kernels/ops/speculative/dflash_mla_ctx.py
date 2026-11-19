"""One launch normalizing every DFLASH MLA draft layer's context-KV latent rows."""

import torch
import triton
import triton.language as tl


@triton.jit
def _latent_rmsnorm_kernel(
    lat_ptr,  # [n, L, R + P]
    w_ptr,  # [L, R]
    out_ptr,  # [L, n, R]
    n,
    eps,
    L: tl.constexpr,
    R: tl.constexpr,
    P: tl.constexpr,
):
    row = tl.program_id(0)
    layer = tl.program_id(1)
    cols = tl.arange(0, R)
    x = tl.load(lat_ptr + (row * L + layer) * (R + P) + cols).to(tl.float32)
    rstd = tl.rsqrt(tl.sum(x * x, axis=0) / R + eps)
    w = tl.load(w_ptr + layer * R + cols).to(tl.float32)
    tl.store(
        out_ptr + (layer * n + row) * R + cols,
        (x * rstd * w).to(out_ptr.dtype.element_ty),
    )


def stacked_latent_rmsnorm(
    lat: torch.Tensor, weight: torch.Tensor, eps: float
) -> torch.Tensor:
    """RMSNorm layer ``l``'s latent part of ``lat`` [n, L, R + P] with ``weight[l]``
    ([L, R]); returns [L, n, R]. R must be a power of two."""
    n, L, width = lat.shape
    R = weight.shape[1]
    out = torch.empty((L, n, R), dtype=lat.dtype, device=lat.device)
    if n:
        _latent_rmsnorm_kernel[(n, L)](lat, weight, out, n, eps, L=L, R=R, P=width - R)
    return out
