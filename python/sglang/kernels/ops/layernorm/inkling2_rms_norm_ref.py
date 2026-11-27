"""Reference RMSNorm for the inkling2 family, without CuTe/CUTLASS.

Mirrors ``RMSNormFwd``'s math exactly: the row is scaled by ``pre_weight``
first (when the layer is pre-affine), ``rstd = rsqrt(mean(x^2) + eps)`` is
taken on the scaled row, and ``weight`` is applied after the normalization.
Used on devices whose CUTE kernels cannot compile (pre-sm90).
"""

import torch


def rms_norm_ref(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    eps: float,
    *,
    pre_weight: torch.Tensor | None = None,
    output_dtype: torch.dtype | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    shape = x.shape
    y = x.reshape(-1, shape[-1]).float()
    if pre_weight is not None:
        y = y * pre_weight.float()
    rstd = torch.rsqrt(y.square().mean(-1, keepdim=True) + eps)
    y = y * rstd
    if weight is not None:
        y = y * weight.float()
    y = y.view(shape).to(output_dtype or x.dtype)
    if out is not None:
        out.copy_(y)
        return out
    return y
