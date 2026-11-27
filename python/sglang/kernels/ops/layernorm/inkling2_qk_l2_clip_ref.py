# type: ignore

"""Torch reference for Inkling-2 QK/relative L2 clipping.

The CuTe-DSL kernel in :mod:`inkling2_qk_l2_clip` needs Hopper or newer, so
pre-Hopper devices use this implementation instead. It follows the same math:
``coeff = min(1, threshold / (||x||_2 + eps))`` applied in FP32, with an
optional per-row output scale multiplied after the cast to the output dtype.
"""

import torch


def qk_l2_clip_fwd_ref(
    x: torch.Tensor,
    threshold: float,
    eps: float = 1e-6,
    *,
    output_dtype: torch.dtype | None = None,
    scale: torch.Tensor | None = None,
    output_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    output_dtype = output_dtype or x.dtype
    xf = x.float()
    if scale is not None:
        if scale.ndim == 1:
            xf = xf * scale.float()
        else:
            heads, dim = scale.shape
            xf = xf.reshape(-1, heads, xf.shape[-1]) * scale.float().view(1, heads, dim)
            xf = xf.reshape(x.shape)
    sum_sq = xf.pow(2).sum(-1, keepdim=True)
    norm = torch.where(sum_sq > 0, torch.sqrt(sum_sq), torch.zeros_like(sum_sq))
    coeff = torch.clamp(threshold / (norm + eps), max=1.0)
    y = xf * coeff
    if output_scale is not None:
        rows = y.numel() // y.shape[-1]
        y = y.reshape(rows, -1).to(output_dtype).float()
        y = (
            y
            * output_scale.float().repeat_interleave(rows // output_scale.numel())[
                :, None
            ]
        )
        y = y.reshape(x.shape)
    return y.to(output_dtype)
