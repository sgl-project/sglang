"""Pytorch native based fallbacks for Triton diffusion kernels.

Triton is not available on some platforms, so these pure-PyTorch
implementations replace the Triton kernels

"""

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor


def fuse_scale_shift_kernel_native(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    scale_constant: float = 1.0,
    block_l: int = 128,
    block_c: int = 128,
):
    """Native fallback for fuse_scale_shift_kernel with scale_constant support."""
    B, L, C = x.shape

    def _expand(t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 4:
            # [B, F, 1, C] -> [B, L, C]
            num_frames = t.shape[1]
            frame_seqlen = L // num_frames
            return (
                t.squeeze(2)
                .unsqueeze(2)
                .expand(-1, -1, frame_seqlen, -1)
                .reshape(B, L, C)
            )
        elif t.dim() == 2:
            # [B, C] -> [B, 1, C]
            return t.unsqueeze(1)
        return t

    scale = _expand(scale)
    shift = _expand(shift)

    return x * (scale_constant + scale) + shift


def apply_rotary_embedding_native(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, interleaved: bool = False
) -> torch.Tensor:
    """Native fallback for rotary embedding (shared with NPU implementation)."""
    if interleaved and cos.shape[-1] == x.shape[-1]:
        cos = cos[..., ::2]
        sin = sin[..., ::2]
    cos = cos.unsqueeze(-2).to(x.dtype)
    sin = sin.unsqueeze(-2).to(x.dtype)
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    return torch.stack((o1, o2), dim=-1).flatten(-2)


def norm_infer_native(
    x: Tensor,
    weight: Optional[Tensor],
    bias: Optional[Tensor],
    eps: float,
    is_rms_norm: bool = False,
    out: Optional[Tensor] = None,
) -> Tensor:
    """Native fallback for norm_infer (layer norm / rms norm inference)."""
    normalized_shape = (x.shape[-1],)
    if is_rms_norm:
        # ``F.rms_norm`` returns in the input dtype before a separately added
        # bias is applied. Promote the whole branch when a bias is present or
        # parameters have a different dtype, retaining the old
        # fp32-accumulate-then-cast contract. A bias triggers promotion even
        # when its dtype matches the input; otherwise the normalized value is
        # rounded before the affine add. Keep the native fast path for the
        # common bias-free, same-dtype case.
        needs_fp32 = bias is not None or (
            weight is not None and weight.dtype != x.dtype
        )
        if needs_fp32:
            result = F.rms_norm(
                x.float(),
                normalized_shape,
                weight.float() if weight is not None else None,
                eps,
            )
            if bias is not None:
                result = result + bias.float()
        else:
            result = F.rms_norm(x, normalized_shape, weight, eps)
            if bias is not None:
                result = result + bias
    else:
        result = F.layer_norm(x, normalized_shape, weight, bias, eps)
    # Match the original fallback and Triton kernel contract even when a
    # higher-precision weight or bias promotes PyTorch's intermediate result.
    result = result.to(x.dtype)
    if out is not None:
        out.copy_(result)
        return out
    return result


def triton_one_pass_rms_norm_native(
    x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """Native fallback for triton_one_pass_rms_norm."""
    return F.rms_norm(x, (x.shape[-1],), w, eps)


def rms_norm_fn_native(
    x,
    weight,
    bias,
    residual=None,
    x1=None,
    weight1=None,
    bias1=None,
    eps=1e-6,
    dropout_p=0.0,
    rowscale=None,
    prenorm=False,
    residual_in_fp32=False,
    zero_centered_weight=False,
    return_dropout_mask=False,
    out_dtype=None,
    out=None,
    residual_out=None,
):
    """Native fallback for rms_norm_fn (inference only, no dropout/x1 support)."""
    x_shape_og = x.shape
    orig_dtype = x.dtype
    x = x.reshape(-1, x.shape[-1]).float()
    if residual is not None:
        residual = residual.reshape(-1, residual.shape[-1]).float()
        x = x + residual
        residual_out_val = x.to(torch.float32 if residual_in_fp32 else orig_dtype)
    else:
        residual_out_val = None
    if weight is not None:
        w = weight.float()
        if zero_centered_weight:
            w = w + 1.0
    else:
        w = None
    x_hat = F.rms_norm(x, (x.shape[-1],), w, eps)
    if bias is not None:
        x_hat = x_hat + bias.float()
    final_dtype = out_dtype if out_dtype is not None else orig_dtype
    y = x_hat.to(final_dtype).reshape(x_shape_og)
    if residual is not None and residual_out_val is not None:
        return y, residual_out_val.reshape(x_shape_og)
    return y
