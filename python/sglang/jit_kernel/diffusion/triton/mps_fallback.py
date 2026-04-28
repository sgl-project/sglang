"""MPS (Apple Silicon) fallbacks for Triton diffusion kernels.

Triton is not available on macOS / Metal, so these pure-PyTorch (and
optionally MLX-accelerated) implementations replace the Triton kernels
at import time when ``current_platform.is_mps()`` is True.

MLX acceleration (opt-in via ``SGLANG_USE_MLX=1``):
    Norm ops use ``mx.fast.rms_norm`` / ``mx.fast.layer_norm`` — single fused
    Metal kernels that are 1.4x–2.9x faster than the multi-step PyTorch MPS
    decomposition for medium-to-large tensors.
"""

from typing import Optional

import torch
from torch import Tensor

from sglang.srt.utils.tensor_bridge import mlx_to_torch, torch_to_mlx, use_mlx

from .torch_fallback import (
    apply_rotary_embedding_native,
    fuse_scale_shift_kernel_native,
    norm_infer_native,
    rms_norm_fn_native,
    triton_one_pass_rms_norm_native,
)

_use_mlx = use_mlx()

if _use_mlx:
    import mlx.core as mx

# use the common torch native version form torch_fallback
fuse_scale_shift_kernel_native = fuse_scale_shift_kernel_native
apply_rotary_embedding_native = apply_rotary_embedding_native
norm_infer_native = norm_infer_native
triton_one_pass_rms_norm_native = triton_one_pass_rms_norm_native
rms_norm_fn_native = rms_norm_fn_native

# MLX-accelerated norm ops (1.4x–2.9x faster than torch native on MPS)
# Uses mx.fast.rms_norm / mx.fast.layer_norm — single fused Metal kernels
# instead of 7+ separate PyTorch MPS kernel launches.

if _use_mlx:

    def norm_infer_native(  # noqa: F811
        x: Tensor,
        weight: Optional[Tensor],
        bias: Optional[Tensor],
        eps: float,
        is_rms_norm: bool = False,
        out: Optional[Tensor] = None,
    ) -> Tensor:
        """MLX-accelerated norm_infer (layer norm / rms norm inference)."""
        device = x.device
        orig_dtype = x.dtype
        x_mx = torch_to_mlx(x)
        if is_rms_norm:
            w_mx = (
                torch_to_mlx(weight) if weight is not None else mx.ones(x_mx.shape[-1])
            )
            result_mx = mx.fast.rms_norm(x_mx, w_mx, eps)
        else:
            w_mx = torch_to_mlx(weight) if weight is not None else None
            b_mx = torch_to_mlx(bias) if bias is not None else None
            result_mx = mx.fast.layer_norm(x_mx, w_mx, b_mx, eps)
        result = mlx_to_torch(result_mx, device).to(orig_dtype)
        if out is not None:
            out.copy_(result)
            return out
        return result

    def triton_one_pass_rms_norm_native(  # noqa: F811
        x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6
    ) -> torch.Tensor:
        """MLX-accelerated triton_one_pass_rms_norm."""
        device = x.device
        orig_dtype = x.dtype
        x_mx = torch_to_mlx(x)
        w_mx = torch_to_mlx(w)
        result_mx = mx.fast.rms_norm(x_mx, w_mx, eps)
        return mlx_to_torch(result_mx, device).to(orig_dtype)

    def rms_norm_fn_native(  # noqa: F811
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
        """MLX-accelerated rms_norm_fn (inference only, no dropout/x1 support)."""
        device = x.device
        orig_dtype = x.dtype
        if residual is not None:
            x = x.float() + residual.float()
            residual_out_val = x.to(torch.float32 if residual_in_fp32 else orig_dtype)
        else:
            residual_out_val = None
        if weight is not None and zero_centered_weight:
            w = weight.float() + 1.0
        else:
            w = weight
        x_mx = torch_to_mlx(x)
        w_mx = torch_to_mlx(w) if w is not None else mx.ones(x_mx.shape[-1])
        result_mx = mx.fast.rms_norm(x_mx, w_mx, eps)
        x_hat = mlx_to_torch(result_mx, device)
        if bias is not None:
            x_hat = x_hat + bias.to(x_hat.device, x_hat.dtype)
        final_dtype = out_dtype if out_dtype is not None else orig_dtype
        y = x_hat.to(final_dtype)
        if residual is not None and residual_out_val is not None:
            return y, residual_out_val
        return y
