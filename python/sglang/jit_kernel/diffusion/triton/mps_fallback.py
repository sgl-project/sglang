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

from sglang.srt.environ import envs

# MLX acceleration – opt-in via SGLANG_USE_MLX=1
_MLX_AVAILABLE = False
try:
    import mlx.core as mx

    _MLX_AVAILABLE = True
except ImportError:
    pass

_USE_MLX = envs.SGLANG_USE_MLX.get() and _MLX_AVAILABLE

# Dtype mapping for torch <-> MLX tensor bridge
_TORCH_TO_MLX_DTYPE = (
    {
        torch.float32: mx.float32,
        torch.float16: mx.float16,
        torch.bfloat16: mx.bfloat16,
    }
    if _MLX_AVAILABLE
    else {}
)

_MLX_TO_TORCH_DTYPE = {v: k for k, v in _TORCH_TO_MLX_DTYPE.items()}


def _torch_to_mlx(tensor: torch.Tensor) -> "mx.array":
    """Convert a PyTorch tensor to an MLX array (via numpy on CPU)."""
    t = tensor.cpu().detach()
    if t.dtype == torch.bfloat16:
        return mx.array(t.float().numpy(), dtype=mx.bfloat16)
    return mx.array(t.numpy())


def _mlx_to_torch(array: "mx.array", device: torch.device) -> torch.Tensor:
    """Convert an MLX array to a PyTorch tensor (zero-copy via memoryview)."""
    torch_dtype = _MLX_TO_TORCH_DTYPE.get(array.dtype, torch.float32)
    array = mx.contiguous(array)
    mx.eval(array)
    tensor = torch.frombuffer(memoryview(array), dtype=torch_dtype).reshape(array.shape)
    if device.type == "mps":
        tensor = tensor.to(device)
    return tensor


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
    orig_dtype = x.dtype
    x = x.contiguous().float()
    if is_rms_norm:
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x_hat = x * torch.rsqrt(variance + eps)
    else:
        mean = x.mean(dim=-1, keepdim=True)
        variance = (x - mean).pow(2).mean(dim=-1, keepdim=True)
        x_hat = (x - mean) * torch.rsqrt(variance + eps)
    if weight is not None:
        x_hat = x_hat * weight.float()
    if bias is not None:
        x_hat = x_hat + bias.float()
    result = x_hat.to(orig_dtype)
    if out is not None:
        out.copy_(result)
        return out
    return result


def triton_one_pass_rms_norm_native(
    x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """Native fallback for triton_one_pass_rms_norm."""
    shape = x.shape
    orig_dtype = x.dtype
    x = x.contiguous().float()
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x_hat = x * torch.rsqrt(variance + eps)
    return (x_hat * w.float()).to(orig_dtype).view(shape)


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
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x_hat = x * torch.rsqrt(variance + eps)
    if weight is not None:
        w = weight.float()
        if zero_centered_weight:
            w = w + 1.0
        x_hat = x_hat * w
    if bias is not None:
        x_hat = x_hat + bias.float()
    final_dtype = out_dtype if out_dtype is not None else orig_dtype
    y = x_hat.to(final_dtype).reshape(x_shape_og)
    if residual is not None and residual_out_val is not None:
        return y, residual_out_val.reshape(x_shape_og)
    return y


# MLX-accelerated norm ops (1.4x–2.9x faster than torch native on MPS)
# Uses mx.fast.rms_norm / mx.fast.layer_norm — single fused Metal kernels
# instead of 7+ separate PyTorch MPS kernel launches.

if _USE_MLX:

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
        x_mx = _torch_to_mlx(x)
        if is_rms_norm:
            w_mx = (
                _torch_to_mlx(weight) if weight is not None else mx.ones(x_mx.shape[-1])
            )
            result_mx = mx.fast.rms_norm(x_mx, w_mx, eps)
        else:
            w_mx = _torch_to_mlx(weight) if weight is not None else None
            b_mx = _torch_to_mlx(bias) if bias is not None else None
            result_mx = mx.fast.layer_norm(x_mx, w_mx, b_mx, eps)
        result = _mlx_to_torch(result_mx, device).to(orig_dtype)
        if out is not None:
            out.copy_(result)
            return out
        return result

    def triton_one_pass_rms_norm_native(  # noqa: F811
        x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6
    ) -> torch.Tensor:
        """MLX-accelerated triton_one_pass_rms_norm."""
        shape = x.shape
        device = x.device
        orig_dtype = x.dtype
        x_mx = _torch_to_mlx(x.reshape(-1, x.shape[-1]))
        w_mx = _torch_to_mlx(w)
        result_mx = mx.fast.rms_norm(x_mx, w_mx, eps)
        return _mlx_to_torch(result_mx, device).to(orig_dtype).view(shape)

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
        x_shape_og = x.shape
        device = x.device
        orig_dtype = x.dtype
        x_flat = x.reshape(-1, x.shape[-1])
        if residual is not None:
            residual = residual.reshape(-1, residual.shape[-1]).float()
            x_flat = x_flat.float() + residual
            residual_out_val = x_flat.to(
                torch.float32 if residual_in_fp32 else orig_dtype
            )
        else:
            residual_out_val = None
        if weight is not None and zero_centered_weight:
            w = weight.float() + 1.0
        else:
            w = weight
        x_mx = _torch_to_mlx(x_flat)
        w_mx = _torch_to_mlx(w) if w is not None else mx.ones(x_mx.shape[-1])
        result_mx = mx.fast.rms_norm(x_mx, w_mx, eps)
        x_hat = _mlx_to_torch(result_mx, device)
        if bias is not None:
            x_hat = x_hat + bias.to(x_hat.device, x_hat.dtype)
        final_dtype = out_dtype if out_dtype is not None else orig_dtype
        y = x_hat.to(final_dtype).reshape(x_shape_og)
        if residual is not None and residual_out_val is not None:
            return y, residual_out_val.reshape(x_shape_og)
        return y
