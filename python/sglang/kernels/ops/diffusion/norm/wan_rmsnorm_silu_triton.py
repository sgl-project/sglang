# SPDX-License-Identifier: Apache-2.0
"""Channels-last-3d Wan VAE RMSNorm(+SiLU) Triton kernel.

Fuses the Wan VAE ``WanRMS_norm -> SiLU`` chain
(``SiLU(F.normalize(x, dim=1) * scale * gamma + bias)`` on channel-first 5D
activations) into one kernel for ``channels_last_3d`` tensors. Each program
handles a ``[ROWS, C]`` tile of consecutive (b, t, h, w) pixels: the channel
rows are contiguous in memory, so loads and stores stay fully coalesced, and
tiling several pixels per program keeps every thread busy at the decoder's
small channel counts (96 - 384). A one-pixel-per-program layout ran at
0.63 TB/s on H200 for ``[1, 96, 4, 480, 832]`` bf16; this one runs at ~3 TB/s.

``conv_bias`` folds the preceding convolution's bias into the same pass:
PyTorch's cuDNN conv adds its bias as a separate ``add_`` kernel, so the
caller can run the conv without bias and hand it here. It is applied as
``x.dtype(float(x) + float(conv_bias))`` before the statistics, the exact
arithmetic of aten's ``add_``, so the norm sees the same values it would
have read from the biased tensor.

Numerics contract: fp32 channel-norm statistics, every step materialized at
the same dtype boundary as eager ``WanRMS_norm.forward`` (including the
aten promotion to fp32 at ``* gamma`` for half-precision x with fp32 affine
params -- the autocast case), SiLU in fp32. Bitwise equality with aten is
still not guaranteed (different reduction and SiLU paths), so callers must
keep this behind an opt-in gate.  Support is a predicate
(``can_use_wan_rmsnorm_silu``); the kernel raises on an unsupported input.
"""

from __future__ import annotations

import torch
import triton  # type: ignore
import triton.language as tl  # type: ignore

from sglang.srt.utils.custom_op import register_custom_op

_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}
_MAX_CHANNELS = 1024


@triton.jit
def _wan_rmsnorm_silu_kernel(
    x_ptr,
    gamma_ptr,
    bias_ptr,
    out_ptr,
    conv_bias_ptr,
    n_rows,
    channels: tl.constexpr,
    rms_scale,
    eps,
    has_bias: tl.constexpr,
    has_conv_bias: tl.constexpr,
    rows: tl.constexpr,
    block_c: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    row_ids = pid * rows + tl.arange(0, rows).to(tl.int64)
    col_ids = tl.arange(0, block_c)
    col_mask = col_ids < channels
    mask = (row_ids[:, None] < n_rows) & col_mask[None, :]
    # Dense channels-last-3d stores each pixel as one contiguous channel row.
    # Address it directly instead of recovering b/t/h/w with integer div/mod.
    offsets = row_ids[:, None] * channels + col_ids[None, :]

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    if has_conv_bias:
        # The conv's own bias add (aten add_): fp32 add, one rounding to x.dtype.
        conv_bias = tl.load(conv_bias_ptr + col_ids, mask=col_mask, other=0.0)
        x = (x + conv_bias.to(tl.float32)[None, :]).to(x_ptr.dtype.element_ty)
        x = x.to(tl.float32)
    norm = tl.sqrt(tl.sum(x * x, axis=1))
    inv_norm = 1.0 / tl.maximum(norm, eps)

    # Eager op boundaries: normalize/*scale in x.dtype; *gamma/+bias in the
    # promoted output dtype; SiLU in fp32, stored in the output dtype.
    y = (x * inv_norm[:, None]).to(x_ptr.dtype.element_ty)
    gamma = tl.load(gamma_ptr + col_ids, mask=col_mask, other=1.0)
    y = (y * rms_scale).to(x_ptr.dtype.element_ty)
    y = (y.to(tl.float32) * gamma.to(tl.float32)[None, :]).to(out_ptr.dtype.element_ty)
    if has_bias:
        bias = tl.load(bias_ptr + col_ids, mask=col_mask, other=0.0)
        y = (y.to(tl.float32) + bias.to(tl.float32)[None, :]).to(
            out_ptr.dtype.element_ty
        )
    y = y.to(tl.float32)
    y = y * tl.sigmoid(y)

    tl.store(out_ptr + offsets, y, mask=mask)


def _fake_wan_rmsnorm_silu(
    x: torch.Tensor,
    gamma: torch.Tensor,
    bias: torch.Tensor,
    conv_bias: torch.Tensor,
    rms_scale: float,
    eps: float,
    has_bias: bool,
    has_conv_bias: bool,
) -> torch.Tensor:
    dtype = torch.promote_types(x.dtype, gamma.dtype)
    return torch.empty_strided(x.shape, x.stride(), device=x.device, dtype=dtype)


@register_custom_op(
    op_name="triton_wan_rmsnorm_silu_cuda",
    fake_impl=_fake_wan_rmsnorm_silu,
)
def _triton_wan_rmsnorm_silu_cuda(
    x: torch.Tensor,
    gamma: torch.Tensor,
    bias: torch.Tensor,
    conv_bias: torch.Tensor,
    rms_scale: float,
    eps: float,
    has_bias: bool,
    has_conv_bias: bool,
) -> torch.Tensor:
    bsz, channels, t_size, h_size, w_size = x.shape
    n_rows = bsz * t_size * h_size * w_size
    # Preserve the input strides so the VAE keeps its channels_last_3d layout.
    dtype = torch.promote_types(x.dtype, gamma.dtype)
    out = torch.empty_strided(x.shape, x.stride(), device=x.device, dtype=dtype)
    block_c = triton.next_power_of_2(channels)
    # ~4K elements per program: 32 pixels at C <= 128 down to 4 at C = 1024.
    rows = max(4, min(32, 4096 // block_c))
    num_warps = 8 if rows * block_c >= 2048 else 4

    with torch.get_device_module().device(x.device):
        _wan_rmsnorm_silu_kernel[(triton.cdiv(n_rows, rows),)](
            x,
            gamma,
            bias,
            out,
            conv_bias,
            n_rows,
            channels,
            rms_scale,
            eps,
            has_bias,
            has_conv_bias,
            rows,
            block_c,
            num_warps=num_warps,
        )
    return out


def _affine_supported(x: torch.Tensor, t: torch.Tensor) -> bool:
    # Same dtype, or fp32 affine params on half-precision x (autocast case).
    return (
        t.is_cuda
        and t.device == x.device
        and (t.dtype == x.dtype or t.dtype == torch.float32)
        and t.numel() == x.shape[1]
    )


def can_use_wan_rmsnorm_silu(
    x: torch.Tensor,
    gamma: torch.Tensor,
    bias: torch.Tensor | None,
    conv_bias: torch.Tensor | None = None,
) -> bool:
    return (
        x.is_cuda
        and not torch.is_grad_enabled()
        and not x.requires_grad
        and x.dtype in _SUPPORTED_DTYPES
        and x.ndim == 5
        and x.numel() > 0
        and 0 < x.shape[1] <= _MAX_CHANNELS
        and x.is_contiguous(memory_format=torch.channels_last_3d)
        # Size-one channel tensors can satisfy the memory-format predicate
        # while retaining channel-first strides, so require dense rows too.
        and x.stride(1) == 1
        and _affine_supported(x, gamma)
        and (bias is None or _affine_supported(x, bias))
        and (conv_bias is None or _affine_supported(x, conv_bias))
    )


def wan_rmsnorm_silu(
    x: torch.Tensor,
    gamma: torch.Tensor,
    bias: torch.Tensor | None = None,
    rms_scale: float | None = None,
    eps: float = 1e-12,
    conv_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fused ``SiLU(F.normalize(x, dim=1) * rms_scale * gamma + bias)``.

    With ``conv_bias`` the input is first biased exactly like aten's conv bias
    ``add_`` (``x.dtype(x + conv_bias)``), so ``wan_rmsnorm_silu(conv_nobias,
    ..., conv_bias=b)`` equals ``wan_rmsnorm_silu(conv_nobias.add_(b), ...)``
    bit for bit. Guard with :func:`can_use_wan_rmsnorm_silu`.
    """
    if not can_use_wan_rmsnorm_silu(x, gamma, bias, conv_bias):
        raise ValueError("unsupported input for wan_rmsnorm_silu")

    channels = x.shape[1]
    gamma = gamma.reshape(channels).contiguous()
    has_bias = bias is not None
    bias = gamma if bias is None else bias.reshape(channels).contiguous()
    has_conv_bias = conv_bias is not None
    # autocast casts the conv bias to the activation dtype before the conv.
    conv_bias = (
        gamma
        if conv_bias is None
        else conv_bias.reshape(channels).to(x.dtype).contiguous()
    )
    if rms_scale is None:
        rms_scale = channels**0.5
    return _triton_wan_rmsnorm_silu_cuda(
        x, gamma, bias, conv_bias, float(rms_scale), eps, has_bias, has_conv_bias
    )


__all__ = ["can_use_wan_rmsnorm_silu", "wan_rmsnorm_silu"]
