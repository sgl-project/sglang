# SPDX-License-Identifier: Apache-2.0
"""Qwen-Image 2.1 VAE decoder fast paths (lossless, always on).

Both paths are verified bit-exact against the original module chain on first
sight and disable themselves on a mismatch:

- ``FusedChannelRMSNormSiLU`` keeps aten's fp32 channel L2 norm and runs the
  normalize / scale / gamma / SiLU tail as one CUDA launch wherever the decoder
  applies ``RMS_norm -> SiLU`` (residual blocks and the output head).
- ``FoldedPadConv2d`` folds the symmetric ``F.pad`` that precedes every causal
  3x3 conv into the convolution's own padding. cuDNN may pick another algorithm
  for the new descriptor, so the folded conv is compared per input shape and
  kept only where the result is identical.

Parameter names are preserved (``...norm1.gamma``, ``...conv1.weight``), so
checkpoint loading and weight transfer are unaffected.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from sglang.kernels.ops.diffusion import (
    BitExactFusionGate,
    can_use_channel_rmsnorm_finish_silu,
    channel_rmsnorm_finish_silu,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class FusedChannelRMSNormSiLU(nn.Module):
    """``SiLU(RMS_norm(x))`` with the pointwise tail fused; the eager chain stays the fallback."""

    def __init__(self, norm: nn.Module) -> None:
        super().__init__()
        # Unregistered: it owns the exact off-path chain. ``gamma`` is re-registered
        # here so the state_dict key stays ``...norm1.gamma``.
        object.__setattr__(self, "_norm", norm)
        self.gamma = norm.gamma
        self.scale = float(norm.scale)
        self._gate = BitExactFusionGate("Qwen-Image 2.1 VAE channel RMSNorm + SiLU")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = self._norm
        fused = None
        if (
            norm.channel_first
            and isinstance(norm.bias, float)
            and norm.bias == 0.0
            and can_use_channel_rmsnorm_finish_silu(x, self.gamma)
            and self._gate.can_attempt_once()
        ):
            channel_norm = x.float().norm(p=2, dim=1, keepdim=True)
            fused = channel_rmsnorm_finish_silu(x, channel_norm, self.gamma, self.scale)
            if self._gate.verified:
                return fused
        out = F.silu(norm(x))
        if fused is not None:
            return self._gate.accept_or_fallback(fused, out, logger=logger)
        return out


class FoldedPadConv2d(nn.Module):
    """A causal conv whose explicit symmetric ``F.pad`` is folded into ``F.conv2d``'s padding."""

    def __init__(self, conv: nn.Module) -> None:
        super().__init__()
        object.__setattr__(self, "_conv", conv)
        self.weight = conv.weight
        self.bias = conv.bias
        left, right, top, bottom = conv._padding
        self._symmetric = left == right and top == bottom
        self._padding = (top, left)
        self._gate = BitExactFusionGate(
            "Qwen-Image 2.1 VAE conv padding fold", per_signature=True
        )

    def forward(self, x: torch.Tensor, cache_x=None) -> torch.Tensor:
        conv = self._conv
        if not (
            self._symmetric
            and cache_x is None
            and x.is_cuda
            and not self._gate.disabled
            and not torch.compiler.is_compiling()
        ):
            return conv(x, cache_x)
        sig = (x.dtype, tuple(x.shape))
        verified = self._gate.is_verified(sig)
        if not verified and torch.cuda.is_current_stream_capturing():
            return conv(x, cache_x)
        folded = F.conv2d(
            x.squeeze(2),
            self.weight,
            self.bias,
            conv.stride,
            self._padding,
            conv.dilation,
            conv.groups,
        ).unsqueeze(2)
        if verified:
            return folded
        return self._gate.accept_or_fallback(folded, conv(x, cache_x), sig=sig, logger=logger)


def _install_norm_silu(decoder: nn.Module, residual_block_cls: type, rms_norm_cls: type) -> int:
    count = 0
    for module in decoder.modules():
        if (
            type(module) is residual_block_cls
            and type(module.nonlinearity) is nn.SiLU
            and not module.nonlinearity.inplace
            and type(module.norm1) is rms_norm_cls
            and type(module.norm2) is rms_norm_cls
        ):
            module.norm1 = FusedChannelRMSNormSiLU(module.norm1)
            module.norm2 = FusedChannelRMSNormSiLU(module.norm2)
            module.nonlinearity = nn.Identity()
            count += 2
    if (
        type(decoder.norm_out) is rms_norm_cls
        and type(decoder.nonlinearity) is nn.SiLU
        and not decoder.nonlinearity.inplace
    ):
        decoder.norm_out = FusedChannelRMSNormSiLU(decoder.norm_out)
        decoder.nonlinearity = nn.Identity()
        count += 1
    return count


def _install_folded_convs(root: nn.Module, conv_cls: type) -> int:
    targets = [
        (parent, name)
        for parent in root.modules()
        for name, child in parent.named_children()
        if type(child) is conv_cls
    ]
    for parent, name in targets:
        setattr(parent, name, FoldedPadConv2d(getattr(parent, name)))
    return len(targets)


def maybe_optimize_qwen_image21_vae(vae: nn.Module) -> nn.Module:
    """Install the lossless Qwen-Image 2.1 decoder fast paths; other VAEs pass through."""
    from sglang.multimodal_gen.runtime.models.vaes.autoencoder_kl_qwenimage21 import (
        AutoencoderKLQwenImage21,
        QwenImage21CausalConv3d,
        QwenImage21Decoder3d,
        QwenImage21ResidualBlock,
        QwenImage21RMS_norm,
    )

    if not isinstance(vae, AutoencoderKLQwenImage21):
        return vae
    if not torch.cuda.is_available() or torch.version.hip is not None:
        return vae
    decoder = vae.decoder if vae.config.load_decoder else None
    if type(decoder) is not QwenImage21Decoder3d or vae.spatial_parallel:
        return vae
    n_norm = _install_norm_silu(decoder, QwenImage21ResidualBlock, QwenImage21RMS_norm)
    n_conv = _install_folded_convs(decoder, QwenImage21CausalConv3d)
    if type(vae.post_quant_conv) is QwenImage21CausalConv3d:
        vae.post_quant_conv = FoldedPadConv2d(vae.post_quant_conv)
        n_conv += 1
    logger.info(
        "Qwen-Image 2.1 VAE: %d fused RMSNorm+SiLU sites, %d convs with folded padding.",
        n_norm,
        n_conv,
    )
    return vae


__all__ = ["FoldedPadConv2d", "FusedChannelRMSNormSiLU", "maybe_optimize_qwen_image21_vae"]
