# SPDX-License-Identifier: Apache-2.0
"""Qwen-Image 2.1 VAE decoder fast paths.

Lossless, always on, verified bit-exact against the original module chain on
first sight and disabled on a mismatch:

- ``FusedChannelRMSNormSiLU`` keeps aten's fp32 channel L2 norm and runs the
  normalize / scale / gamma / SiLU tail as one CUDA launch wherever the decoder
  applies ``RMS_norm -> SiLU`` (residual blocks and the output head).
- ``FoldedPadConv2d`` folds the symmetric ``F.pad`` that precedes every causal
  conv into the convolution's own padding. cuDNN may pick another algorithm for
  the new descriptor, so the folded conv is compared per input signature and
  kept only where the result is identical.
- Each residual block's second conv runs without its bias and ``bias_residual_add``
  applies bias and residual in one pass with aten's per-op bf16 rounding, so the
  separate bias pass over the conv output disappears.

Quality-gated (``quality="extra-high"`` / ``"high"``, decode-scoped through
:class:`VaeFastPathGate`; changes rounding, so never on the lossless path):

- The decoder runs in ``channels_last`` so the cuDNN NHWC conv kernels no
  longer wrap every 3x3 conv in NCHW/NHWC transposes.
- ``FusedChannelRMSNormSiLU`` then reduces each pixel's channels in a lane
  group (``channel_rmsnorm_silu_nhwc``) instead of the NCHW aten reduction, and
  takes the first conv's bias as an input so that conv skips its bias pass too.
- ``ChannelsLastNearestUpsample`` re-expresses the merged-batch view with
  canonical NHWC strides and runs the bit-exact Triton nearest gather, which
  is several times faster than aten's NHWC nearest kernel.
- ``FusedUpsample2xConv`` replaces each upsampler's nearest 2x + conv3x3 with
  one ConvTranspose2d(k4, s2, p1) whose kernel is the fp32 sum of the taps each
  output phase touches: 2.25x fewer MACs and no 4x-sized intermediate.

Parameter names are preserved (``...norm1.gamma``, ``...conv1.weight``), so
checkpoint loading and weight transfer are unaffected.
"""

from __future__ import annotations

from types import MethodType

import torch
import torch.nn.functional as F
from torch import nn

from sglang.kernels.ops.diffusion import (
    BitExactFusionGate,
    bias_residual_add,
    can_use_bias_residual_add,
    can_use_channel_rmsnorm_finish_silu,
    can_use_channel_rmsnorm_silu_nhwc,
    can_use_nearest_upsample_nhwc,
    channel_rmsnorm_finish_silu,
    channel_rmsnorm_silu_nhwc,
    nearest_upsample_nhwc,
)
from sglang.multimodal_gen.runtime.models.vaes.conv_fold import (
    fold_upsample2x_conv2d_weight,
)
from sglang.multimodal_gen.runtime.models.vaes.fast_path_gate import (
    VaeFastPathGate,
    register_vae_fast_path_gate,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)
_BIAS_RESIDUAL_FUSION = BitExactFusionGate(
    "Qwen-Image 2.1 VAE conv bias + residual add"
)


class FusedChannelRMSNormSiLU(nn.Module):
    """``SiLU(RMS_norm(x))``: fused NHWC kernel under the quality gate, fused lossless tail otherwise."""

    def __init__(self, norm: nn.Module, gate: VaeFastPathGate) -> None:
        super().__init__()
        # Unregistered: it owns the exact off-path chain. ``gamma`` is re-registered
        # here so the state_dict key stays ``...norm1.gamma``.
        object.__setattr__(self, "_norm", norm)
        self.gamma = norm.gamma
        self.scale = float(norm.scale)
        self._sgl_gate = gate
        self._exact_gate = BitExactFusionGate(
            "Qwen-Image 2.1 VAE channel RMSNorm + SiLU"
        )

    def forward(
        self, x: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor:
        """``bias`` is the preceding conv's bias when that conv ran without it."""
        norm = self._norm
        plain = norm.channel_first and isinstance(norm.bias, float) and norm.bias == 0.0
        if (
            plain
            and self._sgl_gate.enabled
            and can_use_channel_rmsnorm_silu_nhwc(x, self.gamma, bias)
        ):
            return channel_rmsnorm_silu_nhwc(x, self.gamma, self.scale, bias)
        if bias is not None:
            # the conv's own bias pass, same aten rounding
            x = x + bias
        fused = None
        if (
            plain
            and can_use_channel_rmsnorm_finish_silu(x, self.gamma)
            and self._exact_gate.can_attempt_once()
        ):
            channel_norm = x.float().norm(p=2, dim=1, keepdim=True)
            fused = channel_rmsnorm_finish_silu(x, channel_norm, self.gamma, self.scale)
            if self._exact_gate.verified:
                return fused
        out = F.silu(norm(x))
        if fused is not None:
            return self._exact_gate.accept_or_fallback(fused, out, logger=logger)
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
        # cuDNN's algorithm choice depends on the layout too, so NCHW and NHWC
        # inputs verify against separate gates.
        self._gates = {
            "nchw": BitExactFusionGate(
                "Qwen-Image 2.1 VAE conv padding fold (NCHW)", per_signature=True
            ),
            "nhwc": BitExactFusionGate(
                "Qwen-Image 2.1 VAE conv padding fold (NHWC)", per_signature=True
            ),
        }

    def forward(self, x: torch.Tensor, cache_x=None) -> torch.Tensor:
        return self._run(x, cache_x, with_bias=True)

    def forward_no_bias(self, x: torch.Tensor) -> torch.Tensor:
        """The conv without its bias; the caller adds the bias in a later fused op."""
        return self._run(x, None, with_bias=False)

    def _reference(self, x: torch.Tensor, cache_x, with_bias: bool) -> torch.Tensor:
        conv = self._conv
        if with_bias:
            return conv(x, cache_x)
        padded = F.pad(x.squeeze(2), list(conv._padding))
        return F.conv2d(
            padded, self.weight, None, conv.stride, 0, conv.dilation, conv.groups
        ).unsqueeze(2)

    def _run(self, x: torch.Tensor, cache_x, with_bias: bool) -> torch.Tensor:
        conv = self._conv
        if not (
            self._symmetric
            and cache_x is None
            and x.is_cuda
            and not torch.compiler.is_compiling()
        ):
            return self._reference(x, cache_x, with_bias)
        gate = self._gates["nhwc" if x.stride(1) == 1 and x.shape[1] > 1 else "nchw"]
        if gate.disabled:
            return self._reference(x, cache_x, with_bias)
        sig = (x.dtype, tuple(x.shape), tuple(x.stride()), with_bias)
        verified = gate.is_verified(sig)
        if not verified and torch.cuda.is_current_stream_capturing():
            return self._reference(x, cache_x, with_bias)
        folded = F.conv2d(
            x.squeeze(2),
            self.weight,
            self.bias if with_bias else None,
            conv.stride,
            self._padding,
            conv.dilation,
            conv.groups,
        ).unsqueeze(2)
        if verified:
            return folded
        return gate.accept_or_fallback(
            folded, self._reference(x, cache_x, with_bias), sig=sig, logger=logger
        )


class ChannelsLastNearestUpsample(nn.Module):
    """``Resample``'s nearest 2x upsample, kept channels_last and run as the Triton NHWC gather."""

    def __init__(self, upsample: nn.Upsample, gate: VaeFastPathGate) -> None:
        super().__init__()
        self._sgl_upsample = upsample
        self._sgl_gate = gate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        up = self._sgl_upsample
        if torch.compiler.is_compiling() or x.dim() != 4:
            return up(x)
        _, c, h, w = x.shape
        canonical = (h * w * c, 1, w * c, c)
        # Resample merges batch and time into a size-1 dim whose stride is
        # arbitrary; re-labelling it is a pure view of the same elements.
        if (
            self._sgl_gate.enabled
            and x.stride() != canonical
            and x.is_contiguous(memory_format=torch.channels_last)
        ):
            x = x.as_strided(x.shape, canonical)
        if up.size is None and can_use_nearest_upsample_nhwc(
            x, up.scale_factor, up.mode
        ):
            return nearest_upsample_nhwc(x, up.scale_factor)
        return up(x)


class FusedUpsample2xConv(nn.Sequential):
    """``Resample.resample`` (nearest 2x, conv3x3 p1) as one ConvTranspose2d(k4, s2, p1) under the quality gate.

    The children keep their indices (``0`` upsample, ``1`` conv), so parameter
    names are unchanged. Gate off runs the original two-module chain.
    """

    def __init__(self, resample: nn.Sequential, gate: VaeFastPathGate) -> None:
        super().__init__(*resample.children())
        self._sgl_gate = gate
        self._sgl_folded = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._sgl_gate.enabled or x.dim() != 4 or torch.compiler.is_compiling():
            return super().forward(x)
        conv = self[1]
        folded = self._sgl_folded
        if (
            folded is None
            or folded.dtype is not conv.weight.dtype
            or folded.device != conv.weight.device
        ):
            folded = fold_upsample2x_conv2d_weight(conv)
            self._sgl_folded = folded
        return F.conv_transpose2d(x, folded, conv.bias, stride=2, padding=1)


def _foldable_resample(module, upsample_cls) -> bool:
    chain = module.resample
    if module.mode not in ("upsample2d", "upsample3d"):
        return False
    if type(chain) is not nn.Sequential or len(chain) != 2:
        return False
    up, conv = chain[0], chain[1]
    if type(up) is ChannelsLastNearestUpsample:
        up = up._sgl_upsample
    return (
        type(up) is upsample_cls
        and up.size is None
        and up.mode == "nearest-exact"
        and tuple(up.scale_factor) == (2.0, 2.0)
        and type(conv) is nn.Conv2d
        and conv.kernel_size == (3, 3)
        and conv.stride == (1, 1)
        and conv.padding == (1, 1)
        and conv.dilation == (1, 1)
        and conv.groups == 1
        and conv.padding_mode == "zeros"
    )


def _conv_bias_residual_add(conv, x, h):
    """``conv(x) + h`` with the conv's bias applied inside the residual add when the fused kernel fits."""
    if not (
        type(conv) is FoldedPadConv2d
        and conv.bias is not None
        and _BIAS_RESIDUAL_FUSION.can_attempt_once()
    ):
        return conv(x) + h
    y = conv.forward_no_bias(x)
    if not can_use_bias_residual_add(y, conv.bias, h):
        return (y + conv.bias.view(1, -1, *([1] * (y.ndim - 2)))) + h
    fused = bias_residual_add(y, conv.bias, h)
    if _BIAS_RESIDUAL_FUSION.verified:
        return fused
    return _BIAS_RESIDUAL_FUSION.accept_or_fallback(fused, conv(x) + h, logger=logger)


def _residual_block_forward(self, x, feat_cache=None, feat_idx=None):
    """``QwenImage21ResidualBlock.forward`` with the two conv bias passes fused away."""
    h = self.conv_shortcut(x)
    y = self.nonlinearity(self.norm1(x))
    conv1, norm2 = self.conv1, self.norm2
    if (
        type(conv1) is FoldedPadConv2d
        and conv1.bias is not None
        and type(norm2) is FusedChannelRMSNormSiLU
        and norm2._sgl_gate.enabled
    ):
        y = norm2(conv1.forward_no_bias(y), conv1.bias.view(norm2.gamma.shape))
    else:
        y = norm2(conv1(y))
    y = self.dropout(self.nonlinearity(y))
    return _conv_bias_residual_add(self.conv2, y, h)


def _decoder_layout_forward(self, x, *args, **kwargs):
    want_channels_last = self._sgl_gate.enabled
    if want_channels_last != self._sgl_channels_last:
        # Layout swaps permute parameter memory only; flipping back restores the
        # NCHW cuDNN kernel selection of the lossless path exactly.
        self.to(
            memory_format=(
                torch.channels_last if want_channels_last else torch.contiguous_format
            )
        )
        self._sgl_channels_last = want_channels_last
    if want_channels_last and x.dim() == 5:
        x = x.contiguous(memory_format=torch.channels_last_3d)
    return type(self).forward(self, x, *args, **kwargs)


def _install_norm_silu(decoder, gate, residual_block_cls, rms_norm_cls) -> int:
    count = 0
    for module in decoder.modules():
        if (
            type(module) is residual_block_cls
            and type(module.nonlinearity) is nn.SiLU
            and not module.nonlinearity.inplace
            and type(module.norm1) is rms_norm_cls
            and type(module.norm2) is rms_norm_cls
        ):
            module.norm1 = FusedChannelRMSNormSiLU(module.norm1, gate)
            module.norm2 = FusedChannelRMSNormSiLU(module.norm2, gate)
            module.nonlinearity = nn.Identity()
            count += 2
    if (
        type(decoder.norm_out) is rms_norm_cls
        and type(decoder.nonlinearity) is nn.SiLU
        and not decoder.nonlinearity.inplace
    ):
        decoder.norm_out = FusedChannelRMSNormSiLU(decoder.norm_out, gate)
        decoder.nonlinearity = nn.Identity()
        count += 1
    return count


def _replace_children(root: nn.Module, child_cls: type, make) -> int:
    targets = [
        (parent, name)
        for parent in root.modules()
        for name, child in parent.named_children()
        if type(child) is child_cls
    ]
    for parent, name in targets:
        setattr(parent, name, make(getattr(parent, name)))
    return len(targets)


def maybe_optimize_qwen_image21_vae(vae: nn.Module) -> nn.Module:
    """Install the Qwen-Image 2.1 decoder fast paths; other VAEs pass through."""
    from sglang.multimodal_gen.runtime.models.vaes.autoencoder_kl_qwenimage21 import (
        AutoencoderKLQwenImage21,
        QwenImage21CausalConv3d,
        QwenImage21Decoder3d,
        QwenImage21Resample,
        QwenImage21ResidualBlock,
        QwenImage21RMS_norm,
        QwenImage21Upsample,
    )

    if not isinstance(vae, AutoencoderKLQwenImage21):
        return vae
    if not torch.cuda.is_available() or torch.version.hip is not None:
        return vae
    decoder = vae.decoder if vae.config.load_decoder else None
    if type(decoder) is not QwenImage21Decoder3d or vae.spatial_parallel:
        return vae
    gate = VaeFastPathGate()
    n_norm = _install_norm_silu(
        decoder, gate, QwenImage21ResidualBlock, QwenImage21RMS_norm
    )
    n_conv = _replace_children(decoder, QwenImage21CausalConv3d, FoldedPadConv2d)
    for module in decoder.modules():
        if type(module) is QwenImage21ResidualBlock:
            module.forward = MethodType(_residual_block_forward, module)
    if type(vae.post_quant_conv) is QwenImage21CausalConv3d:
        vae.post_quant_conv = FoldedPadConv2d(vae.post_quant_conv)
        n_conv += 1
    n_up = _replace_children(
        decoder, QwenImage21Upsample, lambda up: ChannelsLastNearestUpsample(up, gate)
    )
    n_fold = 0
    for module in decoder.modules():
        if type(module) is QwenImage21Resample and _foldable_resample(
            module, QwenImage21Upsample
        ):
            module.resample = FusedUpsample2xConv(module.resample, gate)
            n_fold += 1
    decoder._sgl_gate = gate
    decoder._sgl_channels_last = False
    decoder.forward = MethodType(_decoder_layout_forward, decoder)
    register_vae_fast_path_gate(vae, gate)
    logger.info(
        "Qwen-Image 2.1 VAE: %d fused RMSNorm+SiLU sites, %d convs with folded padding, "
        "%d channels_last upsamplers, %d folded upsample+conv pairs "
        "(channels_last decode and the fold at quality extra-high/high).",
        n_norm,
        n_conv,
        n_up,
        n_fold,
    )
    return vae


__all__ = [
    "ChannelsLastNearestUpsample",
    "FoldedPadConv2d",
    "FusedUpsample2xConv",
    "FusedChannelRMSNormSiLU",
    "maybe_optimize_qwen_image21_vae",
]
