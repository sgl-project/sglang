# SPDX-License-Identifier: Apache-2.0
"""CUDA fast path for the Wan-family VAE decoders (AutoencoderKLWan and the
Qwen-Image VAE, which is the Wan 2.1 VAE under other class names).

Fuses every decoder ``WanRMS_norm -> SiLU`` chain into one Triton kernel on
the channels_last_3d layout. Wrappers are installed once at VAE load and
dispatch on a decode-scoped :class:`VaeFastPathGate`: ``quality="extra-high"``
and ``quality="high"`` run the fused kernel (not bitwise-identical to aten,
hence gated), while the ``"lossless"`` default runs the original module path
bit-for-bit. Install is all-or-nothing and fail-closed.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from sglang.multimodal_gen.runtime.models.vaes.fast_path_gate import (
    VaeFastPathGate,
    register_vae_fast_path_gate,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

try:
    from sglang.kernels.ops.diffusion import (
        can_use_wan_rmsnorm_silu,
        wan_rmsnorm_silu,
    )

    _HAS_TRITON = True
except ImportError:  # pragma: no cover
    _HAS_TRITON = False


class FusedWanRMSNormSiLU(nn.Module):
    """``WanRMS_norm`` + SiLU fused via the channels_last_3d Triton kernel;
    falls back to the original op chain (bit-identical to norm + ``nn.SiLU``)
    for unsupported inputs and whenever the gate is off. Steps aside under
    ``torch.compile``, where Inductor already fuses this chain."""

    def __init__(self, norm: nn.Module, gate: VaeFastPathGate) -> None:
        super().__init__()
        # Keep the norm's parameters registered directly on the wrapper so
        # parameter names stay `...norm1.gamma` (weight transfer and
        # state_dict load match by name).
        self.gamma = norm.gamma
        self.bias = norm.bias
        self.scale = float(norm.scale)
        self._sgl_gate = gate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._sgl_gate.enabled and not torch.compiler.is_compiling():
            bias = self.bias if isinstance(self.bias, torch.Tensor) else None
            if can_use_wan_rmsnorm_silu(x, self.gamma, bias):
                return wan_rmsnorm_silu(x, self.gamma, bias, rms_scale=self.scale)
        # WanRMS_norm.forward (channel-first) + SiLU, same ops in the same
        # order, so the off-path stays bit-identical.
        return F.silu(F.normalize(x, dim=1) * self.scale * self.gamma + self.bias)


class GatedChannelsLastUpsample(nn.Module):
    """Wan-style ``Resample`` upsample that keeps the decoder channels_last.

    ``Resample.forward`` reaches its 2D ``Upsample -> Conv2d`` through
    ``x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)``. On a channels_last_3d
    tensor with ``b * t == 1`` that view keeps a degenerate batch stride which
    ``is_contiguous(channels_last)`` accepts but aten's
    ``suggest_memory_format`` does not, so the upsample and the conv2d run
    their NCHW kernels (cuDNN converts back and forth internally) and every
    up block starts with an NCDHW tensor that the fused RMSNorm+SiLU cannot
    take. With the gate on, re-express the same memory with canonical NHWC
    strides (a pure view) so the conv2d runs channels_last end-to-end. The
    NHWC conv2d is not guaranteed to pick the same cuDNN algorithm as the NCHW
    one, so the stride canonicalisation stays behind the quality gate; with
    the gate off the layout is left exactly as the eager module sees it.
    """

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
        # ``is_contiguous(channels_last)`` guarantees every dim of size > 1
        # already has its canonical stride; only size-1 dims (the merged
        # batch dim here) can carry a stray one, and their stride is never
        # used to address memory. So ``as_strided`` to the canonical strides
        # is a pure re-labelling of the same elements, for any batch size.
        if (
            self._sgl_gate.enabled
            and x.stride() != canonical
            and x.is_contiguous(memory_format=torch.channels_last)
        ):
            x = x.as_strided(x.shape, canonical)
        return up(x)


def _install_module_gates(
    decoder: nn.Module, gate: VaeFastPathGate, module_classes: tuple[type, ...]
) -> int:
    """Hand the gate to layout-sensitive modules that consult ``_sgl_gate`` in
    their own forward (the attention block's residual operand order, so the
    sum keeps the channels_last_3d layout)."""
    count = 0
    for m in decoder.modules():
        if type(m) in module_classes:
            m._sgl_gate = gate
            count += 1
    return count


def _is_plain_silu(act: object) -> bool:
    return isinstance(act, nn.SiLU) and not act.inplace


def _norm_fusable(norm: object, wan_rms_norm_cls: type) -> bool:
    return (
        type(norm) is wan_rms_norm_cls
        and getattr(norm, "channel_first", False)
        and isinstance(getattr(norm, "gamma", None), torch.Tensor)
    )


def _install_norm_silu(
    decoder: nn.Module,
    gate: VaeFastPathGate,
    *,
    residual_block_cls: type,
    rms_norm_cls: type,
    label: str,
) -> int | None:
    """Wrap every decoder ``RMS_norm -> SiLU`` chain; ``None`` = fail closed.

    ``residual_block_cls`` / ``rms_norm_cls`` name the model's own Wan-style
    residual block and channel-first RMSNorm (``WanResidualBlock`` /
    ``WanRMS_norm`` or their Qwen-Image twins).
    """
    res_blocks = [m for m in decoder.modules() if isinstance(m, residual_block_cls)]
    eligible = [
        m
        for m in res_blocks
        if type(m) is residual_block_cls
        and _is_plain_silu(m.nonlinearity)
        and _norm_fusable(m.norm1, rms_norm_cls)
        and _norm_fusable(m.norm2, rms_norm_cls)
    ]
    if len(eligible) != len(res_blocks):
        logger.warning(
            "%s: %d/%d residual blocks non-standard; skipping fast path.",
            label,
            len(res_blocks) - len(eligible),
            len(res_blocks),
        )
        return None
    if not (
        _norm_fusable(getattr(decoder, "norm_out", None), rms_norm_cls)
        and _is_plain_silu(getattr(decoder, "nonlinearity", None))
    ):
        logger.warning("%s: non-standard output head; skipping fast path.", label)
        return None

    count = 0
    for m in eligible:
        m.norm1 = FusedWanRMSNormSiLU(m.norm1, gate)
        m.norm2 = FusedWanRMSNormSiLU(m.norm2, gate)
        m.nonlinearity = nn.Identity()
        count += 2
    decoder.norm_out = FusedWanRMSNormSiLU(decoder.norm_out, gate)
    decoder.nonlinearity = nn.Identity()
    return count + 1


def _install_channels_last_upsample(
    decoder: nn.Module, gate: VaeFastPathGate, upsample_cls: type
) -> int:
    """Wrap the 2D nearest upsample of every ``Resample`` (index 0 of its
    ``resample`` Sequential, no parameters, so state_dict names are kept)."""
    count = 0
    for m in decoder.modules():
        seq = getattr(m, "resample", None)
        if (
            isinstance(seq, nn.Sequential)
            and len(seq) >= 1
            and type(seq[0]) is upsample_cls
        ):
            seq[0] = GatedChannelsLastUpsample(seq[0], gate)
            count += 1
    return count


def maybe_optimize_wan_vae(vae: nn.Module) -> nn.Module:
    """Install the quality-gated CUDA Wan VAE decoder fast path."""
    from sglang.multimodal_gen.runtime.models.vaes.wanvae import (
        AutoencoderKLWan,
        WanDecoder3d,
        WanResidualBlock,
        WanRMS_norm,
    )

    if not isinstance(vae, AutoencoderKLWan):
        return vae
    return _maybe_optimize_wan_family_vae(
        vae,
        decoder_cls=WanDecoder3d,
        residual_block_cls=WanResidualBlock,
        rms_norm_cls=WanRMS_norm,
        label="Wan VAE",
    )


def maybe_optimize_qwen_image_vae(vae: nn.Module) -> nn.Module:
    """Install the quality-gated CUDA Qwen-Image VAE decoder fast path.

    The Qwen-Image VAE is the Wan 2.1 VAE (same ``F.normalize``-based
    channel-first RMSNorm, same residual block layout), so it takes the same
    ``RMSNorm -> SiLU`` fusion under the same gate.
    """
    from sglang.multimodal_gen.runtime.models.vaes.autoencoder_kl_qwenimage import (
        AutoencoderKLQwenImage,
        QwenImageAttentionBlock,
        QwenImageDecoder3d,
        QwenImageResidualBlock,
        QwenImageRMS_norm,
        QwenImageUpsample,
    )

    if not isinstance(vae, AutoencoderKLQwenImage):
        return vae
    return _maybe_optimize_wan_family_vae(
        vae,
        decoder_cls=QwenImageDecoder3d,
        residual_block_cls=QwenImageResidualBlock,
        rms_norm_cls=QwenImageRMS_norm,
        upsample_cls=QwenImageUpsample,
        gated_module_classes=(QwenImageAttentionBlock,),
        label="Qwen-Image VAE",
    )


def _maybe_optimize_wan_family_vae(
    vae: nn.Module,
    *,
    decoder_cls: type,
    residual_block_cls: type,
    rms_norm_cls: type,
    label: str,
    upsample_cls: type | None = None,
    gated_module_classes: tuple[type, ...] = (),
) -> nn.Module:
    decoder = getattr(vae, "decoder", None)
    if type(decoder) is not decoder_cls:
        return vae
    if decoder.use_parallel_decode and decoder.world_size > 1:
        logger.info("%s: spatial-parallel decode; skipping fast path.", label)
        return vae
    if not _HAS_TRITON:
        logger.warning("%s: Triton unavailable; skipping fast path.", label)
        return vae

    gate = VaeFastPathGate()
    n_norm = _install_norm_silu(
        decoder,
        gate,
        residual_block_cls=residual_block_cls,
        rms_norm_cls=rms_norm_cls,
        label=label,
    )
    if n_norm is None:
        return vae
    n_up = 0
    if upsample_cls is not None:
        n_up = _install_channels_last_upsample(decoder, gate, upsample_cls)
    if gated_module_classes:
        _install_module_gates(decoder, gate, gated_module_classes)
    register_vae_fast_path_gate(vae, gate)
    logger.info(
        "%s: installed quality-gated fast path (%d RMSNorm+SiLU fusions, "
        "%d channels_last upsamples).",
        label,
        n_norm,
        n_up,
    )
    return vae
