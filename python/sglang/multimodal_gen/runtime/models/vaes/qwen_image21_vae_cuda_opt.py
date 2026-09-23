# SPDX-License-Identifier: Apache-2.0
"""CUDA fast path for the Qwen-Image 2.1 VAE encoder and decoder.

With the gate on (``quality="extra-high"`` or ``"high"``) every convolution
takes channels_last input and lets cuDNN apply the zero padding itself, so the
explicit ``F.pad`` copy and the NCHW/NHWC transposes cuDNN otherwise inserts
around each conv disappear; every ``RMS_norm -> SiLU`` chain runs the fused
channels_last_3d Triton kernel; and the 2D nearest upsample runs the NHWC
gather. With the gate off the original module code runs bit-for-bit. Installed
once at VAE load; all-or-nothing and fail-closed like the Wan-family path.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from sglang.multimodal_gen.runtime.models.vaes.autoencoder_kl_qwenimage21 import (
    AutoencoderKLQwenImage21,
    QwenImage21CausalConv3d,
    QwenImage21ResidualBlock,
    QwenImage21RMS_norm,
    QwenImage21Upsample,
)
from sglang.multimodal_gen.runtime.models.vaes.fast_path_gate import (
    VaeFastPathGate,
    register_vae_fast_path_gate,
)
from sglang.multimodal_gen.runtime.models.vaes.wan_vae_cuda_opt import (
    GatedChannelsLastUpsample,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

try:
    from sglang.kernels.ops.diffusion import can_use_wan_rmsnorm_silu, wan_rmsnorm_silu

    _HAS_TRITON = True
except ImportError:  # pragma: no cover
    _HAS_TRITON = False


def _as_nhwc(x: torch.Tensor) -> torch.Tensor:
    """Canonical channels_last strides; a pure re-labelling when the elements
    are already laid out NHWC (size-1 dims may carry stray strides), else one
    conversion copy at a layout boundary."""
    n, c, h, w = x.shape
    canonical = (h * w * c, 1, w * c, c)
    if x.stride() == canonical:
        return x
    if x.is_contiguous(memory_format=torch.channels_last):
        return x.as_strided(x.shape, canonical)
    return x.contiguous(memory_format=torch.channels_last)


def _nhwc_weight(conv: nn.Conv2d) -> torch.Tensor:
    """channels_last copy of the conv weight, rebuilt only when the parameter
    storage or version changes (LoRA merges, weight reloads)."""
    weight = conv.weight
    key = (weight.data_ptr(), weight._version, weight.dtype)
    cached = conv.__dict__.get("_sgl_nhwc_weight")
    if cached is None or cached[0] != key:
        cached = (key, weight.detach().contiguous(memory_format=torch.channels_last))
        conv.__dict__["_sgl_nhwc_weight"] = cached
    return cached[1]


class _GatedCausalConv3d(QwenImage21CausalConv3d):
    """``QwenImage21CausalConv3d`` whose padding moves into cuDNN and whose
    input runs channels_last while the gate is on."""

    def forward(self, x, cache_x=None):
        if not self._sgl_gate.enabled or torch.compiler.is_compiling():
            return QwenImage21CausalConv3d.forward(self, x, cache_x)
        assert cache_x is None
        # _padding is (w, w, h, h) for F.pad; the spatial pad is symmetric
        padding = (self._padding[2], self._padding[0])
        y = F.conv2d(
            _as_nhwc(x.squeeze(2)),
            _nhwc_weight(self),
            self.bias,
            self.stride,
            padding,
            self.dilation,
            self.groups,
        )
        return y.unsqueeze(2)


class _GatedConv2d(nn.Conv2d):
    """Plain ``nn.Conv2d`` (Resample and attention projections) running
    channels_last while the gate is on."""

    def forward(self, x):
        if not self._sgl_gate.enabled or torch.compiler.is_compiling():
            return nn.Conv2d.forward(self, x)
        return F.conv2d(
            _as_nhwc(x),
            _nhwc_weight(self),
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class _GatedRMSNormSiLU(QwenImage21RMS_norm):
    """``QwenImage21RMS_norm`` that is always followed by SiLU. With the gate
    on it applies the SiLU itself (fused when the layout allows) and the
    paired :class:`_GatedSiLU` steps aside."""

    def forward(self, x):
        if not self._sgl_gate.enabled or torch.compiler.is_compiling():
            return QwenImage21RMS_norm.forward(self, x)
        if (
            self.channel_first
            and isinstance(self.bias, float)
            and self.bias == 0
            and can_use_wan_rmsnorm_silu(x, self.gamma, None)
        ):
            return wan_rmsnorm_silu(x, self.gamma, None, rms_scale=self.scale)
        return F.silu(QwenImage21RMS_norm.forward(self, x))


class _GatedSiLU(nn.SiLU):
    def forward(self, x):
        if self._sgl_gate.enabled and not torch.compiler.is_compiling():
            return x
        return nn.SiLU.forward(self, x)


def _norm_silu_pairs(part: nn.Module) -> list[tuple[nn.Module, str, nn.Module, str]]:
    """(owner, norm attribute, owner, activation attribute) for every
    ``RMS_norm -> SiLU`` chain of an encoder or decoder; ``[]`` if any chain
    is non-standard so the install fails closed."""
    pairs = []
    blocks = [m for m in part.modules() if isinstance(m, QwenImage21ResidualBlock)]
    for block in blocks:
        if type(block) is not QwenImage21ResidualBlock:
            return []
        for name in ("norm1", "norm2"):
            pairs.append((block, name, block, "nonlinearity"))
    pairs.append((part, "norm_out", part, "nonlinearity"))
    for owner, norm_name, act_owner, act_name in pairs:
        norm, act = getattr(owner, norm_name), getattr(act_owner, act_name)
        if not (
            type(norm) is QwenImage21RMS_norm
            and norm.channel_first
            and isinstance(norm.gamma, torch.Tensor)
            and isinstance(norm.bias, float)
            and type(act) in (nn.SiLU, _GatedSiLU)
            and not act.inplace
        ):
            return []
    return pairs


def _install(part: nn.Module, gate: VaeFastPathGate) -> tuple[int, int, int] | None:
    pairs = _norm_silu_pairs(part)
    if not pairs:
        return None
    for owner, norm_name, act_owner, act_name in pairs:
        norm, act = getattr(owner, norm_name), getattr(act_owner, act_name)
        # class swaps keep every parameter registered under its original name
        norm.__class__ = _GatedRMSNormSiLU
        norm._sgl_gate = gate
        act.__class__ = _GatedSiLU
        act._sgl_gate = gate
    convs = 0
    for m in part.modules():
        if type(m) is QwenImage21CausalConv3d:
            m.__class__ = _GatedCausalConv3d
        elif type(m) is nn.Conv2d:
            m.__class__ = _GatedConv2d
        else:
            continue
        m._sgl_gate = gate
        convs += 1
    upsamples = 0
    for m in part.modules():
        seq = getattr(m, "resample", None)
        if isinstance(seq, nn.Sequential) and type(seq[0]) is QwenImage21Upsample:
            seq[0] = GatedChannelsLastUpsample(seq[0], gate)
            upsamples += 1
    return len(pairs), convs, upsamples


def maybe_optimize_qwen_image21_vae(vae: nn.Module) -> nn.Module:
    """Install the quality-gated CUDA fast path on a Qwen-Image 2.1 VAE."""
    if not isinstance(vae, AutoencoderKLQwenImage21):
        return vae
    if vae.spatial_parallel:
        logger.info("Qwen-Image 2.1 VAE: spatial-parallel decode; skipping fast path.")
        return vae
    if not _HAS_TRITON:
        logger.warning("Qwen-Image 2.1 VAE: Triton unavailable; skipping fast path.")
        return vae
    parts = [
        getattr(vae, name) for name in ("encoder", "decoder") if hasattr(vae, name)
    ]
    gate = VaeFastPathGate()
    counts = [_install(part, gate) for part in parts]
    if any(count is None for count in counts):
        logger.warning("Qwen-Image 2.1 VAE: non-standard blocks; skipping fast path.")
        return vae
    register_vae_fast_path_gate(vae, gate)
    logger.info(
        "Qwen-Image 2.1 VAE: installed quality-gated fast path (%d RMSNorm+SiLU "
        "fusions, %d channels_last convs, %d channels_last upsamples).",
        sum(c[0] for c in counts),
        sum(c[1] for c in counts),
        sum(c[2] for c in counts),
    )
    return vae
