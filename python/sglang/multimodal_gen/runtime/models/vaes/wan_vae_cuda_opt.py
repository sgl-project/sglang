# SPDX-License-Identifier: Apache-2.0
"""CUDA fast path for the Wan VAE decoder (AutoencoderKLWan).

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


def _is_plain_silu(act: object) -> bool:
    return isinstance(act, nn.SiLU) and not act.inplace


def _norm_fusable(norm: object, wan_rms_norm_cls: type) -> bool:
    return (
        type(norm) is wan_rms_norm_cls
        and getattr(norm, "channel_first", False)
        and isinstance(getattr(norm, "gamma", None), torch.Tensor)
    )


def _install_norm_silu(decoder: nn.Module, gate: VaeFastPathGate) -> int | None:
    """Wrap every decoder ``WanRMS_norm -> SiLU`` chain; ``None`` = fail closed."""
    from sglang.multimodal_gen.runtime.models.vaes.wanvae import (
        WanResidualBlock,
        WanRMS_norm,
    )

    res_blocks = [m for m in decoder.modules() if isinstance(m, WanResidualBlock)]
    eligible = [
        m
        for m in res_blocks
        if type(m) is WanResidualBlock
        and _is_plain_silu(m.nonlinearity)
        and _norm_fusable(m.norm1, WanRMS_norm)
        and _norm_fusable(m.norm2, WanRMS_norm)
    ]
    if len(eligible) != len(res_blocks):
        logger.warning(
            "Wan VAE: %d/%d residual blocks non-standard; skipping fast path.",
            len(res_blocks) - len(eligible),
            len(res_blocks),
        )
        return None
    if not (
        _norm_fusable(getattr(decoder, "norm_out", None), WanRMS_norm)
        and _is_plain_silu(getattr(decoder, "nonlinearity", None))
    ):
        logger.warning("Wan VAE: non-standard output head; skipping fast path.")
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


def maybe_optimize_wan_vae(vae: nn.Module) -> nn.Module:
    """Install the quality-gated CUDA Wan VAE decoder fast path."""
    from sglang.multimodal_gen.runtime.models.vaes.wanvae import (
        AutoencoderKLWan,
        WanDecoder3d,
    )

    if not isinstance(vae, AutoencoderKLWan):
        return vae
    decoder = getattr(vae, "decoder", None)
    if type(decoder) is not WanDecoder3d:
        return vae
    if decoder.use_parallel_decode and decoder.world_size > 1:
        logger.info("Wan VAE: spatial-parallel decode; skipping fast path.")
        return vae
    if not _HAS_TRITON:
        logger.warning("Wan VAE: Triton unavailable; skipping fast path.")
        return vae

    gate = VaeFastPathGate()
    n_norm = _install_norm_silu(decoder, gate)
    if n_norm is None:
        return vae
    register_vae_fast_path_gate(vae, gate)
    logger.info(
        "Wan VAE: installed quality-gated fast path (%d RMSNorm+SiLU fusions).",
        n_norm,
    )
    return vae
