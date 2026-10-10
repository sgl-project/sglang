"""MiniMax-H3 fused RMSNorm + indexed AdaLN, gated by request quality.

Each H3 block runs ``nn.RMSNorm`` and then the indexed AdaLN modulation
``x * (1 + scale[idx]) + shift[idx]``, rounding to bf16 after the norm and after
every modulation op. The fused kernel keeps the same math and operand precision
in one fp32 pass with a single rounding, so ``quality="lossless"`` and ``"high"``
mount it, while ``"exact"`` keeps the reference chain bit-for-bit.
"""

from __future__ import annotations

import logging

import torch.nn as nn

from sglang.kernels.ops.diffusion.sites.quality_gate import QualityGatedFusion

logger = logging.getLogger(__name__)

_FUSION = QualityGatedFusion(
    name="MiniMax-H3 fused RMSNorm + AdaLN",
    marker_attr="_sgl_minimax_h3_norm_modulate_site",
    enabled_attr="_sgl_minimax_h3_norm_modulate_enabled",
)


def mark_minimax_h3_norm_modulate_site(module: nn.Module) -> None:
    """Mark an H3 DiT block; it starts on the reference path."""
    _FUSION.mark(module)


def minimax_h3_norm_modulate_active(module: nn.Module) -> bool:
    return _FUSION.is_enabled(module)


def mount_minimax_h3_norm_modulate(root: nn.Module) -> bool:
    return _FUSION.mount(root, logger=logger)


def unmount_minimax_h3_norm_modulate(root: nn.Module) -> None:
    _FUSION.unmount(root)
