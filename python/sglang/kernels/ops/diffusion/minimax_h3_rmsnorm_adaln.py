# SPDX-License-Identifier: Apache-2.0
"""Request-scoped MiniMax H3 indexed RMSNorm+AdaLN fusion controls.

The Triton implementation is numerically close to, but not bit-exact with,
the eager RMSNorm reduction. H3 therefore exposes it through the shared
``quality="high"`` fusion protocol while ``quality="lossless"`` keeps the
reference operation unchanged.
"""

from __future__ import annotations

from torch import nn

from sglang.kernels.ops.diffusion.quality_gate import QualityGatedFusion

_SITE_MARKER_ATTR = "_sgl_minimax_h3_indexed_rmsnorm_adaln_site"
_SITE_ENABLED_ATTR = "_sgl_minimax_h3_indexed_rmsnorm_adaln_enabled"
_FUSION = QualityGatedFusion(
    name="MiniMax H3 indexed RMSNorm+AdaLN",
    marker_attr=_SITE_MARKER_ATTR,
    enabled_attr=_SITE_ENABLED_ATTR,
)


def mark_minimax_h3_indexed_rmsnorm_adaln_site(module: nn.Module) -> None:
    """Mark an H3 DiT block as an eligible fusion site, disabled by default."""
    _FUSION.mark(module)


def minimax_h3_indexed_rmsnorm_adaln_active(module: nn.Module) -> bool:
    return _FUSION.is_enabled(module)


def mount_minimax_h3_indexed_rmsnorm_adaln(root: nn.Module) -> bool:
    return _FUSION.mount(root)


def unmount_minimax_h3_indexed_rmsnorm_adaln(root: nn.Module) -> None:
    _FUSION.unmount(root)


__all__ = [
    "mark_minimax_h3_indexed_rmsnorm_adaln_site",
    "minimax_h3_indexed_rmsnorm_adaln_active",
    "mount_minimax_h3_indexed_rmsnorm_adaln",
    "unmount_minimax_h3_indexed_rmsnorm_adaln",
]
