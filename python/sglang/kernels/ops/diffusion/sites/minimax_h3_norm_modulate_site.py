"""MiniMax-H3 fused RMSNorm + indexed AdaLN, gated by request quality.

Each H3 block runs ``nn.RMSNorm`` and then the indexed AdaLN modulation
``x * (1 + scale[idx]) + shift[idx]``, rounding to bf16 after the norm and after
every modulation op. The fused kernel keeps the same math and operand precision
in one fp32 pass with a single rounding, so ``quality="lossless"`` and ``"high"``
mount it, while ``"exact"`` keeps the reference chain bit-for-bit.
"""

from __future__ import annotations

from sglang.kernels.ops.diffusion.sites.quality_gate import QualityGatedFusion

MINIMAX_H3_NORM_MODULATE = QualityGatedFusion(
    name="MiniMax-H3 fused RMSNorm + AdaLN",
    marker_attr="_sgl_minimax_h3_norm_modulate_site",
    enabled_attr="_sgl_minimax_h3_norm_modulate_enabled",
)
