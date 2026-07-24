# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 SGLang authors
#
# KWLFusions -- BUILD-time transform that enables the KWL operator fusions
# (fused QK+RoPE, RMS/AdaLN, dual/CA-dual modulation, FFN proj_in+gelu,
# compiled gate-to-out, audio QKVG, tiled-VAE compile, ...). These are op-level
# kernel swaps inside the model's modules, chosen at build via env -- writes the
# SHARED Seam.KERNEL_FUSION (composes with everything; not exclusive). set_env()
# emits the SGLANG_HQ_KWL_* flags the existing run pipeline reads.

from __future__ import annotations

from sglang.multimodal_gen.runtime.efficiency.registry import register_transform
from sglang.multimodal_gen.runtime.efficiency.technique import Seam
from sglang.multimodal_gen.runtime.efficiency.transform import (
    ModelTransform,
    TransformContext,
    TransformPhase,
)

# the full-opt KWL bundle
_KWL_FLAGS = (
    "SHARE_BLOCK0_SELF_ATTN",
    "SHARE_GUIDANCE_PREFIX",
    "FUSED_QK_ROPE",
    "FUSED_RMS_ADALN",
    "FUSED_ADALN",
    "FUSED_QKNORM_ROPE",
    "FUSED_DUAL_MODULATE",
    "FUSED_CA_DUAL_MODULATE",
    "FUSED_ADA_VALUES_ALL",
    "FUSED_RESIDUAL_GATE",
    "FUSED_FFN_PROJ_IN_GELU",
    "COMPILE_GATE_TO_OUT",
    "FUSED_AUDIO_QKVG",
    "ENABLE_FUSED_QKNORM_ROPE",
    "COMPILE_TILED_VAE",
)


@register_transform("kwl_fusions")
class KWLFusions(ModelTransform):
    """Enable the KWL operator-fusion bundle. ``flags`` overrides the default
    full-opt bundle (a subset of _KWL_FLAGS) for ablations."""

    name = "kwl_fusions"
    phase = TransformPhase.BUILD
    writes = frozenset({Seam.KERNEL_FUSION})  # SHARED -> never a false conflict

    def __init__(self, flags: tuple[str, ...] | None = None):
        self.flags = tuple(flags) if flags is not None else _KWL_FLAGS

    def set_env(self, ctx: TransformContext) -> None:
        e = ctx.env
        for f in self.flags:
            e[f"SGLANG_HQ_KWL_{f}"] = "1"
