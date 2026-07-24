# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 SGLang authors
#
# SparseAttention (PISA) -- BUILD-time transform that installs the piecewise
# sparse-attention backend for a component. The kernel is selected at build;
# the per-step dense->sparse schedule is read by the kernel at runtime. We do
# NOT reimplement the kernel -- set_env() emits the exact env the existing
# piecewise_attn backend already consumes.

from __future__ import annotations

from sglang.multimodal_gen.runtime.efficiency.registry import register_transform
from sglang.multimodal_gen.runtime.efficiency.technique import Capability, Seam
from sglang.multimodal_gen.runtime.efficiency.transform import (
    ModelTransform,
    TransformContext,
    TransformPhase,
)


@register_transform("sparse_attention")
class SparseAttention(ModelTransform):
    """Install PISA piecewise sparse attention on the given component(s).

    Parameters mirror the existing SGLANG_PIECEWISE_ATTN_* knobs. ``dense_steps``
    is the per-step schedule's warmup (first-N dense) expressed as an int for
    the existing kernel; ``component`` selects which transformer the backend
    applies to (e.g. ``transformer_2`` = stage-2 only, the full-opt setting)."""

    name = "sparse_attention"
    phase = TransformPhase.BUILD
    writes = frozenset({Seam.ATTENTION_BACKEND})
    required_capabilities = frozenset({Capability.SWAPPABLE_ATTENTION})

    def __init__(
        self,
        sparsity: float = 0.9,
        block_size: int = 64,
        only_video_self: bool = True,
        component: str = "transformer_2",
        stage1_dense: bool = False,
        dense_steps: int = 0,
        route_mode: str = "score",
        dense_fallback: str = "fa",
        stage2_dense_layers: str = "0-1",
    ):
        self.sparsity = sparsity
        self.block_size = block_size
        self.only_video_self = only_video_self
        self.component = component
        self.stage1_dense = stage1_dense
        self.dense_steps = dense_steps
        self.route_mode = route_mode
        self.dense_fallback = dense_fallback
        self.stage2_dense_layers = stage2_dense_layers

    def set_env(self, ctx: TransformContext) -> None:
        e = ctx.env
        # other components stay dense (fa); the chosen component goes piecewise.
        backends = {"transformer": "fa", "transformer_2": "fa"}
        backends[self.component] = "piecewise_attn"
        e["SGLANG_HQ_COMPONENT_ATTENTION_BACKENDS"] = ",".join(
            f"{k}={v}" for k, v in backends.items()
        )
        cfg = (
            f"piecewise_sparsity={self.sparsity},"
            f"piecewise_block_size={self.block_size},"
            f"piecewise_only_video_self_attention={str(self.only_video_self).lower()},"
            f"piecewise_stage1_schedule={str(self.stage1_dense).lower()},"
            f"piecewise_stage1_dense_steps={self.dense_steps},"
            f"piecewise_stage2_dense_layers={self.stage2_dense_layers},"
            f"piecewise_approx_remainder=true,"
            f"piecewise_route_mode={self.route_mode},"
            f"piecewise_dense_fallback={self.dense_fallback}"
        )
        e["SGLANG_HQ_ATTENTION_BACKEND_CONFIG"] = cfg
