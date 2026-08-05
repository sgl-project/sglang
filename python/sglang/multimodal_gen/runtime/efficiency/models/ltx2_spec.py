# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 SGLang authors
#
# ModelSpec for LTX-2.3 -- declares the seams the efficiency framework plugs
# into. Registered under "LTX2" plus the transformer class name so
# get_model_spec(transformer) resolves a live model.

from __future__ import annotations

from sglang.multimodal_gen.runtime.efficiency.registry import register_model_spec
from sglang.multimodal_gen.runtime.efficiency.spec import ModelSpec
from sglang.multimodal_gen.runtime.efficiency.technique import Capability


def _ltx2_prunable_segment(hidden, ctx):
    """The video-token span that is prunable (midpoint token-prune target).

    LTX-2 concatenates video + audio tokens; the prunable segment is the video
    patch tokens only. Until the exact layout is wired, default to the whole
    sequence (correct on the single-video path where the hidden passed to the
    block loop is the video stream)."""
    n = hidden.shape[ctx.spec.seq_dim]
    return (0, n)


@register_model_spec("LTX2", "LTX2Transformer3DModel", "LTXVideoTransformer3DModel")
def _ltx2_spec() -> ModelSpec:
    return ModelSpec(
        name="LTX2",
        capabilities=frozenset(
            {
                Capability.BLOCKS,
                Capability.PRUNABLE_TOKENS,
                Capability.SWAPPABLE_ATTENTION,
                Capability.RESIDUAL_TUPLE,
            }
        ),
        get_blocks=lambda tf: getattr(tf, "transformer_blocks"),
        prunable_segment=_ltx2_prunable_segment,
        seq_dim=1,
        sp_local_prune=False,  # set True under sequence-parallel (per-rank-local)
    )
