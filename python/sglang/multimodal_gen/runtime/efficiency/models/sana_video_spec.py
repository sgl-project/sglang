# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 SGLang authors
#
# ModelSpec for SANA-Video -- declares the seams the efficiency framework plugs
# into. Registered under "SanaVideo" + the transformer class name so
# get_model_spec(transformer) resolves a live model.
#
# NOTE: SANA-Video's FFN (GLUMBTempConv) reshapes tokens back to the
# (frames, height, width) grid for conv, so arbitrary per-token pruning is NOT
# grid-safe for the FFN. The prunable unit is frames (temporal) or a spatial
# sub-grid; see the token-prune toggle in the DiT for the grid-preserving impl.

from __future__ import annotations

from sglang.multimodal_gen.runtime.efficiency.registry import register_model_spec
from sglang.multimodal_gen.runtime.efficiency.spec import ModelSpec
from sglang.multimodal_gen.runtime.efficiency.technique import Capability


def _sana_video_prunable_segment(hidden, ctx):
    """Whole sequence is the video-token stream (no audio); the grid-safety
    constraint (frames/spatial) is enforced in the DiT forward, not here."""
    n = hidden.shape[ctx.spec.seq_dim]
    return (0, n)


@register_model_spec("SanaVideo", "SanaVideoTransformer3DModel")
def _sana_video_spec() -> ModelSpec:
    return ModelSpec(
        name="SanaVideo",
        capabilities=frozenset(
            {
                Capability.BLOCKS,
                Capability.PRUNABLE_TOKENS,
                Capability.SWAPPABLE_ATTENTION,
            }
        ),
        get_blocks=lambda tf: getattr(tf, "transformer_blocks"),
        prunable_segment=_sana_video_prunable_segment,
        seq_dim=1,
        sp_local_prune=False,
    )
