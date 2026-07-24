# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 SGLang authors
#
# Presets -- named assemblies of techniques + transforms that reproduce a known
# optimization configuration. ltx_full_opt() reproduces the LTX-2.3 HQ full-opt
# as a single composed list:
#
#   build/load transforms (delegate to the existing env mechanism):
#     KWLFusions     (KWL operator fusions)
#     NVFP4FFN       (NVFP4 video FFN, load-time quant)
#     SparseAttention(PISA piecewise sparse attn on transformer_2 = stage2 only)
#   runtime techniques (framework-native):
#     StepCache      (SCSP stage-1 cache-core, stage1 only)
#     TokenPrune     (stage-2 midpoint token-prune, stage2 steps 1-2)
#
# Per-stage gating is via Schedules: StepCache only fires in "stage1", TokenPrune
# only in "stage2" (so one composed Plan drives both stages).

from __future__ import annotations

from sglang.multimodal_gen.runtime.efficiency.schedule import at_steps, by_stage, const
from sglang.multimodal_gen.runtime.efficiency.techniques.step_cache import StepCache
from sglang.multimodal_gen.runtime.efficiency.techniques.token_prune import TokenPrune
from sglang.multimodal_gen.runtime.efficiency.transforms.kwl_fusions import KWLFusions
from sglang.multimodal_gen.runtime.efficiency.transforms.nvfp4_ffn import NVFP4FFN
from sglang.multimodal_gen.runtime.efficiency.transforms.sparse_attention import (
    SparseAttention,
)

# SCSP preset 8of15_last_29calls skips a fixed set of late stage-1 calls.
# Represented here as the skipped step indices (approx; the tuned delta/EMA
# controller stays available via the existing cache-core path if exactness is
# required). 29 res2s calls over 15 steps; skip the late cluster.
_SCSP_SKIP_STEPS = "16-28"


def ltx_full_opt(
    *,
    nvfp4: bool = True,
    prune_ratio: float = 0.5,
    prune_steps: str = "1-2",
    pisa_sparsity: float = 0.9,
) -> list:
    """Return the LTX-2.3 full-opt item list (compose() it against the LTX2 spec).

    Set ``nvfp4=False`` for the no-FP4 variant. ``prune_ratio>=1`` or empty
    ``prune_steps`` disables token-prune (== full-opt minus midpoint)."""
    items = [
        KWLFusions(),
        SparseAttention(
            sparsity=pisa_sparsity,
            component="transformer_2",  # stage-2 only
            stage2_dense_layers="0-1",
        ),
        StepCache(
            skip=by_stage(
                {"stage1": at_steps(_SCSP_SKIP_STEPS, True, False)}, default=False
            )
        ),
        TokenPrune(
            keep_ratio=by_stage({"stage2": const(prune_ratio)}, default=1.0),
            method="feat_norm",
            compensation="prev",
            enabled=by_stage(
                {"stage2": at_steps(prune_steps, True, False)}, default=False
            ),
        ),
    ]
    if nvfp4:
        items.insert(1, NVFP4FFN())
    return items


def sana_video_step_cache(*, skip_steps: str = "20-40") -> list:
    """A minimal SANA-Video preset: whole-step reuse on the listed steps.

    This is the framework-native form of the SANA-Video EasyCache; it does NOT
    replace the request-scoped ``EasyCacheController`` already wired into
    ``SanaVideoTransformer3DModel.forward``. Use this preset when driving the
    denoising loop through the framework's Plan (see ``compose(items, spec)``)."""
    return [
        StepCache(skip=at_steps(skip_steps, True, False)),
    ]
