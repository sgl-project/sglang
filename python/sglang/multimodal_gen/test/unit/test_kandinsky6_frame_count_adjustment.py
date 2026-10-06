# SPDX-License-Identifier: Apache-2.0
"""GAP 5 regression: ``SamplingParams._adjust``'s multi-GPU frame-count
rounding must not compose with ``Kandinsky6LatentPreparationStage``'s own
``num_frames % 4 == 1`` rounding (the diffusers reference's own
``latent_frames = (num_frames - 1) // 4 + 1`` convention, the single source
of truth for this model -- the reference pipeline has no GPU-count-based
frame adjustment at all).

Pre-fix, the two rounding rules composed for ``num_gpus > 1``: the
documented default ``num_frames=121`` decoded to 125/129/125/125 frames at
2/3/4/8 GPUs instead of the diffusers reference's 121. Fix:
``Kandinsky6TI2VASamplingParams.adjust_frames = False`` opts out of the
GPU-count rounding, leaving ``Kandinsky6LatentPreparationStage``'s own
rounding as the only one applied, at any ``num_gpus``.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from sglang.multimodal_gen.configs.pipeline_configs.kandinsky6 import (
    Kandinsky6TI2VAPipelineConfig,
)
from sglang.multimodal_gen.configs.sample.kandinsky6 import (
    Kandinsky6TI2VADistilledSamplingParams,
    Kandinsky6TI2VASamplingParams,
)


def _final_num_frames(num_frames: int, num_gpus: int, *, adjust_frames) -> int:
    """SamplingParams._adjust's multi-GPU rounding, followed by
    Kandinsky6LatentPreparationStage's own num_frames % 4 == 1 rounding
    (``latent_preparation.py``'s temporal_ratio=4 alignment) -- the same
    two-step composition the audit measured end to end.
    """
    params = Kandinsky6TI2VASamplingParams(
        num_frames=num_frames, adjust_frames=adjust_frames
    )
    server_args = SimpleNamespace(
        pipeline_config=Kandinsky6TI2VAPipelineConfig(),
        output_path=None,
        comfyui_mode=True,
        num_gpus=num_gpus,
    )
    params._adjust(server_args)
    after_gpu_adjust = params.num_frames
    if after_gpu_adjust % 4 != 1:
        after_gpu_adjust = after_gpu_adjust // 4 * 4 + 1
    return after_gpu_adjust


def test_adjust_frames_defaults_to_false_for_kandinsky6():
    assert Kandinsky6TI2VASamplingParams().adjust_frames is False
    # Inherited by the distilled sampling params too.
    assert Kandinsky6TI2VADistilledSamplingParams().adjust_frames is False


@pytest.mark.parametrize("num_gpus", [1, 2, 3, 4, 8])
def test_default_num_frames_decodes_to_121_at_every_gpu_count(num_gpus):
    assert _final_num_frames(121, num_gpus, adjust_frames=False) == 121


@pytest.mark.parametrize(
    "num_gpus,old_buggy_result",
    [(2, 125), (3, 129), (4, 125), (8, 125)],
)
def test_old_default_composed_two_rounding_rules(num_gpus, old_buggy_result):
    """Pins the pre-fix bug itself (``adjust_frames=True``, the base
    ``SamplingParams`` default every other model still uses): the
    documented default num_frames=121 does NOT decode to 121 once the two
    rounding rules compose for num_gpus > 1. Confirms the fix in this file
    is what changes the outcome, not some other code path.
    """
    assert _final_num_frames(121, num_gpus, adjust_frames=True) == old_buggy_result
