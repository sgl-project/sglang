# SPDX-License-Identifier: Apache-2.0
"""GAP 1 regression: Kandinsky6's RoPE ``scale_factor`` must come from the
checkpoint's ``transformer/config.json`` (a fixed per-checkpoint constant,
matching the diffusers reference's ``Kandinsky6TI2VAPipeline.__init__``,
``pipeline_kandinsky6_ti2va.py:807-809``:
``self.scale_factor = tuple(transformer_config.get("scale_factor", (1.0, 2.0,
2.0)))``), not a resolution-bucket heuristic recomputed per request from
``(height, width)``.
"""

from __future__ import annotations

import json

import pytest
from huggingface_hub import hf_hub_download

from sglang.multimodal_gen.configs.models.dits.kandinsky6 import (
    Kandinsky6ArchConfig,
    Kandinsky6VideoAudioConfig,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.kandinsky6.denoising import (
    Kandinsky6DenoisingStage,
)

PRO_SFT_REPO = "kandinskylab/Kandinsky-6.0-Pro-sft-5s-Diffusers"


def _deleted_resolution_bucket_heuristic(
    height: int, width: int
) -> tuple[float, float, float]:
    """The exact resolution-bucket heuristic GAP 1's fix deletes from
    ``Kandinsky6DenoisingStage`` -- reproduced here, standalone, ONLY so this
    regression test can show what the old (buggy) code would have computed,
    for comparison against the fixed, checkpoint-driven value. This is not
    imported from production code: the heuristic no longer exists there.
    """
    if 480 <= height <= 854 and 480 <= width <= 854:
        return (1.0, 2.0, 2.0)
    return (1.0, 3.16, 3.16)


def test_scale_factor_field_defaults_to_diffusers_fixed_constant():
    # Both real Pro checkpoints (sft and distill) ship "scale_factor":
    # [1.0, 2.0, 2.0] in transformer/config.json; the diffusers reference
    # falls back to this exact tuple when a checkpoint config omits the key.
    # On pre-fix code, `scale_factor` is not a declared field at all (only
    # reachable via `extra_attrs` after an explicit `update_model_arch`
    # call), so a fresh `Kandinsky6ArchConfig()` raises AttributeError here.
    assert Kandinsky6ArchConfig().scale_factor == (1.0, 2.0, 2.0)


def test_denoising_stage_no_longer_has_a_resolution_based_heuristic():
    # GAP 1's fix deletes `_scale_factor(height, width)` entirely -- the
    # resolved value must come from the DiT's arch config instead.
    assert not hasattr(Kandinsky6DenoisingStage, "_scale_factor")


@pytest.mark.parametrize(
    "height,width",
    [(512, 768), (480, 864), (720, 1280)],
)
def test_real_checkpoint_scale_factor_is_fixed_not_bucketed(height, width):
    """The real Kandinsky-6.0-Pro-sft-5s-Diffusers checkpoint's own
    transformer/config.json resolves to a fixed (1.0, 2.0, 2.0) scale_factor
    at every request resolution -- unlike the deleted resolution-bucket
    heuristic, which returned the wrong (1.0, 3.16, 3.16) for requests
    outside its hardcoded [480, 854] band (480x864 and 720x1280 both fall
    outside it on at least one axis).
    """
    config_path = hf_hub_download(PRO_SFT_REPO, "transformer/config.json")
    with open(config_path) as f:
        real_transformer_config = json.load(f)

    dit_config = Kandinsky6VideoAudioConfig()
    dit_config.update_model_arch(real_transformer_config)
    resolved = dit_config.arch_config.scale_factor

    assert resolved == (1.0, 2.0, 2.0)

    old_heuristic_value = _deleted_resolution_bucket_heuristic(height, width)
    if old_heuristic_value != (1.0, 2.0, 2.0):
        assert resolved != old_heuristic_value
