# SPDX-License-Identifier: Apache-2.0
"""H3 image schedules: PDD grids, step counts, task dedup."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.constants import (
    MINIMAX_H3_SIGMAS_EXTRA_KEY,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.request_validation import (
    minimax_h3_validate_canonical_request,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.resolved_plan import (
    MINIMAX_H3_CANONICAL_REQUEST_EXTRA_KEY,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.stages.timestep_preparation import (
    MiniMaxH3TimestepPreparationStage,
)


def _bits(values):
    return [f"{bits:08x}" for bits in np.asarray(values, np.float32).view(np.uint32)]


def _stage_batch(task, steps):
    conditions = [{"role": "reference", "type": "image", "uri": "ref.png"}]
    canonical = minimax_h3_validate_canonical_request(
        task=task,
        prompt="p",
        conditions=conditions if task == "edit" else [],
        target={"width": 64, "height": 32},
        seed=0,
    )
    return SimpleNamespace(
        extra={MINIMAX_H3_CANONICAL_REQUEST_EXTRA_KEY: canonical},
        num_inference_steps=steps,
        is_warmup=False,
        sigmas=None,
        timesteps=None,
    )


def test_timestep_dedup_separates_tasks_with_equal_step_counts():
    stage = MiniMaxH3TimestepPreparationStage(sigma_shift_scales=None)
    t2i, edit = (
        stage.build_dedup_fingerprint(_stage_batch(task, 8), SimpleNamespace())
        for task in ("t2i", "edit")
    )
    assert t2i != edit


# pdd_config.json sigmas of a 6-step image PDD release.
PDD_SIGMAS = [
    1.0,
    0.9675236940383911,
    0.8875485062599182,
    0.75,
    0.5327739119529724,
    0.23200887441635132,
    0.0,
]


def write_pdd_release(directory, config=None):
    """Fused heads and pdd_config.json of a PDD image release in ``directory``."""
    config = config or {"num_inference_steps": 6, "sigmas": PDD_SIGMAS}
    heads = directory / "pdd_fused_heads.safetensors"
    steps = config["num_inference_steps"]
    save_file(
        {
            "video_out.weight": torch.zeros(steps, 1, 1),
            "video_out.bias": torch.zeros(steps, 1),
        },
        str(heads),
    )
    (directory / "pdd_config.json").write_text(json.dumps(config))
    return heads


def _pdd_stage(tmp_path, config):
    return MiniMaxH3TimestepPreparationStage(
        pdd_heads=str(write_pdd_release(tmp_path, config))
    )


def _float32_timestep_schedule(pdd_sigmas):
    """Each sigma rounded through its float32 timestep 1 - sigma, then 0 appended."""
    timesteps = np.asarray([1 - s for s in pdd_sigmas[:-1]], dtype=np.float32)
    sigmas = np.append(1 - timesteps, np.float32(0))
    return sigmas, (1 - sigmas)[:-1]


@pytest.mark.parametrize("task", ["t2i", "edit"])
def test_pdd_schedule_rounds_sigmas_through_float32_timesteps(tmp_path, task):
    stage = _pdd_stage(tmp_path, {"num_inference_steps": 6, "sigmas": PDD_SIGMAS})
    batch = _stage_batch(task, 6)
    stage.forward(batch, SimpleNamespace())

    sigmas, timesteps = _float32_timestep_schedule(PDD_SIGMAS)
    assert _bits(batch.sigmas) == _bits(sigmas)
    assert _bits(batch.timesteps.numpy()) == _bits(timesteps)
    assert "full" not in batch.extra[MINIMAX_H3_SIGMAS_EXTRA_KEY]


@pytest.mark.parametrize("task", ["t2i", "edit"])
def test_pdd_schedule_pins_the_step_count_except_for_warmup(tmp_path, task):
    stage = _pdd_stage(tmp_path, {"num_inference_steps": 6, "sigmas": PDD_SIGMAS})
    with pytest.raises(ValueError, match="num_inference_steps must be 6"):
        stage.forward(_stage_batch(task, 5), SimpleNamespace())

    # A warmup runs a prefix of the same grid.
    warmup = _stage_batch(task, 2)
    warmup.is_warmup = True
    stage.forward(warmup, SimpleNamespace())
    assert _bits(warmup.sigmas) == _bits(_float32_timestep_schedule(PDD_SIGMAS)[0][:3])
    assert _bits(warmup.extra[MINIMAX_H3_SIGMAS_EXTRA_KEY]["full"]) == _bits(
        _float32_timestep_schedule(PDD_SIGMAS)[0]
    )
    warmup.is_warmup = False
    warmup.num_inference_steps = 6
    stage.forward(warmup, SimpleNamespace())
    assert "full" not in warmup.extra[MINIMAX_H3_SIGMAS_EXTRA_KEY]


@pytest.mark.parametrize(
    "config,match",
    [
        ({"num_inference_steps": 6, "sigmas": PDD_SIGMAS[:-1]}, "sigmas must"),
        ({"num_inference_steps": 6, "sigmas": PDD_SIGMAS[::-1]}, "sigmas must"),
        (
            {"num_inference_steps": 6, "sigmas": PDD_SIGMAS, "sigma_max": 0.999},
            "not both",
        ),
        ({"num_inference_steps": 5, "sigmas": PDD_SIGMAS[1:]}, "does not match"),
    ],
)
def test_pdd_config_must_list_the_grid_of_its_heads(tmp_path, config, match):
    heads = tmp_path / "pdd_fused_heads.safetensors"
    save_file(
        {"video_out.weight": torch.zeros(6, 1, 1), "video_out.bias": torch.zeros(6, 1)},
        str(heads),
    )
    (tmp_path / "pdd_config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match=match):
        MiniMaxH3TimestepPreparationStage(pdd_heads=str(heads))


def test_image_pdd_release_requires_an_explicit_grid(tmp_path):
    stage = _pdd_stage(
        tmp_path, {"num_inference_steps": 6, "video_shift": 3.0, "audio_shift": 3.0}
    )
    with pytest.raises(ValueError, match="as sigmas"):
        stage.forward(_stage_batch("t2i", 6), SimpleNamespace())


@pytest.mark.parametrize("task", ["t2i", "edit"])
def test_image_requests_need_pdd_heads(task):
    # There is no image sampler without the release's fused heads.
    stage = MiniMaxH3TimestepPreparationStage()
    with pytest.raises(ValueError, match="PDD releases only"):
        stage.forward(_stage_batch(task, 6), SimpleNamespace())
