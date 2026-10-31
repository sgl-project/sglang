# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for the opt-in four-step SMD Ref2VA sampling preset."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from sglang.multimodal_gen import envs
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3 import (
    resolved_plan,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.constants import (
    MINIMAX_H3_SIGMAS_EXTRA_KEY,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.release_metadata import (
    MiniMaxH3PartitionAdmissionStage,
    MiniMaxH3ReleaseMetadata,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.stages.timestep_preparation import (
    MiniMaxH3TimestepPreparationStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.time_request import (
    MINIMAX_H3_SMD_SAMPLING_PRESET,
    minimax_h3_smd_sigmas,
    minimax_h3_time_shift_sigmas,
)


def _metadata(**overrides):
    return MiniMaxH3ReleaseMetadata.from_model_index(
        {
            "_minimax_h3": {
                "schema_version": 1,
                "partition": "ref2va",
                "tasks": ["ref2va"],
                "sigma_shift_scales": {"video": 12.0, "audio": 3.0},
                "sampling_preset": MINIMAX_H3_SMD_SAMPLING_PRESET,
                **overrides,
            }
        }
    )


def _stage(preset=MINIMAX_H3_SMD_SAMPLING_PRESET):
    stage = MiniMaxH3TimestepPreparationStage.__new__(MiniMaxH3TimestepPreparationStage)
    stage.sigma_shift_scales = {"video": 12.0, "audio": 3.0}
    stage.dmd_denoising_steps = None
    stage.sampling_preset = preset
    stage._pdd_config = None
    return stage


def _plan(**overrides):
    return SimpleNamespace(
        **{
            "task": "ref2va",
            "flow_shift": None,
            "audio_flow_shift": None,
            "default_flow_shift": 12.0,
            "default_audio_flow_shift": 3.0,
            **overrides,
        }
    )


def _batch(steps=4, warmup=False):
    return SimpleNamespace(
        num_inference_steps=steps,
        is_warmup=warmup,
        extra={},
        sampling_params=SimpleNamespace(task="ref2va", quality="lossless"),
    )


def test_smd_grid_halves_the_last_raw_interval_before_modality_shifts():
    sigmas = minimax_h3_smd_sigmas(num_steps=4, video_shift=12.0, audio_shift=3.0)
    assert sigmas["video"] == pytest.approx([1, 30 / 31, 0.9, 2 / 3, 0])
    assert sigmas["audio"] == pytest.approx([1, 15 / 17, 9 / 13, 1 / 3, 0])


def test_smd_stage_publishes_four_updates_and_preserves_warmup_prefixes():
    stage = _stage()
    request = _batch()
    with patch.object(
        resolved_plan, "minimax_h3_plan_from_batch", return_value=_plan()
    ):
        stage.forward(request, SimpleNamespace())
        assert request.timesteps.device.type == "cpu"
        assert len(request.timesteps) == 4
        for count in (1, 2, 4):
            warmup = _batch(count, warmup=True)
            # A copied serving schedule must not bypass warmup truncation.
            warmup.extra = dict(request.extra)
            stage.forward(warmup, SimpleNamespace())
            assert len(warmup.timesteps) == count
            for modality, values in warmup.extra[MINIMAX_H3_SIGMAS_EXTRA_KEY].items():
                assert (
                    values
                    == request.extra[MINIMAX_H3_SIGMAS_EXTRA_KEY][modality][: count + 1]
                )


@pytest.mark.parametrize(
    "steps,warmup", [(3, False), (0, True), (5, True), (True, False)]
)
def test_smd_rejects_mismatched_serving_or_warmup_step_count(steps, warmup):
    with pytest.raises(ValueError, match="exactly 4 inference steps"):
        minimax_h3_smd_sigmas(
            num_steps=steps, video_shift=12.0, audio_shift=3.0, is_warmup=warmup
        )


def test_smd_admission_rejects_request_overrides_before_encoding():
    with patch.object(PipelineStage, "__init__", return_value=None):
        admission = MiniMaxH3PartitionAdmissionStage(_metadata())
    server_args = SimpleNamespace(minimax_h3_adaln_online=False)
    with patch.object(
        resolved_plan, "minimax_h3_plan_from_batch", return_value=_plan()
    ):
        # release_metadata binds the resolver at import time.
        with (
            patch(
                f"{MiniMaxH3ReleaseMetadata.__module__}.minimax_h3_plan_from_batch",
                return_value=_plan(flow_shift=8.0),
            ),
            pytest.raises(ValueError, match="shifts 12/3"),
        ):
            admission.forward(_batch(), server_args)
        with pytest.raises(ValueError, match="exactly 4 inference steps"):
            admission.forward(_batch(3), server_args)


@pytest.mark.parametrize(
    "overrides",
    [
        {"partition": "fl2va", "tasks": ["t2va"]},
        {"dmd_denoising_steps": [999, 750, 500, 250]},
        {"pdd_fused_heads": "heads.safetensors"},
        {"sampling_preset": "unknown"},
        {"sigma_shift_scales": {"video": 8.0, "audio": 3.0}},
    ],
)
def test_smd_metadata_rejects_incompatible_release_contracts(overrides):
    with pytest.raises(ValueError):
        _metadata(**overrides)


def test_smd_does_not_admit_untrained_tasks_via_a_hybrid_override():
    metadata = replace(_metadata(), partition="hybrid", tasks=("ref2va", "t2va"))
    with pytest.raises(ValueError, match="only Ref2VA"):
        metadata.canonical_task("t2va")


def test_smd_rejects_pdd_or_dmd_before_loading_head_files():
    with patch.object(PipelineStage, "__init__", return_value=None):
        with (
            patch.object(envs, "SGLANG_DIFFUSION_MINIMAX_H3_PDD_HEADS", None),
            pytest.raises(ValueError, match="must not stack PDD"),
        ):
            MiniMaxH3TimestepPreparationStage(
                sampling_preset=MINIMAX_H3_SMD_SAMPLING_PRESET,
                pdd_heads="unused.safetensors",
            )
        with (
            patch.object(envs, "SGLANG_DIFFUSION_MINIMAX_H3_PDD_HEADS", "unused"),
            pytest.raises(ValueError, match="must not stack PDD"),
        ):
            MiniMaxH3TimestepPreparationStage(
                sampling_preset=MINIMAX_H3_SMD_SAMPLING_PRESET
            )
        with (
            patch.object(envs, "SGLANG_DIFFUSION_MINIMAX_H3_PDD_HEADS", None),
            pytest.raises(ValueError, match="DMD schedule"),
        ):
            MiniMaxH3TimestepPreparationStage(
                sampling_preset=MINIMAX_H3_SMD_SAMPLING_PRESET,
                dmd_denoising_steps=(999, 750, 500, 250),
            )


def test_default_schedule_and_smd_dedup_identity_stay_distinct():
    stage = _stage(preset=None)
    request = _batch()
    assert _metadata(sampling_preset=None).sampling_preset is None
    assert (
        _metadata(
            sampling_preset=None, pdd_fused_heads="heads.safetensors"
        ).pdd_fused_heads
        == "heads.safetensors"
    )
    with patch.object(
        resolved_plan, "minimax_h3_plan_from_batch", return_value=_plan()
    ):
        stage.forward(request, SimpleNamespace())
        assert request.extra[MINIMAX_H3_SIGMAS_EXTRA_KEY] == {
            modality: minimax_h3_time_shift_sigmas(num_steps=4, shift_scale=shift)
            for modality, shift in (("video", 12.0), ("audio", 3.0))
        }
        original_key = stage.build_dedup_fingerprint(request, None)
        smd_key = _stage().build_dedup_fingerprint(request, None)
        assert len(original_key) == 7
        assert smd_key == (*original_key, MINIMAX_H3_SMD_SAMPLING_PRESET)
