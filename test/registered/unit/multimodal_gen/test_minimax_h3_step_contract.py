# SPDX-License-Identifier: Apache-2.0

import json
import os
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from safetensors.torch import save_file

from sglang.multimodal_gen.configs.sample.minimax_h3 import (
    FastH3SamplingParams,
    MiniMaxH3SamplingParams,
)
from sglang.multimodal_gen.configs.sample.minimax_h3_vdn import VDNH3SamplingParams
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.constants import (
    MINIMAX_H3_SIGMAS_EXTRA_KEY,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.denoise_loop import (
    MiniMaxH3DenoiseBranch,
    minimax_h3_denoise_loop,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.packed_sequence import (
    minimax_h3_packed_sequence,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.release_metadata import (
    MiniMaxH3PartitionAdmissionStage,
    MiniMaxH3ReleaseMetadata,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.resolved_plan import (
    MINIMAX_H3_CANONICAL_REQUEST_EXTRA_KEY,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.stages.denoising import (
    MiniMaxH3DenoisingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.stages.timestep_preparation import (
    MiniMaxH3TimestepPreparationStage,
)
from sglang.multimodal_gen.runtime.server_args import server_args as server_args_module
from sglang.multimodal_gen.tools import (
    build_minimax_h3_adaln_cache,
    fuse_minimax_h3_pdd_heads,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@pytest.fixture(autouse=True)
def global_server_args(monkeypatch):
    monkeypatch.setattr(
        server_args_module,
        "_global_server_args",
        SimpleNamespace(comfyui_mode=False),
    )


@pytest.fixture
def admission():
    metadata = MiniMaxH3ReleaseMetadata.from_model_index(
        {
            "_minimax_h3": {
                "schema_version": 1,
                "partition": "fl2va",
                "tasks": ["t2va", "fl2va"],
                "task_aliases": {},
                "sigma_shift_scales": {"video": 12.0, "audio": 3.0},
            }
        }
    )
    return MiniMaxH3PartitionAdmissionStage(metadata)


def _request(steps):
    sampling = MiniMaxH3SamplingParams(
        prompt="step contract",
        task="t2va",
        conditions=[],
        target={"short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 5.0},
        num_inference_steps=steps,
    )
    return Req(sampling_params=sampling, extra=sampling.build_request_extra())


@pytest.mark.parametrize("steps", [1, 2, 10, 50])
def test_admission_schedule_loop_and_workload_agree(admission, steps):
    request = _request(steps)
    server_args = SimpleNamespace(minimax_h3_adaln_online=False)
    assert admission.forward(request, server_args) is request
    assert MINIMAX_H3_CANONICAL_REQUEST_EXTRA_KEY in request.extra
    preparation = MiniMaxH3TimestepPreparationStage()
    preparation.forward(request, server_args)
    sigmas = request.extra[MINIMAX_H3_SIGMAS_EXTRA_KEY]
    assert len(request.timesteps) == steps
    for schedule in sigmas.values():
        assert len(schedule) == steps + 1
        assert schedule[0] == 1.0 and schedule[-1] == 0.0
    stage = MiniMaxH3DenoisingStage.__new__(MiniMaxH3DenoisingStage)
    assert stage.default_workload_iterations(request, steps) == steps
    packed = minimax_h3_packed_sequence(
        text_len=3,
        latent_t=2,
        latent_h=4,
        latent_w=4,
        audio_t=3,
        include_keyframe_cond=False,
    )
    branch = MiniMaxH3DenoiseBranch(
        packed=packed,
        text_embeddings=torch.zeros(3, 5120, device="cuda"),
        token_tags=packed["token_tags"],
        device=torch.device("cuda"),
    )
    calls = []
    plans = []

    def prepare_plans(timesteps):
        plans.extend(timesteps)
        return None

    def forward(_model, _kwargs, step):
        calls.append(step)
        return (
            torch.ones(int(branch.update_mask.sum()), 96, device="cuda"),
            torch.ones(branch.audio_pos.numel(), 32, device="cuda"),
        )

    video, audio = minimax_h3_denoise_loop(
        model=SimpleNamespace(prepare_adaln_plans=prepare_plans),
        model_forward=forward,
        positive=branch,
        initial_video_rows=torch.zeros(branch.img_pos.numel(), 96, device="cuda"),
        initial_audio_rows=torch.zeros(branch.audio_pos.numel(), 32, device="cuda"),
        keyframe_cond_rows=None,
        sigmas_video=sigmas["video"],
        sigmas_audio=sigmas["audio"],
        device=torch.device("cuda"),
    )
    assert calls == list(range(steps)) and len(plans) == steps
    # H3's velocity convention gives a +1 denoised target for a unit
    # prediction from zero rows once the schedule reaches sigma=0.
    torch.testing.assert_close(
        video[branch.video_target_slice],
        torch.ones_like(video[branch.video_target_slice]),
    )
    torch.testing.assert_close(
        audio[branch.audio_target_slice],
        torch.ones_like(audio[branch.audio_target_slice]),
    )


def test_online_adaln_admission_capacity_boundary(admission):
    server_args = SimpleNamespace(minimax_h3_adaln_online=True)
    with patch.dict(os.environ, {"SGLANG_DIFFUSION_MINIMAX_H3_ADALN_GPU_PLANS": "8"}):
        for steps in (1, 8):
            request = _request(steps)
            assert admission.forward(request, server_args) is request
        with pytest.raises(ValueError, match="9 AdaLN plans.*slab holds 8"):
            admission.forward(_request(9), server_args)


@pytest.mark.parametrize(
    "params_class,steps", [(FastH3SamplingParams, 8), (VDNH3SamplingParams, 8)]
)
def test_distilled_models_keep_their_trained_grid(params_class, steps):
    request = Req(sampling_params=params_class(prompt="step contract"))
    assert request.num_inference_steps == steps
    with pytest.raises(ValueError):
        params_class(prompt="step contract", num_inference_steps=steps + 1)


def test_fasth3_trained_rungs_preserve_serving_and_warmup_counts(admission):
    sampling = FastH3SamplingParams(
        prompt="step contract",
        task="t2va",
        conditions=[],
        target={"short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 5.0},
        flow_shift=10.0,
        audio_flow_shift=3.0,
    )
    request = Req(sampling_params=sampling, extra=sampling.build_request_extra())
    admission.forward(request, SimpleNamespace(minimax_h3_adaln_online=False))
    preparation = MiniMaxH3TimestepPreparationStage(
        dmd_denoising_steps=(999, 874, 749, 624, 500, 375, 250, 125)
    )
    warmup_requests = [request.copy_as_warmup(steps) for steps in (1, 2, 8)]
    preparation.forward(request, SimpleNamespace())
    assert len(request.timesteps) == 8
    assert all(
        len(schedule) == 9 and schedule[-1] == 0.0
        for schedule in request.extra[MINIMAX_H3_SIGMAS_EXTRA_KEY].values()
    )
    for steps, warmup in zip((1, 2, 8), warmup_requests):
        preparation.forward(warmup, SimpleNamespace())
        assert len(warmup.timesteps) == steps
        for modality, schedule in warmup.extra[MINIMAX_H3_SIGMAS_EXTRA_KEY].items():
            assert (
                schedule
                == request.extra[MINIMAX_H3_SIGMAS_EXTRA_KEY][modality][: steps + 1]
            )


@pytest.mark.parametrize("steps", [1, 2, 10, 50])
@pytest.mark.parametrize(
    "mode", ["t2va", "fl2va", "ref2va-image", "ref2va-audio", "ref2va-mixed"]
)
def test_offline_adaln_builder_prepares_one_plan_per_transition(steps, mode):
    plans = build_minimax_h3_adaln_cache._cache_timestep_plans(
        SimpleNamespace(
            timesteps=None,
            num_inference_steps=steps,
            flow_shift=12.0,
            audio_flow_shift=3.0,
            imgvid_cond_noise_aug=0.999,
            audio_cond_noise_aug=1.0,
            mode=mode,
        )
    )
    assert len(plans) == steps
    assert all(plan.numel() > 0 for plan in plans)


def test_fused_pdd_config_and_warmup_match_head_count(tmp_path, admission):
    heads = {
        name: torch.ones(32, 2, 3) if name.endswith("weight") else torch.ones(32, 2)
        for name in (
            "proj_out.weight",
            "proj_out.bias",
            "audio_proj_out.weight",
            "audio_proj_out.bias",
        )
    }
    save_file(heads, tmp_path / "pdd_heads.safetensors")
    config_path = tmp_path / "pdd_config.json"
    config_path.write_text(json.dumps({"num_steps": 32, "block_size": 4}))
    with patch.object(sys, "argv", ["fuse_minimax_h3_pdd_heads", str(tmp_path)]):
        assert fuse_minimax_h3_pdd_heads.main() == 0
    config = json.loads(config_path.read_text())
    assert config["fused_steps"] == config["num_inference_steps"] == 8
    with patch.dict(
        os.environ,
        {
            "SGLANG_DIFFUSION_MINIMAX_H3_PDD_HEADS": str(
                tmp_path / "pdd_fused_heads.safetensors"
            )
        },
    ):
        preparation = MiniMaxH3TimestepPreparationStage()
    serving = _request(8)
    admission.forward(serving, SimpleNamespace(minimax_h3_adaln_online=False))
    preparation.forward(serving, SimpleNamespace())
    for steps in (1, 2, 8):
        warmup = serving.copy_as_warmup(steps)
        preparation.forward(warmup, SimpleNamespace())
        assert len(warmup.timesteps) == steps
        for modality, schedule in warmup.extra[MINIMAX_H3_SIGMAS_EXTRA_KEY].items():
            assert (
                schedule
                == serving.extra[MINIMAX_H3_SIGMAS_EXTRA_KEY][modality][: steps + 1]
            )

    config["num_inference_steps"] = 9
    config_path.write_text(json.dumps(config))
    with patch.dict(
        os.environ,
        {
            "SGLANG_DIFFUSION_MINIMAX_H3_PDD_HEADS": str(
                tmp_path / "pdd_fused_heads.safetensors"
            )
        },
    ):
        with pytest.raises(ValueError, match="config does not match the fused heads"):
            MiniMaxH3TimestepPreparationStage()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
