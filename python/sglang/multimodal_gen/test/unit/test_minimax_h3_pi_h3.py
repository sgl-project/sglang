# SPDX-License-Identifier: Apache-2.0
"""pi-H3 (2/4-step adapters of the AdaLN-pruned MiniMax-H3) contracts."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from sglang.multimodal_gen.configs.sample.minimax_h3 import PiH3SamplingParams
from sglang.multimodal_gen.registry import get_model_info
from sglang.multimodal_gen.runtime.models.dits.minimax_h3 import (
    _diffusers_h3_checkpoint,
)
from sglang.multimodal_gen.runtime.models.dits.minimax_h3_dwd import (
    MiniMaxH3DWDNormOut,
    MiniMaxH3DWDPixelRefiner,
)
from sglang.multimodal_gen.runtime.pipelines.minimax_h3_pipeline import PiH3Pipeline
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.release_metadata import (
    MiniMaxH3ReleaseMetadata,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.time_request import (
    minimax_h3_time_shift_sigmas,
)

# LakonLab FlowEulerODEScheduler knots (final_step_size_scale 0.6)
RAW_SIGMAS = {2: [0.93, 0.34875], 4: [0.96, 0.693333333333, 0.426666666667, 0.16]}


def _release(**overrides):
    raw = {
        "schema_version": 1,
        "partition": "fl2va",
        "tasks": ["t2va", "fl2va"],
        "sigma_shift_scales": {"video": 12.0, "audio": 3.0},
        "raw_sigmas": RAW_SIGMAS[4],
        "video_output": "dwd",
    }
    return MiniMaxH3ReleaseMetadata.from_model_index(
        {"_minimax_h3": {**raw, **overrides}}
    )


def test_registry_variants_and_sampling() -> None:
    assert get_model_info("Lakonik/pi-H3").sampling_param_cls is PiH3SamplingParams
    assert PiH3Pipeline.model_subfolder_for_variant("4STEP_vae") == "4step_vae"
    assert PiH3Pipeline.partition_for_variant("2step") == "fl2va"
    assert PiH3SamplingParams.trained_steps("2step-vae") == 2
    assert PiH3SamplingParams(prompt="p").num_inference_steps == 4
    for kwargs, match in (
        ({"num_inference_steps": 8}, "2 or 4 steps"),
        ({"task": "ref2va"}, "t2va and fl2va only"),
        ({"flow_shift": 10.0}, "trained at"),
    ):
        with pytest.raises(ValueError, match=match):
            PiH3SamplingParams(prompt="p", **kwargs)


def test_default_steps_follow_variant_or_subfolder(monkeypatch) -> None:
    from sglang.multimodal_gen.configs.sample.minimax_h3 import (
        MiniMaxH3SamplingParams,
    )

    # only the step default is under test, not the generic path adjustments
    monkeypatch.setattr(MiniMaxH3SamplingParams, "_adjust", lambda self, args: None)
    for variant, subfolder, steps in (
        ("2step", None, 2),
        (None, "/m/2step_vae/", 2),
        (None, None, 4),
    ):
        params = PiH3SamplingParams(prompt="p")
        params._explicit_fields = set()
        params._adjust(
            SimpleNamespace(model_variant=variant, model_subfolder=subfolder)
        )
        assert params.num_inference_steps == steps


def test_release_metadata() -> None:
    metadata = _release()
    assert metadata.video_output == "dwd" and metadata.num_trained_steps == 4
    for overrides, match in (
        ({"dmd_denoising_steps": [1000, 500]}, "not both"),
        ({"raw_sigmas": [0.5, 0.9]}, "strictly decreasing"),
        ({"video_output": "pixels"}, "'vae' or 'dwd'"),
    ):
        with pytest.raises(ValueError, match=match):
            _release(**overrides)


@pytest.mark.parametrize("steps", [2, 4])
def test_raw_sigma_schedule(steps: int) -> None:
    base = torch.tensor(RAW_SIGMAS[steps] + [0.0], dtype=torch.float64)
    expected = (12 * base / (1 + 11 * base)).tolist()
    sigmas = minimax_h3_time_shift_sigmas(
        num_steps=steps, shift_scale=12.0, raw_sigmas=tuple(RAW_SIGMAS[steps])
    )
    assert sigmas == pytest.approx(expected, abs=1e-6)


def test_diffusers_checkpoint_synthesizes_rope_frequencies() -> None:
    inv_freq = dict(_diffusers_h3_checkpoint([]))["rope.inv_freq"]
    expected = 10000.0 ** (-torch.arange(0, 32, 2, dtype=torch.float64) / 32)
    torch.testing.assert_close(inv_freq.double(), expected, rtol=1e-6, atol=0)
    stored = torch.full((16,), 2.0)
    out = list(_diffusers_h3_checkpoint([("rope.inv_freq", stored)]))
    assert len(out) == 1 and out[0][1] is stored


def test_dwd_norm_out_takes_scale_then_shift() -> None:
    norm = MiniMaxH3DWDNormOut(hidden_size=4, condition_dim=2, eps=1e-5)
    with torch.no_grad():
        norm.linear.weight.zero_()
        norm.linear.bias.copy_(torch.tensor([1.0] * 4 + [0.5] * 4))
    x = torch.randn(3, 4, dtype=torch.bfloat16)
    out = norm(x, torch.zeros(1, 2), torch.zeros(3, dtype=torch.long))
    expected = torch.nn.functional.layer_norm(x, (4,), eps=1e-5) * 2 + 0.5
    torch.testing.assert_close(out, expected.to(torch.bfloat16))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="varlen attention needs CUDA")
def test_dwd_refiner_frame_layout() -> None:
    refiner = MiniMaxH3DWDPixelRefiner(64, (4, 32, 32), 5, num_layers=2)
    for param in refiner.parameters():
        torch.nn.init.normal_(param, std=0.02)
    refiner = refiner.to("cuda", torch.bfloat16)
    # 7 latents: one native block (1 + 4 * 4 frames) and a 2-latent tail (1 + 4)
    features = torch.randn(1, 7 * 2 * 3, 64, device="cuda", dtype=torch.bfloat16)
    pixels = refiner(features, (7, 2, 3))
    assert pixels.shape == (1, 3, 22, 2 * 32, 3 * 32)
    assert torch.isfinite(pixels).all()
