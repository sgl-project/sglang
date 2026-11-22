# SPDX-License-Identifier: Apache-2.0
"""pi-H3 (2/4-step pi-Flow adapters of the AdaLN-pruned MiniMax-H3) contracts."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from sglang.multimodal_gen.configs.pipeline_configs.minimax_h3 import (
    PiH3PipelineConfig,
)
from sglang.multimodal_gen.configs.sample.minimax_h3 import PiH3SamplingParams
from sglang.multimodal_gen.registry import get_model_info
from sglang.multimodal_gen.runtime.models.dits.minimax_h3 import (
    _diffusers_h3_checkpoint,
)
from sglang.multimodal_gen.runtime.models.dits.minimax_h3_dwd import (
    MiniMaxH3DWDNormOut,
    MiniMaxH3DWDPixelRefiner,
)
from sglang.multimodal_gen.runtime.pipelines.minimax_h3_pipeline import (
    MiniMaxH3Pipeline,
    PiH3Pipeline,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.release_metadata import (
    MiniMaxH3ReleaseMetadata,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.time_request import (
    minimax_h3_time_shift_sigmas,
)
from sglang.multimodal_gen.runtime.utils.model_overlay import (
    is_model_overlay,
    resolve_model_overlay,
)


def _lakonlab_raw_sigmas(steps: int, max_raw_t: float) -> list[float]:
    # LakonLab FlowEulerODEScheduler(final_step_size_scale=0.6, max_raw_t=...)
    scale = 0.6
    end = max_raw_t * scale / (steps - 1 + scale)
    return torch.linspace(max_raw_t, end, steps, dtype=torch.float64).tolist()


RAW_SIGMAS = {2: _lakonlab_raw_sigmas(2, 0.93), 4: _lakonlab_raw_sigmas(4, 0.96)}


def _release(**overrides):
    raw = {
        "schema_version": 1,
        "partition": "fl2va",
        "tasks": ["t2va", "fl2va"],
        "sigma_shift_scales": {"video": 12.0, "audio": 3.0},
        "raw_sigmas": RAW_SIGMAS[4],
        "lora_adapter": "pi_h3/lora.safetensors",
        "video_output": "dwd",
    }
    raw.update(overrides)
    return MiniMaxH3ReleaseMetadata.from_model_index({"_minimax_h3": raw})


def test_registry_and_sampling_contract() -> None:
    info = get_model_info("Lakonik/pi-H3")
    assert info.sampling_param_cls is PiH3SamplingParams
    assert info.pipeline_config_cls is PiH3PipelineConfig
    assert resolve_model_overlay("Lakonik/pi-H3")["overlay_repo_id"] == (
        "Niehen6174/pi-H3-overlay"
    )
    assert is_model_overlay("Lakonik/pi-H3")
    assert PiH3SamplingParams(prompt="p").num_inference_steps == 4
    PiH3SamplingParams(prompt="p", num_inference_steps=2)
    with pytest.raises(ValueError, match="2 or 4 steps"):
        PiH3SamplingParams(prompt="p", num_inference_steps=8)
    with pytest.raises(ValueError, match="t2va and fl2va only"):
        PiH3SamplingParams(prompt="p", task="ref2va")
    PiH3SamplingParams(prompt="p", flow_shift=12.0, audio_flow_shift=3.0)
    for shifts in ({"flow_shift": 10.0}, {"audio_flow_shift": 6.0}):
        with pytest.raises(ValueError, match="trained at"):
            PiH3SamplingParams(prompt="p", **shifts)
    assert PiH3SamplingParams.trained_steps(None) == 4
    assert PiH3SamplingParams.trained_steps("2step-vae") == 2
    assert PiH3SamplingParams.trained_steps("4step") == 4


def test_sampling_steps_follow_the_variant(monkeypatch) -> None:
    from sglang.multimodal_gen.configs.sample.minimax_h3 import (
        MiniMaxH3SamplingParams,
    )

    # only the step default is under test, not the generic path adjustments
    monkeypatch.setattr(MiniMaxH3SamplingParams, "_adjust", lambda self, args: None)
    params = PiH3SamplingParams(prompt="p")
    params._explicit_fields = set()
    params._adjust(SimpleNamespace(model_variant="2step"))
    assert params.num_inference_steps == 2
    explicit = PiH3SamplingParams(prompt="p", num_inference_steps=4)
    explicit._explicit_fields = {"num_inference_steps"}
    explicit._adjust(SimpleNamespace(model_variant="2step"))
    assert explicit.num_inference_steps == 4


def test_variant_subfolders() -> None:
    assert PiH3Pipeline.model_subfolder_for_variant("2step") == "2step"
    assert PiH3Pipeline.model_subfolder_for_variant("4STEP_vae") == "4step_vae"
    assert PiH3Pipeline.partition_for_variant("2step-vae") == "fl2va"
    with pytest.raises(ValueError, match="unsupported pi-H3 model variant"):
        PiH3Pipeline.model_subfolder_for_variant("fl2va")
    # the base pipeline keeps its partition names
    assert MiniMaxH3Pipeline.partition_for_variant("ref2va") == "ref2va"


def test_release_metadata_raw_sigmas() -> None:
    metadata = _release()
    assert metadata.raw_sigmas == pytest.approx(RAW_SIGMAS[4])
    assert metadata.video_output == "dwd"
    assert metadata.num_trained_steps == 4
    assert _release(video_output="vae", raw_sigmas=RAW_SIGMAS[2]).num_trained_steps == 2
    for overrides, match in (
        ({"dmd_denoising_steps": [1000, 500]}, "not both"),
        ({"raw_sigmas": [0.5, 0.9]}, "strictly decreasing"),
        ({"raw_sigmas": [1.5, 0.5]}, "strictly decreasing"),
        ({"raw_sigmas": [True]}, "strictly decreasing"),
        ({"video_output": "pixels"}, "'vae' or 'dwd'"),
    ):
        with pytest.raises(ValueError, match=match):
            _release(**overrides)
    # re-noising releases may carry their knots as raw sigmas
    assert _release(stochastic_sampling=True).stochastic_sampling


@pytest.mark.parametrize("steps", [2, 4])
@pytest.mark.parametrize("shift", [12.0, 3.0])
def test_raw_sigma_schedule(steps: int, shift: float) -> None:
    base = torch.tensor(RAW_SIGMAS[steps] + [0.0], dtype=torch.float64)
    expected = (shift * base / (1 + (shift - 1) * base)).tolist()
    sigmas = minimax_h3_time_shift_sigmas(
        num_steps=steps, shift_scale=shift, raw_sigmas=tuple(RAW_SIGMAS[steps])
    )
    assert sigmas == pytest.approx(expected, abs=1e-6)
    with pytest.raises(ValueError, match="distilled for"):
        minimax_h3_time_shift_sigmas(
            num_steps=steps + 1, shift_scale=shift, raw_sigmas=tuple(RAW_SIGMAS[steps])
        )


def test_diffusers_checkpoint_synthesizes_rope_frequencies() -> None:
    weight = torch.zeros(4, 4)
    names = [name for name, _ in _diffusers_h3_checkpoint([("proj_in.weight", weight)])]
    assert names.count("rope.inv_freq") == 1
    inv_freq = dict(_diffusers_h3_checkpoint([]))["rope.inv_freq"]
    assert inv_freq.dtype == torch.float32
    expected = 10000.0 ** (-torch.arange(0, 32, 2, dtype=torch.float64) / 32)
    torch.testing.assert_close(inv_freq.double(), expected, rtol=1e-6, atol=0)
    # a checkpoint that stores the frequencies keeps them
    stored = torch.full((16,), 2.0)
    out = dict(_diffusers_h3_checkpoint([("rope.inv_freq", stored)]))
    torch.testing.assert_close(out["rope.inv_freq"], stored)


def test_dwd_norm_out_takes_scale_then_shift() -> None:
    norm = MiniMaxH3DWDNormOut(hidden_size=4, condition_dim=2, eps=1e-5)
    with torch.no_grad():
        norm.linear.weight.zero_()
        norm.linear.bias.copy_(torch.tensor([1.0] * 4 + [0.5] * 4))
    x = torch.randn(3, 4, dtype=torch.bfloat16)
    out = norm(x, torch.zeros(1, 2), torch.zeros(3, dtype=torch.long))
    expected = torch.nn.functional.layer_norm(x, (4,), eps=1e-5) * 2 + 0.5
    torch.testing.assert_close(out, expected.to(torch.bfloat16))


def test_dwd_refiner_rejects_foreign_geometry() -> None:
    with pytest.raises(ValueError, match="frame trimming"):
        MiniMaxH3DWDPixelRefiner(64, (4, 24, 32), 5)
    with pytest.raises(ValueError, match="frame trimming"):
        MiniMaxH3DWDPixelRefiner(64, (4, 32, 32), 4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="varlen attention needs CUDA")
def test_dwd_refiner_frame_layout() -> None:
    refiner = MiniMaxH3DWDPixelRefiner(64, (4, 32, 32), 5, num_layers=2)
    for param in refiner.parameters():
        torch.nn.init.normal_(param, std=0.02)
    refiner = refiner.to("cuda", torch.bfloat16)
    # 7 latents: one native block (1 + 4 * 4 frames) and a 2-latent tail (1 + 4)
    pixels = refiner(
        torch.randn(1, 7 * 2 * 3, 64, device="cuda", dtype=torch.bfloat16), (7, 2, 3)
    )
    assert pixels.shape == (1, 3, 17 + 5, 2 * 32, 3 * 32)
    assert torch.isfinite(pixels).all()
