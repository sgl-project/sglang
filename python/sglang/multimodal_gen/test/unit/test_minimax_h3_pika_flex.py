# SPDX-License-Identifier: Apache-2.0
"""Pika H3 Flex (MiniMax-H3 + six-step LoRA adapters) release contracts."""

from __future__ import annotations

import pytest
import torch

from sglang.multimodal_gen.configs.pipeline_configs.minimax_h3 import (
    PikaH3FlexPipelineConfig,
)
from sglang.multimodal_gen.configs.sample.minimax_h3 import PikaH3FlexSamplingParams
from sglang.multimodal_gen.registry import get_model_info
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.denoise_loop import (
    _minimax_h3_renoise_target_rows_,
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

PIKA_KNOTS = (1000, 750, 500, 250, 100, 40)


def test_registry_overlay_and_six_step_contract() -> None:
    info = get_model_info("Pika-Labs/Pika-H3-Flex")
    assert info.sampling_param_cls is PikaH3FlexSamplingParams
    assert info.pipeline_config_cls is PikaH3FlexPipelineConfig
    overlay = resolve_model_overlay("Pika-Labs/Pika-H3-Flex")
    assert overlay["overlay_repo_id"] == "Niehen6174/Pika-H3-Flex-overlay"
    assert is_model_overlay("Pika-Labs/Pika-H3-Flex")
    assert PikaH3FlexSamplingParams(prompt="p").num_inference_steps == 6
    with pytest.raises(ValueError, match="fixed six-step schedule"):
        PikaH3FlexSamplingParams(prompt="p", num_inference_steps=8)
    PikaH3FlexSamplingParams(prompt="p", flow_shift=12.0, audio_flow_shift=3.0)
    for shifts in ({"flow_shift": 10.0}, {"audio_flow_shift": 6.0}):
        with pytest.raises(ValueError, match="trained at"):
            PikaH3FlexSamplingParams(prompt="p", **shifts)


def test_release_metadata_fields() -> None:
    raw = {
        "schema_version": 1,
        "partition": "fl2va",
        "tasks": ["t2va", "fl2va"],
        "sigma_shift_scales": {"video": 12.0, "audio": 3.0},
        "dmd_denoising_steps": list(PIKA_KNOTS),
        "stochastic_sampling": True,
        "lora_adapter": "pika/adapter.safetensors",
    }
    metadata = MiniMaxH3ReleaseMetadata.from_model_index({"_minimax_h3": raw})
    assert metadata.stochastic_sampling
    assert metadata.lora_adapter == "pika/adapter.safetensors"
    for overrides, match in (
        ({"dmd_denoising_steps": None}, "requires dmd_denoising_steps"),
        ({"lora_adapter": "../adapter.safetensors"}, "relative .safetensors"),
    ):
        with pytest.raises(ValueError, match=match):
            MiniMaxH3ReleaseMetadata.from_model_index(
                {"_minimax_h3": {**raw, **overrides}}
            )


@pytest.mark.parametrize("shift", [12.0, 3.0])
def test_knot_schedule(shift: float) -> None:
    # diffusers MiniMaxH3Scheduler(shift_provided_sigmas=True) on the same knots
    base = torch.tensor([k / 1000.0 for k in PIKA_KNOTS] + [0.0], dtype=torch.float64)
    expected = (shift * base / (1 + (shift - 1) * base)).tolist()
    sigmas = minimax_h3_time_shift_sigmas(
        num_steps=6, shift_scale=shift, dmd_steps=PIKA_KNOTS
    )
    assert sigmas == pytest.approx(expected, abs=1e-6)


def test_renoise_update() -> None:
    state, velocity = torch.randn(4, 8), torch.randn(4, 8)
    denoised = state + 0.8 * velocity
    out = state.clone()
    _minimax_h3_renoise_target_rows_(
        out,
        velocity,
        sigma=0.8,
        sigma_next=0.5,
        generator=torch.Generator().manual_seed(7),
    )
    noise = torch.randn(4, 8, generator=torch.Generator().manual_seed(7))
    torch.testing.assert_close(out, 0.5 * denoised + 0.5 * noise)
    last = state.clone()
    _minimax_h3_renoise_target_rows_(
        last, velocity, sigma=0.8, sigma_next=0.0, generator=torch.Generator()
    )
    torch.testing.assert_close(last, denoised)
