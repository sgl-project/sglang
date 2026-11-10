# SPDX-License-Identifier: Apache-2.0
"""Open-H3-Regenerate-2K (MiniMax-H3 draft refiner) contracts."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from sglang.multimodal_gen.configs.pipeline_configs.minimax_h3 import (
    OpenH3RegeneratePipelineConfig,
)
from sglang.multimodal_gen.configs.sample.minimax_h3 import (
    OpenH3RegenerateSamplingParams,
)
from sglang.multimodal_gen.registry import (
    get_model_info,
    get_non_diffusers_pipeline_name,
)
from sglang.multimodal_gen.runtime.models.dits.minimax_h3 import (
    MiniMaxH3AdalnProj,
    MiniMaxH3DiTModel,
)
from sglang.multimodal_gen.runtime.models.upsampler.minimax_h3_latent_upscaler import (
    MiniMaxH3LatentUpscaler,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.request_validation import (
    minimax_h3_validate_canonical_request,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.resolved_plan import (
    minimax_h3_resolve_plan,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.stages.regenerate_source import (
    regenerate_shifted_grid,
    regenerate_sigmas,
)

MODEL_ID = "Efficient-Large-Model/Open-H3-Regenerate-2K"


def test_registry_and_request_contract() -> None:
    info = get_model_info(MODEL_ID)
    assert info.sampling_param_cls is OpenH3RegenerateSamplingParams
    assert info.pipeline_config_cls is OpenH3RegeneratePipelineConfig
    assert get_non_diffusers_pipeline_name(MODEL_ID) == "OpenH3RegeneratePipeline"
    params = OpenH3RegenerateSamplingParams(prompt="p")
    assert (params.task, params.num_inference_steps) == ("t2va", 3)
    assert params.target["short_edge"] == 1440 and params.keep_source_audio
    with pytest.raises(ValueError, match="2- and 3-step"):
        OpenH3RegenerateSamplingParams(prompt="p", num_inference_steps=4)
    OpenH3RegenerateSamplingParams(prompt="p", num_inference_steps=4, start_sigma=0.8)
    with pytest.raises(ValueError, match="source_short_edge"):
        OpenH3RegenerateSamplingParams(prompt="p", source_short_edge=512)


def test_release_sigma_schedules() -> None:
    # open_h3/sigmas.py
    assert regenerate_sigmas(3, None) == [
        0.909375,
        0.8571428571428571,
        0.6666666666666666,
        0.0,
    ]
    assert regenerate_sigmas(2, None) == pytest.approx([0.7, 0.494118, 0.0], abs=1e-6)
    assert regenerate_shifted_grid(0.909375, 4) == pytest.approx(
        [0.909375, 0.841446, 0.732075, 0.526697, 0.0], abs=1e-6
    )


@pytest.mark.parametrize(
    ("short_edge", "max_pixels", "size"),
    [(1440, 2560 * 1440, (2560, 1440)), (1440, None, (1344, 768))],
)
def test_canvas_area_cap(short_edge, max_pixels, size) -> None:
    canonical = minimax_h3_validate_canonical_request(
        task="t2va",
        prompt="p",
        conditions=[],
        target={
            "short_edge": short_edge,
            "aspect_ratio": "16:9",
            "duration_seconds": 5.0,
        },
        seed=0,
        max_pixels=max_pixels,
    )
    shape = minimax_h3_resolve_plan(canonical).shape
    assert (shape["width"], shape["height"]) == size


def test_adaln_curve_branch() -> None:
    table = torch.arange(9 * 8, dtype=torch.float32).reshape(9, 8)
    proj = SimpleNamespace(curve_coords=None)
    model = SimpleNamespace(
        _adaln_curve_table=table, blocks=[SimpleNamespace(adaln_proj=proj)]
    )
    MiniMaxH3DiTModel._set_adaln_curve_coords(model, torch.tensor([0.0, 0.0625]))
    # t * (grid - 1) = 0.5 -> halfway between rows 0 and 1
    torch.testing.assert_close(proj.curve_coords[1], (table[0] + table[1]) / 2)

    weight = torch.randn(6, 4)
    proj.linear = lambda x: (x @ weight.T, None)
    proj.curve_a, proj.curve_b = torch.randn(3, 8), torch.randn(6, 3)
    x = torch.randn(2, 4)
    expected = x @ weight.T + proj.curve_coords @ proj.curve_a.T @ proj.curve_b.T
    torch.testing.assert_close(MiniMaxH3AdalnProj.project_local(proj, x), expected)


def test_latent_upscaler_loads_its_checkpoint_layout() -> None:
    reference = MiniMaxH3LatentUpscaler(
        in_channels=24, in_blocks=2, out_blocks=2, channels=32
    )
    model = MiniMaxH3LatentUpscaler.from_state_dict(reference.state_dict()).eval()
    latent = torch.randn(1, 24, 2, 3, 4)
    out = model.upscale(
        latent,
        target_hw=(6, 8),
        mean=torch.zeros(1, 24, 1, 1, 1),
        std=torch.ones(1, 24, 1, 1, 1),
    )
    assert out.shape == (1, 24, 2, 6, 8)


def test_release_loads_on_the_pinned_base() -> None:
    from sglang.multimodal_gen.runtime.pipelines.minimax_h3_pipeline import (
        OpenH3RegeneratePipeline,
    )

    pipeline = OpenH3RegeneratePipeline
    assert pipeline.base_model == "MiniMaxAI/MiniMax-H3"
    assert len(pipeline.base_revision) == len(pipeline.release_revision) == 40
    # only the add-ons are fetched from the release repo, never its base_model/
    assert pipeline.release_lora_file in pipeline.release_files
    assert all(not name.startswith("base_model/") for name in pipeline.release_files)
    fake = SimpleNamespace(
        release_lora_file=pipeline.release_lora_file, release_path="/release"
    )
    assert (
        OpenH3RegeneratePipeline._release_lora_adapter_path(fake)
        == "/release/lora/open_h3_regenerate_2k_lora.safetensors"
    )
