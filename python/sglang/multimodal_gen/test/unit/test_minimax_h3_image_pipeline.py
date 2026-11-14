# SPDX-License-Identifier: Apache-2.0
"""Weight-free CPU integration: real image stages on a 6-step PDD release."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from sglang.multimodal_gen.configs.pipeline_configs.minimax_h3 import (
    MiniMaxH3PipelineConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.minimax_h3_image import (
    MiniMaxH3ImagePipelineConfig,
)
from sglang.multimodal_gen.configs.sample.minimax_h3_image import (
    MiniMaxH3ImageSamplingParams,
)
from sglang.multimodal_gen.registry import _get_config_info
from sglang.multimodal_gen.runtime.entrypoints.utils import prepare_request
from sglang.multimodal_gen.runtime.managers.forward_context import get_forward_context
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.constants import (
    MINIMAX_H3_DENOISE_STATE_EXTRA_KEY as STATE,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.constants import (
    MINIMAX_H3_REFERENCE_IMAGE_ROWS_EXTRA_KEY as REFS,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.constants import (
    MINIMAX_H3_TEXT_EMBEDDINGS_EXTRA_KEY as TEXT,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.packed_tokens import (
    minimax_h3_unpatchify_video_tokens,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.stages.decoding import (
    MiniMaxH3DecodingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.stages.denoising import (
    MiniMaxH3DenoisingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.stages.latent_preparation import (
    MiniMaxH3LatentPreparationStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.stages.timestep_preparation import (
    MiniMaxH3TimestepPreparationStage,
)
from sglang.multimodal_gen.runtime.server_args import (
    get_global_server_args,
    set_global_server_args,
)
from sglang.multimodal_gen.test.unit.test_minimax_h3_image_pipeline_config import (
    normalized_args,
)
from sglang.multimodal_gen.test.unit.test_minimax_h3_image_schedule import (
    write_pdd_release,
)


@pytest.fixture
def server(default_global_server_args, tmp_path, monkeypatch):
    # Stages built without a pipeline load the release's heads from the env var.
    heads = write_pdd_release(tmp_path)
    monkeypatch.setenv("SGLANG_DIFFUSION_MINIMAX_H3_PDD_HEADS", str(heads))
    previous = get_global_server_args()
    args = normalized_args()
    arch = args.pipeline_config.vae_config.arch_config
    arch.latents_mean, arch.latents_std = [1.0] * 24, [2.0] * 24
    # The tiny test references keep their own size for Qwen and the VAE.
    args.pipeline_config.edit_reference_qwen_min_short_edge = 32
    args.pipeline_config.edit_reference_vae_min_short_edge = 32
    set_global_server_args(args)
    try:
        yield args
    finally:
        set_global_server_args(previous)


def make_req(server, task="t2i", seed=17):
    req = prepare_request(
        server,
        MiniMaxH3ImageSamplingParams(
            prompt="p",
            width=64,
            height=32,
            seed=seed,
            # The server's _adjust fills the release's head count; direct
            # construction passes it.
            num_inference_steps=6,
            image_path=["one.png", "two.png"] if task == "edit" else None,
        ),
    )

    def embedding(length):
        return {
            "text_len": length,
            "hidden_states": torch.ones(length, 5120, dtype=torch.bfloat16),
            "text_token_tags": torch.ones(length, dtype=torch.int64),
        }

    req.extra[TEXT] = {"positive": embedding(3)}
    if task == "edit":
        # Clean rows as published by the shared visual stage.
        req.extra[REFS] = {
            "images": [
                {"latent_h": 2, "latent_w": 2, "rows": torch.full((1, 96), 0.3)},
                {"latent_h": 4, "latent_w": 2, "rows": torch.full((2, 96), 0.6)},
            ]
        }
    return req


class ZeroVelocityModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []
        # The image loop requires the release's fused heads.
        self.final_layer = SimpleNamespace(_pdd_heads={})

    def prepare_adaln_plans(self, plans):
        return None

    def forward(self, **kwargs):
        # The real DiT's CUDA attention asserts a forward context.
        get_forward_context()
        positions = kwargs["img_pos_for_infer_output_info"]["position_ids"]
        self.calls.append(kwargs)
        return torch.zeros(len(positions), 96), torch.empty(0, 32)


class RecordingVAE:
    def decode_image_raw(self, x):
        # The real ViT decoder's CUDA attention asserts a forward context.
        get_forward_context()
        self.raw = x.clone()
        return torch.full((1, 3, x.shape[-2] * 16, x.shape[-1] * 16), 0.25)


@pytest.mark.skipif(
    torch.cuda.is_available(),
    reason="CPU-only weight-free test; it mixes CPU and CUDA tensors on GPU hosts",
)
@pytest.mark.parametrize("task", ["t2i", "edit"])
def test_real_denoise_loop_from_stage_state_to_rgb(server, task):
    req = make_req(server, task)
    before = torch.random.get_rng_state().clone()
    MiniMaxH3TimestepPreparationStage().forward(req, server)
    MiniMaxH3LatentPreparationStage().forward(req, server)
    assert torch.equal(before, torch.random.get_rng_state())
    state = req.extra[STATE]
    original = minimax_h3_unpatchify_video_tokens(
        state["initial_video_rows"], latent_shape=(1, 1, 2, 24), patch_size=(1, 2, 2)
    )
    expected = torch.randn(
        (1, 24, 1, 2, 4), generator=torch.Generator().manual_seed(17)
    )
    torch.testing.assert_close(original, expected, rtol=0, atol=0)
    model = ZeroVelocityModel()
    stage = MiniMaxH3DenoisingStage(model)
    assert stage.component_uses(server)[0].target_dtype is None
    stage.forward(req, server)
    # One positive forward per fused head.
    assert len(model.calls) == 6
    assert {int(c["prompt_embeds"].shape[0]) for c in model.calls} == {3}
    assert req.latents.shape == (1, 24, 1, 2, 4)
    assert req.latents.dtype == torch.float32
    assert torch.isfinite(req.latents).all()
    vae = RecordingVAE()
    output = MiniMaxH3DecodingStage(vae, None).forward(req, server)
    assert output.output.shape == (1, 3, 32, 64)
    assert torch.equal(output.output, torch.full((1, 3, 32, 64), 0.25))
    assert torch.equal(vae.raw, req.latents * 2 + 1)
    assert output.audio is None


def _pattern(latent_h, latent_w):
    generator = torch.Generator().manual_seed(3)
    return torch.rand((1, 3, latent_h * 16, latent_w * 16), generator=generator)


class PatternVAE:
    def decode_image_raw(self, x):
        get_forward_context()
        return _pattern(x.shape[-2], x.shape[-1])


@pytest.mark.skipif(
    torch.cuda.is_available(),
    reason="CPU-only weight-free test; it mixes CPU and CUDA tensors on GPU hosts",
)
def test_min_generation_area_generates_the_canvas_and_delivers_the_request(server):
    # 40x20 is below the 2048-pixel area: generated as 64x32, delivered as 40x20.
    req = prepare_request(
        server,
        MiniMaxH3ImageSamplingParams(
            prompt="p",
            width=40,
            height=20,
            seed=17,
            min_generation_area=2048,
            num_inference_steps=6,
        ),
    )
    req.extra[TEXT] = make_req(server).extra[TEXT]
    MiniMaxH3TimestepPreparationStage().forward(req, server)
    MiniMaxH3LatentPreparationStage().forward(req, server)
    # Same initial noise as requesting the 64x32 canvas directly.
    canvas = make_req(server)
    MiniMaxH3TimestepPreparationStage().forward(canvas, server)
    MiniMaxH3LatentPreparationStage().forward(canvas, server)
    assert torch.equal(
        req.extra[STATE]["initial_video_rows"],
        canvas.extra[STATE]["initial_video_rows"],
    )
    MiniMaxH3DenoisingStage(ZeroVelocityModel()).forward(req, server)
    assert req.latents.shape == (1, 24, 1, 2, 4)
    output = MiniMaxH3DecodingStage(PatternVAE(), None).forward(req, server)
    decoded = _pattern(2, 4)[0]
    array = (decoded * 255).clamp(0, 255).to(torch.uint8).permute(1, 2, 0).numpy()
    expected = np.asarray(
        Image.fromarray(array).resize((40, 20), Image.Resampling.LANCZOS)
    )
    assert output.output.shape == (1, 3, 20, 40)
    # The writer's (x * 255).to(uint8) maps the result back to the PIL bytes.
    delivered = (output.output[0] * 255).clamp(0, 255).to(torch.uint8)
    assert np.array_equal(delivered.permute(1, 2, 0).numpy(), expected)


@pytest.mark.parametrize(
    "name,expected",
    [
        ("MiniMaxH3ImagePipeline", MiniMaxH3ImagePipelineConfig),
        ("MiniMaxH3Pipeline", MiniMaxH3PipelineConfig),
    ],
)
def test_registry_disambiguates_image_and_video(tmp_path, name, expected):
    path = tmp_path / "neutral-model-name"
    path.mkdir()
    (path / "transformer").mkdir()
    (path / "model_index.json").write_text(
        json.dumps(
            {
                "_class_name": name,
                "_diffusers_version": "0.35.1",
                "transformer": ["sglang", "MiniMaxH3DiTModel"],
            }
        )
    )
    _get_config_info.cache_clear()
    assert _get_config_info(str(path)).pipeline_config_cls is expected
