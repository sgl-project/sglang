from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.multimodal_gen.runtime.layers.layernorm import (
    _can_use_npu_fused_scale_shift,
)
from sglang.multimodal_gen.runtime.models.dits import glm_image as glm_model
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages import (
    glm_image as glm_stage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.glm_image import (
    GlmImageAR,
    GlmImageBeforeDenoisingStage,
)


class _DummyVisionLanguageEncoder:
    device = torch.device("cpu")


class _RecordingGlmImageAR(GlmImageAR):
    def __init__(self):
        super().__init__(
            processor=None, vision_language_encoder=_DummyVisionLanguageEncoder()
        )
        self.initial_seeds = []

    def generate_prior_tokens(self, **kwargs):
        self.initial_seeds.append(torch.initial_seed())
        output_idx = len(self.initial_seeds)
        return torch.full((1, 4), output_idx, dtype=torch.long), None


class _DummySchedulerConfig(dict):
    num_train_timesteps = 1000


class _DummyScheduler:
    config = _DummySchedulerConfig(
        base_image_seq_len=256,
        base_shift=0.25,
        max_shift=0.75,
    )

    def set_timesteps(self, timesteps=None, sigmas=None, device=None, **kwargs):
        self.timesteps = torch.as_tensor(timesteps, device=device)


class _RecordingBeforeDenoisingStage(GlmImageBeforeDenoisingStage):
    def __init__(self):
        self.transformer = SimpleNamespace(
            config=SimpleNamespace(
                in_channels=4,
                num_layers=1,
                patch_size=2,
            )
        )
        self.vae = SimpleNamespace(config=SimpleNamespace(block_out_channels=[1]))
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.scheduler = _DummyScheduler()

    def encode_prompt(self, *args, **kwargs):
        return torch.ones(1, 3, 5), torch.zeros(1, 3, 5)

    def prepare_latents(
        self, batch_size, num_channels_latents, height, width, **kwargs
    ):
        return torch.zeros(batch_size, num_channels_latents, height // 2, width // 2)


class _NpuForwardProbe(glm_model.GlmImageTransformer2DModel):
    def __init__(self):
        pass

    def forward(self, **kwargs):
        batch_size = kwargs["hidden_states"].shape[0]
        if batch_size > 1:
            return super().forward(**kwargs)
        marker = kwargs["prior_token_id"].float().view(1, 1, 1, 1)
        return marker.expand(1, 1, 2, 2)


def test_npu_fused_scale_shift_accepts_broadcast_modulation_shapes():
    hidden_size = 8
    x = torch.zeros(1, 4, hidden_size)

    for shape in ((), (1,), (hidden_size,), (1, hidden_size), (1, 1, hidden_size)):
        scale = torch.zeros(shape)
        shift = torch.zeros(shape)
        assert _can_use_npu_fused_scale_shift(x, shift, scale)

    assert not _can_use_npu_fused_scale_shift(
        x, torch.zeros(1, 2, hidden_size), torch.zeros(1, 2, hidden_size)
    )


def test_ar_stage_generates_one_prior_per_requested_output():
    stage = _RecordingGlmImageAR()
    batch = SimpleNamespace(
        prompt="a cat",
        height=64,
        width=64,
        image_path=None,
        num_outputs_per_prompt=2,
        seed=11,
    )

    with patch.object(
        glm_stage, "get_local_torch_device", return_value=torch.device("cpu")
    ):
        result = stage.forward(batch, SimpleNamespace())

    assert result.prior_token_id.shape == (2, 4)
    assert result.prior_token_id.tolist() == [[1, 1, 1, 1], [2, 2, 2, 2]]
    assert stage.initial_seeds == [11, 12]


def test_before_denoising_expands_latents_and_conditions_for_requested_outputs():
    stage = _RecordingBeforeDenoisingStage()
    batch = SimpleNamespace(
        prompt="a cat",
        height=64,
        width=64,
        image_path=None,
        guidance_scale=4.5,
        num_inference_steps=2,
        num_outputs_per_prompt=2,
        seed=7,
        prior_token_id=torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]),
        prior_token_image_ids=None,
    )

    with patch.object(
        glm_stage, "get_local_torch_device", return_value=torch.device("cpu")
    ):
        result = stage.forward(batch, SimpleNamespace())

    assert result.latents.shape[0] == 2
    assert result.prompt_embeds[0].shape[0] == 2
    assert result.negative_prompt_embeds[0].shape[0] == 2
    assert result.prior_token_id.shape == (2, 4)
    assert result.prior_token_drop_cond.shape == (2, 4)
    assert result.target_size.shape == (2, 2)
    assert result.crop_coords.shape == (2, 2)


def test_npu_transformer_fallback_runs_each_batch_item_independently():
    model = _NpuForwardProbe()
    hidden_states = torch.zeros(2, 1, 2, 2)
    encoder_hidden_states = torch.zeros(2, 3, 4)
    prior_token_id = torch.tensor([[3], [9]])
    attention_mask = torch.tensor([[True, True, False], [True, False, False]])

    with patch.object(glm_model.current_platform, "is_npu", return_value=True):
        output = glm_model.GlmImageTransformer2DModel.forward(
            model,
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            prior_token_id=prior_token_id,
            prior_token_drop=torch.zeros_like(prior_token_id, dtype=torch.bool),
            timestep=torch.ones(2),
            target_size=torch.ones(2, 2),
            crop_coords=torch.zeros(2, 2),
            attention_mask=attention_mask,
        )

    assert output.shape == (2, 1, 2, 2)
    assert output[:, 0, 0, 0].tolist() == [3.0, 9.0]
