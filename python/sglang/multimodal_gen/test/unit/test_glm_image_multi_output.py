from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.multimodal_gen.configs.models.dits.glmimage import GlmImageDitConfig
from sglang.multimodal_gen.configs.pipeline_configs.glm_image import (
    GlmImagePipelineConfig,
)
from sglang.multimodal_gen.runtime.layers.layernorm import (
    _can_use_npu_fused_scale_shift,
)
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.models.dits import glm_image as glm_model
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages import (
    glm_image as glm_stage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.glm_image import (
    GlmImageAR,
    GlmImageBeforeDenoisingStage,
)
from sglang.multimodal_gen.test.unit.host_sync_guard import forbid_host_syncs

# the DiT's fused kernels need a GPU on a GPU platform; elsewhere run on the CPU
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        return torch.full((1, 4), output_idx, dtype=torch.long), None, None


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
        self.text_encoder = None
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
        extra={},
    )

    with patch.object(
        glm_stage.current_platform,
        "get_local_torch_device",
        return_value=torch.device("cpu"),
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
        glm_stage.current_platform,
        "get_local_torch_device",
        return_value=torch.device("cpu"),
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


def _tiny_glm_dit():
    config = GlmImageDitConfig()
    # a 256-wide hidden state keeps CUDA's fused kernels on supported widths
    for key, value in dict(
        num_layers=2,
        attention_head_dim=128,
        num_attention_heads=2,
        condition_dim=8,
        prior_vq_quantizer_codebook_size=32,
        text_embed_dim=24,
        time_embed_dim=16,
    ).items():
        setattr(config.arch_config, key, value)
    model = glm_model.GlmImageTransformer2DModel(config, {})
    generator = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for param in model.parameters():
            param.copy_(0.2 * torch.randn(param.shape, generator=generator))
    return model


@torch.no_grad()
def test_prior_token_drop_does_not_sync(single_process_model_parallel):
    """Dropping the unconditional branch's prior tokens is a select: a boolean-mask
    write runs a nonzero, which blocks the host on every forward."""
    model = _tiny_glm_dit().to(device=_DEVICE, dtype=torch.bfloat16)
    generator = torch.Generator().manual_seed(1)
    inputs = dict(
        hidden_states=torch.randn(1, 16, 4, 4, generator=generator),
        encoder_hidden_states=torch.randn(1, 3, 24, generator=generator),
        prior_token_id=torch.randint(0, 32, (1, 4), generator=generator),
        prior_token_drop=torch.tensor([[True, False, True, False]]),
        timestep=torch.tensor([500.0]),
        target_size=torch.tensor([[64.0, 64.0]]),
        crop_coords=torch.zeros(1, 2),
    )
    inputs = {
        key: value.to(_DEVICE, torch.bfloat16 if value.is_floating_point() else None)
        for key, value in inputs.items()
    }
    with set_forward_context(current_timestep=0, attn_metadata=None):
        # the first call verifies the fused kernels against eager, once
        eager = model(**inputs)
        with forbid_host_syncs():
            model(**inputs)
        # the graph path zeroes the same rows with masked_fill
        with patch.object(glm_model, "is_in_breakable_cuda_graph", lambda: True):
            graphed = model(**inputs)
    torch.testing.assert_close(eager, graphed, atol=0, rtol=0)


def test_decode_latent_stats_are_uploaded_once():
    # decode scales with the first request's device tensors: a per-request
    # upload would stall the host until the denoising loop drains
    config = GlmImagePipelineConfig()
    vae_config = config.vae_config
    vae_config.latents_mean = tuple(0.25 * i - 2.0 for i in range(16))
    vae_config.latents_std = tuple(0.5 + 0.125 * i for i in range(16))
    device = torch.device("cpu")
    first = config.get_decode_scale_and_shift(device, torch.bfloat16, None)
    second = config.get_decode_scale_and_shift(device, torch.bfloat16, None)
    mean = torch.tensor(vae_config.latents_mean).reshape(1, 16, 1, 1)
    std = torch.tensor(vae_config.latents_std).reshape(1, 16, 1, 1)
    torch.testing.assert_close(
        first, (1.0 / std.to(torch.bfloat16), mean.to(torch.bfloat16)), atol=0, rtol=0
    )
    assert first[0] is second[0] and first[1] is second[1]
