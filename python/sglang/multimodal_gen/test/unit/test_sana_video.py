from types import SimpleNamespace

import torch

from sglang.multimodal_gen.configs.pipeline_configs.sana_video import (
    SanaVideoPipelineConfig,
)
from sglang.multimodal_gen.configs.sample.sana_video import SanaVideoSamplingParams
from sglang.multimodal_gen.registry import get_model_info
from sglang.multimodal_gen.runtime.models.dits.sana_video import (
    SanaVideoRotaryPosEmbed,
)
from sglang.multimodal_gen.runtime.pipelines.sana_video import (
    select_sana_video_prompt_window,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.latent_preparation import (
    LatentPreparationStage,
)


def test_sana_video_registry_resolution(monkeypatch):
    monkeypatch.setattr(
        "sglang.multimodal_gen.registry.maybe_download_model_index",
        lambda _: {"_class_name": "SanaVideoPipeline"},
    )
    get_model_info.cache_clear()
    model_info = get_model_info("Efficient-Large-Model/SANA-Video_2B_480p_diffusers")

    assert model_info is not None
    assert model_info.pipeline_config_cls is SanaVideoPipelineConfig
    assert model_info.sampling_param_cls is SanaVideoSamplingParams
    get_model_info.cache_clear()


def test_sana_video_pipeline_latent_shape_and_frame_alignment():
    config = SanaVideoPipelineConfig()
    sampling = SanaVideoSamplingParams()

    assert config.adjust_num_frames(81) == 81
    assert config.adjust_num_frames(80) == 77
    batch = SimpleNamespace(height=480, width=832, num_frames=81)
    server_args = SimpleNamespace(pipeline_config=config)
    latent_frames = LatentPreparationStage(
        scheduler=None, transformer=None
    ).adjust_video_length(batch, server_args)
    assert latent_frames == 21
    # LatentPreparationStage applies temporal compression before calling the config.
    assert config.prepare_latent_shape(
        batch, batch_size=2, num_frames=latent_frames
    ) == (
        2,
        16,
        21,
        60,
        104,
    )
    assert config.get_latent_dtype(torch.bfloat16) is torch.float32
    assert not config.enable_autocast
    assert not config.vae_config.load_encoder
    assert config.vae_config.load_decoder
    assert (sampling.width, sampling.height, sampling.num_frames) == (832, 480, 81)
    assert sampling.fps == 16
    assert sampling.num_inference_steps == 50
    assert sampling.guidance_scale == 6.0


def test_select_sana_video_prompt_window_keeps_first_and_tail_tokens():
    tensor = torch.arange(10).view(1, 10, 1)

    selected = select_sana_video_prompt_window(tensor, max_sequence_length=4)

    assert selected.flatten().tolist() == [0, 7, 8, 9]


def test_sana_video_rotary_embeddings_follow_video_token_order():
    rotary = SanaVideoRotaryPosEmbed(
        attention_head_dim=12,
        patch_size=(1, 2, 2),
        max_seq_len=16,
    )

    cos, sin = rotary(torch.zeros(1, 4, 3, 4, 4))

    assert cos.shape == (1, 12, 1, 12)
    assert sin.shape == (1, 12, 1, 12)
    assert torch.isfinite(cos).all()
    assert torch.isfinite(sin).all()
