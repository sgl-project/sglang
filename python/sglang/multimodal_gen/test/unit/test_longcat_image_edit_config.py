"""Unit tests for LongCat-Image-Edit pipeline config hooks (CPU only)."""

import types

import pytest
import torch

from sglang.multimodal_gen.configs.pipeline_configs.longcat_image import (
    LongCatImageEditPipelineConfig,
    LongCatImagePipelineConfig,
    _calculate_edit_dimensions,
)
from sglang.multimodal_gen.configs.pipeline_configs.base import ModelTaskType


@pytest.fixture
def edit_config():
    return LongCatImageEditPipelineConfig()


def _make_batch(**kwargs):
    return types.SimpleNamespace(**kwargs)


def test_edit_dimensions_match_diffusers_formula():
    import math

    for w, h in [(1000, 1500), (1024, 1024), (1920, 1080), (333, 777)]:
        ratio = w / h
        got_w, got_h = _calculate_edit_dimensions(1024 * 1024, ratio)
        # diffusers reference (ceil to /16)
        ref_w = math.sqrt(1024 * 1024 * ratio)
        ref_h = ref_w / ratio
        ref_w = ref_w if ref_w % 16 == 0 else (ref_w // 16 + 1) * 16
        ref_h = ref_h if ref_h % 16 == 0 else (ref_h // 16 + 1) * 16
        assert (got_w, got_h) == (int(ref_w), int(ref_h))
        assert got_w % 16 == 0 and got_h % 16 == 0


def test_edit_config_task_type_and_generator(edit_config):
    assert edit_config.task_type == ModelTaskType.I2I
    assert edit_config.generator_device == "cpu"


def test_slice_noise_pred_drops_reference_tokens(edit_config):
    latents = torch.zeros(1, 100, 64)
    noise = torch.arange(200, dtype=torch.float32).view(1, 200, 1).expand(1, 200, 64)
    sliced = edit_config.slice_noise_pred(noise, latents)
    assert sliced.shape == (1, 100, 64)
    assert torch.equal(sliced, noise[:, :100])


def test_maybe_prepare_latent_ids_returns_none(edit_config):
    assert edit_config.maybe_prepare_latent_ids(torch.zeros(1, 16, 8, 8)) is None


def test_preprocess_vae_encode_squeezes_frames_dim(edit_config):
    image_5d = torch.zeros(1, 3, 1, 64, 64)
    assert edit_config.preprocess_vae_encode(image_5d, None).shape == (1, 3, 64, 64)
    image_4d = torch.zeros(1, 3, 64, 64)
    assert edit_config.preprocess_vae_encode(image_4d, None).shape == (1, 3, 64, 64)


def test_postprocess_image_latent_packs_2x2(edit_config):
    latent = torch.randn(1, 16, 8, 8)
    batch = _make_batch(batch_size=1)
    packed = edit_config.postprocess_image_latent(latent, batch)
    assert packed.shape == (1, 16, 64)  # (8//2 * 8//2, 16*4)


def test_edit_img_ids_modalities_and_offset(edit_config):
    # 1024x1024 -> latent 128x128 -> packed grid 64x64 -> 4096 tokens per image
    batch = _make_batch(height=1024, width=1024)
    num_token = 600  # text length including VL image tokens
    img_ids = edit_config._edit_img_ids(batch, num_token, torch.device("cpu"))
    assert img_ids.shape == (2 * 64 * 64, 3)
    noisy_ids, ref_ids = img_ids[: 64 * 64], img_ids[64 * 64 :]
    assert torch.all(noisy_ids[:, 0] == 1)
    assert torch.all(ref_ids[:, 0] == 2)
    # same grid layout, offset by text length
    assert torch.equal(noisy_ids[:, 1:], ref_ids[:, 1:])
    assert noisy_ids[0, 1].item() == num_token
    assert noisy_ids[0, 2].item() == num_token
    assert noisy_ids[-1, 1].item() == num_token + 63
    assert noisy_ids[-1, 2].item() == num_token + 63


def test_prepare_pos_cond_kwargs_uses_prompt_embeds_length(edit_config):
    prompt_embeds = torch.zeros(1, 700, 3072)
    batch = _make_batch(height=512, width=512, prompt_embeds=[prompt_embeds])
    kwargs = edit_config.prepare_pos_cond_kwargs(
        batch, torch.device("cpu"), None, torch.float32
    )
    txt_ids, img_ids = kwargs["txt_ids"], kwargs["img_ids"]
    assert txt_ids.shape == (700, 3)
    assert torch.all(txt_ids[:, 0] == 0)
    # 512 -> latent 64 -> packed grid 32x32
    assert img_ids.shape == (2 * 32 * 32, 3)
    assert img_ids[0, 1].item() == 700


def test_t2i_config_unchanged():
    t2i = LongCatImagePipelineConfig()
    assert t2i.task_type == ModelTaskType.T2I
    # T2I still slices nothing and builds latent ids at preparation time
    noise = torch.zeros(1, 10, 64)
    assert t2i.slice_noise_pred(noise, torch.zeros(1, 5, 64)).shape == (1, 10, 64)
    ids = t2i.maybe_prepare_latent_ids(torch.zeros(1, 16, 8, 8))
    assert ids is not None and ids.shape == (16, 3)
