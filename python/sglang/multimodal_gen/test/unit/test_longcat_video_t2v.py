import json
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def _longcat_sampling_params_cls():
    module = import_module("sglang.multimodal_gen.configs.sample.longcat_video")
    return module.LongCatVideoT2VSamplingParams


def _longcat_pipeline_config_cls():
    module = import_module(
        "sglang.multimodal_gen.configs.pipeline_configs.longcat_video"
    )
    return module.LongCatVideoPipelineConfig


def _longcat_helpers():
    return import_module("sglang.multimodal_gen.runtime.pipelines.longcat_video")


def _longcat_cfg_helpers():
    return import_module(
        "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.longcat_video"
    )


def test_longcat_t2v_sampling_params_defaults():
    params = _longcat_sampling_params_cls()()

    assert params.height == 480
    assert params.width == 832
    assert params.num_frames == 93
    assert params.num_inference_steps == 50
    assert params.guidance_scale == 4.0
    assert params.fps == 16


@pytest.mark.parametrize(
    ("num_frames", "expected"),
    [
        (90, 89),
        (92, 93),
        (93, 93),
    ],
)
def test_longcat_pipeline_config_adjusts_num_frames_to_vae_stride(num_frames, expected):
    config = _longcat_pipeline_config_cls()()

    assert config.adjust_num_frames(num_frames) == expected


def test_longcat_latent_shape_helper_uses_official_stride_and_channels():
    config = _longcat_pipeline_config_cls()()
    batch = SimpleNamespace(height=480, width=832)

    # prepare_latent_shape receives the already-compressed latent frame count
    # (after LatentPreparationStage.adjust_video_length). For 93 pixel frames:
    #   latent_T = (93 - 1) // 4 + 1 = 23
    # The function must NOT re-compress; it uses num_frames as-is.
    assert config.prepare_latent_shape(batch, batch_size=2, num_frames=23) == [
        2,
        16,
        23,
        60,
        104,
    ]


def test_longcat_text_postprocess_keeps_mask_and_adds_embed_view_dim():
    helpers = _longcat_helpers()
    hidden_state = torch.randn(2, 512, 4096)
    attention_mask = torch.ones(2, 512, dtype=torch.long)
    outputs = SimpleNamespace(
        last_hidden_state=hidden_state,
        attention_mask=attention_mask,
    )

    embeds, mask = helpers.longcat_text_postprocess(outputs, None)

    assert embeds.shape == (2, 1, 512, 4096)
    assert mask.shape == (2, 512)
    assert torch.equal(embeds[:, 0], hidden_state)
    assert torch.equal(mask, attention_mask)


def test_longcat_text_postprocess_uses_tokenizer_mask_when_encoder_omits_it():
    helpers = _longcat_helpers()
    hidden_state = torch.randn(1, 512, 4096)
    attention_mask = torch.zeros(1, 512, dtype=torch.long)
    attention_mask[:, :17] = 1
    outputs = SimpleNamespace(
        last_hidden_state=hidden_state,
        attention_mask=None,
    )

    embeds, mask = helpers.longcat_text_postprocess(
        outputs, {"attention_mask": attention_mask}
    )

    assert embeds.shape == (1, 1, 512, 4096)
    assert torch.equal(mask, attention_mask)


def test_longcat_optimized_cfg_helper_matches_st_star_and_sign_flip():
    helpers = _longcat_cfg_helpers()
    noise_cond = torch.tensor([[[[[2.0, -3.0]]]]])
    noise_uncond = torch.tensor([[[[[0.5, 1.0]]]]])
    guidance_scale = 4.0

    before_flip, after_flip = helpers.longcat_optimized_cfg(
        noise_cond=noise_cond,
        noise_uncond=noise_uncond,
        guidance_scale=guidance_scale,
        return_before_sign_flip=True,
    )

    st_star = (noise_cond * noise_uncond).sum() / noise_uncond.square().sum()
    projected_uncond = noise_uncond * st_star
    expected_before_flip = projected_uncond + guidance_scale * (
        noise_cond - projected_uncond
    )
    expected_after_flip = -expected_before_flip

    assert torch.allclose(before_flip, expected_before_flip)
    assert torch.allclose(after_flip, expected_after_flip)


def test_longcat_synthesizes_diffusers_model_index_for_official_layout(tmp_path: Path):
    helpers = _longcat_helpers()
    model_dir = tmp_path / "LongCat-Video"
    (model_dir / "dit").mkdir(parents=True)
    (model_dir / "tokenizer").mkdir()
    (model_dir / "text_encoder").mkdir()
    (model_dir / "vae").mkdir()
    (model_dir / "scheduler").mkdir()
    (model_dir / "model_index.json").write_text(
        json.dumps(
            {
                "_class_name": "LongCatVideoPipeline",
                "_diffusers_version": "0.35.0",
                "dit": ["longcat_video", "LongCatVideoTransformer3DModel"],
            }
        )
    )

    model_index = helpers.synthesize_longcat_model_index(model_dir)

    assert model_index["_class_name"] == "LongCatVideoPipeline"
    assert set(["tokenizer", "text_encoder", "vae", "scheduler", "transformer"]) <= set(
        model_index
    )
    # path field was removed from LongCatComponentSpec (it was dead data — never read by load_modules)
    assert model_index["transformer"]["library"] == "diffusers"
    assert model_index["transformer"]["architecture"] == "LongCatVideoTransformer3DModel"


@pytest.mark.parametrize(
    ("model_path", "expected"),
    [
        ("meituan-longcat/LongCat-Video", True),
        ("/models/LongCat-Video", True),
        ("/models/Avatar-Video", False),
    ],
)
def test_longcat_registry_model_detection(model_path, expected):
    registry = import_module("sglang.multimodal_gen.registry")

    assert registry._is_longcat_video_model_path(model_path) is expected


def test_longcat_registry_resolves_hf_and_local_paths_without_model_index_download():
    registry = import_module("sglang.multimodal_gen.registry")
    server_args = import_module("sglang.multimodal_gen.runtime.server_args")

    for model_path in [
        "meituan-longcat/LongCat-Video",
        "/tmp/models/LongCat-Video",
    ]:
        info = registry.get_model_info(model_path, backend=server_args.Backend.SGLANG)

        assert info.pipeline_cls.__name__ == "LongCatVideoPipeline"
        assert info.sampling_param_cls.__name__ == "LongCatVideoT2VSamplingParams"
        assert info.pipeline_config_cls.__name__ == "LongCatVideoPipelineConfig"


def test_longcat_registry_does_not_resolve_avatar_variant():
    registry = import_module("sglang.multimodal_gen.registry")
    server_args = import_module("sglang.multimodal_gen.runtime.server_args")

    info = registry.get_model_info(
        "/tmp/models/LongCat-Video-Avatar-1.5", backend=server_args.Backend.SGLANG
    )

    assert info is None


# --- Error path tests ---


def test_longcat_sampling_params_rejects_invalid_num_frames_stride():
    """(num_frames - 1) % 4 != 0 should raise ValueError."""
    with pytest.raises(ValueError, match=r"num_frames"):
        _longcat_sampling_params_cls()(num_frames=91)  # (91-1)%4 = 2 ≠ 0


def test_longcat_sampling_params_accepts_valid_num_frames():
    """Valid num_frames values (satisfying (n-1)%4==0) should not raise."""
    for n in [1, 5, 9, 13, 25, 93]:
        _longcat_sampling_params_cls()(num_frames=n)


def test_longcat_latent_shape_no_double_compression():
    """prepare_latent_shape must use num_frames as-is (already latent frames).

    Regression test for the double-compression bug: adjust_video_length already
    converts 25 pixel-frames → 7 latent frames; prepare_latent_shape must NOT
    apply (n-1)//4+1 again.
    """
    config = _longcat_pipeline_config_cls()()
    batch = SimpleNamespace(height=480, width=832)
    shape = config.prepare_latent_shape(batch, batch_size=1, num_frames=7)
    assert shape[2] == 7, (
        f"Expected latent frame dim == 7 (no double compression), got {shape[2]}"
    )


def test_longcat_synthesize_model_index_structure(tmp_path):
    """synthesize_longcat_model_index returns a dict with all required keys."""
    helpers = _longcat_helpers()
    model_dir = tmp_path / "LongCat-Video"
    model_dir.mkdir()
    index = helpers.synthesize_longcat_model_index(model_dir)
    required_keys = {"_class_name", "tokenizer", "text_encoder", "vae", "scheduler", "transformer"}
    assert required_keys <= set(index.keys()), (
        f"Missing keys: {required_keys - set(index.keys())}"
    )
    assert index["_class_name"] == "LongCatVideoPipeline"
