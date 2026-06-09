import json
import sys
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
    return import_module(
        "sglang.multimodal_gen.runtime.pipelines.longcat_video_pipeline"
    )


def _longcat_cfg_helpers():
    return import_module(
        "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.longcat_video"
    )


def _longcat_dit_module():
    return import_module("sglang.multimodal_gen.runtime.models.dits.longcat_video")


def _mock_longcat_server_args():
    return SimpleNamespace(
        attention_backend=None,
        pipeline_config=SimpleNamespace(
            dit_config=SimpleNamespace(hidden_size=4096, num_attention_heads=32),
        ),
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
        (96, 97),
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


def _longcat_reference_rope_cache(head_dim: int, grid_size: tuple[int, int, int]):
    num_frames, height, width = grid_size
    dim_t = head_dim - 4 * (head_dim // 6)
    dim_h = 2 * (head_dim // 6)
    dim_w = 2 * (head_dim // 6)
    freqs = []
    for size, dim in [(num_frames, dim_t), (height, dim_h), (width, dim_w)]:
        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, dim, 2).float()[: (dim // 2)] / dim)
        )
        freqs.append(torch.outer(torch.arange(size, dtype=torch.float32), inv_freq))

    freqs_t = freqs[0][:, None, None, :].expand(num_frames, height, width, -1)
    freqs_h = freqs[1][None, :, None, :].expand(num_frames, height, width, -1)
    freqs_w = freqs[2][None, None, :, :].expand(num_frames, height, width, -1)
    freqs = torch.cat((freqs_t, freqs_h, freqs_w), dim=-1).reshape(
        num_frames * height * width, -1
    )
    return torch.cat((freqs.cos(), freqs.sin()), dim=-1)


def _longcat_reference_apply_rope(x: torch.Tensor, cos_sin_cache: torch.Tensor):
    cos, sin = cos_sin_cache.chunk(2, dim=-1)
    x = x.transpose(1, 2)
    x1 = x[..., ::2].float()
    x2 = x[..., 1::2].float()
    cos = cos.unsqueeze(0).unsqueeze(2)
    sin = sin.unsqueeze(0).unsqueeze(2)
    out = torch.stack((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1).flatten(-2)
    return out.transpose(1, 2).type_as(x)


def test_longcat_rope_reuses_shared_nd_embedding_without_changing_math():
    module = _longcat_dit_module()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    head_dim = 128
    grid_size = (2, 3, 4)
    seq_len = grid_size[0] * grid_size[1] * grid_size[2]
    rope = module.RotaryPositionalEmbedding(head_dim)

    q = torch.randn(2, 3, seq_len, head_dim, device=device)
    k = torch.randn(2, 3, seq_len, head_dim, device=device)
    q_out, k_out = rope(q, k, grid_size)

    expected_cache = _longcat_reference_rope_cache(head_dim, grid_size).to(device)
    assert q_out.dtype == q.dtype
    assert k_out.dtype == k.dtype
    assert torch.allclose(rope.freqs_dict[grid_size], expected_cache)
    assert torch.allclose(
        q_out, _longcat_reference_apply_rope(q, expected_cache), rtol=1e-5, atol=1e-6
    )
    assert torch.allclose(
        k_out, _longcat_reference_apply_rope(k, expected_cache), rtol=1e-5, atol=1e-6
    )


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
    assert (
        model_index["transformer"]["architecture"] == "LongCatVideoTransformer3DModel"
    )


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


def test_longcat_is_excluded_from_component_accuracy_suite(monkeypatch):
    testcase_configs = import_module(
        "sglang.multimodal_gen.test.server.testcase_configs"
    )
    monkeypatch.setattr(
        testcase_configs, "_infer_modality_from_model_path", lambda _: "video"
    )
    monkeypatch.setattr(
        testcase_configs,
        "get_default_sampling_params_for_server_args",
        lambda _: testcase_configs.T2V_sampling_params,
    )

    case_module_names = (
        "sglang.multimodal_gen.test.server.gpu_cases",
        "sglang.multimodal_gen.test.server.accuracy_testcase_configs",
    )
    missing = object()
    previous_modules = {
        name: sys.modules.get(name, missing) for name in case_module_names
    }

    try:
        for name in case_module_names:
            sys.modules.pop(name, None)

        gpu_cases = import_module("sglang.multimodal_gen.test.server.gpu_cases")
        accuracy_cases = import_module(
            "sglang.multimodal_gen.test.server.accuracy_testcase_configs"
        )

        longcat_case = next(
            case for case in gpu_cases.ONE_GPU_CASES if case.id == "longcat_video_t2v"
        )

        assert longcat_case.run_component_accuracy_check is False
        assert "longcat_video_t2v" not in {
            case.id for case in accuracy_cases.ACCURACY_ONE_GPU_CASES
        }
    finally:
        for name, previous_module in previous_modules.items():
            if previous_module is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous_module


# --- Error path tests ---


def test_longcat_sampling_params_allows_unaligned_num_frames_before_adjust():
    """Frame stride alignment is handled by pipeline config adjust_num_frames."""
    params = _longcat_sampling_params_cls()(num_frames=91)

    assert params.num_frames == 91


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
    assert (
        shape[2] == 7
    ), f"Expected latent frame dim == 7 (no double compression), got {shape[2]}"


def test_longcat_synthesize_model_index_structure(tmp_path):
    """synthesize_longcat_model_index returns a dict with all required keys."""
    helpers = _longcat_helpers()
    model_dir = tmp_path / "LongCat-Video"
    model_dir.mkdir()
    index = helpers.synthesize_longcat_model_index(model_dir)
    required_keys = {
        "_class_name",
        "tokenizer",
        "text_encoder",
        "vae",
        "scheduler",
        "transformer",
    }
    assert required_keys <= set(
        index.keys()
    ), f"Missing keys: {required_keys - set(index.keys())}"
    assert index["_class_name"] == "LongCatVideoPipeline"


def test_longcat_predict_noise_with_cfg_sets_forward_context_for_non_cfg():
    helpers = _longcat_cfg_helpers()
    fc_mod = import_module("sglang.multimodal_gen.runtime.managers.forward_context")
    base_mod = import_module("sglang.multimodal_gen.runtime.pipelines_core.stages.base")
    selector_mod = import_module(
        "sglang.multimodal_gen.runtime.layers.attention.selector"
    )

    original_get_global_server_args = base_mod.get_global_server_args
    original_selector_get_global_server_args = selector_mod.get_global_server_args
    base_mod.get_global_server_args = _mock_longcat_server_args
    selector_mod.get_global_server_args = base_mod.get_global_server_args

    try:
        stage = helpers.LongCatVideoDenoisingStage(
            transformer=None,
            scheduler=None,
            vae=None,
            pipeline=None,
        )
        batch = SimpleNamespace(
            do_classifier_free_guidance=False,
            guidance_rescale=0.0,
            cfg_normalization=0.0,
        )
        latents = torch.randn(1, 16, 7, 60, 104)
        latent_model_input = latents.clone()
        timestep = torch.tensor([1.0])
        encoder_hidden_states = torch.randn(1, 1, 8, 16)
        encoder_attention_mask = torch.ones(1, 8, dtype=torch.long)
        server_args = SimpleNamespace(
            pipeline_config=SimpleNamespace(slice_noise_pred=lambda noise, _: noise)
        )
        seen = {}

        def fake_model(**kwargs):
            ctx = fc_mod.get_forward_context()
            seen["current_timestep"] = ctx.current_timestep
            seen["attn_metadata"] = ctx.attn_metadata
            seen["forward_batch"] = ctx.forward_batch
            return kwargs["hidden_states"] + 1

        noise_pred = stage._predict_noise_with_cfg(
            current_model=fake_model,
            latent_model_input=latent_model_input,
            timestep=timestep,
            batch=batch,
            timestep_index=3,
            attn_metadata={"meta": "non_cfg"},
            target_dtype=torch.bfloat16,
            current_guidance_scale=4.0,
            image_kwargs={},
            pos_cond_kwargs={
                "encoder_hidden_states": encoder_hidden_states,
                "encoder_attention_mask": encoder_attention_mask,
            },
            neg_cond_kwargs={},
            server_args=server_args,
            guidance=torch.tensor([4.0]),
            latents=latents,
        )

        assert seen["current_timestep"] == 3
        assert seen["attn_metadata"] == {"meta": "non_cfg"}
        assert seen["forward_batch"] is batch
        assert torch.allclose(noise_pred, -(latent_model_input + 1))
    finally:
        base_mod.get_global_server_args = original_get_global_server_args
        selector_mod.get_global_server_args = original_selector_get_global_server_args


def test_longcat_predict_noise_with_cfg_sets_forward_context_for_cfg():
    helpers = _longcat_cfg_helpers()
    fc_mod = import_module("sglang.multimodal_gen.runtime.managers.forward_context")
    base_mod = import_module("sglang.multimodal_gen.runtime.pipelines_core.stages.base")
    selector_mod = import_module(
        "sglang.multimodal_gen.runtime.layers.attention.selector"
    )

    original_get_global_server_args = base_mod.get_global_server_args
    original_selector_get_global_server_args = selector_mod.get_global_server_args
    base_mod.get_global_server_args = _mock_longcat_server_args
    selector_mod.get_global_server_args = base_mod.get_global_server_args

    try:
        stage = helpers.LongCatVideoDenoisingStage(
            transformer=None,
            scheduler=None,
            vae=None,
            pipeline=None,
        )
        batch = SimpleNamespace(
            do_classifier_free_guidance=True,
            guidance_rescale=0.0,
            cfg_normalization=0.0,
        )
        latents = torch.randn(1, 16, 7, 60, 104)
        latent_model_input = latents.clone()
        timestep = torch.tensor([1.0])
        encoder_hidden_states = torch.randn(1, 1, 8, 16)
        encoder_attention_mask = torch.ones(1, 8, dtype=torch.long)
        server_args = SimpleNamespace(
            pipeline_config=SimpleNamespace(slice_noise_pred=lambda noise, _: noise)
        )
        seen = {}

        def fake_model(**kwargs):
            ctx = fc_mod.get_forward_context()
            seen["current_timestep"] = ctx.current_timestep
            seen["attn_metadata"] = ctx.attn_metadata
            seen["forward_batch"] = ctx.forward_batch
            return torch.zeros_like(kwargs["hidden_states"])

        noise_pred = stage._predict_noise_with_cfg(
            current_model=fake_model,
            latent_model_input=latent_model_input,
            timestep=timestep,
            batch=batch,
            timestep_index=5,
            attn_metadata={"meta": "cfg"},
            target_dtype=torch.bfloat16,
            current_guidance_scale=4.0,
            image_kwargs={},
            pos_cond_kwargs={
                "encoder_hidden_states": encoder_hidden_states,
                "encoder_attention_mask": encoder_attention_mask,
            },
            neg_cond_kwargs={
                "encoder_hidden_states": encoder_hidden_states,
                "encoder_attention_mask": encoder_attention_mask,
            },
            server_args=server_args,
            guidance=torch.tensor([4.0]),
            latents=latents,
        )

        assert seen["current_timestep"] == 5
        assert seen["attn_metadata"] == {"meta": "cfg"}
        assert seen["forward_batch"] is batch
        assert noise_pred.shape == latents.shape
    finally:
        base_mod.get_global_server_args = original_get_global_server_args
        selector_mod.get_global_server_args = original_selector_get_global_server_args
