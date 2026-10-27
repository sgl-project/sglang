# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace

import pytest
import torch

from sglang.multimodal_gen.configs.models.dits.moss_image import (
    MOSSImageArchConfig,
    MOSSImageDiTConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.moss_image import (
    MOSSImagePipelineConfig,
)
from sglang.multimodal_gen.configs.sample.moss_image import MOSSImageSamplingParams
from sglang.multimodal_gen.runtime.distributed.cfg_policy import CFGPolicy
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.models.dits.moss_image import (
    MOSSImageTransformer,
    _RoPE,
)
from sglang.multimodal_gen.runtime.models.schedulers.moss_image import (
    MOSSImageScheduler,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.decoding import (
    scale_and_shift_latents,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.input_validation import (
    InputValidationStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.moss_image import (
    MOSSImageBeforeDenoisingStage,
    MOSSImageDenoisingStage,
)


@pytest.fixture(autouse=True)
def cpu_attention_backend(monkeypatch):
    from sglang.multimodal_gen.runtime.server_args import get_global_server_args

    monkeypatch.setattr(get_global_server_args(), "attention_backend", "torch_sdpa")


def test_v2_schedule_and_shift_override():
    scheduler = MOSSImageScheduler(shift=1)
    scheduler.set_shift(5)
    scheduler.set_timesteps(5)
    # Golden V2 nearest-training-grid values from the standalone reference.
    torch.testing.assert_close(
        scheduler.timesteps,
        torch.tensor([1000.0, 938.0, 835.0, 630.0, 24.0]),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        scheduler.sigmas,
        torch.tensor([1.0, 0.938, 0.835, 0.630, 0.024, 0.0]),
        rtol=0,
        atol=0,
    )
    sample = torch.ones(1, 8, 2, 2)
    scheduler.step(torch.ones_like(sample), scheduler.timesteps[0], sample)
    scheduler.set_timesteps(5)
    assert scheduler.step_index is None
    with pytest.raises(ValueError):
        scheduler.set_shift(0)


@pytest.mark.parametrize("invalid", [None, 22, [True], [-1], [42], [1.5], ["22"]])
def test_formal_skip_layer_validation(invalid):
    from sglang.multimodal_gen.runtime.models.dits.moss_image import (
        validate_skip_layers,
    )

    with pytest.raises(ValueError, match="skip_layers"):
        validate_skip_layers(invalid, 42)


def test_unsupported_quantization_is_not_silently_ignored():
    with pytest.raises(NotImplementedError, match="quantization"):
        MOSSImageTransformer(MOSSImageDiTConfig(), quant_config=object())


def test_formal_guidance_is_unconditional_only_and_request_scoped():
    from sglang.multimodal_gen.configs.sample.moss_image import MOSSImageSamplingParams

    config = MOSSImagePipelineConfig()
    default = MOSSImageSamplingParams()
    disabled = MOSSImageSamplingParams(uncond_skip_layers=[])
    for params, expected in ((default, (22,)), (disabled, ()), (default, (22,))):
        batch = SimpleNamespace(sampling_params=params)
        positive = config.prepare_pos_cond_kwargs(batch, "cpu", None, torch.bfloat16)
        negative = config.prepare_neg_cond_kwargs(batch, "cpu", None, torch.bfloat16)
        assert "skip_layers" not in positive
        assert negative["skip_layers"] == expected


@pytest.mark.parametrize(
    "extra",
    [
        {"uncond_skip_layers": [0, 41]},
        {"uncond_skip_layers": []},
        {"uncond_skip_layers": [22], "slg_start": 0.01, "slg_stop": 1.0},
        {
            "uncond_skip_layers": [],
            "slg_layers": [22],
            "slg_scale": 3.0,
            "slg_start": 0.01,
            "slg_stop": 1.0,
        },
    ],
)
@pytest.mark.parametrize("nested", [False, True])
def test_image_api_preserves_model_owned_guidance_fields(extra, nested):
    from sglang.multimodal_gen.configs.sample.moss_image import MOSSImageSamplingParams
    from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
    from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import (
        ImageGenerationsRequest,
    )
    from sglang.multimodal_gen.runtime.entrypoints.openai.utils import (
        request_model_kwargs,
    )

    request = ImageGenerationsRequest(
        prompt="example", **({"extra_params": extra} if nested else extra)
    )
    kwargs = request_model_kwargs(request, MOSSImageSamplingParams, "image")
    assert kwargs == extra
    params = MOSSImageSamplingParams(**kwargs)
    for name, value in extra.items():
        assert getattr(params, name) == value
    assert request_model_kwargs(request, SamplingParams, "image") == {}


def test_dit_skips_requested_layer_and_restores_full_forward():
    config = MOSSImageDiTConfig(
        arch_config=MOSSImageArchConfig(
            dim=64,
            ffn_dim=128,
            num_heads=2,
            num_layers=2,
            text_dim=32,
            freq_dim=16,
            latent_channels=8,
        )
    )
    torch.manual_seed(0)
    model = MOSSImageTransformer(config).eval()
    calls = []
    hooks = [
        layer.register_forward_pre_hook(lambda module, args, i=i: calls.append(i))
        for i, layer in enumerate(model.layers)
    ]
    latent = torch.randn(1, 8, 2, 3)
    text = torch.randn(1, 3, 32)
    with torch.no_grad(), set_forward_context(0, None):
        expected = model(latent, torch.tensor(500.0), text)
        skipped = model(latent, torch.tensor(500.0), text, skip_layers=[1])
        restored = model(latent, torch.tensor(500.0), text)
    for hook in hooks:
        hook.remove()
    assert calls == [0, 1, 0, 0, 1]
    assert torch.equal(expected, restored)
    assert not torch.equal(expected, skipped)


@pytest.mark.parametrize("name", ["Base", "SFT"])
def test_model_identifiers_resolve_native_configs(name, monkeypatch, tmp_path):
    """Resolve registered model IDs and local snapshots through native configs."""
    from sglang.multimodal_gen import registry

    config = {"_class_name": "MOSSImagePipeline", "_diffusers_version": "0.38.0"}
    monkeypatch.setattr(registry, "maybe_download_model_index", lambda _: config)
    for path in (
        f"OpenMOSS-Team/MOSS-Image-{name}",
        str(tmp_path / f"MOSS-Image-{name}"),
    ):
        info = registry.get_model_info(path, backend="sglang")
        assert info.pipeline_cls.__name__ == "MOSSImagePipeline"
        assert info.pipeline_config_cls is MOSSImagePipelineConfig


def test_rope_keeps_fp32_derived_buffer_after_assign_loading():
    rope = _RoPE(128, 10000)
    bands = rope.bands.clone()
    rope.to(torch.bfloat16)
    assert rope.bands.dtype == torch.float32
    assert torch.equal(rope.bands, bands)
    rope.load_state_dict(
        {"bands": torch.nn.Parameter(bands.bfloat16(), requires_grad=False)},
        assign=True,
    )
    assert "bands" in dict(rope.named_buffers())
    assert "bands" not in dict(rope.named_parameters())
    assert torch.equal(rope.bands, bands)


def test_joint_attention_ignores_padded_text_tokens():
    torch.manual_seed(0)
    config = MOSSImageDiTConfig(
        arch_config=MOSSImageArchConfig(
            dim=64,
            ffn_dim=128,
            num_heads=2,
            num_layers=2,
            text_dim=32,
            freq_dim=16,
            latent_channels=8,
        )
    )
    model = MOSSImageTransformer(config).eval()
    latents = torch.randn(1, 8, 2, 3)
    text = torch.randn(1, 3, 32)
    padded = torch.cat([text, torch.randn(1, 5, 32) * 1000], 1)
    with torch.no_grad(), set_forward_context(0, None):
        unpadded = model(latents, torch.tensor([630.0]), text)
        masked = model(
            latents,
            torch.tensor([630.0]),
            padded,
            torch.tensor([[1, 1, 1, 0, 0, 0, 0, 0]], dtype=torch.bool),
        )
        assert unpadded.abs().max() > 0
        assert torch.equal(unpadded, masked)
        with pytest.raises(ValueError, match="at least one valid text token"):
            model(
                latents,
                torch.tensor([630.0]),
                text,
                torch.zeros(1, 3, dtype=torch.bool),
            )


def test_vae_bn_denormalization_precedes_channel_major_unpacking():
    config = MOSSImagePipelineConfig()
    latents = torch.arange(48, dtype=torch.bfloat16).reshape(1, 8, 2, 3)
    mean = torch.arange(8, dtype=torch.float32)
    var = torch.full((8,), 4.0)
    vae = SimpleNamespace(
        bn=SimpleNamespace(running_mean=mean, running_var=var),
        config=SimpleNamespace(arch_config=SimpleNamespace(batch_norm_eps=0.0)),
    )
    scaled = scale_and_shift_latents(
        latents, SimpleNamespace(pipeline_config=config), vae
    )
    actual = config.preprocess_decoding(scaled, vae=vae)
    assert actual.dtype == torch.float32
    expected = torch.empty(1, 2, 4, 6)
    for c in range(2):
        for h in range(2):
            for w in range(3):
                for i in range(2):
                    for j in range(2):
                        k = 4 * c + 2 * i + j
                        expected[0, c, 2 * h + i, 2 * w + j] = (
                            2 * latents[0, k, h, w] + mean[k]
                        )
    assert torch.equal(actual, expected)


def test_encoder_respects_active_sdpa_policy():
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from transformers import BatchEncoding

    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.moss_image import (
        MOSSImageBeforeDenoisingStage,
    )

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs["enable_thinking"] and kwargs["add_generation_prompt"]
            return messages[0]["content"]

        def __call__(self, prompt, **kwargs):
            assert kwargs["max_length"] == 4096
            return BatchEncoding({"input_ids": torch.ones(1, 2, dtype=torch.long)})

    class Encoder:
        def __call__(self, **kwargs):
            assert not torch.backends.cuda.cudnn_sdp_enabled()
            assert torch._C._get_sdp_priority_order() == expected_priority
            return SimpleNamespace(last_hidden_state=torch.ones(1, 2, 4))

    stage = MOSSImageBeforeDenoisingStage(Encoder(), Tokenizer(), None)
    with sdpa_kernel(
        [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH],
        set_priority=True,
    ):
        expected_priority = torch._C._get_sdp_priority_order()
        for _ in range(2):
            stage._encode_prompt(["a prompt"], torch.device("cpu"))
            assert not torch.backends.cuda.cudnn_sdp_enabled()
            assert torch._C._get_sdp_priority_order() == expected_priority


@pytest.mark.parametrize(
    "library", ["pipeline_moss_image", "diffusers", "another_pipeline"]
)
def test_pipeline_uses_generic_component_loaders(tmp_path, monkeypatch, library):
    from sglang.multimodal_gen.runtime.loader.component_loaders.component_loader import (
        ComponentLoader,
    )
    from sglang.multimodal_gen.runtime.loader.component_loaders.scheduler_loader import (
        SchedulerLoader,
    )
    from sglang.multimodal_gen.runtime.loader.component_loaders.text_encoder_loader import (
        TextEncoderLoader,
    )
    from sglang.multimodal_gen.runtime.loader.component_loaders.transformer_loader import (
        TransformerLoader,
    )
    from sglang.multimodal_gen.runtime.loader.component_loaders.vae_loader import (
        VAELoader,
    )
    from sglang.multimodal_gen.runtime.pipelines.moss_image import MOSSImagePipeline
    from sglang.multimodal_gen.runtime.pipelines_core import composed_pipeline_base

    model_index = {
        "_class_name": "MOSSImagePipeline",
        "_diffusers_version": "0.38.0",
        "transformer": [library, "MOSSImageTransformer"],
        "scheduler": [library, "MOSSImageScheduler"],
        "text_encoder": ["transformers", "Qwen3_5Model"],
        "vae": ["diffusers", "AutoencoderKLFlux2"],
    }
    for name in ("transformer", "scheduler", "text_encoder", "vae"):
        (tmp_path / name).mkdir()
    source = json.dumps(model_index)
    index_path = tmp_path / "model_index.json"
    index_path.write_text(source)
    monkeypatch.setattr(
        composed_pipeline_base, "maybe_download_model", lambda path, **kwargs: path
    )
    pipeline = object.__new__(MOSSImagePipeline)
    pipeline.model_path = str(tmp_path)
    pipeline.server_args = SimpleNamespace(model_subfolder=None, revision=None)
    config = pipeline._load_config()
    expected_library = "diffusers" if library == "pipeline_moss_image" else library
    assert not MOSSImagePipeline.component_loaders
    for name, loader in (
        ("transformer", TransformerLoader),
        ("scheduler", SchedulerLoader),
    ):
        assert config[name] == [expected_library, model_index[name][1]]
        if expected_library == "diffusers":
            assert (
                type(ComponentLoader.for_component_type(name, *config[name])) is loader
            )
        else:
            with pytest.raises(AssertionError, match="must be loaded from diffusers"):
                ComponentLoader.for_component_type(name, *config[name])
        with pytest.raises(AssertionError, match="must be loaded from diffusers"):
            ComponentLoader.for_component_type(name, "pipeline_moss_image")

    assert config["text_encoder"] == model_index["text_encoder"]
    assert config["vae"] == model_index["vae"]
    assert index_path.read_text() == source
    assert (
        composed_pipeline_base.ComposedPipelineBase._load_config(pipeline)
        == model_index
    )
    assert (
        type(ComponentLoader.for_component_type("text_encoder", "transformers"))
        is TextEncoderLoader
    )
    assert type(ComponentLoader.for_component_type("vae", "diffusers")) is VAELoader
    with pytest.raises(AssertionError, match="text_encoder must be loaded"):
        ComponentLoader.for_component_type("text_encoder", "pipeline_moss_image")


def test_request_shift_and_text_limit_do_not_leak(monkeypatch):
    from sglang.multimodal_gen.configs.sample.moss_image import MOSSImageSamplingParams
    from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages import (
        moss_image,
    )

    monkeypatch.setattr(
        moss_image, "get_local_torch_device", lambda: torch.device("cpu")
    )
    template = MOSSImageScheduler(shift=5)
    stage = moss_image.MOSSImageBeforeDenoisingStage(None, None, template)
    limits = []

    def encode(prompts, device, max_sequence_length):
        limits.append(max_sequence_length)
        return torch.zeros(1, 2, 4), torch.ones(1, 2, dtype=torch.bool)

    monkeypatch.setattr(stage, "_encode_prompt", encode)
    args = SimpleNamespace(
        pipeline_config=MOSSImagePipelineConfig(), enable_cfg_parallel=False
    )
    for shift in (3.0, None):
        params = MOSSImageSamplingParams(
            prompt="test",
            height=16,
            width=16,
            num_inference_steps=5,
            guidance_scale=1.0,
            flow_shift=shift,
            max_sequence_length=128,
            seed=0,
        )
        batch = Req(sampling_params=params)
        batch.sampling_params.generator_device = "cpu"
        InputValidationStage().forward(batch, args)
        batch = stage.forward(batch, args)
        assert batch.scheduler.config.shift == (3.0 if shift is not None else 5.0)
        assert template.config.shift == 5.0
    assert limits == [128, 128]


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_prepared_rope_matches_original_rotation(dtype):
    from sglang.multimodal_gen.runtime.models.dits.moss_image import (
        _prepare_rope,
        _rotate,
    )

    torch.manual_seed(0)
    x = torch.randn(1, 15, 2, 32).to(dtype)
    packed = _RoPE(32, 10000)(3, 4, 3, torch.device("cpu")).to(dtype)
    sin, cos = packed.chunk(2, -1)
    a, b = x.chunk(2, -1)
    expected = torch.cat(
        (
            a * cos.unsqueeze(-2) - b * sin.unsqueeze(-2),
            b * cos.unsqueeze(-2) + a * sin.unsqueeze(-2),
        ),
        -1,
    )
    assert torch.equal(_rotate(x, _prepare_rope(packed)), expected)


@pytest.fixture
def small_model():
    config = MOSSImageDiTConfig(
        arch_config=MOSSImageArchConfig(
            dim=64,
            ffn_dim=128,
            num_heads=2,
            num_layers=2,
            text_dim=32,
            freq_dim=16,
            latent_channels=8,
        )
    )
    return MOSSImageTransformer(config).eval()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_prepared_conditioning_matches_steps_and_requests(
    small_model, dtype, monkeypatch
):
    model = small_model.to(dtype)
    torch.manual_seed(0)

    def unexpected(*args, **kwargs):
        pytest.fail("A denoising step repeated static conditioning work")

    # Different requests, spatial shapes and non-prefix masks must stay independent.
    for height, width in [(2, 3), (3, 2), (2, 3)]:
        text = torch.randn(2, 5, 32).to(dtype)
        mask = torch.tensor([[1, 0, 1, 0, 0], [0, 1, 1, 1, 1]])
        with torch.no_grad(), set_forward_context(0, None):
            prepared = model.prepare_conditioning(text, mask, height, width, dtype)
            saved = [
                tensor.clone()
                for group in prepared.groups
                for tensor in (group.text, *group.rope)
            ]
            for timestep in (938.0, 24.0):
                latents = torch.randn(2, 8, height, width)
                expected = model(latents, timestep, text, mask)
                with monkeypatch.context() as patch:
                    patch.setattr(model, "_unpad_text", unexpected)
                    patch.setattr(model, "_prepare_sample_conditioning", unexpected)
                    actual = model(latents, timestep, text, mask, prepared)
                assert torch.equal(actual, expected)
            current = [
                tensor
                for group in prepared.groups
                for tensor in (group.text, *group.rope)
            ]
            assert all(torch.equal(a, b) for a, b in zip(saved, current))


@pytest.mark.parametrize("skip", [(), (1,)])
@pytest.mark.parametrize("batch_samples", [False, True])
def test_batch_groups_true_lengths_and_restores_sample_order(
    small_model, skip, batch_samples
):
    torch.manual_seed(42)
    x, text = torch.randn(4, 8, 2, 3), torch.randn(4, 5, 32)
    mask = torch.tensor(
        [[1, 0, 1, 0, 0], [1, 1, 1, 1, 0], [0, 1, 0, 1, 0], [1, 1, 1, 0, 0]]
    )
    times = torch.tensor([1000.0, 630.0, 24.0, 835.0])
    with torch.no_grad(), set_forward_context(0, None):
        expected = torch.cat(
            [
                small_model(
                    x[i : i + 1],
                    times[i],
                    text[i : i + 1],
                    mask[i : i + 1],
                    skip_layers=skip,
                )
                for i in range(4)
            ]
        )
        calls = []
        hook = small_model.layers[0].register_forward_pre_hook(
            lambda _, args: calls.append(args[0].shape[0])
        )
        try:
            prepared = small_model.prepare_conditioning(
                text, mask, 2, 3, torch.float32, batch_samples=batch_samples
            )
            actual = small_model(
                x, times, text, mask, prepared_conditioning=prepared, skip_layers=skip
            )
        finally:
            hook.remove()
    assert calls == ([2, 1, 1] if batch_samples else [1, 1, 1, 1])
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    "mask", [torch.zeros(1, 3), torch.tensor([[1.0, 0.5, 0.0]]), torch.ones(1, 2)]
)
def test_prepare_conditioning_rejects_invalid_masks(small_model, mask):
    with torch.no_grad(), set_forward_context(0, None), pytest.raises(ValueError):
        small_model.prepare_conditioning(
            torch.randn(1, 3, 32), mask, 2, 3, torch.float32
        )


@pytest.mark.parametrize("shape", [(1, 8, 3, 2), (2, 8, 2, 3)])
def test_prepared_conditioning_requires_matching_latents(small_model, shape):
    text = torch.randn(1, 3, 32)
    with torch.no_grad(), set_forward_context(0, None):
        prepared = small_model.prepare_conditioning(text, None, 2, 3, torch.float32)
        with pytest.raises(ValueError, match="latent batch and size"):
            small_model(
                torch.randn(shape),
                630.0,
                text.expand(shape[0], -1, -1),
                prepared_conditioning=prepared,
            )


@pytest.mark.parametrize("use_cfg", [False, True])
def test_denoising_stage_prepares_actual_cfg_branches(
    small_model, use_cfg, monkeypatch
):
    from sglang.multimodal_gen.runtime.distributed.cfg_policy import CFGPolicy
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.moss_image import (
        MOSSImageDenoisingStage,
    )

    batch = SimpleNamespace(
        do_classifier_free_guidance=use_cfg,
        sampling_params=SimpleNamespace(cfg_gate_step=1.0, quality="exact"),
    )
    latents = torch.randn(1, 8, 2, 3)
    pos = {"encoder_hidden_states": torch.randn(1, 3, 32)}
    neg = {"encoder_hidden_states": torch.randn(1, 2, 32)}
    ctx = SimpleNamespace(
        cfg_policy=CFGPolicy().build(batch, {}, pos, neg),
        pos_cond_kwargs=pos,
        neg_cond_kwargs=neg,
        latents=latents,
        target_dtype=torch.float32,
        scheduler=MOSSImageScheduler(),
        timesteps=torch.tensor([938.0, 24.0]),
        extra={},
    )
    stage = object.__new__(MOSSImageDenoisingStage)
    stage.transformer = small_model
    uses = []
    monkeypatch.setattr(stage, "_manage_dit_use_site", lambda *args: uses.append(args))
    with torch.no_grad(), set_forward_context(0, None):
        stage._before_denoising_loop(
            ctx, batch, SimpleNamespace(enable_cfg_parallel=False)
        )
        assert uses == [(small_model, "transformer", batch)]
        assert len(ctx.cfg_policy.branches) == (2 if use_cfg else 1)
        for branch in ctx.cfg_policy.branches:
            assert "prepared_conditioning" in branch.kwargs
            expected = small_model(
                latents, 938.0, branch.kwargs["encoder_hidden_states"]
            )
            actual = small_model(latents, 938.0, **branch.kwargs)
            assert torch.equal(actual, expected)


@pytest.fixture
def text_stage():
    from transformers import BatchEncoding

    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.moss_image import (
        MOSSImageBeforeDenoisingStage,
    )

    class Tokenizer:
        prefix = ""
        token_offset = 0

        def apply_chat_template(self, messages, **kwargs):
            return self.prefix + messages[0]["content"]

        def __call__(self, prompt, **kwargs):
            ids = [ord(char) % 13 + 1 + self.token_offset for char in prompt] + [2]
            ids = torch.tensor([ids[: kwargs["max_length"]]], dtype=torch.long)
            return BatchEncoding(
                {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
            )

    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.arange(1, 5, dtype=torch.float32))
            self.calls = 0

        def forward(self, input_ids, **kwargs):
            self.calls += 1
            return SimpleNamespace(
                last_hidden_state=input_ids.to(self.scale.dtype).unsqueeze(-1)
                * self.scale
            )

    return MOSSImageBeforeDenoisingStage(
        Encoder().eval(), Tokenizer(), MOSSImageScheduler()
    )


def test_conditioning_cache_preserves_padding_and_request_ownership(
    text_stage,
):
    from sglang.multimodal_gen.runtime.cache.conditioning import ConditioningCache

    stage = text_stage
    prompts = ["short", "short", "a longer negative"]
    device = torch.device("cpu")
    expected, expected_mask = stage._encode_prompt(prompts, device)
    stage.text_encoder.calls = 0
    cache = ConditioningCache(max_bytes=1024 * 1024)
    with cache.scope():
        first, mask = stage._encode_prompt(prompts, device)
        first.add_(1000)
        actual, _ = stage._encode_prompt(prompts, device)
    assert stage.text_encoder.calls == 2
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal(mask, expected_mask)


@pytest.mark.parametrize("change", ["template", "vocabulary", "truncation"])
def test_conditioning_cache_keys_actual_tokens(text_stage, change):
    from sglang.multimodal_gen.runtime.cache.conditioning import ConditioningCache

    stage = text_stage
    device = torch.device("cpu")
    cache = ConditioningCache(max_bytes=1024 * 1024)
    with cache.scope():
        stage._encode_prompt(["negative"], device)
        limit = 4096
        if change == "template":
            stage.tokenizer.prefix = "prefix: "
        elif change == "vocabulary":
            stage.tokenizer.token_offset = 1
        else:
            limit = 3
        actual, mask = stage._encode_prompt(["negative"], device, limit)
        assert stage.text_encoder.calls == 2
    expected, expected_mask = stage._encode_prompt(["negative"], device, limit)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal(mask, expected_mask)


def test_pipeline_residency_declares_only_components_used_by_each_stage(
    small_model, monkeypatch
):
    from sglang.multimodal_gen.runtime.pipelines.moss_image import MOSSImagePipeline
    from sglang.multimodal_gen.runtime.pipelines_core.stages.decoding import (
        DecodingStage,
    )
    from sglang.multimodal_gen.runtime.server_args import get_global_server_args

    args = get_global_server_args()
    monkeypatch.setattr(args, "pipeline_config", MOSSImagePipelineConfig())
    monkeypatch.setattr(args, "component_precisions", {}, raising=False)
    pipe = object.__new__(MOSSImagePipeline)
    pipe.modules = {
        "transformer": small_model,
        "text_encoder": torch.nn.Identity(),
        "tokenizer": object(),
        "scheduler": MOSSImageScheduler(),
        "vae": torch.nn.Identity(),
    }
    stages = []
    monkeypatch.setattr(pipe, "add_stage", stages.append)
    monkeypatch.setattr(
        pipe, "add_stage_factory", lambda role, factory, name: stages.append(factory())
    )
    monkeypatch.setattr(
        pipe,
        "add_standard_decoding_stage",
        lambda: stages.append(DecodingStage(vae=pipe.modules["vae"], pipeline=pipe)),
    )
    pipe.create_pipeline_stages(args)
    # Offload/auto-residency must see encoding -> DiT -> VAE. MOSSImage never
    # uses the VAE in denoising, unlike the shared stage's TI2V callers.
    assert [
        [use.component_name for use in stage.component_uses(args)] for stage in stages
    ] == [[], ["text_encoder"], ["transformer"], ["vae"]]
    assert stages[1].component_uses(args)[0].memory_intensive
    assert stages[2].vae is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("resident_layers", [0, 1])
@pytest.mark.parametrize("skip_layers", [(), (0,), (1,), (0, 1)])
def test_layerwise_offload_preserves_requests_and_streams(
    small_model, resident_layers, skip_layers
):
    torch.manual_seed(0)
    model = small_model.to(device="cuda", dtype=torch.bfloat16)
    text = torch.randn(1, 5, 32, device="cuda", dtype=torch.bfloat16)
    timestep = torch.tensor([630.0], device="cuda")
    latents = [
        torch.randn(1, 8, h, w, device="cuda", dtype=torch.bfloat16)
        for h, w in [(2, 3), (3, 2), (2, 3)]
    ]
    with torch.inference_mode(), set_forward_context(0, None):
        expected = [model(x, timestep, text, skip_layers=skip_layers) for x in latents]
    model.configure_layerwise_offload(
        SimpleNamespace(
            performance_mode="manual",
            pin_cpu_memory=True,
            layerwise_tuning_for=lambda *args, **kwargs: (
                1,
                resident_layers,
                "strided",
                "forward",
            ),
        )
    )
    assert len(model.layerwise_offload_managers) == 1
    manager = model.layerwise_offload_managers[0]
    assert manager.enabled and manager.num_layers == 2
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    torch.cuda.synchronize()
    try:
        for index, (x, reference) in enumerate(zip(latents, expected)):
            with (
                torch.cuda.stream(streams[index % 2]),
                torch.inference_mode(),
                set_forward_context(0, None),
            ):
                model.prepare_for_next_req()
                prepared = model.prepare_conditioning(
                    text, None, *x.shape[-2:], x.dtype
                )
                output = model(
                    x,
                    timestep,
                    text,
                    prepared_conditioning=prepared,
                    skip_layers=skip_layers,
                )
                manager.release_all()
            torch.cuda.synchronize()
            torch.testing.assert_close(output, reference, rtol=0, atol=0)
    finally:
        model.disable_offload()
        manager.release_host_stores()


@pytest.mark.parametrize(
    "guidance,expected_cfg", [(0.0, True), (0.5, True), (1.0, False), (5.0, True)]
)
@pytest.mark.parametrize("negative", [None, ""])
def test_cfg_rule_applies_before_input_validation(guidance, expected_cfg, negative):
    """Sub-unit guidance needs a negative branch before shared input validation.

    Previously only the model stage enabled it, so request validation and CFG
    policy disagreed. Missing negative text must become the reference empty text.
    """
    batch = Req(
        sampling_params=MOSSImageSamplingParams(
            prompt="test",
            negative_prompt=negative,
            guidance_scale=guidance,
            generator_device="cpu",
        )
    )
    assert batch.do_classifier_free_guidance is expected_cfg
    result = InputValidationStage().forward(
        batch,
        SimpleNamespace(
            pipeline_config=MOSSImagePipelineConfig(),
            enable_cfg_parallel=False,
        ),
    )
    assert result.do_classifier_free_guidance is expected_cfg
    assert result.negative_prompt == ""


VELOCITIES = {
    (True, False): torch.tensor([1.0078125, -0.00390625]).reshape(1, 2, 1, 1),
    (False, False): torch.tensor([-0.0078125, 0.015625]).reshape(1, 2, 1, 1),
    (True, True): torch.tensor([0.00390625, -1.015625]).reshape(1, 2, 1, 1),
    (False, True): torch.tensor([-1.0234375, 0.00390625]).reshape(1, 2, 1, 1),
}


@pytest.mark.parametrize(
    "guidance,options,uncond_steps,extra_steps",
    [
        (3.0, {"uncond_skip_layers": []}, (), ()),
        (None, {}, (1, 2, 3), ()),
        (3.0, {"uncond_skip_layers": [22], "slg_start": 0.0}, (0, 1, 2, 3), ()),
        (
            3.0,
            {"uncond_skip_layers": [22], "slg_start": 0.01, "slg_stop": 1.0},
            (1, 2, 3),
            (),
        ),
        (
            3.0,
            {"uncond_skip_layers": [22], "slg_start": 0.25, "slg_stop": 0.75},
            (1, 2),
            (),
        ),
        (
            5.0,
            {
                "slg_layers": [22],
                "slg_scale": 3.0,
                "slg_start": 0.01,
            },
            (),
            (1, 2, 3),
        ),
        (
            5.0,
            {
                "uncond_skip_layers": [22],
                "slg_layers": [22],
                "slg_scale": 3.0,
                "slg_start": 0.01,
            },
            (1, 2, 3),
            (1, 2, 3),
        ),
        (
            1.0,
            {
                "slg_layers": [22],
                "slg_scale": 3.0,
                "slg_start": 0.25,
                "slg_stop": 0.75,
            },
            (),
            (1, 2),
        ),
        (1.0, {"slg_layers": [22], "slg_scale": 1.0}, (), (1, 2, 3)),
        (5.0, {"uncond_skip_layers": [], "slg_layers": [22], "slg_scale": 0.0}, (), ()),
        (5.0, {"uncond_skip_layers": [], "slg_layers": [], "slg_scale": 3.0}, (), ()),
        (5.0, {"uncond_skip_layers": [], "slg_layers": None, "slg_scale": 3.0}, (), ()),
        (
            5.0,
            {"slg_layers": [22], "slg_scale": 3.0, "slg_start": 0.5, "slg_stop": 0.5},
            (),
            (),
        ),
        (0.0, {"uncond_skip_layers": []}, (), ()),
        (0.5, {"uncond_skip_layers": []}, (), ()),
    ],
)
def test_guidance_formula_window_branch_count_and_request_isolation(
    monkeypatch, guidance, options, uncond_steps, extra_steps
):
    """Check reference branch topology, FP32 guidance and window boundaries."""
    if guidance is None:
        params = MOSSImageSamplingParams(**options)
        guidance = 3.0  # Recommended default from the reference recipe.
        assert params.guidance_scale == guidance
    else:
        params = MOSSImageSamplingParams(guidance_scale=guidance, **options)
    batch = Req(
        sampling_params=params,
        timesteps=torch.tensor([1000.0, 750.0, 500.0, 250.0]),
    )
    config = MOSSImagePipelineConfig()
    pos = {"encoder_hidden_states": torch.ones(1, 1, 2)}
    neg = {
        "encoder_hidden_states": -torch.ones(1, 1, 2),
        **config.prepare_neg_cond_kwargs(batch, "cpu", None, torch.float32),
    }
    policy = CFGPolicy().build(batch, {}, pos, neg)
    initial_kwargs = [dict(branch.kwargs) for branch in policy.branches]
    stage = object.__new__(MOSSImageDenoisingStage)
    calls = []

    def predict(**kwargs):
        conditional = bool(kwargs["encoder_hidden_states"][0, 0, 0] > 0)
        skip = tuple(kwargs.get("skip_layers", ()))
        assert batch.is_cfg_negative == (not conditional)
        calls.append((conditional, skip))
        return VELOCITIES[(conditional, bool(skip))].clone()

    monkeypatch.setattr(stage, "_predict_noise", predict)
    for index, timestep in enumerate(batch.timesteps):
        start = len(calls)
        actual = stage._predict_noise_with_cfg(
            current_model=None,
            latent_model_input=torch.zeros(1, 2, 1, 1),
            timestep=timestep,
            batch=batch,
            timestep_index=index,
            attn_metadata=None,
            target_dtype=torch.float32,
            current_guidance_scale=guidance,
            cfg_policy=policy,
            cfg_gate_state=None,
            server_args=SimpleNamespace(
                pipeline_config=config, enable_cfg_parallel=False
            ),
            guidance=None,
            latents=torch.zeros(1, 2, 1, 1),
        )
        conditional = VELOCITIES[(True, False)]
        expected = conditional
        expected_calls = [(True, ())]
        if guidance != 1:
            negative = VELOCITIES[(False, index in uncond_steps)]
            expected = negative + guidance * (conditional - negative)
            expected_calls.append((False, (22,) if index in uncond_steps else ()))
        if index in extra_steps:
            skipped = VELOCITIES[(True, True)]
            if guidance == 1:
                expected = skipped
            expected = expected + params.slg_scale * (conditional - skipped)
            expected_calls.append((True, (22,)))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert calls[start:] == expected_calls
        assert [branch.kwargs for branch in policy.branches] == initial_kwargs


@pytest.mark.parametrize(
    "options,message",
    [
        ({"uncond_skip_layers": 22}, "skip_layers"),
        ({"slg_layers": [42]}, "skip_layers"),
        ({"slg_scale": -1.0}, "slg_scale"),
        ({"slg_scale": None}, "slg_scale"),
        ({"slg_scale": "3"}, "slg_scale"),
        ({"slg_scale": float("nan")}, "slg_scale"),
        ({"slg_scale": float("inf")}, "slg_scale"),
        ({"slg_start": -0.01}, "slg_start"),
        ({"slg_stop": 1.01}, "slg_start"),
        ({"slg_start": 0.8, "slg_stop": 0.2}, "slg_start"),
        ({"slg_start": float("nan")}, "slg_start"),
        ({"slg_stop": float("inf")}, "slg_start"),
    ],
)
def test_invalid_guidance_is_rejected_before_encoding(options, message):
    from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req

    stage = MOSSImageBeforeDenoisingStage(None, None, None)
    batch = Req(sampling_params=MOSSImageSamplingParams(prompt="test", **options))
    with pytest.raises(ValueError, match=message):
        stage.forward(batch, SimpleNamespace(pipeline_config=MOSSImagePipelineConfig()))


@pytest.mark.parametrize(
    "mode,available_gib,explicit,expected",
    [
        ("auto", 24, None, "layerwise-offload"),
        ("auto", 39, None, "layerwise-offload"),
        ("auto", 40, None, "resident"),
        ("auto", 80, "component-offload", "component-offload"),
        ("auto", 24, "resident", "resident"),
        ("auto", 24, "layerwise-offload", "layerwise-offload"),
        ("memory", 80, None, "layerwise-offload"),
    ],
)
def test_pipeline_auto_uses_supported_encoder_offload(
    monkeypatch, mode, available_gib, explicit, expected
):
    from sglang.multimodal_gen.configs.pipeline_configs.base import PipelineConfig
    from sglang.multimodal_gen.runtime.pipelines.moss_image import MOSSImagePipeline
    from sglang.multimodal_gen.runtime.platforms import current_platform
    from sglang.multimodal_gen.runtime.server_args import ServerArgs

    for name, value in {
        "is_cpu": False,
        "is_mps": False,
        "is_cuda": True,
        "get_device_total_memory": 80 * 1024**3,
        "get_available_gpu_memory": available_gib,
        "enable_dit_layerwise_offload_by_default": True,
    }.items():
        monkeypatch.setattr(
            current_platform, name, lambda *args, _value=value, **kwargs: _value
        )
    monkeypatch.setattr(
        PipelineConfig, "from_kwargs", lambda kwargs: MOSSImagePipelineConfig()
    )
    args = ServerArgs.from_dict(
        {
            "model_path": "/fake",
            "performance_mode": mode,
            "component_residency": {"text_encoder": explicit} if explicit else None,
        }
    )
    pipe = object.__new__(MOSSImagePipeline)
    pipe.modules = {}
    pipe.initialize_pipeline(args)
    assert args.residency_mode("text_encoder") == expected
    if expected == "component-offload":
        assert "text_encoder" not in (args.layerwise_offload_components or [])
    assert not args.enable_torch_compile and not args.enable_cfg_parallel
