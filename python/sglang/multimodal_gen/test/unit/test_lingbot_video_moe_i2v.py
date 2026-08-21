# SPDX-License-Identifier: Apache-2.0

from contextlib import nullcontext
from types import SimpleNamespace

import PIL.Image
import pytest
import torch
from diffusers.utils.torch_utils import randn_tensor

from sglang.multimodal_gen.configs.pipeline_configs.base import ModelTaskType
from sglang.multimodal_gen.configs.pipeline_configs.lingbot_video_moe import (
    LingBotVideoMoEPipelineConfig,
)
from sglang.multimodal_gen.configs.sample.lingbot_video_moe import (
    DEFAULT_NEGATIVE_PROMPT,
    DEFAULT_NEGATIVE_PROMPT_IMAGE,
    REFINER_PIPELINE_NAME,
    LingBotVideoMoESamplingParams,
)
from sglang.multimodal_gen.configs.sample.sampling_params import DataType
from sglang.multimodal_gen.runtime.pipelines.lingbot_video_moe_refiner import (
    LingBotVideoRefinerPipeline,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.lingbot_video_moe import (
    LingBotVideoDenoisingStage,
    LingBotVideoRefinementStage,
    LingBotVideoRefinerUpscaleStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.lingbot_video_moe import (
    refiner_stages as refiner_stages_module,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.lingbot_video_moe.auto_negative import (
    LingBotVideoAutoNegativeStage,
    prune_negative,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.lingbot_video_moe.i2v import (
    COND_LATENT_KEY,
    VLM_IMAGE_KEY,
    VLM_PATCH_FACTOR,
    apply_first_frame_prefix,
    pixel_to_vlm_image,
    preprocess_condition_image,
    smart_resize,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.lingbot_video_moe.refiner import (
    compute_refiner_sigmas,
    prepare_refiner_latent,
    resize_video_pixels,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.lingbot_video_moe.rewriter import (
    LingBotVideoPromptRewriteStage,
    build_expand_prompt,
    build_map_prompt,
    needs_rewrite,
    parse_caption,
    resolve_mode,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.lingbot_video_moe.rewriter_prompts import (
    VIDEO_DURATION_EN,
    VIDEO_DURATION_ZH,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.lingbot_video_moe.text_encoding import (
    IMG_PROMPT_TEMPLATE,
    PROMPT_TEMPLATE,
    LingBotVideoTextEncodingStage,
)


class _RecordingProcessor:
    def __init__(self, width=8):
        self.width = width
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "input_ids": torch.zeros(1, self.width, dtype=torch.long),
            "attention_mask": torch.ones(1, self.width, dtype=torch.long),
        }


class _StubVae:
    # The fast-path gate registry is keyed by weak reference, so this cannot be a
    # SimpleNamespace.
    def __init__(self):
        self.use_tiling = False


class _StubScheduler:
    sigma_max = 1.0
    sigma_min = 0.0

    def __init__(self):
        self.timesteps = torch.zeros(1)

    def set_timesteps(self, device, sigmas, shift):
        self.timesteps = torch.as_tensor(sigmas)


def _text_encoding_stage(processor):
    stage = object.__new__(LingBotVideoTextEncodingStage)
    stage.tokenizers = [processor]
    stage.token_length = 128
    stage.prompt_template = PROMPT_TEMPLATE
    stage._crop_start = None
    return stage


def test_condition_image_prepends_vision_block_before_caption():
    processor = _RecordingProcessor()
    stage = _text_encoding_stage(processor)
    image = PIL.Image.new("RGB", (64, 64))

    stage._build_prompt_inputs("a structured caption", [image])

    call = processor.calls[0]
    assert call["images"] == [image]
    text = call["text"][0]
    assert IMG_PROMPT_TEMPLATE + "a structured caption" in text
    assert text.index(IMG_PROMPT_TEMPLATE) < text.index("a structured caption")
    assert text.index("<|im_start|>user") < text.index(IMG_PROMPT_TEMPLATE)


def test_text_only_prompt_keeps_no_vision_block():
    processor = _RecordingProcessor()
    stage = _text_encoding_stage(processor)

    stage._build_prompt_inputs("a structured caption")

    call = processor.calls[0]
    assert call["images"] is None
    assert IMG_PROMPT_TEMPLATE not in call["text"][0]


def test_crop_start_is_measured_on_the_text_only_prefix():
    processor = _RecordingProcessor()
    stage = _text_encoding_stage(processor)

    assert stage._compute_crop_start() == processor.width
    assert processor.calls[0]["images"] is None


def test_vlm_images_read_the_condition_frame_from_batch_extra():
    image = PIL.Image.new("RGB", (32, 32))
    read = LingBotVideoTextEncodingStage._vlm_images
    assert read(SimpleNamespace(extra={})) is None
    assert read(SimpleNamespace(extra={VLM_IMAGE_KEY: image})) == [image]


def test_condition_image_is_cover_resized_then_center_cropped():
    height, width = 32, 64
    # Portrait source: cover-resize scales on width, so the crop trims the height.
    source = PIL.Image.new("RGB", (32, 128), (255, 128, 0))
    pixel = preprocess_condition_image(source, height, width)

    assert tuple(pixel.shape) == (1, 3, 1, height, width)
    assert pixel.dtype == torch.float32
    assert 0.0 <= float(pixel.min()) and float(pixel.max()) <= 1.0


def test_condition_image_preserves_pixel_values_when_already_sized():
    array = torch.randint(0, 256, (16, 24, 3), dtype=torch.uint8)
    image = PIL.Image.fromarray(array.numpy(), mode="RGB")

    pixel = preprocess_condition_image(image, 16, 24)

    expected = array.permute(2, 0, 1).float().div(255.0)[None, :, None]
    torch.testing.assert_close(pixel, expected)


def test_smart_resize_aligns_to_the_patch_factor():
    height, width = smart_resize(480, 832, factor=VLM_PATCH_FACTOR)
    assert (height, width) == (480, 832)

    # Below the four-patch minimum the frame is enlarged, still factor-aligned.
    height, width = smart_resize(30, 30, factor=VLM_PATCH_FACTOR)
    assert height % VLM_PATCH_FACTOR == 0 and width % VLM_PATCH_FACTOR == 0
    assert height * width >= 4 * VLM_PATCH_FACTOR**2


def test_vlm_image_is_patch_aligned():
    pixel = preprocess_condition_image(PIL.Image.new("RGB", (100, 60)), 48, 80)

    image = pixel_to_vlm_image(pixel)

    assert image.mode == "RGB"
    assert image.width % VLM_PATCH_FACTOR == 0
    assert image.height % VLM_PATCH_FACTOR == 0


def test_first_frame_prefix_overwrites_only_the_conditioned_frames():
    latents = torch.zeros(1, 16, 5, 2, 3)
    cond = torch.arange(16 * 2 * 3, dtype=torch.float32).reshape(1, 16, 1, 2, 3)

    out = apply_first_frame_prefix(latents, cond)

    torch.testing.assert_close(out[:, :, :1], cond)
    assert float(out[:, :, 1:].abs().max()) == 0.0


def test_first_frame_prefix_follows_the_latent_dtype():
    latents = torch.zeros(1, 16, 2, 1, 1, dtype=torch.float32)
    cond = torch.ones(1, 16, 1, 1, 1, dtype=torch.bfloat16)

    out = apply_first_frame_prefix(latents, cond)

    assert out.dtype == torch.float32
    assert float(out[:, :, 0].min()) == 1.0


def test_denoising_reapplies_the_condition_latent_after_each_step():
    stage = object.__new__(LingBotVideoDenoisingStage)
    cond = torch.full((1, 16, 1, 2, 2), 7.0)
    batch = SimpleNamespace(extra={COND_LATENT_KEY: cond})
    latents = torch.zeros(1, 16, 3, 2, 2)

    out = stage.post_forward_for_ti2v_task(batch, None, None, latents, None)

    torch.testing.assert_close(out[:, :, :1], cond)
    assert float(out[:, :, 1:].abs().max()) == 0.0


def test_denoising_leaves_latents_untouched_without_a_condition_latent():
    stage = object.__new__(LingBotVideoDenoisingStage)
    latents = torch.randn(1, 16, 3, 2, 2)

    out = stage.post_forward_for_ti2v_task(
        SimpleNamespace(extra={}), None, None, latents, None
    )

    torch.testing.assert_close(out, latents)


def test_config_takes_an_optional_image_and_loads_the_vae_encoder():
    config = LingBotVideoMoEPipelineConfig()

    assert config.task_type == ModelTaskType.TI2V
    assert config.task_type.accepts_image_input()
    assert not config.task_type.requires_image_input()
    assert config.skip_input_image_preprocess
    assert config.vae_config.load_encoder and config.vae_config.load_decoder
    # The base class excludes TI2V; the text-only path must keep batching.
    assert config.supports_dynamic_batching()


def test_single_frame_request_outputs_an_image():
    params = LingBotVideoMoESamplingParams(prompt="a caption", num_frames=1)

    params._set_output_file_name()

    assert params.data_type == DataType.IMAGE


def _adjusted(num_frames, explicit_fields, pipeline_class_name=None, **kwargs):
    params = LingBotVideoMoESamplingParams(
        prompt="a caption", num_frames=num_frames, **kwargs
    )
    params._explicit_fields = set(explicit_fields)
    params._adjust(
        SimpleNamespace(
            pipeline_config=LingBotVideoMoEPipelineConfig(),
            pipeline_class_name=pipeline_class_name,
            num_gpus=4,
            output_path=None,
            comfyui_mode=False,
        )
    )
    return params


def test_single_frame_request_swaps_in_the_image_negative_prompt():
    params = _adjusted(1, [])

    assert params.negative_prompt == DEFAULT_NEGATIVE_PROMPT_IMAGE
    assert "temporal_and_motion_stability" not in params.negative_prompt


def test_explicit_negative_prompt_survives_the_single_frame_default():
    params = _adjusted(1, ["negative_prompt"], negative_prompt="keep me")

    assert params.negative_prompt == "keep me"


def test_single_frame_survives_multi_gpu_frame_alignment():
    params = _adjusted(1, [])

    assert params.num_frames == 1


def test_refiner_rejects_single_frame_requests():
    with pytest.raises(ValueError, match="single-frame"):
        _adjusted(1, [], pipeline_class_name=REFINER_PIPELINE_NAME)


def test_refiner_accepts_video_requests():
    params = _adjusted(81, [], pipeline_class_name=REFINER_PIPELINE_NAME)

    assert params.data_type == DataType.VIDEO
    assert params.num_frames > 1


def test_video_request_keeps_the_video_negative_prompt():
    params = _adjusted(81, [])

    assert params.negative_prompt == DEFAULT_NEGATIVE_PROMPT
    assert params.data_type == DataType.VIDEO


def test_sequence_shard_is_on_and_frames_stay_as_requested():
    # The DiT shards the joint sequence itself, so latent frames must not be
    # rounded up to num_gpus: 9 frames would otherwise become 13.
    params = _adjusted(9, [])

    assert params.enable_sequence_shard
    assert not params.adjust_frames
    assert params.num_frames == 9


def test_sequence_shard_can_be_turned_off_by_the_request():
    params = _adjusted(81, ["enable_sequence_shard"], enable_sequence_shard=False)

    assert not params.enable_sequence_shard


def test_refiner_sigmas_start_at_the_threshold_and_descend():
    sigmas = compute_refiner_sigmas(
        sigma_max=1.0,
        sigma_min=0.0,
        num_inference_steps=8,
        shift=3.0,
        t_thresh=0.85,
        tail_steps=2,
    )

    assert abs(float(sigmas[0]) - 0.85) < 1e-6
    assert all(b < a for a, b in zip(sigmas, sigmas[1:]))
    assert float(sigmas.min()) >= 0.0 and float(sigmas.max()) <= 1.0


def test_refiner_sigma_tail_steps_extend_the_schedule():
    kwargs = dict(
        sigma_max=1.0,
        sigma_min=0.0,
        num_inference_steps=8,
        shift=3.0,
        t_thresh=0.85,
    )

    plain = compute_refiner_sigmas(tail_steps=0, **kwargs)
    tailed = compute_refiner_sigmas(tail_steps=2, **kwargs)

    assert len(tailed) == len(plain) + 2


@pytest.mark.parametrize("t_thresh", [0.0, 1.5])
def test_refiner_rejects_a_threshold_outside_the_unit_range(t_thresh):
    with pytest.raises(ValueError, match="t_thresh"):
        compute_refiner_sigmas(
            sigma_max=1.0,
            sigma_min=0.0,
            num_inference_steps=8,
            shift=3.0,
            t_thresh=t_thresh,
        )


def test_refiner_latent_interpolates_toward_noise_at_the_threshold():
    upscaled = torch.zeros(1, 16, 2, 2, 2)
    noise = torch.ones(1, 16, 2, 2, 2)

    blended = prepare_refiner_latent(upscaled, noise, 0.85)

    torch.testing.assert_close(blended, torch.full_like(upscaled, 0.85))


def test_pixel_resize_keeps_the_clip_layout():
    pixels = torch.rand(1, 3, 5, 24, 32)

    resized = resize_video_pixels(pixels, 48, 64)

    assert tuple(resized.shape) == (1, 3, 5, 48, 64)
    assert 0.0 <= float(resized.min()) and float(resized.max()) <= 1.0


def test_refiner_weights_resolve_to_the_refiner_subfolder():
    pipeline = object.__new__(LingBotVideoRefinerPipeline)
    pipeline.model_path = "/models/lingbot"
    server_args = SimpleNamespace(component_paths={})

    path = pipeline._resolve_component_path(
        server_args, "transformer_2", "transformer_2"
    )

    assert path == "/models/lingbot/refiner"


def test_transformer_2_path_override_wins_over_the_refiner_subfolder(tmp_path):
    override = tmp_path / "custom_refiner"
    override.mkdir()
    pipeline = object.__new__(LingBotVideoRefinerPipeline)
    pipeline.model_path = "/models/lingbot"
    server_args = SimpleNamespace(component_paths={"transformer_2": str(override)})

    path = pipeline._resolve_component_path(
        server_args, "transformer_2", "transformer_2"
    )

    assert path == str(override)


def test_refiner_reseeds_every_output_of_a_batched_request(monkeypatch):
    monkeypatch.setattr(
        refiner_stages_module, "get_local_torch_device", lambda: torch.device("cpu")
    )
    batch = SimpleNamespace(seeds=[11, 22])

    generators = refiner_stages_module._refiner_generators(batch, torch.device("cpu"))
    batched = randn_tensor(
        (2, 1, 2, 2), generator=generators, device=torch.device("cpu")
    )

    for index, seed in enumerate(batch.seeds):
        alone = randn_tensor(
            (1, 1, 2, 2),
            generator=[torch.Generator().manual_seed(seed)],
            device=torch.device("cpu"),
        )
        torch.testing.assert_close(batched[index : index + 1], alone)


def _refiner_prepared(monkeypatch, *, base_scale, refiner_scale):
    monkeypatch.setattr(
        refiner_stages_module, "get_local_torch_device", lambda: torch.device("cpu")
    )
    monkeypatch.setattr(
        LingBotVideoDenoisingStage,
        "_prepare_denoising_loop",
        lambda self, batch, server_args: batch,
    )
    stage = object.__new__(LingBotVideoRefinementStage)
    config = LingBotVideoMoEPipelineConfig()
    config.refiner_guidance_scale = refiner_scale
    batch = SimpleNamespace(
        scheduler=_StubScheduler(),
        timesteps=None,
        num_inference_steps=None,
        guidance_scale=base_scale,
        do_classifier_free_guidance=base_scale > 1.0,
        prompt_embeds=[torch.ones(1, 4, 8)],
        prompt_attention_mask=torch.ones(1, 4, dtype=torch.long),
        negative_prompt_embeds=None,
        negative_attention_mask=None,
        extra={},
    )
    stage._prepare_denoising_loop(batch, SimpleNamespace(pipeline_config=config))
    return batch


def test_refiner_guidance_turns_cfg_on_when_the_base_pass_ran_without_it(monkeypatch):
    batch = _refiner_prepared(monkeypatch, base_scale=1.0, refiner_scale=3.0)

    assert batch.guidance_scale == 3.0
    assert batch.do_classifier_free_guidance
    torch.testing.assert_close(
        batch.negative_prompt_embeds[0], torch.zeros_like(batch.prompt_embeds[0])
    )


def test_refiner_guidance_turns_cfg_off_when_the_refiner_scale_is_one(monkeypatch):
    batch = _refiner_prepared(monkeypatch, base_scale=6.0, refiner_scale=1.0)

    assert not batch.do_classifier_free_guidance
    assert batch.negative_prompt_embeds is None


def test_refiner_round_trip_matches_the_declared_decode_precision(monkeypatch):
    monkeypatch.setattr(
        refiner_stages_module, "get_local_torch_device", lambda: torch.device("cpu")
    )
    vae = _StubVae()
    stage = object.__new__(LingBotVideoRefinerUpscaleStage)
    stage.component_name = "vae"
    stage.vae = vae
    stage.use_declared_component = lambda **kwargs: nullcontext(vae)

    seen = {}

    def decode(latents, server_args, *, vae_dtype):
        seen["decode"] = vae_dtype
        return torch.zeros(1, 3, 1, 4, 4)

    def encode(pixels, server_args, *, vae_dtype, generators):
        seen["encode"] = vae_dtype
        return torch.zeros(1, 4, 1, 2, 2)

    stage.decode = decode
    stage._encode = encode

    config = LingBotVideoMoEPipelineConfig()
    # The decode-only override must win over vae_precision on both halves.
    config.vae_decode_precision = "fp32"
    config.refiner_height, config.refiner_width = 8, 16
    server_args = SimpleNamespace(pipeline_config=config, disable_autocast=False)
    batch = SimpleNamespace(
        latents=torch.zeros(1, 4, 1, 2, 2),
        extra={},
        seeds=[],
        sampling_params=SimpleNamespace(quality="lossless"),
        height=480,
        width=832,
    )

    stage.forward(batch, server_args)

    declared = stage.component_uses(server_args)[0].target_dtype
    assert declared == torch.float32
    assert seen["decode"] == declared
    assert seen["encode"] == declared
    assert (batch.height, batch.width) == (8, 16)


def test_rewrite_mode_follows_the_request_shape():
    assert resolve_mode(SimpleNamespace(num_frames=1, image_path=None)) == "t2i"
    assert resolve_mode(SimpleNamespace(num_frames=121, image_path=None)) == "t2v"
    assert resolve_mode(SimpleNamespace(num_frames=121, image_path="f.png")) == "ti2v"


def test_structured_captions_skip_rewriting():
    assert needs_rewrite("a red fox in the snow")
    assert not needs_rewrite('{"comprehensive_description": "..."}')
    assert not needs_rewrite('  {"caption": {}}')


def test_expand_prompt_carries_the_duration_for_video_modes():
    video = build_expand_prompt("t2v", "a red fox", 5)
    assert "Video Duration: 5 seconds" in video
    assert video.endswith("Video Duration: 5 seconds")

    assert "Duration" not in build_expand_prompt("t2i", "a glass of tea", 5)


def test_expand_prompt_matches_the_caption_language():
    cjk_prompt = VIDEO_DURATION_ZH.format(duration=3)
    expanded = build_expand_prompt("t2v", cjk_prompt, 5)

    assert VIDEO_DURATION_ZH.format(duration=5) in expanded
    assert VIDEO_DURATION_EN.format(duration=5) not in expanded


def test_map_prompt_feeds_the_expansion_back():
    mapped = build_map_prompt("t2v", "A red fox trots through snow.", 5)

    assert "DETAILED CAPTION:\nA red fox trots through snow." in mapped
    assert mapped.endswith("Output the JSON now.")


def test_caption_parsing_tolerates_fences_and_prose():
    assert parse_caption('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_caption('Here you go: {"a": 1} hope that helps') == {"a": 1}
    assert parse_caption("no json here") is None
    assert parse_caption('["not", "an", "object"]') is None


class _FakeBackend:
    def __init__(self, replies):
        self.replies = replies
        self.calls = []

    def generate(self, text, image, use_lora):
        self.calls.append((use_lora, image))
        return self.replies.pop(0)


def _rewrite_stage(replies):
    stage = object.__new__(LingBotVideoPromptRewriteStage)
    stage.backend = _FakeBackend(replies)
    return stage


def test_rewriting_replaces_the_prompt_with_a_compact_caption():
    stage = _rewrite_stage(["A red fox trots through snow.", '{"b": 2, "a": 1}'])
    batch = SimpleNamespace(prompt="a red fox", num_frames=121, fps=24, image_path=None)

    stage.forward(batch, None)

    assert batch.prompt == '{"b":2,"a":1}'
    # Only the mapping turn enables the adapter.
    assert [use_lora for use_lora, _ in stage.backend.calls] == [False, True]
    assert all(image is None for _, image in stage.backend.calls)


def test_rewriting_refuses_to_fall_back_to_the_free_text_prompt():
    stage = _rewrite_stage(["expanded", "sorry, I cannot"])
    batch = SimpleNamespace(prompt="a red fox", num_frames=121, fps=24, image_path=None)

    with pytest.raises(ValueError, match="structured caption"):
        stage.forward(batch, None)


def test_rewriting_leaves_a_structured_caption_alone():
    stage = _rewrite_stage([])
    caption = '{"comprehensive_description": "..."}'
    batch = SimpleNamespace(prompt=caption, num_frames=121, fps=24, image_path=None)

    stage.forward(batch, None)

    assert batch.prompt == caption
    assert stage.backend.calls == []


_DEFAULT_NEGATIVE = {
    "universal_negative": {
        "visual_quality": ["low quality", "underexposed", "crushed blacks"],
        "artistic_style": ["painting", "cartoon"],
        "physical_plausibility": ["objects defying gravity"],
        "temporal_and_motion_stability": ["motion blur", "warping"],
    }
}


def test_auto_negative_keeps_the_default_order_and_drops_only_deletions():
    pruned = {
        "universal_negative": {
            # Reordered, with an invented term the model must not be able to add.
            "visual_quality": ["invented", "low quality"],
            "artistic_style": ["painting", "cartoon"],
            "physical_plausibility": ["objects defying gravity"],
            "temporal_and_motion_stability": ["warping", "motion blur"],
        }
    }

    out = prune_negative(_DEFAULT_NEGATIVE, pruned, "a bright meadow at noon")

    assert out["universal_negative"]["visual_quality"] == ["low quality"]
    assert out["universal_negative"]["temporal_and_motion_stability"] == [
        "motion blur",
        "warping",
    ]


def test_auto_negative_restores_a_block_the_caption_never_asked_to_drop():
    emptied = {
        "universal_negative": {"physical_plausibility": [], "artistic_style": []}
    }

    live_action = prune_negative(_DEFAULT_NEGATIVE, emptied, "a cinematic car chase")
    assert live_action["universal_negative"]["physical_plausibility"] == [
        "objects defying gravity"
    ]
    assert live_action["universal_negative"]["artistic_style"] == [
        "painting",
        "cartoon",
    ]

    surreal = prune_negative(
        _DEFAULT_NEGATIVE, emptied, "a surreal dreamlike watercolor painting"
    )
    assert surreal["universal_negative"]["physical_plausibility"] == []
    assert surreal["universal_negative"]["artistic_style"] == []


def test_auto_negative_deletes_terms_the_caption_clearly_wants():
    kept = {"universal_negative": dict(_DEFAULT_NEGATIVE["universal_negative"])}

    out = prune_negative(
        _DEFAULT_NEGATIVE, kept, "a moody night scene with deep shadows"
    )

    assert out["universal_negative"]["visual_quality"] == ["low quality"]


def test_auto_negative_turns_dynamic_batching_off():
    # A merged request carries one negative for several captions, so pruning it
    # per request is only possible while merging is off.
    config = LingBotVideoMoEPipelineConfig()
    assert config.supports_dynamic_batching()

    config.rewriter_auto_negative = True
    assert not config.supports_dynamic_batching()


def test_auto_negative_needs_a_rewriter_backend():
    config = LingBotVideoMoEPipelineConfig()
    config.check_pipeline_config()

    config.rewriter_auto_negative = True
    with pytest.raises(ValueError, match="rewriter_auto_negative"):
        config.check_pipeline_config()

    config.rewriter_url = "http://host:30000"
    config.check_pipeline_config()

    local = LingBotVideoMoEPipelineConfig()
    local.rewriter_auto_negative = True
    local.rewriter_model_path = "/models/base"
    local.rewriter_adapter_path = "/models/adapter"
    local.check_pipeline_config()


@pytest.mark.parametrize(
    "negative_prompt",
    [
        "blurry, low quality",
        '{"universal_negative": []}',
        '{"universal_negative": "bad"}',
        '{"universal_negative": {"visual_quality": "not a list"}}',
        '{"universal_negative": {"visual_quality": [1, 2]}}',
    ],
)
def test_auto_negative_skips_a_negative_it_cannot_prune(negative_prompt):
    # A request may send any string here, so an unprunable one is passed through
    # rather than indexed as categories.
    stage = object.__new__(LingBotVideoAutoNegativeStage)
    stage.backend = _FakeBackend([])
    batch = SimpleNamespace(
        prompt='{"comprehensive_description": "a night scene"}',
        negative_prompt=negative_prompt,
        num_frames=121,
        fps=24,
        image_path=None,
    )

    stage.forward(batch, None)

    assert batch.negative_prompt == negative_prompt
    assert stage.backend.calls == []


def test_auto_negative_hints_match_whole_words_only():
    default = {
        "universal_negative": {
            "visual_quality": [
                "low quality",
                "underexposed",
                "subject hidden in darkness",
                "crushed blacks",
            ]
        }
    }

    def survivors(caption):
        return prune_negative(default, default, caption)["universal_negative"][
            "visual_quality"
        ]

    # "knight" contains "night" and "three-dimensional" contains "dim".
    assert (
        survivors("A knight in bright daylight")
        == default["universal_negative"]["visual_quality"]
    )
    assert (
        survivors("A bright three-dimensional sculpture")
        == default["universal_negative"]["visual_quality"]
    )

    assert survivors("A dim night scene") == ["low quality"]
    assert survivors("dimly lit room") == ["low quality"]
    # Multi-word and hyphenated hints still match.
    assert survivors("shot in low-light") == ["low quality"]
    assert survivors("a low light interior") == ["low quality"]


def test_auto_negative_leaves_a_free_text_negative_alone():
    stage = object.__new__(LingBotVideoAutoNegativeStage)
    stage.backend = _FakeBackend([])
    batch = SimpleNamespace(
        prompt='{"comprehensive_description": "a fox"}',
        negative_prompt="blurry, low quality",
        num_frames=121,
        fps=24,
        image_path=None,
    )

    stage.forward(batch, None)

    assert batch.negative_prompt == "blurry, low quality"
    assert stage.backend.calls == []


def test_rewriting_is_off_until_a_server_is_configured():
    assert LingBotVideoMoEPipelineConfig().rewriter_url is None
