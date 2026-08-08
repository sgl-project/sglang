# SPDX-License-Identifier: Apache-2.0

from contextlib import nullcontext
from types import SimpleNamespace

import PIL.Image
import torch

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
    LingBotVideoRefinerUpscaleStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.lingbot_video_moe import (
    refiner_stages as refiner_stages_module,
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
    pixel = preprocess_condition_image(PIL.Image.new("RGB", (32, 128)), height, width)

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
    try:
        _adjusted(1, [], pipeline_class_name=REFINER_PIPELINE_NAME)
        raise AssertionError("expected ValueError for a single-frame refiner request")
    except ValueError as err:
        assert "single-frame" in str(err)


def test_refiner_accepts_video_requests():
    params = _adjusted(81, [], pipeline_class_name=REFINER_PIPELINE_NAME)

    assert params.data_type == DataType.VIDEO
    assert params.num_frames > 1


def test_video_request_keeps_the_video_negative_prompt():
    params = _adjusted(81, [])

    assert params.negative_prompt == DEFAULT_NEGATIVE_PROMPT
    assert params.data_type == DataType.VIDEO


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


def test_refiner_rejects_a_threshold_outside_the_unit_range():
    for t_thresh in (0.0, 1.5):
        try:
            compute_refiner_sigmas(
                sigma_max=1.0,
                sigma_min=0.0,
                num_inference_steps=8,
                shift=3.0,
                t_thresh=t_thresh,
            )
            raise AssertionError(f"expected ValueError for t_thresh={t_thresh}")
        except ValueError:
            pass


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

    path = pipeline._resolve_component_path(None, "transformer_2", "transformer_2")

    assert path == "/models/lingbot/refiner"


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

    def encode(pixels, server_args, *, vae_dtype, generator):
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
    assert "视频时长：5 秒" in build_expand_prompt("t2v", "一只红狐狸", 5)


def test_map_prompt_feeds_the_expansion_back():
    mapped = build_map_prompt("t2v", "A red fox trots through snow.", 5)

    assert "DETAILED CAPTION:\nA red fox trots through snow." in mapped
    assert mapped.endswith("Output the JSON now.")


def test_caption_parsing_tolerates_fences_and_prose():
    assert parse_caption('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_caption('Here you go: {"a": 1} hope that helps') == {"a": 1}
    assert parse_caption("no json here") is None
    assert parse_caption('["not", "an", "object"]') is None


def _rewrite_stage(replies):
    stage = object.__new__(LingBotVideoPromptRewriteStage)
    stage.expand_model = "expand"
    stage.map_model = "map"
    stage.calls = []

    def chat(model, text, image):
        stage.calls.append((model, image))
        return replies.pop(0)

    stage._chat = chat
    return stage


def test_rewriting_replaces_the_prompt_with_a_compact_caption():
    stage = _rewrite_stage(["A red fox trots through snow.", '{"b": 2, "a": 1}'])
    batch = SimpleNamespace(prompt="a red fox", num_frames=121, fps=24, image_path=None)

    stage.forward(batch, None)

    assert batch.prompt == '{"b":2,"a":1}'
    assert [model for model, _ in stage.calls] == ["expand", "map"]
    assert all(image is None for _, image in stage.calls)


def test_rewriting_keeps_the_prompt_when_no_caption_comes_back():
    stage = _rewrite_stage(["expanded", "sorry, I cannot"])
    batch = SimpleNamespace(prompt="a red fox", num_frames=121, fps=24, image_path=None)

    stage.forward(batch, None)

    assert batch.prompt == "a red fox"


def test_rewriting_leaves_a_structured_caption_alone():
    stage = _rewrite_stage([])
    caption = '{"comprehensive_description": "..."}'
    batch = SimpleNamespace(prompt=caption, num_frames=121, fps=24, image_path=None)

    stage.forward(batch, None)

    assert batch.prompt == caption
    assert stage.calls == []


def test_rewriting_is_off_until_a_server_is_configured():
    assert LingBotVideoMoEPipelineConfig().rewriter_url is None
