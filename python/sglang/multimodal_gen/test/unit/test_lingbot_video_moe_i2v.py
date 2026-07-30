# SPDX-License-Identifier: Apache-2.0

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
from sglang.multimodal_gen.runtime.loader.component_loaders.component_loader import (
    component_name_to_loader_cls,
)
from sglang.multimodal_gen.runtime.loader.utils import _normalize_component_type
from sglang.multimodal_gen.runtime.pipelines.lingbot_video_moe import (
    LingBotVideoPipeline,
)
from sglang.multimodal_gen.runtime.pipelines.lingbot_video_moe_refiner import (
    LingBotVideoRefinerPipeline,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.lingbot_video_moe import (
    LingBotVideoDenoisingStage,
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


def test_refiner_pipeline_declares_the_second_transformer():
    assert "transformer_2" in LingBotVideoRefinerPipeline._required_config_modules


def test_refiner_weights_resolve_to_the_refiner_subfolder():
    pipeline = object.__new__(LingBotVideoRefinerPipeline)
    pipeline.model_path = "/models/lingbot"

    path = pipeline._resolve_component_path(None, "transformer_2", "transformer_2")

    assert path == "/models/lingbot/refiner"


def test_refiner_component_name_selects_the_transformer_loader():
    # The loader registry is keyed on the component name, so the refiner must keep
    # the transformer_2 name rather than borrow its subfolder name.
    assert _normalize_component_type("transformer_2") == "transformer"
    assert "transformer" in component_name_to_loader_cls
    assert "refiner" not in component_name_to_loader_cls


def test_refiner_config_defaults_match_the_reference_scripts():
    config = LingBotVideoMoEPipelineConfig()

    assert (config.refiner_height, config.refiner_width) == (1088, 1920)
    assert config.refiner_num_inference_steps == 8
    assert config.refiner_t_thresh == 0.85
    assert config.refiner_sigma_tail_steps == 2


def test_single_pass_pipeline_adds_no_refiner_stages():
    pipeline = object.__new__(LingBotVideoPipeline)

    assert pipeline._maybe_add_refiner_stages(None) is None
