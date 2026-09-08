"""Resolution-contract tests for HunyuanImage-3 image generation and editing."""

from types import SimpleNamespace

import torch
from PIL import Image

from sglang.multimodal_gen.configs.pipeline_configs.hunyuan_image3 import (
    HunyuanImage3PipelineConfig,
)
from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch, Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.decoding import DecodingStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.input_validation import (
    InputValidationStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.hunyuan_image3.ar_stage import (
    HunyuanImage3AR,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.hunyuan_image3.decoding import (
    HunyuanImage3DecodingStage,
    _build_spatial_plan,
    apply_hunyuan_image3_spatial_plan,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.hunyuan_image3.resolution import (
    OUTPUT_GEOMETRY_EXTRA_KEY,
    build_hunyuan_image3_output_geometry,
    resolve_hunyuan_image3_output_resolution,
)


def test_edit_without_explicit_size_follows_reference_aspect_ratio():
    width, height = resolve_hunyuan_image3_output_resolution(
        width=1280,
        height=720,
        explicit_fields=set(),
        reference_size=(1000, 333),
    )

    assert (width, height) == (1000, 333)


def test_explicit_size_is_not_pre_aligned_before_processor():
    width, height = resolve_hunyuan_image3_output_resolution(
        width=1025,
        height=577,
        explicit_fields={"width", "height"},
    )

    assert (width, height) == (1025, 577)


def test_explicit_size_takes_precedence_over_reference_image():
    width, height = resolve_hunyuan_image3_output_resolution(
        width=1025,
        height=577,
        explicit_fields={"width", "height"},
        reference_size=(1000, 333),
    )

    assert (width, height) == (1025, 577)


def test_one_explicit_dimension_prevents_reference_size_override():
    width, height = resolve_hunyuan_image3_output_resolution(
        width=640,
        height=720,
        explicit_fields={"width"},
        reference_size=(333, 1000),
    )

    assert (width, height) == (640, 720)


def test_text_to_image_without_reference_keeps_raw_request_size():
    width, height = resolve_hunyuan_image3_output_resolution(
        width=1025,
        height=577,
        explicit_fields=set(),
    )

    assert (width, height) == (1025, 577)


def test_generation_saves_requested_ratio_before_selecting_native_bucket(monkeypatch):
    requested_size = (1000, 700)
    native_size = (1216, 832)
    image_info = SimpleNamespace(
        image_width=native_size[0],
        image_height=native_size[1],
        token_width=native_size[0] // 16,
        token_height=native_size[1] // 16,
    )
    processor = SimpleNamespace(
        build_gen_image_info=lambda image_size: image_info,
    )
    request = SimpleNamespace(
        width=requested_size[0],
        height=requested_size[1],
        sampling_params=SimpleNamespace(_explicit_fields={"width", "height"}),
        original_condition_image_size=None,
        guidance_scale=2.5,
        num_inference_steps=50,
        extra={},
    )
    stage = object.__new__(HunyuanImage3AR)
    stage._processor = processor
    monkeypatch.setattr(stage, "_rebuild_image_info", lambda info: info)

    width, height, *_ = stage._resolve_generation_params([request], [None])

    assert (width, height) == native_size
    assert (request.width, request.height) == native_size
    assert request.extra[OUTPUT_GEOMETRY_EXTRA_KEY] == {
        "requested_size": [1000, 700],
        "requested_aspect_ratio": [10, 7],
        "size_mode": "aspect_ratio",
        "strategy": "native_crop",
        "ratio_policy": "exact",
        "crop_anchor": [0.5, 0.5],
        "max_ratio_error": 0.0005,
        "pad_value": 0.0,
        "native_bucket_size": [1216, 832],
    }


def test_native_crop_is_maximum_area_strict_ratio_and_preserves_pixels():
    geometry = build_hunyuan_image3_output_geometry(1000, 700)
    frames = torch.arange(1216 * 832, dtype=torch.float32).reshape(1, 1, 832, 1216)

    plan, metadata = _build_spatial_plan((1216, 832), geometry)
    cropped = apply_hunyuan_image3_spatial_plan(frames, plan)

    assert metadata["crop_box"] == [18, 3, 1198, 829]
    assert metadata["output_size"] == [1180, 826]
    assert metadata["relative_ratio_error"] == 0.0
    assert metadata["resampled"] is False
    assert cropped.shape == (1, 1, 826, 1180)
    assert torch.equal(cropped, frames[..., 3:829, 18:1198])


def test_native_crop_reuses_the_same_plan_for_trajectory_frames():
    geometry = build_hunyuan_image3_output_geometry(1000, 700)
    image = torch.zeros(2, 3, 1, 832, 1216)
    trajectory = torch.ones(2, 3, 1, 832, 1216)

    plan, _ = _build_spatial_plan((1216, 832), geometry)

    assert apply_hunyuan_image3_spatial_plan(image, plan).shape == (2, 3, 1, 826, 1180)
    assert torch.equal(
        apply_hunyuan_image3_spatial_plan(trajectory, plan),
        trajectory[..., 3:829, 18:1198],
    )


def test_decoding_stage_crops_images_and_trajectories_with_one_plan(monkeypatch):
    frames = torch.zeros(2, 3, 1, 832, 1216)
    trajectory = torch.ones(2, 3, 1, 832, 1216)

    def fake_decode(_stage, _batch, _server_args):
        return OutputBatch(output=frames, trajectory_decoded=[trajectory])

    monkeypatch.setattr(DecodingStage, "forward", fake_decode)
    stage = object.__new__(HunyuanImage3DecodingStage)
    batch = Req(sampling_params=SamplingParams(prompt="test", width=1000, height=700))
    batch.extra[OUTPUT_GEOMETRY_EXTRA_KEY] = build_hunyuan_image3_output_geometry(
        1000, 700
    )

    output = stage.forward(batch, SimpleNamespace())

    assert output.output.shape == (2, 3, 1, 826, 1180)
    assert output.trajectory_decoded[0].shape == (2, 3, 1, 826, 1180)
    assert (batch.width, batch.height) == (1180, 826)
    assert batch.extra[OUTPUT_GEOMETRY_EXTRA_KEY]["crop_box"] == [18, 3, 1198, 829]


def test_exact_size_performs_one_uniform_final_resample():
    geometry = build_hunyuan_image3_output_geometry(1000, 700, size_mode="exact_size")
    frames = torch.zeros(1, 3, 1, 832, 1216)

    plan, metadata = _build_spatial_plan((1216, 832), geometry)
    output = apply_hunyuan_image3_spatial_plan(frames, plan)

    assert plan.crop_box == (18, 3, 1198, 829)
    assert plan.resize_size == (1000, 700)
    assert output.shape == (1, 3, 1, 700, 1000)
    assert metadata["resampled"] is True


def test_native_pad_retains_all_decoded_pixels():
    geometry = build_hunyuan_image3_output_geometry(1000, 700, strategy="native_pad")
    frames = torch.ones(1, 1, 832, 1216)

    plan, metadata = _build_spatial_plan((1216, 832), geometry)
    padded = apply_hunyuan_image3_spatial_plan(frames, plan)

    assert plan.padding == (2, 2, 11, 11)
    assert padded.shape == (1, 1, 854, 1220)
    assert torch.equal(padded[..., 11:843, 2:1218], frames)
    assert metadata["retained_pixel_fraction"] == 1.0


def test_multi_output_request_keeps_native_bucket_size(monkeypatch):
    outputs = [
        SimpleNamespace(
            latents=torch.zeros(1, 1, 1, 1, 1),
            width=1216,
            height=832,
        )
        for _ in range(2)
    ]
    stage = object.__new__(HunyuanImage3AR)
    monkeypatch.setattr(stage, "_expand_multi_output", lambda _batch: [object()] * 2)
    monkeypatch.setattr(stage, "_forward_batched", lambda _batches: outputs)
    batch = Req(
        sampling_params=SamplingParams(
            prompt="test",
            width=1000,
            height=700,
            num_outputs_per_prompt=2,
        )
    )

    output = stage.forward(batch, SimpleNamespace())

    assert output is batch
    assert output.latents.shape[0] == 2
    assert (output.width, output.height) == (1216, 832)


def test_pipeline_config_delegates_condition_image_sizing_to_native_stage():
    config = HunyuanImage3PipelineConfig()
    reference = Image.new("RGB", (1000, 333))

    assert config.calculate_condition_image_size(reference, 1280, 720) is None
    assert config.prepare_calculated_size(reference) is None


def test_input_validation_does_not_apply_generic_32_pixel_edit_resize():
    reference = Image.new("RGB", (1000, 333))
    batch = Req(
        sampling_params=SamplingParams(prompt="edit", width=1008, height=336),
        condition_image=reference,
    )
    batch.extra["explicit_fields"] = ["width", "height"]
    server_args = SimpleNamespace(pipeline_config=HunyuanImage3PipelineConfig())

    InputValidationStage().preprocess_condition_image(
        batch,
        server_args,
        condition_image_width=reference.width,
        condition_image_height=reference.height,
    )

    assert batch.condition_image[0].size == (1000, 333)
    assert (batch.width, batch.height) == (1008, 336)
