from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from sglang.multimodal_gen.configs.pipeline_configs.qwen_image import (
    QwenImageEditPipelineConfig,
    QwenImageEditPlusPipelineConfig,
)
from sglang.multimodal_gen.configs.sample.qwenimage import QwenImageSamplingParams
from sglang.multimodal_gen.runtime.managers.scheduler import Scheduler
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.input_validation import (
    InputValidationStage,
)


def _scheduler():
    scheduler = object.__new__(Scheduler)
    scheduler.server_args = SimpleNamespace(
        pipeline_config=QwenImageEditPipelineConfig()
    )
    return scheduler


def _request(request_id, prompt, image_path, *, width=1024):
    params = QwenImageSamplingParams(
        prompt=prompt,
        image_path=image_path,
        width=width,
        height=1024,
        seed=7,
    )
    return Req(request_id=request_id, sampling_params=params)


def test_scheduler_merges_distinct_single_image_edit_requests():
    scheduler = _scheduler()
    requests = [
        _request("first", "turn it red", "first.png"),
        _request("second", "turn it blue", "second.png"),
    ]

    merged = scheduler._try_merge_generation_reqs(requests)

    assert merged is not None
    assert merged.prompt == ["turn it red", "turn it blue"]
    assert merged.image_path == ["first.png", "second.png"]
    assert merged.extra["dynamic_batch_image_conditioning"] is True


def test_scheduler_rejects_multi_reference_and_mismatched_output_shape():
    scheduler = _scheduler()
    single = _request("first", "turn it red", "first.png")
    multiple = _request("second", "turn it blue", ["a.png", "b.png"])
    different_width = _request("third", "turn it green", "third.png", width=768)
    multiple_outputs = _request("fourth", "make it purple", "fourth.png")
    multiple_outputs.num_outputs_per_prompt = 2

    assert scheduler._try_merge_generation_reqs([single, multiple]) is None
    assert scheduler._try_merge_generation_reqs([single, different_width]) is None
    assert scheduler._try_merge_generation_reqs([single, multiple_outputs]) is None


def test_only_standard_qwen_image_edit_opts_in():
    assert QwenImageEditPipelineConfig().supports_dynamic_batching()
    assert not QwenImageEditPlusPipelineConfig().supports_dynamic_batching()


def test_dynamic_image_prompt_mapping_is_one_image_per_prompt():
    config = QwenImageEditPipelineConfig()
    images = [Image.new("RGB", (32, 32)), Image.new("RGB", (32, 32))]
    batch = SimpleNamespace(
        prompt=["first edit", "second edit"],
        negative_prompt=["", ""],
        condition_image=images,
        extra={"dynamic_batch_image_conditioning": True},
    )

    kwargs = config.prepare_image_processor_kwargs(batch)

    assert kwargs["per_prompt_images"] == [[images[0]], [images[1]]]
    assert len(kwargs["text"]) == 2

    negative_kwargs = config.prepare_image_processor_kwargs(batch, neg=True)
    assert negative_kwargs["per_prompt_images"] == [[images[0]], [images[1]]]
    assert len(negative_kwargs["text"]) == 2


def test_dynamic_condition_latents_merge_on_batch_dimension():
    config = QwenImageEditPipelineConfig()
    first = torch.full((1, 3, 4), 1.0)
    second = torch.full((1, 3, 4), 2.0)
    batch = SimpleNamespace(extra={"dynamic_batch_image_conditioning": True})

    merged = config.merge_condition_image_latents([first, second], batch)

    assert merged.shape == (2, 3, 4)
    torch.testing.assert_close(merged[0], first[0])
    torch.testing.assert_close(merged[1], second[0])


def test_dynamic_batch_condition_images_require_one_equal_size_image_per_prompt():
    batch = SimpleNamespace(
        condition_image=[Image.new("RGB", (32, 32)), Image.new("RGB", (64, 32))],
        batch_size=2,
    )

    with pytest.raises(ValueError, match="same processed size"):
        InputValidationStage._validate_dynamic_batch_condition_images(batch)


def test_dynamic_edit_conditioning_builds_one_shape_per_request():
    config = QwenImageEditPipelineConfig()
    prompt_embeds = [torch.zeros(2, 5, 8)]
    batch = SimpleNamespace(
        latents=torch.zeros(2, 16, 8),
        height=1024,
        width=1024,
        original_condition_image_size=(1024, 1024),
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=None,
        prompt_embeds_mask=None,
        negative_prompt_embeds_mask=None,
        prompt_seq_lens=[[5, 5]],
        negative_prompt_seq_lens=None,
        extra={
            "dynamic_batch_condition_image_sizes": [(1024, 1024), (1024, 1024)]
        },
    )

    kwargs = config._prepare_edit_cond_kwargs(
        batch, prompt_embeds, None, "cpu", torch.float32
    )

    assert len(kwargs["img_shapes"]) == 2
    assert kwargs["img_shapes"][0] == kwargs["img_shapes"][1]
    assert kwargs["txt_seq_lens"] == [5, 5]
