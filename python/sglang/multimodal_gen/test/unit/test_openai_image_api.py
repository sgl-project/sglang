import os

from fastapi import HTTPException
from PIL import Image

from sglang.multimodal_gen.configs.sample.glmimage import GlmImageSamplingParams
from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.runtime.entrypoints.openai.image_api import (
    _build_image_response_kwargs,
    _fallback_image_urls,
    _get_response_resize,
    _raise_if_image_variant_not_found,
    _runtime_sampling_quality,
    _select_image_variant_cloud_url,
    _select_image_variant_path,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch


def test_url_response_returns_one_item_per_output_path():
    paths = ["first.png", "second.png"]

    response = _build_image_response_kwargs(
        paths,
        "url",
        "a lantern",
        "req-123",
        OutputBatch(),
        cloud_urls=["https://cdn.example/first.png", "https://cdn.example/second.png"],
        fallback_urls=_fallback_image_urls("req-123", len(paths), True),
        is_persistent=True,
    )

    assert [item.url for item in response["data"]] == [
        "https://cdn.example/first.png",
        "https://cdn.example/second.png",
    ]
    assert [item.file_path for item in response["data"]] == [
        os.path.abspath("first.png"),
        os.path.abspath("second.png"),
    ]


def test_runtime_sampling_quality_preserves_the_openai_default():
    assert _runtime_sampling_quality(None) is None
    assert _runtime_sampling_quality("auto") is None
    assert _runtime_sampling_quality("lossless") == "lossless"
    assert _runtime_sampling_quality("high") == "high"


def test_image_response_includes_resize_for_every_output():
    response = _build_image_response_kwargs(
        ["first.png", "second.png"],
        "b64_json",
        "a lantern",
        "req-123",
        OutputBatch(),
        b64_list=["first", "second"],
        resize="1280x736",
    )

    assert [item.resize for item in response["data"]] == [
        "1280x736",
        "1280x736",
    ]


def test_response_resize_is_only_populated_for_glm_image():
    glm_sampling = GlmImageSamplingParams(width=1280, height=736)

    assert _get_response_resize(glm_sampling) == "1280x736"
    assert _get_response_resize(SamplingParams(width=1280, height=736)) is None


def test_response_resize_uses_actual_generated_image_size(tmp_path):
    output_path = tmp_path / "output.png"
    Image.new("RGB", (1280, 736)).save(output_path)
    glm_sampling = GlmImageSamplingParams(image_path="input.png")

    assert _get_response_resize(glm_sampling, str(output_path)) == "1280x736"


def test_response_resize_prefers_final_output_over_sampling_canvas(tmp_path):
    output_path = tmp_path / "upscaled.png"
    Image.new("RGB", (2560, 1472)).save(output_path)
    glm_sampling = GlmImageSamplingParams(width=1280, height=736)

    assert _get_response_resize(glm_sampling, str(output_path)) == "2560x1472"


def test_url_response_uses_variant_fallback_urls_for_multiple_persistent_outputs():
    paths = ["first.png", "second.png"]

    response = _build_image_response_kwargs(
        paths,
        "url",
        "a lantern",
        "req-123",
        OutputBatch(),
        fallback_urls=_fallback_image_urls("req-123", len(paths), True),
        is_persistent=True,
    )

    assert [item.url for item in response["data"]] == [
        "/v1/images/req-123/content?variant=0",
        "/v1/images/req-123/content?variant=1",
    ]


def test_select_image_variant_path_reads_indexed_file_paths():
    item = {
        "file_path": "first.png",
        "file_paths": ["first.png", "second.png"],
    }

    assert _select_image_variant_path(item, None) == "first.png"
    assert _select_image_variant_path(item, "1") == "second.png"


def test_raise_if_image_variant_not_found_handles_out_of_range_variant():
    item = {
        "file_path": "first.png",
        "file_paths": ["first.png", "second.png"],
    }

    try:
        _raise_if_image_variant_not_found(item, "5")
    except HTTPException as exc:
        assert exc.status_code == 404
        assert exc.detail == "Image variant 5 not found"
    else:
        raise AssertionError("Expected HTTPException")


def test_select_image_variant_path_returns_none_for_cloud_only_variant():
    item = {
        "file_path": None,
        "file_paths": [None, "second.png"],
        "urls": ["https://cdn.example/first.png"],
    }

    assert _select_image_variant_path(item, "0") is None
    assert _select_image_variant_path(item, "1") == "second.png"


def test_select_image_variant_cloud_url_keeps_variant_alignment():
    item = {
        "url": "https://cdn.example/first.png",
        "urls": [
            "https://cdn.example/first.png",
            None,
            "https://cdn.example/third.png",
        ],
    }

    assert (
        _select_image_variant_cloud_url(item, None) == "https://cdn.example/first.png"
    )
    assert _select_image_variant_cloud_url(item, "1") is None
    assert _select_image_variant_cloud_url(item, "2") == "https://cdn.example/third.png"


def test_select_image_variant_cloud_url_falls_back_to_single_url():
    item = {"url": "https://cdn.example/only.png"}

    assert _select_image_variant_cloud_url(item, None) == "https://cdn.example/only.png"
    assert _select_image_variant_cloud_url(item, "0") == "https://cdn.example/only.png"
    assert _select_image_variant_cloud_url(item, "1") is None


def test_select_image_variant_path_falls_back_to_single_file_path():
    item = {"file_path": "only.png"}

    assert _select_image_variant_path(item, None) == "only.png"
    assert _select_image_variant_path(item, "0") == "only.png"
