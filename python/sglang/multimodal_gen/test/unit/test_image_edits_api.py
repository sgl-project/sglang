# SPDX-License-Identifier: Apache-2.0

import asyncio
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException, UploadFile
from starlette.requests import Request

from sglang.multimodal_gen.runtime.entrypoints.openai.image_api import edits


class _CapturedSampling(Exception):
    pass


def _call_edits(**sources) -> None:
    request = Request({"type": "http", "headers": []})
    asyncio.run(
        edits(
            raw_request=request,
            image=sources.get("image"),
            image_array=sources.get("image_array"),
            url=sources.get("url"),
            url_array=sources.get("url_array"),
            prompt="edit",
            mask=None,
            model=None,
            n=1,
            response_format=None,
            size="512x1024",
            output_format="png",
            background="auto",
            seed=9,
            generator_device="cuda",
            user=None,
            negative_prompt=None,
            guidance_scale=4.0,
            true_cfg_scale=2.0,
            num_inference_steps=8,
            output_quality="default",
            output_compression=None,
            enable_teacache=False,
            enable_taylorseer=sources.get("enable_taylorseer"),
            enable_upscaling=False,
            upscaling_model_path=None,
            upscaling_scale=4,
            num_frames=1,
        )
    )


def test_edits_rejects_mask_instead_of_silently_ignoring_it() -> None:
    request = Request({"type": "http", "headers": []})
    mask = UploadFile(filename="mask.png", file=BytesIO(b"not-used"))

    with patch(
        "sglang.multimodal_gen.runtime.entrypoints.openai.image_api."
        "get_global_server_args",
        return_value=SimpleNamespace(pipeline_class_name="BagelEditPipeline"),
    ):
        with pytest.raises(HTTPException, match="masks are not supported") as error:
            asyncio.run(edits(raw_request=request, prompt="edit", mask=mask))

    assert error.value.status_code == 400


@pytest.mark.parametrize("source_field", ["image", "image_array", "url", "url_array"])
def test_edits_sources_flow_into_one_sampling_image_path(source_field: str) -> None:
    if source_field.startswith("image"):
        source = UploadFile(filename="source.png", file=BytesIO(b"image"))
    else:
        source = "https://example.com/source.png"
    build_sampling = patch(
        "sglang.multimodal_gen.runtime.entrypoints.openai.image_api."
        "build_sampling_params",
        side_effect=_CapturedSampling,
    )
    save_image = AsyncMock(return_value="/tmp/saved-source.png")

    with (
        patch(
            "sglang.multimodal_gen.runtime.entrypoints.openai.image_api."
            "get_global_server_args",
            return_value=SimpleNamespace(input_save_path=None, output_path=None),
        ),
        patch(
            "sglang.multimodal_gen.runtime.entrypoints.openai.image_api."
            "save_image_to_path",
            save_image,
        ),
        build_sampling as sampling_mock,
    ):
        with pytest.raises(_CapturedSampling):
            _call_edits(
                **{source_field: [source]},
                enable_taylorseer=True,
            )

    kwargs = sampling_mock.call_args.kwargs
    assert kwargs["image_path"] == ["/tmp/saved-source.png"]
    assert kwargs["size"] == "512x1024"
    assert kwargs["seed"] == 9
    assert kwargs["guidance_scale"] == 4.0
    assert kwargs["true_cfg_scale"] == 2.0
    assert kwargs["enable_taylorseer"] is True
    save_image.assert_awaited_once()


def test_edits_normalizes_explicit_false_taylorseer_control() -> None:
    source = UploadFile(filename="source.png", file=BytesIO(b"image"))
    save_image = AsyncMock(return_value="/tmp/saved-source.png")

    with (
        patch(
            "sglang.multimodal_gen.runtime.entrypoints.openai.image_api."
            "get_global_server_args",
            return_value=SimpleNamespace(input_save_path=None, output_path=None),
        ),
        patch(
            "sglang.multimodal_gen.runtime.entrypoints.openai.image_api."
            "save_image_to_path",
            save_image,
        ),
        patch(
            "sglang.multimodal_gen.runtime.entrypoints.openai.image_api."
            "build_sampling_params",
            side_effect=_CapturedSampling,
        ) as sampling_mock,
    ):
        with pytest.raises(_CapturedSampling):
            _call_edits(image=[source], enable_taylorseer=False)

    assert sampling_mock.call_args.kwargs["enable_taylorseer"] is None


def test_edits_requires_an_upload_or_url() -> None:
    with patch(
        "sglang.multimodal_gen.runtime.entrypoints.openai.image_api."
        "get_global_server_args",
        return_value=SimpleNamespace(input_save_path=None, output_path=None),
    ):
        with pytest.raises(HTTPException, match="image.*url") as error:
            _call_edits()

    assert error.value.status_code == 422
