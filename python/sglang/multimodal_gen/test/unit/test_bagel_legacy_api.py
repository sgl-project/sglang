# SPDX-License-Identifier: Apache-2.0

import asyncio
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from sglang.multimodal_gen.configs.pipeline_configs.bagel import (
    BagelEditPipelineConfig,
    BagelPipelineConfig,
    BagelThinkingPipelineConfig,
    BagelUnderstandingPipelineConfig,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.image_api import (
    _reject_legacy_bagel_generation_fields,
    generations,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import (
    ImageGenerationsRequest,
)


def _bagel_server_args(**overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "model_path": "ByteDance-Seed/BAGEL-7B-MoT",
        "output_path": None,
        "pipeline_class_name": None,
        "pipeline_config": BagelPipelineConfig(),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize(
    ("field_name", "value", "replacement"),
    [
        ("enable_think", True, "BagelThinkingPipeline"),
        ("enable_editing", True, "/v1/images/edits"),
        ("enable_understanding", True, "/v1/chat/completions"),
        ("cfg_img_scale", 2.0, "true_cfg_scale"),
        ("image_path", "/tmp/input.png", "image_url"),
        ("max_understanding_tokens", 256, "max_completion_tokens"),
    ],
)
def test_active_legacy_bagel_fields_return_actionable_400(
    field_name: str, value: object, replacement: str
) -> None:
    request = ImageGenerationsRequest(prompt="test", **{field_name: value})

    with pytest.raises(HTTPException) as error:
        _reject_legacy_bagel_generation_fields(request, _bagel_server_args())

    assert error.value.status_code == 400
    assert field_name in error.value.detail
    assert replacement in error.value.detail


def test_legacy_bagel_guard_allows_inactive_defaults_and_current_fields() -> None:
    request = ImageGenerationsRequest(
        prompt="test",
        enable_think=False,
        enable_editing=False,
        enable_understanding=False,
        cfg_img_scale=None,
        image_path="",
        max_understanding_tokens=None,
        enable_taylorseer=True,
        max_think_tokens=64,
        think_do_sample=False,
        think_temperature=0.3,
    )

    _reject_legacy_bagel_generation_fields(request, _bagel_server_args())


@pytest.mark.parametrize(
    "pipeline_config_cls",
    [
        BagelPipelineConfig,
        BagelThinkingPipelineConfig,
        BagelUnderstandingPipelineConfig,
        BagelEditPipelineConfig,
    ],
)
def test_legacy_guard_covers_every_bagel_pipeline_config(
    pipeline_config_cls: type[BagelPipelineConfig],
) -> None:
    request = ImageGenerationsRequest(prompt="test", enable_think=True)
    server_args = _bagel_server_args(
        pipeline_class_name=None,
        pipeline_config=pipeline_config_cls(),
    )

    with pytest.raises(HTTPException, match="enable_think"):
        _reject_legacy_bagel_generation_fields(request, server_args)


@pytest.mark.parametrize(
    "container_name", ["extra_body", "extra_json", "extra_args", "extra_params"]
)
def test_legacy_bagel_guard_reads_nested_extra_containers(
    container_name: str,
) -> None:
    request = ImageGenerationsRequest(
        prompt="test",
        enable_editing=False,
        cfg_img_scale=None,
        **{container_name: {"enable_editing": True, "cfg_img_scale": 2.0}},
    )
    server_args = _bagel_server_args(
        pipeline_class_name="BagelEditPipeline",
        pipeline_config=object(),
    )

    with pytest.raises(HTTPException) as error:
        _reject_legacy_bagel_generation_fields(request, server_args)

    assert "cfg_img_scale, enable_editing" in error.value.detail


def test_legacy_bagel_guard_does_not_change_other_pipelines() -> None:
    request = ImageGenerationsRequest(
        prompt="test",
        enable_think=True,
        enable_editing=True,
        enable_understanding=True,
        cfg_img_scale=2.0,
        image_path="/tmp/input.png",
        max_understanding_tokens=256,
    )
    server_args = SimpleNamespace(
        pipeline_class_name="OtherPipeline",
        pipeline_config=object(),
    )

    _reject_legacy_bagel_generation_fields(request, server_args)


def test_generation_handler_rejects_legacy_mode_before_sampling() -> None:
    request = ImageGenerationsRequest(prompt="test", enable_understanding=True)
    raw_request = Request({"type": "http", "headers": []})
    sampling_mock = Mock(
        side_effect=AssertionError("legacy mode must fail before sampling")
    )

    with patch(
        "sglang.multimodal_gen.runtime.entrypoints.openai.image_api."
        "get_global_server_args",
        return_value=_bagel_server_args(),
    ), patch(
        "sglang.multimodal_gen.runtime.entrypoints.openai.image_api."
        "build_sampling_params",
        sampling_mock,
    ):
        with pytest.raises(HTTPException) as error:
            asyncio.run(generations(request, raw_request))

    assert error.value.status_code == 400
    sampling_mock.assert_not_called()
