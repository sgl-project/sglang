# SPDX-License-Identifier: Apache-2.0
"""Check SR prompt admission and request fields without running a GPU scheduler.

JSON and multipart use TestClient to exercise FastAPI field parsing. The
full-checkpoint test_server_kandinsky6_sr.py also exercises actual file uploads.
"""

import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from sglang.multimodal_gen.configs.pipeline_configs.kandinsky6_sr import (
    Kandinsky6SRPipelineConfig,
)
from sglang.multimodal_gen.configs.sample.kandinsky6_sr import (
    Kandinsky6SRSamplingParams,
)
from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.runtime.entrypoints.openai import video_api
from sglang.multimodal_gen.runtime.entrypoints.openai.utils import (
    resolve_sampling_params_cls,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.video_api import (
    _build_video_sampling_params,
    _multipart_video_extras,
)

_SR_EXTRA_FIELDS = frozenset(
    {
        "sr_resolution_scale",
        "sr_tiles_batch_size",
        "sr_tile_min_overlap",
        "sr_target_resolution",
        "sr_target_resize_mode",
    }
)


def _fake_server_args(**overrides) -> SimpleNamespace:
    base = dict(
        pipeline_config=Kandinsky6SRPipelineConfig(),
        pipeline_class_name="Kandinsky6SRPipeline",
        backend="auto",
        model_id=None,
        model_path="kandinskylab/Kandinsky-6.0-VSR-5s-Diffusers",
        served_model_name="kandinsky6-sr-test",
        input_save_path=os.path.expanduser(
            "~/.cache/k6_agent_sg-sr-gaps-fix/http_request_test/uploads"
        ),
        output_path=os.path.expanduser(
            "~/.cache/k6_agent_sg-sr-gaps-fix/http_request_test/output"
        ),
        attention_backend_config={},
        enable_trace=False,
        comfyui_mode=False,
        num_gpus=1,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _patched_global_server_args(server_args):
    return patch(
        "sglang.multimodal_gen.runtime.server_args.server_args._global_server_args",
        server_args,
    )


# --------------------------------------------------------------------------- #
# The two declared pieces (SamplingParams-level)
# --------------------------------------------------------------------------- #
def test_prompt_optional_defaults_false_and_sr_overrides_it():
    """A generic pipeline keeps requiring ``prompt``; only the SR params opt out."""
    assert SamplingParams.prompt_optional is False
    assert Kandinsky6SRSamplingParams.prompt_optional is True


@pytest.mark.parametrize(
    "sampling_params_cls, status",
    [(Kandinsky6SRSamplingParams, 200), (SamplingParams, 400)],
)
def test_json_prompt_requirement_matches_pipeline(
    tmp_path, sampling_params_cls, status
):
    app = FastAPI()
    app.include_router(video_api.router)
    server_args = _fake_server_args(
        input_save_path=str(tmp_path / "uploads"), output_path=str(tmp_path / "outputs")
    )
    dispatch = AsyncMock()
    with (
        _patched_global_server_args(server_args),
        patch.object(
            video_api, "resolve_sampling_params_cls", return_value=sampling_params_cls
        ),
        patch.object(video_api, "_dispatch_job_async", dispatch),
        TestClient(app) as client,
    ):
        response = client.post(
            "/v1/videos",
            json={"video_path": "unused.mp4", "sr_resolution_scale": 4},
        )
    assert response.status_code == status, response.text
    if status == 200:
        assert response.json()["status"] == "queued"
        dispatch.assert_awaited_once()
        _, batch = dispatch.call_args.args
        assert batch.prompt == ""
        assert batch.sampling_params.sr_resolution_scale == 4
    else:
        dispatch.assert_not_called()


def test_sr_declares_its_cli_flags_as_multipart_extra_fields():
    assert Kandinsky6SRSamplingParams.video_request_extra_fields() == _SR_EXTRA_FIELDS
    # The base default (what every other model still gets) stays empty.
    assert SamplingParams.video_request_extra_fields() == frozenset()


# --------------------------------------------------------------------------- #
# The real multipart-field allowlist (video_api._multipart_video_extras)
# --------------------------------------------------------------------------- #
def test_multipart_extras_keep_sr_fields_only_for_the_sr_params_class():
    raw_form = {
        "video_path": "unused.mp4",
        "sr_resolution_scale": "4",
        "sr_tiles_batch_size": "3",
        "not_a_declared_field": "dropped",
    }
    sr_extras = _multipart_video_extras(
        raw_form,
        extra_body=None,
        extra_params=None,
        sampling_params_cls=Kandinsky6SRSamplingParams,
    )
    assert sr_extras["sr_resolution_scale"] == 4  # JSON-decoded, not the raw "4"
    assert sr_extras["sr_tiles_batch_size"] == 3
    assert "not_a_declared_field" not in sr_extras

    # A model without the override (the pre-fix behaviour) drops the sr_* fields too.
    base_extras = _multipart_video_extras(
        raw_form,
        extra_body=None,
        extra_params=None,
        sampling_params_cls=SamplingParams,
    )
    assert "sr_resolution_scale" not in base_extras
    assert "sr_tiles_batch_size" not in base_extras


# --------------------------------------------------------------------------- #
# The registry-backed class resolution the admission gate reads
# --------------------------------------------------------------------------- #
def test_video_sampling_params_cls_resolves_to_sr_for_the_sr_pipeline():
    server_args = _fake_server_args()
    assert resolve_sampling_params_cls(server_args) is Kandinsky6SRSamplingParams


# --------------------------------------------------------------------------- #
# End-to-end request parsing: a prompt-less request field set -> sampling params
# --------------------------------------------------------------------------- #
def test_build_video_sampling_params_accepts_no_prompt_and_carries_sr_fields():
    """The real kwargs-building code (``_build_video_sampling_params``), given a
    request built the way the multipart branch builds one for a prompt-optional
    pipeline (``prompt=prompt or ""`` + the surviving sr_* extras), must construct a
    real ``Kandinsky6SRSamplingParams`` with those fields set and no prompt error."""
    from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import (
        VideoGenerationsRequest,
    )

    server_args = _fake_server_args()
    raw_form = {
        "video_path": "unused.mp4",
        "sr_resolution_scale": "4",
        "sr_tiles_batch_size": "3",
    }
    extras = _multipart_video_extras(
        raw_form,
        extra_body=None,
        extra_params=None,
        sampling_params_cls=Kandinsky6SRSamplingParams,
    )
    req = VideoGenerationsRequest(
        prompt="",  # what video_api.py substitutes for a missing prompt
        video_path=extras.pop("video_path"),
        **extras,
    )

    with _patched_global_server_args(server_args):
        sampling_params = _build_video_sampling_params("gap1-request-id", req)

    assert isinstance(sampling_params, Kandinsky6SRSamplingParams)
    assert sampling_params.prompt == ""
    assert sampling_params.sr_resolution_scale == 4
    assert sampling_params.sr_tiles_batch_size == 3


# --------------------------------------------------------------------------- #
# The full endpoint with real multipart parsing
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "sampling_params_cls, status",
    [(Kandinsky6SRSamplingParams, 200), (SamplingParams, 400)],
)
def test_multipart_sr_request_without_prompt_is_accepted(
    tmp_path, sampling_params_cls, status
):
    app = FastAPI()
    app.include_router(video_api.router)
    server_args = _fake_server_args(
        input_save_path=str(tmp_path / "uploads"), output_path=str(tmp_path / "outputs")
    )
    dispatch = AsyncMock()
    with (
        _patched_global_server_args(server_args),
        patch.object(
            video_api, "resolve_sampling_params_cls", return_value=sampling_params_cls
        ),
        patch.object(video_api, "_dispatch_job_async", dispatch),
        TestClient(app) as client,
    ):
        response = client.post(
            "/v1/videos",
            files={
                "video_path": (None, "unused.mp4"),
                "sr_resolution_scale": (None, "4"),
                "sr_tiles_batch_size": (None, "3"),
            },
        )
    assert response.status_code == status, response.text
    if status == 200:
        assert response.json()["status"] == "queued"
        dispatch.assert_awaited_once()
        _, batch = dispatch.call_args.args
        assert batch.prompt == ""
        assert batch.sampling_params.video_path == "unused.mp4"
        assert batch.sampling_params.sr_resolution_scale == 4
        assert batch.sampling_params.sr_tiles_batch_size == 3
    else:
        dispatch.assert_not_called()
