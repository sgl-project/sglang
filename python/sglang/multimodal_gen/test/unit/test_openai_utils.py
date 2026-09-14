# SPDX-License-Identifier: Apache-2.0

import asyncio
import io
from types import SimpleNamespace

from starlette.datastructures import UploadFile as StarletteUploadFile

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.runtime.entrypoints.openai import utils as openai_utils
from sglang.multimodal_gen.runtime.entrypoints.openai.utils import (
    _parse_size_or_raise,
    _save_upload_to_path,
    _validate_positive_int,
    build_sampling_params,
)


def test_save_upload_to_path_accepts_starlette_upload_file(tmp_path):
    upload = StarletteUploadFile(
        io.BytesIO(b"image-bytes"),
        filename="input.png",
    )
    target_path = tmp_path / "input.png"

    saved_path = asyncio.run(_save_upload_to_path(upload, str(target_path)))

    assert saved_path == str(target_path)
    assert target_path.read_bytes() == b"image-bytes"


def test_parse_size_or_raise_accepts_positive_size():
    assert _parse_size_or_raise("512x768") == (512, 768)


def test_parse_size_or_raise_rejects_malformed_size():
    try:
        _parse_size_or_raise("not-a-size")
    except Exception as exc:
        assert exc.status_code == 400
        assert "positive WIDTHxHEIGHT" in exc.detail
    else:
        raise AssertionError("expected bad request")


def test_parse_size_or_raise_rejects_non_positive_size():
    try:
        _parse_size_or_raise("0x512")
    except Exception as exc:
        assert exc.status_code == 400
        assert "positive WIDTHxHEIGHT" in exc.detail
    else:
        raise AssertionError("expected bad request")


def test_validate_positive_int_rejects_non_positive_sampling_fields():
    try:
        _validate_positive_int({"num_frames": 0}, "num_frames")
    except Exception as exc:
        assert exc.status_code == 400
        assert "num_frames must be positive" in exc.detail
    else:
        raise AssertionError("expected bad request")


def test_build_sampling_params_resolves_size_and_explicit_dimensions(monkeypatch):
    server_args = SimpleNamespace(model_path="zai-org/GLM-Image")
    monkeypatch.setattr(openai_utils, "get_global_server_args", lambda: server_args)
    captured = {}

    def fake_from_user_sampling_params_args(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(
        SamplingParams,
        "from_user_sampling_params_args",
        fake_from_user_sampling_params_args,
    )

    cases = [
        (
            {"size": "500x500", "width": None, "height": None},
            (500, 500),
        ),
        (
            {"size": "1024x1024", "width": None, "height": 600},
            (1024, 600),
        ),
        (
            {"size": "500x500", "width": None, "height": 600},
            (500, 600),
        ),
        (
            {"size": "500x500", "width": 600, "height": None},
            (600, 500),
        ),
        (
            {"size": "500x500", "width": 600, "height": 700},
            (600, 700),
        ),
    ]

    for request_fields, expected in cases:
        captured.clear()

        build_sampling_params("request-id", **request_fields)

        assert (captured["width"], captured["height"]) == expected
