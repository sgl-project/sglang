# SPDX-License-Identifier: Apache-2.0

import asyncio
import io
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from starlette.datastructures import UploadFile

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.runtime.entrypoints.openai import (
    image_api,
    mesh_api,
    utils,
    video_api,
)
from sglang.multimodal_gen.runtime.utils.image_io import ensure_path_within_root


@pytest.mark.parametrize(
    "filename,expected",
    [
        ("../../evil.txt", "evil.txt"),
        (r"..\..\evil.txt", "evil.txt"),
        ("/tmp/evil.txt", "evil.txt"),
        ("..", "fallback"),
        ("", "fallback"),
        ("unsafe\x00 name.png", "unsafe_name.png"),
    ],
)
def test_sanitize_upload_filename(filename, expected):
    assert utils.sanitize_upload_filename(filename, "fallback") == expected


def test_ensure_path_within_root_rejects_escape(tmp_path):
    root = tmp_path / "uploads"
    with pytest.raises(ValueError, match="escapes"):
        ensure_path_within_root(root / ".." / "evil.txt", root)


@pytest.fixture(params=["upload", "bytes", "base64", "url"])
def image_source(request, monkeypatch):
    if request.param == "upload":
        return UploadFile(io.BytesIO(b"image"), filename="image.png"), ""
    if request.param == "bytes":
        return b"image", ""
    if request.param == "base64":
        return "data:image/png;base64,aW1hZ2U=", ".png"

    client_cls = httpx.AsyncClient
    transport = httpx.MockTransport(
        lambda req: httpx.Response(
            200, headers={"content-type": "image/png"}, content=b"image"
        )
    )
    monkeypatch.setattr(
        utils.httpx,
        "AsyncClient",
        lambda **kwargs: client_cls(transport=transport, **kwargs),
    )
    return "https://example.invalid/image.png", ".png"


def test_save_image_sources_within_uploads_root(tmp_path, image_source):
    source, suffix = image_source
    root = tmp_path / "uploads"
    saved = asyncio.run(
        utils.save_image_to_path(source, str(root / "input"), uploads_root=str(root))
    )
    assert Path(saved) == root / f"input{suffix}"
    assert Path(saved).read_bytes() == b"image"


def test_save_image_sources_reject_escape_before_writing(tmp_path, image_source):
    source, _ = image_source
    root = tmp_path / "uploads"
    with pytest.raises(ValueError, match="escapes"):
        asyncio.run(
            utils.save_image_to_path(
                source, str(root / ".." / "outside"), uploads_root=str(root)
            )
        )
    assert not list(tmp_path.iterdir())


def test_final_upload_path_cannot_follow_symlink_outside_root(tmp_path, image_source):
    source, suffix = image_source
    root = tmp_path / "uploads"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.write_bytes(b"unchanged")
    (root / f"input{suffix}").symlink_to(outside)
    with pytest.raises(Exception, match="escapes"):
        asyncio.run(
            utils.save_image_to_path(
                source, str(root / "input"), uploads_root=str(root)
            )
        )
    assert outside.read_bytes() == b"unchanged"


@pytest.mark.parametrize(
    "source", ["https://example.invalid/image.png", "data:image/png;base64,aW1hZ2U="]
)
def test_prefer_remote_source_still_avoids_disk_io(tmp_path, source):
    root = tmp_path / "uploads"
    saved = asyncio.run(
        utils.save_image_to_path(
            source,
            str(root / "input"),
            uploads_root=str(root),
            prefer_remote_source=True,
        )
    )
    assert saved == source
    assert not root.exists()


def test_video_upload_sanitizes_filename_and_preserves_metadata(tmp_path):
    filename = "../../../evil.png"
    upload = UploadFile(io.BytesIO(b"image"), filename=filename)
    saved = asyncio.run(
        video_api._save_first_input_image([upload], "request", str(tmp_path))
    )
    assert Path(saved) == tmp_path / "request_evil.png"
    assert Path(saved).read_bytes() == b"image"
    assert upload.filename == filename


@pytest.mark.parametrize(
    "api,endpoint,stop_at,relative_path",
    [
        (
            image_api,
            "/v1/images/edits",
            "build_sampling_params",
            "uploads/request_0_evil.png",
        ),
        (
            mesh_api,
            "/v1/meshes",
            "_build_sampling_params_from_request",
            "outputs/uploads/request_evil.png",
        ),
    ],
)
def test_multipart_upload_is_sanitized_before_generation(
    monkeypatch, tmp_path, api, endpoint, stop_at, relative_path
):
    monkeypatch.chdir(tmp_path)
    args = SimpleNamespace(
        input_save_path=str(tmp_path / "uploads"),
        output_path=str(tmp_path / "outputs"),
    )
    monkeypatch.setattr(api, "get_global_server_args", lambda: args)
    monkeypatch.setattr(api, "generate_request_id", lambda: "request")
    monkeypatch.setattr(
        image_api, "resolve_sampling_params_cls", lambda args: SamplingParams
    )

    def stop_before_generation(*args, **kwargs):
        assert (tmp_path / relative_path).read_bytes() == b"image"
        raise HTTPException(status_code=418, detail="upload verified")

    monkeypatch.setattr(api, stop_at, stop_before_generation)
    app = FastAPI()
    app.include_router(api.router)
    with TestClient(app) as client:
        response = client.post(
            endpoint,
            data={"prompt": "test"},
            files={"image": ("../../../evil.png", b"image", "image/png")},
        )
    assert response.status_code == 418, response.text
    assert not (tmp_path / "evil.png").exists()
