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


def test_invalid_output_quality_is_a_400_not_a_500(monkeypatch):
    """A bad output_quality must surface as HTTP 400, not an unhandled ValueError."""
    monkeypatch.setattr(
        utils, "get_global_server_args", lambda: SimpleNamespace(model_path="m")
    )
    monkeypatch.setattr(
        SamplingParams,
        "from_user_sampling_params_args",
        classmethod(
            lambda cls, **kw: SimpleNamespace(
                data_type=utils.DataType.IMAGE, output_compression=None
            )
        ),
    )
    with pytest.raises(HTTPException) as exc_info:
        utils.build_sampling_params("request", output_quality="bogus")
    assert exc_info.value.status_code == 400
    assert "output_quality" in exc_info.value.detail


@pytest.mark.parametrize(
    "endpoint,kwargs",
    [
        ("/v1/images/generations", {"json": {"prompt": "p", "response_format": "url"}}),
        (
            "/v1/images/edits",
            {
                "data": {"prompt": "p", "response_format": "url"},
                "files": {"image": ("a.png", b"image", "image/png")},
            },
        ),
    ],
)
def test_url_response_without_destination_is_rejected_before_generation(
    monkeypatch, tmp_path, endpoint, kwargs
):
    """response_format='url' with no cloud storage and no output_path must fail
    fast instead of running the whole generation first."""
    args = SimpleNamespace(
        input_save_path=None, output_path=None, pipeline_class_name=None
    )
    monkeypatch.setattr(image_api, "get_global_server_args", lambda: args)
    monkeypatch.setattr(
        image_api, "resolve_sampling_params_cls", lambda args: SamplingParams
    )
    monkeypatch.setattr(image_api.cloud_storage, "enabled", False)

    def generation_reached(*args, **kwargs):
        raise HTTPException(status_code=599, detail="generation reached")

    monkeypatch.setattr(image_api, "build_sampling_params", generation_reached)
    app = FastAPI()
    app.include_router(image_api.router)
    with TestClient(app) as client:
        response = client.post(endpoint, **kwargs)
    assert response.status_code == 400, response.text
    assert "cloud storage" in response.json()["detail"]


def _patch_download(monkeypatch, handler):
    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        utils.httpx,
        "AsyncClient",
        lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw),
    )


def _chunked_png(request):
    async def body():
        for _ in range(8):
            yield b"x" * 512

    return httpx.Response(200, content=body(), headers={"content-type": "image/png"})


@pytest.mark.parametrize(
    "handler",
    [
        lambda request: httpx.Response(
            200, content=b"x" * 4096, headers={"content-type": "image/png"}
        ),
        _chunked_png,
    ],
    ids=["content-length", "chunked-no-length"],
)
def test_image_download_over_size_limit_is_rejected(monkeypatch, tmp_path, handler):
    """An oversized remote image must be aborted, not buffered and written."""
    monkeypatch.setattr(utils, "_MAX_IMAGE_DOWNLOAD_BYTES", 1024)
    _patch_download(monkeypatch, handler)
    target = tmp_path / "in" / "img.png"
    with pytest.raises(Exception, match="exceeds"):
        asyncio.run(utils._save_url_image_to_path("https://x.test/a.png", str(target)))
    assert not target.exists()


def test_image_download_within_limit_is_saved(monkeypatch, tmp_path):
    monkeypatch.setattr(utils, "_MAX_IMAGE_DOWNLOAD_BYTES", 1024)
    _patch_download(
        monkeypatch,
        lambda request: httpx.Response(
            200, content=b"x" * 512, headers={"content-type": "image/png"}
        ),
    )
    target = tmp_path / "in" / "img.png"
    saved = asyncio.run(
        utils._save_url_image_to_path("https://x.test/a.png", str(target))
    )
    assert Path(saved).read_bytes() == b"x" * 512


@pytest.mark.parametrize(
    "field", ["upscaling_model_path", "frame_interpolation_model_path"]
)
@pytest.mark.parametrize(
    "value", ["http://169.254.169.254/latest/x.pth", "https://internal.svc/w.pth"]
)
def test_remote_url_model_paths_are_rejected_over_http(monkeypatch, field, value):
    """A client-supplied URL must not make the server fetch it (SSRF)."""
    monkeypatch.setattr(
        utils, "get_global_server_args", lambda: SimpleNamespace(model_path="m")
    )
    with pytest.raises(HTTPException) as exc_info:
        utils.build_sampling_params("request", **{field: value})
    assert exc_info.value.status_code == 400
    assert field in exc_info.value.detail


def test_unsupported_response_format_is_rejected_before_generation(monkeypatch):
    """An unsupported response_format must fail fast, not after generation."""
    args = SimpleNamespace(
        input_save_path=None, output_path="out", pipeline_class_name=None
    )
    monkeypatch.setattr(image_api, "get_global_server_args", lambda: args)
    monkeypatch.setattr(
        image_api, "resolve_sampling_params_cls", lambda args: SamplingParams
    )
    monkeypatch.setattr(
        image_api,
        "build_sampling_params",
        lambda *a, **k: (_ for _ in ()).throw(HTTPException(599, "generation reached")),
    )
    app = FastAPI()
    app.include_router(image_api.router)
    with TestClient(app) as client:
        response = client.post(
            "/v1/images/generations", json={"prompt": "p", "response_format": "bmp"}
        )
    assert response.status_code == 400, response.text
    assert "not supported" in response.json()["detail"]


def test_cloud_upload_failure_is_not_reported_as_a_client_error(monkeypatch):
    """With cloud storage enabled but the upload failed, the server is at fault."""
    monkeypatch.setattr(image_api.cloud_storage, "enabled", True)
    with pytest.raises(HTTPException) as exc_info:
        image_api._build_image_response_kwargs(
            ["a.png"],
            "url",
            "p",
            "req",
            SimpleNamespace(),
            cloud_urls=[None],
            is_persistent=False,
        )
    assert exc_info.value.status_code == 502
