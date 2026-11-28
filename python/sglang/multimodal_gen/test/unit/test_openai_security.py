# SPDX-License-Identifier: Apache-2.0

import asyncio
import base64
import errno
import gzip
import http.server
import io
import ipaddress
import os
import ssl
import subprocess
import tempfile
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, wait
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import pytest
import torch
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from openai import AsyncOpenAI
from starlette.datastructures import UploadFile

from sglang.multimodal_gen.configs.pipeline_configs.hunyuan3d import (
    Hunyuan3D2PipelineConfig,
)
from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.runtime.entrypoints.openai import (
    image_api,
    mesh_api,
    utils,
    video_api,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.stores import AsyncDictStore
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch
from sglang.multimodal_gen.runtime.postprocess import (
    bounded_cache,
    realesrgan_upscaler,
    rife_interpolator,
)
from sglang.multimodal_gen.runtime.postprocess.bounded_cache import (
    cache_get,
    cache_put,
)
from sglang.multimodal_gen.runtime.utils import image_io
from sglang.multimodal_gen.runtime.utils.image_io import ensure_path_within_root


@pytest.mark.parametrize("api,kind", [(video_api, "video"), (mesh_api, "mesh")])
def test_job_list_is_bounded_and_has_complete_cursor_metadata(monkeypatch, api, kind):
    store = AsyncDictStore()
    monkeypatch.setattr(api, f"{kind.upper()}_STORE", store)
    jobs = [{"id": str(i), "created_at": i % 4} for i in range(45)]

    async def populate():
        for job in jobs:
            await store.upsert(job["id"], job)

    asyncio.run(populate())
    app = FastAPI()
    app.include_router(api.router)
    endpoint = api.router.prefix
    with TestClient(app) as client:
        assert len(client.get(endpoint).json()["data"]) == 20
        for order in ("asc", "desc"):
            expected = sorted(
                jobs, key=lambda item: item["created_at"], reverse=order == "desc"
            )
            ids = []
            params = {"order": order, "limit": 7}
            for start in range(0, len(jobs), 7):
                page = client.get(endpoint, params=params).json()
                page_ids = [job["id"] for job in page["data"]]
                assert page_ids == [job["id"] for job in expected[start : start + 7]]
                assert page["first_id"] == page_ids[0]
                assert page["last_id"] == page_ids[-1]
                assert page["has_more"] is (start + 7 < len(jobs))
                ids.extend(page_ids)
                params["after"] = page["last_id"]
            assert len(ids) == len(set(ids)) == len(jobs)
        empty = client.get(endpoint, params={"after": "missing"}).json()
        assert empty == {
            "object": "list",
            "data": [],
            "first_id": None,
            "last_id": None,
            "has_more": False,
        }
        for params in ({"limit": 0}, {"limit": 101}, {"order": "invalid"}):
            assert client.get(endpoint, params=params).status_code == 422


def test_openai_video_iterator_stops_at_the_last_nonempty_page(monkeypatch):
    store = AsyncDictStore()
    monkeypatch.setattr(video_api, "VIDEO_STORE", store)
    app = FastAPI()
    app.include_router(video_api.router)
    requests = []

    async def record_request(request):
        requests.append(request)

    async def run():
        for i in range(21):
            await store.upsert(str(i), {"id": str(i), "created_at": i})
        async with AsyncOpenAI(
            api_key="unused",
            base_url="http://test/v1",
            http_client=httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                event_hooks={"request": [record_request]},
            ),
        ) as client:
            ids = [job.id async for job in client.videos.list(limit=7)]
        assert ids == [str(i) for i in reversed(range(21))]
        assert len(requests) == 3

    asyncio.run(run())


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
        _streamed(
            lambda req: httpx.Response(
                200, headers={"content-type": "image/png"}, content=b"image"
            )
        )
    )
    monkeypatch.setattr(
        utils.httpx,
        "AsyncClient",
        lambda **kwargs: client_cls(transport=transport, **kwargs),
    )
    return "https://93.184.216.34/image.png", ".png"


def test_save_image_sources_within_uploads_root(tmp_path, image_source):
    source, suffix = image_source
    root = tmp_path / "uploads"
    saved = asyncio.run(
        utils.save_image_to_path(source, str(root / "input"), uploads_root=str(root))
    )
    assert Path(saved) == root / f"input{suffix}"
    assert Path(saved).read_bytes() == b"image"


def _streamed(handler):
    """Serve a handler's body as an unread stream, as a real socket would."""

    def respond(request):
        response = handler(request)
        if response.is_stream_consumed:
            return httpx.Response(
                response.status_code,
                headers=response.headers,
                stream=httpx.ByteStream(response.content),
            )
        return response

    return respond


def _reason(error: BaseException) -> str:
    """The cause of a failed image save; a URL download raises a generic error from it."""
    return str(error.__cause__ if error.__cause__ is not None else error)


@pytest.mark.parametrize("existing", [False, True])
def test_failed_image_write_preserves_destination(
    monkeypatch, tmp_path, image_source, existing
):
    """A failed input write must not publish partial bytes or truncate an old file."""
    source, suffix = image_source
    target = tmp_path / f"input{suffix}"
    if existing:
        target.write_bytes(b"previous image")

    @contextmanager
    def failing_write(opener, *args, **kwargs):
        with opener(*args, **kwargs) as stream:
            write = stream.write

            def write_partial(data):
                write(data[:1])
                raise OSError(errno.ENOSPC, "No space left on device")

            with patch.object(stream, "write", side_effect=write_partial):
                yield stream

    with monkeypatch.context() as fault:
        fault.setattr(utils, "open", partial(failing_write, open), raising=False)
        fault.setattr(image_io, "open", partial(failing_write, open), raising=False)
        fault.setattr(os, "fdopen", partial(failing_write, os.fdopen))
        with pytest.raises(Exception) as caught:
            asyncio.run(
                utils.save_image_to_path(
                    source, str(tmp_path / "input"), uploads_root=str(tmp_path)
                )
            )
        assert "No space left on device" in _reason(caught.value)
    assert set(tmp_path.iterdir()) == ({target} if existing else set())
    if existing:
        assert target.read_bytes() == b"previous image"
    # a later request must replace the file with a complete input
    saved = asyncio.run(
        utils.save_image_to_path(
            b"complete image", str(target), uploads_root=str(tmp_path)
        )
    )
    assert Path(saved).read_bytes() == b"complete image"


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
    with pytest.raises(Exception) as caught:
        asyncio.run(
            utils.save_image_to_path(
                source, str(root / "input"), uploads_root=str(root)
            )
        )
    assert "escapes" in _reason(caught.value)
    assert outside.read_bytes() == b"unchanged"


@pytest.mark.parametrize("persistent", [True, False])
@pytest.mark.parametrize("as_url", [True, False])
def test_input_storage_modes_admit_identical_bytes(
    tmp_path, monkeypatch, persistent, as_url
):
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), "navy").save(buffer, format="PNG")
    data = buffer.getvalue()
    _patch_download(
        monkeypatch,
        tmp_path,
        lambda _: httpx.Response(
            200, content=data, headers={"content-type": "image/png"}
        ),
    )
    source = (
        "https://x.test/image.png"
        if as_url
        else "data:image/png;base64," + base64.b64encode(data).decode()
    )
    with utils.temp_dir_if_disabled(
        str(tmp_path / "uploads") if persistent else None
    ) as root:
        path = Path(
            asyncio.run(
                utils.save_image_to_path(
                    source, str(Path(root) / "input"), uploads_root=root
                )
            )
        )
        assert path.read_bytes() == data
        with Image.open(path) as decoded:
            assert decoded.getpixel((0, 0)) == (0, 0, 128)
    assert path.exists() is persistent


def test_video_upload_sanitizes_filename_and_preserves_metadata(tmp_path):
    filename = "../../../evil.png"
    upload = UploadFile(io.BytesIO(b"image"), filename=filename)
    saved = asyncio.run(
        utils.save_first_input_image([upload], "request", str(tmp_path))
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
            "uploads/request_evil.png",
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


def test_mesh_upload_is_temporary_when_input_saving_is_disabled(monkeypatch, tmp_path):
    """With `--input-save-path ""` the mesh route keeps the uploaded image only
    for the lifetime of its background job, and never under the working directory."""
    monkeypatch.chdir(tmp_path)
    store = AsyncDictStore()
    mesh_path = tmp_path / "mesh.glb"
    seen = {}

    def build_sampling_params(request_id, req, image_path):
        seen["upload"] = Path(image_path)
        return SimpleNamespace(output_file_path=lambda: str(mesh_path))

    async def run_job(scheduler_client, batch):
        seen["alive_during_job"] = seen["upload"].read_bytes() == b"image"
        mesh_path.write_bytes(b"mesh")
        return [str(mesh_path)], OutputBatch()

    monkeypatch.setattr(mesh_api, "MESH_STORE", store)
    monkeypatch.setattr(
        mesh_api,
        "get_global_server_args",
        lambda: SimpleNamespace(input_save_path=None, output_path=str(tmp_path)),
    )
    monkeypatch.setattr(mesh_api, "generate_request_id", lambda: "request")
    monkeypatch.setattr(
        mesh_api, "_build_sampling_params_from_request", build_sampling_params
    )
    monkeypatch.setattr(mesh_api, "prepare_request", lambda **kwargs: SimpleNamespace())
    monkeypatch.setattr(mesh_api, "process_generation_batch", run_job)
    app = FastAPI()
    app.include_router(mesh_api.router)

    async def run_test():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/meshes",
                data={"prompt": "test"},
                files={"image": ("input.png", b"image", "image/png")},
            )
            assert response.status_code == 200, response.text

            async def job_left_the_queue():
                while (await store.get("request"))["status"] == "queued":
                    await asyncio.sleep(0)

            await asyncio.wait_for(job_left_the_queue(), 5)
        assert (await store.get("request"))["status"] == "completed"

    asyncio.run(run_test())
    assert seen["alive_during_job"]
    assert not seen["upload"].exists()
    assert sorted(path.name for path in tmp_path.iterdir()) == ["mesh.glb"]


@pytest.mark.parametrize("status", ["queued", "in_progress", "failed", "completed"])
def test_mesh_content_is_not_published_before_success(monkeypatch, tmp_path, status):
    """A partial mesh may exist while generation is running or after it fails."""
    path = tmp_path / "mesh.glb"
    path.write_bytes(b"mesh bytes")
    store = AsyncDictStore()
    monkeypatch.setattr(mesh_api, "MESH_STORE", store)
    app = FastAPI()
    app.include_router(mesh_api.router)

    async def run_test():
        await store.upsert("mesh", {"status": status, "file_path": str(path)})
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/v1/meshes/mesh/content")
        assert response.status_code == (200 if status == "completed" else 404)
        if status == "completed":
            assert response.content == path.read_bytes()

    asyncio.run(run_test())


@pytest.mark.parametrize("output_exists", [False, True])
def test_mesh_completion_requires_a_downloadable_file(
    monkeypatch, tmp_path, output_exists
):
    """A worker-returned path is not proof that a local output can be published."""
    path = tmp_path / "mesh.glb"
    if output_exists:
        path.write_bytes(b"mesh bytes")
    store = AsyncDictStore()
    monkeypatch.setattr(mesh_api, "MESH_STORE", store)
    monkeypatch.setattr(mesh_api.cloud_storage, "enabled", False)

    async def generate(*args, **kwargs):
        return [str(path)], OutputBatch()

    monkeypatch.setattr(mesh_api, "process_generation_batch", generate)

    async def run_test():
        await store.upsert("mesh", {"status": "queued", "file_path": str(path)})
        await mesh_api._dispatch_job_async("mesh", None)
        job = await store.get("mesh")
        assert job["status"] == ("completed" if output_exists else "failed")
        if output_exists:
            assert job["file_size_bytes"] == path.stat().st_size
        else:
            assert job["error"] and job["file_path"] is None

    asyncio.run(run_test())


def test_rejected_mesh_preparation_does_not_create_a_queued_job(monkeypatch, tmp_path):
    """Failure before scheduler submission must not leave a permanently queued job."""
    store = AsyncDictStore()
    monkeypatch.setattr(mesh_api, "MESH_STORE", store)
    monkeypatch.setattr(
        mesh_api,
        "get_global_server_args",
        lambda: SimpleNamespace(input_save_path=None, output_path=str(tmp_path)),
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        mesh_api,
        "_build_sampling_params_from_request",
        lambda *args: SamplingParams(
            prompt="mesh", output_path=str(tmp_path), output_file_name="mesh.glb"
        ),
    )
    rejected = asyncio.Event()

    def reject(**kwargs):
        rejected.set()
        raise ValueError("request preparation failed")

    monkeypatch.setattr(mesh_api, "prepare_request", reject)
    app = FastAPI()
    app.include_router(mesh_api.router)

    async def run_test():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/v1/meshes", json={"input_image": "data:image/png;base64,aW1hZ2U="}
            )
        assert response.status_code == 500
        assert rejected.is_set()
        assert (await store.list_page())["data"] == []

    asyncio.run(run_test())


@pytest.mark.parametrize("seed,device", [(None, None), (0, "cpu"), ([0, 7], "cuda")])
def test_mesh_optional_sampling_fields_preserve_defaults(
    monkeypatch, tmp_path, seed, device
):
    args = SimpleNamespace(
        model_path="mesh",
        backend="native",
        model_id=None,
        pipeline_config=Hunyuan3D2PipelineConfig(),
        output_path=str(tmp_path),
        comfyui_mode=False,
    )
    monkeypatch.setattr(mesh_api, "get_global_server_args", lambda: args)
    monkeypatch.setattr(
        SamplingParams, "from_pretrained", lambda *a, **kw: SamplingParams()
    )
    request = mesh_api.MeshGenerationsRequest(
        seed=seed,
        generator_device=device,
        num_inference_steps=4,
        guidance_scale=5.0,
        negative_prompt="test",
    )
    sampling = mesh_api._build_sampling_params_from_request(
        "mesh", request, "input.png"
    )
    defaults = SamplingParams()
    assert sampling.seed == (defaults.seed if seed is None else seed)
    assert sampling.generator_device == (
        defaults.generator_device if device is None else device
    )
    assert sampling.num_inference_steps == 4
    assert sampling.guidance_scale == 5.0
    assert sampling.negative_prompt == "test"


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


def _build_sampling_params_with_captured_kwargs(monkeypatch, server_args):
    """Patch build_sampling_params to capture (not construct) its kwargs."""
    captured = {}
    monkeypatch.setattr(utils, "get_global_server_args", lambda: server_args)
    monkeypatch.setattr(
        SamplingParams,
        "from_user_sampling_params_args",
        lambda **kw: captured.update(kw) or SimpleNamespace(),
    )
    return captured


def test_build_sampling_params_rejects_perf_dump_path_outside_output_root(
    monkeypatch, tmp_path
):
    """Regression test for sglang#51555: perf_dump_path reaches code that
    creates directories and atomically overwrites files, so a client must
    not point it outside the operator-configured output root."""
    captured = _build_sampling_params_with_captured_kwargs(
        monkeypatch,
        SimpleNamespace(model_path="m", output_path=str(tmp_path)),
    )
    with pytest.raises(HTTPException) as exc_info:
        utils.build_sampling_params(
            "request", prompt="p", perf_dump_path=str(tmp_path.parent / "evil.json")
        )
    assert exc_info.value.status_code == 400
    assert "perf_dump_path" in exc_info.value.detail

    utils.build_sampling_params(
        "request", prompt="p", perf_dump_path=str(tmp_path / "report.json")
    )
    assert captured["perf_dump_path"] == str(tmp_path / "report.json")


def test_build_sampling_params_rejects_output_path_traversal(monkeypatch, tmp_path):
    captured = _build_sampling_params_with_captured_kwargs(
        monkeypatch,
        SimpleNamespace(model_path="m", output_path=str(tmp_path)),
    )
    with pytest.raises(HTTPException) as exc_info:
        utils.build_sampling_params(
            "request", prompt="p", output_path=str(tmp_path / ".." / "evil")
        )
    assert exc_info.value.status_code == 400
    assert "output_path" in exc_info.value.detail

    utils.build_sampling_params(
        "request", prompt="p", output_path=str(tmp_path / "sub")
    )
    assert captured["output_path"] == str(tmp_path / "sub")


def test_build_sampling_params_rejects_traversal_output_file_name(
    monkeypatch, tmp_path
):
    captured = _build_sampling_params_with_captured_kwargs(
        monkeypatch,
        SimpleNamespace(model_path="m", output_path=str(tmp_path)),
    )
    # output_file_name is joined onto output_path, so a ".." in it would
    # escape even a contained output_path.
    for bad_name in ("../evil.mp4", "..\\evil.mp4", "..", "sub/evil.mp4"):
        with pytest.raises(HTTPException) as exc_info:
            utils.build_sampling_params(
                "request", prompt="p", output_file_name=bad_name
            )
        assert exc_info.value.status_code == 400
        assert "output_file_name" in exc_info.value.detail

    utils.build_sampling_params("request", prompt="p", output_file_name="clip.mp4")
    assert captured["output_file_name"] == "clip.mp4"


def test_build_sampling_params_falls_back_to_tempdir_when_output_disabled(
    monkeypatch,
):
    """With no operator-configured output root, only the system temp dir
    (where the server itself generates temporary output dirs) is allowed."""
    captured = _build_sampling_params_with_captured_kwargs(
        monkeypatch, SimpleNamespace(model_path="m")
    )
    with pytest.raises(HTTPException) as exc_info:
        utils.build_sampling_params(
            "request", prompt="p", perf_dump_path="/var/lib/sglang/evil.json"
        )
    assert exc_info.value.status_code == 400

    in_tmp = os.path.join(tempfile.gettempdir(), "report.json")
    utils.build_sampling_params("request", prompt="p", perf_dump_path=in_tmp)
    assert captured["perf_dump_path"] == in_tmp


@pytest.mark.parametrize(
    "endpoint,kwargs,detail",
    [
        (
            "/v1/images/generations",
            {"json": {"prompt": "p", "response_format": "url"}},
            "cloud storage",
        ),
        (
            "/v1/images/edits",
            {
                "data": {"prompt": "p", "response_format": "url"},
                "files": {"image": ("a.png", b"image", "image/png")},
            },
            "cloud storage",
        ),
        (
            "/v1/images/generations",
            {"json": {"prompt": "p", "response_format": "bmp"}},
            "not supported",
        ),
    ],
)
def test_undeliverable_response_format_is_rejected_before_generation(
    monkeypatch, endpoint, kwargs, detail
):
    """An unsupported response_format, or 'url' with no cloud storage and no
    output_path, must fail fast instead of running the whole generation first."""
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
    assert detail in response.json()["detail"]


def _patch_download(monkeypatch, tmp_path, handler):
    """Route the download through handler; return the dir holding partial files."""
    # Only the synthetic origin is allowed; neither DNS nor HTTP needs a network.
    monkeypatch.setattr(
        "sglang.srt.multimodal.media_io._allowed_media_domains", frozenset({"x.test"})
    )
    monkeypatch.setattr(
        "sglang.srt.multimodal.media_io._resolve_host_ips",
        lambda _: [ipaddress.ip_address("127.0.0.1")],
    )
    monkeypatch.setenv("no_proxy", "*")
    part_dir = tmp_path / "parts"
    part_dir.mkdir()
    monkeypatch.setattr(utils.tempfile, "tempdir", str(part_dir))
    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        utils.httpx,
        "AsyncClient",
        lambda **kw: real_client(
            transport=httpx.MockTransport(_streamed(handler)), **kw
        ),
    )
    return part_dir


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
    monkeypatch.setattr(
        "sglang.srt.multimodal.media_io._media_url_max_file_size_bytes", 1024
    )
    part_dir = _patch_download(monkeypatch, tmp_path, handler)
    target = tmp_path / "in" / "img.png"
    with pytest.raises(Exception) as caught:
        asyncio.run(utils._save_url_image_to_path("https://x.test/a.png", str(target)))
    assert "exceeds" in _reason(caught.value)
    # The message reaches the client as a 400; neither the URL nor the cause is in it.
    assert str(caught.value) == "Failed to download image from URL"
    assert not target.exists()
    assert not any(part_dir.iterdir())
    assert not list(target.parent.iterdir())


def test_image_download_requests_identity_and_rejects_compressed_body(
    monkeypatch, tmp_path
):
    """The size cap counts written bytes, so the body is never decompressed here."""
    seen = []

    def handler(request):
        seen.append(request.headers["accept-encoding"])
        return httpx.Response(
            200,
            content=gzip.compress(b"x" * 4096),
            headers={"content-type": "image/png", "content-encoding": "gzip"},
        )

    part_dir = _patch_download(monkeypatch, tmp_path, handler)
    target = tmp_path / "in" / "img.png"
    with pytest.raises(Exception) as caught:
        asyncio.run(utils._save_url_image_to_path("https://x.test/a.png", str(target)))
    assert "Content-Encoding" in _reason(caught.value)
    assert str(caught.value) == "Failed to download image from URL"
    assert seen and set(seen) == {"identity"}
    assert not target.exists()
    assert not any(part_dir.iterdir())


def test_image_download_has_one_deadline(monkeypatch, tmp_path):
    """A slow drip is cut at --media-url-timeout, not after every chunk's own timeout."""
    monkeypatch.setattr("sglang.srt.multimodal.media_io._media_url_timeout", 0.3)

    async def drip():
        for _ in range(20):
            await asyncio.sleep(0.1)
            yield b"x"

    part_dir = _patch_download(
        monkeypatch,
        tmp_path,
        lambda request: httpx.Response(
            200, content=drip(), headers={"content-type": "image/png"}
        ),
    )
    target = tmp_path / "in" / "img.png"
    with pytest.raises(Exception) as caught:
        asyncio.run(utils._save_url_image_to_path("https://x.test/a.png", str(target)))
    assert isinstance(caught.value.__cause__, TimeoutError)
    assert not target.exists()
    assert not any(part_dir.iterdir())


def test_image_download_within_limit_is_saved(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "sglang.srt.multimodal.media_io._media_url_max_file_size_bytes", 1024
    )
    part_dir = _patch_download(
        monkeypatch,
        tmp_path,
        lambda request: httpx.Response(
            200, content=b"x" * 512, headers={"content-type": "image/png"}
        ),
    )
    target = tmp_path / "in" / "img.png"
    saved = asyncio.run(
        utils._save_url_image_to_path("https://x.test/a.png", str(target))
    )
    assert Path(saved).read_bytes() == b"x" * 512
    assert not any(part_dir.iterdir())
    assert list(target.parent.iterdir()) == [target]


def test_cancelled_image_download_leaves_no_partial_file(monkeypatch, tmp_path):
    """Executor congestion must not orphan a fully downloaded temporary file."""
    target = tmp_path / "in" / "image.png"
    finish = threading.Event()
    payload = b"image bytes"

    async def run_test():
        loop = asyncio.get_running_loop()
        loop.set_default_executor(ThreadPoolExecutor(max_workers=1))
        occupied = asyncio.Event()
        worker = None

        def occupy_executor():
            loop.call_soon_threadsafe(occupied.set)
            assert finish.wait(5)

        class Stream(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield payload

            async def aclose(self):
                nonlocal worker
                worker = loop.run_in_executor(None, occupy_executor)
                await occupied.wait()

        part_dir = _patch_download(
            monkeypatch,
            tmp_path,
            lambda _: httpx.Response(200, stream=Stream()),
        )
        task = asyncio.create_task(
            utils._save_url_image_to_path("https://x.test/image.png", str(target))
        )
        try:
            await asyncio.wait_for(occupied.wait(), 5)
            await asyncio.sleep(0)
            if not task.done():
                task.cancel()
            try:
                saved = await task
            except asyncio.CancelledError:
                assert not target.exists()
            else:
                assert Path(saved).read_bytes() == payload
        finally:
            finish.set()
            if worker is not None:
                await worker
        assert not list(part_dir.iterdir())
        assert list(target.parent.iterdir()) == ([target] if target.exists() else [])

    asyncio.run(run_test())


def test_image_download_rejects_destination_directory(monkeypatch, tmp_path):
    """A path collision must not return a directory or hide an upload inside it."""
    part_dir = _patch_download(
        monkeypatch, tmp_path, lambda _: httpx.Response(200, content=b"image")
    )
    target = tmp_path / "image.png"
    target.mkdir()
    with pytest.raises(Exception) as caught:
        asyncio.run(utils._save_url_image_to_path("https://x.test/a.png", str(target)))
    assert "directory" in _reason(caught.value)
    assert not list(target.iterdir())
    assert not list(part_dir.iterdir())
    assert set(tmp_path.iterdir()) == {target, part_dir}


@pytest.mark.parametrize("recover", [False, True])
def test_image_download_retry_keeps_only_complete_output(
    monkeypatch, tmp_path, recover
):
    """Failed attempts must leave neither partial files nor a truncated destination."""
    attempts = 0

    async def broken_body():
        yield b"partial image"
        raise httpx.ReadError("incomplete response")

    def handler(request):
        nonlocal attempts
        attempts += 1
        return httpx.Response(
            200,
            content=b"complete image" if recover and attempts == 3 else broken_body(),
        )

    part_dir = _patch_download(monkeypatch, tmp_path, handler)
    target = tmp_path / "image.png"
    target.write_bytes(b"existing image")
    request = utils._save_url_image_to_path("https://x.test/a.png", str(target))
    if recover:
        assert asyncio.run(request) == str(target)
    else:
        with pytest.raises(Exception) as caught:
            asyncio.run(request)
        assert "incomplete response" in _reason(caught.value)
    assert attempts == 3
    assert target.read_bytes() == (b"complete image" if recover else b"existing image")
    assert not list(part_dir.iterdir())
    assert set(tmp_path.iterdir()) == {target, part_dir}


_CPU = SimpleNamespace(get_local_torch_device=lambda: torch.device("cpu"))


def test_cache_evicts_least_recently_used():
    """A get must refresh recency, so eviction drops the stale entry, not the oldest insert."""
    cache = OrderedDict()
    cache_put(cache, "a", 1, max_size=2)
    cache_put(cache, "b", 2, max_size=2)
    assert cache_get(cache, "a") == 1
    cache_put(cache, "c", 3, max_size=2)
    assert list(cache) == ["a", "c"]


def test_cache_put_waits_for_the_shared_lock():
    """Request threads share these caches; an unlocked move_to_end can race an
    eviction and raise KeyError."""
    cache = OrderedDict()
    with ThreadPoolExecutor(max_workers=1) as pool:
        with bounded_cache._CACHE_LOCK:
            future = pool.submit(cache_put, cache, "a", 1, max_size=1)
            done, _ = wait([future], timeout=0.1)
            assert not done
        future.result(timeout=5)
    assert cache == {"a": 1}


def test_upscaler_uses_bounded_cache(monkeypatch):
    monkeypatch.setattr(realesrgan_upscaler, "_MODEL_CACHE", OrderedDict())
    monkeypatch.setattr(realesrgan_upscaler, "_MAX_CACHED_MODELS", 1)
    monkeypatch.setattr(realesrgan_upscaler, "_resolve_model_path", lambda p: p)
    monkeypatch.setattr(realesrgan_upscaler.torch, "load", lambda *a, **k: {})
    monkeypatch.setattr(
        realesrgan_upscaler,
        "_build_net_from_state_dict",
        lambda sd: torch.nn.Identity(),
    )
    monkeypatch.setattr(realesrgan_upscaler, "current_platform", _CPU)
    for path in ["a.pth", "b.pth"]:
        realesrgan_upscaler.ImageUpscaler(model_path=path)._ensure_model_loaded()
    assert list(realesrgan_upscaler._MODEL_CACHE) == ["b.pth"]


def test_rife_uses_bounded_cache(monkeypatch):
    import sglang.multimodal_gen.runtime.utils.hf_diffusers_utils as hf

    class FakeModel:
        def __init__(self):
            self.flownet = torch.nn.Identity()

        def load_model(self, *a, **k):
            pass

        def eval(self):
            pass

    monkeypatch.setattr(rife_interpolator, "_MODEL_CACHE", OrderedDict())
    monkeypatch.setattr(rife_interpolator, "_MAX_CACHED_MODELS", 1)
    monkeypatch.setattr(hf, "maybe_download_model", lambda p: p)
    monkeypatch.setattr(rife_interpolator, "Model", FakeModel)
    monkeypatch.setattr(rife_interpolator, "current_platform", _CPU)
    for path in ["a", "b"]:
        rife_interpolator.FrameInterpolator(model_path=path)._ensure_model_loaded()
    assert list(rife_interpolator._MODEL_CACHE) == ["b"]


def test_pinned_https_preserves_original_sni_host_and_redirect_policy(
    tmp_path, monkeypatch
):
    from PIL import Image

    from sglang.srt.multimodal import media_io

    crt = tmp_path / "server.crt"
    key = tmp_path / "server.key"
    subprocess.run(
        [
            "openssl",
            "req",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-x509",
            "-days",
            "1",
            "-subj",
            "/CN=one.media.test",
            "-addext",
            "subjectAltName=DNS:one.media.test,DNS:two.media.test",
            "-keyout",
            str(key),
            "-out",
            str(crt),
        ],
        check=True,
        capture_output=True,
    )
    seen, names = [], []
    image = io.BytesIO()
    Image.new("RGB", (8, 8), "navy").save(image, format="PNG")
    body = image.getvalue()

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            seen.append(self.headers["Host"])
            if self.path == "/first":
                self.send_response(302)
                self.send_header(
                    "Location",
                    f"https://two.media.test:{self.server.server_port}/image.png",
                )
            else:
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
            self.end_headers()
            if self.path != "/first":
                self.wfile.write(body)

        def log_message(self, *_):
            pass

    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(crt, key)
    ctx.set_servername_callback(lambda sock, name, context: names.append(name))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.socket = ctx.wrap_socket(server.socket, server_side=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    previous = media_io.media_url_security_config()
    monkeypatch.setenv("no_proxy", "*")
    monkeypatch.setenv("REQUESTS_CA_BUNDLE", str(crt))
    monkeypatch.setenv("SSL_CERT_FILE", str(crt))
    # The operating system cannot resolve these fixture hostnames. Any unpinned dial fails.
    monkeypatch.setattr(
        media_io, "_resolve_host_ips", lambda _: [ipaddress.ip_address("127.0.0.1")]
    )
    media_io.configure_media_url_security(
        ["one.media.test", "two.media.test", "other.media.test"]
    )
    port = server.server_port
    url = f"https://one.media.test:{port}/first"
    try:
        assert media_io.get_image_bytes(url) == body
        saved = asyncio.run(utils._save_url_image_to_path(url, str(tmp_path / "saved")))
        assert Path(saved).read_bytes() == body
        assert seen == [f"one.media.test:{port}", f"two.media.test:{port}"] * 2
        assert names == ["one.media.test", "two.media.test"] * 2
        import requests

        with pytest.raises(requests.exceptions.SSLError):
            media_io.get_image_bytes(f"https://other.media.test:{port}/image.png")
        monkeypatch.setenv("no_proxy", "")
        monkeypatch.setenv("NO_PROXY", "")
        monkeypatch.setenv("https_proxy", "http://proxy.media.test:3000")
        with pytest.raises(ValueError, match="direct route"):
            media_io.get_image_bytes(url)
        assert len(seen) == 4
    finally:
        media_io.configure_media_url_security(*previous)
        server.shutdown()
        server.server_close()
        thread.join()
