# SPDX-License-Identifier: Apache-2.0

import asyncio
import io
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path
from types import SimpleNamespace

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
    monkeypatch.setattr(mesh_api, "get_global_server_args", lambda: SimpleNamespace())
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
    part_dir = tmp_path / "parts"
    part_dir.mkdir()
    monkeypatch.setattr(utils.tempfile, "tempdir", str(part_dir))
    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        utils.httpx,
        "AsyncClient",
        lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw),
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
    monkeypatch.setattr(utils, "_MAX_IMAGE_DOWNLOAD_BYTES", 1024)
    part_dir = _patch_download(monkeypatch, tmp_path, handler)
    target = tmp_path / "in" / "img.png"
    with pytest.raises(Exception, match="exceeds"):
        asyncio.run(utils._save_url_image_to_path("https://x.test/a.png", str(target)))
    assert not target.exists()
    assert not any(part_dir.iterdir())


def test_image_download_within_limit_is_saved(monkeypatch, tmp_path):
    monkeypatch.setattr(utils, "_MAX_IMAGE_DOWNLOAD_BYTES", 1024)
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
