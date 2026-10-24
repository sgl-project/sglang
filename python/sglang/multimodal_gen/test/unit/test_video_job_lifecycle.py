import asyncio
import os
import shutil
import tempfile
import threading
from contextlib import AsyncExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import FastAPI

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.runtime.entrypoints.openai import utils as openai_utils
from sglang.multimodal_gen.runtime.entrypoints.openai import video_api
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch, Req
from sglang.multimodal_gen.runtime.scheduler_client import async_scheduler_client


def test_worker_error_with_output_cannot_complete_video_job(monkeypatch, tmp_path):
    output = tmp_path / "partial.mp4"
    output.write_bytes(b"partial")
    monkeypatch.setattr(
        async_scheduler_client,
        "forward",
        AsyncMock(
            return_value=OutputBatch(
                output_file_paths=[str(output)], error="save failed"
            )
        ),
    )
    monkeypatch.setattr(
        openai_utils,
        "get_global_server_args",
        lambda: SimpleNamespace(batching_max_size=1),
    )

    async def run():
        await video_api.VIDEO_STORE.upsert("failed-output", {"id": "failed-output"})
        await video_api._dispatch_job_async(
            "failed-output",
            Req(sampling_params=SamplingParams(prompt="test")),
            resources=AsyncExitStack(),
        )
        return await video_api.VIDEO_STORE.pop("failed-output")

    job = asyncio.run(run())
    assert job["status"] == "failed"
    assert "save failed" in job["error"]["message"]
    assert job["file_paths"] is None and job["url"] is None


def _request_app(monkeypatch, tmp_path):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.setattr(
        video_api,
        "get_global_server_args",
        lambda: SimpleNamespace(
            input_save_path=None, output_path=None, served_model_name="test"
        ),
    )
    monkeypatch.setattr(
        video_api, "resolve_sampling_params_cls", lambda _: SamplingParams
    )
    app = FastAPI()
    app.include_router(video_api.router)
    return app


@pytest.mark.parametrize("multipart", [False, True])
def test_invalid_video_request_removes_temporary_uploads(
    monkeypatch, tmp_path, multipart
):
    """Parsing can fail after creating a directory or persisting an upload."""
    app = _request_app(monkeypatch, tmp_path)

    async def run_test():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
            base_url="http://test",
        ) as client:
            if multipart:
                response = await client.post(
                    "/v1/videos",
                    data={"prompt": "test", "extra_body": '{"seed": "invalid"}'},
                    files={"input_reference": ("input.png", b"upload", "image/png")},
                )
            else:
                response = await client.post("/v1/videos", json={"seed": "invalid"})
            assert response.status_code == 400
            assert not list(tmp_path.glob("sglang*"))

    asyncio.run(run_test())


def test_cancelled_preparation_finishes_before_request_cleanup(monkeypatch, tmp_path):
    """Cancelling to_thread does not stop its writes; cleanup must run afterwards."""
    app = _request_app(monkeypatch, tmp_path)
    started, finish = threading.Event(), threading.Event()
    cleaned = []

    class Sampling:
        def prepare_video_request_for_queue(self, batch):
            started.set()
            assert finish.wait(timeout=5)
            (Path(batch.output_path) / "prepared.bin").write_bytes(b"prepared")

        def cleanup_video_request(self, batch):
            cleaned.append((Path(batch.output_path) / "prepared.bin").exists())

    sampling = Sampling()

    def build_sampling(_request_id, req):
        sampling.output_path = req.output_path
        return sampling

    monkeypatch.setattr(video_api, "_build_video_sampling_params", build_sampling)
    monkeypatch.setattr(
        video_api,
        "prepare_request",
        lambda **_: SimpleNamespace(
            sampling_params=sampling,
            output_path=sampling.output_path,
        ),
    )

    async def run_test():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            task = asyncio.create_task(
                client.post("/v1/videos", json={"prompt": "test"})
            )
            try:
                assert await asyncio.to_thread(started.wait, 5)
                for _ in range(2):
                    task.cancel()
                    await asyncio.sleep(0)
                    assert not task.done()
                    assert cleaned == []
            finally:
                finish.set()
                with pytest.raises(asyncio.CancelledError):
                    await task
            assert cleaned == [True]
            assert not list(tmp_path.glob("sglang*"))
            assert not video_api._VIDEO_JOB_TASKS

    asyncio.run(run_test())


@pytest.mark.parametrize("persistent", [False, True])
@pytest.mark.parametrize("failure", ["sampling", "prepare", "cleanup"])
def test_rejected_video_request_cleans_only_owned_resources(
    monkeypatch, tmp_path, persistent, failure
):
    app = _request_app(monkeypatch, tmp_path)
    inputs, outputs = tmp_path / "inputs", tmp_path / "outputs"
    if persistent:
        inputs.mkdir()
        outputs.mkdir()
        (inputs / "keep").write_bytes(b"input")
        (outputs / "keep").write_bytes(b"output")
        monkeypatch.setattr(
            video_api,
            "get_global_server_args",
            lambda: SimpleNamespace(
                input_save_path=str(inputs),
                output_path=str(outputs),
                served_model_name="test",
            ),
        )
    cleaned = []

    class Sampling:
        def prepare_video_request_for_queue(self, batch):
            (Path(batch.output_path) / "prepared.bin").write_bytes(b"prepared")
            raise ValueError("invalid preparation")

        def cleanup_video_request(self, batch):
            prepared = Path(batch.output_path) / "prepared.bin"
            cleaned.append(prepared.read_bytes())
            prepared.unlink()
            if failure == "cleanup":
                raise RuntimeError("cleanup failure")

    sampling = Sampling()

    def build_sampling(_request_id, req):
        if failure == "sampling":
            raise RuntimeError("sampling failure")
        sampling.output_path = req.output_path
        return sampling

    monkeypatch.setattr(video_api, "_build_video_sampling_params", build_sampling)
    monkeypatch.setattr(
        video_api,
        "prepare_request",
        lambda **_: SimpleNamespace(
            sampling_params=sampling, output_path=sampling.output_path
        ),
    )

    async def run_test():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
            base_url="http://test",
        ) as client:
            response = await client.post("/v1/videos", json={"prompt": "test"})
        assert response.status_code == (500 if failure == "sampling" else 400)
        assert cleaned == ([] if failure == "sampling" else [b"prepared"])
        assert not list(tmp_path.glob("sglang*"))
        assert not video_api._VIDEO_JOB_TASKS
        if persistent:
            assert (inputs / "keep").read_bytes() == b"input"
            assert (outputs / "keep").read_bytes() == b"output"

    asyncio.run(run_test())


def test_video_job_registry_holds_task_until_completion():
    async def run_test():
        started = asyncio.Event()
        finish = asyncio.Event()

        async def job():
            started.set()
            await finish.wait()

        task = video_api._start_video_job("job-id", job())
        await started.wait()
        assert video_api._VIDEO_JOB_TASKS["job-id"] is task

        finish.set()
        await task
        await asyncio.sleep(0)
        assert "job-id" not in video_api._VIDEO_JOB_TASKS

    asyncio.run(run_test())


@pytest.mark.parametrize("already_started", [False, True])
def test_shutdown_video_jobs_cancels_and_awaits_cleanup(already_started):
    async def run_test():
        started = asyncio.Event()
        cleaned_up = asyncio.Event()

        async def job():
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cleaned_up.set()

        task = video_api._start_video_job("job-id", job())
        if already_started:
            await started.wait()
        await video_api.shutdown_video_jobs()

        assert task.cancelled()
        assert cleaned_up.is_set()
        assert "job-id" not in video_api._VIDEO_JOB_TASKS

    asyncio.run(run_test())


class _FakeSamplingParams:
    def validate_video_final_outputs(self, paths, batch):
        return {}

    def cleanup_video_request(self, batch):
        pass


class _FakeBatch:
    sampling_params = _FakeSamplingParams()


def _run_dispatch(monkeypatch, tmp_path, *, output_persistent, upload):
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    paths = []
    for i in range(2):
        path = output_dir / f"video_{i}.mp4"
        path.write_bytes(b"video")
        paths.append(str(path))

    async def process_generation_batch(client, batch, scheduler_batches=None):
        return paths, object()

    async def upload_and_cleanup(path):
        url = upload(path)
        if url:
            os.remove(path)
        return url

    monkeypatch.setattr(video_api, "process_generation_batch", process_generation_batch)
    monkeypatch.setattr(
        video_api.cloud_storage, "upload_and_cleanup", upload_and_cleanup
    )
    monkeypatch.setattr(
        video_api,
        "add_common_data_to_response",
        lambda response, request_id, result: response,
    )

    async def run_test():
        await video_api.VIDEO_STORE.upsert("job-id", {"id": "job-id"})
        resources = AsyncExitStack()
        if not output_persistent:
            resources.callback(shutil.rmtree, output_dir, ignore_errors=True)
        batch = _FakeBatch()
        resources.push_async_callback(video_api._cleanup_video_request, batch)
        await video_api._dispatch_job_async(
            "job-id",
            batch,
            resources=resources,
            output_persistent=output_persistent,
        )
        return await video_api.VIDEO_STORE.pop("job-id")

    return asyncio.run(run_test()), paths


def test_temp_output_without_cloud_upload_fails_job(monkeypatch, tmp_path):
    """A job whose temp outputs are deleted unpublished must not report completed."""
    job, _ = _run_dispatch(
        monkeypatch, tmp_path, output_persistent=False, upload=lambda path: None
    )
    assert job["status"] == "failed"
    assert job["url"] is None and job["file_paths"] is None


def test_temp_output_with_partial_cloud_upload_fails_job(monkeypatch, tmp_path):
    job, _ = _run_dispatch(
        monkeypatch,
        tmp_path,
        output_persistent=False,
        upload=lambda path: "s3://a" if path.endswith("_0.mp4") else None,
    )
    assert job["status"] == "failed"


def test_temp_output_uploads_every_variant(monkeypatch, tmp_path):
    job, paths = _run_dispatch(
        monkeypatch,
        tmp_path,
        output_persistent=False,
        upload=lambda path: f"s3://{os.path.basename(path)}",
    )
    assert job["status"] == "completed"
    assert job["url"] == "s3://video_0.mp4"
    assert job["urls"] == ["s3://video_0.mp4", "s3://video_1.mp4"]
    assert job["file_path"] is None and job["file_paths"] is None
    assert video_api._select_video_variant_url(job, "1") == "s3://video_1.mp4"


def test_persistent_output_reports_only_local_files(monkeypatch, tmp_path):
    job, paths = _run_dispatch(
        monkeypatch,
        tmp_path,
        output_persistent=True,
        upload=lambda path: "s3://a" if path.endswith("_0.mp4") else None,
    )
    assert job["status"] == "completed"
    assert job["urls"] == ["s3://a", None]
    assert job["file_path"] is None
    assert job["file_paths"] == [None, paths[1]]
    assert os.path.exists(paths[1])
    assert video_api._select_video_variant_url(job, "1") is None
    assert video_api._select_video_variant_path(job, "1") == paths[1]


def test_content_variant_serves_lidar_artifacts_by_key(tmp_path):
    """A joint camera/LiDAR job exposes its range-map files under the keys of the
    status JSON's ``lidar.files``; numeric variants must keep selecting camera videos
    and unknown keys must not fall through to a video."""
    video = tmp_path / "job.mp4"
    rangemap = tmp_path / "job_lidar.safetensors"
    job = {
        "file_paths": [str(video)],
        "lidar": {
            "files": {"rangemap": str(rangemap), "range_video": str(tmp_path / "r.mp4")}
        },
    }

    assert video_api._select_lidar_artifact_path(job, "rangemap") == str(rangemap)
    assert video_api._select_lidar_artifact_path(job, "0") is None
    assert video_api._select_lidar_artifact_path(job, "bev_video") is None
    assert (
        video_api._select_lidar_artifact_path({"file_paths": [str(video)]}, "rangemap")
        is None
    )
    assert video_api._select_video_variant_path(job, None) == str(video)
    assert video_api._select_video_variant_path(job, "rangemap") is None
    assert video_api._ARTIFACT_MEDIA_TYPES[".safetensors"] == "application/octet-stream"
