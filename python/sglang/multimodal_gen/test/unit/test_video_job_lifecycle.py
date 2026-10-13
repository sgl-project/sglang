import asyncio
import os

from sglang.multimodal_gen.runtime.entrypoints.openai import video_api


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


def test_shutdown_video_jobs_cancels_and_awaits_cleanup():
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
        await video_api._dispatch_job_async(
            "job-id",
            _FakeBatch(),
            temp_dirs=[] if output_persistent else [str(output_dir)],
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
