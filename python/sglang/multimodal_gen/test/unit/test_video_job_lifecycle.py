import asyncio

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
