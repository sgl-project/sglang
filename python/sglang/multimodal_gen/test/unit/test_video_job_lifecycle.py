import asyncio
import os
import threading
import uuid
from contextlib import suppress

import pytest
import zmq
import zmq.asyncio
from tqdm.auto import tqdm

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.runtime.entrypoints.openai import video_api, video_progress
from sglang.multimodal_gen.runtime.entrypoints.openai.stores import AsyncDictStore
from sglang.multimodal_gen.runtime.managers.progress_reporter import (
    ProgressReporter,
    ReportingTqdm,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages import base as stage_base


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


@pytest.fixture
def progress_channel():
    context = zmq.Context()
    endpoint = f"inproc://progress-{uuid.uuid4().hex}"
    pull = context.socket(zmq.PULL)
    pull.bind(endpoint)
    yield ProgressReporter(context, endpoint), pull
    context.destroy(linger=0)


def _received(pull):
    messages = []
    while pull.poll(200):
        messages.append(pull.recv_json())
    return messages


def test_progress_bar_reports_once_per_percent(progress_channel):
    reporter, pull = progress_channel
    with ReportingTqdm(
        total=1000, disable=True, reporter=reporter, request_ids=["job"]
    ) as bar:
        for _ in range(1000):
            bar.update()

    assert [m[1] for m in _received(pull)] == list(range(10, 1001, 10))


def test_visible_progress_bar_counts_each_step_once(progress_channel):
    reporter, pull = progress_channel
    # mininterval=0 makes tqdm call update() from its own __iter__ every step
    bar = ReportingTqdm(
        enumerate("wxyz"), total=4, mininterval=0, reporter=reporter, request_ids=["a"]
    )

    sent = []
    for i, _ in bar:
        sent.append([m[1] for m in _received(pull)])
        if i == 1:
            break
    bar.close()

    assert sent == [[], [1]]


@pytest.mark.parametrize("peer", ["no-peer", "never-reads"])
def test_progress_reporter_never_blocks(peer):
    context = zmq.Context()
    endpoint = f"inproc://progress-{uuid.uuid4().hex}"
    pull = context.socket(zmq.PULL)
    pull.setsockopt(zmq.RCVHWM, 10)
    if peer == "never-reads":
        pull.bind(endpoint)
    reporter = ProgressReporter(context, endpoint)

    def send_all():
        for k in range(50_000):
            reporter.step(request_ids=["job"], completed=k, total=50_000)

    sender = threading.Thread(target=send_all, daemon=True)
    try:
        sender.start()
        sender.join(timeout=10)
        assert not sender.is_alive()
    finally:
        context.destroy(linger=0)


@pytest.mark.parametrize("is_warmup", [False, True], ids=["request", "warmup"])
def test_progress_bar_reports_only_real_requests(monkeypatch, is_warmup):
    monkeypatch.setattr(stage_base, "get_progress_reporter", lambda: object())
    batch = Req(
        sampling_params=SamplingParams(prompt="p", seed=1, request_id="job"),
        is_warmup=is_warmup,
    )

    bar = stage_base.PipelineStage.progress_bar(None, total=3, batch=batch)

    assert type(bar) is (tqdm if is_warmup else ReportingTqdm)


def test_progress_tracker_averages_outputs():
    tracker = video_progress.VideoProgressTracker()
    tracker.register("job", num_outputs=2)

    assert tracker.update("job:0", completed=10, total=10) == ("job", 50)
    assert tracker.update("job:1", completed=5, total=10) == ("job", 75)
    assert tracker.update("job:1", completed=3, total=10) == ("job", 75)
    assert tracker.update("job", completed=10, total=10) == ("job", 100)
    assert tracker.update("job:2", completed=1, total=2) is None
    assert tracker.update("other", completed=1, total=2) is None
    tracker.unregister("job")
    assert tracker.update("job:0", completed=1, total=2) is None


def test_apply_progress_only_moves_running_jobs(monkeypatch):
    store = AsyncDictStore()
    monkeypatch.setattr(video_progress, "VIDEO_STORE", store)

    async def run_test():
        await store.upsert("job", {"status": "queued", "progress": 0})
        await video_progress.apply_progress("job", progress=30)
        await video_progress.apply_progress("job", progress=20)
        assert await store.get("job") == {"status": "in_progress", "progress": 30}
        await video_progress.apply_progress("job", progress=100)
        assert (await store.get("job"))["progress"] == 99

        await store.upsert("done", {"status": "completed", "progress": 100})
        await video_progress.apply_progress("done", progress=40)
        assert await store.get("done") == {"status": "completed", "progress": 100}
        await video_progress.apply_progress("deleted", progress=40)

    asyncio.run(run_test())


def test_progress_listener_updates_store(monkeypatch):
    store = AsyncDictStore()
    tracker = video_progress.VideoProgressTracker()
    monkeypatch.setattr(video_progress, "VIDEO_STORE", store)
    monkeypatch.setattr(video_progress, "VIDEO_PROGRESS", tracker)
    # macOS limits an ipc path to 103 bytes, which pytest's tmp_path exceeds
    endpoint = f"ipc:///tmp/sglang-progress-test-{uuid.uuid4().hex[:8]}"

    async def run_test():
        await store.upsert("job", {"status": "queued", "progress": 0})
        tracker.register("job", num_outputs=1)
        listener = asyncio.create_task(video_progress.run_progress_listener(endpoint))
        context = zmq.asyncio.Context()
        push = context.socket(zmq.PUSH)
        push.connect(endpoint)
        try:
            await push.send(b"not json")
            await push.send_json(["job", 5, 10])
            for _ in range(200):
                if (await store.get("job"))["progress"] == 50:
                    break
                await asyncio.sleep(0.01)
            assert await store.get("job") == {"status": "in_progress", "progress": 50}
            assert not listener.done()
        finally:
            context.destroy(linger=0)
            listener.cancel()
            with suppress(asyncio.CancelledError):
                await listener

    asyncio.run(run_test())
    assert not os.path.exists(endpoint.removeprefix("ipc://"))
