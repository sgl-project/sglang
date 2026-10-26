import asyncio
import os
import threading
import uuid
from contextlib import suppress
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import zmq
import zmq.asyncio
from tqdm.auto import tqdm

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.entrypoints.openai import utils as openai_utils
from sglang.multimodal_gen.runtime.entrypoints.openai import video_api, video_progress
from sglang.multimodal_gen.runtime.entrypoints.openai.stores import AsyncDictStore
from sglang.multimodal_gen.runtime.managers import progress_reporter
from sglang.multimodal_gen.runtime.managers.progress_reporter import (
    ProgressReporter,
    ReportingTqdm,
)
from sglang.multimodal_gen.runtime.pipelines_core.executors.sync_executor import (
    SyncExecutor,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch, Req
from sglang.multimodal_gen.runtime.pipelines_core.stages import base as stage_base
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
            "failed-output", Req(sampling_params=SamplingParams(prompt="test"))
        )
        return await video_api.VIDEO_STORE.pop("failed-output")

    job = asyncio.run(run())
    assert job["status"] == "failed"
    assert "save failed" in job["error"]["message"]
    assert job["file_paths"] is None and job["url"] is None


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


@pytest.mark.parametrize("peer", ["no-peer", "never-reads", "terminated"])
def test_progress_reporter_never_blocks(peer):
    context = zmq.Context()
    endpoint = f"inproc://progress-{uuid.uuid4().hex}"
    pull = context.socket(zmq.PULL)
    pull.setsockopt(zmq.RCVHWM, 10)
    if peer == "never-reads":
        pull.bind(endpoint)
    reporter = ProgressReporter(context, endpoint)
    if peer == "terminated":
        context.destroy(linger=0)

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


@pytest.mark.parametrize(
    "is_warmup, report_progress", [(False, True), (True, True), (False, False)]
)
def test_progress_bar_reports_only_real_requests(
    monkeypatch, is_warmup, report_progress
):
    monkeypatch.setattr(stage_base, "get_progress_reporter", lambda: object())
    batch = Req(
        sampling_params=SamplingParams(prompt="p", seed=1, request_id="job"),
        is_warmup=is_warmup,
    )
    batch.report_progress = report_progress

    bar = stage_base.PipelineStage.progress_bar(
        SimpleNamespace(_progress_range=(0.0, 1.0)), total=3, batch=batch
    )

    assert type(bar) is (ReportingTqdm if report_progress and not is_warmup else tqdm)


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


@pytest.mark.parametrize(
    "invalid_message",
    [
        b"not json",
        b"[1, 5, 10]",
        b"[null, 5, 10]",
        b'["job", "5", 10]',
        b'["job", 5, "10"]',
        b'["job", -1, 10]',
        b'["job", 11, 10]',
        b'["job", 5, -10]',
        b'["job", true, 10]',
        b'["job", NaN, 10]',
        b'["job", 5, Infinity]',
        pytest.param(b'["job", 0.5, 1' + b"0" * 400 + b"]", id="overflow"),
        b'["job", 5]',
        b"\xff",
    ],
)
def test_progress_listener_updates_store(monkeypatch, invalid_message):
    store = AsyncDictStore()
    tracker = video_progress.VideoProgressTracker()
    monkeypatch.setattr(video_progress, "VIDEO_STORE", store)
    monkeypatch.setattr(video_progress, "VIDEO_PROGRESS", tracker)
    # macOS limits an ipc path to 103 bytes, which pytest's tmp_path exceeds
    endpoint = f"ipc:///tmp/sglang-progress-test-{uuid.uuid4().hex[:8]}"

    async def run_test():
        await store.upsert("job", {"status": "queued", "progress": 0})
        await store.upsert("barrier", {"status": "queued", "progress": 0})
        tracker.register("job", num_outputs=1)
        tracker.register("barrier", num_outputs=1)
        listener = asyncio.create_task(video_progress.run_progress_listener(endpoint))
        context = zmq.asyncio.Context()
        push = context.socket(zmq.PUSH)
        push.connect(endpoint)
        try:
            await push.send(invalid_message)
            # The same PUSH socket preserves order: the barrier proves that
            # the malformed message was consumed before checking the job.
            await push.send_json(["barrier", 0, 0])
            for _ in range(200):
                if (await store.get("barrier"))["status"] == "in_progress":
                    break
                await asyncio.sleep(0.01)
            assert (await store.get("barrier"))["status"] == "in_progress"
            assert await store.get("job") == {"status": "queued", "progress": 0}
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


def test_progress_reporter_reconnects_after_listener_restart(monkeypatch):
    """Restarting telemetry must not require restarting the generating worker."""
    store = AsyncDictStore()
    tracker = video_progress.VideoProgressTracker()
    monkeypatch.setattr(video_progress, "VIDEO_STORE", store)
    monkeypatch.setattr(video_progress, "VIDEO_PROGRESS", tracker)
    endpoint = f"ipc:///tmp/sglang-progress-test-{uuid.uuid4().hex[:8]}"
    context = zmq.Context()
    reporter = ProgressReporter(context, endpoint)

    async def run_test():
        await store.upsert("job", {"status": "queued", "progress": 0})
        tracker.register("job", num_outputs=1)
        for completed in (25, 75):
            listener = asyncio.create_task(
                video_progress.run_progress_listener(endpoint)
            )
            try:
                for _ in range(500):
                    reporter.step(["job"], completed=completed, total=100)
                    await asyncio.sleep(0.01)
                    if (await store.get("job"))["progress"] == completed:
                        break
                assert (await store.get("job"))["progress"] == completed
                assert not listener.done()
            finally:
                listener.cancel()
                with suppress(asyncio.CancelledError):
                    await listener
            assert not os.path.exists(endpoint.removeprefix("ipc://"))
            # Updates during the outage are advisory and may be dropped.
            for _ in range(1000):
                reporter.step(["job"], completed=50, total=100)

    try:
        asyncio.run(run_test())
    finally:
        context.destroy(linger=0)


@pytest.mark.parametrize("terminal_status", ["completed", "failed", "deleted"])
def test_late_progress_cannot_revive_terminal_job(monkeypatch, terminal_status):
    store = AsyncDictStore()
    tracker = video_progress.VideoProgressTracker()
    monkeypatch.setattr(video_progress, "VIDEO_STORE", store)

    async def run_test():
        tracker.register("job", num_outputs=2)
        await store.upsert("job", {"status": "queued", "progress": 0})
        tracker.update("job:0", completed=10, total=10)
        await store.update_fields("job", {"status": terminal_status, "progress": 100})
        if terminal_status == "deleted":
            await store.pop("job")
        result = tracker.update("job:1", completed=8, total=10)
        await video_progress.apply_progress(*result)
        job = await store.get("job")
        if terminal_status == "deleted":
            assert job is None
        else:
            assert job == {"status": terminal_status, "progress": 100}
        tracker.unregister("job")
        assert tracker.update("job:1", completed=10, total=10) is None
        assert not tracker._jobs

    asyncio.run(run_test())


@pytest.mark.parametrize("outcome", ["failed", "cancelled"])
def test_dispatch_always_unregisters_progress(monkeypatch, outcome):
    store = AsyncDictStore()
    tracker = video_progress.VideoProgressTracker()
    monkeypatch.setattr(video_api, "VIDEO_STORE", store)
    monkeypatch.setattr(video_api, "VIDEO_PROGRESS", tracker)

    async def run_test():
        started = asyncio.Event()

        async def generate(*args, **kwargs):
            started.set()
            if outcome == "failed":
                raise RuntimeError("generation failed")
            await asyncio.Event().wait()

        monkeypatch.setattr(video_api, "process_generation_batch", generate)
        await store.upsert("job", {"status": "queued", "progress": 0})
        tracker.register("job", num_outputs=2)
        task = asyncio.create_task(video_api._dispatch_job_async("job", _FakeBatch()))
        await started.wait()
        if outcome == "cancelled":
            task.cancel()
        with suppress(asyncio.CancelledError):
            await task
        assert not tracker._jobs
        if outcome == "failed":
            assert (await store.get("job"))["status"] == "failed"

    asyncio.run(run_test())


@pytest.mark.parametrize("num_stages", [1, 2, 3])
def test_progress_covers_all_denoising_stages(
    monkeypatch, progress_channel, num_stages
):
    """Refinement must advance progress rather than freeze after the first pass."""
    reporter, pull = progress_channel
    monkeypatch.setattr(progress_reporter, "_reporter", reporter)

    class DenoisingStage(stage_base.PipelineStage):
        role_affinity = RoleType.DENOISER

        def __init__(self):
            pass

        def forward(self, batch, server_args):
            return batch

    args = SimpleNamespace(disable_conditioning_cache=True, use_fsdp_inference=False)
    executor = SyncExecutor(args)
    executor.component_residency_manager = SimpleNamespace(
        begin_request=lambda *args: None, finish_request=lambda: None
    )
    batch = Req(sampling_params=SamplingParams(prompt="p", seed=1, request_id="job"))
    batch.report_progress = True
    stages = [DenoisingStage() for _ in range(num_stages)]
    with executor._component_residency_request(stages, batch, args):
        for stage in stages:
            with stage.progress_bar(total=2, disable=True, batch=batch) as bar:
                bar.update()
                bar.update()
    percentages = [completed / total for _, completed, total in _received(pull)]
    assert percentages == pytest.approx(
        [(i + 1) / (num_stages * 2) for i in range(num_stages * 2)]
    )
