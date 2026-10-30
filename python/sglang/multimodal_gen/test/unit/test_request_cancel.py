import asyncio
import json
import os
import subprocess
import sys
import threading
import time
import uuid
from contextlib import AsyncExitStack
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import torch.distributed as dist
import torch.multiprocessing as mp
import zmq
from fastapi import HTTPException
from tqdm.auto import tqdm

import sglang.multimodal_gen.runtime.managers.gpu_worker as gpu_worker_module
from sglang.multimodal_gen.configs.pipeline_configs.wan import WanT2V480PConfig
from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.entrypoints import http_server
from sglang.multimodal_gen.runtime.entrypoints.openai import utils as openai_utils
from sglang.multimodal_gen.runtime.entrypoints.openai import video_api, video_cancel
from sglang.multimodal_gen.runtime.entrypoints.openai.stores import AsyncDictStore
from sglang.multimodal_gen.runtime.managers import request_cancel
from sglang.multimodal_gen.runtime.managers import scheduler as scheduler_module
from sglang.multimodal_gen.runtime.managers.gpu_worker import GPUWorker
from sglang.multimodal_gen.runtime.managers.progress_reporter import ReportingTqdm
from sglang.multimodal_gen.runtime.managers.request_cancel import (
    CancelListener,
    RequestCancelledError,
    raise_if_cancelled,
)
from sglang.multimodal_gen.runtime.managers.scheduler import Scheduler
from sglang.multimodal_gen.runtime.pipelines_core.executors import (
    pipeline_executor as pipeline_executor_module,
)
from sglang.multimodal_gen.runtime.pipelines_core.executors.sync_executor import (
    SyncExecutor,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch, Req
from sglang.multimodal_gen.runtime.pipelines_core.stages import base as stage_base
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import (
    StageParallelismType,
)
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.scheduler_client import AsyncSchedulerClient
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.perf_logger import StageProfiler


def _req(request_id="job", cancellable=True, is_warmup=False):
    req = Req(
        sampling_params=SamplingParams(prompt="p", seed=1, request_id=request_id),
        is_warmup=is_warmup,
    )
    req.cancellable = cancellable
    return req


class _FakeListener:
    def __init__(self, cancelled_ids):
        self.cancelled_ids = set(cancelled_ids)

    def is_cancelled(self, request_ids):
        return bool(request_ids) and all(
            request_id in self.cancelled_ids for request_id in request_ids
        )


@pytest.fixture
def cancel_state(monkeypatch):
    """Enable the checks on a single-rank replica with a fake listener."""

    def set_cancelled(*request_ids):
        monkeypatch.setattr(request_cancel, "_listener", _FakeListener(request_ids))

    monkeypatch.setattr(request_cancel, "_enabled", True)
    monkeypatch.setattr(request_cancel, "model_parallel_is_initialized", lambda: False)
    set_cancelled()
    return set_cancelled


def _cancel_endpoint():
    return f"ipc:///tmp/sglang-cancel-test-{uuid.uuid4().hex[:8]}"


def _deliver_cancel(publish, listener, request_id="job"):
    for _ in range(500):
        try:
            publish(request_id)
        except video_cancel.CancelDeliveryError:
            pass
        if listener.is_cancelled([request_id]):
            return
        time.sleep(0.01)
    pytest.fail("Cancellation did not reach the listener")


def test_published_cancel_reaches_listener():
    endpoint = _cancel_endpoint()
    publisher = video_cancel.CancelPublisher(endpoint)
    context = zmq.Context()
    listener = CancelListener(context, endpoint)
    try:
        _deliver_cancel(publisher.publish, listener)
        assert listener.is_cancelled(["job"])
        assert listener.is_cancelled(["job:0", "job:1"])
        assert not listener.is_cancelled(["other"])
        assert not listener.is_cancelled(["job", "other"])
        assert not listener.is_cancelled([])
    finally:
        context.destroy(linger=0)
        publisher.close()
    assert not os.path.exists(endpoint.removeprefix("ipc://"))


def test_listener_forgets_oldest_ids(monkeypatch):
    monkeypatch.setattr(request_cancel, "_MAX_CANCELLED_IDS", 2)
    endpoint = _cancel_endpoint()
    publisher = video_cancel.CancelPublisher(endpoint)
    context = zmq.Context()
    listener = CancelListener(context, endpoint)
    try:
        _deliver_cancel(publisher.publish, listener, "probe")
        for request_id in ("a", "b"):
            publisher.publish(request_id)
        for _ in range(500):
            if listener.is_cancelled(["b"]):
                break
            time.sleep(0.01)
        assert listener.is_cancelled(["a", "b"])
        assert not listener.is_cancelled(["probe"])
    finally:
        context.destroy(linger=0)
        publisher.close()


def test_publish_without_publisher_is_a_no_op(monkeypatch):
    monkeypatch.setattr(video_cancel, "_publisher", None)
    video_cancel.publish_cancel("job")


@pytest.mark.parametrize(
    "module",
    [
        "sglang.multimodal_gen.runtime.managers.request_cancel",
        "sglang.multimodal_gen.runtime.managers.request_cancel_error",
        "sglang.multimodal_gen.runtime.utils.logging_utils",
        "sglang.multimodal_gen.runtime.utils.perf_logger",
        "sglang.multimodal_gen.runtime.entrypoints.openai.video_cancel",
        "sglang.multimodal_gen.runtime.pipelines_core.executors.pipeline_executor",
    ],
)
def test_module_imports_first(module):
    """The stages import request_cancel, so it must not import the stages."""
    subprocess.run([sys.executable, "-c", f"import {module}"], check=True)


def test_raise_if_cancelled(cancel_state):
    raise_if_cancelled([_req()])
    cancel_state("job")
    with pytest.raises(RequestCancelledError, match="job"):
        raise_if_cancelled([_req()])


@pytest.mark.parametrize(
    "reqs",
    [
        [],
        [_req(cancellable=False)],
        [_req(is_warmup=True)],
        # a group stops only when each of its requests is cancelled
        [_req(), _req("other")],
    ],
)
def test_raise_if_cancelled_skips(cancel_state, reqs):
    cancel_state("job")
    raise_if_cancelled(reqs)


def test_raise_if_cancelled_is_disabled_without_endpoint(cancel_state, monkeypatch):
    cancel_state("job")
    monkeypatch.setattr(request_cancel, "_enabled", False)
    raise_if_cancelled([_req()])
    assert request_cancel.step_cancel_check(_req()) is None


def _agree_on_cancel(rank, init_file, driver_cancelled, result_dir):
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=2
    )
    try:
        request_cancel._enabled = True
        # only the driver reads the cancelled ids; rank 1 answers the opposite
        # to show that its own view does not count
        cancelled = driver_cancelled if rank == 0 else not driver_cancelled
        request_cancel._listener = _FakeListener(["job"] if cancelled else [])
        request_cancel.model_parallel_is_initialized = lambda: True
        request_cancel.get_replica_group = lambda: SimpleNamespace(
            world_size=2, ranks=[0, 1], cpu_group=dist.group.WORLD
        )
        try:
            raise_if_cancelled([_req()])
            stopped = False
        except RequestCancelledError:
            stopped = True
        with open(os.path.join(result_dir, str(rank)), "w") as f:
            json.dump(stopped, f)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("driver_cancelled", [False, True])
def test_replica_ranks_follow_the_driver(tmp_path, driver_cancelled):
    """Each rank must stop at the same step, or the others wait in a collective."""
    mp.spawn(
        _agree_on_cancel,
        args=(str(tmp_path / "init"), driver_cancelled, str(tmp_path)),
        nprocs=2,
    )
    results = [json.loads((tmp_path / str(rank)).read_text()) for rank in (0, 1)]
    assert results == [driver_cancelled, driver_cancelled]


def test_step_check_runs_after_each_step():
    calls = []
    with ReportingTqdm(
        total=3,
        disable=True,
        reporter=None,
        request_ids=[],
        on_step=lambda: calls.append("update"),
    ) as bar:
        for _ in range(3):
            bar.update()
    # mininterval=0 makes tqdm call update() from its own __iter__ every step
    for _ in ReportingTqdm(
        range(3),
        mininterval=0,
        reporter=None,
        request_ids=[],
        on_step=lambda: calls.append("iter"),
    ):
        pass
    assert calls == ["update"] * 3 + ["iter"] * 3


def test_step_check_stops_the_loop(cancel_state):
    batch = _req()
    steps = 0
    with pytest.raises(RequestCancelledError):
        with ReportingTqdm(
            total=10,
            disable=True,
            reporter=None,
            request_ids=[],
            on_step=request_cancel.step_cancel_check(batch),
        ) as bar:
            for step in range(10):
                steps += 1
                if step == 2:
                    cancel_state("job")
                bar.update()
    assert steps == 3


class _Stage(stage_base.PipelineStage):
    def __init__(self, parallelism=StageParallelismType.REPLICATED):
        self._progress_range = (0.0, 1.0)
        self._parallelism = parallelism

    @property
    def parallelism_type(self):
        return self._parallelism

    def forward(self, batch, server_args):
        return batch


@pytest.mark.parametrize(
    "batch, parallelism, checked",
    [
        (_req(), StageParallelismType.REPLICATED, True),
        (_req(), StageParallelismType.CFG_PARALLEL, True),
        (_req(), StageParallelismType.MAIN_RANK_ONLY, False),
        (_req(), StageParallelismType.MAIN_RANK_ONLY_AND_SEND_TO_OTHERS, False),
        (_req(cancellable=False), StageParallelismType.REPLICATED, False),
        (_req(is_warmup=True), StageParallelismType.REPLICATED, False),
    ],
)
def test_progress_bar_checks_only_replicated_stages(
    cancel_state, monkeypatch, batch, parallelism, checked
):
    monkeypatch.setattr(stage_base, "get_progress_reporter", lambda: None)
    bar = _Stage(parallelism).progress_bar(total=3, batch=batch, disable=True)
    if checked:
        assert type(bar) is ReportingTqdm and bar._on_step is not None
        assert bar._reporter is None
    else:
        assert type(bar) is tqdm


def _bare_worker(monkeypatch):
    worker = GPUWorker.__new__(GPUWorker)
    worker.is_output_rank = True
    worker._realtime_sessions = SimpleNamespace(attach=Mock())
    worker._release_warmup_pool = Mock()
    worker._record_output_peak_memory = Mock()
    monkeypatch.setattr(current_platform, "is_cpu", lambda: True)
    return worker


def _cancelled(_reqs):
    raise RequestCancelledError(["job"])


def test_worker_skips_forward_of_cancelled_request(monkeypatch):
    worker = _bare_worker(monkeypatch)
    monkeypatch.setattr(gpu_worker_module, "raise_if_cancelled", _cancelled)
    forward = Mock()

    result = worker._execute_forward_common(
        SimpleNamespace(is_warmup=False, extra={}),
        forward_fn=forward,
        log_reqs=[],
        return_req=False,
        save_output_paths=Mock(),
        error_context="request job",
    )

    forward.assert_not_called()
    assert isinstance(result, OutputBatch)
    assert result.cancelled and "cancelled" in result.error


def test_stage_profiler_does_not_log_a_cancel_as_an_error():
    logger = Mock()
    profiler = StageProfiler("DenoisingStage", logger=logger, metrics=None)
    profiler.log_stage_start_end = True
    profiler.prometheus = None
    with pytest.raises(RequestCancelledError):
        with profiler:
            raise RequestCancelledError(["job"])
    logger.error.assert_not_called()
    with pytest.raises(RuntimeError):
        with profiler:
            raise RuntimeError("stage failed")
    logger.error.assert_called_once()


def test_cancelled_output_raises_a_cancel(monkeypatch):
    client = SimpleNamespace(
        forward=AsyncMock(
            return_value=OutputBatch(error="Request job was cancelled", cancelled=True)
        )
    )
    error = Mock()
    monkeypatch.setattr(openai_utils.logger, "error", error)
    batch = Req(sampling_params=SamplingParams(prompt="p", seed=1, request_id="job"))

    with pytest.raises(RequestCancelledError, match="job"):
        asyncio.run(openai_utils.process_generation_batch(client, batch))
    error.assert_not_called()


def test_sequential_worker_does_not_skip_a_request(monkeypatch):
    """A skipped request would leave its output to the next request of the group."""
    worker = _bare_worker(monkeypatch)
    monkeypatch.setattr(gpu_worker_module, "raise_if_cancelled", _cancelled)
    output = OutputBatch()

    result = worker._execute_forward_common(
        SimpleNamespace(is_warmup=False, extra={}),
        forward_fn=lambda: output,
        log_reqs=[],
        return_req=False,
        save_output_paths=Mock(),
        error_context="grouped request job",
        propagate_forward_errors=True,
        check_cancelled=False,
    )

    assert result is output


@pytest.mark.parametrize("propagate", [False, True])
def test_worker_reports_cancel_inside_forward(monkeypatch, propagate):
    worker = _bare_worker(monkeypatch)

    def run():
        return worker._execute_forward_common(
            SimpleNamespace(is_warmup=False, extra={}),
            forward_fn=lambda: _cancelled([]),
            log_reqs=[],
            return_req=False,
            save_output_paths=Mock(),
            error_context="request job",
            propagate_forward_errors=propagate,
        )

    if propagate:
        with pytest.raises(RequestCancelledError):
            run()
    else:
        assert "cancelled" in run().error


def _sequential_worker(monkeypatch):
    """Return a worker whose pipeline yields one output per request it runs."""
    worker = _bare_worker(monkeypatch)
    worker.server_args = SimpleNamespace(
        pipeline_config=SimpleNamespace(
            supports_sequential_multi_output_inference=lambda: False
        )
    )
    worker._execute_forward_common = lambda req, *, forward_fn, **kwargs: forward_fn()
    groups = []
    outputs = {}

    def forward_batch_sequentially(reqs, server_args):
        groups.append([req.request_id for req in reqs])
        return (outputs.setdefault(req.request_id, OutputBatch()) for req in reqs)

    worker.pipeline = SimpleNamespace(
        forward_batch_sequentially=forward_batch_sequentially
    )
    return worker, groups, outputs


def test_sequential_worker_skips_a_cancelled_request(monkeypatch, cancel_state):
    worker, groups, outputs = _sequential_worker(monkeypatch)
    cancel_state("a")

    results = list(worker.execute_forward_sequentially([_req("a"), _req("b")]))

    assert groups == [["b"]]
    assert results[0].cancelled and "cancelled" in results[0].error
    assert results[1] is outputs["b"]


def test_sequential_worker_does_not_start_a_cancelled_group(monkeypatch, cancel_state):
    worker, groups, _ = _sequential_worker(monkeypatch)
    cancel_state("a")

    results = list(worker.execute_forward_sequentially([_req("a")]))

    assert groups == []
    assert len(results) == 1 and results[0].cancelled


def _sequential_executor(execute):
    executor = object.__new__(SyncExecutor)
    executor.execute_group = lambda stages, batches, server_args: batches
    executor.execute = execute
    first_stage = SimpleNamespace(
        iter_sequential_requests=lambda batch, server_args: iter((batch,))
    )
    return executor, [first_stage, object()]


def test_sequential_outputs_stop_after_a_cancel(monkeypatch, cancel_state):
    error = Mock()
    monkeypatch.setattr(pipeline_executor_module.logger, "error", error)
    executed = []

    def execute(stages, batch, server_args):
        executed.append(batch.request_id)
        # the client cancels the job while its first output denoises
        cancel_state("job:0", "job:1", "job:2")
        raise RequestCancelledError(["job"])

    executor, stages = _sequential_executor(execute)
    batches = [_req(f"job:{i}") for i in range(3)]

    outputs = list(executor.execute_group_sequentially(stages, batches, None))

    assert executed == ["job:0"]
    assert [output.cancelled for output in outputs] == [True, True, True]
    error.assert_not_called()


def test_sequential_cancel_skips_only_the_cancelled_request(cancel_state):
    executed = []

    def execute(stages, batch, server_args):
        executed.append(batch.request_id)
        return OutputBatch()

    executor, stages = _sequential_executor(execute)
    cancel_state("job")

    outputs = list(
        executor.execute_group_sequentially(stages, [_req("job"), _req("other")], None)
    )

    assert executed == ["other"]
    assert [output.cancelled for output in outputs] == [True, False]


@pytest.mark.parametrize(
    "first_error, cancelled", [(None, True), ("Error executing job:0", False)]
)
def test_merged_outputs_keep_the_cancel(first_error, cancelled):
    merged = GPUWorker._merge_expanded_output_batches(
        [
            OutputBatch(error=first_error),
            OutputBatch(error="Request job:1 was cancelled", cancelled=True),
        ]
    )
    assert merged.error is not None
    assert merged.cancelled is cancelled


_VIDEO_JOB = {
    "id": "job",
    "object": "video",
    "model": "m",
    "status": "in_progress",
    "progress": 40,
    "created_at": 0,
    "size": "1x1",
    "seconds": "1",
}


def test_delete_publishes_cancel(monkeypatch):
    store = AsyncDictStore()
    published = []
    monkeypatch.setattr(video_api, "VIDEO_STORE", store)
    monkeypatch.setattr(video_api, "publish_cancel", published.append)

    async def run_test():
        await store.upsert("job", dict(_VIDEO_JOB))
        response = await video_api.delete_video("job")
        assert response.status == "deleted"
        assert await store.get("job") is None
        with pytest.raises(HTTPException):
            await video_api.delete_video("job")

    asyncio.run(run_test())
    assert published == ["job"]


def test_delete_preserves_job_when_cancel_needs_retry(monkeypatch):
    store = AsyncDictStore()
    endpoint = _cancel_endpoint()
    publisher = video_cancel.CancelPublisher(endpoint)
    context = zmq.Context()
    monkeypatch.setattr(video_api, "VIDEO_STORE", store)
    monkeypatch.setattr(video_api, "publish_cancel", publisher.publish)

    async def run_test():
        await store.upsert("job", dict(_VIDEO_JOB))
        with pytest.raises(HTTPException) as error:
            await video_api.delete_video("job")
        assert error.value.status_code == 503
        assert error.value.headers == {"Retry-After": "1"}
        assert (await store.get("job"))["status"] == "in_progress"
        listener = CancelListener(context, endpoint)
        _deliver_cancel(publisher.publish, listener, "ready")
        assert (await video_api.delete_video("job")).status == "deleted"
        assert await store.get("job") is None
        for _ in range(500):
            if listener.is_cancelled(["job"]):
                break
            await asyncio.sleep(0.01)
        assert listener.is_cancelled(["job"])

    try:
        asyncio.run(run_test())
    finally:
        context.destroy(linger=0)
        publisher.close()


def test_delete_completed_job_needs_no_cancel_channel(monkeypatch):
    store = AsyncDictStore()
    publish = Mock(side_effect=video_cancel.CancelDeliveryError("unavailable"))
    monkeypatch.setattr(video_api, "VIDEO_STORE", store)
    monkeypatch.setattr(video_api, "publish_cancel", publish)

    async def run_test():
        await store.upsert("job", {**_VIDEO_JOB, "status": "completed"})
        assert (await video_api.delete_video("job")).status == "deleted"

    asyncio.run(run_test())
    publish.assert_not_called()


def test_delete_after_rpc_timeout_publishes_cancel(monkeypatch):
    """A timeout marks the job failed, but the GPU still runs it."""
    store = AsyncDictStore()
    published = []
    monkeypatch.setattr(video_api, "VIDEO_STORE", store)
    monkeypatch.setattr(video_api, "publish_cancel", published.append)

    async def timed_out(*args, **kwargs):
        raise TimeoutError("Scheduler did not respond in time.")

    monkeypatch.setattr(video_api, "process_generation_batch", timed_out)

    async def run_test():
        await store.upsert("job", dict(_VIDEO_JOB))
        task = video_api._start_video_job(
            "job",
            video_api._dispatch_job_async(
                "job",
                Req(sampling_params=SamplingParams(prompt="p")),
                resources=AsyncExitStack(),
            ),
        )
        await task
        assert (await store.get("job"))["status"] == "failed"
        await video_api.delete_video("job")

    asyncio.run(run_test())
    assert published == ["job"]


def test_stopped_job_of_deleted_video_is_not_failed(monkeypatch):
    store = AsyncDictStore()
    monkeypatch.setattr(video_api, "VIDEO_STORE", store)

    async def stopped(*args, **kwargs):
        # DELETE removed the job while it ran
        await store.pop("job")
        raise RuntimeError("Request job was cancelled")

    monkeypatch.setattr(video_api, "process_generation_batch", stopped)
    exception = Mock()
    monkeypatch.setattr(video_api.logger, "exception", exception)

    async def run_test():
        await store.upsert("job", {"status": "in_progress", "progress": 40})
        await video_api._dispatch_job_async(
            "job",
            Req(sampling_params=SamplingParams(prompt="test")),
            resources=AsyncExitStack(),
        )
        assert await store.get("job") is None

    asyncio.run(run_test())
    exception.assert_not_called()


@pytest.mark.parametrize(
    "port, role, expected",
    [
        (None, RoleType.MONOLITHIC, None),
        (30000, RoleType.DENOISER, None),
        (30000, RoleType.MONOLITHIC, "sglang-diffusion-cancel-30000"),
    ],
)
def test_cancel_endpoint(port, role, expected):
    endpoint = ServerArgs.cancel_endpoint.fget(
        SimpleNamespace(port=port, disagg_role=role)
    )
    if expected is None:
        assert endpoint is None
    else:
        assert endpoint.startswith("ipc://") and endpoint.endswith(expected)


def test_init_request_cancel(monkeypatch):
    monkeypatch.setattr(request_cancel, "_enabled", False)
    monkeypatch.setattr(request_cancel, "_listener", None)
    # a rank that is not the driver takes part in each decision without a socket
    request_cancel.init_request_cancel(_cancel_endpoint(), context=None)
    assert request_cancel._enabled and request_cancel._listener is None
    context = zmq.Context()
    try:
        request_cancel.init_request_cancel(_cancel_endpoint(), context=context)
        assert isinstance(request_cancel._listener, CancelListener)
    finally:
        context.destroy(linger=0)


def test_init_request_cancel_survives_a_failed_connect(monkeypatch):
    monkeypatch.setattr(request_cancel, "_enabled", False)
    monkeypatch.setattr(request_cancel, "_listener", None)
    context = zmq.Context()
    try:
        request_cancel.init_request_cancel("bogus://endpoint", context=context)
    finally:
        context.destroy(linger=0)
    assert request_cancel._enabled and request_cancel._listener is None


def test_cancel_publisher_start_and_stop(monkeypatch):
    monkeypatch.setattr(video_cancel, "_publisher", None)
    endpoint = _cancel_endpoint()
    video_cancel.start_cancel_publisher(endpoint)
    context = zmq.Context()
    listener = CancelListener(context, endpoint)
    try:
        _deliver_cancel(video_cancel.publish_cancel, listener)
        assert listener.is_cancelled(["job"])
    finally:
        context.destroy(linger=0)
        video_cancel.stop_cancel_publisher()
    assert video_cancel._publisher is None
    assert not os.path.exists(endpoint.removeprefix("ipc://"))
    video_cancel.stop_cancel_publisher()


def test_cancel_publisher_failures(monkeypatch):
    monkeypatch.setattr(video_cancel, "_publisher", None)
    with pytest.raises(zmq.ZMQError):
        video_cancel.start_cancel_publisher("bogus://endpoint")
    assert video_cancel._publisher is None

    publisher = video_cancel.CancelPublisher(_cancel_endpoint())
    try:
        with pytest.raises(video_cancel.CancelDeliveryError, match="connected"):
            publisher.publish("job")
        publisher._socket.close()
        with pytest.raises(video_cancel.CancelDeliveryError):
            publisher.publish("job")
    finally:
        publisher.close()


def test_cancel_backpressure_does_not_drop_accepted_commands():
    endpoint = _cancel_endpoint()
    publisher = video_cancel.CancelPublisher(endpoint)
    context = zmq.Context()
    listener = CancelListener(context, endpoint)
    try:
        _deliver_cancel(publisher.publish, listener, "ready")
        accepted = set()
        rejected = set()
        for index in range(3500):
            request_id = f"job-{index}"
            try:
                publisher.publish(request_id)
                accepted.add(request_id)
            except video_cancel.CancelDeliveryError:
                rejected.add(request_id)
        assert rejected, "The burst must exercise backpressure"
        for _ in range(500):
            listener.is_cancelled([])
            if accepted.issubset(listener._cancelled):
                break
            time.sleep(0.01)
        assert accepted.issubset(listener._cancelled)
        for request_id in rejected:
            _deliver_cancel(publisher.publish, listener, request_id)
        assert all(listener._is_cancelled(f"job-{i}") for i in range(3500))
    finally:
        context.destroy(linger=0)
        publisher.close()


def test_cancel_waits_for_all_replica_subscriptions():
    endpoint = _cancel_endpoint()
    publisher = video_cancel.CancelPublisher(endpoint, replicas=2)
    context = zmq.Context()
    first = CancelListener(context, endpoint)
    try:
        for _ in range(100):
            with pytest.raises(video_cancel.CancelDeliveryError):
                publisher.publish("job")
            time.sleep(0.001)
        second = CancelListener(context, endpoint)
        _deliver_cancel(publisher.publish, second)
        assert first.is_cancelled(["job"])
        first._socket.close(linger=0)
        for _ in range(500):
            try:
                publisher.publish("next")
            except video_cancel.CancelDeliveryError:
                break
            time.sleep(0.01)
        else:
            pytest.fail("Disconnected replica was not detected")
    finally:
        context.destroy(linger=0)
        publisher.close()


def test_idle_scheduler_drains_cancellations(monkeypatch):
    endpoint = _cancel_endpoint()
    publisher = video_cancel.CancelPublisher(endpoint)
    context = zmq.Context()
    listener = CancelListener(context, endpoint)
    monkeypatch.setattr(request_cancel, "_listener", listener)
    scheduler = object.__new__(Scheduler)
    scheduler.receiver = Mock()
    scheduler.receiver.recv_multipart.side_effect = zmq.Again
    scheduler._socket_lock = threading.Lock()
    scheduler.server_args = SimpleNamespace(
        comfyui_mode=False, sp_degree=1, enable_cfg_parallel=False, tp_size=1
    )
    try:
        _deliver_cancel(publisher.publish, listener, "ready")
        publisher.publish("idle")
        for _ in range(500):
            assert scheduler.recv_reqs() == []
            if "idle" in listener._cancelled:
                break
            time.sleep(0.01)
        assert "idle" in listener._cancelled
    finally:
        context.destroy(linger=0)
        publisher.close()


@pytest.mark.parametrize(
    "members, cancellable",
    [
        ([(True, True), (True, True)], True),
        ([(True, True), (False, True)], False),
        # a member that reports no progress is missing from merged_request_ids
        ([(True, True), (True, False)], False),
    ],
)
def test_merged_batch_is_cancellable_only_when_each_member_is(
    monkeypatch, members, cancellable
):
    scheduler = object.__new__(Scheduler)
    monkeypatch.setattr(scheduler, "_can_dynamic_batch", lambda base, other: True)
    reqs = []
    for index, (member_cancellable, report_progress) in enumerate(members):
        req = _req(f"job{index}", cancellable=member_cancellable)
        req.report_progress = report_progress
        reqs.append(req)

    merged = scheduler._try_merge_generation_reqs(reqs)

    assert merged.cancellable is cancellable


@pytest.mark.parametrize("cancelled", [(), ("b",), ("a", "b"), ("a", "b", "c")])
def test_queued_cancel_is_filtered_before_dynamic_batch(cancel_state, cancelled):
    scheduler = object.__new__(Scheduler)
    scheduler._async_output_save = False
    scheduler.server_args = SimpleNamespace(pipeline_config=WanT2V480PConfig())
    scheduler._batching_max_size = 3
    scheduler._batching_delay_s = 0
    reqs = [_req(name) for name in ("a", "b", "c")]
    for req in reqs:
        req.report_progress = True
    assert scheduler.server_args.pipeline_config.supports_dynamic_batching()
    assert scheduler._can_dynamic_batch(reqs[0], reqs[1])
    executed = []

    def forward(batches):
        raise_if_cancelled(batches)
        ids = [rid for batch in batches for rid in batch.client_request_ids()]
        executed.append(ids)
        return OutputBatch(output=ids)

    scheduler.worker = SimpleNamespace(
        is_sleeping=lambda: False, execute_forward=forward
    )
    scheduler._split_batched_output = lambda output, batches: [
        OutputBatch(output=[batch.request_id]) for batch in batches
    ]
    cancel_state(*cancelled)

    outputs = scheduler._handle_generation(reqs)

    live = [req.request_id for req in reqs if req.request_id not in cancelled]
    assert executed == ([live] if live else [])
    for req, output in zip(reqs, outputs, strict=True):
        assert output.cancelled is (req.request_id in cancelled)
        if not output.cancelled:
            assert output.output == [req.request_id]


def test_explicit_group_keeps_single_reply(cancel_state):
    scheduler = object.__new__(Scheduler)
    scheduler._async_output_save = False
    scheduler.server_args = SimpleNamespace(pipeline_config=WanT2V480PConfig())
    output = OutputBatch(output=["grouped output"])
    forward = Mock(return_value=output)
    scheduler.worker = SimpleNamespace(
        is_sleeping=lambda: False, execute_forward=forward
    )
    reqs = [_req("cancelled"), _req("live"), _req("another")]
    cancel_state("cancelled")

    assert scheduler._handle_generation(reqs, allow_dynamic_batching=False) is output
    forward.assert_called_once_with(reqs)


@pytest.mark.parametrize("streaming", [False, True])
def test_filtered_batch_preserves_reply_mode(cancel_state, streaming):
    scheduler = object.__new__(Scheduler)
    scheduler._async_output_save = not streaming
    scheduler._inflight_finalizes = []
    scheduler._reap_finalizes = lambda: None
    scheduler.server_args = SimpleNamespace(
        pipeline_config=SimpleNamespace(
            supports_sequential_multi_output_inference=lambda: streaming,
        )
    )
    output = OutputBatch(output=["live"])
    result = (
        scheduler_module._SequentiallyReturnedOutputs(iter([output]))
        if streaming
        else scheduler_module._DeferredOutput(output, lambda: None)
    )
    scheduler.worker = SimpleNamespace(is_sleeping=lambda: False)
    scheduler._iter_grouped_outputs_sequentially = lambda reqs: result.outputs
    scheduler._execute_forward_with_deferred_save = lambda reqs: result
    live = _req("live")
    live.num_outputs_per_prompt = 2 if streaming else 1
    cancel_state("cancelled")

    outputs = scheduler._handle_generation([_req("cancelled"), live])

    if streaming:
        assert isinstance(outputs, scheduler_module._SequentiallyReturnedOutputs)
        outputs = list(outputs.outputs)
    assert outputs[0].cancelled
    assert outputs[1] is (output if streaming else result)


@pytest.mark.parametrize("native_grouped", [False, True])
def test_cancelled_batch_returns_cancelled_outputs(monkeypatch, native_grouped):
    """A merged dynamic batch and a native grouped batch keep the cancel."""
    scheduler = object.__new__(Scheduler)
    scheduler._async_output_save = False
    scheduler.server_args = SimpleNamespace(
        pipeline_config=SimpleNamespace(
            supports_sequential_multi_output_inference=lambda: False,
            supports_native_grouped_requests=lambda: native_grouped,
            supports_sequential_dit_inference=lambda: False,
        )
    )
    scheduler.worker = SimpleNamespace(
        is_sleeping=lambda: False,
        execute_forward=Mock(
            return_value=OutputBatch(
                error="Request job0, job1 was cancelled", cancelled=True
            )
        ),
    )
    monkeypatch.setattr(scheduler, "_try_merge_generation_reqs", lambda reqs: reqs[0])
    error = Mock()
    monkeypatch.setattr(scheduler_module.logger, "error", error)

    outputs = scheduler._handle_generation([_req("job0"), _req("job1")])

    assert [output.cancelled for output in outputs] == [True, True]
    assert all("cancelled" in output.error for output in outputs)
    error.assert_not_called()


def test_lifespan_starts_and_stops_the_cancel_publisher(monkeypatch):
    started = Mock()
    stopped = Mock()
    monkeypatch.setattr(http_server, "start_cancel_publisher", started)
    monkeypatch.setattr(http_server, "stop_cancel_publisher", stopped)
    with zmq.Context() as context, context.socket(zmq.REP) as probe:
        broker_port = probe.bind_to_random_port("tcp://127.0.0.1")
        probe.unbind(probe.last_endpoint)
    args = SimpleNamespace(
        enable_metrics=False,
        warmup_mode="off",
        progress_endpoint=None,
        cancel_endpoint="ipc:///tmp/sglang-cancel-lifespan",
        dp_size=2,
        broker_port=broker_port,
        prompt_enhancer_config=None,
        log_requests=False,
        log_requests_level=0,
        log_requests_format="text",
        log_requests_target=None,
        pipeline_config=Mock(
            supports_action_endpoint=lambda: False,
            supports_openpi_endpoint=lambda: False,
        ),
    )

    async def run_test():
        client = AsyncSchedulerClient()
        monkeypatch.setattr(http_server, "async_scheduler_client", client)
        app = http_server.create_app(args)
        async with http_server.lifespan(app):
            started.assert_called_once_with("ipc:///tmp/sglang-cancel-lifespan", 2)
            stopped.assert_not_called()
        stopped.assert_called_once()

    asyncio.run(run_test())
