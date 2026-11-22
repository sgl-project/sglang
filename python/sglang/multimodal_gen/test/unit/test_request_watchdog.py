# SPDX-License-Identifier: Apache-2.0
"""Request watchdog (--request-time-limit): limits, timing and the failed state."""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from types import SimpleNamespace

import psutil
import pytest
import zmq

from sglang.multimodal_gen.configs.sample.action import ActionSamplingParams
from sglang.multimodal_gen.configs.sample.minimax_h3 import MiniMaxH3SamplingParams
from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.runtime.entrypoints import http_server
from sglang.multimodal_gen.runtime.managers import request_watchdog
from sglang.multimodal_gen.runtime.managers.request_watchdog import (
    ExecutionReporter,
    RequestWatchdog,
    parse_request_time_limits,
    request_time_limit,
    request_watchdog_endpoint,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.scheduler_client import AsyncSchedulerClient
from sglang.multimodal_gen.runtime.utils.perf_logger import StageProfiler

LIMITS = {"t2va": 20.0, "fl2va": 30.0, "ref2va": 120.0}


@dataclass
class _TaskParams(SamplingParams):
    task: str | None = None

    def request_task(self) -> str | None:
        return self.task


def _req(request_id: str, task: str | None = "t2va", **kwargs) -> Req:
    return Req(request_id=request_id, sampling_params=_TaskParams(task=task), **kwargs)


def test_parse_limits() -> None:
    assert parse_request_time_limits(None) == {}
    assert parse_request_time_limits(["t2va=20", "fl2va=30", "ref2va=120"]) == LIMITS
    assert parse_request_time_limits(["90", "ref2va=120"]) == {"*": 90, "ref2va": 120}
    # later entries win; 0 and off remove a limit
    assert parse_request_time_limits(["90", "ref2va=120", "0"]) == {"ref2va": 120}
    assert parse_request_time_limits(["off"]) == {}
    assert parse_request_time_limits(["T2VA = 1.5"]) == {"t2va": 1.5}
    for bad in ("abc", "t2va=", "t2va=-1", "nan", "inf", "ref2va=1m"):
        with pytest.raises(ValueError, match="--request-time-limit"):
            parse_request_time_limits([bad])


def test_limit_lookup_and_request_task() -> None:
    assert request_time_limit(LIMITS, "ref2va") == 120
    assert request_time_limit(LIMITS, "i2va") is None
    assert request_time_limit({**LIMITS, "*": 30}, "i2va") == 30
    assert request_time_limit({"*": 30}, None) == 30
    assert MiniMaxH3SamplingParams(prompt="p", task="fl2va").request_task() == "fl2va"
    assert SamplingParams().request_task() is None
    # action policies' sampling params are a separate class
    assert ActionSamplingParams().request_task() is None


def test_off_unless_a_limit_is_set() -> None:
    args = SimpleNamespace(
        progress_endpoint="ipc:///tmp/progress", request_time_limit=None, port=61000
    )
    assert request_watchdog_endpoint(args) is None
    args.request_time_limit = ["0"]
    assert request_watchdog_endpoint(args) is None
    args.request_time_limit = ["ref2va=120"]
    assert request_watchdog_endpoint(args).endswith("sglang-diffusion-watchdog-61000")
    # no HTTP server to report to (offline generation, disaggregated roles)
    args.progress_endpoint = None
    assert request_watchdog_endpoint(args) is None


class _Pair:
    """An ExecutionReporter wired to a PULL socket the test reads."""

    def __init__(self, tmp_path, limits=LIMITS):
        self.context = zmq.Context()
        self.pull = self.context.socket(zmq.PULL)
        endpoint = f"ipc://{tmp_path}/watchdog"
        self.pull.bind(endpoint)
        self.reporter = ExecutionReporter(self.context, endpoint, limits, replica=1)

    def messages(self) -> list:
        out = []
        while self.pull.poll(200):
            out.append(self.pull.recv_json())
        return out

    def close(self) -> None:
        self.context.destroy(linger=0)


def test_reporter_times_only_served_requests(tmp_path) -> None:
    pair = _Pair(tmp_path)
    try:
        before = time.monotonic()
        with pair.reporter.watch(_req("r1", "ref2va")):
            pair.reporter.enter_region("MiniMaxH3DenoisingStage", False)
            pair.reporter.enter_region("denoising_step_3", True)
            # the output finalizer thread is not part of the timed request
            other = threading.Thread(
                target=pair.reporter.enter_region, args=("save", False)
            )
            other.start()
            other.join()
        pair.reporter.enter_region("after", False)
        begin, stage, step, end = pair.messages()
        assert begin[:5] == [1, "begin", "r1", "ref2va", 120]
        assert before <= begin[5] <= time.monotonic()
        assert stage == [1, "stage", "MiniMaxH3DenoisingStage"]
        assert step == [1, "step", "denoising_step_3"]
        assert end == [1, "end"]

        warmup = _req("w", is_warmup=True)
        prewarm = _req("p", extra={"server_internal_prewarm": True})
        for req in (None, warmup, prewarm, _req("x", "i2va"), Req(request_id="y")):
            with pair.reporter.watch(req):
                pair.reporter.enter_region("stage", False)
        assert pair.messages() == []
    finally:
        pair.close()


def test_unwatched_region_skips_step_classification(monkeypatch) -> None:
    monkeypatch.setattr(request_watchdog, "execution_reporter", None)
    region = StageProfiler("denoising_step_0", logger=None, metrics=None)
    monkeypatch.setattr(
        region, "_should_record_as_step", lambda: pytest.fail("unexpected step lookup")
    )
    with region:
        pass


def test_watch_deferred_dispatch_reports_region_and_end(tmp_path, monkeypatch) -> None:
    pair = _Pair(tmp_path)
    monkeypatch.setattr(request_watchdog, "execution_reporter", pair.reporter)

    def results():
        with StageProfiler("denoising_step_0", logger=None, metrics=None):
            yield "frame"

    try:
        dispatch = pair.reporter.watch_dispatch(lambda items: results())
        consume = pair.reporter.watch_dispatch(lambda items, outputs: list(outputs))
        items = [(None, [_req("r1")])]
        outputs = dispatch(items)
        assert consume(items, outputs) == ["frame"]
        messages = pair.messages()
        assert [m[1] for m in messages] == ["begin", "end", "begin", "step", "end"]
        assert messages[3] == [1, "step", "denoising_step_0"]
    finally:
        pair.close()


def _begin(watchdog, request_id, started, limit=10.0, replica=0) -> None:
    watchdog.handle([replica, "begin", request_id, "t2va", limit, started])


def test_queued_time_does_not_count() -> None:
    watchdog = RequestWatchdog([], lambda error: None)
    # three requests sent at t=0, executed back to back for 8 s each
    for index, request_id in enumerate(("a", "b", "c")):
        started = 8.0 * index
        _begin(watchdog, request_id, started)
        assert watchdog.overdue(started + 8.0) is None
        watchdog.handle([0, "end"])
    assert watchdog.overdue(1000.0) is None
    # a request executing past its limit is overdue, whatever else runs
    _begin(watchdog, "slow", 100.0, replica=0)
    _begin(watchdog, "other", 109.0, replica=1)
    watchdog.handle([0, "stage", "MiniMaxH3DenoisingStage"])
    watchdog.handle([0, "step", "denoising_step_4"])
    assert watchdog.overdue(110.0) is None
    overdue = watchdog.overdue(110.5)
    assert (overdue.request_id, overdue.stage, overdue.step) == (
        "slow",
        "MiniMaxH3DenoisingStage",
        "denoising_step_4",
    )
    # a stage resets the step; events of an idle replica are ignored
    watchdog.handle([0, "stage", "MiniMaxH3DecodingStage"])
    assert overdue.step is None
    watchdog.handle([2, "step", "denoising_step_1"])
    assert 2 not in watchdog._executing


_DUMPER = """
import sys, time
from types import SimpleNamespace
from sglang.multimodal_gen.runtime.managers.request_watchdog import install_stack_dump_handler

def stuck_in_step():
    print("ready", flush=True)
    time.sleep(60)

install_stack_dump_handler(SimpleNamespace(
    progress_endpoint="ipc:///tmp/x", request_time_limit=["1"], port=1))
stuck_in_step()
"""


def test_trip_turns_health_unhealthy_and_dumps_worker_stacks(monkeypatch) -> None:
    client = AsyncSchedulerClient()
    monkeypatch.setattr(http_server, "async_scheduler_client", client)
    monkeypatch.setattr(request_watchdog, "STACK_DUMP_GAP_S", 0.5)
    with subprocess.Popen(
        [sys.executable, "-c", _DUMPER],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ) as worker:
        try:
            assert worker.stdout.readline().strip() == "ready"
            watchdog = RequestWatchdog([psutil.Process(worker.pid)], client.mark_failed)
            _begin(watchdog, "r7", 0.0, limit=2.0)
            asyncio.run(watchdog.trip(watchdog.overdue(5.0), 5.0))
            assert worker.poll() is None
        finally:
            worker.kill()
        stderr = worker.stderr.read()
    assert "most recent call first" in stderr and "stuck_in_step" in stderr

    state = SimpleNamespace(
        server_warmup_done=asyncio.Event(),
        worker_processes=[],
        disagg_server=None,
        broker_task=None,
    )
    state.server_warmup_done.set()
    response = asyncio.run(
        http_server.health(SimpleNamespace(app=SimpleNamespace(state=state)))
    )
    assert response.status_code == 503
    assert json.loads(response.body) == {
        "status": "unhealthy",
        "error": "request r7 exceeded 2 s",
    }


def test_watchdog_trips_from_reported_executions(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(request_watchdog, "CHECK_INTERVAL_MS", 50)
    endpoint = f"ipc://{tmp_path}/watchdog"
    failures = []

    async def run() -> float:
        watchdog = RequestWatchdog([], failures.append)
        task = asyncio.create_task(watchdog.run(endpoint))
        context = zmq.Context()
        try:
            reporter = ExecutionReporter(context, endpoint, {"*": 0.3}, replica=0)
            # bound and connected
            await asyncio.sleep(0.3)
            # back-to-back requests under the limit each, over it together
            for request_id in ("q1", "q2", "q3"):
                with reporter.watch(_req(request_id)):
                    await asyncio.sleep(0.2)
            await asyncio.sleep(0.3)
            assert not failures and not task.done()
            started = time.monotonic()
            with reporter.watch(_req("hung")):
                await asyncio.wait_for(task, 5)
            return time.monotonic() - started
        finally:
            context.destroy(linger=0)

    elapsed = asyncio.run(run())
    assert failures == ["request hung exceeded 0.3 s"]
    assert 0.3 < elapsed < 1.0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
