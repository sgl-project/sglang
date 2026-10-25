"""Native reference events plus the actual virtual barrier; no performance claims."""

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace as NS

import pytest
from sglang_simulator.simulation.coordinator import (
    Cancel,
    Ready,
    Submit,
    decode_grant,
    encode_message,
)
from sglang_simulator.simulation.manager import StateManager
from sglang_simulator.simulation.sglang import dspark_runtime, scheduler
from sglang_simulator.simulation.sglang.dspark import STEP_ATTRIBUTE
from sglang_simulator.time_predictor.base import PredictorError
from test_dspark_reference import native_page_sizing
from test_dspark_reference_driver import install_native_hooks, selected
from test_dspark_reference_driver import run as run_fixture
from test_dspark_runtime import runtime as runtime_fixture
from test_scheduler_predictor_hooks import hooked_scheduler as scheduler_fixture
from test_simulation_coordinated_bridge import TokenizedGenerateReqInput
from test_simulation_coordinated_bridge import harness as harness_fixture

run = run_fixture
runtime = runtime_fixture
hooked_scheduler = scheduler_fixture
harness = harness_fixture
pytestmark = pytest.mark.usefixtures(native_page_sizing.__name__)


@pytest.mark.parametrize("sync", [False, True])
@pytest.mark.parametrize("cancel", [False, True])
def test_reference_fifo_timer_visible_commits_and_discarded_tail_drain(
    run, hooked_scheduler, harness, monkeypatch, sync, cancel
):
    target = install_native_hooks(run, hooked_scheduler)
    hooked_scheduler.hook.COORDINATOR = harness.bridge
    harness.bridge.scheduler = target
    harness.bridge._pending_work = lambda: scheduler._has_pending_dspark_work(
        target, hooked_scheduler.predictor
    )
    target.waiting_queue = []
    target.chunked_req = target._pending_chunked_abort_req = None
    target.process_pending_chunked_abort = lambda: None
    target.result_queue = deque()
    monkeypatch.setattr(
        dspark_runtime,
        "sampling_snapshot",
        lambda params: '{"ignore_eos":true,"is_normalized":true,"max_new_tokens":3}',
    )
    worker_call = target.after_forward

    def forward(batch, result):
        worker_call(batch, result)
        result.auxiliary_host_output = None

    target.after_forward = forward
    initial = selected(run)
    req = initial.reqs[0]
    req.origin_input_ids = [10] * 512
    req.cached_tokens = 0
    req.to_finish = None
    target.collect_inflight_reqs = lambda: [req]
    target.abort_request = lambda abort: setattr(req, "to_finish", NS(aborted=True))
    native_events = []

    def process(batch, result):
        prediction = getattr(result, STEP_ATTRIBUTE)
        commits = []
        for current, (_, count) in zip(
            batch.reqs, prediction.progress.committed_tokens, strict=True
        ):
            if current.finished() or current.to_finish is not None:
                current.done = True
                commits.append(None)
                continue
            prior = len(current.output_ids)
            current.output_ids.extend([1] * count)
            if prediction.local_group.forward_mode == "TARGET_VERIFY":
                current.kv.kv_committed_len += count
            current.done = len(current.output_ids) >= 3
            commits.append(NS(output_index=prior, token_ids=(1,) * count))
            native_events.append((prediction.step.forward_id, count))
        result.auxiliary_host_output.consume(batch, commits)

    target.after_process = process
    payload = TokenizedGenerateReqInput("a")
    payload.input_ids = list(req.origin_input_ids)
    payload.sampling_params.max_new_tokens = 3

    def driver(control):
        control.send(
            encode_message(Ready("native-reference", 0, [Submit("a")], 20_000_000))
        )
        harness.payloads.put(payload)
        assert decode_grant(control.recv()).now_ns == 20_000_000
        control.send(encode_message(Ready("native-reference", 1, [], 70_000_000)))
        assert decode_grant(control.recv()).now_ns == 70_000_000
        control.send(
            encode_message(
                Ready(
                    "native-reference",
                    2,
                    [Cancel("a")] if cancel else [],
                    finished=cancel,
                )
            )
        )
        terminal = decode_grant(control.recv())
        assert [(event.kind, event.at_ns) for event in terminal.terminals] == [
            (
                "CANCELLED" if cancel else "COMPLETED",
                70_000_000 if cancel else 90_000_000,
            )
        ]
        control.send(encode_message(Ready("native-reference", 3, [], finished=True)))
        return decode_grant(control.recv())

    def launch(batch):
        result = target.run_batch(batch)
        target.result_queue.append((batch, result))
        return result

    def retire():
        batch, result = target.result_queue.popleft()
        target.process_batch_result(batch, result)

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(harness.run_driver, driver)
        assert [item.rid for item in harness.receive()] == ["a"]
        first = launch(initial)
        current = selected(run, "DECODE", req=req, sync=sync)
        if sync:
            retire()
        second = launch(current)
        if not sync:
            retire()
        assert harness.bridge.clock.now_ns == 40_000_000
        assert getattr(first, scheduler._DSPARK_ACCOUNTED) is True
        assert getattr(second, scheduler._DSPARK_ACCOUNTED) is False
        successor = selected(run, "DECODE", req=req)
        tail = launch(successor)
        retire()
        assert harness.bridge.clock.now_ns == 90_000_000
        assert harness.receive() == []
        assert not future.done() and harness.bridge._has_native_work()
        queued = target.result_queue
        target.result_queue = deque()
        setattr(target, scheduler._DSPARK_PENDING_RESULTS, 0)
        assert harness.bridge._has_native_work(), (
            "reference pending tail must hold FINISHED"
        )
        setattr(target, scheduler._DSPARK_PENDING_RESULTS, 1)
        target.result_queue = queued
        retire()
        assert harness.bridge.clock.now_ns == 140_000_000
        assert harness.receive() == []
        final = future.result(timeout=3)
    assert final.status == "FINISHED" and final.now_ns == 140_000_000
    assert getattr(target, scheduler._DSPARK_PENDING_RESULTS) == 0
    assert getattr(tail, scheduler._DSPARK_ACCOUNTED) is True
    assert run.driver.adapter.ledger.pending == ()
    stats = scheduler.request_stats_manager.get_req_stats("a")
    assert len(stats.gen_token_latencies) == (1 if cancel else 2)
    assert len(req.output_ids) == (1 if cancel else 3)
    assert native_events == ([(0, 1)] if cancel else [(0, 1), (1, 2)])
    now = harness.bridge.clock.now_ns
    with pytest.raises(PredictorError, match="already consumed"):
        target.process_batch_result(successor, tail)
    assert harness.bridge.clock.now_ns == now


@pytest.mark.parametrize("change", ["missing", "owner", "order"])
def test_bad_coordinated_accounting_fails_before_clock_or_native_result(
    hooked_scheduler, change
):
    from test_scheduler_predictor_hooks import dspark_prediction, launch_dspark

    test = hooked_scheduler
    prediction = dspark_prediction(
        0, prefixes=(("a", 512), ("b", 1024)), seconds=0.05, wall_seconds=0.3
    )
    batch, result = launch_dspark(test, prediction)
    if change == "missing":
        delattr(result, scheduler._DSPARK_ACCOUNTING)
    elif change == "owner":
        setattr(test.target, scheduler._DSPARK_ACCOUNTING_OWNER, object())
    else:
        batch.reqs.reverse()
    test.hook.COORDINATOR = NS(
        process_batch_result=lambda *args, **kwargs: pytest.fail(
            "invalid result reached clock"
        )
    )
    with pytest.raises(PredictorError):
        test.target.process_batch_result(batch, result)
    assert StateManager.get_global_clock() == 100
