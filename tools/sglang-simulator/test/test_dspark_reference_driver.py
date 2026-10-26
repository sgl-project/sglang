"""Production event wiring on CPU protocols; no qualified timing/progress rows."""

import ast
import asyncio
import json
from collections import deque
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
from sglang_simulator.compat import (
    apply_simulator_server_args,
    validate_simulator_server_args,
)
from sglang_simulator.simulation.sglang.dspark import STEP_ATTRIBUTE
from sglang_simulator.simulation.sglang.dspark_reference import ReferenceCPUView
from sglang_simulator.simulation.sglang.dspark_reference_driver import (
    DSparkReferenceDriver,
)
from sglang_simulator.time_predictor.base import PredictorError
from test_dspark_reference import native_page_sizing
from test_dspark_runtime import batch, forward
from test_dspark_runtime import runtime as runtime_fixture
from test_scheduler_predictor_hooks import hooked_scheduler as scheduler_fixture
from test_scheduler_predictor_hooks import (
    mock_profile_output,
)

runtime = runtime_fixture
hooked_scheduler = scheduler_fixture

pytestmark = pytest.mark.usefixtures(native_page_sizing.__name__)


@pytest.fixture
def run(runtime):
    runtime.reference_driver = DSparkReferenceDriver(runtime)
    driver = runtime.reference_driver
    seen = []

    def resolve(**kwargs):
        assert type(kwargs["reference_view"]) is ReferenceCPUView
        seen.append(kwargs)
        return {"path": "CPU fixture"}

    runtime.execution_resolver = resolve
    runtime.session.cost_model.native_source_policy = NS(
        reference_needs_cpu_seq_lens=False
    )
    return NS(
        runtime=runtime,
        driver=driver,
        seen=seen,
        scheduler=NS(enable_overlap=True, future_map=NS(needs_cpu_seq_lens=False)),
    )


def selected(
    run,
    mode="EXTEND",
    *,
    prefix=0,
    extend=512,
    chunks=0,
    req=None,
    rid="a",
    relay=True,
    sync=False,
):
    current = batch(mode, prefix=prefix, extend=extend, chunks=chunks)
    if req is not None:
        current.reqs = [req]
        req.inflight_middle_chunks = chunks
    req = current.reqs[0]
    if not hasattr(req, "done"):
        req.rid = rid
    current.chunked_req = req if chunks else None
    if not hasattr(req, "done"):
        req.done = False
        req.finished = lambda: req.done
        req.is_retracted = False
    current.token_to_kv_pool_allocator = NS(page_size=256)
    current.req_pool_indices = [1]
    current.spec_info = None
    if mode == "DECODE":
        row = run.driver.adapter.ledger.request_for(req.rid)
        req.kv.kv_committed_len = row.kv_committed_len
        current.seq_lens = current.seq_lens_cpu = [row.kv_committed_len]
        from sglang_simulator.simulation.sglang.dspark_reference import (
            _decode_alloc_lens,
        )

        _, nxt, _ = _decode_alloc_lens((row,), reserve=12, page_size=256)
        current.spec_info = NS(
            future_indices=[1] if relay else None, nxt_kv_lens_cpu=nxt
        )
    else:
        req.kv.kv_committed_len = prefix + extend
    run.driver.selected(run.scheduler, current)
    run.driver.overlap_decision(current, sync)
    return current


def launch(run, current, publish=None):
    selection = run.driver.selection
    if selection.future_indices_ready:
        published = selection.allocation.prepared.published_seq_lens
        current.seq_lens = list(published)
        current.seq_lens_cpu = (
            list(published) if selection.simulator_needs_cpu_seq_lens else None
        )
    return forward(run.runtime, current, publish)


def process(run, current, result, *, finish=False):
    run.driver.before_result(current, result)
    prediction = getattr(result, STEP_ATTRIBUTE)
    for req, (_, count) in zip(
        current.reqs, prediction.progress.committed_tokens, strict=True
    ):
        if req.finished():
            continue
        req.output_ids.extend([1] * count)
        if prediction.local_group.forward_mode == "TARGET_VERIFY":
            req.kv.kv_committed_len += count
        req.done = finish
        if finish and prediction.local_group.forward_mode == "TARGET_VERIFY":
            run.runtime.note_request_finished(rid=req.rid, natural_stop=False)
    run.driver.after_result()


@pytest.mark.parametrize("rank", [0, 1])
def test_native_rank_binds_after_scheduler_before_work(run, hooked_scheduler, rank):
    # The overlap-loop hook owns the Scheduler before its first empty poll.
    hooked_scheduler.predictor.dspark_runtime = run.runtime
    run.scheduler = hooked_scheduler.target
    run.scheduler.enable_overlap = True
    assert run.scheduler.event_loop_overlap() == "native overlap"
    run.driver.bind_native_dp_rank(rank)
    bound = run.driver.adapter
    assert run.driver.owner is run.scheduler
    assert bound.ledger.local_attention_dp_rank == rank
    run.driver.bind_native_dp_rank(rank)
    assert run.driver.adapter is bound
    run.driver.drained_reset()
    run.driver.bind_native_dp_rank(rank)
    with pytest.raises(PredictorError, match="local DP owner changed"):
        run.driver.bind_native_dp_rank(1 - rank)
    with pytest.raises(PredictorError, match="scheduler owner"):
        run.driver.bind(NS(enable_overlap=True))
    assert run.driver.owner is run.scheduler


@pytest.mark.parametrize("rank", [0, 1])
def test_first_explicit_native_rank_is_sealed_even_before_scheduler_bind(run, rank):
    run.driver.bind_native_dp_rank(rank)
    assert run.driver.owner is None
    with pytest.raises(PredictorError, match="local DP owner changed"):
        run.driver.bind_native_dp_rank(1 - rank)
    assert run.driver.adapter.ledger.local_attention_dp_rank == rank


@pytest.mark.parametrize("state", ["selected", "published", "processing", "drained"])
@pytest.mark.parametrize("rank", [0, 1])
def test_late_native_binding_cannot_reinterpret_existing_reference_work(
    run, state, rank
):
    current = selected(run)
    if state != "selected":
        result = launch(run, current)
        if state == "processing":
            run.driver.before_result(current, result)
        elif state == "drained":
            process(run, current, result, finish=True)
            assert run.driver.adapter.ledger.requests == ()
            assert run.driver.adapter.ledger.last_forward_id == 0
    original = run.driver.adapter
    with pytest.raises(PredictorError, match="local DP owner changed"):
        run.driver.bind_native_dp_rank(rank)
    assert run.driver.adapter is original


@pytest.mark.parametrize("sync", [False, True])
def test_real_event_activation_keeps_allocation_and_worker_output_snapshots(run, sync):
    prefill = selected(run, prefix=1024, extend=64)
    first = launch(run, prefill)
    second_batch = selected(run, "DECODE", req=prefill.reqs[0], sync=sync)
    seed = run.driver.selection.allocation.prepared.seed_seq_lens_cpu
    if sync:
        process(run, prefill, first)
    second = launch(run, second_batch)
    assert run.seen[-1]["reference_view"].seq_lens_cpu is None
    assert seed == (1088,)
    assert getattr(second, STEP_ATTRIBUTE).step.requests[0].output_tokens_before == int(
        sync
    )
    assert getattr(second, STEP_ATTRIBUTE).step.requests[0].draft_history_tokens == 64
    if not sync:
        assert len(run.driver.adapter.ledger.pending) == 2
        process(run, prefill, first)
    process(run, second_batch, second)
    third_batch = selected(run, "DECODE", req=prefill.reqs[0])
    third = launch(run, third_batch)
    process(run, third_batch, third)
    assert run.runtime._forward_id == 3
    assert run.driver.adapter.ledger.pending == ()
    assert run.driver.adapter.ledger.request_for("a").processed_output_tokens == 5


@pytest.mark.parametrize(
    "relay,needs_cpu", [(False, False), (True, False), (True, True)]
)
def test_native_absent_relay_and_present_without_cpu_mirror_remain_distinct(
    run, relay, needs_cpu
):
    first_batch = selected(run)
    first = launch(run, first_batch)
    process(run, first_batch, first)
    run.scheduler.future_map.needs_cpu_seq_lens = needs_cpu
    run.runtime.session.cost_model.native_source_policy.reference_needs_cpu_seq_lens = (
        needs_cpu
    )
    current = selected(run, "DECODE", req=first_batch.reqs[0], relay=relay)
    launch(run, current)
    view = run.seen[-1]["reference_view"]
    assert view.seq_lens_cpu == ((512,) if not relay or needs_cpu else None)
    assert view.nxt_kv_lens_cpu == (768,)


def test_cpu_engine_backend_cannot_supply_missing_physical_reference_policy(run):
    first_batch = selected(run)
    first = launch(run, first_batch)
    process(run, first_batch, first)
    run.runtime.session.cost_model.native_source_policy = NS()
    with pytest.raises(
        PredictorError, match="reference backend CPU mirror policy"
    ) as failure:
        selected(run, "DECODE", req=first_batch.reqs[0])
    assert failure.value.code == "data_unavailable"
    assert failure.value.details["missing_source_facts"] == [
        "reference_backend_cpu_mirror_policy"
    ]


def test_cpu_backend_mirror_stays_distinct_from_verified_reference_mirror(run):
    first_batch = selected(run)
    first = launch(run, first_batch)
    process(run, first_batch, first)
    run.scheduler.future_map.needs_cpu_seq_lens = True
    current = selected(run, "DECODE", req=first_batch.reqs[0])
    launch(run, current)
    assert current.seq_lens_cpu == [512]
    assert run.seen[-1]["reference_view"].seq_lens_cpu is None


def test_cached_chunks_final_bonus_and_prefill_finish_have_actual_result_event(run):
    first_batch = selected(run, prefix=1024, extend=32, chunks=1)
    first = launch(run, first_batch)
    assert getattr(first, STEP_ATTRIBUTE).progress.committed_tokens == (("a", 0),)
    second_batch = selected(
        run, prefix=1056, extend=32, chunks=1, req=first_batch.reqs[0], sync=True
    )
    process(run, first_batch, first)
    second = launch(run, second_batch)
    process(run, second_batch, second)
    final_batch = selected(run, prefix=1088, extend=32, req=first_batch.reqs[0])
    final = launch(run, final_batch)
    assert getattr(final, STEP_ATTRIBUTE).progress.committed_tokens == (("a", 1),)
    process(run, final_batch, final, finish=True)
    assert run.driver.adapter.ledger.requests == ()
    with pytest.raises(PredictorError, match="prior injection"):
        run.runtime.session.history_for("a")


def test_final_chunk_identity_survives_an_unsettled_middle_chunk_counter(run):
    middle = selected(run, extend=128, chunks=1)
    result = launch(run, middle)
    final = selected(run, prefix=128, extend=64, req=middle.reqs[0])
    # Native final selection has chunked_req=None while an earlier middle
    # chunk can still own the shared request's inflight counter.
    final.reqs[0].inflight_middle_chunks = 1
    output = launch(run, final)
    assert getattr(output, STEP_ATTRIBUTE).progress.committed_tokens == (("a", 1),)
    process(run, middle, result)
    process(run, final, output)


def test_worker_device_lengths_cannot_hide_behind_a_known_absent_cpu_mirror(run):
    initial = selected(run)
    first = launch(run, initial)
    process(run, initial, first)
    current = selected(run, "DECODE", req=initial.reqs[0])
    current.seq_lens = [900]
    current.seq_lens_cpu = None
    with pytest.raises(PredictorError, match="device lengths"):
        forward(run.runtime, current)
    assert run.runtime._forward_id == 1


def test_natural_decode_finish_then_pending_successor_skip_and_queued_cancel(run):
    initial = selected(run)
    first = launch(run, initial)
    process(run, initial, first)
    previous = selected(run, "DECODE", req=initial.reqs[0])
    result = launch(run, previous)
    successor = selected(run, "DECODE", req=initial.reqs[0])
    tail = launch(run, successor)
    process(run, previous, result, finish=True)
    assert run.driver.adapter.retiring_request_ids == ("a",)
    assert run.runtime.session.history_for("a").target_prefix_len == 516
    process(run, successor, tail)
    assert run.driver.adapter.ledger.requests == ()
    run.runtime.note_request_finished(rid="never-admitted", natural_stop=False)
    assert run.driver.adapter.ledger.requests == ()


@pytest.mark.parametrize("failure", ["cost", "build", "callback", "stale"])
def test_failed_forward_never_installs_reference_session_buffer_or_fid(run, failure):
    current = selected(run)
    base = run.driver.adapter

    def fail(*args, **kwargs):
        raise RuntimeError("injected failure")

    publish = None
    if failure == "cost":
        run.runtime.session.cost_model.predict_step = fail
    elif failure == "build":
        run.runtime.results.build = fail
    elif failure == "callback":
        publish = fail
    else:
        build = run.runtime.results.build

        def stale(**kwargs):
            result = build(**kwargs)
            run.driver.adapter = replace(base)
            return result

        run.runtime.results.build = stale
    with pytest.raises((RuntimeError, PredictorError)):
        launch(run, current, publish)
    assert run.driver.adapter.ledger == base.ledger
    assert run.runtime.session._history == {}
    assert run.runtime._forward_id == run.runtime._draft_buffer_capacity == 0
    if failure == "callback":
        assert run.driver.failure["event"] == "publication/session commit"
        with pytest.raises(PredictorError, match="restart the run"):
            launch(run, current)


def test_publication_cancellation_after_external_effect_latches_and_cannot_retry(run):
    current = selected(run)
    base, selection = run.driver.adapter, run.driver.selection
    effects = []
    cancellation = asyncio.CancelledError("publication was already visible")

    def publish(lengths):
        effects.append(list(lengths))
        raise cancellation

    with pytest.raises(asyncio.CancelledError) as caught:
        launch(run, current, publish)
    assert caught.value is cancellation and effects == [[512]]
    assert run.driver.failure["event"] == "publication/session commit"
    assert run.driver.adapter is base and run.driver.selection is selection
    assert run.runtime.session._history == {}
    assert run.runtime._forward_id == run.runtime._draft_buffer_capacity == 0
    with pytest.raises(PredictorError, match="restart the run"):
        launch(run, current, publish)
    assert effects == [[512]] and run.driver.adapter.ledger.pending == ()


def test_cancel_pending_drain_and_reset_keep_buffer_and_monotonic_fid(run):
    current = selected(run)
    result = launch(run, current)
    process(run, current, result)
    current = selected(run, "DECODE", req=current.reqs[0])
    result = launch(run, current)
    with pytest.raises(PredictorError, match="drain"):
        run.runtime.clear_cache_pool()
    current.reqs[0].done = True
    process(run, current, result)
    run.runtime.clear_cache_pool()
    assert run.runtime.session._history == {}
    assert run.driver.adapter.ledger.requests == ()
    assert run.runtime._draft_buffer_capacity == 1 and run.runtime._forward_id == 2


def test_native_retraction_with_outputs_fails_explicitly(run):
    current = selected(run)
    result = launch(run, current)
    current.reqs[0].is_retracted = True
    with pytest.raises(PredictorError, match="retraction"):
        process(run, current, result)


def test_dynamic_join_reorder_filter_keep_exact_rids_and_prior_fifo(run):
    a = selected(run, prefix=1024, extend=64)
    first = launch(run, a)
    b = selected(run, rid="b", prefix=2048, extend=128)
    second = launch(run, b)
    assert [
        result.step.requests[0].rid for result in run.driver.adapter.ledger.pending
    ] == ["a", "b"]
    process(run, a, first)
    process(run, b, second)

    def select_members(requests):
        from sglang_simulator.simulation.sglang.dspark_reference import (
            _decode_alloc_lens,
        )

        rows = tuple(run.driver.adapter.ledger.request_for(req.rid) for req in requests)
        _, nxt, _ = _decode_alloc_lens(rows, reserve=12, page_size=256)
        current = batch("DECODE")
        current.reqs = requests
        current.seq_lens = current.seq_lens_cpu = [row.kv_committed_len for row in rows]
        current.req_pool_indices = list(range(1, len(rows) + 1))
        current.spec_info = NS(
            future_indices=current.req_pool_indices, nxt_kv_lens_cpu=nxt
        )
        current.token_to_kv_pool_allocator = NS(page_size=256)
        run.driver.selected(run.scheduler, current)
        run.driver.overlap_decision(current, False)
        return current

    combined = select_members([b.reqs[0], a.reqs[0]])
    third = launch(run, combined)
    assert tuple(request.rid for request in run.seen[-1]["group"].requests) == (
        "b",
        "a",
    )
    assert run.seen[-1]["reference_view"].nxt_kv_lens_cpu == (2304, 1280)
    process(run, combined, third)
    run.runtime.note_request_finished(rid="b", natural_stop=False)
    filtered = select_members([a.reqs[0]])
    fourth = launch(run, filtered)
    assert tuple(request.rid for request in run.seen[-1]["group"].requests) == ("a",)
    process(run, filtered, fourth)
    assert tuple(row.rid for row in run.driver.adapter.ledger.requests) == ("a",)


def test_installed_scheduler_and_worker_hooks_activate_driver_and_consume_timing_once(
    run, hooked_scheduler, monkeypatch
):
    from sglang_simulator.simulation.sglang import scheduler
    from sglang_simulator.simulation.sglang.dspark_worker import C_DSparkWorkerHook

    target = hooked_scheduler.target
    hooked_scheduler.predictor.supports_dspark = True
    hooked_scheduler.predictor.dspark_runtime = run.runtime
    target.enable_overlap = True
    target.future_map = run.scheduler.future_map
    assert target.event_loop_overlap() == "native overlap"

    def add_emulation_wall(function, seconds):
        def call(*args, **kwargs):
            result = function(*args, **kwargs)
            hooked_scheduler.clock.now += seconds
            return result

        return call

    monkeypatch.setattr(
        run.driver, "selected", add_emulation_wall(run.driver.selected, 2)
    )
    monkeypatch.setattr(
        run.driver, "before_result", add_emulation_wall(run.driver.before_result, 3)
    )
    monkeypatch.setattr(
        run.driver, "after_result", add_emulation_wall(run.driver.after_result, 5)
    )
    monkeypatch.setattr(
        run.runtime.results, "build", add_emulation_wall(run.runtime.results.build, 4)
    )

    class Worker:
        pass

    C_DSparkWorkerHook.hook(Worker)
    worker = Worker.__new__(Worker)
    worker.query_token_num = 5
    worker.verify_num_draft_tokens = 6
    current = batch()
    current.chunked_req = None
    current.spec_info = None
    current.token_to_kv_pool_allocator = NS(page_size=256)
    current.reqs[0].done = False
    current.reqs[0].kv.kv_committed_len = 512
    current.reqs[0].finished = lambda: current.reqs[0].done
    current.forward_mode.is_extend = lambda: True
    current.forward_mode.is_decode = lambda: False
    target.plan = NS(batch_to_run=current, running_batch=None)
    target.disable_overlap = False
    assert target.get_next_batch_to_run().batch_to_run is current
    target.is_disable_overlap_for_batch(current)
    publications = []

    def worker_call(native_batch, result):
        vars(result).update(
            vars(
                worker.forward_batch_generation(
                    native_batch, on_publish=publications.append
                )
            )
        )

    target.after_forward = worker_call
    target.after_process = lambda batch, result: batch.reqs[0].output_ids.append(1)
    result = target.run_batch(current)
    assert len(publications) == 1 and len(run.seen) == 1
    target.process_batch_result(current, result)
    assert run.driver.adapter.ledger.pending == ()
    assert getattr(target, scheduler._DSPARK_PENDING_RESULTS) == 0
    assert getattr(result, scheduler._DSPARK_ACCOUNTED) is True
    assert hooked_scheduler.hook.ITERATION_STATS[-1]["cpu_overhead"] == pytest.approx(
        0.21
    )
    assert getattr(target, scheduler._DSPARK_WALL_CREDIT) == 0.0
    with pytest.raises(PredictorError, match="already consumed"):
        target.process_batch_result(current, result)
    target.on_idle()
    assert run.runtime.session.history_for("a").target_prefix_len == 512


def install_native_hooks(run, hooked_scheduler):
    from sglang_simulator.simulation.sglang.dspark_worker import C_DSparkWorkerHook

    target = hooked_scheduler.target
    hooked_scheduler.predictor.supports_dspark = True
    hooked_scheduler.predictor.dspark_runtime = run.runtime
    target.enable_overlap = True
    target.future_map = run.scheduler.future_map
    run.scheduler = target

    class Worker:
        pass

    C_DSparkWorkerHook.hook(Worker)
    worker = Worker.__new__(Worker)
    worker.query_token_num = 5
    worker.verify_num_draft_tokens = 6

    def worker_call(native_batch, result):
        selection = run.driver.selection
        if selection.future_indices_ready:
            published = selection.allocation.prepared.published_seq_lens
            native_batch.seq_lens = list(published)
            native_batch.seq_lens_cpu = (
                list(published) if selection.simulator_needs_cpu_seq_lens else None
            )
        vars(result).update(vars(worker.forward_batch_generation(native_batch)))

    target.after_forward = worker_call
    return target


def test_native_result_cancellation_after_response_latches_and_cannot_retry(
    run, hooked_scheduler
):
    from sglang_simulator.simulation.sglang import scheduler

    target = install_native_hooks(run, hooked_scheduler)
    current = selected(run)
    result = target.run_batch(current)
    effects = []
    cancellation = asyncio.CancelledError("response was already visible")

    def process_then_cancel(batch, result):
        batch.reqs[0].output_ids.append(1)
        effects.append(batch.reqs[0].rid)
        raise cancellation

    target.after_process = process_then_cancel
    with pytest.raises(asyncio.CancelledError) as caught:
        target.process_batch_result(current, result)
    assert caught.value is cancellation and effects == ["a"]
    assert run.driver.failure["event"] == "native result processing"
    assert getattr(result, scheduler._DSPARK_ACCOUNTED) is False
    assert getattr(target, scheduler._DSPARK_PENDING_RESULTS) == 1
    with pytest.raises(PredictorError, match="restart the run"):
        target.process_batch_result(current, result)
    assert effects == ["a"]
    assert hooked_scheduler.hook.ITERATION_STATS == []


@pytest.mark.parametrize("failure", [None, "pending", "save"])
def test_profile_reset_clears_old_idle_credit_only_after_success(
    run, hooked_scheduler, monkeypatch, tmp_path, failure
):
    from sglang_simulator.simulation.manager.state import StateManager
    from sglang_simulator.simulation.sglang import scheduler

    target = install_native_hooks(run, hooked_scheduler)
    current = selected(run)
    process(run, current, launch(run, current))
    current = selected(run, "DECODE", req=current.reqs[0])
    process(run, current, launch(run, current))
    assert run.runtime._draft_buffer_capacity > 0
    target.plan = NS(batch_to_run=None, running_batch=None)
    original_selected = run.driver.selected

    def idle_selection(*args, **kwargs):
        result = original_selected(*args, **kwargs)
        hooked_scheduler.clock.now += 2.0
        return result

    monkeypatch.setattr(run.driver, "selected", idle_selection)
    target.get_next_batch_to_run()
    run.driver.callback_wall_seconds = 3.0
    target.on_idle()
    assert getattr(target, scheduler._DSPARK_WALL_CREDIT) == 2.0
    assert run.driver.callback_wall_seconds == 3.0
    mock_profile_output(monkeypatch)
    monkeypatch.setattr(
        scheduler.Envs, "output_dir", classmethod(lambda cls: str(tmp_path))
    )
    if failure == "pending":
        pending = selected(run, "DECODE", req=current.reqs[0])
        target.run_batch(pending)
        assert getattr(target, scheduler._DSPARK_PENDING_RESULTS) == 1
    elif failure == "save":
        stats = scheduler.request_stats_manager.get_req_stats("a")
        stats.created_time = stats.queue_start = stats.queue_end = 100
        stats.last_event_time = 101
        stats.gen_token_latencies = [1]

        def fail_open(*args, **kwargs):
            raise OSError("disk full")

        monkeypatch.setattr(scheduler, "open", fail_open, raising=False)
    base = run.driver.adapter
    history = dict(run.runtime.session._history)
    capacity, fid = run.runtime._draft_buffer_capacity, run.runtime._forward_id
    clock_before = StateManager.get_global_clock()
    profile = next(iter(target._request_dispatcher._mapping.values()))
    result = profile(hooked_scheduler.profile_req)
    assert result.success is (failure is None)
    assert getattr(target, scheduler._DSPARK_WALL_CREDIT) == (
        0.0 if failure is None else 2.0
    )
    assert run.driver.callback_wall_seconds == (0.0 if failure is None else 3.0)
    if failure is not None:
        assert StateManager.get_global_clock() == clock_before
    assert run.driver.adapter is base and run.driver.owner is target
    assert run.runtime.session._history == history
    assert (
        run.runtime._draft_buffer_capacity == capacity
        and run.runtime._forward_id == fid
    )


def test_finished_successor_skips_response_metrics_but_accounts_launched_cost_once(
    run, hooked_scheduler, monkeypatch
):
    from sglang_simulator.simulation.manager.state import StateManager
    from sglang_simulator.simulation.sglang import dspark_runtime, scheduler

    target = install_native_hooks(run, hooked_scheduler)
    monkeypatch.setattr(
        dspark_runtime,
        "sampling_snapshot",
        lambda params: '{"ignore_eos":true,"is_normalized":true,"max_new_tokens":3}',
    )

    def settle(batch, result):
        prediction = getattr(result, STEP_ATTRIBUTE)
        for req, (_, count) in zip(
            batch.reqs, prediction.progress.committed_tokens, strict=True
        ):
            if req.finished():
                continue
            req.output_ids.extend([1] * count)
            if prediction.local_group.forward_mode == "TARGET_VERIFY":
                req.kv.kv_committed_len += count
            req.done = len(req.output_ids) >= 3
            if req.done:
                run.runtime.note_request_finished(rid=req.rid, natural_stop=False)

    target.after_process = settle
    initial = selected(run)
    first = target.run_batch(initial)
    target.process_batch_result(initial, first)
    previous = selected(run, "DECODE", req=initial.reqs[0])
    result = target.run_batch(previous)
    successor = selected(run, "DECODE", req=initial.reqs[0])
    tail = target.run_batch(successor)
    target.process_batch_result(previous, result)
    req = initial.reqs[0]
    assert req.finished() and len(req.output_ids) == 3
    stats = scheduler.request_stats_manager.get_req_stats(req.rid)
    events_before = tuple(stats.gen_token_latencies)
    event_time_before = stats.last_event_time
    assert len(events_before) == 2
    clock_before = StateManager.get_global_clock()
    tail_seconds = getattr(tail, STEP_ATTRIBUTE).timing.seconds
    target.process_batch_result(successor, tail)
    assert len(req.output_ids) == 3
    assert tuple(stats.gen_token_latencies) == events_before
    assert stats.last_event_time == event_time_before
    assert StateManager.get_global_clock() >= clock_before + tail_seconds
    assert len(hooked_scheduler.hook.ITERATION_STATS) == 3
    assert hooked_scheduler.hook.ITERATION_STATS[-1]["forward_latency"] == tail_seconds
    assert getattr(tail, scheduler._DSPARK_ACCOUNTED) is True
    assert getattr(target, scheduler._DSPARK_PENDING_RESULTS) == 0
    assert run.driver.adapter.ledger.pending == ()
    clock_after = StateManager.get_global_clock()
    with pytest.raises(PredictorError, match="already consumed"):
        target.process_batch_result(successor, tail)
    assert StateManager.get_global_clock() == clock_after


def test_native_abort_release_retires_admitted_chunks_without_a_later_result(
    run, hooked_scheduler
):
    hooked_scheduler.predictor.dspark_runtime = run.runtime
    current = selected(run, chunks=1)
    result = launch(run, current)
    process(run, current, result)
    hooked_scheduler.target._release_aborted_request(current.reqs[0])
    assert run.driver.adapter.ledger.requests == ()
    with pytest.raises(PredictorError, match="prior injection"):
        run.runtime.session.history_for("a")
    hooked_scheduler.target._release_aborted_request(NS(rid="never-admitted"))
    assert run.driver.adapter.ledger.requests == ()


@pytest.mark.parametrize("native", [False, True])
def test_launch_preparation_and_validation_enable_overlap_only_for_native_regions(
    tmp_path, monkeypatch, native
):
    config = tmp_path / "sim.json"
    predictor = {"name": "infercast_dspark", "proposal_profile_id": "a" * 64}
    if native:
        predictor = {"name": "infercast_dspark", "native_segment_profile_id": "b" * 64}
    config.write_text(json.dumps({"predictor": predictor}), encoding="utf-8")
    monkeypatch.setenv("SGLANG_SIMULATOR_CONFIG_PATH", "")
    monkeypatch.delenv("SGLANG_SIMULATOR_COORDINATOR", raising=False)
    args = NS(sim_config_path=str(config))
    apply_simulator_server_args(args)
    assert args.disable_overlap_schedule is (not native)
    validate_simulator_server_args(args)


def source_overlap_loop():
    """Execute the pinned native loop body, avoiding optional engine imports."""
    source = (
        Path(__file__).resolve().parents[3] / "python/sglang/srt/managers/scheduler.py"
    )
    tree = ast.parse(source.read_text(encoding="utf-8"))
    owner = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Scheduler"
    )
    method = next(
        node
        for node in owner.body
        if isinstance(node, ast.FunctionDef) and node.name == "event_loop_overlap"
    )
    method.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    namespace = {
        "deque": deque,
        "envs": NS(SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY=NS(get=lambda: False)),
    }
    exec(compile(module, str(source), "exec"), namespace)  # noqa: S102 - pinned owned source, no external input
    return namespace[method.name]


@pytest.mark.parametrize("sync", [False, True])
def test_original_overlap_loop_preserves_selection_sync_launch_and_fifo_order(sync):
    events = []
    batches = iter(
        [
            NS(rid="previous", copy=lambda: "previous"),
            NS(rid="current", copy=lambda: "current"),
            None,
        ]
    )
    target = NS(
        gracefully_exit=False,
        _engine_paused=False,
        running_batch=None,
        last_batch=None,
        enable_unified_memory=False,
        is_generation=False,
        ingest_requests=lambda: None,
        is_disable_overlap_for_batch=lambda batch, **kwargs: (
            sync and batch is not None and target.last_batch is not None
        ),
        _apply_war_barrier=lambda: None,
        on_idle=lambda: setattr(target, "gracefully_exit", True),
    )

    def select(**kwargs):
        current = next(batches, None)
        events.append(("select", None if current is None else current.rid))
        return NS(running_batch=None, batch_to_run=current)

    target.get_next_batch_to_run = select
    target.run_batch = lambda batch: events.append(("launch", batch.rid)) or batch.rid
    target.process_batch_result = lambda batch, result: events.append(
        ("process", result)
    )
    source_overlap_loop()(target)
    selected_current = events.index(("select", "current"))
    launched_current = events.index(("launch", "current"))
    processed_previous = events.index(("process", "previous"))
    assert selected_current < processed_previous
    assert (processed_previous < launched_current) is sync
