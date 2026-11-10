import json

import pytest
from test_simulation_sglang_serving import (
    SIM_CONFIGS,
    SGLangServingRunner,
    assert_decode_metrics,
)

REQUEST_RATE = 1
SEED = 123
ARRIVAL_TOLERANCE_S = 0.05
DURATION_TOLERANCE = 0.05
LATENCY_TOLERANCE = 0.25
RELATIVE_TOLERANCES = {
    "duration": DURATION_TOLERANCE,
    "request_throughput": DURATION_TOLERANCE,
    "input_throughput": DURATION_TOLERANCE,
    "output_throughput": DURATION_TOLERANCE,
    "mean_e2e_latency_ms": LATENCY_TOLERANCE,
    "mean_ttft_ms": LATENCY_TOLERANCE,
    "mean_tpot_ms": LATENCY_TOLERANCE,
    "mean_itl_ms": LATENCY_TOLERANCE,
}


def _relative_error(actual, expected):
    return abs(actual - expected) / abs(expected)


def _run_mode(mode, tmp_path):
    case_dir = tmp_path / mode
    case_dir.mkdir()
    runner = SGLangServingRunner(SIM_CONFIGS["aisimulate_sol"], case_dir, mode=mode)
    try:
        metrics = runner.benchmark(
            case_dir / "benchmark.json", request_rate=REQUEST_RATE, seed=SEED
        )
    finally:
        runner.shutdown()

    requests = [
        json.loads(line)
        for line in (runner.output_dir / "request.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    requests.sort(key=lambda request: request["created_time"])
    return metrics, requests


def test_request_rate_offline_matches_blocking(tmp_path):
    offline_metrics, offline_requests = _run_mode("offline", tmp_path)
    blocking_metrics, blocking_requests = _run_mode("blocking", tmp_path)

    for metrics in (offline_metrics, blocking_metrics):
        assert_decode_metrics(metrics)

    assert len(offline_requests) == len(blocking_requests) == 3

    offline_arrivals = [request["created_time"] for request in offline_requests]
    blocking_arrivals = [request["created_time"] for request in blocking_requests]
    assert offline_arrivals[1] > 0.5
    assert blocking_arrivals[1] > 0.5
    assert offline_arrivals == pytest.approx(blocking_arrivals, abs=ARRIVAL_TOLERANCE_S)
    assert (
        offline_metrics["max_concurrent_requests"]
        == blocking_metrics["max_concurrent_requests"]
        == 1
    )

    for key in ("completed", "total_input", "total_output"):
        assert offline_metrics[key] == blocking_metrics[key]

    for key, tolerance in RELATIVE_TOLERANCES.items():
        error = _relative_error(offline_metrics[key], blocking_metrics[key])
        assert error <= tolerance, (
            key,
            offline_metrics[key],
            blocking_metrics[key],
            error,
            tolerance,
        )


def _run_offline_iteration(target, batch, clock, run_cost, process_cost):
    """Advance the virtual clock through one run_batch + process_batch_result pair.

    run_batch advances the clock by run_cost; process_batch_result by process_cost.
    The predictor's own cost is applied inside the predictor mock.
    """
    target.run_batch(None, batch)
    target.process_batch_result(None, batch)


def _build_sim_target(hook, clock, predict_fn, mode, run_cost, process_cost):
    """Build a minimal Scheduler-like target with C_SchedulerHook attached.

    Only the attributes touched by wrapped_run_batch / wrapped_process_batch_result
    are wired; everything else is a no-op Mock. ``run_batch`` returns an instance
    whose class name is ``GenerationBatchResult`` so the simulation-batch branch runs.
    """
    from types import SimpleNamespace
    from unittest.mock import Mock

    from sglang_simulator.simulation.sglang import scheduler
    from sglang_simulator.simulation.types import SimulationMode

    scheduler.time = SimpleNamespace(
        time=lambda: clock["now"],
        perf_counter=lambda: clock["now"],
        sleep=lambda d: clock.__setitem__("now", clock["now"] + abs(d)),
    )
    hook.SIM_MODE = SimulationMode(mode)
    hook.OVERLAP_SCHEDULE = False
    hook.ITERATION_STATS = []
    hook.TOTAL_PREDICTOR_TIME_COST = 0.0
    hook.SIMULATION_BATCH = None
    hook.INFERENCE_PREDICTOR = SimpleNamespace(predict_infer_time=predict_fn)
    scheduler.get_obj_from_args = lambda _type_name, obj: obj

    class GenerationBatchResult:
        pass

    def run_batch(_self, _batch):
        clock["now"] += run_cost
        return GenerationBatchResult()

    def process_batch_result(_self, batch):
        clock["now"] += process_cost
        # Flip the request to decode so its token latency is recorded on the next iter.
        for req in batch.reqs:
            req.output_ids.append(1)

    target = SimpleNamespace(
        _prefetch_kvcache=Mock(),
        get_new_batch_prefill=Mock(),
        run_batch=run_batch,
        process_batch_result=process_batch_result,
        event_loop_normal=Mock(),
        init_request_dispatcher=Mock(),
    )
    hook.hook(target)
    return target, scheduler


def _make_extend_batch():
    """A one-request extend (prefill) batch whose forward_mode flips to decode."""
    from types import SimpleNamespace

    req = SimpleNamespace(rid="r1", extend_input_len=4, prefix_len=0, output_ids=[])
    batch = SimpleNamespace(
        reqs=[req],
        forward_mode=SimpleNamespace(
            is_extend=lambda: not req.output_ids, is_decode=lambda: bool(req.output_ids)
        ),
    )
    return batch, req


def _setup_req_stats(req, rid="r1"):
    from sglang_simulator.simulation.sglang.scheduler import request_stats_manager

    req_stats = request_stats_manager.get_req_stats(rid)
    req_stats.created_time = 0.0
    req_stats.input_length = 4
    req_stats.output_length = 2
    req_stats.queue_start = req_stats.queue_end = 0.0
    req_stats.last_event_time = 0.0
    return req_stats


@pytest.mark.parametrize("mode", ["OFFLINE", "BLOCKING"])
def test_predictor_cost_excluded_from_cpu_overhead(mode):
    """A predictor query must not inflate cpu_overhead (#39042).

    In OFFLINE the host thread is blocked by the predictor but no wall clock advances
    on its own, so last_real_time_ts is advanced by the query cost and cpu_overhead
    excludes it. In BLOCKING the same query advances the real clock via time.sleep,
    so the wall-time delta already accounts for it. In both modes cpu_overhead equals
    only the non-predictor CPU work per iteration (gap + run + process).
    """
    from sglang_simulator.simulation.sglang.scheduler import (
        C_SchedulerHook,
        StateManager,
        request_stats_manager,
    )

    predictor_delay = 0.3
    predicted_latency = 0.1
    run_cost = 0.03
    process_cost = 0.04
    gap = 0.02
    clock = {"now": 100.0}

    def predict(_batch):
        clock["now"] += predictor_delay
        return predicted_latency

    target, scheduler = _build_sim_target(
        C_SchedulerHook, clock, predict, mode, run_cost, process_cost
    )
    StateManager.reset()
    request_stats_manager.reset()
    StateManager.set_last_real_time_ts(clock["now"])
    batch, _req = _make_extend_batch()
    _setup_req_stats(_req)

    try:
        for _ in range(2):
            clock["now"] += gap
            StateManager.inc_hicache_l2_load_dur(0.0)
            _run_offline_iteration(target, batch, clock, run_cost, process_cost)

        cpu_overheads = [
            item["cpu_overhead"] for item in C_SchedulerHook.ITERATION_STATS
        ]
        # Non-predictor CPU work per iter = gap(0.02) + run(0.03) + process(0.04) = 0.09.
        # The 0.3s predictor query must NOT appear here in either mode.
        assert cpu_overheads == pytest.approx([0.09, 0.09])
        assert C_SchedulerHook.TOTAL_PREDICTOR_TIME_COST == pytest.approx(
            predictor_delay * 2
        )
    finally:
        StateManager.reset()
        request_stats_manager.reset()
        import time as real_time

        scheduler.time = real_time


def test_offline_token_latency_independent_of_predictor_query_duration():
    """Varying predictor query cost must not change offline TTFT (#39042).

    The regression reported up to ~41% TTFT error: a slow predictor inflated the
    first-token latency because its query time leaked into the simulated clock as
    scheduler overhead. With the fix, the recorded token latency tracks the
    predicted latency across query costs.
    """
    from sglang_simulator.simulation.sglang.scheduler import (
        C_SchedulerHook,
        StateManager,
        request_stats_manager,
    )

    predicted_latency = 0.1
    run_cost = 0.03
    process_cost = 0.04
    gap = 0.02

    def first_token_latency_for(predictor_delay):
        clock = {"now": 100.0}

        def predict(_batch):
            clock["now"] += predictor_delay
            return predicted_latency

        target, scheduler = _build_sim_target(
            C_SchedulerHook, clock, predict, "OFFLINE", run_cost, process_cost
        )
        StateManager.reset()
        request_stats_manager.reset()
        StateManager.set_last_real_time_ts(clock["now"])
        batch, req = _make_extend_batch()
        _setup_req_stats(req)
        try:
            # prefill (extend); process flips req to decode.
            clock["now"] += gap
            StateManager.inc_hicache_l2_load_dur(0.0)
            _run_offline_iteration(target, batch, clock, run_cost, process_cost)
            # decode: this iter records the first token latency.
            clock["now"] += gap
            StateManager.inc_hicache_l2_load_dur(0.0)
            _run_offline_iteration(target, batch, clock, run_cost, process_cost)
            req_stats = request_stats_manager.get_req_stats(req.rid)
            return req_stats.gen_token_latencies[0]
        finally:
            StateManager.reset()
            request_stats_manager.reset()
            import time as real_time

            scheduler.time = real_time

    ttfts = [first_token_latency_for(d) for d in (0.0, 0.1, 0.5, 1.0)]
    # TTFT is invariant to predictor query cost: it is the decode iteration's
    # simulated latency, not inflated by the predictor's host-thread blocking.
    assert max(ttfts) - min(ttfts) < 1e-9
