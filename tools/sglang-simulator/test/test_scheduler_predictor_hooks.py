"""Exercise installed scheduler hooks without importing the GPU runtime."""

import importlib
from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace

import pytest
from sglang_simulator.simulation.manager.state import StateManager
from sglang_simulator.simulation.sglang import scheduler
from sglang_simulator.simulation.sglang.prepared_batch import (
    SNAPSHOT_ATTRIBUTE,
    capture_prepared_batch,
)
from sglang_simulator.simulation.types import SimulationMode
from sglang_simulator.time_predictor.base import (
    PredictorError,
    ScheduleBatch,
    ScheduleRequest,
)


@pytest.fixture
def hooked_scheduler(monkeypatch):
    clock = SimpleNamespace(now=100.0)

    def advance(duration):
        clock.now += duration

    monkeypatch.setattr(scheduler.time, "time", lambda: clock.now)
    monkeypatch.setattr(scheduler.time, "perf_counter", lambda: clock.now)
    monkeypatch.setattr(scheduler.time, "sleep", advance)
    monkeypatch.setattr(
        scheduler, "get_obj_from_args", lambda name, *args, **kwargs: args[0]
    )
    hook = scheduler.C_SchedulerHook
    for attr, value in (
        ("ITERATION_STATS", []),
        ("TOTAL_PREDICTOR_TIME_COST", 0),
        ("CURRENT_PREDICTOR_TIME_COST", 0),
        ("CURRENT_PREDICTED_LATENCY", 0),
        ("SIMULATION_BATCH", None),
        ("GET_NEW_BATCH_PREFILL_TIME_COST", 0),
        ("SIM_MODE", SimulationMode.OFFLINE),
        ("OVERLAP_SCHEDULE", False),
        ("COORDINATOR", None),
    ):
        monkeypatch.setattr(hook, attr, value)

    class Target:
        def __init__(self):
            pass

        def recv_requests(self):
            return []

        def _prefetch_kvcache(self, req):
            pass

        def _release_aborted_request(self, req):
            self.released_request_id = req.rid

        def get_new_batch_prefill(self):
            return None

        def get_next_batch_to_run(self, running_batch=None, last_batch=None):
            if hasattr(self, "prefill_decode_interval"):
                self.selection_calls.append("native selection")
                if self._should_defer_prefill():
                    return SimpleNamespace(batch_to_run=None)
            return self.plan

        def _should_defer_prefill(self):
            self.selection_calls.append("native defer")
            if self._prefill_decode_interval_remaining == 0:
                return False
            self._prefill_decode_interval_remaining -= 1
            return True

        def _arm_prefill_decode_interval(self, batch):
            if self.prefill_decode_interval and batch is not None:
                self._prefill_decode_interval_remaining = self.prefill_decode_interval

        def is_disable_overlap_for_batch(self, batch, **kwargs):
            return self.disable_overlap

        def run_batch(self, batch):
            advance(0.2)
            result = type("GenerationBatchResult", (), {})()
            result.logits_output = SimpleNamespace()
            if predictor.requires_prepared_batch:
                setattr(
                    result.logits_output,
                    SNAPSHOT_ATTRIBUTE,
                    capture_prepared_batch(batch.prepared_forward),
                )
            if getattr(self, "after_forward", None) is not None:
                self.after_forward(batch, result)
            return result

        def process_batch_result(self, batch, result=None):
            advance(0.01)
            if getattr(self, "process_error", None) is not None:
                raise self.process_error
            if getattr(self, "after_process", None) is not None:
                self.after_process(batch, result)

        def event_loop_normal(self):
            pass

        def event_loop_overlap(self):
            return "native overlap"

        def on_idle(self):
            pass

        def init_request_dispatcher(self):
            self._request_dispatcher = SimpleNamespace(_mapping={ProfileReq: None})

    class ProfileReq:
        req_type = SimpleNamespace(name="STOP_PROFILE")

    class Predictor:
        requires_prepared_batch = False
        latency = 0.05

        def predict_infer_time(self, batch):
            self.last_batch = batch
            advance(0.3)
            return self.latency

        def get_metrics(self):
            return {}

        def reset_metrics(self):
            pass

    predictor = Predictor()
    monkeypatch.setattr(hook, "INFERENCE_PREDICTOR", predictor)
    hook.hook(Target)
    target = Target.__new__(Target)
    target.init_request_dispatcher()
    StateManager.reset()
    StateManager.set_global_clock(clock.now)
    StateManager.set_last_real_time_ts(clock.now)
    scheduler.request_stats_manager.reset()
    yield SimpleNamespace(
        target=target,
        predictor=predictor,
        hook=hook,
        clock=clock,
        profile_req=ProfileReq(),
    )
    StateManager.reset()
    scheduler.request_stats_manager.reset()


def make_batch(rid="request"):
    return SimpleNamespace(
        forward_mode=SimpleNamespace(
            name="EXTEND", is_extend=lambda: True, is_decode=lambda: False
        ),
        reqs=[
            SimpleNamespace(rid=rid, extend_input_len=2, prefix_len=10, output_ids=[9])
        ],
        extend_lens=[8],
        prefix_lens=[0],
        prepared_forward=SimpleNamespace(
            forward_mode=SimpleNamespace(name="EXTEND"),
            batch_size=1,
            seq_lens_cpu=[8],
            extend_seq_lens_cpu=[8],
            extend_prefix_lens_cpu=[0],
        ),
    )


@pytest.mark.parametrize("native_dp", [None, False, True])
@pytest.mark.parametrize("rank", [0, 1])
def test_initialized_scheduler_passes_native_rank_only_to_paired_predictor(
    hooked_scheduler, monkeypatch, native_dp, rank
):
    from sglang_simulator.simulation.manager.config import ConfigManager
    from sglang_simulator.simulation.sglang import coordinated, dp_coordinated

    target = hooked_scheduler.target
    target.tp_group = SimpleNamespace(rank_in_group=rank)
    native_args = SimpleNamespace(disable_overlap_schedule=False)
    calls = []
    monkeypatch.setattr(scheduler, "get_obj_from_args", lambda *a, **kw: native_args)
    monkeypatch.setattr(scheduler, "validate_simulator_server_args", lambda *a: None)
    monkeypatch.setattr(scheduler, "validate_required_class_hooks", lambda: None)
    monkeypatch.setattr(
        scheduler.Envs,
        "coordinator_endpoint",
        lambda: None if native_dp is None else "tcp://127.0.0.1:12345",
    )
    monkeypatch.setattr(dp_coordinated, "native_dp_clock_enabled", lambda: native_dp)
    monkeypatch.setattr(dp_coordinated, "internal_endpoint", lambda *a: "ipc://unused")
    monkeypatch.setattr(dp_coordinated, "DPCoordinatedScheduler", lambda **kw: object())
    monkeypatch.setattr(coordinated, "CoordinatedScheduler", lambda **kw: object())
    model, hw, config = object(), object(), object()
    monkeypatch.setattr(ConfigManager, "get_model_info", lambda: model)
    monkeypatch.setattr(ConfigManager, "get_accelerator_info", lambda: hw)
    monkeypatch.setattr(ConfigManager, "get_scheduler_config", lambda: config)

    def create(*args, **kwargs):
        calls.append((args, kwargs))
        return hooked_scheduler.predictor

    monkeypatch.setattr(ConfigManager, "get_inference_time_predictor", create)
    type(target).__init__(target)
    assert calls == [
        ((model, hw, config), {"attention_dp_rank": rank} if native_dp else {})
    ]


@pytest.mark.parametrize("interval", [0, 1, 192])
@pytest.mark.parametrize("reference", [False, True])
def test_cadence_and_reference_hooks_compose_without_replacing_native_selection(
    hooked_scheduler, interval, reference
):
    from sglang_simulator.simulation.sglang.coordinated import (
        _validate_prefill_cadence_runtime,
    )

    test = hooked_scheduler
    target = test.target
    target.prefill_decode_interval = interval
    target._prefill_decode_interval_remaining = min(interval, 1)
    target.selection_calls = []
    target.plan = SimpleNamespace(batch_to_run=make_batch())
    bridge = SimpleNamespace(
        _in_scheduler_pass=False, _cadence_deferred=False, _cadence_only_pass=False
    )
    test.hook.COORDINATOR = bridge
    selected = []

    def selected_batch(owner, batch, *, prior_requests):
        assert owner is target
        assert not bridge._in_scheduler_pass
        selected.append(batch)

    if reference:
        test.predictor.dspark_runtime = SimpleNamespace(
            reference_driver=SimpleNamespace(selected=selected_batch)
        )

    _validate_prefill_cadence_runtime(target)
    first = target.get_next_batch_to_run(running_batch=None, last_batch=None)
    assert target.selection_calls == ["native selection", "native defer"]
    assert bridge._in_scheduler_pass is False
    assert bridge._cadence_deferred is bool(interval)
    assert bridge._cadence_only_pass is bool(interval)
    assert first.batch_to_run is (None if interval else target.plan.batch_to_run)
    assert selected == ([first.batch_to_run] if reference else [])

    target.selection_calls.clear()
    second = target.get_next_batch_to_run(None, None)
    assert target.selection_calls == ["native selection", "native defer"]
    assert target._prefill_decode_interval_remaining == 0
    assert bridge._cadence_deferred is False
    assert bridge._cadence_only_pass is False
    assert second is target.plan
    assert selected == ([first.batch_to_run, second.batch_to_run] if reference else [])


def test_composed_selection_error_keeps_native_exception_and_clears_cadence(
    hooked_scheduler,
):
    test = hooked_scheduler
    error = RuntimeError("native selection failed")

    class FailingTrace(list):
        def append(self, value):
            super().append(value)
            if value == "native defer":
                raise error

    target = test.target
    target.prefill_decode_interval = 1
    target._prefill_decode_interval_remaining = 0
    target.selection_calls = FailingTrace()
    bridge = SimpleNamespace(
        _in_scheduler_pass=False, _cadence_deferred=False, _cadence_only_pass=False
    )
    test.hook.COORDINATOR = bridge
    test.predictor.dspark_runtime = SimpleNamespace(
        reference_driver=SimpleNamespace(
            selected=lambda *args, **kwargs: pytest.fail("failed selection published")
        )
    )
    with pytest.raises(RuntimeError) as caught:
        target.get_next_batch_to_run(None, None)
    assert caught.value is error
    assert target.selection_calls == ["native selection", "native defer"]
    assert bridge._in_scheduler_pass is False
    assert bridge._cadence_only_pass is False


def test_prepared_metadata_is_opt_in(hooked_scheduler):
    test = hooked_scheduler
    batch = make_batch()
    test.target.run_batch(batch)
    assert test.predictor.last_batch.request_info() == [[2, 11]]
    assert test.predictor.last_batch.forward_mode is None
    test.predictor.requires_prepared_batch = True
    test.target.run_batch(batch)
    assert test.predictor.last_batch.request_info() == [[8, 0]]
    assert test.predictor.last_batch.forward_mode == "EXTEND"


def test_prepared_snapshot_survives_post_forward_batch_mutation(hooked_scheduler):
    test = hooked_scheduler
    test.predictor.requires_prepared_batch = True

    def mutate(batch, result):
        batch.prepared_forward.seq_lens_cpu[0] = 999
        batch.prepared_forward.extend_seq_lens_cpu[0] = 999
        batch.extend_lens[0] = 500
        batch.prefix_lens[0] = 499

    test.target.after_forward = mutate
    test.target.run_batch(make_batch())
    assert test.predictor.last_batch.request_info() == [[8, 0]]


@pytest.mark.parametrize("local_idle", [False, True])
def test_dspark_uses_complete_staged_timing_without_ordinary_forward_query(
    hooked_scheduler, local_idle
):
    from sglang_simulator.simulation.sglang.dspark import (
        STEP_ATTRIBUTE,
        DSparkGroup,
        DSparkPhaseCost,
        DSparkProgress,
        DSparkRequest,
        DSparkStep,
        DSparkStepResult,
        DSparkStepTiming,
        DSparkTopology,
    )

    test = hooked_scheduler
    test.predictor.supports_dspark = True
    request = DSparkRequest("request", 512, 6, 0, 512, 1, 20, '{"max_new_tokens":20}')
    groups = (DSparkGroup(0, "TARGET_VERIFY", (request,), '{"path":"CPU fixture"}'),)
    if local_idle:
        groups += (DSparkGroup(1, "IDLE", (), '{"path":"coordinated idle"}'),)
    before = DSparkStep(
        0, DSparkTopology(2, 2 if local_idle else 1), groups, '{"source":"CPU fixture"}'
    )
    result = DSparkStepResult(
        before,
        DSparkStepTiming(
            0,
            tuple(
                DSparkPhaseCost(p, 0.01, "CPU fixture") for p in before.required_phases
            ),
        ),
        DSparkProgress(0, (("request", 2),), "CPU fixture"),
        local_attention_dp_rank=1 if local_idle else 0,
        prediction_wall_seconds=0.3,
    )

    def attach(batch, native_result):
        test.clock.now += 0.3
        setattr(native_result, STEP_ATTRIBUTE, result)

    test.target.after_forward = attach
    batch = make_batch()
    batch.forward_mode.name = "IDLE" if local_idle else "DECODE"
    if local_idle:
        batch.reqs = []
    test.target.run_batch(batch)
    assert not hasattr(test.predictor, "last_batch")
    assert StateManager.get_current_inference_dur() == pytest.approx(0.05)
    assert StateManager.get_iteration() == 1
    assert test.hook.CURRENT_PREDICTOR_TIME_COST == pytest.approx(0.3)
    assert test.hook.SIMULATION_BATCH.forward_mode == (
        "IDLE" if local_idle else "TARGET_VERIFY"
    )


def test_dspark_result_cannot_relabel_an_extend_batch(hooked_scheduler):
    from sglang_simulator.simulation.sglang.dspark import STEP_ATTRIBUTE
    from test_dspark_step import session, step

    test = hooked_scheduler
    test.predictor.supports_dspark = True
    before = step(requests=None)
    result = session(before).execute(before)
    test.target.after_forward = lambda batch, ret: setattr(ret, STEP_ATTRIBUTE, result)
    batch = make_batch("a")
    batch.forward_mode.name = "EXTEND"
    with pytest.raises(PredictorError, match="worker mode differs"):
        test.target.run_batch(batch)
    assert StateManager.get_iteration() == 0


def dspark_prediction(fid, *, prefixes, seconds, wall_seconds):
    from sglang_simulator.simulation.sglang.dspark import (
        DSparkPhaseCost,
        DSparkStepTiming,
    )
    from test_dspark_step import request, session, step

    before = step(
        fid=fid,
        requests=tuple(request(rid, prefix=prefix) for rid, prefix in prefixes),
    )
    prediction = session(before).predict(before)
    return replace(
        prediction,
        timing=DSparkStepTiming(
            fid,
            tuple(
                DSparkPhaseCost(
                    phase, seconds / len(before.required_phases), "CPU fixture"
                )
                for phase in before.required_phases
            ),
        ),
        prediction_wall_seconds=wall_seconds,
    )


def launch_dspark(test, prediction, *, preprocess=0.0):
    from sglang_simulator.simulation.sglang.dspark import STEP_ATTRIBUTE

    test.predictor.supports_dspark = True
    group = prediction.local_group
    batch = make_batch(group.requests[0].rid)
    batch.forward_mode.name = "DECODE"
    batch.reqs = [
        SimpleNamespace(rid=request.rid, output_ids=[9]) for request in group.requests
    ]

    def attach(batch, native_result):
        test.clock.now += prediction.prediction_wall_seconds
        setattr(native_result, STEP_ATTRIBUTE, prediction)

    test.target.after_forward = attach
    test.hook.GET_NEW_BATCH_PREFILL_TIME_COST = preprocess
    for request in group.requests:
        scheduler.request_stats_manager.get_req_stats(request.rid).last_event_time = 100
    return batch, test.target.run_batch(batch)


@pytest.mark.parametrize("launch_both_first", [False, True])
def test_dspark_processing_uses_each_result_and_excludes_each_launch_wall_once(
    hooked_scheduler, launch_both_first
):
    test = hooked_scheduler
    first = dspark_prediction(0, prefixes=(("a", 512),), seconds=0.05, wall_seconds=0.3)
    second = dspark_prediction(
        1, prefixes=(("b", 1024), ("c", 2048)), seconds=0.15, wall_seconds=0.7
    )
    batch0, result0 = launch_dspark(test, first, preprocess=0.02)
    if not launch_both_first:
        test.target.process_batch_result(batch0, result0)
    batch1, result1 = launch_dspark(test, second, preprocess=0.07)
    if launch_both_first:
        test.hook.CURRENT_PREDICTED_LATENCY = 99
        test.hook.CURRENT_PREDICTOR_TIME_COST = 99
        test.hook.SIMULATION_BATCH = None
        test.hook.GET_NEW_BATCH_PREFILL_TIME_COST = 99
        test.target.process_batch_result(batch0, result0)
    test.target.process_batch_result(batch1, result1)

    first_stats, second_stats = test.hook.ITERATION_STATS
    assert first_stats["requests"] == [[6, 512]]
    assert second_stats["requests"] == [[6, 1024], [6, 2048]]
    assert (
        first_stats["forward_mode"] == second_stats["forward_mode"] == "TARGET_VERIFY"
    )
    assert first_stats["predicted_forward_latency"] == pytest.approx(0.05)
    assert second_stats["predicted_forward_latency"] == pytest.approx(0.15)
    assert first_stats["forward_latency"] == pytest.approx(0.05)
    assert second_stats["forward_latency"] == pytest.approx(0.15)
    assert first_stats["preprocess_latency"] == pytest.approx(0.02)
    assert second_stats["preprocess_latency"] == pytest.approx(0.07)
    assert [row["cpu_overhead"] for row in test.hook.ITERATION_STATS] == pytest.approx(
        [0.41, 0.01] if launch_both_first else [0.21, 0.21]
    )
    assert StateManager.get_global_clock() == pytest.approx(100.62)
    assert getattr(test.target, scheduler._DSPARK_WALL_CREDIT) == 0
    assert test.hook.TOTAL_PREDICTOR_TIME_COST == pytest.approx(1.0)
    assert not hasattr(test.predictor, "last_batch")
    assert getattr(result0, scheduler._DSPARK_ACCOUNTING).prediction is first
    with pytest.raises(FrozenInstanceError):
        getattr(result0, scheduler._DSPARK_ACCOUNTING).forward_duration = 5


@pytest.mark.parametrize("change", ["order", "mode", "prediction", "accounting"])
def test_dspark_process_revalidates_exact_result_and_ordered_batch_before_processing(
    hooked_scheduler, change
):
    from sglang_simulator.simulation.sglang.dspark import STEP_ATTRIBUTE

    test = hooked_scheduler
    prediction = dspark_prediction(
        0, prefixes=(("a", 512), ("b", 1024)), seconds=0.05, wall_seconds=0.3
    )
    batch, result = launch_dspark(test, prediction)
    if change == "order":
        batch.reqs.reverse()
    elif change == "mode":
        batch.forward_mode.name = "EXTEND"
    elif change == "prediction":
        setattr(result, STEP_ATTRIBUTE, replace(prediction))
    else:
        delattr(result, scheduler._DSPARK_ACCOUNTING)
    now = test.clock.now
    with pytest.raises(PredictorError):
        test.target.process_batch_result(batch, result)
    assert test.clock.now == now
    assert test.hook.ITERATION_STATS == []
    assert StateManager.get_global_clock() == 100


def test_dspark_result_accounting_cannot_be_consumed_twice(hooked_scheduler):
    test = hooked_scheduler
    prediction = dspark_prediction(
        0, prefixes=(("a", 512),), seconds=0.05, wall_seconds=0.3
    )
    batch, result = launch_dspark(test, prediction)
    test.target.process_batch_result(batch, result=result)
    now = test.clock.now
    with pytest.raises(PredictorError, match="already consumed"):
        test.target.process_batch_result(batch, result)
    assert test.clock.now == now
    assert len(test.hook.ITERATION_STATS) == 1


def test_dspark_failed_native_processing_preserves_unconsumed_accounting(
    hooked_scheduler,
):
    test = hooked_scheduler
    prediction = dspark_prediction(
        0, prefixes=(("a", 512),), seconds=0.05, wall_seconds=0.3
    )
    batch, result = launch_dspark(test, prediction)
    test.target.process_error = RuntimeError("native processing failed")
    with pytest.raises(RuntimeError, match="native processing failed"):
        test.target.process_batch_result(batch, result)
    assert getattr(result, scheduler._DSPARK_ACCOUNTED) is False
    assert getattr(test.target, scheduler._DSPARK_WALL_CREDIT) == pytest.approx(0.3)
    assert test.hook.ITERATION_STATS == []
    assert StateManager.get_global_clock() == 100


def test_dspark_coordinator_processing_keeps_existing_delegation(hooked_scheduler):
    test = hooked_scheduler
    prediction = dspark_prediction(
        0, prefixes=(("a", 512),), seconds=0.05, wall_seconds=0.3
    )
    batch, result = launch_dspark(test, prediction)
    observed = []

    def process(actual_batch, actual_result, *, process, duration):
        assert duration == pytest.approx(0.05)
        observed.append((actual_batch, actual_result))
        process()
        return "coordinator result"

    test.hook.COORDINATOR = SimpleNamespace(process_batch_result=process)
    assert test.target.process_batch_result(batch, result) == "coordinator result"
    assert observed == [(batch, result)]
    assert test.hook.ITERATION_STATS == []
    assert getattr(result, scheduler._DSPARK_ACCOUNTED) is True
    assert getattr(test.target, scheduler._DSPARK_PENDING_RESULTS) == 0
    assert StateManager.get_global_clock() == 100


def test_dspark_result_cannot_be_processed_by_a_different_scheduler(hooked_scheduler):
    test = hooked_scheduler
    prediction = dspark_prediction(
        0, prefixes=(("a", 512),), seconds=0.05, wall_seconds=0.3
    )
    batch, result = launch_dspark(test, prediction)
    other = type(test.target).__new__(type(test.target))
    with pytest.raises(PredictorError, match="accounting"):
        other.process_batch_result(batch, result)
    assert getattr(test.target, scheduler._DSPARK_PENDING_RESULTS) == 1
    assert getattr(result, scheduler._DSPARK_ACCOUNTED) is False


def mock_profile_output(monkeypatch):
    original_import = importlib.import_module
    monkeypatch.setattr(
        scheduler.importlib,
        "import_module",
        lambda name: (
            SimpleNamespace(ProfileReqOutput=SimpleNamespace)
            if name == "sglang.srt.managers.io_struct"
            else original_import(name)
        ),
    )


def test_dspark_profile_reset_requires_pending_results_to_drain(
    hooked_scheduler, monkeypatch, tmp_path
):
    test = hooked_scheduler
    mock_profile_output(monkeypatch)
    monkeypatch.setattr(
        scheduler.Envs, "output_dir", classmethod(lambda cls: str(tmp_path))
    )
    prediction = dspark_prediction(
        0, prefixes=(("a", 512),), seconds=0.05, wall_seconds=0.3
    )
    batch, result = launch_dspark(test, prediction)
    profile = next(iter(test.target._request_dispatcher._mapping.values()))
    rejected = profile(test.profile_req)
    assert rejected.success is False
    assert "must drain" in rejected.message
    assert StateManager.get_global_clock() == 100
    assert getattr(test.target, scheduler._DSPARK_WALL_CREDIT) == pytest.approx(0.3)
    assert not list(tmp_path.iterdir())

    test.target.process_batch_result(batch, result)
    assert getattr(test.target, scheduler._DSPARK_PENDING_RESULTS) == 0
    assert profile(test.profile_req).success is True
    assert getattr(test.target, scheduler._DSPARK_WALL_CREDIT) == 0


@pytest.mark.parametrize("writer", [False, True])
def test_dp_profile_uses_existing_metrics_once_for_combined_request_stats(
    hooked_scheduler, monkeypatch, tmp_path, writer
):
    """Exercise the installed real profile callback, not a replacement writer."""
    import json

    from sglang_simulator.simulation.types import RequestStats

    test = hooked_scheduler
    mock_profile_output(monkeypatch)
    monkeypatch.setattr(
        scheduler.Envs, "output_dir", classmethod(lambda cls: str(tmp_path))
    )
    combined = [
        RequestStats(
            rid=rid,
            input_length=10,
            output_length=2,
            created_time=1.0,
            queue_start=1.0,
            queue_end=1.1,
            last_event_time=1.5,
            gen_token_latencies=[0.2, 0.2],
        )
        for rid in ("dp0", "dp1")
    ]
    predictor_metrics = [{"global_cost_once": 5}, {"global_cost_once": 0}]

    def profile_boundary(req, *, snapshot, execute):
        local = snapshot()
        assert set(local) == {"profile_stats", "iterations", "predictor_metrics"}
        return execute(
            {
                "profile_stats": combined,
                "iterations": [],
                "local_predictor_metrics": predictor_metrics,
            },
            writer,
        )

    test.hook.COORDINATOR = SimpleNamespace(profile_boundary=profile_boundary)
    profile = next(iter(test.target._request_dispatcher._mapping.values()))
    assert profile(test.profile_req).success
    if writer:
        metrics = json.loads((tmp_path / "metrics.json").read_text())
        rows = [
            json.loads(line)
            for line in (tmp_path / "request.jsonl").read_text().splitlines()
        ]
        assert [row["rid"] for row in rows] == ["dp0", "dp1"]
        assert metrics["global_cost_once"] == 5
        assert metrics["attention_dp_local_predictor_metrics"] == predictor_metrics
    else:
        assert not list(tmp_path.iterdir())
    assert test.hook.ITERATION_STATS == []


def test_coordinator_active_at_dspark_launch_consumes_result_owned_accounting(
    hooked_scheduler, monkeypatch, tmp_path
):
    test = hooked_scheduler
    mock_profile_output(monkeypatch)
    monkeypatch.setattr(
        scheduler.Envs, "output_dir", classmethod(lambda cls: str(tmp_path))
    )
    observed = []

    def process(batch, result, *, process, duration):
        observed.append((batch, result, duration))
        value = process()
        for req in batch.reqs:
            scheduler.request_stats_manager.get_req_stats(
                req.rid
            ).gen_token_latencies = [0.05]
        return value

    test.hook.COORDINATOR = SimpleNamespace(process_batch_result=process)
    prediction = dspark_prediction(
        0, prefixes=(("a", 512),), seconds=0.05, wall_seconds=0.3
    )
    batch, result = launch_dspark(test, prediction)
    test.target.process_batch_result(batch, result)
    assert observed == [(batch, result, pytest.approx(0.05))]
    assert getattr(test.target, scheduler._DSPARK_PENDING_RESULTS, 0) == 0
    assert getattr(test.target, scheduler._DSPARK_WALL_CREDIT, 0) == 0
    assert test.hook.ITERATION_STATS == []
    profile = next(iter(test.target._request_dispatcher._mapping.values()))
    assert profile(test.profile_req).success is True


def test_dspark_blocking_keeps_native_host_time_in_result_duration(hooked_scheduler):
    test = hooked_scheduler
    test.hook.SIM_MODE = SimulationMode.BLOCKING
    prediction = dspark_prediction(
        0, prefixes=(("a", 512),), seconds=0.05, wall_seconds=0.3
    )
    batch, result = launch_dspark(test, prediction)
    test.target.process_batch_result(batch, result)
    stats = test.hook.ITERATION_STATS[-1]
    assert stats["forward_latency"] == pytest.approx(0.55)
    assert stats["cpu_overhead"] == pytest.approx(0.01)
    assert StateManager.get_global_clock() == pytest.approx(100.56)
    assert getattr(test.target, scheduler._DSPARK_WALL_CREDIT, 0) == 0


@pytest.mark.parametrize("snapshot", [None, object()])
def test_prepared_predictor_rejects_missing_or_invalid_result_snapshot(
    hooked_scheduler, snapshot
):
    test = hooked_scheduler
    test.predictor.requires_prepared_batch = True
    test.target.after_forward = lambda batch, result: setattr(
        result.logits_output, SNAPSHOT_ATTRIBUTE, snapshot
    )
    with pytest.raises(PredictorError, match="prepared snapshot"):
        test.target.run_batch(make_batch())
    assert not hasattr(test.predictor, "last_batch")
    assert StateManager.get_iteration() == 0


def test_legacy_negative_prediction_remains_supported(hooked_scheduler):
    test = hooked_scheduler
    test.predictor.latency = -1
    test.target.run_batch(make_batch())
    assert StateManager.get_current_inference_dur() == -1
    assert StateManager.get_iteration() == 1


@pytest.mark.parametrize("prepared, iterations", [(False, 1), (True, 0)])
def test_failed_prediction_counts_only_legacy_attempts(
    hooked_scheduler, monkeypatch, prepared, iterations
):
    test = hooked_scheduler
    test.predictor.requires_prepared_batch = prepared

    def fail_prediction(batch):
        raise RuntimeError("missing performance data")

    monkeypatch.setattr(test.predictor, "predict_infer_time", fail_prediction)
    with pytest.raises(RuntimeError, match="missing performance data"):
        test.target.run_batch(make_batch())
    assert StateManager.get_iteration() == iterations
    assert StateManager.get_current_inference_dur() == 0
    assert StateManager.get_global_clock() == 100


@pytest.mark.parametrize("prepared, cpu_time", [(False, 0.51), (True, 0.21)])
def test_offline_excludes_query_time_only_for_opted_in_predictor(
    hooked_scheduler, prepared, cpu_time
):
    test = hooked_scheduler
    test.predictor.requires_prepared_batch = prepared
    batch = make_batch()
    scheduler.request_stats_manager.get_req_stats("request").last_event_time = 100
    test.target.run_batch(batch)
    test.target.process_batch_result(batch)
    assert test.hook.ITERATION_STATS[-1]["cpu_overhead"] == pytest.approx(cpu_time)


@pytest.mark.parametrize("prepared", [False, True])
def test_blocking_accounts_host_time_for_consecutive_arrivals(
    hooked_scheduler, prepared
):
    test = hooked_scheduler
    test.hook.SIM_MODE = SimulationMode.BLOCKING
    test.predictor.requires_prepared_batch = prepared
    for rid in ("first", "second"):
        stats = scheduler.request_stats_manager.get_req_stats(rid)
        stats.created_time = stats.last_event_time = test.clock.now
        batch = make_batch(rid)
        test.target.run_batch(batch)
        test.target.process_batch_result(batch)
        assert stats.gen_token_latencies == pytest.approx([0.56])
        assert stats.last_event_time == pytest.approx(test.clock.now)
    if prepared:
        assert test.hook.ITERATION_STATS[-1]["predicted_forward_latency"] == 0.05


def test_failed_profile_save_preserves_live_timestamps(
    hooked_scheduler, monkeypatch, tmp_path
):
    test = hooked_scheduler
    stats = scheduler.request_stats_manager.get_req_stats("request")
    stats.created_time = 100
    stats.queue_start = stats.queue_end = 100
    stats.last_event_time = 101
    stats.gen_token_latencies = [1]
    StateManager.set_global_clock(101)
    monkeypatch.setattr(
        scheduler.Envs, "output_dir", classmethod(lambda cls: str(tmp_path))
    )

    def fail_open(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(scheduler, "open", fail_open, raising=False)
    original_import = importlib.import_module
    monkeypatch.setattr(
        scheduler.importlib,
        "import_module",
        lambda name: (
            SimpleNamespace(ProfileReqOutput=SimpleNamespace)
            if name == "sglang.srt.managers.io_struct"
            else original_import(name)
        ),
    )
    profile = next(iter(test.target._request_dispatcher._mapping.values()))
    result = profile(test.profile_req)
    assert result.success is False
    assert stats.created_time == 100
    assert stats.last_event_time == 101
    assert StateManager.get_global_clock() == 101
    test.clock.now = 101
    StateManager.set_last_real_time_ts(101)
    batch = make_batch()
    test.target.run_batch(batch)
    test.target.process_batch_result(batch)
    assert stats.gen_token_latencies == pytest.approx([1, 0.56])


def test_forward_metadata_does_not_change_legacy_batch_repr_or_equality():
    legacy = ScheduleBatch([ScheduleRequest(1, 10)])
    prepared = ScheduleBatch([ScheduleRequest(1, 10)], forward_mode="EXTEND")
    assert legacy == prepared
    assert repr(legacy) == repr(prepared)
    assert legacy.is_decode()
    assert prepared.is_prefill()
