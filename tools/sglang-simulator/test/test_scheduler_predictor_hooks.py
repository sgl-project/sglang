"""Exercise installed scheduler hooks without importing the GPU runtime."""

import importlib
from types import SimpleNamespace

import pytest
from sglang_simulator.simulation.manager.state import StateManager
from sglang_simulator.simulation.sglang import scheduler
from sglang_simulator.simulation.types import SimulationMode
from sglang_simulator.time_predictor.base import ScheduleBatch, ScheduleRequest


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
    ):
        monkeypatch.setattr(hook, attr, value)

    class Target:
        def __init__(self):
            pass

        def recv_requests(self):
            return []

        def _prefetch_kvcache(self, req):
            pass

        def get_new_batch_prefill(self):
            return None

        def run_batch(self, batch):
            advance(0.2)
            return type("GenerationBatchResult", (), {})()

        def process_batch_result(self, batch):
            advance(0.01)

        def event_loop_normal(self):
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
    )


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
