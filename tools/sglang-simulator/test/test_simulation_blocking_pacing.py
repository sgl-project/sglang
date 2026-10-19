"""Check response delivery at the native scheduler boundary without wall sleeps."""

from types import SimpleNamespace

import pytest
from sglang_simulator.simulation.manager import StateManager
from sglang_simulator.simulation.sglang import scheduler
from sglang_simulator.simulation.sglang.req_stats_manager import request_stats_manager
from sglang_simulator.simulation.types import SimulationMode
from sglang_simulator.time_predictor import ScheduleBatch as SimulationScheduleBatch


class ScheduleBatch:
    __module__ = "sglang.srt.managers.schedule_batch"

    def __init__(self, reqs):
        self.reqs = reqs


@pytest.fixture
def hooked_scheduler(monkeypatch):
    clock = SimpleNamespace(now=10.2, sleeps=[], deliveries=[])

    def sleep(delay):
        clock.sleeps.append(delay)
        clock.now += delay

    class NativeScheduler:
        def __init__(self):
            pass

        def _noop(self, *args, **kwargs):
            pass

        recv_requests = _noop
        _prefetch_kvcache = _noop
        get_new_batch_prefill = _noop
        run_batch = _noop
        event_loop_normal = _noop
        init_request_dispatcher = _noop
        on_idle = _noop

        def process_batch_result(self, batch, result):
            clock.deliveries.append(clock.now)
            clock.now += 0.02
            return result

    monkeypatch.setattr(scheduler.time, "time", lambda: clock.now)
    monkeypatch.setattr(scheduler.time, "perf_counter", lambda: clock.now)
    monkeypatch.setattr(scheduler.time, "sleep", sleep)
    monkeypatch.setattr(scheduler.C_SchedulerHook, "COORDINATOR", None)
    monkeypatch.setattr(scheduler.C_SchedulerHook, "ITERATION_STATS", [])
    monkeypatch.setattr(
        scheduler.C_SchedulerHook, "SIMULATION_BATCH", SimulationScheduleBatch()
    )
    StateManager.reset()
    request_stats_manager.reset()
    StateManager.set_global_clock(10.0)
    StateManager.set_last_real_time_ts(clock.now)
    StateManager.set_current_inference_dur(0.1)
    StateManager.set_current_inference_dur(0.2)
    scheduler.C_SchedulerHook.hook(NativeScheduler)
    yield object.__new__(NativeScheduler), clock
    StateManager.reset()
    request_stats_manager.reset()


@pytest.mark.parametrize("mode", [SimulationMode.BLOCKING, SimulationMode.OFFLINE])
@pytest.mark.parametrize("overlap, load", [(False, 0.3), (True, 0.3), (True, 0.05)])
def test_cache_load_delays_blocking_response_delivery(
    monkeypatch, hooked_scheduler, mode, overlap, load
):
    native, clock = hooked_scheduler
    monkeypatch.setattr(scheduler.C_SchedulerHook, "SIM_MODE", mode)
    monkeypatch.setattr(scheduler.C_SchedulerHook, "OVERLAP_SCHEDULE", overlap)
    StateManager.inc_hicache_l2_load_dur(load)
    request = SimpleNamespace(rid="request", output_ids=[1])
    stats = request_stats_manager.get_req_stats(request.rid)
    stats.last_event_time = 10.0
    batch = ScheduleBatch([request])
    result = object()

    assert native.process_batch_result(batch, result) is result

    visible_delay = max(load - 0.1, 0) if overlap else load
    wall_delay = visible_delay if mode == SimulationMode.BLOCKING else 0
    assert clock.deliveries == pytest.approx([10.2 + wall_delay])
    assert clock.sleeps == pytest.approx([wall_delay] if wall_delay else [])
    # Moving the wait must not count it again as CPU overhead or change OFFLINE.
    assert StateManager.get_global_clock() == pytest.approx(10.22 + visible_delay)
    assert stats.gen_token_latencies == pytest.approx([0.22 + visible_delay])
    iteration = scheduler.C_SchedulerHook.ITERATION_STATS[-1]
    assert iteration["cpu_overhead"] == pytest.approx(0.02)
    assert iteration["postprocess_latency"] == pytest.approx(0.02)
    assert iteration["l2_blocking_wall_latency"] == pytest.approx(wall_delay)
    assert StateManager.get_hicache_l2_load_dur() == 0

    native.process_batch_result(batch, result)
    assert clock.sleeps == pytest.approx([wall_delay] if wall_delay else [])


def test_empty_batch_does_not_consume_cache_load(monkeypatch, hooked_scheduler):
    native, clock = hooked_scheduler
    monkeypatch.setattr(scheduler.C_SchedulerHook, "SIM_MODE", SimulationMode.BLOCKING)
    StateManager.inc_hicache_l2_load_dur(0.3)

    native.process_batch_result(ScheduleBatch([]), object())

    assert clock.sleeps == []
    assert StateManager.get_hicache_l2_load_dur() == 0.3
