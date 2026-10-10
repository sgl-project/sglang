# SPDX-License-Identifier: Apache-2.0

from collections import deque
from types import SimpleNamespace

import pytest

import sglang.multimodal_gen.runtime.managers.pi05_segment_scheduler as module
from sglang.multimodal_gen.runtime.managers.pi05_segment_scheduler import (
    Pi05SegmentSchedulerMixin,
)


class Request:
    def __init__(self, name, steps=5, signature="a"):
        self.request_id = name
        self.num_inference_steps = steps
        self.extra = {"vla": {}}
        self.is_warmup = False
        self.signature = signature
        self.reject = False


class Scheduler(Pi05SegmentSchedulerMixin):
    def __init__(self, rows, rank=0):
        self.waiting_queue = deque((b"client", req, ts) for req, ts in rows)
        self.server_args = SimpleNamespace(
            pipeline_config=SimpleNamespace(enable_segmented_actions=True), tp_size=1
        )
        self.worker = SimpleNamespace(
            tp_group=SimpleNamespace(rank=rank, ranks=[0, 1]), tp_cpu_group=None
        )
        self._batching_max_size = 3
        self._batching_delay_s = 0.05
        self._batch_admission = SimpleNamespace(
            batch_is_full=lambda rs: False,
            reject_reason_for_candidate=lambda rs, r: "capacity" if r.reject else None,
        )
        self.metrics = None

    def _dynamic_batching_enabled(self):
        return True

    def _can_dynamic_batch(self, a, b):
        return a.signature == b.signature


@pytest.fixture(autouse=True)
def request_type(monkeypatch):
    monkeypatch.setattr(module, "Req", Request)
    monkeypatch.setattr(module.time, "monotonic", lambda: 10.0)


def test_original_arrival_beats_requeue_position_and_checks_admission():
    old, young, incompatible, rejected = [
        Request(n) for n in ["old", "young", "other", "reject"]
    ]
    old._pi05_arrival = (1.0, 0)
    old.extra["vla"]["pi05_flow"] = SimpleNamespace(remaining=2)
    incompatible.signature = "b"
    rejected.reject = True
    scheduler = Scheduler(
        [(young, 8.0), (incompatible, 9.0), (rejected, 9.1), (old, 1.0)]
    )
    items = scheduler._get_next_pi05_segment_batch()
    assert [r.request_id for _, r in items] == ["old", "young"]
    assert [r.request_id for _, r, _ in scheduler.waiting_queue] == ["other", "reject"]
    assert old._pi05_arrival == (1.0, 0)


def test_fresh_wait_but_continuation_bypasses_batch_delay():
    req = Request("fresh")
    scheduler = Scheduler([(req, 9.99)])
    assert scheduler._get_next_pi05_segment_batch() is None
    req.extra["vla"]["pi05_flow"] = SimpleNamespace(remaining=2)
    assert scheduler._get_next_pi05_segment_batch() == [(b"client", req)]


def test_control_is_a_batching_barrier():
    old, new = Request("old"), Request("new")
    control = SimpleNamespace(request_id="update_weights")
    scheduler = Scheduler([(old, 1.0), (control, 2.0), (new, 3.0)])
    assert scheduler._get_next_pi05_segment_batch() == [(b"client", old)]
    assert scheduler._get_next_pi05_segment_batch() == [(b"client", control)]
    assert scheduler._get_next_pi05_segment_batch() == [(b"client", new)]


def test_tp_uses_rank_zero_packet_and_rejects_queue_mismatch(monkeypatch):
    packet = []

    def broadcast(value, rank, group, src):
        if rank == 0:
            packet[:] = [value]
        return packet[0]

    monkeypatch.setattr(module, "broadcast_pyobj", broadcast)
    a = Scheduler([(Request("a"), 1.0), (Request("b"), 2.0)], rank=0)
    b = Scheduler([(Request("a"), 99.0), (Request("b"), 100.0)], rank=1)
    a.server_args.tp_size = b.server_args.tp_size = 2
    ai, bi = a._get_next_pi05_segment_batch(), b._get_next_pi05_segment_batch()
    assert [r.request_id for _, r in ai] == [r.request_id for _, r in bi]
    assert [r._pi05_arrival for _, r in ai] == [r._pi05_arrival for _, r in bi]
    bad = Scheduler([(Request("changed"), 1.0)], rank=1)
    bad.server_args.tp_size = 2
    with pytest.raises(RuntimeError, match="differs"):
        bad._get_next_pi05_segment_batch()


def test_retire_refill_and_only_return_completed_actions():
    a, b = Request("a", 3), Request("b", 5)
    scheduler = Scheduler([(a, 1.0), (b, 2.0)])
    items = scheduler._get_next_pi05_segment_batch()

    def dispatch(items):
        outputs = []
        for _, req in items:
            state = req.extra["vla"]
            flow = state.setdefault(
                "pi05_flow",
                SimpleNamespace(
                    num_steps=req.num_inference_steps,
                    steps_done=0,
                    remaining=req.num_inference_steps,
                ),
            )
            flow.steps_done += state["pi05_segment_length"]
            flow.remaining = flow.num_steps - flow.steps_done
            outputs.append(
                module.OutputBatch(output=[None if flow.remaining else "actions"])
            )
        return outputs

    scheduler._dispatch_items = dispatch
    completed = scheduler._run_pi05_segment(items)
    assert [r.request_id for (_, r), _ in completed] == ["a"]
    assert completed[0][1].output == ["actions"]
    assert "pi05_flow" not in a.extra["vla"]
    assert b.extra["vla"]["pi05_flow"].remaining == 2
    assert scheduler.waiting_queue[0][2] == 2.0
    c = Request("c", 4)
    scheduler.waiting_queue.append((b"client", c, 9.0))
    completed = scheduler._run_pi05_segment(scheduler._get_next_pi05_segment_batch())
    assert [r.request_id for (_, r), _ in completed] == ["b"]
    assert c.extra["vla"]["pi05_flow"].remaining == 2


@pytest.mark.parametrize(
    "failure", ["exception", "missing_output", "missing_state", "changed_budget"]
)
def test_error_terminates_request_and_releases_state(failure):
    req = Request("a", 3)
    scheduler = Scheduler([(req, 1.0)])
    items = scheduler._get_next_pi05_segment_batch()

    def dispatch(items):
        if failure == "exception":
            raise RuntimeError("device failed")
        if failure == "missing_output":
            return []
        if failure == "changed_budget":
            req.extra["vla"]["pi05_flow"] = SimpleNamespace(
                num_steps=4, steps_done=3, remaining=1
            )
        return [module.OutputBatch(output=[None])]

    scheduler._dispatch_items = dispatch
    completed = scheduler._run_pi05_segment(items)
    assert len(completed) == 1 and completed[0][1].error
    assert not scheduler.waiting_queue
    assert "pi05_flow" not in req.extra["vla"]
    assert "pi05_segment_length" not in req.extra["vla"]


def test_other_pipeline_is_disabled():
    scheduler = Scheduler([])
    scheduler.server_args.pipeline_config = SimpleNamespace()
    assert scheduler._pi05_segments_enabled() is False


def test_sampling_signature_changes_only_for_opted_in_pi05_requests():
    from dataclasses import dataclass

    from sglang.multimodal_gen.runtime.managers.scheduler import (
        Scheduler as ProductionScheduler,
    )

    @dataclass
    class Params:
        num_inference_steps: int
        action_horizon: int = 50

    scheduler = ProductionScheduler.__new__(ProductionScheduler)
    cfg = SimpleNamespace(
        enable_segmented_actions=False, supports_sequential_dit_inference=lambda: False
    )
    scheduler.server_args = SimpleNamespace(pipeline_config=cfg)
    a, b = Request("a", 3), Request("b", 5)
    a.sampling_params, b.sampling_params = Params(3), Params(5)
    signature = scheduler._sampling_param_signature_items
    assert signature(a) != signature(b)
    cfg.enable_segmented_actions = True
    assert signature(a) == signature(b)
    b.sampling_params.action_horizon = 25
    assert signature(a) != signature(b)
    a.is_warmup = True
    assert ("num_inference_steps", 3) in signature(a)


def test_disabled_scheduler_keeps_original_dispatch():
    from sglang.multimodal_gen.runtime.managers.scheduler import (
        Scheduler as ProductionScheduler,
    )

    scheduler = ProductionScheduler.__new__(ProductionScheduler)
    scheduler.server_args = SimpleNamespace(
        pipeline_config=SimpleNamespace(enable_segmented_actions=False)
    )
    req = Request("native")
    scheduler.waiting_queue = deque([(b"client", req, 1.0)])
    scheduler._dynamic_batching_enabled = lambda: False
    scheduler._record_batch_dispatch_metrics = lambda **kwargs: None
    assert scheduler.get_next_batch_to_run() == [(b"client", req)]
    assert not scheduler.waiting_queue


@pytest.mark.parametrize(
    "field,value",
    [
        ("realtime_session_id", "session"),
        ("session", object()),
        ("num_outputs_per_prompt", 2),
    ],
)
def test_unsupported_stateful_or_multioutput_requests_use_ordinary_path(field, value):
    scheduler = Scheduler([])
    req = Request("a")
    setattr(req, field, value)
    assert scheduler._is_pi05_segment_request(req) is False
