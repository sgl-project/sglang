"""A PD decode retraction must back up state at the same step as output_ids."""

from collections import deque
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

import sglang.srt.managers.scheduler as scheduler_module  # noqa: E402
from sglang.srt.disaggregation.utils import DisaggregationMode  # noqa: E402
from sglang.srt.managers.scheduler import Scheduler  # noqa: E402

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

# One speculative verify accepts this many tokens.
ACCEPTED = 4


class _Req:
    def __init__(self, rid):
        self.rid = rid
        self.output_ids = list(range(10))
        # Tokens the device state (KV, draft, recurrent slot) has consumed.
        self.device_step = len(self.output_ids)
        self.is_retracted = False
        self.done = False
        self.backup = None

    def finished(self):
        return self.done


class _Batch:
    def __init__(self, reqs, has_room):
        self.reqs = list(reqs)
        self.has_room = has_room
        self.batch_is_full = True
        self.prepared = False

    def batch_size(self):
        return len(self.reqs)

    def is_empty(self):
        return not self.reqs

    def filter_batch(self, reqs_to_exclude=None):
        excluded = set(reqs_to_exclude or ())
        self.reqs = [r for r in self.reqs if not r.finished() and r not in excluded]

    def check_decode_mem(self):
        return self.has_room

    def retract_decode(self):
        req = self.reqs.pop()
        # Offload: the backup carries the device state as it is now.
        req.backup = (len(req.output_ids), req.device_step)
        req.is_retracted = True
        return [req], 0.5, []

    def prepare_for_decode(self):
        self.prepared = True


def _scheduler(
    reqs,
    *,
    has_room,
    mode=DisaggregationMode.DECODE,
    overlap=True,
    finish_rids=(),
    frees_memory=False,
):
    batch = _Batch(reqs, has_room)
    events = []
    # The in-flight verify has already advanced the device state.
    for req in reqs:
        req.device_step += ACCEPTED
    inflight = SimpleNamespace(reqs=list(reqs))

    def process_batch_result(pending_batch, result):
        events.append("settle")
        for req in pending_batch.reqs:
            # Result processing skips retracted requests.
            if req.is_retracted:
                continue
            req.output_ids.extend(range(ACCEPTED))
            req.done = req.rid in finish_rids
        if frees_memory:
            batch.has_room = True

    s = Scheduler.__new__(Scheduler)
    vars(s).update(
        disaggregation_mode=mode,
        enable_overlap=overlap,
        enable_skip_finishing_decode=False,
        result_queue=deque([(inflight, object())]),
        last_batch=inflight,
        process_batch_result=process_batch_result,
        forward_ct=1,
        decode_offload_manager=None,
        token_to_kv_pool_allocator=SimpleNamespace(available_size=lambda: 0),
        tree_cache=SimpleNamespace(
            token_to_kv_pool_allocator=SimpleNamespace(state_slots=None)
        ),
        new_token_ratio_tracker=SimpleNamespace(
            current=0.5, decay_step=lambda: events.append("decay")
        ),
        metrics_reporter=SimpleNamespace(enable_metrics=False, num_retracted_reqs=0),
        _add_request_to_queue=lambda req, is_retracted: events.append("requeue"),
    )
    return s, batch, events


@pytest.mark.parametrize("forced", [False, True])
def test_backup_matches_output_ids_after_an_inflight_verify(forced):
    reqs = [_Req("a"), _Req("b")]
    s, batch, events = _scheduler(reqs, has_room=forced)
    with (
        patch.object(scheduler_module, "TEST_RETRACT", forced),
        patch.object(scheduler_module, "TEST_RETRACT_INTERVAL", 1),
    ):
        Scheduler.update_running_batch(s, batch)

    retracted = reqs[1]
    num_output_ids, device_step = retracted.backup
    # Resuming feeds output_ids[-1] again; the backed-up state must not have
    # consumed tokens beyond the ones output_ids records.
    assert device_step == num_output_ids
    assert events.index("settle") < events.index("requeue")
    assert not s.result_queue
    assert s.last_batch is None


@pytest.mark.parametrize("finished", [0, 1, 2])
def test_settled_completions_leave_the_batch_and_free_memory(finished):
    reqs = [_Req("a"), _Req("b")]
    s, batch, events = _scheduler(
        reqs,
        has_room=False,
        finish_rids=[r.rid for r in reqs[:finished]],
        frees_memory=True,
    )
    with patch.object(scheduler_module, "TEST_RETRACT", False):
        out = Scheduler.update_running_batch(s, batch)

    assert out is batch
    assert [r.rid for r in batch.reqs] == [r.rid for r in reqs[finished:]]
    assert all(r.backup is None for r in reqs)
    assert "requeue" not in events
    assert batch.prepared == (finished < 2)
    if finished:
        assert batch.batch_is_full is False


@pytest.mark.parametrize(
    "mode,overlap,has_room",
    [
        (DisaggregationMode.DECODE, True, True),
        (DisaggregationMode.NULL, True, False),
        (DisaggregationMode.DECODE, False, False),
    ],
)
def test_other_steps_keep_the_result_in_flight(mode, overlap, has_room):
    reqs = [_Req("a"), _Req("b")]
    s, batch, events = _scheduler(reqs, has_room=has_room, mode=mode, overlap=overlap)
    with patch.object(scheduler_module, "TEST_RETRACT", False):
        Scheduler.update_running_batch(s, batch)

    assert "settle" not in events
    assert len(s.result_queue) == 1
    assert s.last_batch is not None
