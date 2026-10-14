"""Real cancellation entry methods with controlled CPU-only ownership.

No CUDA or RDMA is simulated as hardware evidence. The sender's quarantine
check, scheduler abort, and result retirement run from their actual source.
SGLANG_COMPRESSION_STANDALONE_TEST=1 bypasses serving package initializers.
"""

# ruff: noqa: E402
import ast
import logging
import os
import sys
import types
from collections import deque
from http import HTTPStatus
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import pytest

ROOT = Path(__file__).resolve().parents[4]
SRT = ROOT / "python/sglang/srt"
if os.environ.get("SGLANG_COMPRESSION_STANDALONE_TEST") == "1":
    for name, relative in (
        ("sglang", "python/sglang"),
        ("sglang.srt", "python/sglang/srt"),
    ):
        if name not in sys.modules:
            module = types.ModuleType(name)
            module.__path__ = [str(ROOT / relative)]
            sys.modules[name] = module

from sglang.srt.kv_compression.types import BufferDrainError
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def definition(file, name, owner=None, **globals):
    path = SRT / file
    nodes = ast.parse(path.read_text()).body
    if owner:
        nodes = next(
            x for x in nodes if isinstance(x, ast.ClassDef) and x.name == owner
        ).body
    node = next(x for x in nodes if getattr(x, "name", None) == name)
    node.decorator_list = []
    module = ast.parse("from __future__ import annotations")
    module.body.append(node)
    globals.setdefault("logger", logging.getLogger(__name__))
    exec(compile(module, str(path), "exec"), globals)
    return globals[name]


def prepare_abort(req, message, status_code=None):
    req.finished_reason = NS(message=message, status_code=status_code)


class Harness:
    abort_request = definition(
        "managers/scheduler.py",
        "abort_request",
        "Scheduler",
        DisaggregationMode=NS(PREFILL="prefill", DECODE="decode"),
        get_parallel=lambda: NS(pp_size=1),
        maybe_release_metadata_buffer=lambda req, alloc: alloc.free(
            req.metadata_buffer_index
        ),
        _make_abort_req=lambda *a, **kw: NS(),
        prepare_abort=prepare_abort,
        release_kv_cache=lambda req, tree, **kw: tree.release(req),
    )
    compression_failure_reason = definition(
        "managers/scheduler.py",
        "compression_failure_reason",
        "Scheduler",
        DisaggregationMode=NS(PREFILL="prefill", DECODE="decode"),
    )
    _retire_aborted_prefill_result = definition(
        "disaggregation/prefill.py",
        "_retire_aborted_prefill_result",
        "SchedulerDisaggregationPrefillMixin",
        BufferDrainError=BufferDrainError,
        CacheRequestOutcome=NS(ABORT="abort"),
        maybe_release_metadata_buffer=lambda req, alloc: alloc.free(
            req.metadata_buffer_index
        ),
        release_kv_cache=lambda req, tree, **kw: tree.release(req),
    )

    release_aborted_prefill_waiting_req = definition(
        "disaggregation/prefill.py",
        "release_aborted_prefill_waiting_req",
        "SchedulerDisaggregationPrefillMixin",
        maybe_release_metadata_buffer=lambda req, alloc: alloc.free(
            req.metadata_buffer_index
        ),
        release_kv_cache=lambda req, tree, **kw: tree.release(req),
    )

    def __init__(self, req, queue):
        self.enable_lmcache = False
        self.chunked_req = self.mm_receiver = self.dllm_config = None
        self.waiting_queue = [req] if queue == "waiting" else []
        self.disagg_prefill_bootstrap_queue = NS(
            queue=[req] if queue == "bootstrap" else []
        )
        self.disagg_prefill_inflight_queue = [req] if queue == "inflight" else []
        self.disaggregation_mode = "prefill"
        self.tree_cache = NS(finish=Mock(), release=Mock())
        self._release_aborted_request = lambda req: self.tree_cache.finish(
            req.cache_request_handle, "abort"
        )
        self.req_to_metadata_buffer_idx_allocator = NS(free=Mock())
        self.beam_coordinator = NS(retire_group=Mock())
        self.ipc_channels = NS(send_to_tokenizer=NS(send_output=Mock()))
        self.grammar_manager = NS(abort_requests=Mock())
        self.collect_inflight_reqs = lambda: []
        self.clear_pending_chunk_send = Mock()
        self.output_streamer = NS(stream_output=Mock())


# Bind new methods when present, leaving the baseline entry methods untouched
# for the red run. Missing methods cannot make the fixed tests pass.
for name in (
    "_abort_prefill_sender",
    "_quarantine_prefill_abort",
    "_is_prefill_abort_quarantined",
):
    tree = ast.parse((SRT / "disaggregation/prefill.py").read_text())
    if any(isinstance(x, ast.FunctionDef) and x.name == name for x in ast.walk(tree)):
        setattr(
            Harness,
            name,
            definition(
                "disaggregation/prefill.py",
                name,
                "SchedulerDisaggregationPrefillMixin",
                BufferDrainError=BufferDrainError,
                prepare_abort=prepare_abort,
                HTTPStatus=HTTPStatus,
            ),
        )

Harness._reject_quarantined_waiting = definition(
    "managers/scheduler.py",
    "_reject_quarantined_waiting",
    "Scheduler",
    prepare_abort=prepare_abort,
    HTTPStatus=HTTPStatus,
)
Harness.is_fully_idle = definition(
    "managers/scheduler.py",
    "is_fully_idle",
    "Scheduler",
    DisaggregationMode=NS(PREFILL="prefill", DECODE="decode"),
)
Harness.maybe_send_health_check_signal = definition(
    "managers/scheduler.py",
    "maybe_send_health_check_signal",
    "Scheduler",
    HealthCheckOutput=NS,
)
Harness.add_one_req = definition(
    "managers/schedule_policy.py",
    "add_one_req",
    "PrefillAdder",
    AddReqResult=NS(OTHER="rejected"),
)
POLL = NS(Success=1, Failed=2, WaitingForInput=3, Transferring=4)
Harness.process_disagg_prefill_inflight_queue = definition(
    "disaggregation/prefill.py",
    "process_disagg_prefill_inflight_queue",
    "SchedulerDisaggregationPrefillMixin",
    KVPoll=POLL,
    poll_and_all_reduce_attn_cp_tp_group=lambda *args: [POLL.Success],
)


def request(*, uncertain=True):
    r = NS(
        rid="target",
        bootstrap_room=77,
        pending_bootstrap=True,
        cache_request_handle=object(),
        metadata_buffer_index=9,
        kv=NS(holds_kv=True, holds_mamba=True),
        to_finish=None,
        finished_reason=None,
        return_logprob=False,
    )
    r.finished = lambda: r.finished_reason is not None
    r.update_finish_state = lambda: setattr(r, "finished_reason", r.to_finish)
    if uncertain:
        sender = NS(bootstrap_room=77, kv_mgr=NS(compression_uncertain_rooms={77}))
        abort = definition(
            "disaggregation/mooncake/conn.py",
            "abort",
            "MooncakeKVSender",
            BufferDrainError=BufferDrainError,
        )
        sender.abort = Mock(side_effect=lambda: abort(sender))
        r.disagg_kv_sender = sender
    else:
        r.disagg_kv_sender = NS(abort=Mock())
    return r


def assert_owned(s, r):
    s.tree_cache.finish.assert_not_called()
    s.tree_cache.release.assert_not_called()
    s.req_to_metadata_buffer_idx_allocator.free.assert_not_called()
    assert r.metadata_buffer_index == 9 and r.kv.holds_kv and r.kv.holds_mamba
    assert sum(held is r for held in s._compression_quarantined_waiting) == 1
    assert s.compression_failure_reason()


@pytest.mark.parametrize("queue", ["waiting", "bootstrap", "inflight"])
def test_uncertain_abort_preserves_owners_and_does_not_escape(queue):
    r = request()
    s = Harness(r, queue)
    for _ in range(3):
        s.abort_request(
            NS(rid=r.rid, abort_all=False, finished_reason=None, abort_message=None)
        )
    assert_owned(s, r)
    assert r.finished_reason.status_code == 503
    r.disagg_kv_sender.abort.assert_called_once()
    s.output_streamer.stream_output.assert_called_once_with([r], False)
    assert (
        r
        in s.waiting_queue
        + s.disagg_prefill_bootstrap_queue.queue
        + s.disagg_prefill_inflight_queue
    )


def test_uncertain_final_result_does_not_release_or_unpin():
    r = request()
    s = Harness(r, "result")
    for _ in range(3):
        assert s._retire_aborted_prefill_result(r) is False
    assert_owned(s, r)
    s.clear_pending_chunk_send.assert_not_called()
    r.disagg_kv_sender.abort.assert_called_once()
    s.output_streamer.stream_output.assert_called_once()


def test_existing_abort_reason_is_preserved():
    r = request()
    s = Harness(r, "inflight")
    reason = NS(status_code=400, message="existing rejection")
    r.finished_reason = reason
    s.abort_request(NS(rid=r.rid, abort_all=False))
    assert r.finished_reason is reason
    assert_owned(s, r)


def test_pending_abort_reason_is_preserved():
    r = request()
    s = Harness(r, "inflight")
    reason = NS(status_code=503, message="existing timeout")
    r.to_finish = reason
    s.abort_request(NS(rid=r.rid, abort_all=False))
    assert r.finished_reason is reason
    assert_owned(s, r)


@pytest.mark.parametrize("queue", ["waiting", "bootstrap", "inflight"])
def test_safe_abort_keeps_existing_retirement(queue):
    r = request(uncertain=False)
    s = Harness(r, queue)
    s.abort_request(
        NS(rid=r.rid, abort_all=False, finished_reason=None, abort_message=None)
    )
    r.disagg_kv_sender.abort.assert_called_once()
    assert not getattr(s, "_compression_quarantined_waiting", {})
    if queue == "waiting":
        assert s.waiting_queue == []
        s.tree_cache.finish.assert_called_once()
        s.req_to_metadata_buffer_idx_allocator.free.assert_called_once_with(9)
    elif queue == "bootstrap":
        s.tree_cache.finish.assert_called_once()
    else:
        s.tree_cache.finish.assert_not_called()


def test_notify_failure_keeps_existing_final_result_cleanup():
    r = request(uncertain=False)
    s = Harness(r, "result")
    r.disagg_kv_sender.abort.side_effect = RuntimeError(
        "notification failed after drain"
    )
    assert s._retire_aborted_prefill_result(r) is True
    assert not getattr(s, "_compression_quarantined_waiting", {})
    s.tree_cache.release.assert_called_once()
    s.req_to_metadata_buffer_idx_allocator.free.assert_called_once_with(9)
    s.tree_cache.finish.assert_called_once()


def test_non_drain_error_still_propagates_from_intake_abort():
    r = request(uncertain=False)
    s = Harness(r, "inflight")
    r.disagg_kv_sender.abort.side_effect = ValueError("unexpected programming error")
    with pytest.raises(ValueError):
        s.abort_request(NS(rid=r.rid, abort_all=False))
    assert not getattr(s, "_compression_quarantined_waiting", {})


def test_other_request_abort_does_not_borrow_quarantined_ownership():
    r = request()
    s = Harness(r, "inflight")
    s.abort_request(NS(rid="another", abort_all=False))
    r.disagg_kv_sender.abort.assert_not_called()
    assert not getattr(s, "_compression_quarantined_waiting", {})


@pytest.mark.parametrize("poll", [POLL.Success, POLL.Failed])
def test_late_terminal_poll_cannot_release_quarantined_request(poll):
    r = request()
    s = Harness(r, "inflight")
    s.abort_request(NS(rid=r.rid, abort_all=False))
    s.attn_cp_cpu_group = s.attn_tp_cpu_group = None
    fn = s.process_disagg_prefill_inflight_queue
    with patch.dict(
        fn.__func__.__globals__, poll_and_all_reduce_attn_cp_tp_group=lambda *a: [poll]
    ):
        assert fn() == []
    assert s.disagg_prefill_inflight_queue == [r]
    assert_owned(s, r)


def test_waiting_sweep_does_not_duplicate_terminal_output_or_clear_isolation():
    r = request()
    s = Harness(r, "waiting")
    s.abort_request(NS(rid=r.rid, abort_all=False))
    reason = r.finished_reason
    s._reject_quarantined_waiting("later quarantine observation")
    assert s.waiting_queue == [] and r.finished_reason is reason
    s.output_streamer.stream_output.assert_called_once()
    assert_owned(s, r)


def test_quarantine_blocks_admission_health_and_false_idle():
    r = request()
    s = Harness(r, "inflight")
    s.abort_request(NS(rid=r.rid, abort_all=False))
    other = NS(set_finish_with_abort=Mock())
    s._admission_failure_reason = s.compression_failure_reason
    assert s.add_one_req(other, False, None) == "rejected"
    assert other.set_finish_with_abort.call_args.kwargs["status_code"] == 503
    s.return_health_check_ipcs = deque([object()])
    s.maybe_send_health_check_signal()
    assert s.ipc_channels.send_to_tokenizer.send_output.call_count == 2
    assert all(
        call.args[0].quarantine_reason
        for call in s.ipc_channels.send_to_tokenizer.send_output.call_args_list
    )
    s.disagg_prefill_inflight_queue = []
    s.running_batch = NS(is_empty=lambda: True)
    s.last_batch = None
    s.dllm_manager = NS(any_staging_reqs=lambda: False)
    s.enable_overlap = s.enable_hisparse = s.enable_hierarchical_cache = False
    s._pp_microbatches_drained = lambda: True
    s.grammar_manager.grammar_queue = []
    assert s.is_fully_idle() is False
    assert_owned(s, r)


def test_quarantined_ownership_uses_object_identity_for_reused_rid():
    first = request()
    second = request()
    second.metadata_buffer_index = 10
    s = Harness(first, "inflight")
    s.disagg_prefill_inflight_queue.append(second)
    s.abort_request(NS(rid=first.rid, abort_all=False))
    assert len(s._compression_quarantined_waiting) == 2
    assert any(r is first for r in s._compression_quarantined_waiting)
    assert any(r is second for r in s._compression_quarantined_waiting)
    s.tree_cache.finish.assert_not_called()
    s.req_to_metadata_buffer_idx_allocator.free.assert_not_called()


@pytest.mark.parametrize("queue", ["waiting", "bootstrap"])
def test_bootstrap_failure_cannot_retire_undrained_sender(queue):
    r = request()
    s = Harness(r, queue)
    r.disagg_kv_sender.failure_exception = Mock(
        side_effect=BufferDrainError("undrained")
    )
    fn = definition(
        "disaggregation/prefill.py",
        "handle_bootstrap_failure",
        "SchedulerDisaggregationPrefillMixin",
        BufferDrainError=BufferDrainError,
        get_parallel=lambda: NS(tp_rank=0),
    )
    fn(s, r)
    assert_owned(s, r)
    s.tree_cache.release.assert_not_called()
    s.req_to_metadata_buffer_idx_allocator.free.assert_not_called()
    s.output_streamer.stream_output.assert_called_once()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
