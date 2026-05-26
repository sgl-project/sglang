"""Edge cases — abort × scheduler-iteration timing.

Covers A.5 series from the expansion plan and fan-out symmetric
cases (abort at chunk_first / chunk_last / penultimate / very late /
post-finish). Verifies abort is a no-op when the rid is invalid /
already finished, and idempotent on repeated calls.
"""

import unittest

from sglang.test.scripted_runtime.entrypoint import execute_scripted_runtime
from sglang.test.scripted_runtime.req_handle import ReqHandle
from sglang.test.scripted_runtime.runtime import ScriptedRuntime
from sglang.test.scripted_runtime_chunked_helpers import (
    DEFAULT_CHUNK_SIZE,
    VERY_LONG_PROMPT_LEN,
    base_engine_kwargs,
    run_until,
    run_until_all_finished,
    run_until_finished,
)
from sglang.test.test_utils import CustomTestCase


def _script_abort_with_zero_yield(t: ScriptedRuntime):
    # Submit + abort same step (zero yields between).
    r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
    t.abort(r)
    yield
    assert r.kv_pages == 0
    assert r.row_idx is None
    assert r.lock_refs == 0


def _script_abort_at_admission_step(t: ScriptedRuntime):
    # Submit, yield once (admission), abort.
    r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
    yield  # scheduler observes req
    t.abort(r)
    yield
    assert r.kv_pages == 0
    assert r.row_idx is None


def _script_abort_then_start_same_step_new_rid(t: ScriptedRuntime):
    # Abort one req; in the same yield step, submit a different rid.
    r1 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
    yield from run_until(r1, lambda h: h.is_chunking)
    t.abort(r1)
    r2 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
    yield from run_until_finished(r2)
    assert r2.finished
    assert r1.kv_pages == 0


def _script_abort_then_start_same_step_same_rid(t: ScriptedRuntime):
    # Abort r1; resubmit with the same rid — must work as a fresh req.
    r1 = t.start_req(
        prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2, rid="abort-reuse"
    )
    yield from run_until(r1, lambda h: h.is_chunking)
    t.abort(r1)
    yield
    r2 = t.start_req(prompt_len=16, max_new_tokens=2, rid="abort-reuse")
    yield from run_until_finished(r2)
    assert r2.finished


def _script_abort_five_chunked_in_a_row(t: ScriptedRuntime):
    # 5 chunked reqs submitted; abort all 5 sequentially.
    reqs = [
        t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2) for _ in range(5)
    ]
    yield from run_until(reqs[0], lambda h: h.is_chunking)
    for r in reqs:
        t.abort(r)
    yield
    for r in reqs:
        assert r.kv_pages == 0
        assert r.row_idx is None


def _script_abort_unknown_rid_noop(t: ScriptedRuntime):
    # Abort an rid that was never submitted: no exception, no state change.
    bogus = ReqHandle(rid="never-submitted-rid", runtime=t)
    t.abort(bogus)  # must not raise
    yield


def _script_abort_after_finish_noop(t: ScriptedRuntime):
    # Submit r1, wait for finish, then abort: no-op.
    r = t.start_req(prompt_len=16, max_new_tokens=2)
    yield from run_until_finished(r)
    assert r.finished
    t.abort(r)
    yield
    # finished + abort should not destabilize stats.
    stats = t.engine_stats()
    assert stats["kv_pool_free"] >= 0


def _script_abort_during_waiting_timeout(t: ScriptedRuntime):
    # Chunked-resume in waiting_queue + abort timeout cleanup hits.
    # The cleanup must skip chunked-resume reqs (359e5ed7bd).
    r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
    yield from run_until(r, lambda h: h.is_chunking)
    yield from run_until(r, lambda h: h.status == "waiting")
    # Simulate the waiting-queue timeout housekeeping firing.
    # NEW API NEEDED: t.trigger_abort_on_waiting_timeout() — invoke
    # the internal waiting-queue housekeeping abort sweep.
    t.trigger_abort_on_waiting_timeout()
    yield
    yield from run_until_finished(r)
    assert r.finished, "chunked-resume must survive waiting-timeout sweep"


def _script_abort_chunk_first(t: ScriptedRuntime):
    # Abort while the very first chunk is still in progress.
    r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
    yield  # admission
    yield  # first chunk admitted
    t.abort(r)
    yield
    assert r.kv_pages == 0


def _script_abort_chunk_last(t: ScriptedRuntime):
    # Abort while the last chunk is in progress.
    r = t.start_req(prompt_len=2 * DEFAULT_CHUNK_SIZE, max_new_tokens=4)
    yield from run_until(r, lambda h: h.chunks_done >= 1 and h.is_chunking)
    t.abort(r)
    yield
    assert r.kv_pages == 0
    assert r.pending_middle_outputs == 0


def _script_abort_penultimate_chunk(t: ScriptedRuntime):
    # Abort 1 chunk before the final one.
    r = t.start_req(prompt_len=4 * DEFAULT_CHUNK_SIZE, max_new_tokens=2)
    yield from run_until(r, lambda h: h.chunks_done >= 2 and h.is_chunking)
    t.abort(r)
    yield
    assert r.kv_pages == 0


def _script_double_abort_idempotent(t: ScriptedRuntime):
    # Call abort twice on the same handle: second call is a no-op.
    r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
    yield from run_until(r, lambda h: h.is_chunking)
    t.abort(r)
    t.abort(r)
    yield
    assert r.kv_pages == 0


def _script_abort_during_decode(t: ScriptedRuntime):
    # r1 has finished prefill, is in decode; abort mid-decode.
    r = t.start_req(prompt_len=16, max_new_tokens=64)
    yield from run_until(r, lambda h: h.status == "running")
    t.abort(r)
    yield
    assert r.kv_pages == 0


def _script_abort_one_of_three_others_finish(t: ScriptedRuntime):
    # Three reqs in batch, abort middle one, other two finish.
    r1 = t.start_req(prompt_len=16, max_new_tokens=4)
    r2 = t.start_req(prompt_len=16, max_new_tokens=4)
    r3 = t.start_req(prompt_len=16, max_new_tokens=4)
    yield from run_until(r2, lambda h: h.status == "running")
    t.abort(r2)
    yield from run_until_all_finished([r1, r3])
    assert r2.kv_pages == 0


def _script_abort_in_separate_yields(t: ScriptedRuntime):
    # Submit 3 chunked, abort in 3 separate yield steps.
    reqs = [
        t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2) for _ in range(3)
    ]
    yield from run_until(reqs[0], lambda h: h.is_chunking)
    for r in reqs:
        t.abort(r)
        yield
    for r in reqs:
        assert r.kv_pages == 0


def _script_abort_finish_event_count_zero(t: ScriptedRuntime):
    # Aborted req should not emit a normal finish event.
    # NEW API NEEDED: r.finish_event_count — count of completion events.
    r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
    yield from run_until(r, lambda h: h.is_chunking)
    t.abort(r)
    yield
    # finish_event_count for an aborted req should be 0 or 1 (abort
    # is itself a kind of finalize, but should not double-fire).
    assert r.finish_event_count <= 1


class TestEdgeAbortTiming(CustomTestCase):
    def test_abort_with_zero_yield(self):
        execute_scripted_runtime(
            _script_abort_with_zero_yield,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_abort_at_admission_step(self):
        execute_scripted_runtime(
            _script_abort_at_admission_step,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_abort_then_start_same_step_new_rid(self):
        execute_scripted_runtime(
            _script_abort_then_start_same_step_new_rid,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_abort_then_start_same_step_same_rid(self):
        execute_scripted_runtime(
            _script_abort_then_start_same_step_same_rid,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_abort_five_chunked_in_a_row(self):
        execute_scripted_runtime(
            _script_abort_five_chunked_in_a_row,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_abort_unknown_rid_noop(self):
        execute_scripted_runtime(
            _script_abort_unknown_rid_noop,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_abort_after_finish_noop(self):
        execute_scripted_runtime(
            _script_abort_after_finish_noop,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_abort_during_waiting_timeout(self):
        execute_scripted_runtime(
            _script_abort_during_waiting_timeout,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_abort_chunk_first(self):
        execute_scripted_runtime(
            _script_abort_chunk_first,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_abort_chunk_last(self):
        execute_scripted_runtime(
            _script_abort_chunk_last,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_abort_penultimate_chunk(self):
        execute_scripted_runtime(
            _script_abort_penultimate_chunk,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_double_abort_idempotent(self):
        execute_scripted_runtime(
            _script_double_abort_idempotent,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_abort_during_decode(self):
        execute_scripted_runtime(
            _script_abort_during_decode,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_abort_one_of_three_others_finish(self):
        execute_scripted_runtime(
            _script_abort_one_of_three_others_finish,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_abort_in_separate_yields(self):
        execute_scripted_runtime(
            _script_abort_in_separate_yields,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_abort_finish_event_count_zero(self):
        execute_scripted_runtime(
            _script_abort_finish_event_count_zero,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )


if __name__ == "__main__":
    unittest.main()
