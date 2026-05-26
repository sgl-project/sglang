"""Edge cases — abort × chunked.

Verifies that aborting a chunked-resume
request (1) releases all owned resources (row, KV, lock_ref), (2)
does not double-finalize, and (3) does not poison other reqs in
flight.

Many of these scripts read ``r.kv_pages`` / ``r.pending_middle_outputs``
which are whitebox views into Req state — necessary to verify
resource ownership invariants. The wishlist (§4 P1 (7)(8)) calls them
out as deliberate whitebox API.
"""

import unittest

from sglang.test.scripted_runtime.entrypoint import execute_scripted_runtime
from sglang.test.scripted_runtime.runtime import ScriptedRuntime
from sglang.test.scripted_runtime_chunked_helpers import (
    DEFAULT_CHUNK_SIZE,
    VERY_LONG_PROMPT_LEN,
    base_engine_kwargs,
    run_until,
    run_until_finished,
)
from sglang.test.test_utils import CustomTestCase


# abort a chunked-resume req while it's still in waiting_queue
# (chunked-resume parked between chunks). Resources must release.
def _script_abort_waiting_chunked_resume(t: ScriptedRuntime):
    r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
    # Push it into chunked-resume parked-in-waiting state.
    yield from run_until(r, lambda h: h.is_chunking)
    yield from run_until(r, lambda h: h.status == "waiting" and h.chunks_done >= 1)

    pages_before = r.kv_pages
    assert pages_before > 0, "chunked req should own KV pages mid-chunk"

    t.abort(r)
    yield  # let scheduler process the abort

    assert r.status in (
        "finished",
        "unknown",
    ), f"after abort r should be finished/unknown, got {r.status}"
    assert (
        r.kv_pages == 0
    ), f"abort must release KV; r.kv_pages={r.kv_pages} after abort"
    assert (
        r.row_idx is None
    ), f"abort must release row; r.row_idx={r.row_idx} after abort"
    assert r.lock_refs == 0, f"abort must release lock_refs; r.lock_refs={r.lock_refs}"


# abort during chunk_0 (first chunk not yet flushed).
def _script_abort_at_chunk_0(t: ScriptedRuntime):
    r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
    yield  # request observed by scheduler; first chunk starts
    yield from run_until(r, lambda h: h.is_chunking)

    t.abort(r)
    yield

    assert r.kv_pages == 0
    assert r.row_idx is None


# abort at chunk_mid (some chunks done, some pending).
def _script_abort_at_chunk_mid(t: ScriptedRuntime):
    r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
    yield from run_until(r, lambda h: h.chunks_done >= 2 and h.is_chunking)

    t.abort(r)
    yield

    assert r.kv_pages == 0


# abort at last_chunk (pending_middle_outputs has been ++ for
# the last admission; abort must zero it so Stage A doesn't revive
# the released row). Tied to scheduler.py:3567-3569 defensive cleanup.
def _script_abort_at_last_chunk(t: ScriptedRuntime):
    # Use a prompt that's exactly 2 chunks so the second chunk is the
    # last one.
    r = t.start_req(prompt_len=2 * DEFAULT_CHUNK_SIZE, max_new_tokens=4)
    yield from run_until(r, lambda h: h.chunks_done >= 1 and h.is_chunking)

    # pending_middle_outputs is incremented when the last chunk is
    # admitted into the running batch (b3a7b9f2a1).
    assert r.pending_middle_outputs > 0

    t.abort(r)
    yield

    # After abort, the defensive cleanup should have zeroed it.
    assert r.pending_middle_outputs == 0, (
        f"abort must zero pending_middle_outputs to prevent revival; "
        f"got {r.pending_middle_outputs}"
    )


# abort at last_chunk_in_flight under PP (forward still in
# flight when the abort hits). Verifies the PP cross-microbatch
# dedup (b823c16e60) and double-finalize guard (02b1785f0a).
#
# Requires multi-rank ScriptedRuntime + PP=2 (wishlist §4 P3 (15)).
def _script_abort_at_last_chunk_in_flight_pp(t: ScriptedRuntime):
    r = t.start_req(prompt_len=2 * DEFAULT_CHUNK_SIZE, max_new_tokens=4)
    # Drive to last_chunk_in_flight via batch_composition introspection
    # — when ``mb='b'`` has the chunked extend and ``mb='a'`` has the
    # last-chunk forward still running.
    yield from run_until(
        r,
        lambda h: h.chunks_done >= 1 and h.is_chunking,
    )

    t.abort(r)
    yield

    assert r.kv_pages == 0
    # No double-finalize: the result_queue should have one finished
    # event for r, not two.
    assert (
        r.finish_event_count == 1
    ), f"abort must not double-finalize; got {r.finish_event_count} events"


# abort one chunked req does not disturb another in flight.
def _script_abort_one_does_not_disturb_other(t: ScriptedRuntime):
    r1 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
    r2 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
    yield from run_until(r1, lambda h: h.is_chunking)

    t.abort(r1)
    yield

    assert r1.kv_pages == 0
    # r2 should make forward progress after r1 is gone.
    yield from run_until_finished(r2)
    assert r2.finished, "r2 should still complete after r1 is aborted"


class TestEdgeAbort(CustomTestCase):
    def test_abort_waiting_chunked_resume(self):
        execute_scripted_runtime(
            _script_abort_waiting_chunked_resume,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_abort_at_chunk_0(self):
        execute_scripted_runtime(
            _script_abort_at_chunk_0,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_abort_at_chunk_mid(self):
        execute_scripted_runtime(
            _script_abort_at_chunk_mid,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_abort_at_last_chunk(self):
        execute_scripted_runtime(
            _script_abort_at_last_chunk,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_abort_at_last_chunk_in_flight_pp(self):
        execute_scripted_runtime(
            _script_abort_at_last_chunk_in_flight_pp,
            **base_engine_kwargs(
                chunked_prefill_size=DEFAULT_CHUNK_SIZE,
                tp_size=2,
                pp_size=2,
            ),
        )

    def test_abort_one_does_not_disturb_other(self):
        execute_scripted_runtime(
            _script_abort_one_does_not_disturb_other,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )


if __name__ == "__main__":
    unittest.main()
