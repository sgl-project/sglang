"""Edge cases — multi-req submission timing.

Covers A.3 series from the expansion plan plus parametrised
stress / interleaving / rid-reuse scenarios. Verifies that
arbitrary submission timings (back-to-back, mid-yield, sustained
trickle) all reach a clean terminal state for every req.
"""

import unittest

from sglang.test.scripted_runtime.entrypoint import execute_scripted_runtime
from sglang.test.scripted_runtime.req_handle import ReqHandle
from sglang.test.scripted_runtime.runtime import ScriptedRuntime
from sglang.test.scripted_runtime_chunked_helpers import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_MAX_STEPS,
    VERY_LONG_PROMPT_LEN,
    base_engine_kwargs,
    run_until,
    run_until_all_finished,
    run_until_finished,
)
from sglang.test.test_utils import CustomTestCase


def _script_hundred_short_reqs(t: ScriptedRuntime):
    # 100 short reqs back-to-back: all complete, no leak.
    reqs = [t.start_req(prompt_len=16, max_new_tokens=2) for _ in range(100)]
    yield from run_until_all_finished(reqs, max_steps=2000)
    for r in reqs:
        assert r.finished


def _script_two_hundred_short_reqs(t: ScriptedRuntime):
    # 200 short reqs: stability check.
    reqs = [t.start_req(prompt_len=16, max_new_tokens=2) for _ in range(200)]
    yield from run_until_all_finished(reqs, max_steps=4000)
    for r in reqs:
        assert r.finished


def _script_five_hundred_short_reqs(t: ScriptedRuntime):
    # 500 short reqs: sustained pressure.
    reqs = [t.start_req(prompt_len=8, max_new_tokens=1) for _ in range(500)]
    yield from run_until_all_finished(reqs, max_steps=8000)
    for r in reqs:
        assert r.finished


def _script_mixed_ten_chunked_ten_short(t: ScriptedRuntime):
    # 10 chunked + 10 short, all submitted back-to-back. Single
    # chunked-in-flight invariant preserved at every step.
    chunked = [
        t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        for _ in range(10)
    ]
    shorts = [t.start_req(prompt_len=8, max_new_tokens=2) for _ in range(10)]
    all_reqs = chunked + shorts
    for _ in range(DEFAULT_MAX_STEPS * 20):
        assert t.chunked_in_flight_count() <= 1
        if all(r.finished for r in all_reqs):
            return
        yield
    raise AssertionError("not all reqs finished")


def _script_submit_during_chunk_mid(t: ScriptedRuntime):
    # r1 in mid-chunk; r2 submitted after 1 yield; r3 after another.
    # All three complete.
    r1 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
    yield from run_until(r1, lambda h: h.is_chunking)
    r2 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
    yield
    r3 = t.start_req(prompt_len=16, max_new_tokens=2)
    yield from run_until_all_finished([r1, r2, r3])


def _script_five_identical_prompts(t: ScriptedRuntime):
    # 5 identical prompts: r1 chunks; r2..r5 hit radix.
    r1 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
    yield from run_until_finished(r1)

    others = [
        t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        for _ in range(4)
    ]
    yield from run_until_all_finished(others)
    for r in others:
        assert r.finished
        # Cached prefix — should not re-chunk.
        assert r.chunks_done == 0


def _script_sibling_shared_prefix(t: ScriptedRuntime):
    # Two reqs share the first N tokens: each runs to completion.
    r1 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
    yield from run_until_finished(r1)
    r2 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN + 8, max_new_tokens=2)
    yield from run_until_finished(r2)
    assert r1.finished and r2.finished


def _script_trickle_per_yield_50(t: ScriptedRuntime):
    # Submit one new req per yield for 50 yields. All complete.
    reqs = []
    for _ in range(50):
        reqs.append(t.start_req(prompt_len=8, max_new_tokens=2))
        yield
    yield from run_until_all_finished(reqs, max_steps=2000)
    for r in reqs:
        assert r.finished


def _script_submit_then_immediate_abort(t: ScriptedRuntime):
    # start_req then abort in same yield step: clean state.
    r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
    t.abort(r)
    yield
    assert r.kv_pages == 0
    assert r.row_idx is None


def _script_rid_reuse_after_finish(t: ScriptedRuntime):
    # Submit r1, wait for finish, then submit r2 with same rid.
    # NEW API NEEDED: start_req(..., rid="...") — explicit rid control.
    r1 = t.start_req(prompt_len=16, max_new_tokens=2, rid="reuse-rid")
    yield from run_until_finished(r1)
    r2 = t.start_req(prompt_len=16, max_new_tokens=2, rid="reuse-rid")
    yield from run_until_finished(r2)
    assert r2.finished
    # r2 must be a distinct request lifecycle, not a resurrection of r1.


def _script_concurrent_short_and_long(t: ScriptedRuntime):
    # 5 short + 1 long, all concurrent; verify long does not starve.
    long = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=4)
    shorts = [t.start_req(prompt_len=16, max_new_tokens=2) for _ in range(5)]
    yield from run_until_all_finished([long] + shorts)


def _script_three_long_back_to_back(t: ScriptedRuntime):
    # Three long chunked reqs submitted back-to-back.
    reqs = [
        t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        for _ in range(3)
    ]
    yield from run_until_all_finished(reqs, max_steps=1500)
    for r in reqs:
        assert r.finished


def _script_alternating_long_short(t: ScriptedRuntime):
    # Alternate long / short submissions, 10 total.
    reqs = []
    for i in range(10):
        if i % 2 == 0:
            reqs.append(t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2))
        else:
            reqs.append(t.start_req(prompt_len=8, max_new_tokens=2))
    yield from run_until_all_finished(reqs, max_steps=2000)


def _script_burst_then_pause_then_burst(t: ScriptedRuntime):
    # Burst 10, wait, burst 10 more.
    first = [t.start_req(prompt_len=16, max_new_tokens=2) for _ in range(10)]
    yield from run_until_all_finished(first)
    second = [t.start_req(prompt_len=16, max_new_tokens=2) for _ in range(10)]
    yield from run_until_all_finished(second)


def _script_submit_pause_n_resubmit_same_rid(t: ScriptedRuntime):
    # Submit and complete r1, then 200 yields, then resubmit with same rid.
    r1 = t.start_req(prompt_len=16, max_new_tokens=2, rid="reuse-200")
    yield from run_until_finished(r1)
    for _ in range(200):
        yield
    r2 = t.start_req(prompt_len=16, max_new_tokens=2, rid="reuse-200")
    yield from run_until_finished(r2)
    assert r2.finished


def _script_submit_during_decode_of_other(t: ScriptedRuntime):
    # r1 in decode phase; submit r2 (chunked). Both complete.
    r1 = t.start_req(prompt_len=16, max_new_tokens=16)
    yield from run_until(r1, lambda h: h.status == "running")
    r2 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
    yield from run_until_all_finished([r1, r2])


def _script_unique_rids_distinct(t: ScriptedRuntime):
    # Many reqs with unique explicit rids.
    reqs = [
        t.start_req(prompt_len=16, max_new_tokens=2, rid=f"unique-{i}")
        for i in range(20)
    ]
    yield from run_until_all_finished(reqs)
    rids = {r.rid for r in reqs}
    assert len(rids) == 20


def _script_chunked_reqs_each_yield_15(t: ScriptedRuntime):
    # 15 chunked reqs trickled one per yield.
    reqs = []
    for _ in range(15):
        reqs.append(t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2))
        yield
    yield from run_until_all_finished(reqs, max_steps=3000)


class TestEdgeMultiReqTiming(CustomTestCase):
    def test_hundred_short_reqs(self):
        execute_scripted_runtime(
            _script_hundred_short_reqs,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_two_hundred_short_reqs(self):
        execute_scripted_runtime(
            _script_two_hundred_short_reqs,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_five_hundred_short_reqs(self):
        execute_scripted_runtime(
            _script_five_hundred_short_reqs,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_mixed_ten_chunked_ten_short(self):
        execute_scripted_runtime(
            _script_mixed_ten_chunked_ten_short,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_submit_during_chunk_mid(self):
        execute_scripted_runtime(
            _script_submit_during_chunk_mid,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_five_identical_prompts(self):
        execute_scripted_runtime(
            _script_five_identical_prompts,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_sibling_shared_prefix(self):
        execute_scripted_runtime(
            _script_sibling_shared_prefix,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_trickle_per_yield_50(self):
        execute_scripted_runtime(
            _script_trickle_per_yield_50,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_submit_then_immediate_abort(self):
        execute_scripted_runtime(
            _script_submit_then_immediate_abort,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_rid_reuse_after_finish(self):
        execute_scripted_runtime(
            _script_rid_reuse_after_finish,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_concurrent_short_and_long(self):
        execute_scripted_runtime(
            _script_concurrent_short_and_long,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_three_long_back_to_back(self):
        execute_scripted_runtime(
            _script_three_long_back_to_back,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_alternating_long_short(self):
        execute_scripted_runtime(
            _script_alternating_long_short,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_burst_then_pause_then_burst(self):
        execute_scripted_runtime(
            _script_burst_then_pause_then_burst,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_submit_pause_n_resubmit_same_rid(self):
        execute_scripted_runtime(
            _script_submit_pause_n_resubmit_same_rid,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_submit_during_decode_of_other(self):
        execute_scripted_runtime(
            _script_submit_during_decode_of_other,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_unique_rids_distinct(self):
        execute_scripted_runtime(
            _script_unique_rids_distinct,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_chunked_reqs_each_yield_15(self):
        execute_scripted_runtime(
            _script_chunked_reqs_each_yield_15,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )


if __name__ == "__main__":
    unittest.main()
