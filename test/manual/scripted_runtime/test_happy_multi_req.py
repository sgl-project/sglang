"""Happy path — multi-req parallel completion.

Covers B.2 series from the expansion plan plus fan-out across
batch sizes and mixed-shape concurrency.
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


def _script_two_small_parallel(t: ScriptedRuntime):
    r1 = t.start_req(prompt_len=16, max_new_tokens=4)
    r2 = t.start_req(prompt_len=16, max_new_tokens=4)
    yield from run_until_all_finished([r1, r2])
    assert r1.finished and r2.finished


def _script_three_small_parallel(t: ScriptedRuntime):
    reqs = [t.start_req(prompt_len=16, max_new_tokens=4) for _ in range(3)]
    yield from run_until_all_finished(reqs)


def _script_five_small_parallel(t: ScriptedRuntime):
    reqs = [t.start_req(prompt_len=16, max_new_tokens=4) for _ in range(5)]
    yield from run_until_all_finished(reqs)


def _script_ten_small_parallel(t: ScriptedRuntime):
    reqs = [t.start_req(prompt_len=16, max_new_tokens=2) for _ in range(10)]
    yield from run_until_all_finished(reqs)


def _script_one_chunked_plus_many_short(t: ScriptedRuntime):
    # 1 long chunked + 5 short, all parallel.
    chunked = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
    shorts = [t.start_req(prompt_len=16, max_new_tokens=2) for _ in range(5)]
    yield from run_until_all_finished([chunked] + shorts)
    assert chunked.chunks_done >= 2


def _script_multiple_chunked_staggered(t: ScriptedRuntime):
    # Submit chunked reqs every few yields, serial chunking.
    reqs = []
    for _ in range(4):
        reqs.append(t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2))
        yield
        yield
    yield from run_until_all_finished(reqs, max_steps=2000)


def _script_eight_concurrent_chunked(t: ScriptedRuntime):
    # 8 chunked reqs submitted together. Single-in-flight invariant holds.
    reqs = [
        t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        for _ in range(8)
    ]
    for _ in range(DEFAULT_MAX_STEPS * 5):
        assert t.chunked_in_flight_count() <= 1
        if all(r.finished for r in reqs):
            return
        yield


def _script_decode_only_batch(t: ScriptedRuntime):
    # 10 short reqs — pure decode batch.
    reqs = [t.start_req(prompt_len=4, max_new_tokens=8) for _ in range(10)]
    yield from run_until_all_finished(reqs)
    # Decode-only batch: chunked_in_flight should never have been > 0.


def _script_mixed_prefill_lengths(t: ScriptedRuntime):
    # Variable prompt lengths in same batch.
    lens = [8, 16, 32, 64, 128, 256, 512, 1024]
    reqs = [t.start_req(prompt_len=L, max_new_tokens=2) for L in lens]
    yield from run_until_all_finished(reqs, max_steps=1500)


def _script_two_chunked_one_decode(t: ScriptedRuntime):
    # 2 chunked + 1 decode-only.
    chunked1 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
    chunked2 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
    short = t.start_req(prompt_len=8, max_new_tokens=4)
    yield from run_until_all_finished([chunked1, chunked2, short])


def _script_batch_with_finish_event_count(t: ScriptedRuntime):
    # Each req emits exactly 1 finish event.
    reqs = [t.start_req(prompt_len=16, max_new_tokens=2) for _ in range(6)]
    yield from run_until_all_finished(reqs)
    for r in reqs:
        assert r.finish_event_count == 1


def _script_batch_state_query_during_run(t: ScriptedRuntime):
    # Query batch_composition every step while batch is active.
    reqs = [t.start_req(prompt_len=16, max_new_tokens=4) for _ in range(4)]
    for _ in range(DEFAULT_MAX_STEPS):
        comp = t.batch_composition()
        assert isinstance(comp, dict)
        if all(r.finished for r in reqs):
            return
        yield


def _script_mixed_lengths_then_more_arrivals(t: ScriptedRuntime):
    # First batch starts; midway, more reqs arrive.
    initial = [t.start_req(prompt_len=16, max_new_tokens=4) for _ in range(3)]
    yield
    yield
    more = [t.start_req(prompt_len=16, max_new_tokens=4) for _ in range(3)]
    yield from run_until_all_finished(initial + more)


def _script_parallel_with_priority(t: ScriptedRuntime):
    # 3 normal + 2 high-priority reqs.
    normal = [t.start_req(prompt_len=16, max_new_tokens=2) for _ in range(3)]
    high = [
        t.start_req(prompt_len=16, max_new_tokens=2, priority="high")
        for _ in range(2)
    ]
    yield from run_until_all_finished(normal + high)


class TestHappyMultiReq(CustomTestCase):
    def test_two_small_parallel(self):
        execute_scripted_runtime(
            _script_two_small_parallel,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_three_small_parallel(self):
        execute_scripted_runtime(
            _script_three_small_parallel,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_five_small_parallel(self):
        execute_scripted_runtime(
            _script_five_small_parallel,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_ten_small_parallel(self):
        execute_scripted_runtime(
            _script_ten_small_parallel,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_one_chunked_plus_many_short(self):
        execute_scripted_runtime(
            _script_one_chunked_plus_many_short,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_multiple_chunked_staggered(self):
        execute_scripted_runtime(
            _script_multiple_chunked_staggered,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_eight_concurrent_chunked(self):
        execute_scripted_runtime(
            _script_eight_concurrent_chunked,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_decode_only_batch(self):
        execute_scripted_runtime(
            _script_decode_only_batch,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_mixed_prefill_lengths(self):
        execute_scripted_runtime(
            _script_mixed_prefill_lengths,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_two_chunked_one_decode(self):
        execute_scripted_runtime(
            _script_two_chunked_one_decode,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_batch_with_finish_event_count(self):
        execute_scripted_runtime(
            _script_batch_with_finish_event_count,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_batch_state_query_during_run(self):
        execute_scripted_runtime(
            _script_batch_state_query_during_run,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_mixed_lengths_then_more_arrivals(self):
        execute_scripted_runtime(
            _script_mixed_lengths_then_more_arrivals,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_parallel_with_priority(self):
        execute_scripted_runtime(
            _script_parallel_with_priority,
            **base_engine_kwargs(
                chunked_prefill_size=DEFAULT_CHUNK_SIZE,
                enable_priority_scheduling=True,
            ),
        )


if __name__ == "__main__":
    unittest.main()
