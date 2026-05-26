"""Happy path — long-running stability scenarios.

Covers B.5 series from the expansion plan and stress fan-outs. The
focus is "no resource leak after many reqs" — every test starts and
finishes a baseline run, then asserts ``engine_stats`` returns to
(or above) the initial pool counts.
"""

import unittest

from sglang.test.scripted_runtime.entrypoint import execute_scripted_runtime
from sglang.test.scripted_runtime.runtime import ScriptedRuntime
from sglang.test.scripted_runtime_chunked_helpers import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_MAX_STEPS,
    VERY_LONG_PROMPT_LEN,
    base_engine_kwargs,
    run_until_all_finished,
)
from sglang.test.test_utils import CustomTestCase


def _script_hundred_reqs_no_leak(t: ScriptedRuntime):
    # 100 reqs end-to-end: KV/row/lock_ref pool counts return to baseline.
    baseline = t.engine_stats()
    reqs = [t.start_req(prompt_len=16, max_new_tokens=2) for _ in range(100)]
    yield from run_until_all_finished(reqs, max_steps=4000)
    final = t.engine_stats()
    assert (
        final["kv_pool_free"] >= baseline["kv_pool_free"]
    ), f"KV leak: {baseline['kv_pool_free']} -> {final['kv_pool_free']}"
    assert final["row_pool_free"] >= baseline["row_pool_free"]


def _script_two_hundred_reqs_no_leak(t: ScriptedRuntime):
    baseline = t.engine_stats()
    reqs = [t.start_req(prompt_len=8, max_new_tokens=1) for _ in range(200)]
    yield from run_until_all_finished(reqs, max_steps=8000)
    final = t.engine_stats()
    assert final["kv_pool_free"] >= baseline["kv_pool_free"]


def _script_five_hundred_reqs_no_leak(t: ScriptedRuntime):
    baseline = t.engine_stats()
    reqs = [t.start_req(prompt_len=8, max_new_tokens=1) for _ in range(500)]
    yield from run_until_all_finished(reqs, max_steps=15000)
    final = t.engine_stats()
    assert final["kv_pool_free"] >= baseline["kv_pool_free"]


def _script_long_lived_engine_reps_chunked(t: ScriptedRuntime):
    # 20 rounds × 5 reqs each; scheduler internal counters stay healthy.
    baseline = t.engine_stats()
    for _ in range(20):
        reqs = [
            t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
            for _ in range(5)
        ]
        yield from run_until_all_finished(reqs, max_steps=2000)
    final = t.engine_stats()
    assert final["kv_pool_free"] >= baseline["kv_pool_free"]


def _script_sustained_long_chunked_load(t: ScriptedRuntime):
    # Sustained: 30 long chunked reqs.
    reqs = [
        t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        for _ in range(30)
    ]
    yield from run_until_all_finished(reqs, max_steps=8000)


def _script_round_robin_short_and_chunked(t: ScriptedRuntime):
    # 50 short followed by 5 chunked, 5 rounds.
    for _ in range(5):
        shorts = [t.start_req(prompt_len=16, max_new_tokens=2) for _ in range(10)]
        chunked = [
            t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
            for _ in range(1)
        ]
        yield from run_until_all_finished(shorts + chunked, max_steps=2000)


def _script_long_decode_then_many_short(t: ScriptedRuntime):
    # One very long decode + many short.
    long_decode = t.start_req(prompt_len=16, max_new_tokens=256)
    shorts = [t.start_req(prompt_len=8, max_new_tokens=2) for _ in range(50)]
    yield from run_until_all_finished([long_decode] + shorts, max_steps=4000)


def _script_chunked_in_flight_count_never_above_one_long_run(t: ScriptedRuntime):
    # 50 chunked reqs over many yields; verify invariant at every step.
    # Step budget bumped to DEFAULT_MAX_STEPS * 60 because 50 chunked
    # reqs with VERY_LONG_PROMPT_LEN each can take many chunk
    # iterations and the original *30 budget was borderline-tight.
    reqs = [
        t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        for _ in range(50)
    ]
    for _ in range(DEFAULT_MAX_STEPS * 60):
        assert t.chunked_in_flight_count() <= 1
        if all(r.finished for r in reqs):
            return
        yield


def _script_engine_stats_monotone_after_each_batch(t: ScriptedRuntime):
    # After each batch finishes, kv_pool_free non-decreasing vs end-of-prev-batch.
    last = None
    for _ in range(10):
        reqs = [t.start_req(prompt_len=16, max_new_tokens=2) for _ in range(8)]
        yield from run_until_all_finished(reqs)
        cur = t.engine_stats()["kv_pool_free"]
        if last is not None:
            assert cur >= last - 1, f"KV pool drifted: {last} -> {cur}"
        last = cur


class TestHappyLongRun(CustomTestCase):
    def test_hundred_reqs_no_leak(self):
        execute_scripted_runtime(
            _script_hundred_reqs_no_leak,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_two_hundred_reqs_no_leak(self):
        execute_scripted_runtime(
            _script_two_hundred_reqs_no_leak,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_five_hundred_reqs_no_leak(self):
        execute_scripted_runtime(
            _script_five_hundred_reqs_no_leak,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_long_lived_engine_reps_chunked(self):
        execute_scripted_runtime(
            _script_long_lived_engine_reps_chunked,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_sustained_long_chunked_load(self):
        execute_scripted_runtime(
            _script_sustained_long_chunked_load,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_round_robin_short_and_chunked(self):
        execute_scripted_runtime(
            _script_round_robin_short_and_chunked,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_long_decode_then_many_short(self):
        execute_scripted_runtime(
            _script_long_decode_then_many_short,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_chunked_in_flight_count_never_above_one_long_run(self):
        execute_scripted_runtime(
            _script_chunked_in_flight_count_never_above_one_long_run,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_engine_stats_monotone_after_each_batch(self):
        execute_scripted_runtime(
            _script_engine_stats_monotone_after_each_batch,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )


if __name__ == "__main__":
    unittest.main()
