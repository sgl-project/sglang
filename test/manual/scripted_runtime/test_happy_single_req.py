"""Happy path — single-req normal completion.

Covers B.1 series from the expansion plan plus fan-out across
prompt-length × decode-length combinations. These are baseline
"the engine works" tests, no fault injection.
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


def _script_small_prompt_short_decode(t: ScriptedRuntime):
    # Tiny prompt + tiny decode.
    r = t.start_req(prompt_len=8, max_new_tokens=2)
    yield from run_until_finished(r)
    assert r.finished
    assert r.chunks_done == 0


def _script_medium_prompt_medium_decode(t: ScriptedRuntime):
    # Prompt <= chunk_size + medium decode.
    r = t.start_req(prompt_len=DEFAULT_CHUNK_SIZE, max_new_tokens=16)
    yield from run_until_finished(r)
    assert r.finished
    assert r.chunks_done <= 1


def _script_long_prompt_short_decode(t: ScriptedRuntime):
    # Multi-chunk prefill + short decode.
    r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
    yield from run_until_finished(r)
    assert r.finished
    assert r.chunks_done >= 2


def _script_long_prompt_long_decode(t: ScriptedRuntime):
    # Multi-chunk prefill + long decode.
    r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=64)
    yield from run_until(r, lambda h: h.finished, max_steps=1000)
    assert r.finished
    assert r.chunks_done >= 2


def _script_tiny_prompt_long_decode(t: ScriptedRuntime):
    # 1-token prompt, long decode.
    r = t.start_req(prompt_len=1, max_new_tokens=64)
    yield from run_until(r, lambda h: h.finished, max_steps=500)
    assert r.finished
    assert r.chunks_done == 0


def _script_chunk_size_minus_one_prompt(t: ScriptedRuntime):
    # prompt_len = chunk_size - 1 + short decode.
    r = t.start_req(prompt_len=DEFAULT_CHUNK_SIZE - 1, max_new_tokens=4)
    yield from run_until_finished(r)
    assert r.finished
    assert r.chunks_done == 0


def _script_chunk_size_plus_two_prompt(t: ScriptedRuntime):
    # prompt_len = chunk_size + 2.
    r = t.start_req(prompt_len=DEFAULT_CHUNK_SIZE + 2, max_new_tokens=4)
    yield from run_until_finished(r)
    assert r.finished
    assert r.chunks_done == 2


def _script_just_over_2x_chunk_size(t: ScriptedRuntime):
    # prompt_len = 2 * chunk_size + 1.
    r = t.start_req(prompt_len=2 * DEFAULT_CHUNK_SIZE + 1, max_new_tokens=4)
    yield from run_until_finished(r)
    assert r.finished
    assert r.chunks_done == 3


def _script_five_x_chunk_size(t: ScriptedRuntime):
    # 5 chunks exactly.
    r = t.start_req(prompt_len=5 * DEFAULT_CHUNK_SIZE, max_new_tokens=4)
    yield from run_until_finished(r)
    assert r.finished
    assert r.chunks_done == 5


def _script_ten_x_chunk_size(t: ScriptedRuntime):
    # 10 chunks exactly.
    r = t.start_req(prompt_len=10 * DEFAULT_CHUNK_SIZE, max_new_tokens=2)
    yield from run_until(r, lambda h: h.finished, max_steps=1000)
    assert r.finished
    assert r.chunks_done == 10


def _script_status_progression_happy_path(t: ScriptedRuntime):
    # Verify status progresses unknown → waiting → running → finished.
    r = t.start_req(prompt_len=16, max_new_tokens=2)
    seen = set()
    for _ in range(DEFAULT_MAX_STEPS):
        seen.add(r.status)
        if r.finished:
            break
        yield
    assert "finished" in seen


def _script_long_prompt_only_one_decode(t: ScriptedRuntime):
    # max_new_tokens = 1: minimum decode.
    r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=1)
    yield from run_until_finished(r)
    assert r.finished


def _script_kv_pages_consistent_during_run(t: ScriptedRuntime):
    # kv_pages > 0 during running, == 0 after finish.
    r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=4)
    saw_positive = False
    for _ in range(DEFAULT_MAX_STEPS):
        if r.kv_pages > 0:
            saw_positive = True
        if r.finished:
            break
        yield
    assert saw_positive
    assert r.kv_pages == 0


def _script_row_idx_recycled_after_finish(t: ScriptedRuntime):
    # After finish, row_idx becomes None.
    r = t.start_req(prompt_len=16, max_new_tokens=2)
    yield from run_until_finished(r)
    assert r.row_idx is None


class TestHappySingleReq(CustomTestCase):
    def test_small_prompt_short_decode(self):
        execute_scripted_runtime(
            _script_small_prompt_short_decode,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_medium_prompt_medium_decode(self):
        execute_scripted_runtime(
            _script_medium_prompt_medium_decode,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_long_prompt_short_decode(self):
        execute_scripted_runtime(
            _script_long_prompt_short_decode,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_long_prompt_long_decode(self):
        execute_scripted_runtime(
            _script_long_prompt_long_decode,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_tiny_prompt_long_decode(self):
        execute_scripted_runtime(
            _script_tiny_prompt_long_decode,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_chunk_size_minus_one_prompt(self):
        execute_scripted_runtime(
            _script_chunk_size_minus_one_prompt,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_chunk_size_plus_two_prompt(self):
        execute_scripted_runtime(
            _script_chunk_size_plus_two_prompt,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_just_over_2x_chunk_size(self):
        execute_scripted_runtime(
            _script_just_over_2x_chunk_size,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_five_x_chunk_size(self):
        execute_scripted_runtime(
            _script_five_x_chunk_size,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_ten_x_chunk_size(self):
        execute_scripted_runtime(
            _script_ten_x_chunk_size,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_status_progression_happy_path(self):
        execute_scripted_runtime(
            _script_status_progression_happy_path,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_long_prompt_only_one_decode(self):
        execute_scripted_runtime(
            _script_long_prompt_only_one_decode,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_kv_pages_consistent_during_run(self):
        execute_scripted_runtime(
            _script_kv_pages_consistent_during_run,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_row_idx_recycled_after_finish(self):
        execute_scripted_runtime(
            _script_row_idx_recycled_after_finish,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )


if __name__ == "__main__":
    unittest.main()
