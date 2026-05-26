"""Happy path — default and common sampling configurations.

Covers B.4 series from the expansion plan plus parametric fan-out
across (default / greedy / high temperature / top-k / top-p) ×
(short / chunked prompt).
"""

import unittest

from sglang.test.scripted_runtime.entrypoint import execute_scripted_runtime
from sglang.test.scripted_runtime.runtime import ScriptedRuntime
from sglang.test.scripted_runtime_chunked_helpers import (
    DEFAULT_CHUNK_SIZE,
    VERY_LONG_PROMPT_LEN,
    base_engine_kwargs,
    run_until_finished,
)
from sglang.test.test_utils import CustomTestCase


def _script_default_sampling_chunked(t: ScriptedRuntime):
    # All defaults + chunked: just complete cleanly.
    r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=4)
    yield from run_until_finished(r)
    assert r.finished


def _script_greedy_chunked(t: ScriptedRuntime):
    r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=4, temperature=0.0)
    yield from run_until_finished(r)
    assert r.finished


def _script_high_temperature_short(t: ScriptedRuntime):
    r = t.start_req(prompt_len=16, max_new_tokens=4, temperature=1.8)
    yield from run_until_finished(r)
    assert r.finished


def _script_high_temperature_chunked(t: ScriptedRuntime):
    r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=4, temperature=1.8)
    yield from run_until_finished(r)
    assert r.finished


def _script_low_temperature_short(t: ScriptedRuntime):
    r = t.start_req(prompt_len=16, max_new_tokens=4, temperature=0.1)
    yield from run_until_finished(r)
    assert r.finished


def _script_default_top_p(t: ScriptedRuntime):
    r = t.start_req(prompt_len=16, max_new_tokens=4, top_p=0.95)
    yield from run_until_finished(r)
    assert r.finished


def _script_default_top_k(t: ScriptedRuntime):
    r = t.start_req(prompt_len=16, max_new_tokens=4, top_k=50)
    yield from run_until_finished(r)
    assert r.finished


def _script_combined_sampling_chunked(t: ScriptedRuntime):
    # All sampling knobs on at once.
    r = t.start_req(
        prompt_len=VERY_LONG_PROMPT_LEN,
        max_new_tokens=4,
        temperature=0.7,
        top_p=0.9,
        top_k=40,
    )
    yield from run_until_finished(r)
    assert r.finished


def _script_default_sampling_short(t: ScriptedRuntime):
    r = t.start_req(prompt_len=8, max_new_tokens=2)
    yield from run_until_finished(r)
    assert r.finished


def _script_sampling_diversity_two_reqs(t: ScriptedRuntime):
    # Two reqs with same prompt and non-greedy temp: outputs may differ.
    r1 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=4, temperature=1.0)
    yield from run_until_finished(r1)
    r2 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=4, temperature=1.0)
    yield from run_until_finished(r2)


class TestHappySampling(CustomTestCase):
    def test_default_sampling_chunked(self):
        execute_scripted_runtime(
            _script_default_sampling_chunked,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_greedy_chunked(self):
        execute_scripted_runtime(
            _script_greedy_chunked,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_high_temperature_short(self):
        execute_scripted_runtime(
            _script_high_temperature_short,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_high_temperature_chunked(self):
        execute_scripted_runtime(
            _script_high_temperature_chunked,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_low_temperature_short(self):
        execute_scripted_runtime(
            _script_low_temperature_short,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_default_top_p(self):
        execute_scripted_runtime(
            _script_default_top_p,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_default_top_k(self):
        execute_scripted_runtime(
            _script_default_top_k,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_combined_sampling_chunked(self):
        execute_scripted_runtime(
            _script_combined_sampling_chunked,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_default_sampling_short(self):
        execute_scripted_runtime(
            _script_default_sampling_short,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_sampling_diversity_two_reqs(self):
        execute_scripted_runtime(
            _script_sampling_diversity_two_reqs,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )


if __name__ == "__main__":
    unittest.main()
