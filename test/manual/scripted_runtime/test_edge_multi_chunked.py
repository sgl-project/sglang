"""Edge cases — multiple chunked reqs in flight.

Verifies the single-in-flight chunked
invariant from main-upstream (``len(chunked_reqs()) <= 1``), and the
mixed chunked-in-flight + normal-decode batch composition.
"""

import unittest
from sglang.test.scripted_runtime_chunked_helpers import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_MAX_STEPS,
    VERY_LONG_PROMPT_LEN,
    base_engine_kwargs,
    run_until,
    run_until_all_finished,
)

from sglang.test.scripted_runtime.entrypoint import execute_scripted_runtime
from sglang.test.scripted_runtime.runtime import ScriptedRuntime
from sglang.test.test_utils import CustomTestCase


# two long requests submitted back-to-back; main-upstream
# invariant says at most one is chunked-in-flight at any moment.
def _script_at_most_one_chunked_in_flight(t: ScriptedRuntime):
    r1 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
    r2 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)

    # Drive the scheduler until both complete, asserting the
    # single-in-flight invariant at every step.
    for _ in range(DEFAULT_MAX_STEPS):
        in_flight = t.chunked_in_flight_count()
        assert in_flight <= 1, (
            f"single-in-flight invariant violated: "
            f"chunked_in_flight_count()={in_flight}"
        )
        if r1.finished and r2.finished:
            return
        yield
    raise AssertionError("r1 and r2 did not both finish within step budget")


# r1 chunked mid-stream + r2 submitted long. r2 must wait for
# r1's chunk loop to clear before starting its own chunking.
def _script_second_chunked_waits(t: ScriptedRuntime):
    r1 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
    yield from run_until(r1, lambda h: h.is_chunking)
    # r1 is mid-chunk-loop. Submit a second long req.
    r2 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
    yield  # give scheduler one step to observe r2

    assert r1.is_chunking, "r1 should still be chunking"
    assert (
        not r2.is_chunking
    ), "r2 must wait for r1's chunk loop to clear before chunking"

    yield from run_until_all_finished([r1, r2])


# r1 chunked mid-stream + r2 short decode-only. r2 should be
# admittable into the running batch alongside r1's chunked extend.
def _script_chunked_plus_decode_in_batch(t: ScriptedRuntime):
    r1 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
    yield from run_until(r1, lambda h: h.is_chunking)

    r2 = t.start_req(prompt_len=8, max_new_tokens=4)
    yield  # admission step

    comp = t.batch_composition()
    assert r1.rid in comp.get(
        "chunked", []
    ), f"r1 should be in chunked subset of batch; got {comp}"
    assert r2.rid in comp.get("prefill", []) + comp.get(
        "decode", []
    ), f"r2 should be in prefill or decode subset; got {comp}"

    yield from run_until_all_finished([r1, r2])


class TestEdgeMultiChunked(CustomTestCase):
    def test_at_most_one_chunked_in_flight(self):
        execute_scripted_runtime(
            _script_at_most_one_chunked_in_flight,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_second_chunked_waits(self):
        execute_scripted_runtime(
            _script_second_chunked_waits,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_chunked_plus_decode_in_batch(self):
        execute_scripted_runtime(
            _script_chunked_plus_decode_in_batch,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )


if __name__ == "__main__":
    unittest.main()
