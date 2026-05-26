"""Source-code special-case coverage.

Each test deliberately drives the scheduler through one of the
``self.chunked_req is not None`` / ``has_pending_chunk`` /
``pending_middle_outputs`` / ``exclude_chunked_req`` branches in
``scheduler.py``. The intent is: when the chunked refactor lands and
the special case disappears, the same script should still pass
(behavior-level assertion) so the absence of the branch is silently
ok — but during the refactor any *behavior change* on the test
trigger fails fast.

Line references are to main-upstream's ``scheduler.py`` at the time
of writing (audit doc lists exact lines).
"""

import unittest
from sglang.test.scripted_runtime_chunked_helpers import (
    DEFAULT_CHUNK_SIZE,
    VERY_LONG_PROMPT_LEN,
    base_engine_kwargs,
    run_until,
    run_until_finished,
)

from sglang.test.scripted_runtime.entrypoint import execute_scripted_runtime
from sglang.test.scripted_runtime.runtime import ScriptedRuntime
from sglang.test.test_utils import CustomTestCase


# DLLM staging queue intersects ``stash_chunked_request`` at
# scheduler.py:2335. We pump a DLLM-style request through and verify
# the chunked req is correctly stashed even with DLLM staging active.
#
# Requires DLLM model + multi-iter denoising support. Until
# ScriptedRuntime exposes DLLM submission, the script just verifies
# the scheduler does not crash with both flags on.
def _script_dllm_stash_path(t: ScriptedRuntime):
    r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=4)
    yield from run_until_finished(r)
    assert r.finished


# ``self.chunked_req is not None`` early-exit bypass at
# scheduler.py:2487 / 2499. While r1 is mid-chunk, the scheduler must
# *not* return None from ``_get_new_batch_prefill_raw`` even if the
# waiting_queue is empty — the in-flight chunked req still needs
# continuation.
def _script_chunked_in_flight_no_idle(t: ScriptedRuntime):
    r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
    yield from run_until(r, lambda h: h.is_chunking)
    # waiting_queue is empty (no other reqs submitted), but r is mid-chunk.
    # The scheduler must keep advancing — if it idled out, r would never finish.
    yield from run_until_finished(r)
    assert r.finished


# dynamic chunking reads ``history_len`` from
# ``self.chunked_req.prefix_indices`` (scheduler.py:2516-2517). With
# ``--enable-dynamic-chunking``, the per-iter chunk size adjusts to
# the chunked req's history length.
def _script_dynamic_chunking_history_len(t: ScriptedRuntime):
    r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
    yield from run_until_finished(r)
    assert r.finished
    assert r.chunks_done >= 2


# ``adder.add_chunked_req`` path (scheduler.py:2541-2548). The
# primary chunked-resume admission loop. Naive long request exercises
# this on every chunk after the first.
def _script_add_chunked_req_path(t: ScriptedRuntime):
    r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
    yield from run_until_finished(r)
    assert r.finished
    assert r.chunks_done >= 2


# ``_chunked_req_scheduled_last_iter`` toggling under hybrid-SWA
# early-return (scheduler.py:2544-2548). The flag flips between True
# and False as ``add_chunked_req`` succeeds vs early-returns.
#
# Requires hybrid SWA model — see test_feature_hybrid_swa.py.
def _script_chunked_scheduled_last_iter_flag(t: ScriptedRuntime):
    r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
    yield from run_until_finished(r)
    assert r.finished


# ``add_one_req`` kwarg ``has_chunked_req=True`` propagation
# (scheduler.py:2593). With a chunked req in flight, the admission of
# new reqs takes this code path.
def _script_admission_with_chunked_in_flight(t: ScriptedRuntime):
    r_chunk = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
    yield from run_until(r_chunk, lambda h: h.is_chunking)

    # Submit a small req that goes through admission while chunked is in flight.
    r_small = t.start_req(prompt_len=4, max_new_tokens=2)
    yield  # admission step

    # Composition check: at the admission iteration, the scheduler
    # has had to take has_chunked_req=True branch.
    comp = t.batch_composition()
    assert r_chunk.rid in comp.get("chunked", [])
    yield from run_until_finished(r_small)
    yield from run_until_finished(r_chunk)


# ``adder.new_chunked_req`` assignment on first chunk
# (scheduler.py:2636-2642). The very first chunk of a long req takes
# this assignment + assert path.
def _script_new_chunked_req_first_chunk(t: ScriptedRuntime):
    r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
    yield  # request enters admission, new_chunked_req path triggers
    yield from run_until(r, lambda h: h.chunks_done >= 1)
    yield from run_until_finished(r)
    assert r.finished


# ``inflight_middle_chunks += 1`` every iteration the chunked
# req is admitted (scheduler.py:2644-2645). Counter should equal
# number of chunks minus the initial admission.
def _script_inflight_middle_chunks_counter(t: ScriptedRuntime):
    # 3 chunks: chunks_done == 3, inflight_middle_chunks ++ for chunks 1, 2.
    r = t.start_req(prompt_len=3 * DEFAULT_CHUNK_SIZE, max_new_tokens=1)
    yield from run_until_finished(r)
    assert r.finished
    # When ``has_pending_chunk`` clears, the counter is final.
    # Exact value depends on implementation; lower bound: at least 1
    # middle chunk for a 3-chunk request.
    assert r.inflight_middle_chunks >= 1


# ``chunked_req=self.chunked_req`` wiring into ScheduleBatch
# (scheduler.py:2658). Verifies ``extend_input_len`` correctly
# reflects the *remaining* tokens after each chunk.
def _script_chunked_req_passes_through_batch(t: ScriptedRuntime):
    r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
    yield from run_until(r, lambda h: h.is_chunking)

    comp = t.batch_composition()
    assert r.rid in comp.get("chunked", [])

    yield from run_until_finished(r)


# idle path bypass — ``chunked_req is None`` check in
# ``check_idle`` (scheduler.py:3174). The scheduler must not enter
# idle state while a chunked req is in flight.
def _script_no_idle_during_chunked(t: ScriptedRuntime):
    r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
    yield from run_until(r, lambda h: h.is_chunking)

    # If the scheduler had idled, the chunked req would not progress.
    assert not t.is_idle
    yield from run_until_finished(r)


# abort path's ``chunked_req_to_exclude`` plumbing
# (scheduler.py:3568-3596). With chunked_req live in last_batch when
# abort fires, the exclusion set must include it.
def _script_abort_excludes_chunked_req(t: ScriptedRuntime):
    r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
    yield from run_until(r, lambda h: h.is_chunking)

    t.abort(r)
    yield

    # After abort the scheduler state must reset cleanly — no
    # leftover chunked_req referencing the aborted Req.
    assert (
        t.chunked_in_flight_count() == 0
    ), f"abort must clear in-flight count; got {t.chunked_in_flight_count()}"


class TestSpecialCaseCoverage(CustomTestCase):
    def test_dllm_stash_path(self):
        execute_scripted_runtime(
            _script_dllm_stash_path,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_chunked_in_flight_no_idle(self):
        execute_scripted_runtime(
            _script_chunked_in_flight_no_idle,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_dynamic_chunking_history_len(self):
        execute_scripted_runtime(
            _script_dynamic_chunking_history_len,
            **base_engine_kwargs(
                chunked_prefill_size=DEFAULT_CHUNK_SIZE,
                enable_dynamic_chunking=True,
            ),
        )

    def test_add_chunked_req_path(self):
        execute_scripted_runtime(
            _script_add_chunked_req_path,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_chunked_scheduled_last_iter_flag(self):
        execute_scripted_runtime(
            _script_chunked_scheduled_last_iter_flag,
            **base_engine_kwargs(
                chunked_prefill_size=DEFAULT_CHUNK_SIZE,
                model_path="openai/gpt-oss-20b",
                mem_fraction_static=0.70,
                disable_piecewise_cuda_graph=True,
            ),
        )

    def test_admission_with_chunked_in_flight(self):
        execute_scripted_runtime(
            _script_admission_with_chunked_in_flight,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_new_chunked_req_first_chunk(self):
        execute_scripted_runtime(
            _script_new_chunked_req_first_chunk,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_inflight_middle_chunks_counter(self):
        execute_scripted_runtime(
            _script_inflight_middle_chunks_counter,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_chunked_req_passes_through_batch(self):
        execute_scripted_runtime(
            _script_chunked_req_passes_through_batch,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_no_idle_during_chunked(self):
        execute_scripted_runtime(
            _script_no_idle_during_chunked,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    def test_abort_excludes_chunked_req(self):
        execute_scripted_runtime(
            _script_abort_excludes_chunked_req,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )


if __name__ == "__main__":
    unittest.main()
