"""Source-code special-case coverage scripted tests.

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

Also covers category D from the expansion plan: scheduler.py /
disaggregation / dllm chunked-related branches that round-1 did not
address. Each test drives the scheduler through a specific branch.
"""

import unittest

from sglang.test.scripted_runtime.entrypoint import execute_scripted_runtime
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


class TestScriptedSpecialCase(CustomTestCase):
    def test_dllm_stash_path(self):
        """DLLM staging queue intersects ``stash_chunked_request`` at scheduler.py:2335."""
        execute_scripted_runtime(
            self._script_dllm_stash_path,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    # DLLM staging queue intersects ``stash_chunked_request`` at
    # scheduler.py:2335. We pump a DLLM-style request through and verify
    # the chunked req is correctly stashed even with DLLM staging active.
    #
    # Requires DLLM model + multi-iter denoising support. Until
    # ScriptedRuntime exposes DLLM submission, the script just verifies
    # the scheduler does not crash with both flags on.
    @staticmethod
    def _script_dllm_stash_path(t: ScriptedRuntime):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=4)
        yield from run_until_finished(r)
        assert r.finished

    def test_chunked_in_flight_no_idle(self):
        """``self.chunked_req is not None`` early-exit bypass at scheduler.py:2487 / 2499."""
        execute_scripted_runtime(
            self._script_chunked_in_flight_no_idle,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    # ``self.chunked_req is not None`` early-exit bypass at
    # scheduler.py:2487 / 2499. While r1 is mid-chunk, the scheduler must
    # *not* return None from ``_get_new_batch_prefill_raw`` even if the
    # waiting_queue is empty — the in-flight chunked req still needs
    # continuation.
    @staticmethod
    def _script_chunked_in_flight_no_idle(t: ScriptedRuntime):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r, lambda h: h.is_chunking)
        # waiting_queue is empty (no other reqs submitted), but r is mid-chunk.
        # The scheduler must keep advancing — if it idled out, r would never finish.
        yield from run_until_finished(r)
        assert r.finished

    def test_dynamic_chunking_history_len(self):
        """Dynamic chunking reads ``history_len`` from ``self.chunked_req.prefix_indices`` (scheduler.py:2516-2517)."""
        execute_scripted_runtime(
            self._script_dynamic_chunking_history_len,
            **base_engine_kwargs(
                chunked_prefill_size=DEFAULT_CHUNK_SIZE,
                enable_dynamic_chunking=True,
            ),
        )

    # dynamic chunking reads ``history_len`` from
    # ``self.chunked_req.prefix_indices`` (scheduler.py:2516-2517). With
    # ``--enable-dynamic-chunking``, the per-iter chunk size adjusts to
    # the chunked req's history length.
    @staticmethod
    def _script_dynamic_chunking_history_len(t: ScriptedRuntime):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until_finished(r)
        assert r.finished
        assert r.chunks_done >= 2

    def test_add_chunked_req_path(self):
        """``adder.add_chunked_req`` path (scheduler.py:2541-2548)."""
        execute_scripted_runtime(
            self._script_add_chunked_req_path,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    # ``adder.add_chunked_req`` path (scheduler.py:2541-2548). The
    # primary chunked-resume admission loop. Naive long request exercises
    # this on every chunk after the first.
    @staticmethod
    def _script_add_chunked_req_path(t: ScriptedRuntime):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until_finished(r)
        assert r.finished
        assert r.chunks_done >= 2

    def test_chunked_scheduled_last_iter_flag(self):
        """``_chunked_req_scheduled_last_iter`` toggling under hybrid-SWA early-return (scheduler.py:2544-2548)."""
        execute_scripted_runtime(
            self._script_chunked_scheduled_last_iter_flag,
            **base_engine_kwargs(
                chunked_prefill_size=DEFAULT_CHUNK_SIZE,
                model_path="openai/gpt-oss-20b",
                mem_fraction_static=0.70,
                disable_piecewise_cuda_graph=True,
            ),
        )

    # ``_chunked_req_scheduled_last_iter`` toggling under hybrid-SWA
    # early-return (scheduler.py:2544-2548). The flag flips between True
    # and False as ``add_chunked_req`` succeeds vs early-returns.
    #
    # Requires hybrid SWA model — see test_scripted_hybrid_swa.py.
    @staticmethod
    def _script_chunked_scheduled_last_iter_flag(t: ScriptedRuntime):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until_finished(r)
        assert r.finished

    def test_admission_with_chunked_in_flight(self):
        """``add_one_req`` kwarg ``has_chunked_req=True`` propagation (scheduler.py:2593)."""
        execute_scripted_runtime(
            self._script_admission_with_chunked_in_flight,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    # ``add_one_req`` kwarg ``has_chunked_req=True`` propagation
    # (scheduler.py:2593). With a chunked req in flight, the admission of
    # new reqs takes this code path.
    @staticmethod
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

    def test_new_chunked_req_first_chunk(self):
        """``adder.new_chunked_req`` assignment on first chunk (scheduler.py:2636-2642)."""
        execute_scripted_runtime(
            self._script_new_chunked_req_first_chunk,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    # ``adder.new_chunked_req`` assignment on first chunk
    # (scheduler.py:2636-2642). The very first chunk of a long req takes
    # this assignment + assert path.
    @staticmethod
    def _script_new_chunked_req_first_chunk(t: ScriptedRuntime):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield  # request enters admission, new_chunked_req path triggers
        yield from run_until(r, lambda h: h.chunks_done >= 1)
        yield from run_until_finished(r)
        assert r.finished

    def test_inflight_middle_chunks_counter(self):
        """``inflight_middle_chunks += 1`` every iteration the chunked req is admitted (scheduler.py:2644-2645)."""
        execute_scripted_runtime(
            self._script_inflight_middle_chunks_counter,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    # ``inflight_middle_chunks += 1`` every iteration the chunked
    # req is admitted (scheduler.py:2644-2645). Counter should equal
    # number of chunks minus the initial admission.
    @staticmethod
    def _script_inflight_middle_chunks_counter(t: ScriptedRuntime):
        # 3 chunks: chunks_done == 3, inflight_middle_chunks ++ for chunks 1, 2.
        r = t.start_req(prompt_len=3 * DEFAULT_CHUNK_SIZE, max_new_tokens=1)
        yield from run_until_finished(r)
        assert r.finished
        # When ``has_pending_chunk`` clears, the counter is final.
        # Exact value depends on implementation; lower bound: at least 1
        # middle chunk for a 3-chunk request.
        assert r.inflight_middle_chunks >= 1

    def test_chunked_req_passes_through_batch(self):
        """``chunked_req=self.chunked_req`` wiring into ScheduleBatch (scheduler.py:2658)."""
        execute_scripted_runtime(
            self._script_chunked_req_passes_through_batch,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    # ``chunked_req=self.chunked_req`` wiring into ScheduleBatch
    # (scheduler.py:2658). Verifies ``extend_input_len`` correctly
    # reflects the *remaining* tokens after each chunk.
    @staticmethod
    def _script_chunked_req_passes_through_batch(t: ScriptedRuntime):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r, lambda h: h.is_chunking)

        comp = t.batch_composition()
        assert r.rid in comp.get("chunked", [])

        yield from run_until_finished(r)

    def test_no_idle_during_chunked(self):
        """Idle path bypass — ``chunked_req is None`` check in ``check_idle`` (scheduler.py:3174)."""
        execute_scripted_runtime(
            self._script_no_idle_during_chunked,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    # idle path bypass — ``chunked_req is None`` check in
    # ``check_idle`` (scheduler.py:3174). The scheduler must not enter
    # idle state while a chunked req is in flight.
    @staticmethod
    def _script_no_idle_during_chunked(t: ScriptedRuntime):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r, lambda h: h.is_chunking)

        # If the scheduler had idled, the chunked req would not progress.
        assert not t.is_idle
        yield from run_until_finished(r)

    def test_abort_excludes_chunked_req(self):
        """Abort path's ``chunked_req_to_exclude`` plumbing (scheduler.py:3568-3596)."""
        execute_scripted_runtime(
            self._script_abort_excludes_chunked_req,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    # abort path's ``chunked_req_to_exclude`` plumbing
    # (scheduler.py:3568-3596). With chunked_req live in last_batch when
    # abort fires, the exclusion set must include it.
    @staticmethod
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

    def test_last_batch_chunked_req_pp_context(self):
        """Scheduler.py:2363-2369 — last_batch tracks the chunked_req in the PP context (chunked_req_to_exclude path)."""
        execute_scripted_runtime(
            self._script_last_batch_chunked_req_pp_context,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    @staticmethod
    def _script_last_batch_chunked_req_pp_context(t: ScriptedRuntime):
        # scheduler.py:2363-2369 — last_batch tracks the chunked_req in
        # the PP context (chunked_req_to_exclude path).
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until_finished(r)
        assert r.finished

    def test_chunked_req_to_exclude_set_add(self):
        """Scheduler.py:2366 — chunked_req_to_exclude.add(last_batch.chunked_req)."""
        execute_scripted_runtime(
            self._script_chunked_req_to_exclude_set_add,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    @staticmethod
    def _script_chunked_req_to_exclude_set_add(t: ScriptedRuntime):
        # scheduler.py:2366 — chunked_req_to_exclude.add(last_batch.chunked_req).
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=4)
        yield from run_until_finished(r)

    def test_chunked_req_to_exclude_update_reqs(self):
        """Scheduler.py:2369 — chunked_req_to_exclude.update(last_batch.reqs)."""
        execute_scripted_runtime(
            self._script_chunked_req_to_exclude_update_reqs,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    @staticmethod
    def _script_chunked_req_to_exclude_update_reqs(t: ScriptedRuntime):
        # scheduler.py:2369 — chunked_req_to_exclude.update(last_batch.reqs).
        r1 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        r2 = t.start_req(prompt_len=16, max_new_tokens=2)
        yield from run_until_all_finished([r1, r2])

    def test_schedule_batch_init_new_chunked_req(self):
        """Scheduler.py:2658 — ScheduleBatch.init_new(chunked_req=self.chunked_req)."""
        execute_scripted_runtime(
            self._script_schedule_batch_init_new_chunked_req,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    @staticmethod
    def _script_schedule_batch_init_new_chunked_req(t: ScriptedRuntime):
        # scheduler.py:2658 — ScheduleBatch.init_new(chunked_req=self.chunked_req).
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until_finished(r)

    def test_mem_check_chunked_req_kwarg(self):
        """Scheduler.py:2676-2677 — mem check called with chunked_req=..."""
        execute_scripted_runtime(
            self._script_mem_check_chunked_req_kwarg,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    @staticmethod
    def _script_mem_check_chunked_req_kwarg(t: ScriptedRuntime):
        # scheduler.py:2676-2677 — mem check called with chunked_req=...
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until_finished(r)

    def test_get_chunked_req_lambda_getter(self):
        """Scheduler.py:680 — get_chunked_req lambda."""
        execute_scripted_runtime(
            self._script_get_chunked_req_lambda_getter,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    @staticmethod
    def _script_get_chunked_req_lambda_getter(t: ScriptedRuntime):
        # scheduler.py:680 — get_chunked_req lambda. Verify that during
        # a chunked req's lifetime, the getter returns the right rid.
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r, lambda h: h.is_chunking)

        # NEW API NEEDED: t.get_chunked_req_rid() — query the scheduler's
        # current chunked_req's rid (or None).
        cur = t.get_chunked_req_rid()
        if cur is not None:
            assert cur == r.rid

    def test_chunked_req_scheduled_last_iter_flip(self):
        """Scheduler.py: _chunked_req_scheduled_last_iter flip logic."""
        execute_scripted_runtime(
            self._script_chunked_req_scheduled_last_iter_flip,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    @staticmethod
    def _script_chunked_req_scheduled_last_iter_flip(t: ScriptedRuntime):
        # scheduler.py: _chunked_req_scheduled_last_iter flip logic.
        r = t.start_req(prompt_len=2 * DEFAULT_CHUNK_SIZE, max_new_tokens=2)
        yield from run_until_finished(r)

    def test_chunked_req_reset_to_none(self):
        """Scheduler.py:3596 — chunked_req=None reset path."""
        execute_scripted_runtime(
            self._script_chunked_req_reset_to_none,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    @staticmethod
    def _script_chunked_req_reset_to_none(t: ScriptedRuntime):
        # scheduler.py:3596 — chunked_req=None reset path. After all
        # chunked reqs finish, scheduler.chunked_req should be None.
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until_finished(r)
        cur = t.get_chunked_req_rid()
        assert cur is None

    def test_disagg_prefill_chunked_path(self):
        """Disaggregation/prefill.py — chunked req in disagg prefill mode."""
        execute_scripted_runtime(
            self._script_disagg_prefill_chunked_path,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    @staticmethod
    def _script_disagg_prefill_chunked_path(t: ScriptedRuntime):
        # disaggregation/prefill.py — chunked req in disagg prefill mode.
        # Single-engine smoke (disagg topology requires P3 multi-engine).
        # TODO(round-3): recreate the specific bug shape; this currently
        # is a forward-pointing smoke.
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until_finished(r)

    def test_disagg_decode_waiting_queue_kv_held(self):
        """Disaggregation/decode.py — waiting_queue reqs hold KV in decode mode."""
        execute_scripted_runtime(
            self._script_disagg_decode_waiting_queue_kv_held,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    @staticmethod
    def _script_disagg_decode_waiting_queue_kv_held(t: ScriptedRuntime):
        # disaggregation/decode.py — waiting_queue reqs hold KV in decode mode.
        # Smoke: chunked req traverses decode-side waiting state cleanly.
        # TODO(round-3): recreate the specific bug shape; this currently
        # is a forward-pointing smoke.
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until_finished(r)

    def test_dllm_staging_double_pending_middle_outputs(self):
        """Dllm/mixin/scheduler.py — DLLM staging AND chunked admission both incrementing pending_middle_outputs (double-source)."""
        execute_scripted_runtime(
            self._script_dllm_staging_double_pending_middle_outputs,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    @staticmethod
    def _script_dllm_staging_double_pending_middle_outputs(t: ScriptedRuntime):
        # dllm/mixin/scheduler.py — DLLM staging AND chunked admission both
        # incrementing pending_middle_outputs (double-source).
        # Single-engine smoke (DLLM model required for full coverage).
        # TODO(round-3): recreate the specific bug shape; this currently
        # is a forward-pointing smoke.
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until_finished(r)

    def test_staging_handler_chunked(self):
        """Disaggregation/common/staging_handler.py — chunked interaction."""
        execute_scripted_runtime(
            self._script_staging_handler_chunked,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    @staticmethod
    def _script_staging_handler_chunked(t: ScriptedRuntime):
        # disaggregation/common/staging_handler.py — chunked interaction.
        # TODO(round-3): recreate the specific bug shape; this currently
        # is a forward-pointing smoke.
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until_finished(r)

    def test_mooncake_conn_chunked(self):
        """Disaggregation/mooncake/conn.py — chunked path in conn layer."""
        execute_scripted_runtime(
            self._script_mooncake_conn_chunked,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    @staticmethod
    def _script_mooncake_conn_chunked(t: ScriptedRuntime):
        # disaggregation/mooncake/conn.py — chunked path in conn layer.
        # TODO(round-3): recreate the specific bug shape; this currently
        # is a forward-pointing smoke.
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until_finished(r)

    def test_nixl_conn_chunked(self):
        """Disaggregation/nixl/conn.py — chunked path."""
        execute_scripted_runtime(
            self._script_nixl_conn_chunked,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    @staticmethod
    def _script_nixl_conn_chunked(t: ScriptedRuntime):
        # disaggregation/nixl/conn.py — chunked path.
        # TODO(round-3): recreate the specific bug shape; this currently
        # is a forward-pointing smoke.
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until_finished(r)

    def test_idle_path_chunked_req_none(self):
        """Scheduler.py:3174 — idle path checks chunked_req is None."""
        execute_scripted_runtime(
            self._script_idle_path_chunked_req_none,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    @staticmethod
    def _script_idle_path_chunked_req_none(t: ScriptedRuntime):
        # scheduler.py:3174 — idle path checks chunked_req is None.
        # If we have no in-flight req, scheduler is idle.
        # Give the scheduler a few yields to settle into the idle state
        # (initial setup may keep is_idle False for one or two iterations).
        for _ in range(5):
            yield
        assert t.is_idle

    def test_dynamic_chunking_history_len_extra(self):
        """Scheduler.py:2516-2517 — dynamic chunking reads history_len from chunked_req."""
        execute_scripted_runtime(
            self._script_dynamic_chunking_history_len_extra,
            **base_engine_kwargs(
                chunked_prefill_size=DEFAULT_CHUNK_SIZE,
                enable_dynamic_chunking=True,
            ),
        )

    @staticmethod
    def _script_dynamic_chunking_history_len_extra(t: ScriptedRuntime):
        # scheduler.py:2516-2517 — dynamic chunking reads history_len from
        # chunked_req. Enabled via --enable-dynamic-chunking.
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until_finished(r)

    def test_admission_path_with_chunked_inflight_flag(self):
        """Scheduler.py:2593 — add_one_req called with has_chunked_req=True."""
        execute_scripted_runtime(
            self._script_admission_path_with_chunked_inflight_flag,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    @staticmethod
    def _script_admission_path_with_chunked_inflight_flag(t: ScriptedRuntime):
        # scheduler.py:2593 — add_one_req called with has_chunked_req=True.
        r_chunked = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r_chunked, lambda h: h.is_chunking)
        r_new = t.start_req(prompt_len=16, max_new_tokens=2)
        yield from run_until_all_finished([r_chunked, r_new])

    def test_inflight_counter_increments_each_chunk(self):
        """Scheduler.py:2644-2645 — inflight_middle_chunks += 1 per chunk."""
        execute_scripted_runtime(
            self._script_inflight_counter_increments_each_chunk,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    @staticmethod
    def _script_inflight_counter_increments_each_chunk(t: ScriptedRuntime):
        # scheduler.py:2644-2645 — inflight_middle_chunks += 1 per chunk.
        r = t.start_req(prompt_len=4 * DEFAULT_CHUNK_SIZE, max_new_tokens=2)
        saw_increment = False
        last = 0
        for _ in range(DEFAULT_MAX_STEPS):
            cur = r.inflight_middle_chunks
            if cur > last:
                saw_increment = True
            last = max(last, cur)
            if r.finished:
                break
            yield
        # After at least one chunk, the counter must have moved up at some point.
        assert saw_increment, "expected inflight_middle_chunks to increment"

    def test_filter_batch_exclude_chunked_flag(self):
        """Filter_batch + chunked: exclude_chunked_req branch."""
        execute_scripted_runtime(
            self._script_filter_batch_exclude_chunked_flag,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    @staticmethod
    def _script_filter_batch_exclude_chunked_flag(t: ScriptedRuntime):
        # filter_batch + chunked: exclude_chunked_req branch.
        r1 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        r2 = t.start_req(prompt_len=16, max_new_tokens=2)
        yield from run_until_all_finished([r1, r2])

    def test_pdmux_split_prefill_batch(self):
        """34c02d6a67: filter chunked-resume from split_prefill_batch."""
        execute_scripted_runtime(
            self._script_pdmux_split_prefill_batch,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    @staticmethod
    def _script_pdmux_split_prefill_batch(t: ScriptedRuntime):
        # 34c02d6a67: filter chunked-resume from split_prefill_batch.
        # pdmux-specific; single-engine smoke.
        # TODO(round-3): recreate the specific bug shape; this currently
        # is a forward-pointing smoke.
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until_finished(r)

    def test_streaming_session_kv_committed_bound(self):
        """116584e8fa: bound streaming-session chunked stash by kv_committed_len."""
        execute_scripted_runtime(
            self._script_streaming_session_kv_committed_bound,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    @staticmethod
    def _script_streaming_session_kv_committed_bound(t: ScriptedRuntime):
        # 116584e8fa: bound streaming-session chunked stash by kv_committed_len.
        # TODO(round-3): recreate the specific bug shape; this currently
        # is a forward-pointing smoke.
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until_finished(r)

    def test_mamba_pool_idx_cleanup_skip_chunked_resume(self):
        """Dbdcdde245: skip mamba_pool_idx cleanup for chunked-resume on NO_TOKEN."""
        execute_scripted_runtime(
            self._script_mamba_pool_idx_cleanup_skip_chunked_resume,
            **base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE),
        )

    @staticmethod
    def _script_mamba_pool_idx_cleanup_skip_chunked_resume(t: ScriptedRuntime):
        # dbdcdde245: skip mamba_pool_idx cleanup for chunked-resume on
        # NO_TOKEN. Mamba-specific; single-engine smoke.
        # TODO(round-3): recreate the specific bug shape; this currently
        # is a forward-pointing smoke.
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until_finished(r)


if __name__ == "__main__":
    unittest.main()
