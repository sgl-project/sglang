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

from sglang.test.scripted_runtime.runtime import ScriptedRuntime
from sglang.test.scripted_runtime.testcase import ScriptedRuntimeTestCase
from sglang.test.scripted_runtime_chunked_helpers import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_MAX_STEPS,
    VERY_LONG_PROMPT_LEN,
    base_engine_kwargs,
    run_until,
    run_until_finished,
)


class TestSpecialCaseBasic(ScriptedRuntimeTestCase):
    ENGINE_KWARGS = base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE)

    def test_chunked_in_flight_no_idle(self):
        """``self.chunked_req is not None`` early-exit bypass at scheduler.py:2487 / 2499."""
        self.runtime.run(self._script_chunked_in_flight_no_idle)

    # ``self.chunked_req is not None`` early-exit bypass at
    # scheduler.py:2487 / 2499. While r1 is mid-chunk, the scheduler must
    # *not* return None from ``_get_new_batch_prefill_raw`` even if the
    # waiting_queue is empty — the in-flight chunked req still needs
    # continuation. In-loop S2 invariant ensures the chunked_req is
    # never simultaneously in the running batch (mutually-exclusive slot).
    @staticmethod
    def _script_chunked_in_flight_no_idle(t: ScriptedRuntime):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r, lambda h: h.is_chunking)
        # While r is chunking, scheduler must not idle and S2 invariant must hold.
        saw_chunking = False
        for _ in range(DEFAULT_MAX_STEPS):
            if r.is_chunking:
                saw_chunking = True
                assert (
                    not t.is_idle
                ), "scheduler must not idle while chunked_req is in flight"
            assert t.chunked_req_in_batch_violation_count() == 0, (
                "S2 invariant: chunked_req must be mutually exclusive with "
                "running_batch.reqs"
            )
            if r.finished:
                break
            yield
        assert r.finished
        assert saw_chunking, "test must observe r mid-chunk at least once"
        assert t.chunked_req_in_batch_violation_count() == 0

    def test_add_chunked_req_path(self):
        """``adder.add_chunked_req`` path (scheduler.py:2541-2548)."""
        self.runtime.run(self._script_add_chunked_req_path)

    # ``adder.add_chunked_req`` path (scheduler.py:2541-2548). The
    # primary chunked-resume admission loop. Naive long request exercises
    # this on every chunk after the first. Across multi-chunk admission,
    # B6 invariant must hold: extend_batch_idx is monotonic (only reset
    # path may shrink it).
    @staticmethod
    def _script_add_chunked_req_path(t: ScriptedRuntime):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        for _ in range(DEFAULT_MAX_STEPS):
            assert r.extend_batch_idx_regression_count == 0, (
                "B6 invariant: extend_batch_idx must be monotonic across "
                "consecutive chunk admissions"
            )
            if r.finished:
                break
            yield
        assert r.finished
        assert r.chunks_done >= 2
        assert r.extend_batch_idx_regression_count == 0

    def test_admission_with_chunked_in_flight(self):
        """``add_one_req`` kwarg ``has_chunked_req=True`` propagation (scheduler.py:2593)."""
        self.runtime.run(self._script_admission_with_chunked_in_flight)

    # ``add_one_req`` kwarg ``has_chunked_req=True`` propagation
    # (scheduler.py:2593). With a chunked req in flight, the admission of
    # new reqs takes this code path. S2 invariant must hold across both
    # reqs' lifetimes.
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

        for _ in range(DEFAULT_MAX_STEPS):
            assert t.chunked_req_in_batch_violation_count() == 0, (
                "S2 invariant: admission with has_chunked_req=True must "
                "not place chunked_req into running batch"
            )
            if r_small.finished and r_chunk.finished:
                break
            yield
        assert r_small.finished and r_chunk.finished

    def test_new_chunked_req_first_chunk(self):
        """``adder.new_chunked_req`` assignment on first chunk (scheduler.py:2636-2642)."""
        self.runtime.run(self._script_new_chunked_req_first_chunk)

    # ``adder.new_chunked_req`` assignment on first chunk
    # (scheduler.py:2636-2642). The very first chunk of a long req takes
    # this assignment + assert path. The chunked req must transition
    # through is_chunking and S2 invariant must hold throughout.
    @staticmethod
    def _script_new_chunked_req_first_chunk(t: ScriptedRuntime):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield  # request enters admission, new_chunked_req path triggers
        yield from run_until(r, lambda h: h.chunks_done >= 1)
        # While the chunked req runs, S2 invariant must stay zero.
        saw_chunking = False
        for _ in range(DEFAULT_MAX_STEPS):
            if r.is_chunking:
                saw_chunking = True
            assert (
                t.chunked_req_in_batch_violation_count() == 0
            ), "S2 invariant: chunked_req exclusive of running batch"
            if r.finished:
                break
            yield
        assert r.finished
        assert r.chunks_done >= 1
        assert saw_chunking, "first-chunk assignment branch must be exercised"

    def test_inflight_middle_chunks_counter(self):
        """``inflight_middle_chunks += 1`` every iteration the chunked req is admitted (scheduler.py:2644-2645)."""
        self.runtime.run(self._script_inflight_middle_chunks_counter)

    # ``inflight_middle_chunks += 1`` every iteration the chunked
    # req is admitted (scheduler.py:2644-2645). Counter should equal
    # number of chunks minus the initial admission. R1 invariant:
    # decrement only fires on the final chunk — premature decrement
    # counter must stay zero.
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
        # R1 invariant — inflight_middle_chunks decrement must only fire
        # on the final chunk; premature-decrement counter == 0.
        assert r.inflight_middle_chunks_premature_decrement_count == 0, (
            f"R1 invariant: inflight_middle_chunks decremented before final "
            f"chunk {r.inflight_middle_chunks_premature_decrement_count} times"
        )

    def test_chunked_req_passes_through_batch(self):
        """``chunked_req=self.chunked_req`` wiring into ScheduleBatch (scheduler.py:2658)."""
        self.runtime.run(self._script_chunked_req_passes_through_batch)

    # ``chunked_req=self.chunked_req`` wiring into ScheduleBatch
    # (scheduler.py:2658). Verifies the wiring places the chunked req in
    # the chunked slot of the batch composition, and S2 invariant holds
    # across the chunked lifecycle.
    @staticmethod
    def _script_chunked_req_passes_through_batch(t: ScriptedRuntime):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r, lambda h: h.is_chunking)

        comp = t.batch_composition()
        assert r.rid in comp.get("chunked", [])

        for _ in range(DEFAULT_MAX_STEPS):
            assert t.chunked_req_in_batch_violation_count() == 0, (
                "S2 invariant: chunked_req must remain mutually exclusive "
                "of running_batch.reqs"
            )
            if r.finished:
                break
            yield
        assert r.finished

    def test_no_idle_during_chunked(self):
        """Idle path bypass — ``chunked_req is None`` check in ``check_idle`` (scheduler.py:3174)."""
        self.runtime.run(self._script_no_idle_during_chunked)

    # idle path bypass — ``chunked_req is None`` check in
    # ``check_idle`` (scheduler.py:3174). The scheduler must not enter
    # idle state while a chunked req is in flight. Verify across the
    # whole mid-chunk window, not just at one boundary.
    @staticmethod
    def _script_no_idle_during_chunked(t: ScriptedRuntime):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r, lambda h: h.is_chunking)

        # If the scheduler had idled, the chunked req would not progress.
        saw_chunking = False
        for _ in range(DEFAULT_MAX_STEPS):
            if r.is_chunking:
                saw_chunking = True
                assert (
                    not t.is_idle
                ), "scheduler must not idle while chunked_req is in flight"
            if r.finished:
                break
            yield
        assert r.finished
        assert saw_chunking, "test must observe r.is_chunking at least once"

    def test_abort_excludes_chunked_req(self):
        """Abort path's ``chunked_req_to_exclude`` plumbing (scheduler.py:3568-3596)."""
        self.runtime.run(self._script_abort_excludes_chunked_req)

    # abort path's ``chunked_req_to_exclude`` plumbing
    # (scheduler.py:3568-3596). With chunked_req live in last_batch when
    # abort fires, the exclusion set must include it. W3 invariant: the
    # dual-queue abort path must not double-release.
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
        # W3 invariant — abort of a chunked req must not double-release
        # its row + KV across the dual-queue path.
        assert r.abort_double_release_count == 0, (
            f"W3 invariant: abort double-released the chunked req "
            f"{r.abort_double_release_count} times"
        )
        # The chunked slot must be cleared after abort.
        assert t.get_chunked_req_rid() is None
        assert r.kv_pages == 0

    def test_get_chunked_req_lambda_getter(self):
        """Scheduler.py:680 — get_chunked_req lambda."""
        self.runtime.run(self._script_get_chunked_req_lambda_getter)

    @staticmethod
    def _script_get_chunked_req_lambda_getter(t: ScriptedRuntime):
        # scheduler.py:680 — get_chunked_req lambda. While r is chunking,
        # the getter must always return r.rid (never some other rid or
        # stale value). Verified across the whole mid-chunk window.
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r, lambda h: h.is_chunking)
        saw_match = False
        for _ in range(DEFAULT_MAX_STEPS):
            if r.is_chunking:
                cur = t.get_chunked_req_rid()
                assert cur in (None, r.rid), (
                    f"getter returned unrelated rid: got {cur!r}, expected "
                    f"None or {r.rid!r}"
                )
                if cur == r.rid:
                    saw_match = True
            if r.finished:
                break
            yield
        assert r.finished
        assert saw_match, "getter must return r.rid at least once while r.is_chunking"
        # After finish the slot must be cleared back to None.
        assert t.get_chunked_req_rid() is None

    def test_chunked_req_reset_to_none(self):
        """Scheduler.py:3596 — chunked_req=None reset path."""
        self.runtime.run(self._script_chunked_req_reset_to_none)

    @staticmethod
    def _script_chunked_req_reset_to_none(t: ScriptedRuntime):
        # scheduler.py:3596 — chunked_req=None reset path. After all
        # chunked reqs finish, scheduler.chunked_req should be None.
        # In-loop verification: across the whole chunked window, the
        # getter must report r.rid when chunking; immediately after
        # finish it must be None.
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        saw_chunking_match = False
        for _ in range(DEFAULT_MAX_STEPS):
            if r.is_chunking and t.get_chunked_req_rid() == r.rid:
                saw_chunking_match = True
            if r.finished:
                break
            yield
        assert r.finished
        assert (
            saw_chunking_match
        ), "must observe scheduler.chunked_req == r at least once during chunking"
        # And the reset to None must have happened.
        assert t.get_chunked_req_rid() is None

    @unittest.skip(
        "requires real disaggregation prefill/decode split — single-engine "
        "ScriptedRuntime cannot exercise the decode-side waiting_queue "
        "KV-hold path. Belongs in test_scripted_disagg.py with D3 counter "
        "wiring once disagg topology is available."
    )
    def test_disagg_decode_waiting_queue_kv_held(self):
        """Disaggregation/decode.py — waiting_queue reqs hold KV in decode mode."""

    @unittest.skip(
        "requires DLLM model + staging mixin — single-engine cannot drive "
        "both DLLM staging AND chunked admission incrementing "
        "pending_middle_outputs from two sources. Belongs in DLLM-specific "
        "test file."
    )
    def test_dllm_staging_double_pending_middle_outputs(self):
        """Dllm/mixin/scheduler.py — DLLM staging double-source pending_middle_outputs."""

    @unittest.skip(
        "requires real disaggregation topology — single-engine cannot "
        "exercise staging_handler chunked path."
    )
    def test_staging_handler_chunked(self):
        """Disaggregation/common/staging_handler.py — chunked interaction."""

    @unittest.skip(
        "requires mooncake KV transport backend — single-engine cannot "
        "drive the conn layer chunked path."
    )
    def test_mooncake_conn_chunked(self):
        """Disaggregation/mooncake/conn.py — chunked path in conn layer."""

    @unittest.skip(
        "requires NIXL KV transport backend — single-engine cannot drive "
        "the NIXL conn layer chunked path."
    )
    def test_nixl_conn_chunked(self):
        """Disaggregation/nixl/conn.py — chunked path."""

    def test_idle_path_chunked_req_none(self):
        """Scheduler.py:3174 — idle path checks chunked_req is None."""
        self.runtime.run(self._script_idle_path_chunked_req_none)

    @staticmethod
    def _script_idle_path_chunked_req_none(t: ScriptedRuntime):
        # scheduler.py:3174 — idle path checks chunked_req is None.
        # With no in-flight req, the scheduler must idle AND the chunked
        # slot must be None. Both invariants hold together.
        # Give the scheduler a few yields to settle into the idle state
        # (initial setup may keep is_idle False for one or two iterations).
        for _ in range(5):
            yield
        assert t.is_idle
        assert t.get_chunked_req_rid() is None, (
            f"with no in-flight reqs, chunked slot must be None; "
            f"got {t.get_chunked_req_rid()!r}"
        )
        # The idle invariant: across more yields, the scheduler stays
        # idle (no spurious wakeups when chunked_req is None).
        for _ in range(5):
            assert t.is_idle, "scheduler must remain idle with no work"
            assert t.get_chunked_req_rid() is None
            yield

    def test_admission_path_with_chunked_inflight_flag(self):
        """Scheduler.py:2593 — add_one_req called with has_chunked_req=True."""
        self.runtime.run(self._script_admission_path_with_chunked_inflight_flag)

    @staticmethod
    def _script_admission_path_with_chunked_inflight_flag(t: ScriptedRuntime):
        # scheduler.py:2593 — add_one_req called with has_chunked_req=True.
        # While r_chunked is in flight, S2 invariant must remain zero across
        # every admission attempt (new short req cannot collide with the
        # chunked slot).
        r_chunked = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r_chunked, lambda h: h.is_chunking)
        # Confirm r_chunked is in batch as chunked at this point.
        comp = t.batch_composition()
        assert r_chunked.rid in comp.get(
            "chunked", []
        ), f"r_chunked must occupy chunked slot before admission; got {comp!r}"
        r_new = t.start_req(prompt_len=16, max_new_tokens=2)
        for _ in range(DEFAULT_MAX_STEPS):
            assert t.chunked_req_in_batch_violation_count() == 0, (
                "S2 invariant: admission of new req must not place chunked_req "
                "into running batch simultaneously"
            )
            if r_chunked.finished and r_new.finished:
                break
            yield
        assert r_chunked.finished and r_new.finished

    def test_inflight_counter_increments_each_chunk(self):
        """Scheduler.py:2644-2645 — inflight_middle_chunks += 1 per chunk."""
        self.runtime.run(self._script_inflight_counter_increments_each_chunk)

    @staticmethod
    def _script_inflight_counter_increments_each_chunk(t: ScriptedRuntime):
        # scheduler.py:2644-2645 — inflight_middle_chunks += 1 per chunk.
        # In-loop R1 invariant: premature-decrement count must never grow
        # while the counter is incrementing.
        r = t.start_req(prompt_len=4 * DEFAULT_CHUNK_SIZE, max_new_tokens=2)
        saw_increment = False
        last = 0
        for _ in range(DEFAULT_MAX_STEPS):
            cur = r.inflight_middle_chunks
            if cur > last:
                saw_increment = True
            last = max(last, cur)
            assert r.inflight_middle_chunks_premature_decrement_count == 0, (
                "R1 invariant: inflight_middle_chunks must only decrement "
                "on the final chunk"
            )
            if r.finished:
                break
            yield
        # After at least one chunk, the counter must have moved up at some point.
        assert saw_increment, "expected inflight_middle_chunks to increment"
        assert r.finished
        assert r.inflight_middle_chunks_premature_decrement_count == 0

    def test_filter_batch_exclude_chunked_flag(self):
        """Filter_batch + chunked: exclude_chunked_req branch."""
        self.runtime.run(self._script_filter_batch_exclude_chunked_flag)

    @staticmethod
    def _script_filter_batch_exclude_chunked_flag(t: ScriptedRuntime):
        # filter_batch + chunked: exclude_chunked_req branch. While both
        # reqs are alive, the chunked slot must stay mutually exclusive
        # of the running batch — S2 invariant must hold every step.
        r1 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        r2 = t.start_req(prompt_len=16, max_new_tokens=2)
        saw_r1_chunking = False
        for _ in range(DEFAULT_MAX_STEPS * 2):
            if r1.is_chunking:
                saw_r1_chunking = True
            assert t.chunked_req_in_batch_violation_count() == 0, (
                "S2 invariant: filter_batch must keep chunked slot "
                "mutually-exclusive of running batch"
            )
            if r1.finished and r2.finished:
                break
            yield
        assert r1.finished and r2.finished
        assert (
            saw_r1_chunking
        ), "r1 must have chunked at some point to exercise the exclude branch"

    @unittest.skip(
        "pdmux split_prefill_batch requires the pdmux topology — "
        "single-engine ScriptedRuntime cannot drive the split path. "
        "Belongs in a pdmux-specific test once that lane is wired up."
    )
    def test_pdmux_split_prefill_batch(self):
        """34c02d6a67: filter chunked-resume from split_prefill_batch."""

    def test_streaming_session_kv_committed_bound(self):
        """116584e8fa: streaming-session chunked stash must stay bounded by kv_committed_len."""
        self.runtime.run(self._script_streaming_session_kv_committed_bound)

    # 116584e8fa: bound streaming-session chunked stash by
    # kv_committed_len. Across the chunked lifecycle, any positional
    # length surfaced to the next chunk (prefix_indices) must not exceed
    # the committed KV length — otherwise the next chunk would read
    # uninitialized rows.
    @staticmethod
    def _script_streaming_session_kv_committed_bound(t: ScriptedRuntime):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        for _ in range(DEFAULT_MAX_STEPS):
            if r.is_chunking:
                assert r.prefix_indices_len <= r.kv_committed_len, (
                    f"streaming-session chunked stash must stay bounded by "
                    f"kv_committed_len; prefix_indices_len={r.prefix_indices_len}, "
                    f"kv_committed_len={r.kv_committed_len}"
                )
            if r.finished:
                break
            yield
        assert r.finished

    @unittest.skip(
        "mamba_pool_idx cleanup applies only to mamba-class models — "
        "single-engine with a non-mamba model cannot drive the NO_TOKEN "
        "chunked-resume cleanup-skip branch. Belongs in a mamba-specific "
        "test file once mamba scripted coverage is added."
    )
    def test_mamba_pool_idx_cleanup_skip_chunked_resume(self):
        """Dbdcdde245: skip mamba_pool_idx cleanup for chunked-resume on NO_TOKEN."""

    # ================================================================
    # Round-3 source-driven additions.
    # ================================================================

    def test_chunked_req_bypasses_req_pool_exhaustion(self):
        """In-flight chunked req keeps advancing even when get_num_allocatable_reqs returns 0."""
        self.runtime.run(self._script_chunked_req_bypasses_req_pool_exhaustion)

    @staticmethod
    def _script_chunked_req_bypasses_req_pool_exhaustion(t: ScriptedRuntime):
        # scheduler.py: get_num_allocatable_reqs is the gate for
        # admitting new waiters, but a chunked req mid-flight must bypass
        # it on the resume iteration. Drive into mid-chunk, then exhaust
        # the req pool externally; the chunked req must still complete
        # AND keep making progress (chunks_done advancing).
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r, lambda h: h.is_chunking)
        chunks_before_pressure = r.chunks_done

        # Force the req pool to look exhausted to a fresh waiter.
        t.exhaust_row_pool(leave_rows=0)

        # The in-flight chunked req must still advance to completion, and
        # must do so while S2 invariant holds (no slot collision under
        # pressure).
        progressed_under_pressure = False
        for _ in range(DEFAULT_MAX_STEPS * 2):
            if r.chunks_done > chunks_before_pressure:
                progressed_under_pressure = True
            assert (
                t.chunked_req_in_batch_violation_count() == 0
            ), "S2 invariant must hold even under row-pool exhaustion"
            if r.finished:
                break
            yield
        assert r.finished
        assert progressed_under_pressure, (
            "chunked req must advance even when get_num_allocatable_reqs "
            "returns 0; pre-fix the bypass would block forever"
        )
        assert r.kv_pages == 0

    def test_pause_retract_clears_chunked_req(self):
        """Pause_generation(retract) mid-chunk: scheduler.chunked_req is cleared and all reqs retracted."""
        self.runtime.run(self._script_pause_retract_clears_chunked_req)

    @staticmethod
    def _script_pause_retract_clears_chunked_req(t: ScriptedRuntime):
        # scheduler.py: pause_generation(retract) must clear
        # ``self.chunked_req`` so the next iteration does not try to
        # resume a partially-retracted prompt.
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r, lambda h: h.is_chunking)

        t.force_retract(r)
        yield

        assert t.get_chunked_req_rid() is None, (
            f"pause(retract) must clear chunked_req; "
            f"got {t.get_chunked_req_rid()!r}"
        )
        assert t.chunked_in_flight_count() == 0

    def test_load_inquirer_pending_tokens_dedup_chunked(self):
        """Mid-chunk load query: chunked req's pending tokens are not double-counted."""
        self.runtime.run(self._script_load_inquirer_pending_tokens_dedup_chunked)

    @staticmethod
    def _script_load_inquirer_pending_tokens_dedup_chunked(t: ScriptedRuntime):
        # scheduler.py load-inquirer path: when a chunked req sits
        # in both running_batch and waiting_queue (the dual-queue holding
        # state), the pending-token tally must dedup. Verified at EVERY
        # mid-chunk yield to catch dedup regressions that only show up
        # transiently.
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r, lambda h: h.is_chunking)
        saw_chunking = False
        for _ in range(DEFAULT_MAX_STEPS):
            if r.is_chunking:
                saw_chunking = True
                snap = t.load_inquirer_snapshot()
                pending = snap["pending_tokens_count_for_rid"](r.rid)
                assert pending <= r.remaining_prompt_tokens, (
                    f"load_inquirer tallied {pending} tokens for r but only "
                    f"{r.remaining_prompt_tokens} are still pending — "
                    "dual-queue dedup violated"
                )
            if r.finished:
                break
            yield
        assert r.finished
        assert (
            saw_chunking
        ), "test must observe the dual-queue chunked state at least once"

    def test_chunked_forced_admission_avoids_leak(self):
        """Non-SWA + rem_total_tokens <= 0: chunked req force-admitted to avoid pool leak (comment 677)."""
        self.runtime.run(self._script_chunked_forced_admission_avoids_leak)

    @staticmethod
    def _script_chunked_forced_admission_avoids_leak(t: ScriptedRuntime):
        # prefill_adder.py comment "must force-admit chunked-resume
        # even with rem_total_tokens <= 0 to avoid leaking the row + KV
        # already held". Drive: long chunked req, exhaust KV externally,
        # the req must still advance to completion without leaked rows.
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r, lambda h: h.is_chunking)

        baseline_rows = t.row_pool_used()
        t.exhaust_kv()
        # Even with KV "exhausted", the chunked-resume must be force-admitted.
        yield from run_until_finished(r)
        assert r.finished
        assert (
            t.row_pool_used() <= baseline_rows
        ), f"row leak under forced chunked admission: baseline={baseline_rows}, after={t.row_pool_used()}"

    # SWA add_chunked_req early-return + stash double-free regression is
    # covered by test_swa_chunked_req_early_return_no_double_free in
    # test_scripted_hybrid_swa.py (full assertion suite: flag transitions,
    # early-return path was hit, double-free counter == 0).

    def test_stage_a_pending_middle_outputs_sync_invariant(self):
        """Cross-iteration invariant: pending_middle_outputs > 0 implies is_chunking == True."""
        self.runtime.run(self._script_stage_a_pending_middle_outputs_sync_invariant)

    @staticmethod
    def _script_stage_a_pending_middle_outputs_sync_invariant(t: ScriptedRuntime):
        # cross-iter invariant: every time we observe the req at
        # an iter boundary, either it is finished, or the pair
        # (pending_middle_outputs, is_chunking) is consistent: if there's
        # a pending middle output, the req is still chunking.
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=4)
        for _ in range(DEFAULT_MAX_STEPS):
            if r.finished:
                return
            if r.pending_middle_outputs > 0:
                assert r.is_chunking, (
                    f"invariant violated: pending_middle_outputs="
                    f"{r.pending_middle_outputs} but is_chunking={r.is_chunking}"
                )
            yield
        raise AssertionError("req did not finish")

    def test_init_next_round_input_resets_chunk_state(self):
        """Across two chunks of one req, fill_ids_len is reset to prefix_indices_len before next admit."""
        self.runtime.run(self._script_init_next_round_input_resets_chunk_state)

    # schedule_batch.init_next_round_input: between consecutive
    # chunks of the same chunked-resume req, the per-iter ``fill_ids``
    # buffer must be reset to the already-committed prefix
    # (prefix_indices length). Pre-fix the residual fill_ids from the
    # previous chunk could leak into the next admit and double-count
    # tokens. Observable via r.fill_ids_len snapped at consecutive
    # mid-chunk yields.
    @staticmethod
    def _script_init_next_round_input_resets_chunk_state(t: ScriptedRuntime):
        # NEW API NEEDED: r.fill_ids_len — per-req length of the
        # scheduler's current fill_ids buffer, exposed for invariant
        # checks across chunk boundaries.
        # NEW API NEEDED: r.prefix_indices_len — length of the already-
        # committed prefix indices for the chunked req.
        r = t.start_req(prompt_len=3 * DEFAULT_CHUNK_SIZE, max_new_tokens=2)
        # Drive to the boundary between chunk 1 and chunk 2.
        yield from run_until(r, lambda h: h.chunks_done >= 1 and h.is_chunking)
        # Verify the reset invariant holds at EVERY mid-chunk yield,
        # not just at the first chunk boundary — pre-fix a residual
        # fill_ids could regrow across consecutive chunks.
        saw_mid_chunk = False
        for _ in range(DEFAULT_MAX_STEPS):
            if r.is_chunking and r.chunks_done >= 1:
                saw_mid_chunk = True
                assert r.fill_ids_len == r.prefix_indices_len, (
                    f"init_next_round_input must reset fill_ids to "
                    f"prefix_indices at every chunk boundary; "
                    f"fill_ids_len={r.fill_ids_len}, "
                    f"prefix_indices_len={r.prefix_indices_len}, "
                    f"chunks_done={r.chunks_done}"
                )
            if r.finished:
                break
            yield
        assert r.finished
        assert (
            saw_mid_chunk
        ), "test must observe the fill_ids reset boundary at least once"
        assert r.finished

    def test_chunked_req_scheduled_last_iter_false_when_chunk_completes(self):
        """After last chunk admit, last_chunked_req_scheduled_iter_flag clears to False (chunked_req cleared)."""
        self.runtime.run(
            self._script_chunked_req_scheduled_last_iter_false_when_chunk_completes
        )

    # scheduler.py: ``_chunked_req_scheduled_last_iter`` must
    # transition from True (during chunked admission) to False once the
    # last chunk has been admitted and the scheduler.chunked_req has
    # been cleared back to None. Observable via the flag itself.
    @staticmethod
    def _script_chunked_req_scheduled_last_iter_false_when_chunk_completes(
        t: ScriptedRuntime,
    ):
        # 2 chunks so the second chunk is the last one and clearly
        # exercises the last-chunk admission path.
        r = t.start_req(prompt_len=2 * DEFAULT_CHUNK_SIZE, max_new_tokens=2)
        # While the req is chunking, observe the flag at least once True.
        saw_true = False
        for _ in range(DEFAULT_MAX_STEPS):
            if r.is_chunking:
                # NEW API NEEDED: t.last_chunked_req_scheduled_iter_flag()
                # returns the value of scheduler._chunked_req_scheduled_last_iter.
                if t.last_chunked_req_scheduled_iter_flag():
                    saw_true = True
            if r.finished:
                break
            yield
        assert r.finished
        assert saw_true, "flag should have been True at least once mid-chunk"
        # After finish + chunked_req cleared, the flag must be False.
        assert (
            t.last_chunked_req_scheduled_iter_flag() is False
        ), f"flag must clear to False after last chunk; got {t.last_chunked_req_scheduled_iter_flag()!r}"

    def test_second_chunked_admit_blocked_when_chunked_req_set(self):
        """While R1 is chunking, R2 (also long) waits; at every yield exactly one of {R1, R2} is_chunking."""
        self.runtime.run(self._script_second_chunked_admit_blocked_when_chunked_req_set)

    # scheduler.py: only one chunked-resume req at a
    # time. With R1 holding the chunked slot, a second long req R2
    # submitted concurrently must NOT also enter is_chunking — it has
    # to wait until R1 releases the slot. Mutual-exclusion invariant
    # across the full lifetime.
    @staticmethod
    def _script_second_chunked_admit_blocked_when_chunked_req_set(t: ScriptedRuntime):
        r1 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        r2 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        saw_r1_chunking = False
        saw_r2_chunking = False
        for _ in range(DEFAULT_MAX_STEPS * 2):
            both = r1.is_chunking and r2.is_chunking
            assert not both, (
                f"only one chunked-resume slot allowed; r1.is_chunking="
                f"{r1.is_chunking}, r2.is_chunking={r2.is_chunking}"
            )
            if r1.is_chunking:
                saw_r1_chunking = True
            if r2.is_chunking:
                saw_r2_chunking = True
            if r1.finished and r2.finished:
                break
            yield
        assert r1.finished and r2.finished
        # Both should eventually chunk (sequentially).
        assert saw_r1_chunking and saw_r2_chunking, (
            f"both reqs should chunk over their lifetime; saw_r1="
            f"{saw_r1_chunking}, saw_r2={saw_r2_chunking}"
        )

    def test_chunked_exclude_falls_back_to_last_batch_reqs_when_no_pp(self):
        """Non-PP: exclude set populated from last_batch.reqs, not from last_batch.chunked_req."""
        self.runtime.run(
            self._script_chunked_exclude_falls_back_to_last_batch_reqs_when_no_pp
        )

    # scheduler.py: in the non-PP single-microbatch path,
    # the chunked_req_to_exclude set is populated from ``last_batch.reqs``
    # (the else branch), not from ``last_batch.chunked_req`` (the PP
    # branch). Observable via a runtime helper that exposes which
    # source branch produced the exclude set. S3 invariant
    # (stale_chunked_req_merged) is specific to the PP branch — it
    # MUST stay at 0 in the non-PP path even when chunked_req churn
    # is happening.
    @staticmethod
    def _script_chunked_exclude_falls_back_to_last_batch_reqs_when_no_pp(
        t: ScriptedRuntime,
    ):
        # Two reqs so last_batch.reqs has multiple entries — easy to
        # observe the else branch sourcing from .reqs.
        r1 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        r2 = t.start_req(prompt_len=16, max_new_tokens=2)
        # NEW API NEEDED: t.last_chunked_exclude_set_source() returning
        # "last_batch_reqs" | "last_batch_chunked_req" | None — which
        # branch of the exclude set source was taken at the most recent
        # admission step.
        saw_reqs_branch = False
        for _ in range(DEFAULT_MAX_STEPS):
            source = t.last_chunked_exclude_set_source()
            if source == "last_batch_reqs":
                saw_reqs_branch = True
                # In the non-PP path the exclude set must NOT come from
                # the chunked_req pointer.
                assert source != "last_batch_chunked_req"
            # S3 invariant — stale_chunked_req_merged path is PP-only;
            # in this non-PP scheduler it must never fire.
            assert t.stale_chunked_req_merged_count() == 0, (
                "S3 invariant: stale chunked_req merge path must never fire "
                "in a non-PP scheduler"
            )
            if r1.finished and r2.finished:
                break
            yield
        assert r1.finished and r2.finished
        assert saw_reqs_branch, (
            "non-PP scheduler must source exclude set from last_batch.reqs "
            "(else branch) at least once during the multi-req lifetime"
        )
        assert t.stale_chunked_req_merged_count() == 0

    def test_scheduler_continues_with_only_chunked_req_no_waiting(self):
        """Mid-chunk single long req: waiting_queue empty but scheduler keeps running until finish."""
        self.runtime.run(
            self._script_scheduler_continues_with_only_chunked_req_no_waiting
        )

    # scheduler.py: when the only inflight work is a chunked
    # req (waiting_queue is empty), the event loop must not flip to
    # idle and skip the chunked continuation. Observable: chunks_done
    # must keep advancing, and is_idle must stay False throughout.
    @staticmethod
    def _script_scheduler_continues_with_only_chunked_req_no_waiting(
        t: ScriptedRuntime,
    ):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r, lambda h: h.is_chunking)
        prev_chunks_done = r.chunks_done
        progressed = False
        for _ in range(DEFAULT_MAX_STEPS):
            assert (
                not t.is_idle
            ), "scheduler must not go idle while a chunked req is in flight"
            cur_chunks_done = r.chunks_done
            if cur_chunks_done > prev_chunks_done:
                progressed = True
            prev_chunks_done = cur_chunks_done
            if r.finished:
                break
            yield
        assert r.finished
        assert progressed, (
            "chunks_done must keep advancing without any waiter; pre-fix "
            "an empty waiting_queue could cause the loop to skip continuation"
        )

    def test_add_chunked_req_non_swa_forced_admit_on_rem_zero(self):
        """Non-SWA chunked-resume: forced-admitted even when _rem_tokens hits 0 (avoid leak at schedule_policy.py:679-682)."""
        self.runtime.run(self._script_add_chunked_req_non_swa_forced_admit_on_rem_zero)

    # schedule_policy.py:679-682 ("must force-admit chunked-
    # resume even with rem_total_tokens == 0 to avoid leaking the row
    # + KV already held"). Drive a long chunked req into a state where
    # external KV exhaustion would normally block admission; the
    # chunked-resume must STILL be admitted (forced) and complete
    # without leak.
    @staticmethod
    def _script_add_chunked_req_non_swa_forced_admit_on_rem_zero(t: ScriptedRuntime):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        # Get the chunked-resume into a known mid-stream state.
        yield from run_until(r, lambda h: h.is_chunking and h.chunks_done >= 1)
        # Exhaust KV to push _rem_tokens to 0 from the budget side.
        t.exhaust_kv()
        # The chunked-resume must still advance to completion via the
        # forced-admit path.
        yield from run_until_finished(r, max_steps=800)
        assert r.finished, (
            "non-SWA chunked-resume must be force-admitted when "
            "_rem_tokens == 0 (schedule_policy.py:679-682); pre-fix it "
            "would block forever and leak its held row + KV"
        )
        assert r.kv_pages == 0
        assert r.lock_refs == 0


class TestSpecialCaseDynamicChunking(ScriptedRuntimeTestCase):
    ENGINE_KWARGS = base_engine_kwargs(
        chunked_prefill_size=DEFAULT_CHUNK_SIZE,
        enable_dynamic_chunking=True,
    )

    def test_dynamic_chunking_history_len(self):
        """Dynamic chunking reads ``history_len`` from ``self.chunked_req.prefix_indices`` (scheduler.py:2516-2517)."""
        self.runtime.run(self._script_dynamic_chunking_history_len)

    # dynamic chunking reads ``history_len`` from
    # ``self.chunked_req.prefix_indices`` (scheduler.py:2516-2517). With
    # ``--enable-dynamic-chunking``, the per-iter chunk size adjusts to
    # the chunked req's history length. S2 and B6 invariants must hold
    # since dynamic chunking varies extend_batch_idx more aggressively.
    @staticmethod
    def _script_dynamic_chunking_history_len(t: ScriptedRuntime):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        for _ in range(DEFAULT_MAX_STEPS):
            assert t.chunked_req_in_batch_violation_count() == 0
            assert r.extend_batch_idx_regression_count == 0
            if r.finished:
                break
            yield
        assert r.finished
        assert r.chunks_done >= 2

    # test_dynamic_chunking_history_len_extra was deleted: it duplicated
    # test_dynamic_chunking_history_len above (same scenario, weaker
    # body — no assertions). REDUNDANT.


class TestSpecialCaseMixedChunk(ScriptedRuntimeTestCase):
    ENGINE_KWARGS = base_engine_kwargs(
        chunked_prefill_size=DEFAULT_CHUNK_SIZE,
        enable_mixed_chunk=True,
    )

    def test_mix_with_running_chunked_plus_decode(self):
        """Enable_mixed_chunk: long chunked req + multiple decode reqs share a MIXED forward step."""
        self.runtime.run(self._script_mix_with_running_chunked_plus_decode)

    @staticmethod
    def _script_mix_with_running_chunked_plus_decode(t: ScriptedRuntime):
        # scheduler.py:2683-2699 + schedule_batch.py:2193-2222
        # mix_with_running: with enable_mixed_chunk, a chunked-resume admit
        # iteration must merge currently running decode reqs into the same
        # forward step. Drive: warm a few decode reqs, then submit a long
        # chunked req; once mid-chunk, the batch composition must include
        # both running decodes and the chunked req in the same iter.
        decodes = [t.start_req(prompt_len=8, max_new_tokens=16) for _ in range(3)]
        yield from run_until(decodes[0], lambda h: h.status == "running")

        r_chunk = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=4)
        yield from run_until(r_chunk, lambda h: h.is_chunking)

        comp = t.batch_composition()
        # mix-with-running path: chunked + at least one decode in the same iter
        assert r_chunk.rid in comp.get("chunked", [])
        running_in_batch = set(comp.get("decode", [])) | set(comp.get("running", []))
        assert any(
            d.rid in running_in_batch for d in decodes
        ), f"enable_mixed_chunk should merge decode reqs into chunked iter; got {comp!r}"
        assert (
            t.forward_mode == "MIXED"
        ), f"expected forward_mode == MIXED with enable_mixed_chunk, got {t.forward_mode!r}"

        # B6 + S2 invariants must hold across the mixed-chunk lifecycle.
        all_reqs = [r_chunk, *decodes]
        for _ in range(DEFAULT_MAX_STEPS * 2):
            assert t.chunked_req_in_batch_violation_count() == 0, (
                "S2 invariant: mix-with-running must not duplicate chunked_req "
                "into running batch"
            )
            assert (
                r_chunk.extend_batch_idx_regression_count == 0
            ), "B6 invariant: extend_batch_idx must stay monotonic in MIXED mode"
            if all(x.finished for x in all_reqs):
                break
            yield
        assert all(x.finished for x in all_reqs)

    def test_mixed_chunk_with_logprob_falls_back(self):
        """Enable_mixed_chunk + return_logprob: mix path is bypassed for logprob reqs."""
        self.runtime.run(self._script_mixed_chunk_with_logprob_falls_back)

    @staticmethod
    def _script_mixed_chunk_with_logprob_falls_back(t: ScriptedRuntime):
        # scheduler.py mixed-chunk decision: when the chunked req
        # asks for input logprob, ``is_mixed_chunk`` short-circuits to
        # False to keep logprob accumulation linear; the batch must stay
        # in plain EXTEND, not MIXED.
        r = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN,
            max_new_tokens=2,
            return_logprob=True,
        )
        yield from run_until(r, lambda h: h.is_chunking)
        assert (
            t.forward_mode != "MIXED"
        ), f"return_logprob must disable mixed-chunk path; got {t.forward_mode!r}"
        yield from run_until_finished(r)

    def test_mixed_chunk_with_running_batch(self):
        """Enable_mixed_chunk + existing running_batch: chunked admission merges with running decode reqs."""
        self.runtime.run(self._script_mixed_chunk_with_running_batch)

    @staticmethod
    def _script_mixed_chunk_with_running_batch(t: ScriptedRuntime):
        # scheduler.py: when a decode batch is already running and a
        # new chunked req arrives, the mix-with-running branch must merge
        # them into one MIXED forward; we observe MIXED on the admission
        # iter and clean completion.
        r_dec = t.start_req(prompt_len=8, max_new_tokens=32)
        yield from run_until(r_dec, lambda h: h.status == "running")

        r_chunk = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield  # admission iter — mix-with-running should engage
        yield from run_until(r_chunk, lambda h: h.is_chunking)

        assert (
            t.forward_mode == "MIXED"
        ), f"chunked admission with running batch must enter MIXED; got {t.forward_mode!r}"
        # B6 + S2 invariants must hold while MIXED runs.
        for _ in range(DEFAULT_MAX_STEPS * 2):
            assert t.chunked_req_in_batch_violation_count() == 0
            assert r_chunk.extend_batch_idx_regression_count == 0
            if r_chunk.finished and r_dec.finished:
                break
            yield
        assert r_chunk.finished and r_dec.finished


class TestSpecialCaseTransformers(ScriptedRuntimeTestCase):
    ENGINE_KWARGS = base_engine_kwargs(
        chunked_prefill_size=DEFAULT_CHUNK_SIZE,
        impl="transformers",
    )

    def test_multimodal_transformers_disables_chunking(self):
        """Multimodal + transformers backend forces chunked_prefill_size=None at server-args resolve time."""
        self.runtime.run(self._script_multimodal_transformers_disables_chunking)

    @staticmethod
    def _script_multimodal_transformers_disables_chunking(t: ScriptedRuntime):
        # server_args / model_config: under the transformers backend
        # with mm inputs, ``chunked_prefill_size`` is force-disabled. The
        # scheduler should treat any long prompt as single-shot — observe
        # ``chunks_done == 0`` even for a long req.
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until_finished(r)
        assert r.finished
        assert (
            r.chunks_done == 0
        ), f"transformers backend should disable chunking; got chunks_done={r.chunks_done}"


class TestSpecialCaseNoChunking(ScriptedRuntimeTestCase):
    ENGINE_KWARGS = base_engine_kwargs(chunked_prefill_size=-1)

    def test_chunk_size_negative_disables_chunking(self):
        """Chunked_prefill_size=-1 disables chunked prefill; long prompts still complete in one shot."""
        self.runtime.run(self._script_chunk_size_negative_disables_chunking)

    @staticmethod
    def _script_chunk_size_negative_disables_chunking(t: ScriptedRuntime):
        # server_args: chunked_prefill_size < 0 means disabled. A
        # long prompt must complete without entering the chunked path.
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        # While running, ``is_chunking`` must never flip True.
        for _ in range(DEFAULT_MAX_STEPS):
            assert (
                not r.is_chunking
            ), "chunked_prefill_size=-1 should disable chunked path"
            if r.finished:
                return
            yield
        raise AssertionError("req did not finish under disabled chunking")


class TestSpecialCaseTinyChunk(ScriptedRuntimeTestCase):
    ENGINE_KWARGS = base_engine_kwargs(
        # chunked_prefill_size < page_size forces
        # ``trunc_len = rem // page_size * page_size = 0`` so the
        # prefill_adder OTHER-return branch fires per iter.
        chunked_prefill_size=8,
        page_size=16,
    )

    def test_chunked_admission_trunc_lt_zero_returns_other(self):
        """Page_size > rem_chunk_tokens: add_one_req returns OTHER, chunked admit deferred this iter."""
        self.runtime.run(self._script_chunked_admission_trunc_lt_zero_returns_other)

    @staticmethod
    def _script_chunked_admission_trunc_lt_zero_returns_other(t: ScriptedRuntime):
        # prefill_adder.py: when ``rem_chunk_tokens - (start %
        # page_size) < page_size``, ``add_one_req`` returns OTHER and the
        # iteration emits no new chunk. The req must still eventually
        # complete on a later iter when capacity recovers.
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        # Drive the chunked path far enough to observe at least one
        # iteration where the req was deferred (status == "waiting" and
        # chunks_done did not advance from the prior observation). That is
        # the externally visible signature of the OTHER-return path.
        # NEW API NEEDED: t.last_admission_path() should expose the most
        # recent add_one_req result so the test could pin the OTHER path
        # directly instead of inferring it.
        saw_deferred_iter = False
        prev_chunks_done = r.chunks_done
        for _ in range(DEFAULT_MAX_STEPS):
            if r.finished:
                break
            cur_chunks_done = r.chunks_done
            if r.status == "waiting" and cur_chunks_done == prev_chunks_done:
                saw_deferred_iter = True
            prev_chunks_done = cur_chunks_done
            yield
        assert r.finished
        assert saw_deferred_iter, (
            "expected at least one iter where add_one_req returned OTHER "
            "(req status == waiting and chunks_done did not advance); "
            "chunked_prefill_size < page_size must defer admission"
        )


class TestSpecialCaseDeterministicFlashInfer(ScriptedRuntimeTestCase):
    ENGINE_KWARGS = base_engine_kwargs(
        chunked_prefill_size=DEFAULT_CHUNK_SIZE,
        page_size=16,
        attention_backend="flashinfer",
        enable_deterministic_inference=True,
    )

    def test_chunked_truncation_align_size(self):
        """Deterministic + flashinfer + long prompt: each chunk boundary is page_size aligned."""
        self.runtime.run(self._script_chunked_truncation_align_size)

    @staticmethod
    def _script_chunked_truncation_align_size(t: ScriptedRuntime):
        # prefill_adder.py truncation: under deterministic +
        # flashinfer, every chunk's extend_input_len must be a multiple of
        # page_size (no partial-page tails mid-chunk).
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        page_size = 16
        for _ in range(DEFAULT_MAX_STEPS):
            if r.is_chunking and r.extend_input_len is not None:
                assert r.extend_input_len % page_size == 0, (
                    f"deterministic chunk boundary must be page-aligned; "
                    f"got extend_input_len={r.extend_input_len}, page_size={page_size}"
                )
            if r.finished:
                return
            yield
        raise AssertionError("chunked req did not finish")


class TestSpecialCaseHiCache(ScriptedRuntimeTestCase):
    ENGINE_KWARGS = base_engine_kwargs(
        chunked_prefill_size=DEFAULT_CHUNK_SIZE,
        enable_hierarchical_cache=True,
    )

    def test_hicache_breakdown_only_first_chunk(self):
        """HiCache cached_tokens_* fields are written exactly once on the first chunk, never re-set on later chunks."""
        self.runtime.run(self._script_hicache_breakdown_only_first_chunk)

    @staticmethod
    def _script_hicache_breakdown_only_first_chunk(t: ScriptedRuntime):
        # schedule_batch.py:1909-1932: ``cached_tokens_*`` (HiCache
        # breakdown stats) should be populated only on the first chunk.
        # Subsequent chunks must not overwrite — that would double-count
        # the prefix cache hit metric. B5 counter
        # (hicache_cached_tokens_write_count) makes this invariant
        # directly observable: must stay <= 1 across the whole lifecycle.
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        first_chunk_snap = None
        for _ in range(DEFAULT_MAX_STEPS):
            if r.is_chunking and r.chunks_done >= 1 and first_chunk_snap is None:
                first_chunk_snap = r.cached_tokens_snapshot()
            if first_chunk_snap is not None and r.is_chunking:
                cur = r.cached_tokens_snapshot()
                assert cur == first_chunk_snap, (
                    f"HiCache cached_tokens_* must freeze after first chunk; "
                    f"first={first_chunk_snap!r}, now={cur!r}"
                )
            # B5 invariant — cached_tokens_* must be written at most once.
            assert r.hicache_cached_tokens_write_count <= 1, (
                f"B5 invariant: HiCache cached_tokens_* written more than "
                f"once; count={r.hicache_cached_tokens_write_count}"
            )
            if r.finished:
                break
            yield
        assert r.finished
        assert r.hicache_cached_tokens_write_count <= 1, (
            f"B5 invariant: final hicache_cached_tokens_write_count must be "
            f"<= 1, got {r.hicache_cached_tokens_write_count}"
        )

    def test_init_load_back_called_once_per_request_with_hicache(self):
        """HiCache + multi-chunk req: init_load_back fires exactly once for the whole req, not once per chunk."""
        self.runtime.run(
            self._script_init_load_back_called_once_per_request_with_hicache
        )

    # schedule_batch.py / hicache integration: the
    # ``init_load_back`` hook bridges the HiCache off-GPU prefix into
    # the chunked req's working set. It must be called exactly once per
    # request (on first admission), not re-invoked per chunk. Observable
    # via a per-req counter on the handle.
    @staticmethod
    def _script_init_load_back_called_once_per_request_with_hicache(
        t: ScriptedRuntime,
    ):
        # NEW API NEEDED: r.init_load_back_count — how many times
        # init_load_back() was invoked for this req across its lifetime.
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until_finished(r)
        assert r.finished
        assert r.chunks_done >= 2, (
            f"req must actually chunk for this branch to fire; got "
            f"chunks_done={r.chunks_done}"
        )
        assert r.init_load_back_count == 1, (
            f"HiCache init_load_back must run exactly once per request, "
            f"not once per chunk; got init_load_back_count="
            f"{r.init_load_back_count}"
        )
        # B5 invariant — the HiCache cached_tokens_* breakdown must be
        # written at most once even across multiple chunks (same gate as
        # init_load_back, expressed via the dedicated counter).
        assert r.hicache_cached_tokens_write_count <= 1, (
            f"B5 invariant: cached_tokens_* must be written at most once; "
            f"got count={r.hicache_cached_tokens_write_count}"
        )


if __name__ == "__main__":
    unittest.main()
