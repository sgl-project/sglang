"""Priority scheduling × chunked: naive ScriptedRuntime smoke plus retract/preempt edge cases.

Submit a low-priority long-prompt request that must be chunked, then
a high-priority short request. With priority preemption enabled the
high-priority req should not starve waiting on the low-priority one's
chunk loop.

Requires the wishlist API extension ``start_req(..., priority=...)``
(§4 P2 (10)). Until it lands the priority kwargs are passed as-is and
will surface as a clear AttributeError at script time.

Edge cases verify that chunked-resume requests participate in
KV-pressure retract and priority preemption like normal requests, and
that resource ownership is correctly released across the transition.
Also covers A.7 series from the expansion plan and fan-out variations
(retract at every chunk position, retract + immediate re-admit,
multi-victim, retract during decode, retract-then-abort).
"""

import time
import unittest

from sglang.srt.environ import envs
from sglang.test.scripted_runtime.runtime import ScriptedRuntime
from sglang.test.scripted_runtime.testcase import ScriptedRuntimeTestCase
from sglang.test.scripted_runtime_chunked_helpers import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_MAX_STEPS,
    VERY_LONG_PROMPT_LEN,
    base_engine_kwargs,
    run_until,
    run_until_all_finished,
    run_until_finished,
)


class TestPriorityBasic(ScriptedRuntimeTestCase):
    ENGINE_KWARGS = base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE)

    def test_retract_mid_chunk_releases_kv(self):
        """KV pressure mid-chunk causes the chunked-resume req to be retracted."""
        self.runtime.run(self._script_retract_mid_chunk_releases_kv)

    # KV pressure mid-chunk causes the chunked-resume req to be
    # retracted. All resources must release immediately (96d4749094 +
    # f38e69f87d).
    @staticmethod
    def _script_retract_mid_chunk_releases_kv(t: ScriptedRuntime):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r, lambda h: h.is_chunking and h.chunks_done >= 1)

        pages_before = r.kv_pages
        assert pages_before > 0

        # Force KV pressure by spamming short reqs until the pool is
        # nearly exhausted. ``exhaust_kv`` is the wishlist §4 P2 (9) helper.
        t.exhaust_kv(leave_pages=1)
        yield

        assert r.status in (
            "waiting",
            "finished",
            "unknown",
        ), f"r should be retracted (back in waiting) or finished; got {r.status}"
        if r.status == "waiting":
            # Retract path: resources released; the req is parked for re-admission.
            assert r.kv_pages == 0, f"retract must release KV; got {r.kv_pages}"

    def test_retract_and_resume(self):
        """Retract chunked mid-stream, release pressure, then complete normally."""
        self.runtime.run(self._script_retract_and_resume)

    # retract-and-resume. Retract chunked mid-stream, then release
    # pressure so it can be re-admitted; must complete normally.
    @staticmethod
    def _script_retract_and_resume(t: ScriptedRuntime):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r, lambda h: h.is_chunking and h.chunks_done >= 1)

        # Use the active fault-injection helper to force a retract — the
        # passive ``exhaust_kv`` is also acceptable, but this is
        # deterministic.
        t.force_retract(r)
        yield
        assert r.status == "waiting"
        assert r.kv_pages == 0

        # Drain pressure (no-op if we used force_retract) and run to completion.
        yield from run_until_finished(r)
        assert r.finished

    def test_force_retract_at_chunk_0(self):
        """Force retract before first chunk completes."""
        self.runtime.run(self._script_force_retract_at_chunk_0)

    @staticmethod
    def _script_force_retract_at_chunk_0(t: ScriptedRuntime):
        # Force retract before first chunk completes.
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield  # admission
        yield  # first chunk
        t.force_retract(r)
        yield
        assert r.kv_pages == 0

    def test_force_retract_at_chunk_mid(self):
        """Force retract in the middle of chunked extend."""
        self.runtime.run(self._script_force_retract_at_chunk_mid)

    @staticmethod
    def _script_force_retract_at_chunk_mid(t: ScriptedRuntime):
        # Force retract in the middle of chunked extend.
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r, lambda h: h.chunks_done >= 2 and h.is_chunking)
        t.force_retract(r)
        yield
        assert r.kv_pages == 0

    def test_force_retract_at_last_chunk(self):
        """Force retract during the last chunk's admit."""
        self.runtime.run(self._script_force_retract_at_last_chunk)

    @staticmethod
    def _script_force_retract_at_last_chunk(t: ScriptedRuntime):
        # Force retract during the last chunk's admit.
        r = t.start_req(prompt_len=2 * DEFAULT_CHUNK_SIZE, max_new_tokens=4)
        yield from run_until(r, lambda h: h.chunks_done >= 1 and h.is_chunking)
        t.force_retract(r)
        yield
        assert r.kv_pages == 0
        assert r.pending_middle_outputs == 0

    def test_force_retract_then_readmit(self):
        """Force retract then req re-admits and finishes cleanly."""
        self.runtime.run(self._script_force_retract_then_readmit)

    @staticmethod
    def _script_force_retract_then_readmit(t: ScriptedRuntime):
        # Force retract; the chunked-resume parked in waiting_queue
        # must not be aborted by the watchdog (commit 359e5ed7bd) —
        # observable as the req successfully re-admitting and finishing.
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r, lambda h: h.is_chunking)
        t.force_retract(r)
        yield
        assert r.kv_pages == 0, "retract must release KV before re-admission"
        # Scheduler should re-admit on its own and the req must finish.
        yield from run_until_finished(r, max_steps=800)
        assert r.finished
        assert r.kv_pages == 0
        assert r.lock_refs == 0

    def test_retract_one_admit_one(self):
        """Retract r1 while admitting r2: r2 finishes first, r1 re-admits and finishes, neither leaks resources."""
        self.runtime.run(self._script_retract_one_admit_one)

    @staticmethod
    def _script_retract_one_admit_one(t: ScriptedRuntime):
        # Force retract r1 + simultaneously admit r2. r2 must complete
        # quickly; r1 must re-admit and complete cleanly. KV / lock_refs
        # must release for both.
        r1 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r1, lambda h: h.is_chunking)
        r2 = t.start_req(prompt_len=8, max_new_tokens=2)
        t.force_retract(r1)
        yield from run_until_finished(r2)
        assert r2.finished
        assert r2.kv_pages == 0
        yield from run_until_finished(r1)
        assert r1.finished
        assert r1.kv_pages == 0
        assert r1.lock_refs == 0

    def test_retract_during_decode(self):
        """Retract during pure decode (no chunked) cleanly releases resources and the req still finishes."""
        self.runtime.run(self._script_retract_during_decode)

    @staticmethod
    def _script_retract_during_decode(t: ScriptedRuntime):
        # Retract during pure decode (no chunked path involved at all).
        # The retract must release KV immediately; the scheduler must
        # re-admit and the req must finish with no leaked resources.
        r = t.start_req(prompt_len=8, max_new_tokens=32)
        yield from run_until(r, lambda h: h.status == "running")
        assert r.kv_pages > 0, "decode-state req must own KV before retract"
        t.force_retract(r)
        yield
        assert r.kv_pages == 0, f"retract must release KV; got {r.kv_pages}"
        yield from run_until_finished(r)
        assert r.finished
        assert r.kv_pages == 0
        assert r.lock_refs == 0

    def test_retract_then_abort_idempotent(self):
        """Retract + abort same step; final state stable."""
        self.runtime.run(self._script_retract_then_abort_idempotent)

    @staticmethod
    def _script_retract_then_abort_idempotent(t: ScriptedRuntime):
        # Retract + abort same step; final state stable.
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r, lambda h: h.is_chunking)
        t.force_retract(r)
        t.abort(r)
        yield
        assert r.kv_pages == 0

    def test_disagg_retract_resets_send_state_extra(self):
        """Disagg path: retract must reset send-side state (414efd4a27)."""
        self.runtime.run(self._script_disagg_retract_resets_send_state_extra)

    @staticmethod
    def _script_disagg_retract_resets_send_state_extra(t: ScriptedRuntime):
        # disagg path: retract must reset send-side state (414efd4a27).
        # Even in non-disagg engine config, the send-side field must be
        # absent/idle after retract — retract still touches the same
        # scheduler code path.
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r, lambda h: h.is_chunking)
        t.force_retract(r)
        yield
        # NEW API NEEDED: r.disagg_send_state — current state of the
        # send-side machine (None when not disagg or after reset).
        assert r.disagg_send_state in (None, "idle")
        yield from run_until_finished(r)
        assert r.finished

    def test_retract_chunked_resume_in_waiting(self):
        """Chunked-resume sitting in waiting → force retract releases row + KV cleanly."""
        self.runtime.run(self._script_retract_chunked_resume_in_waiting)

    @staticmethod
    def _script_retract_chunked_resume_in_waiting(t: ScriptedRuntime):
        # Chunked-resume already parked in waiting — this is the exact
        # state the commit 359e5ed7bd fix protects: watchdog must not
        # abort a chunked-resume here. Observable as the req being
        # retracted cleanly (kv_pages == 0, row_idx released) rather
        # than entering an aborted state.
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r, lambda h: h.is_chunking)
        yield from run_until(r, lambda h: h.status == "waiting")
        t.force_retract(r)
        yield
        assert r.kv_pages == 0
        assert r.row_idx is None
        # Status must be a clean retract-park ("waiting"), not aborted.
        assert r.status in ("waiting", "finished")

    def test_two_retracts_same_yield(self):
        """Two reqs retracted in the same yield release KV simultaneously and both still finish."""
        self.runtime.run(self._script_two_retracts_same_yield)

    @staticmethod
    def _script_two_retracts_same_yield(t: ScriptedRuntime):
        # Two reqs force_retracted in the same yield step — both must
        # release KV in lockstep, both must re-admit and finish (the
        # observable witness that the watchdog did not spuriously abort
        # the chunked-resume in waiting).
        r1 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        r2 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r1, lambda h: h.is_chunking)
        t.force_retract(r1)
        t.force_retract(r2)
        yield
        assert r1.kv_pages == 0
        assert r2.kv_pages == 0
        yield from run_until_all_finished([r1, r2])
        assert r1.finished and r2.finished
        assert r1.lock_refs == 0
        assert r2.lock_refs == 0

    def test_retract_then_re_chunk(self):
        """Retract mid-chunk then resume: re-chunk continues, KV/lock release at finish."""
        self.runtime.run(self._script_retract_then_re_chunk)

    @staticmethod
    def _script_retract_then_re_chunk(t: ScriptedRuntime):
        # Retract a mid-chunk req; subsequent re-chunk must complete
        # without prefix_indices residue. The req successfully finishing
        # is the observable witness that the watchdog did not abort the
        # chunked-resume while it sat in the waiting_queue.
        r = t.start_req(prompt_len=2 * DEFAULT_CHUNK_SIZE, max_new_tokens=2)
        yield from run_until(r, lambda h: h.chunks_done >= 1)
        t.force_retract(r)
        # Explicit yield so the retract side effect happens-before the
        # subsequent run_until_finished observes any state.
        yield
        assert r.kv_pages == 0, "retract must release KV"
        yield from run_until_finished(r, max_steps=800)
        assert r.finished
        assert r.lock_refs == 0
        assert r.kv_pages == 0

    def test_watchdog_skips_chunked_resume_invariant(self):
        """Invariant W2: _abort_on_waiting_timeout must skip chunked-resume reqs even when their wait_queue_entry_time has aged past the timeout.

        Direct access to ``_scheduler`` internals is intentional; this is an
        invariant-tier test (see direct-internals-access plan).
        """
        self.runtime.run(self._script_watchdog_skips_chunked_resume_invariant)

    # W2 — drive a chunked req mid-prefill, simulate the v2 "chunked-resume
    # parked in waiting_queue" condition by manually inserting it into
    # ``s.waiting_queue`` with a backdated entry_time, run the watchdog
    # sweep deterministically by calling ``s._abort_on_waiting_timeout()``,
    # and assert the chunked-resume survives. The watchdog being
    # wall-clock driven makes the natural reproduction flaky in scripted
    # runtime; invoking the sweep directly removes timing variance.
    # See commit 359e5ed7bd ("Skip chunked-resume reqs in
    # _abort_on_waiting_timeout") for the original bug.
    @staticmethod
    def _script_watchdog_skips_chunked_resume_invariant(t: ScriptedRuntime):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        # Drive until the scheduler is mid-chunk on this req — the
        # chunked_req slot is occupied and inflight_middle_chunks > 0,
        # which is the protected state.
        yield from run_until(r, lambda h: h.is_chunking)
        s = t._scheduler
        req = t._find_req_by_rid(r.rid)
        assert req is not None
        assert s.chunked_req is req, (
            f"setup expected s.chunked_req to point at r; got "
            f"s.chunked_req={s.chunked_req!r}, r={req!r}"
        )
        assert req.inflight_middle_chunks > 0, (
            f"setup expected inflight_middle_chunks > 0, got "
            f"{req.inflight_middle_chunks}"
        )

        # Simulate the v2 layout: a chunked-resume req can sit in
        # ``waiting_queue`` across iters with a stale entry_time. Force
        # both conditions by backdating ``wait_queue_entry_time`` and
        # inserting the req into ``waiting_queue`` so the watchdog has a
        # candidate to abort. Note: the req remains the live chunked_req
        # — the duplicate placement only persists for the synchronous
        # sweep call below and is undone before we yield control back.
        was_in_queue = req in s.waiting_queue
        if not was_in_queue:
            s.waiting_queue.append(req)
        original_entry_time = req.time_stats.wait_queue_entry_time
        # ``time.perf_counter()`` starts at process load, so by the time
        # the engine is mid-chunk it is comfortably > 1.0s. An
        # entry_time of 1.0 is unambiguously older than the deadline for
        # any positive timeout.
        req.time_stats.wait_queue_entry_time = 1.0
        try:
            with envs.SGLANG_REQ_WAITING_TIMEOUT.override(0.5):
                # Confirm the deadline math actually selects r as a
                # candidate — otherwise the assertion below would be
                # vacuously satisfied.
                deadline = time.perf_counter() - 0.5
                assert (
                    0 < req.time_stats.wait_queue_entry_time < deadline
                ), "setup did not backdate entry_time past the watchdog deadline"
                s._abort_on_waiting_timeout()

            # Invariant: the watchdog must not have removed the
            # chunked-resume req from waiting_queue. If it did, the
            # gate is missing or broken and the resume would leak KV
            # + req_to_token row on next iter.
            assert req in s.waiting_queue, (
                "watchdog incorrectly aborted a chunked-resume req: "
                "r was removed from waiting_queue despite inflight_middle_chunks > 0"
            )
            # Also: chunked_req slot must still point at r — abort would
            # have nulled / replaced it.
            assert s.chunked_req is req, (
                f"chunked_req slot must still hold r after watchdog "
                f"skip, got s.chunked_req={s.chunked_req!r}"
            )
            assert req.finished_reason is None, (
                f"chunked-resume req must not be marked finished by "
                f"watchdog abort, got finished_reason={req.finished_reason!r}"
            )
        finally:
            # Restore scheduler state so the rest of the script — and
            # any later tests sharing this engine — see clean state.
            if not was_in_queue and req in s.waiting_queue:
                s.waiting_queue.remove(req)
            req.time_stats.wait_queue_entry_time = original_entry_time

        # Drive to completion so the engine returns to idle.
        yield from run_until_finished(r, max_steps=DEFAULT_MAX_STEPS * 2)
        assert r.finished


class TestPriorityPriority(ScriptedRuntimeTestCase):
    ENGINE_KWARGS = base_engine_kwargs(
        chunked_prefill_size=DEFAULT_CHUNK_SIZE,
        enable_priority_scheduling=True,
    )

    def test_naive_priority_chunked(self):
        """Priority scheduling × chunked: naive ScriptedRuntime smoke plus retract/preempt edge cases."""
        self.runtime.run(self._script_naive_priority_chunked)

    @staticmethod
    def _script_naive_priority_chunked(t: ScriptedRuntime):
        low = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=4, priority="low"
        )
        yield  # let scheduler pull `low` and begin its chunk loop

        high = t.start_req(prompt_len=8, max_new_tokens=2, priority="high")

        yield from run_until_all_finished([low, high])
        assert low.finished and high.finished

    def test_priority_preempt_chunked(self):
        """Priority preemption — a high-priority req shows up while a low-priority chunked-resume holds resources."""
        self.runtime.run(self._script_priority_preempt_chunked)

    # priority preemption — a high-priority req shows up while a
    # low-priority chunked-resume holds resources. The victim chunked req
    # must go through retract (not abort) and the high-priority req takes
    # its slot.
    @staticmethod
    def _script_priority_preempt_chunked(t: ScriptedRuntime):
        low = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2, priority="low"
        )
        yield from run_until(low, lambda h: h.is_chunking and h.chunks_done >= 1)
        assert low.kv_pages > 0

        high = t.start_req(prompt_len=8, max_new_tokens=2, priority="high")
        # In a pressured cache, high should force preempt of low.
        t.exhaust_kv(leave_pages=1)
        yield

        # After preempt: low is retracted (status "waiting") with no KV;
        # high gets admitted.
        if low.status == "waiting":
            assert low.kv_pages == 0
        yield from run_until_finished(high)
        assert high.finished

    def test_priority_preempt_chunked_victim(self):
        """Submit chunked low-priority, then high-priority new req."""
        self.runtime.run(self._script_priority_preempt_chunked_victim)

    @staticmethod
    def _script_priority_preempt_chunked_victim(t: ScriptedRuntime):
        # Submit chunked low-priority, then high-priority new req.
        # Victim chunked must be moved from active to waiting.
        r_low = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2, priority="low"
        )
        yield from run_until(r_low, lambda h: h.is_chunking)
        # NEW API NEEDED: t.force_preempt(victim, by) — explicit priority preempt
        # trigger; auto-triggered when high-prio req can't admit otherwise.
        r_high = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2, priority="high"
        )
        t.force_preempt(victim_rid=r_low.rid, by_rid=r_high.rid)
        yield
        assert r_low.status == "waiting"

        yield from run_until_all_finished([r_low, r_high])

    def test_preempt_five_victims(self):
        """5 chunked reqs preempted simultaneously."""
        self.runtime.run(self._script_preempt_five_victims)

    @staticmethod
    def _script_preempt_five_victims(t: ScriptedRuntime):
        # 5 chunked reqs preempted simultaneously.
        victims = [
            t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
            for _ in range(5)
        ]
        yield from run_until(victims[0], lambda h: h.is_chunking)

        r_high = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2, priority="high"
        )
        for v in victims:
            t.force_preempt(victim_rid=v.rid, by_rid=r_high.rid)
        yield
        for v in victims:
            assert v.kv_pages == 0

        yield from run_until_all_finished(victims + [r_high])

    def test_priority_preempt_release_invariant(self):
        """Each preemption releases the victim's kv_pages strictly to 0."""
        self.runtime.run(self._script_priority_preempt_release_invariant)

    @staticmethod
    def _script_priority_preempt_release_invariant(t: ScriptedRuntime):
        # Each preemption releases the victim's kv_pages strictly to 0.
        r_low = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2, priority="low"
        )
        yield from run_until(r_low, lambda h: h.is_chunking)
        pages_before = r_low.kv_pages
        assert pages_before > 0

        r_high = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2, priority="high"
        )
        t.force_preempt(victim_rid=r_low.rid, by_rid=r_high.rid)
        yield
        assert r_low.kv_pages == 0

    def test_priority_preempt_with_chunked_admission_same_yield(self):
        """Same yield: high-priority R1 + new long chunked R2 land while low-priority R3 runs; R3 preempted, R2 chunks_done starts; both finish."""
        self.runtime.run(
            self._script_priority_preempt_with_chunked_admission_same_yield
        )

    # same-yield priority preempt + chunked admission combo.
    # Low-priority R3 is mid-flight; on the same yield we submit a
    # high-priority short R1 and a long chunked R2. R3 must be preempted
    # (kv_pages goes to 0), R2 must begin chunking (chunks_done starts
    # advancing), and ultimately every req completes.
    @staticmethod
    def _script_priority_preempt_with_chunked_admission_same_yield(
        t: ScriptedRuntime,
    ):
        # Low-priority running req to be preempted.
        r3 = t.start_req(
            prompt_len=16,
            max_new_tokens=32,
            priority="low",
        )
        yield from run_until(r3, lambda h: h.status == "running")
        assert r3.kv_pages > 0

        # Same yield: submit high-priority R1 + long chunked R2, plus
        # an explicit preempt to force R3 out of the running batch.
        r1 = t.start_req(prompt_len=8, max_new_tokens=2, priority="high")
        r2 = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN,
            max_new_tokens=2,
            priority="high",
        )
        t.force_preempt(victim_rid=r3.rid, by_rid=r1.rid)
        yield

        # R3 retracted: kv_pages released.
        assert (
            r3.kv_pages == 0
        ), f"preempted low-priority r3 must release KV; got {r3.kv_pages}"

        # R2 must enter chunked admission.
        yield from run_until(r2, lambda h: h.is_chunking)
        assert r2.chunks_done >= 1, (
            f"long chunked r2 should start advancing chunks_done; got "
            f"{r2.chunks_done}"
        )

        # All reqs eventually finish.
        yield from run_until_all_finished([r1, r2, r3], max_steps=800)
        assert r1.finished and r2.finished and r3.finished


class TestPriorityDisagg(ScriptedRuntimeTestCase):
    ENGINE_KWARGS = base_engine_kwargs(
        chunked_prefill_size=DEFAULT_CHUNK_SIZE,
        disaggregation_mode="prefill",
    )

    def test_disagg_retract_resets_send_state(self):
        """Disagg mode retract — disaggregation send-side state must reset on chunked-resume retract (414efd4a27)."""
        self.runtime.run(self._script_disagg_retract_resets_send_state)

    # disagg mode retract — disaggregation send-side state must
    # reset on chunked-resume retract (414efd4a27).
    @staticmethod
    def _script_disagg_retract_resets_send_state(t: ScriptedRuntime):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r, lambda h: h.is_chunking and h.chunks_done >= 1)

        t.force_retract(r)
        yield

        assert r.status == "waiting"
        assert r.kv_pages == 0
        # Disagg-specific: the per-req send-side state must reset.
        assert r.disagg_send_state in (
            None,
            "idle",
        ), f"disagg send state must reset on retract, got {r.disagg_send_state}"

        yield from run_until_finished(r)
        assert r.finished


if __name__ == "__main__":
    unittest.main()
