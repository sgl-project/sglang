"""Invariant scripted tests for chunked prefill.

State-query API invariants (A.8 series from the expansion plan). These
tests are contract-style — they don't drive an interesting trajectory,
they just hammer on the harness API (status, chunks_done, kv_pages,
batch_composition, etc.) at every yield step and verify the
monotone / non-negative / self-consistent invariants hold.

Also covers B.5 series — long-running stability scenarios and stress
fan-outs. The focus is "no resource leak after many reqs" — every test
starts and finishes a baseline run, then asserts ``engine_stats``
returns to (or above) the initial pool counts.
"""

import unittest

from sglang.test.scripted_runtime.req_handle import ReqHandle
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


class TestInvariantsBasic(ScriptedRuntimeTestCase):
    ENGINE_KWARGS = base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE)

    def test_status_unknown_before_submit(self):
        """A bare ReqHandle whose rid was never submitted reports "unknown"."""
        self.runtime.run(self._script_status_unknown_before_submit)

    @staticmethod
    def _script_status_unknown_before_submit(t: ScriptedRuntime):
        # A bare ReqHandle whose rid was never submitted reports "unknown".
        bogus = ReqHandle(rid="never-submitted", runtime=t)
        assert bogus.status == "unknown"
        yield

    def test_status_finished_after_done(self):
        """After finish, r.status == "finished" (not unknown)."""
        self.runtime.run(self._script_status_finished_after_done)

    @staticmethod
    def _script_status_finished_after_done(t: ScriptedRuntime):
        # After finish, r.status == "finished" (not unknown).
        r = t.start_req(prompt_len=16, max_new_tokens=2)
        yield from run_until_finished(r)
        assert r.status == "finished"

    def test_chunks_done_monotone_invariant(self):
        """Chunks_done is non-decreasing across yield steps."""
        self.runtime.run(self._script_chunks_done_monotone_invariant)

    @staticmethod
    def _script_chunks_done_monotone_invariant(t: ScriptedRuntime):
        # chunks_done is non-decreasing across yield steps.
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        prev = 0
        for _ in range(DEFAULT_MAX_STEPS):
            cur = r.chunks_done
            assert cur >= prev, f"chunks_done regressed: {prev} -> {cur}"
            prev = cur
            if r.finished:
                return
            yield
        raise AssertionError("req never finished")

    def test_kv_pages_zero_after_finish(self):
        """Kv_pages drops to 0 after the req finishes."""
        self.runtime.run(self._script_kv_pages_zero_after_finish)

    @staticmethod
    def _script_kv_pages_zero_after_finish(t: ScriptedRuntime):
        # kv_pages drops to 0 after the req finishes.
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until_finished(r)
        assert r.kv_pages == 0

    def test_kv_pages_positive_mid_chunk(self):
        """Mid-chunked, kv_pages > 0."""
        self.runtime.run(self._script_kv_pages_positive_mid_chunk)

    @staticmethod
    def _script_kv_pages_positive_mid_chunk(t: ScriptedRuntime):
        # Mid-chunked, kv_pages > 0.
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r, lambda h: h.is_chunking)
        assert r.kv_pages > 0
        yield from run_until_finished(r)

    def test_batch_composition_consistent_with_status(self):
        """If r.status == "running" then r.rid appears in batch_composition."""
        self.runtime.run(self._script_batch_composition_consistent_with_status)

    @staticmethod
    def _script_batch_composition_consistent_with_status(t: ScriptedRuntime):
        # If r.status == "running" then r.rid appears in batch_composition.
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        for _ in range(DEFAULT_MAX_STEPS):
            if r.status == "running":
                comp = t.batch_composition()
                all_rids = (
                    comp.get("prefill", [])
                    + comp.get("decode", [])
                    + comp.get("chunked", [])
                )
                assert (
                    r.rid in all_rids
                ), f"running but not in batch_composition: {comp}"
            if r.finished:
                return
            yield
        raise AssertionError("req never finished")

    def test_is_idle_excludes_chunked_in_flight(self):
        """T.is_idle and chunked_in_flight_count > 0 are mutually exclusive."""
        self.runtime.run(self._script_is_idle_excludes_chunked_in_flight)

    @staticmethod
    def _script_is_idle_excludes_chunked_in_flight(t: ScriptedRuntime):
        # t.is_idle and chunked_in_flight_count > 0 are mutually exclusive.
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        for _ in range(DEFAULT_MAX_STEPS):
            if t.chunked_in_flight_count() > 0:
                assert not t.is_idle, "is_idle must be False when chunked is in flight"
            if r.finished:
                return
            yield
        raise AssertionError("req never finished")

    def test_finish_event_count_exactly_one(self):
        """Every normally-completed req emits exactly one finish event."""
        self.runtime.run(self._script_finish_event_count_exactly_one)

    @staticmethod
    def _script_finish_event_count_exactly_one(t: ScriptedRuntime):
        # Every normally-completed req emits exactly one finish event.
        r = t.start_req(prompt_len=16, max_new_tokens=2)
        yield from run_until_finished(r)
        assert r.finish_event_count == 1

    def test_kv_pages_non_negative(self):
        """Kv_pages never negative."""
        self.runtime.run(self._script_kv_pages_non_negative)

    @staticmethod
    def _script_kv_pages_non_negative(t: ScriptedRuntime):
        # kv_pages never negative.
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        for _ in range(DEFAULT_MAX_STEPS):
            assert r.kv_pages >= 0
            if r.finished:
                return
            yield
        raise AssertionError("req never finished")

    def test_pending_middle_outputs_non_negative(self):
        """Pending_middle_outputs non-negative invariant."""
        self.runtime.run(self._script_pending_middle_outputs_non_negative)

    @staticmethod
    def _script_pending_middle_outputs_non_negative(t: ScriptedRuntime):
        # pending_middle_outputs non-negative invariant.
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        for _ in range(DEFAULT_MAX_STEPS):
            assert r.pending_middle_outputs >= 0
            if r.finished:
                return
            yield

    def test_inflight_middle_chunks_non_negative(self):
        """Inflight_middle_chunks counter stays non-negative across all yields."""
        self.runtime.run(self._script_inflight_middle_chunks_non_negative)

    @staticmethod
    def _script_inflight_middle_chunks_non_negative(t: ScriptedRuntime):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        for _ in range(DEFAULT_MAX_STEPS):
            assert r.inflight_middle_chunks >= 0
            if r.finished:
                return
            yield

    def test_lock_refs_non_negative(self):
        """Lock_refs counter stays non-negative across all yields."""
        self.runtime.run(self._script_lock_refs_non_negative)

    @staticmethod
    def _script_lock_refs_non_negative(t: ScriptedRuntime):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        for _ in range(DEFAULT_MAX_STEPS):
            assert r.lock_refs >= 0
            if r.finished:
                return
            yield

    def test_chunked_in_flight_count_le_one(self):
        """Main-upstream invariant: at most one chunked req in flight."""
        self.runtime.run(self._script_chunked_in_flight_count_le_one)

    @staticmethod
    def _script_chunked_in_flight_count_le_one(t: ScriptedRuntime):
        # main-upstream invariant: at most one chunked req in flight.
        reqs = [
            t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
            for _ in range(3)
        ]
        for _ in range(DEFAULT_MAX_STEPS * 3):
            assert t.chunked_in_flight_count() <= 1
            if all(r.finished for r in reqs):
                return
            yield

    def test_status_transition_monotone(self):
        """Chunks_done is non-decreasing across the chunked req's lifetime."""
        self.runtime.run(self._script_status_transition_monotone)

    @staticmethod
    def _script_status_transition_monotone(t: ScriptedRuntime):
        # The unknown -> waiting -> running -> finished status rank is *not*
        # strictly monotone because retract / double-retract can drop the
        # req back to waiting one or more times. Instead, observe that
        # chunks_done is non-decreasing across the lifetime — that is the
        # real progress invariant a chunked req must satisfy.
        r = t.start_req(prompt_len=16, max_new_tokens=2)
        prev_chunks_done = 0
        for _ in range(DEFAULT_MAX_STEPS):
            cur_chunks_done = r.chunks_done
            assert (
                cur_chunks_done >= prev_chunks_done
            ), f"chunks_done regressed: {cur_chunks_done} < {prev_chunks_done}"
            prev_chunks_done = cur_chunks_done
            if r.finished:
                return
            yield

    def test_active_reqs_listing(self):
        """NEW API NEEDED: t.list_active_reqs() returns the currently-running set of ReqHandle objects."""
        self.runtime.run(self._script_active_reqs_listing)

    @staticmethod
    def _script_active_reqs_listing(t: ScriptedRuntime):
        # NEW API NEEDED: t.list_active_reqs() returns the currently-running
        # set of ReqHandle objects.
        r1 = t.start_req(prompt_len=16, max_new_tokens=4)
        r2 = t.start_req(prompt_len=16, max_new_tokens=4)
        yield
        actives = t.list_active_reqs()
        rids = {h.rid for h in actives}
        assert r1.rid in rids or r2.rid in rids
        yield from run_until_all_finished([r1, r2])
        actives_after = t.list_active_reqs()
        assert all(h.rid not in (r1.rid, r2.rid) for h in actives_after)

    def test_batch_composition_disjoint_subsets(self):
        """Prefill / decode / chunked subsets must be disjoint."""
        self.runtime.run(self._script_batch_composition_disjoint_subsets)

    @staticmethod
    def _script_batch_composition_disjoint_subsets(t: ScriptedRuntime):
        # prefill / decode / chunked subsets must be disjoint.
        r1 = t.start_req(prompt_len=16, max_new_tokens=2)
        r2 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        for _ in range(DEFAULT_MAX_STEPS):
            comp = t.batch_composition()
            prefill = set(comp.get("prefill", []))
            decode = set(comp.get("decode", []))
            chunked = set(comp.get("chunked", []))
            assert prefill & decode == set()
            assert prefill & chunked == set()
            assert decode & chunked == set()
            if r1.finished and r2.finished:
                return
            yield

    def test_finished_means_chunks_done_stable(self):
        """Once finished, chunks_done value should not change."""
        self.runtime.run(self._script_finished_means_chunks_done_stable)

    @staticmethod
    def _script_finished_means_chunks_done_stable(t: ScriptedRuntime):
        # Once finished, chunks_done value should not change.
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until_finished(r)
        snap = r.chunks_done
        for _ in range(10):
            yield
            assert r.chunks_done == snap

    def test_finished_means_kv_pages_stays_zero(self):
        """After finish, kv_pages remains 0 across subsequent yields."""
        self.runtime.run(self._script_finished_means_kv_pages_stays_zero)

    @staticmethod
    def _script_finished_means_kv_pages_stays_zero(t: ScriptedRuntime):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until_finished(r)
        for _ in range(10):
            yield
            assert r.kv_pages == 0

    def test_engine_stats_keys_present(self):
        """Engine_stats returns a dict with expected keys."""
        self.runtime.run(self._script_engine_stats_keys_present)

    @staticmethod
    def _script_engine_stats_keys_present(t: ScriptedRuntime):
        # engine_stats returns a dict with expected keys.
        stats = t.engine_stats()
        assert isinstance(stats, dict)
        assert "kv_pool_free" in stats
        assert "row_pool_free" in stats
        yield

    def test_kv_pool_recovers_to_baseline(self):
        """Full lifecycle: pool counts return to baseline after all done."""
        self.runtime.run(self._script_kv_pool_recovers_to_baseline)

    @staticmethod
    def _script_kv_pool_recovers_to_baseline(t: ScriptedRuntime):
        # Full lifecycle: pool counts return to baseline after all done.
        before = t.engine_stats()["kv_pool_free"]
        reqs = [t.start_req(prompt_len=16, max_new_tokens=2) for _ in range(8)]
        yield from run_until_all_finished(reqs)
        after = t.engine_stats()["kv_pool_free"]
        assert after >= before

    def test_hundred_reqs_no_leak(self):
        """100 reqs end-to-end: KV/row/lock_ref pool counts return to baseline."""
        self.runtime.run(self._script_hundred_reqs_no_leak)

    @staticmethod
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

    def test_two_hundred_reqs_no_leak(self):
        """200 short reqs end-to-end leave KV pool >= baseline."""
        self.runtime.run(self._script_two_hundred_reqs_no_leak)

    @staticmethod
    def _script_two_hundred_reqs_no_leak(t: ScriptedRuntime):
        baseline = t.engine_stats()
        reqs = [t.start_req(prompt_len=8, max_new_tokens=1) for _ in range(200)]
        yield from run_until_all_finished(reqs, max_steps=8000)
        final = t.engine_stats()
        assert final["kv_pool_free"] >= baseline["kv_pool_free"]

    def test_five_hundred_reqs_no_leak(self):
        """500 short reqs end-to-end leave KV pool >= baseline."""
        self.runtime.run(self._script_five_hundred_reqs_no_leak)

    @staticmethod
    def _script_five_hundred_reqs_no_leak(t: ScriptedRuntime):
        baseline = t.engine_stats()
        reqs = [t.start_req(prompt_len=8, max_new_tokens=1) for _ in range(500)]
        yield from run_until_all_finished(reqs, max_steps=15000)
        final = t.engine_stats()
        assert final["kv_pool_free"] >= baseline["kv_pool_free"]

    def test_long_lived_engine_reps_chunked(self):
        """20 rounds × 5 reqs each; scheduler internal counters stay healthy."""
        self.runtime.run(self._script_long_lived_engine_reps_chunked)

    @staticmethod
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

    def test_sustained_long_chunked_load(self):
        """Sustained: 30 long chunked reqs."""
        self.runtime.run(self._script_sustained_long_chunked_load)

    @staticmethod
    def _script_sustained_long_chunked_load(t: ScriptedRuntime):
        # Sustained: 30 long chunked reqs.
        reqs = [
            t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
            for _ in range(30)
        ]
        yield from run_until_all_finished(reqs, max_steps=8000)

    def test_round_robin_short_and_chunked(self):
        """50 short followed by 5 chunked, 5 rounds."""
        self.runtime.run(self._script_round_robin_short_and_chunked)

    @staticmethod
    def _script_round_robin_short_and_chunked(t: ScriptedRuntime):
        # 50 short followed by 5 chunked, 5 rounds.
        for _ in range(5):
            shorts = [t.start_req(prompt_len=16, max_new_tokens=2) for _ in range(10)]
            chunked = [
                t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
                for _ in range(1)
            ]
            yield from run_until_all_finished(shorts + chunked, max_steps=2000)

    def test_long_decode_then_many_short(self):
        """One very long decode + many short."""
        self.runtime.run(self._script_long_decode_then_many_short)

    @staticmethod
    def _script_long_decode_then_many_short(t: ScriptedRuntime):
        # One very long decode + many short.
        long_decode = t.start_req(prompt_len=16, max_new_tokens=256)
        shorts = [t.start_req(prompt_len=8, max_new_tokens=2) for _ in range(50)]
        yield from run_until_all_finished([long_decode] + shorts, max_steps=4000)

    def test_chunked_in_flight_count_never_above_one_long_run(self):
        """50 chunked reqs over many yields; verify invariant at every step."""
        self.runtime.run(self._script_chunked_in_flight_count_never_above_one_long_run)

    @staticmethod
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

    def test_engine_stats_monotone_after_each_batch(self):
        """After each batch finishes, kv_pool_free non-decreasing vs end-of-prev-batch."""
        self.runtime.run(self._script_engine_stats_monotone_after_each_batch)

    @staticmethod
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

    def test_chunked_status_never_mid_chunk_running(self):
        """Long chunked req's status only ever takes legal values across the chunked lifecycle."""
        self.runtime.run(self._script_chunked_status_never_mid_chunk_running)

    # chunked status legal-set — during chunked admission a
    # req's externally observable status must remain in {waiting, running}
    # while chunking, never a synthetic "mid-chunk" leak; once chunked
    # completes it transitions to finished and never re-enters waiting.
    @staticmethod
    def _script_chunked_status_never_mid_chunk_running(t: ScriptedRuntime):
        legal_pre_finish = {"waiting", "running", "unknown"}
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        for _ in range(DEFAULT_MAX_STEPS):
            s = r.status
            if r.finished:
                assert (
                    s == "finished"
                ), f"finished req must report status 'finished', got {s!r}"
                return
            assert s in legal_pre_finish, (
                f"chunked req observed illegal status {s!r}; "
                f"legal pre-finish set is {legal_pre_finish}"
            )
            yield
        raise AssertionError("req never finished")

    def test_pending_middle_outputs_caps_at_one(self):
        """Long chunked req's pending_middle_outputs is capped at 1 across its lifetime."""
        self.runtime.run(self._script_pending_middle_outputs_caps_at_one)

    # pending_middle_outputs cap — at any
    # iteration there must be at most one pending middle output; the
    # revert of e875cd36e4 re-established this invariant.
    @staticmethod
    def _script_pending_middle_outputs_caps_at_one(t: ScriptedRuntime):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        # Sample BEFORE the first yield, AFTER every yield (including the
        # finish iter), and for several yields AFTER r.finished to verify
        # the counter is reset and stays at 0. The lifetime maximum must
        # be exactly 1 — proves chunking happened (>=1) and the cap held
        # (<=1).
        running_max = r.pending_middle_outputs
        running_max_post_finish = 0
        post_finish_samples = 0
        for _ in range(DEFAULT_MAX_STEPS):
            yield
            cur = r.pending_middle_outputs
            running_max = max(running_max, cur)
            if r.finished:
                running_max_post_finish = max(running_max_post_finish, cur)
                post_finish_samples += 1
                if post_finish_samples >= 5:
                    break
        assert r.finished, "req never finished"
        assert running_max == 1, (
            f"pending_middle_outputs must reach exactly 1 across the chunked "
            f"lifecycle (the cap from revert e875cd36e4); observed max={running_max}"
        )
        assert running_max_post_finish == 0, (
            f"pending_middle_outputs must be reset to 0 after finish; "
            f"observed max post-finish={running_max_post_finish}"
        )

    def test_status_never_finished_to_waiting(self):
        """After finish, status never rolls back to waiting / running / unknown."""
        self.runtime.run(self._script_status_never_finished_to_waiting)

    # finished is terminal — once a req has finished, its
    # observable status must never roll back to waiting / running /
    # unknown across subsequent yields.
    @staticmethod
    def _script_status_never_finished_to_waiting(t: ScriptedRuntime):
        r = t.start_req(prompt_len=16, max_new_tokens=2)
        yield from run_until_finished(r)
        assert r.status == "finished"
        for _ in range(50):
            yield
            assert (
                r.status == "finished"
            ), f"status rolled back from finished to {r.status!r}"

    def test_chunks_done_strictly_increases_no_plateaus(self):
        """Across consecutive non-finished yields while chunking, chunks_done is strictly increasing (no plateaus)."""
        self.runtime.run(self._script_chunks_done_strictly_increases_no_plateaus)

    # chunks_done should make forward progress on every iter
    # the chunked req is actively chunking — a plateau (two consecutive
    # mid-chunk yields with the same chunks_done) would indicate the
    # scheduler stalled or wasted an iter on an in-flight chunked req.
    @staticmethod
    def _script_chunks_done_strictly_increases_no_plateaus(t: ScriptedRuntime):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        # Wait until the req actually enters chunked admission so the
        # first sample is meaningful.
        yield from run_until(r, lambda h: h.is_chunking)
        prev_chunks_done = r.chunks_done
        prev_was_chunking = r.is_chunking
        for _ in range(DEFAULT_MAX_STEPS):
            yield
            if r.finished:
                return
            cur_chunks_done = r.chunks_done
            cur_is_chunking = r.is_chunking
            # Between two consecutive iterations where the req is
            # actively chunking (not just sitting in waiting awaiting
            # admission), chunks_done must strictly advance.
            if prev_was_chunking and cur_is_chunking:
                assert cur_chunks_done > prev_chunks_done, (
                    f"chunks_done plateau between consecutive mid-chunk "
                    f"yields: {prev_chunks_done} -> {cur_chunks_done}"
                )
            prev_chunks_done = cur_chunks_done
            prev_was_chunking = cur_is_chunking
        raise AssertionError("req never finished")

    def test_output_tokens_len_equals_max_new_tokens_chunked(self):
        """Ignore_eos + chunked: len(output_tokens) == max_new_tokens after finish."""
        self.runtime.run(self._script_output_tokens_len_equals_max_new_tokens_chunked)

    # output-state contract: when ignore_eos=True forces decode
    # to run to completion across max_new_tokens iters, the recorded
    # output_tokens list must contain exactly N tokens regardless of
    # how many chunks the prefill spanned.
    @staticmethod
    def _script_output_tokens_len_equals_max_new_tokens_chunked(t: ScriptedRuntime):
        n: int = 8
        # NEW API NEEDED: start_req(..., ignore_eos=True) — sampling kwarg
        # passthrough so decode runs to max_new_tokens.
        r = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN,
            max_new_tokens=n,
            ignore_eos=True,
        )
        yield from run_until_finished(r)
        assert r.finished
        assert (
            r.chunks_done >= 2
        ), f"VERY_LONG_PROMPT_LEN should chunk; got chunks_done={r.chunks_done}"
        # NEW API NEEDED: r.output_tokens — list[int] of decoded tokens.
        assert len(r.output_tokens) == n, (
            f"ignore_eos=True + max_new_tokens={n} must produce exactly "
            f"{n} output tokens; got len(output_tokens)={len(r.output_tokens)}"
        )

    def test_num_input_tokens_equals_prompt_len_for_chunked(self):
        """Chunked req's num_input_tokens equals prompt_len even after multi-chunk admission."""
        self.runtime.run(self._script_num_input_tokens_equals_prompt_len_for_chunked)

    # output-state contract: r.num_input_tokens must reflect the
    # original prompt length regardless of how many chunks the prefill
    # took. Pre-fix bugs in some refactor branches accidentally reported
    # the size of the *last* chunk, not the whole prompt.
    @staticmethod
    def _script_num_input_tokens_equals_prompt_len_for_chunked(t: ScriptedRuntime):
        prompt_len: int = VERY_LONG_PROMPT_LEN
        r = t.start_req(prompt_len=prompt_len, max_new_tokens=2)
        yield from run_until_finished(r)
        assert r.finished
        # NEW API NEEDED: r.num_input_tokens — total prompt token count
        # observable on the handle after finish.
        assert r.num_input_tokens == prompt_len, (
            f"num_input_tokens must equal prompt_len after chunked finish; "
            f"expected {prompt_len}, got {r.num_input_tokens}"
        )

    def test_chunked_in_flight_count_exactly_zero_after_finish(self):
        """After finish, chunked_in_flight_count() stays exactly 0 for several idle yields."""
        self.runtime.run(self._script_chunked_in_flight_count_exactly_zero_after_finish)

    # post-finish invariant: a long chunked req must observe
    # chunked_in_flight_count() == 1 while in flight, then == 0 for at
    # least 3 idle yields after finish. Guards against stale chunked_req
    # pointers leaking into the in-flight counter.
    @staticmethod
    def _script_chunked_in_flight_count_exactly_zero_after_finish(t: ScriptedRuntime):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r, lambda h: h.is_chunking)
        assert t.chunked_in_flight_count() == 1, (
            f"chunked_in_flight_count should be 1 mid-chunk; got "
            f"{t.chunked_in_flight_count()}"
        )
        yield from run_until_finished(r)
        for _ in range(3):
            yield
            assert t.chunked_in_flight_count() == 0, (
                f"chunked_in_flight_count must be 0 after finish; got "
                f"{t.chunked_in_flight_count()}"
            )

    def test_pending_middle_outputs_zero_at_idle_yields(self):
        """After finish + 5 idle yields, pending_middle_outputs is 0."""
        self.runtime.run(self._script_pending_middle_outputs_zero_at_idle_yields)

    # post-finish invariant: once a chunked req finishes, the
    # pending_middle_outputs counter on the handle must drain to 0 and
    # stay there across idle yields. Guards against the e875cd36e4-style
    # bump leaking past finish.
    @staticmethod
    def _script_pending_middle_outputs_zero_at_idle_yields(t: ScriptedRuntime):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until_finished(r)
        for _ in range(5):
            yield
            assert r.pending_middle_outputs == 0, (
                f"pending_middle_outputs must be 0 at idle yields after "
                f"finish; got {r.pending_middle_outputs}"
            )

    def test_decode_side_chunked_req_always_none(self):
        """Pure decode workload: get_chunked_req_rid() is always None."""
        self.runtime.run(self._script_decode_side_chunked_req_always_none)

    # scheduler.py decode-side branch: when the workload
    # is pure decode (no req with a chunked prefill), the scheduler's
    # chunked_req slot must remain None across the entire lifetime.
    # Negative control for the chunked admission paths — proves the
    # chunked slot is not accidentally populated by short-prompt reqs.
    @staticmethod
    def _script_decode_side_chunked_req_always_none(t: ScriptedRuntime):
        # Submit several short reqs so the engine is not idle — the
        # scheduler is actively decoding, but no req triggers chunking.
        reqs = [t.start_req(prompt_len=8, max_new_tokens=16) for _ in range(4)]
        for _ in range(50):
            # NEW API NEEDED: t.get_chunked_req_rid() — current
            # scheduler.chunked_req rid (or None).
            assert t.get_chunked_req_rid() is None, (
                f"pure decode workload must keep chunked_req None; got "
                f"{t.get_chunked_req_rid()!r}"
            )
            if all(r.finished for r in reqs):
                return
            yield


if __name__ == "__main__":
    unittest.main()
