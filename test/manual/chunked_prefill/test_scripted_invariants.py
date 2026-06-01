import unittest

from sglang.test.scripted_runtime.context import ScriptedContext
from sglang.test.scripted_runtime.req_handle import ScriptedReqHandle
from sglang.test.scripted_runtime.test_case import ScriptedTestCase
from sglang.test.scripted_runtime_chunked_helpers import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_MAX_STEPS,
    VERY_LONG_PROMPT_LEN,
    base_engine_kwargs,
    run_until,
    run_until_all_finished,
    run_until_finished,
)


class TestInvariantsBasic(ScriptedTestCase):
    ENGINE_KWARGS = base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE)

    def test_status_unknown_before_submit(self):
        self.server.execute_script(self._script_status_unknown_before_submit)

    @staticmethod
    def _script_status_unknown_before_submit(t: ScriptedContext):
        bogus = ScriptedReqHandle(
            rid="never-submitted", scheduler_hook=t._scheduler_hook
        )
        assert bogus.status == "unknown"
        yield

    def test_status_finished_after_done(self):
        self.server.execute_script(self._script_status_finished_after_done)

    @staticmethod
    def _script_status_finished_after_done(t: ScriptedContext):
        r = t.start_req(prompt_len=16, max_new_tokens=2)
        yield from run_until_finished(r)
        assert r.status == "finished"

    def test_kv_pages_zero_after_finish(self):
        self.server.execute_script(self._script_kv_pages_zero_after_finish)

    @staticmethod
    def _script_kv_pages_zero_after_finish(t: ScriptedContext):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until_finished(r)
        assert r.kv_pages == 0

    def test_kv_pages_positive_continuously_mid_chunk(self):
        self.server.execute_script(
            self._script_kv_pages_positive_continuously_mid_chunk
        )

    @staticmethod
    def _script_kv_pages_positive_continuously_mid_chunk(t: ScriptedContext):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r, lambda h: h.is_chunking)
        observed_chunking = False
        for _ in range(DEFAULT_MAX_STEPS):
            if r.is_chunking:
                observed_chunking = True
                assert (
                    r.kv_pages > 0
                ), f"kv_pages must be > 0 while is_chunking; got {r.kv_pages}"
            if r.finished:
                break
            yield
        assert observed_chunking, "test must observe at least one mid-chunk iter"
        assert r.finished

    def test_batch_composition_consistent_with_status(self):
        self.server.execute_script(
            self._script_batch_composition_consistent_with_status
        )

    @staticmethod
    def _script_batch_composition_consistent_with_status(t: ScriptedContext):
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
        self.server.execute_script(self._script_is_idle_excludes_chunked_in_flight)

    @staticmethod
    def _script_is_idle_excludes_chunked_in_flight(t: ScriptedContext):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        for _ in range(DEFAULT_MAX_STEPS):
            if (1 if t._scheduler.chunked_req is not None else 0) > 0:
                assert not t.is_idle, "is_idle must be False when chunked is in flight"
            if r.finished:
                return
            yield
        raise AssertionError("req never finished")

    def test_finish_event_count_exactly_one(self):
        self.server.execute_script(self._script_finish_event_count_exactly_one)

    @staticmethod
    def _script_finish_event_count_exactly_one(t: ScriptedContext):
        r = t.start_req(prompt_len=16, max_new_tokens=2)
        yield from run_until_finished(r)
        assert r.finish_event_count == 1

    def test_kv_pages_non_negative(self):
        self.server.execute_script(self._script_kv_pages_non_negative)

    @staticmethod
    def _script_kv_pages_non_negative(t: ScriptedContext):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        for _ in range(DEFAULT_MAX_STEPS):
            assert r.kv_pages >= 0
            if r.finished:
                return
            yield
        raise AssertionError("req never finished")

    def test_inflight_middle_chunks_non_negative(self):
        self.server.execute_script(self._script_inflight_middle_chunks_non_negative)

    @staticmethod
    def _script_inflight_middle_chunks_non_negative(t: ScriptedContext):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        for _ in range(DEFAULT_MAX_STEPS):
            assert r.req.inflight_middle_chunks >= 0
            if r.finished:
                return
            yield
        raise AssertionError("req never finished")

    def test_lock_refs_non_negative(self):
        self.server.execute_script(self._script_lock_refs_non_negative)

    @staticmethod
    def _script_lock_refs_non_negative(t: ScriptedContext):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        for _ in range(DEFAULT_MAX_STEPS):
            assert r.lock_refs >= 0
            if r.finished:
                return
            yield
        raise AssertionError("req never finished")

    def test_chunked_in_flight_count_le_one(self):
        self.server.execute_script(self._script_chunked_in_flight_count_le_one)

    @staticmethod
    def _script_chunked_in_flight_count_le_one(t: ScriptedContext):
        reqs = [
            t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
            for _ in range(3)
        ]
        for _ in range(DEFAULT_MAX_STEPS * 3):
            assert (1 if t._scheduler.chunked_req is not None else 0) <= 1
            if all(r.finished for r in reqs):
                return
            yield
        raise AssertionError("not all reqs finished")

    def test_active_reqs_listing(self):
        self.server.execute_script(self._script_active_reqs_listing)

    @staticmethod
    def _script_active_reqs_listing(t: ScriptedContext):
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
        self.server.execute_script(self._script_batch_composition_disjoint_subsets)

    @staticmethod
    def _script_batch_composition_disjoint_subsets(t: ScriptedContext):
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
        self.server.execute_script(self._script_finished_means_chunks_done_stable)

    @staticmethod
    def _script_finished_means_chunks_done_stable(t: ScriptedContext):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until_finished(r)
        snap = r.chunks_done
        for _ in range(10):
            yield
            assert r.chunks_done == snap

    def test_finished_means_kv_pages_stays_zero(self):
        self.server.execute_script(self._script_finished_means_kv_pages_stays_zero)

    @staticmethod
    def _script_finished_means_kv_pages_stays_zero(t: ScriptedContext):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until_finished(r)
        for _ in range(10):
            yield
            assert r.kv_pages == 0

    def test_engine_stats_keys_present(self):
        self.server.execute_script(self._script_engine_stats_keys_present)

    @staticmethod
    def _script_engine_stats_keys_present(t: ScriptedContext):
        stats = t.engine_stats()
        assert isinstance(stats, dict)
        assert "kv_pool_free" in stats
        assert "row_pool_free" in stats
        yield

    def test_kv_pool_recovers_to_baseline(self):
        self.server.execute_script(self._script_kv_pool_recovers_to_baseline)

    @staticmethod
    def _script_kv_pool_recovers_to_baseline(t: ScriptedContext):
        before = t.engine_stats()["kv_pool_free"]
        reqs = [t.start_req(prompt_len=16, max_new_tokens=2) for _ in range(8)]
        yield from run_until_all_finished(reqs)
        after = t.engine_stats()["kv_pool_free"]
        assert after >= before

    def test_hundred_reqs_no_leak(self):
        self.server.execute_script(self._script_hundred_reqs_no_leak)

    @staticmethod
    def _script_hundred_reqs_no_leak(t: ScriptedContext):
        baseline = t.engine_stats()
        reqs = [t.start_req(prompt_len=16, max_new_tokens=2) for _ in range(100)]
        yield from run_until_all_finished(reqs, max_steps=4000)
        final = t.engine_stats()
        assert (
            final["kv_pool_free"] >= baseline["kv_pool_free"]
        ), f"KV leak: {baseline['kv_pool_free']} -> {final['kv_pool_free']}"
        assert final["row_pool_free"] >= baseline["row_pool_free"]

    def test_two_hundred_reqs_no_leak(self):
        self.server.execute_script(self._script_two_hundred_reqs_no_leak)

    @staticmethod
    def _script_two_hundred_reqs_no_leak(t: ScriptedContext):
        baseline = t.engine_stats()
        reqs = [t.start_req(prompt_len=8, max_new_tokens=1) for _ in range(200)]
        yield from run_until_all_finished(reqs, max_steps=8000)
        final = t.engine_stats()
        assert final["kv_pool_free"] >= baseline["kv_pool_free"]

    def test_five_hundred_reqs_no_leak(self):
        self.server.execute_script(self._script_five_hundred_reqs_no_leak)

    @staticmethod
    def _script_five_hundred_reqs_no_leak(t: ScriptedContext):
        baseline = t.engine_stats()
        reqs = [t.start_req(prompt_len=8, max_new_tokens=1) for _ in range(500)]
        yield from run_until_all_finished(reqs, max_steps=15000)
        final = t.engine_stats()
        assert final["kv_pool_free"] >= baseline["kv_pool_free"]

    def test_long_lived_engine_reps_chunked(self):
        self.server.execute_script(self._script_long_lived_engine_reps_chunked)

    @staticmethod
    def _script_long_lived_engine_reps_chunked(t: ScriptedContext):
        baseline = t.engine_stats()
        for _ in range(20):
            reqs = [
                t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
                for _ in range(5)
            ]
            yield from run_until_all_finished(reqs, max_steps=2000)
            for r in reqs:
                assert r.finished
        final = t.engine_stats()
        assert final["kv_pool_free"] >= baseline["kv_pool_free"]

    def test_sustained_long_chunked_load(self):
        self.server.execute_script(self._script_sustained_long_chunked_load)

    @staticmethod
    def _script_sustained_long_chunked_load(t: ScriptedContext):
        reqs = [
            t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
            for _ in range(30)
        ]
        for _ in range(DEFAULT_MAX_STEPS * 20):
            assert (1 if t._scheduler.chunked_req is not None else 0) <= 1
            if all(r.finished for r in reqs):
                break
            yield
        for r in reqs:
            assert r.finished

    def test_round_robin_short_and_chunked(self):
        self.server.execute_script(self._script_round_robin_short_and_chunked)

    @staticmethod
    def _script_round_robin_short_and_chunked(t: ScriptedContext):
        baseline = t.engine_stats()
        for _ in range(5):
            shorts = [t.start_req(prompt_len=16, max_new_tokens=2) for _ in range(10)]
            chunked = [
                t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
                for _ in range(1)
            ]
            yield from run_until_all_finished(shorts + chunked, max_steps=2000)
            for r in shorts + chunked:
                assert r.finished
        final = t.engine_stats()
        assert final["kv_pool_free"] >= baseline["kv_pool_free"]

    def test_long_decode_then_many_short(self):
        self.server.execute_script(self._script_long_decode_then_many_short)

    @staticmethod
    def _script_long_decode_then_many_short(t: ScriptedContext):
        long_decode = t.start_req(prompt_len=16, max_new_tokens=256)
        shorts = [t.start_req(prompt_len=8, max_new_tokens=2) for _ in range(50)]
        all_reqs = [long_decode] + shorts
        for _ in range(DEFAULT_MAX_STEPS * 20):
            assert (
                t._scheduler.chunked_req.rid
                if t._scheduler.chunked_req is not None
                else None
            ) is None
            if all(r.finished for r in all_reqs):
                return
            yield
        raise AssertionError("not all reqs finished")

    def test_chunked_in_flight_count_never_above_one_long_run(self):
        self.server.execute_script(
            self._script_chunked_in_flight_count_never_above_one_long_run
        )

    @staticmethod
    def _script_chunked_in_flight_count_never_above_one_long_run(t: ScriptedContext):
        reqs = [
            t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
            for _ in range(50)
        ]
        for _ in range(DEFAULT_MAX_STEPS * 60):
            assert (1 if t._scheduler.chunked_req is not None else 0) <= 1
            if all(r.finished for r in reqs):
                return
            yield
        raise AssertionError("not all reqs finished")

    def test_engine_stats_monotone_after_each_batch(self):
        self.server.execute_script(self._script_engine_stats_monotone_after_each_batch)

    @staticmethod
    def _script_engine_stats_monotone_after_each_batch(t: ScriptedContext):
        last = None
        for _ in range(10):
            reqs = [t.start_req(prompt_len=16, max_new_tokens=2) for _ in range(8)]
            yield from run_until_all_finished(reqs)
            cur = t.engine_stats()["kv_pool_free"]
            if last is not None:
                assert cur >= last - 1, f"KV pool drifted: {last} -> {cur}"
            last = cur

    def test_chunked_status_never_mid_chunk_running(self):
        self.server.execute_script(self._script_chunked_status_never_mid_chunk_running)

    @staticmethod
    def _script_chunked_status_never_mid_chunk_running(t: ScriptedContext):
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

    def test_inflight_middle_chunks_caps_at_one(self):
        self.server.execute_script(self._script_inflight_middle_chunks_caps_at_one)

    @staticmethod
    def _script_inflight_middle_chunks_caps_at_one(t: ScriptedContext):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        running_max = r.req.inflight_middle_chunks
        running_max_post_finish = 0
        post_finish_samples = 0
        for _ in range(DEFAULT_MAX_STEPS):
            yield
            cur = r.req.inflight_middle_chunks
            running_max = max(running_max, cur)
            if r.finished:
                running_max_post_finish = max(running_max_post_finish, cur)
                post_finish_samples += 1
                if post_finish_samples >= 5:
                    break
        assert r.finished, "req never finished"
        assert running_max == 1, (
            f"inflight_middle_chunks must reach exactly 1 across the chunked "
            f"lifecycle (the cap from revert e875cd36e4); observed max={running_max}"
        )
        assert running_max_post_finish == 0, (
            f"inflight_middle_chunks must be reset to 0 after finish; "
            f"observed max post-finish={running_max_post_finish}"
        )

    def test_status_never_finished_to_waiting(self):
        self.server.execute_script(self._script_status_never_finished_to_waiting)

    @staticmethod
    def _script_status_never_finished_to_waiting(t: ScriptedContext):
        r = t.start_req(prompt_len=16, max_new_tokens=2)
        yield from run_until_finished(r)
        assert r.status == "finished"
        for _ in range(50):
            yield
            assert (
                r.status == "finished"
            ), f"status rolled back from finished to {r.status!r}"

    def test_chunks_done_strictly_increases_no_plateaus(self):
        self.server.execute_script(
            self._script_chunks_done_strictly_increases_no_plateaus
        )

    @staticmethod
    def _script_chunks_done_strictly_increases_no_plateaus(t: ScriptedContext):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r, lambda h: h.is_chunking)
        prev_chunks_done = r.chunks_done
        prev_was_chunking = r.is_chunking
        for _ in range(DEFAULT_MAX_STEPS):
            yield
            if r.finished:
                return
            cur_chunks_done = r.chunks_done
            cur_is_chunking = r.is_chunking
            if prev_was_chunking and cur_is_chunking:
                assert cur_chunks_done > prev_chunks_done, (
                    f"chunks_done plateau between consecutive mid-chunk "
                    f"yields: {prev_chunks_done} -> {cur_chunks_done}"
                )
            prev_chunks_done = cur_chunks_done
            prev_was_chunking = cur_is_chunking
        raise AssertionError("req never finished")

    def test_output_tokens_len_equals_max_new_tokens_chunked(self):
        self.server.execute_script(
            self._script_output_tokens_len_equals_max_new_tokens_chunked
        )

    @staticmethod
    def _script_output_tokens_len_equals_max_new_tokens_chunked(t: ScriptedContext):
        n: int = 8
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
        assert len(r.req.output_ids) == n, (
            f"ignore_eos=True + max_new_tokens={n} must produce exactly "
            f"{n} output tokens; got len(output_tokens)={len(r.req.output_ids)}"
        )

    def test_num_input_tokens_equals_prompt_len_for_chunked(self):
        self.server.execute_script(
            self._script_num_input_tokens_equals_prompt_len_for_chunked
        )

    @staticmethod
    def _script_num_input_tokens_equals_prompt_len_for_chunked(t: ScriptedContext):
        prompt_len: int = VERY_LONG_PROMPT_LEN
        r = t.start_req(prompt_len=prompt_len, max_new_tokens=2)
        yield from run_until_finished(r)
        assert r.finished
        assert len(r.req.origin_input_ids) == prompt_len, (
            f"num_input_tokens must equal prompt_len after chunked finish; "
            f"expected {prompt_len}, got {len(r.req.origin_input_ids)}"
        )

    def test_chunked_in_flight_count_exactly_zero_after_finish(self):
        self.server.execute_script(
            self._script_chunked_in_flight_count_exactly_zero_after_finish
        )

    @staticmethod
    def _script_chunked_in_flight_count_exactly_zero_after_finish(t: ScriptedContext):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r, lambda h: h.is_chunking)
        assert (1 if t._scheduler.chunked_req is not None else 0) == 1, (
            f"chunked_in_flight_count should be 1 mid-chunk; got "
            f"{(1 if t._scheduler.chunked_req is not None else 0)}"
        )
        yield from run_until_finished(r)
        for _ in range(3):
            yield
            assert (1 if t._scheduler.chunked_req is not None else 0) == 0, (
                f"chunked_in_flight_count must be 0 after finish; got "
                f"{(1 if t._scheduler.chunked_req is not None else 0)}"
            )

    def test_inflight_middle_chunks_zero_at_idle_yields(self):
        self.server.execute_script(
            self._script_inflight_middle_chunks_zero_at_idle_yields
        )

    @staticmethod
    def _script_inflight_middle_chunks_zero_at_idle_yields(t: ScriptedContext):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until_finished(r)
        for _ in range(5):
            yield
            assert r.req.inflight_middle_chunks == 0, (
                f"inflight_middle_chunks must be 0 at idle yields after "
                f"finish; got {r.req.inflight_middle_chunks}"
            )

    def test_extend_batch_idx_monotonic_invariant(self):
        self.server.execute_script(self._script_extend_batch_idx_monotonic_invariant)

    @staticmethod
    def _script_extend_batch_idx_monotonic_invariant(t: ScriptedContext):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        prev_extend_batch_idx: int = -1
        prev_is_retracted: bool = False
        for _ in range(DEFAULT_MAX_STEPS):
            req = t._find_req_by_rid(r.rid)
            if req is not None:
                cur_extend_batch_idx = req.extend_batch_idx
                cur_is_retracted = req.is_retracted
                if prev_extend_batch_idx >= 0:
                    if cur_extend_batch_idx < prev_extend_batch_idx:
                        assert cur_is_retracted and not prev_is_retracted, (
                            f"extend_batch_idx regressed without retract: "
                            f"{prev_extend_batch_idx} -> {cur_extend_batch_idx}, "
                            f"prev_is_retracted={prev_is_retracted}, "
                            f"cur_is_retracted={cur_is_retracted}"
                        )
                prev_extend_batch_idx = cur_extend_batch_idx
                prev_is_retracted = cur_is_retracted
            if r.finished:
                return
            yield
        raise AssertionError("req never finished")

    def test_inflight_decrement_only_on_final_invariant(self):
        self.server.execute_script(
            self._script_inflight_decrement_only_on_final_invariant
        )

    @staticmethod
    def _script_inflight_decrement_only_on_final_invariant(t: ScriptedContext):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        prev_inflight: int = 0
        prev_was_chunked_slot: bool = False
        prev_finished: bool = False
        observed_decrement: bool = False
        for _ in range(DEFAULT_MAX_STEPS):
            s = t._scheduler
            req = t._find_req_by_rid(r.rid)
            cur_inflight = req.inflight_middle_chunks if req is not None else 0
            cur_is_chunked_slot = (
                s.chunked_req is not None and s.chunked_req.rid == r.rid
            )
            cur_finished = req.finished() if req is not None else True
            if cur_inflight < prev_inflight:
                observed_decrement = True
                slot_just_released = prev_was_chunked_slot and not cur_is_chunked_slot
                finish_just_happened = (not prev_finished) and cur_finished
                assert slot_just_released or finish_just_happened, (
                    f"inflight_middle_chunks decreased ({prev_inflight} -> "
                    f"{cur_inflight}) without chunked slot release or req "
                    f"finish; prev_was_chunked_slot={prev_was_chunked_slot}, "
                    f"cur_is_chunked_slot={cur_is_chunked_slot}, "
                    f"prev_finished={prev_finished}, cur_finished={cur_finished}"
                )
            prev_inflight = cur_inflight
            prev_was_chunked_slot = cur_is_chunked_slot
            prev_finished = cur_finished
            if r.finished:
                break
            yield
        assert r.finished
        assert observed_decrement, (
            "test must observe at least one inflight_middle_chunks decrement "
            "across the chunked lifecycle"
        )

    def test_decode_side_chunked_req_always_none(self):
        self.server.execute_script(self._script_decode_side_chunked_req_always_none)

    @staticmethod
    def _script_decode_side_chunked_req_always_none(t: ScriptedContext):
        reqs = [t.start_req(prompt_len=8, max_new_tokens=16) for _ in range(4)]
        for _ in range(50):
            assert (
                t._scheduler.chunked_req.rid
                if t._scheduler.chunked_req is not None
                else None
            ) is None, (
                f"pure decode workload must keep chunked_req None; got "
                f"{(t._scheduler.chunked_req.rid if t._scheduler.chunked_req is not None else None)!r}"
            )
            if all(r.finished for r in reqs):
                return
            yield


if __name__ == "__main__":
    unittest.main()
