import unittest

from sglang.test.scripted_runtime.context import ScriptedContext
from sglang.test.scripted_runtime.test_case import ScriptedTestCase
from sglang.test.scripted_runtime_chunked_helpers import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_MAX_STEPS,
    VERY_LONG_PROMPT_LEN,
    base_engine_kwargs,
    chunked_req_of,
    inflight_middle_chunks_of,
    run_until,
    run_until_all_finished,
    run_until_finished,
)


class TestInvariantsBasic(ScriptedTestCase):
    ENGINE_KWARGS = base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE)

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
        for _ in range(12):
            actives_after = t.list_active_reqs()
            if all(h.rid not in (r1.rid, r2.rid) for h in actives_after):
                break
            yield
        actives_after = t.list_active_reqs()
        assert all(h.rid not in (r1.rid, r2.rid) for h in actives_after)

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
        assert "req_pool_free" in stats
        yield

    def test_kv_pool_recovers_to_baseline(self):
        self.server.execute_script(self._script_kv_pool_recovers_to_baseline)

    @staticmethod
    def _script_kv_pool_recovers_to_baseline(t: ScriptedContext):
        before = t.engine_stats()["kv_pool_free"]
        reqs = [t.start_req(prompt_len=16, max_new_tokens=2) for _ in range(8)]
        yield from run_until_all_finished(reqs)
        for _ in range(40):
            if t.is_fully_idle:
                break
            yield
        t.flush_cache()
        yield
        after = t.engine_stats()["kv_pool_free"]
        assert after >= before

    def test_hundred_reqs_no_leak(self):
        self.server.execute_script(self._script_hundred_reqs_no_leak)

    @staticmethod
    def _script_hundred_reqs_no_leak(t: ScriptedContext):
        baseline = t.engine_stats()
        reqs = [t.start_req(prompt_len=16, max_new_tokens=2) for _ in range(100)]
        yield from run_until_all_finished(reqs, max_steps=4000)
        for _ in range(40):
            if t.is_fully_idle:
                break
            yield
        t.flush_cache()
        yield
        final = t.engine_stats()
        assert (
            final["kv_pool_free"] >= baseline["kv_pool_free"]
        ), f"KV leak: {baseline['kv_pool_free']} -> {final['kv_pool_free']}"
        assert final["req_pool_free"] >= baseline["req_pool_free"]

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
        for _ in range(40):
            if t.is_fully_idle:
                break
            yield
        t.flush_cache()
        yield
        final = t.engine_stats()
        assert final["kv_pool_free"] >= baseline["kv_pool_free"]

    def test_sustained_long_chunked_load(self):
        self.server.execute_script(self._script_sustained_long_chunked_load)

    @staticmethod
    def _script_sustained_long_chunked_load(t: ScriptedContext):
        expected_chunks_done = VERY_LONG_PROMPT_LEN // DEFAULT_CHUNK_SIZE
        baseline_kv = t.engine_stats()["kv_pool_free"]
        reqs = [
            t.start_req(
                prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2, prompt_token=10 + i
            )
            for i in range(30)
        ]
        yield from run_until_all_finished(reqs, max_steps=DEFAULT_MAX_STEPS * 20)
        for r in reqs:
            assert r.finished
            assert r.chunks_done == expected_chunks_done, (
                f"VERY_LONG_PROMPT_LEN must take exactly {expected_chunks_done} "
                f"chunks; got chunks_done={r.chunks_done}"
            )
        for _ in range(40):
            if t.is_fully_idle:
                break
            yield
        t.flush_cache()
        yield
        final_kv = t.engine_stats()["kv_pool_free"]
        assert (
            final_kv >= baseline_kv
        ), f"KV leak after sustained chunked load: {baseline_kv} -> {final_kv}"

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
        for _ in range(40):
            if t.is_fully_idle:
                break
            yield
        t.flush_cache()
        yield
        final = t.engine_stats()
        assert final["kv_pool_free"] >= baseline["kv_pool_free"]

    def test_long_decode_then_many_short(self):
        self.server.execute_script(self._script_long_decode_then_many_short)

    @staticmethod
    def _script_long_decode_then_many_short(t: ScriptedContext):
        expected_chunks_done = VERY_LONG_PROMPT_LEN // DEFAULT_CHUNK_SIZE
        long_max_new_tokens = 256
        long_decode = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN,
            max_new_tokens=long_max_new_tokens,
            ignore_eos=True,
        )
        shorts = [t.start_req(prompt_len=8, max_new_tokens=2) for _ in range(50)]
        all_reqs = [long_decode] + shorts
        yield from run_until_all_finished(all_reqs, max_steps=DEFAULT_MAX_STEPS * 20)
        for r in all_reqs:
            assert r.finished
        assert long_decode.chunks_done == expected_chunks_done, (
            f"long req must chunk across exactly {expected_chunks_done} chunks; "
            f"got chunks_done={long_decode.chunks_done}"
        )
        assert len(long_decode.req.output_ids) == long_max_new_tokens, (
            f"ignore_eos long req must decode exactly {long_max_new_tokens} "
            f"tokens; got len(output_ids)={len(long_decode.req.output_ids)}"
        )

    def test_engine_stats_monotone_after_each_batch(self):
        self.server.execute_script(self._script_engine_stats_monotone_after_each_batch)

    @staticmethod
    def _script_engine_stats_monotone_after_each_batch(t: ScriptedContext):
        last = None
        for _ in range(10):
            reqs = [t.start_req(prompt_len=16, max_new_tokens=2) for _ in range(8)]
            yield from run_until_all_finished(reqs)
            for _ in range(40):
                if t.is_fully_idle:
                    break
                yield
            t.flush_cache()
            yield
            cur = t.engine_stats()["kv_pool_free"]
            if last is not None:
                assert cur >= last - 1, f"KV pool drifted: {last} -> {cur}"
            last = cur

    def test_inflight_middle_chunks_caps_at_one(self):
        self.server.execute_script(self._script_inflight_middle_chunks_caps_at_one)

    @staticmethod
    def _script_inflight_middle_chunks_caps_at_one(t: ScriptedContext):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        running_max = 0
        running_max_post_finish = 0
        post_finish_samples = 0
        for _ in range(DEFAULT_MAX_STEPS):
            yield
            req = r.req
            cur = inflight_middle_chunks_of(req) if req is not None else 0
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
        assert r.chunks_done >= 2, (
            f"VERY_LONG_PROMPT_LEN should chunk so this invariant is exercised "
            f"on a real multi-chunk prefill; got chunks_done={r.chunks_done}"
        )
        assert r.remaining_prompt_tokens == 0, (
            f"the whole prompt must be committed after a chunked finish; "
            f"remaining_prompt_tokens={r.remaining_prompt_tokens}"
        )

    def test_chunked_in_flight_count_exactly_zero_after_finish(self):
        self.server.execute_script(
            self._script_chunked_in_flight_count_exactly_zero_after_finish
        )

    @staticmethod
    def _script_chunked_in_flight_count_exactly_zero_after_finish(t: ScriptedContext):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r, lambda h: h.is_chunking)
        assert (1 if chunked_req_of(t.scheduler) is not None else 0) == 1, (
            f"chunked_in_flight_count should be 1 mid-chunk; got "
            f"{(1 if chunked_req_of(t.scheduler) is not None else 0)}"
        )
        yield from run_until_finished(r)
        for _ in range(3):
            yield
            assert (1 if chunked_req_of(t.scheduler) is not None else 0) == 0, (
                f"chunked_in_flight_count must be 0 after finish; got "
                f"{(1 if chunked_req_of(t.scheduler) is not None else 0)}"
            )

    def test_extend_batch_idx_monotonic_invariant(self):
        self.server.execute_script(self._script_extend_batch_idx_monotonic_invariant)

    @staticmethod
    def _script_extend_batch_idx_monotonic_invariant(t: ScriptedContext):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=64)
        observed_regression: bool = False

        yield from run_until(
            r,
            lambda h: (
                h.req is not None
                and not h.req.is_retracted
                and h.req.extend_batch_idx > 0
                and h.remaining_prompt_tokens == 0
                and not h.finished
            ),
        )
        pre_retract_idx: int = r.req.extend_batch_idx
        assert not r.req.is_retracted

        t.pause_generation(mode="retract")
        retracted = t.find_req_by_rid(r.rid)
        assert retracted is not None, "retracted req must stay live in the queue"
        if retracted.extend_batch_idx < pre_retract_idx:
            observed_regression = True
            assert retracted.is_retracted, (
                f"extend_batch_idx regressed without retract flag: "
                f"{pre_retract_idx} -> {retracted.extend_batch_idx}"
            )
        t.continue_generation()

        prev_extend_batch_idx: int = -1
        regressions: int = 0
        for _ in range(DEFAULT_MAX_STEPS):
            req = t.find_req_by_rid(r.rid)
            if req is not None:
                cur_extend_batch_idx = req.extend_batch_idx
                if (
                    prev_extend_batch_idx >= 0
                    and cur_extend_batch_idx < prev_extend_batch_idx
                ):
                    regressions += 1
                    observed_regression = True
                    assert regressions == 1, (
                        f"extend_batch_idx regressed more than once for a single "
                        f"retract episode: "
                        f"{prev_extend_batch_idx} -> {cur_extend_batch_idx}"
                    )
                prev_extend_batch_idx = cur_extend_batch_idx
            if r.finished:
                break
            yield
        assert r.finished, "req never finished"
        assert observed_regression, (
            "retract must reset extend_batch_idx, producing the regression this "
            "test guards"
        )

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
            s = t.scheduler
            req = t.find_req_by_rid(r.rid)
            cur_inflight = inflight_middle_chunks_of(req) if req is not None else 0
            cur_is_chunked_slot = (
                chunked_req_of(s) is not None and chunked_req_of(s).rid == r.rid
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


if __name__ == "__main__":
    unittest.main()
