import unittest

from sglang.test.scripted_runtime.context import ScriptedContext
from sglang.test.scripted_runtime.test_case import ScriptedTestCase
from sglang.test.scripted_runtime_chunked_helpers import (
    BALLAST_MAX_NEW_TOKENS,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_MAX_STEPS,
    VERY_LONG_PROMPT_LEN,
    base_engine_kwargs,
    chunked_req_of,
    run_until,
    run_until_all_finished,
    run_until_finished,
)


def _drain_flush_then_assert_no_kv_leak(t: ScriptedContext, baseline: dict):
    for _ in range(5):
        yield
    t.flush_cache()
    yield
    final = t.engine_stats()
    assert (
        final["kv_pool_free"] >= baseline["kv_pool_free"]
    ), f"KV leak: {baseline['kv_pool_free']} -> {final['kv_pool_free']}"


class TestMultiReqBasic(ScriptedTestCase):
    ENGINE_KWARGS = base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE)

    def test_second_chunked_waits(self):
        self.server.execute_script(self._script_second_chunked_waits)

    @staticmethod
    def _script_second_chunked_waits(t: ScriptedContext):
        r1 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r1, lambda h: h.is_chunking)
        r2 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield

        assert r1.is_chunking, "r1 should still be chunking"
        assert (
            not r2.is_chunking
        ), "r2 must wait for r1's chunk loop to clear before chunking"

        yield from run_until_all_finished([r1, r2])
        assert r1.finished and r2.finished

    def test_hundred_short_reqs(self):
        self.server.execute_script(self._script_hundred_short_reqs)

    @staticmethod
    def _script_hundred_short_reqs(t: ScriptedContext):
        baseline = t.engine_stats()
        reqs = [t.start_req(prompt_len=16, max_new_tokens=2) for _ in range(100)]
        yield from run_until_all_finished(reqs, max_steps=2000)
        for r in reqs:
            assert r.finished
        yield from _drain_flush_then_assert_no_kv_leak(t, baseline)

    def test_five_hundred_short_reqs(self):
        self.server.execute_script(self._script_five_hundred_short_reqs)

    @staticmethod
    def _script_five_hundred_short_reqs(t: ScriptedContext):
        baseline = t.engine_stats()
        reqs = [t.start_req(prompt_len=8, max_new_tokens=1) for _ in range(500)]
        yield from run_until_all_finished(reqs, max_steps=20000)
        for r in reqs:
            assert r.finished
        yield from _drain_flush_then_assert_no_kv_leak(t, baseline)

    def test_mixed_ten_chunked_ten_short(self):
        self.server.execute_script(self._script_mixed_ten_chunked_ten_short)

    @staticmethod
    def _script_mixed_ten_chunked_ten_short(t: ScriptedContext):
        baseline = t.engine_stats()
        chunked = [
            t.start_req(
                prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2, prompt_token=100 + i
            )
            for i in range(10)
        ]
        shorts = [t.start_req(prompt_len=8, max_new_tokens=2) for _ in range(10)]
        all_reqs = chunked + shorts
        finished = False
        for _ in range(DEFAULT_MAX_STEPS * 20):
            assert sum(1 for r in all_reqs if r.is_chunking) <= 1
            if all(r.finished for r in all_reqs):
                finished = True
                break
            yield
        if not finished:
            raise AssertionError("not all reqs finished")
        for r in chunked:
            assert r.chunks_done >= 2
        yield from _drain_flush_then_assert_no_kv_leak(t, baseline)

    def test_submit_during_chunk_mid(self):
        self.server.execute_script(self._script_submit_during_chunk_mid)

    @staticmethod
    def _script_submit_during_chunk_mid(t: ScriptedContext):
        r1 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r1, lambda h: h.is_chunking)
        r2 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield
        r3 = t.start_req(prompt_len=16, max_new_tokens=2)
        yield

        assert r1.is_chunking, "r1 must still hold the single chunked slot"
        assert not r2.is_chunking, "r2 must wait until r1's chunk loop clears"
        assert not r3.is_chunking, "r3 must wait until r1's chunk loop clears"

        yield from run_until_all_finished([r1, r2, r3])
        assert r1.finished and r2.finished and r3.finished

    def test_five_identical_prompts(self):
        self.server.execute_script(self._script_five_identical_prompts)

    @staticmethod
    def _script_five_identical_prompts(t: ScriptedContext):
        r1 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until_finished(r1)

        others = [
            t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
            for _ in range(4)
        ]
        yield from run_until_all_finished(others)
        for r in others:
            assert r.finished
            assert r.chunks_done == 0
        assert r1.finished

    def test_sibling_shared_prefix(self):
        self.server.execute_script(self._script_sibling_shared_prefix)

    @staticmethod
    def _script_sibling_shared_prefix(t: ScriptedContext):
        r1 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until_finished(r1)
        r2 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN + 8, max_new_tokens=2)
        yield from run_until_finished(r2)
        assert r1.finished and r2.finished
        assert (
            r2.chunks_done < r1.chunks_done
        ), "r2 reuses r1's cached prefix, so it should chunk fewer times"

    def test_trickle_per_yield_50(self):
        self.server.execute_script(self._script_trickle_per_yield_50)

    @staticmethod
    def _script_trickle_per_yield_50(t: ScriptedContext):
        baseline = t.engine_stats()
        reqs = []
        for _ in range(50):
            reqs.append(t.start_req(prompt_len=8, max_new_tokens=2))
            yield
        yield from run_until_all_finished(reqs, max_steps=2000)
        for r in reqs:
            assert r.finished
        yield from _drain_flush_then_assert_no_kv_leak(t, baseline)

    def test_submit_then_immediate_abort(self):
        self.server.execute_script(self._script_submit_then_immediate_abort)

    @staticmethod
    def _script_submit_then_immediate_abort(t: ScriptedContext):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        t.abort(r)
        for _ in range(5):
            yield
        assert r.kv_pages == 0
        assert r.req is None or r.req.req_pool_idx is None

    def test_rid_reuse_after_finish(self):
        self.server.execute_script(self._script_rid_reuse_after_finish)

    @staticmethod
    def _script_rid_reuse_after_finish(t: ScriptedContext):
        baseline = t.engine_stats()
        r1 = t.start_req(prompt_len=16, max_new_tokens=2, rid="reuse-rid")
        yield from run_until_finished(r1)
        r2 = t.start_req(prompt_len=16, max_new_tokens=2, rid="reuse-rid")
        yield from run_until_finished(r2)
        assert r1.finished and r2.finished
        yield from _drain_flush_then_assert_no_kv_leak(t, baseline)

    def test_submit_pause_n_resubmit_same_rid(self):
        self.server.execute_script(self._script_submit_pause_n_resubmit_same_rid)

    @staticmethod
    def _script_submit_pause_n_resubmit_same_rid(t: ScriptedContext):
        baseline = t.engine_stats()
        r1 = t.start_req(prompt_len=16, max_new_tokens=2, rid="reuse-200")
        yield from run_until_finished(r1)
        for _ in range(200):
            assert (1 if chunked_req_of(t.scheduler) is not None else 0) == 0
            yield
        r2 = t.start_req(prompt_len=16, max_new_tokens=2, rid="reuse-200")
        yield from run_until_finished(r2)
        assert r2.finished
        yield from _drain_flush_then_assert_no_kv_leak(t, baseline)

    def test_submit_during_decode_of_other(self):
        self.server.execute_script(self._script_submit_during_decode_of_other)

    @staticmethod
    def _script_submit_during_decode_of_other(t: ScriptedContext):
        r1 = t.start_req(prompt_len=16, max_new_tokens=16)
        yield from run_until(r1, lambda h: h.status == "running")
        r2 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r2, lambda h: h.is_chunking)

        comp = t.batch_composition()
        chunked_set = set(comp.get("chunked", []))
        prefill_set = set(comp.get("prefill", []))
        decode_set = set(comp.get("decode", []))
        assert r2.rid in chunked_set, f"r2 should be the chunked req; got {comp}"
        assert chunked_set.isdisjoint(prefill_set)
        assert chunked_set.isdisjoint(decode_set)

        yield from run_until_all_finished([r1, r2], max_steps=DEFAULT_MAX_STEPS * 5)
        assert r1.finished and r2.finished

    def test_two_small_parallel(self):
        self.server.execute_script(self._script_two_small_parallel)

    @staticmethod
    def _script_two_small_parallel(t: ScriptedContext):
        r1 = t.start_req(prompt_len=16, max_new_tokens=4)
        r2 = t.start_req(prompt_len=16, max_new_tokens=4)
        for _ in range(DEFAULT_MAX_STEPS):
            assert (
                chunked_req_of(t.scheduler).rid
                if chunked_req_of(t.scheduler) is not None
                else None
            ) is None
            if r1.finished and r2.finished:
                break
            yield
        assert r1.finished and r2.finished

    def test_one_chunked_plus_many_short(self):
        self.server.execute_script(self._script_one_chunked_plus_many_short)

    @staticmethod
    def _script_one_chunked_plus_many_short(t: ScriptedContext):
        chunked = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        shorts = [t.start_req(prompt_len=16, max_new_tokens=2) for _ in range(5)]
        for _ in range(DEFAULT_MAX_STEPS * 5):
            assert (1 if chunked_req_of(t.scheduler) is not None else 0) <= 1
            if chunked.finished and all(s.finished for s in shorts):
                break
            yield
        assert chunked.chunks_done >= 2
        assert chunked.finished
        for s in shorts:
            assert s.finished

    def test_multiple_chunked_staggered(self):
        self.server.execute_script(self._script_multiple_chunked_staggered)

    @staticmethod
    def _script_multiple_chunked_staggered(t: ScriptedContext):
        reqs = []
        for i in range(4):
            reqs.append(
                t.start_req(
                    prompt_len=VERY_LONG_PROMPT_LEN,
                    max_new_tokens=2,
                    prompt_token=100 + i,
                )
            )
            yield
            assert sum(1 for r in reqs if r.is_chunking) <= 1
            yield
        for _ in range(DEFAULT_MAX_STEPS * 10):
            assert sum(1 for r in reqs if r.is_chunking) <= 1
            if all(r.finished for r in reqs):
                break
            yield
        for r in reqs:
            assert r.finished
            assert r.chunks_done >= 2

    def test_eight_concurrent_chunked(self):
        self.server.execute_script(self._script_eight_concurrent_chunked)

    @staticmethod
    def _script_eight_concurrent_chunked(t: ScriptedContext):
        reqs = [
            t.start_req(
                prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2, prompt_token=100 + i
            )
            for i in range(8)
        ]
        finished = False
        for _ in range(DEFAULT_MAX_STEPS * 5):
            assert sum(1 for r in reqs if r.is_chunking) <= 1
            if all(r.finished for r in reqs):
                finished = True
                break
            yield
        if not finished:
            raise AssertionError("not all reqs finished")
        for r in reqs:
            assert r.chunks_done >= 2

    def test_decode_only_batch(self):
        self.server.execute_script(self._script_decode_only_batch)

    @staticmethod
    def _script_decode_only_batch(t: ScriptedContext):
        reqs = [t.start_req(prompt_len=4, max_new_tokens=8) for _ in range(10)]
        for _ in range(DEFAULT_MAX_STEPS * 3):
            assert (
                chunked_req_of(t.scheduler).rid
                if chunked_req_of(t.scheduler) is not None
                else None
            ) is None, "pure decode workload must never populate chunked_req"
            assert (1 if chunked_req_of(t.scheduler) is not None else 0) == 0
            if all(r.finished for r in reqs):
                return
            yield
        raise AssertionError("not all reqs finished")

    def test_mixed_prefill_lengths(self):
        self.server.execute_script(self._script_mixed_prefill_lengths)

    @staticmethod
    def _script_mixed_prefill_lengths(t: ScriptedContext):
        lens = [8, 16, 32, 64, 128, 256, 512, 1024]
        reqs = [t.start_req(prompt_len=L, max_new_tokens=2) for L in lens]
        by_len = dict(zip(lens, reqs))
        for _ in range(DEFAULT_MAX_STEPS * 10):
            assert sum(1 for r in reqs if r.is_chunking) <= 1
            if all(r.finished for r in reqs):
                break
            yield
        for r in reqs:
            assert r.finished
        for L in (8, 16, 32, 64, 128):
            assert by_len[L].chunks_done <= 1, f"prompt_len={L} should not chunk"
        for L in (512, 1024):
            assert by_len[L].chunks_done >= 2, f"prompt_len={L} should chunk >=2"

    def test_chunked_req_exclusive_of_batch_invariant(self):
        self.server.execute_script(
            self._script_chunked_req_exclusive_of_batch_invariant
        )

    @staticmethod
    def _script_chunked_req_exclusive_of_batch_invariant(t: ScriptedContext):
        r1 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        r2 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        for _ in range(DEFAULT_MAX_STEPS * 5):
            s = t.scheduler
            chunked = chunked_req_of(s)
            running = s.running_batch
            if chunked is not None and running is not None:
                assert chunked not in running.reqs, (
                    f"chunked_req must be exclusive of running_batch.reqs; "
                    f"chunked.rid={chunked.rid!r} appears in running_batch"
                )
            if r1.finished and r2.finished:
                return
            yield
        raise AssertionError("r1 and r2 did not both finish within step budget")

    def test_two_chunked_one_decode(self):
        self.server.execute_script(self._script_two_chunked_one_decode)

    @staticmethod
    def _script_two_chunked_one_decode(t: ScriptedContext):
        chunked1 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        chunked2 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        short = t.start_req(prompt_len=8, max_new_tokens=4)
        all_reqs = [chunked1, chunked2, short]
        for _ in range(DEFAULT_MAX_STEPS * 10):
            comp = t.batch_composition()
            prefill_set = set(comp.get("prefill", []))
            decode_set = set(comp.get("decode", []))
            chunked_set = set(comp.get("chunked", []))
            assert prefill_set.isdisjoint(decode_set)
            assert prefill_set.isdisjoint(chunked_set)
            assert decode_set.isdisjoint(chunked_set)
            if all(r.finished for r in all_reqs):
                break
            yield
        assert chunked1.finished and chunked2.finished and short.finished

    def test_batch_state_query_during_run(self):
        self.server.execute_script(self._script_batch_state_query_during_run)

    @staticmethod
    def _script_batch_state_query_during_run(t: ScriptedContext):
        reqs = [t.start_req(prompt_len=16, max_new_tokens=4) for _ in range(4)]
        for _ in range(DEFAULT_MAX_STEPS):
            comp = t.batch_composition()
            prefill = set(comp.get("prefill", []))
            decode = set(comp.get("decode", []))
            assert prefill & decode == set()
            if all(r.finished for r in reqs):
                return
            yield
        raise AssertionError("not all reqs finished")


class TestMultiReqPriority(ScriptedTestCase):
    ENGINE_KWARGS = base_engine_kwargs(
        chunked_prefill_size=DEFAULT_CHUNK_SIZE,
        enable_priority_scheduling=True,
    )

    def test_parallel_with_priority(self):
        self.server.execute_script(self._script_parallel_with_priority)

    @staticmethod
    def _script_parallel_with_priority(t: ScriptedContext):
        normal = [
            t.start_req(
                prompt_len=16,
                max_new_tokens=BALLAST_MAX_NEW_TOKENS,
                ignore_eos=True,
                priority=0,
            )
            for _ in range(4)
        ]
        yield from run_until(normal[-1], lambda h: h.status == "running")

        t.exhaust_kv(leave_pages=0)
        high = t.start_req(prompt_len=16, max_new_tokens=2, priority=100)

        preempted = False
        for _ in range(DEFAULT_MAX_STEPS * 5):
            if any(r.status == "waiting" for r in normal):
                preempted = True
            if high.finished:
                break
            yield
        assert high.finished, "high-priority req must run to completion"
        assert preempted, "a normal req must be preempted back to the waiting queue"

        t.abort_all()
        for _ in range(DEFAULT_MAX_STEPS):
            if all(r.finished for r in normal):
                break
            yield


class TestMultiReqMixedChunk(ScriptedTestCase):
    ENGINE_KWARGS = base_engine_kwargs(
        chunked_prefill_size=DEFAULT_CHUNK_SIZE,
        enable_mixed_chunk=True,
    )

    def test_long_prefill_chunks_more_with_concurrent_decode(self):
        self.server.execute_script(
            self._script_long_prefill_chunks_more_with_concurrent_decode
        )

    @staticmethod
    def _script_long_prefill_chunks_more_with_concurrent_decode(t: ScriptedContext):
        decoder = t.start_req(prompt_len=16, max_new_tokens=64)
        yield from run_until(decoder, lambda h: h.status == "running")

        long_req = t.start_req(prompt_len=4 * DEFAULT_CHUNK_SIZE, max_new_tokens=2)
        yield from run_until(long_req, lambda h: h.is_chunking)

        saw_mixed_batch: bool = False
        for _ in range(DEFAULT_MAX_STEPS * 5):
            if t.last_batch_forward_mode == "MIXED":
                saw_mixed_batch = True
            if long_req.finished and decoder.finished:
                break
            yield
        assert long_req.finished and decoder.finished

        assert saw_mixed_batch, "expected at least one MIXED (prefill+decode) batch"
        assert long_req.chunks_done >= 4, (
            f"a 4-chunk prompt must chunk >= 4 times under concurrent decode; "
            f"got {long_req.chunks_done}"
        )

    def test_chunked_plus_decode_in_batch(self):
        self.server.execute_script(self._script_chunked_plus_decode_in_batch)

    @staticmethod
    def _script_chunked_plus_decode_in_batch(t: ScriptedContext):
        r2 = t.start_req(prompt_len=8, max_new_tokens=64, ignore_eos=True)
        yield from run_until(
            r2, lambda h: h.status == "running" and len(h.req.output_ids) >= 1
        )

        r1 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r1, lambda h: h.is_chunking)

        r2_out_before = len(r2.req.output_ids)
        saw_chunked_r1 = False
        saw_r2_decode_during_chunking = False
        for _ in range(DEFAULT_MAX_STEPS):
            comp = t.batch_composition()
            if r1.rid in comp.get("chunked", []):
                saw_chunked_r1 = True
                if r2.rid in comp.get("decode", []) + comp.get("prefill", []):
                    saw_r2_decode_during_chunking = True
                chunked_set = set(comp.get("chunked", []))
                assert chunked_set.isdisjoint(set(comp.get("prefill", [])))
                assert chunked_set.isdisjoint(set(comp.get("decode", [])))
            if not r1.is_chunking:
                break
            yield
        assert saw_chunked_r1, "r1 was never observed in the chunked subset"
        r2_out_after = len(r2.req.output_ids) if r2.req is not None else 64
        assert r2_out_after > r2_out_before, (
            f"r2's decode must progress while r1 chunks; output_ids stayed at "
            f"{r2_out_before}"
        )
        assert saw_r2_decode_during_chunking, (
            "r2 was never observed co-batched (decode/mixed) while r1 held "
            "the chunked slot"
        )

        t.abort(r2)
        yield from run_until_finished(r1)
        for _ in range(40):
            if t.is_fully_idle:
                break
            yield
        assert r1.finished


if __name__ == "__main__":
    unittest.main()
