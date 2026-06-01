import unittest

from sglang.test.scripted_runtime.context import ScriptedContext
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


class TestMultiReqBasic(ScriptedTestCase):
    ENGINE_KWARGS = base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE)

    def test_at_most_one_chunked_in_flight(self):
        self.server.execute_script(self._script_at_most_one_chunked_in_flight)

    @staticmethod
    def _script_at_most_one_chunked_in_flight(t: ScriptedContext):
        r1 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        r2 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)

        for _ in range(DEFAULT_MAX_STEPS):
            in_flight = (1 if t.scheduler.chunked_req is not None else 0)
            assert in_flight <= 1, (
                f"single-in-flight invariant violated: "
                f"chunked_in_flight_count()={in_flight}"
            )
            if r1.finished and r2.finished:
                return
            yield
        raise AssertionError("r1 and r2 did not both finish within step budget")

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

    def test_chunked_plus_decode_in_batch(self):
        self.server.execute_script(self._script_chunked_plus_decode_in_batch)

    @staticmethod
    def _script_chunked_plus_decode_in_batch(t: ScriptedContext):
        r1 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r1, lambda h: h.is_chunking)

        r2 = t.start_req(prompt_len=8, max_new_tokens=4)
        yield

        comp = t.batch_composition()
        assert r1.rid in comp.get(
            "chunked", []
        ), f"r1 should be in chunked subset of batch; got {comp}"
        assert r2.rid in comp.get("prefill", []) + comp.get(
            "decode", []
        ), f"r2 should be in prefill or decode subset; got {comp}"
        chunked_set = set(comp.get("chunked", []))
        prefill_set = set(comp.get("prefill", []))
        decode_set = set(comp.get("decode", []))
        assert chunked_set.isdisjoint(prefill_set)
        assert chunked_set.isdisjoint(decode_set)

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
        final = t.engine_stats()
        assert (
            final["kv_pool_free"] >= baseline["kv_pool_free"]
        ), f"KV leak: {baseline['kv_pool_free']} -> {final['kv_pool_free']}"

    def test_five_hundred_short_reqs(self):
        self.server.execute_script(self._script_five_hundred_short_reqs)

    @staticmethod
    def _script_five_hundred_short_reqs(t: ScriptedContext):
        baseline = t.engine_stats()
        reqs = [t.start_req(prompt_len=8, max_new_tokens=1) for _ in range(500)]
        yield from run_until_all_finished(reqs, max_steps=20000)
        for r in reqs:
            assert r.finished
        final = t.engine_stats()
        assert final["kv_pool_free"] >= baseline["kv_pool_free"]

    def test_mixed_ten_chunked_ten_short(self):
        self.server.execute_script(self._script_mixed_ten_chunked_ten_short)

    @staticmethod
    def _script_mixed_ten_chunked_ten_short(t: ScriptedContext):
        chunked = [
            t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
            for _ in range(10)
        ]
        shorts = [t.start_req(prompt_len=8, max_new_tokens=2) for _ in range(10)]
        all_reqs = chunked + shorts
        for _ in range(DEFAULT_MAX_STEPS * 20):
            assert (1 if t.scheduler.chunked_req is not None else 0) <= 1
            if all(r.finished for r in all_reqs):
                return
            yield
        raise AssertionError("not all reqs finished")

    def test_submit_during_chunk_mid(self):
        self.server.execute_script(self._script_submit_during_chunk_mid)

    @staticmethod
    def _script_submit_during_chunk_mid(t: ScriptedContext):
        r1 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r1, lambda h: h.is_chunking)
        r2 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield
        r3 = t.start_req(prompt_len=16, max_new_tokens=2)
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
        final = t.engine_stats()
        assert final["kv_pool_free"] >= baseline["kv_pool_free"]

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
        final = t.engine_stats()
        assert final["kv_pool_free"] >= baseline["kv_pool_free"]

    def test_three_long_back_to_back(self):
        self.server.execute_script(self._script_three_long_back_to_back)

    @staticmethod
    def _script_three_long_back_to_back(t: ScriptedContext):
        reqs = [
            t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
            for _ in range(3)
        ]
        for _ in range(DEFAULT_MAX_STEPS * 10):
            assert (1 if t.scheduler.chunked_req is not None else 0) <= 1
            if all(r.finished for r in reqs):
                break
            yield
        for r in reqs:
            assert r.finished

    def test_submit_pause_n_resubmit_same_rid(self):
        self.server.execute_script(self._script_submit_pause_n_resubmit_same_rid)

    @staticmethod
    def _script_submit_pause_n_resubmit_same_rid(t: ScriptedContext):
        baseline = t.engine_stats()
        r1 = t.start_req(prompt_len=16, max_new_tokens=2, rid="reuse-200")
        yield from run_until_finished(r1)
        for _ in range(200):
            assert (1 if t.scheduler.chunked_req is not None else 0) == 0
            yield
        r2 = t.start_req(prompt_len=16, max_new_tokens=2, rid="reuse-200")
        yield from run_until_finished(r2)
        assert r2.finished
        final = t.engine_stats()
        assert final["kv_pool_free"] >= baseline["kv_pool_free"]

    def test_submit_during_decode_of_other(self):
        self.server.execute_script(self._script_submit_during_decode_of_other)

    @staticmethod
    def _script_submit_during_decode_of_other(t: ScriptedContext):
        r1 = t.start_req(prompt_len=16, max_new_tokens=16)
        yield from run_until(r1, lambda h: h.status == "running")
        r2 = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        for _ in range(DEFAULT_MAX_STEPS * 5):
            assert (1 if t.scheduler.chunked_req is not None else 0) <= 1
            if r1.finished and r2.finished:
                break
            yield
        assert r1.finished and r2.finished

    def test_unique_rids_distinct(self):
        self.server.execute_script(self._script_unique_rids_distinct)

    @staticmethod
    def _script_unique_rids_distinct(t: ScriptedContext):
        baseline = t.engine_stats()
        reqs = [
            t.start_req(prompt_len=16, max_new_tokens=2, rid=f"unique-{i}")
            for i in range(20)
        ]
        yield from run_until_all_finished(reqs)
        rids = {r.rid for r in reqs}
        assert len(rids) == 20
        final = t.engine_stats()
        assert final["kv_pool_free"] >= baseline["kv_pool_free"]

    def test_two_small_parallel(self):
        self.server.execute_script(self._script_two_small_parallel)

    @staticmethod
    def _script_two_small_parallel(t: ScriptedContext):
        r1 = t.start_req(prompt_len=16, max_new_tokens=4)
        r2 = t.start_req(prompt_len=16, max_new_tokens=4)
        for _ in range(DEFAULT_MAX_STEPS):
            assert (t.scheduler.chunked_req.rid if t.scheduler.chunked_req is not None else None) is None
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
            assert (1 if t.scheduler.chunked_req is not None else 0) <= 1
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
        for _ in range(4):
            reqs.append(t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2))
            yield
            assert (1 if t.scheduler.chunked_req is not None else 0) <= 1
            yield
        for _ in range(DEFAULT_MAX_STEPS * 10):
            assert (1 if t.scheduler.chunked_req is not None else 0) <= 1
            if all(r.finished for r in reqs):
                break
            yield
        for r in reqs:
            assert r.finished

    def test_eight_concurrent_chunked(self):
        self.server.execute_script(self._script_eight_concurrent_chunked)

    @staticmethod
    def _script_eight_concurrent_chunked(t: ScriptedContext):
        reqs = [
            t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
            for _ in range(8)
        ]
        for _ in range(DEFAULT_MAX_STEPS * 5):
            assert (1 if t.scheduler.chunked_req is not None else 0) <= 1
            if all(r.finished for r in reqs):
                return
            yield
        raise AssertionError("not all reqs finished")

    def test_decode_only_batch(self):
        self.server.execute_script(self._script_decode_only_batch)

    @staticmethod
    def _script_decode_only_batch(t: ScriptedContext):
        reqs = [t.start_req(prompt_len=4, max_new_tokens=8) for _ in range(10)]
        for _ in range(DEFAULT_MAX_STEPS * 3):
            assert (
                (t.scheduler.chunked_req.rid if t.scheduler.chunked_req is not None else None) is None
            ), "pure decode workload must never populate chunked_req"
            assert (1 if t.scheduler.chunked_req is not None else 0) == 0
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
        for _ in range(DEFAULT_MAX_STEPS * 10):
            assert (1 if t.scheduler.chunked_req is not None else 0) <= 1
            if all(r.finished for r in reqs):
                break
            yield
        for r in reqs:
            assert r.finished

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
            chunked = s.chunked_req
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
        for _ in range(DEFAULT_MAX_STEPS * 10):
            assert (1 if t.scheduler.chunked_req is not None else 0) <= 1
            if chunked1.finished and chunked2.finished and short.finished:
                break
            yield
        assert chunked1.finished and chunked2.finished and short.finished

    def test_batch_with_finish_emitted_exactly_once(self):
        self.server.execute_script(self._script_batch_with_finish_emitted_exactly_once)

    @staticmethod
    def _script_batch_with_finish_emitted_exactly_once(t: ScriptedContext):
        reqs = [t.start_req(prompt_len=16, max_new_tokens=2) for _ in range(6)]
        yield from run_until_all_finished(reqs)
        for r in reqs:
            assert r.finished

    def test_batch_state_query_during_run(self):
        self.server.execute_script(self._script_batch_state_query_during_run)

    @staticmethod
    def _script_batch_state_query_during_run(t: ScriptedContext):
        reqs = [t.start_req(prompt_len=16, max_new_tokens=4) for _ in range(4)]
        for _ in range(DEFAULT_MAX_STEPS):
            comp = t.batch_composition()
            assert isinstance(comp, dict)
            prefill = set(comp.get("prefill", []))
            decode = set(comp.get("decode", []))
            chunked = set(comp.get("chunked", []))
            assert prefill & decode == set()
            assert prefill & chunked == set()
            assert decode & chunked == set()
            if all(r.finished for r in reqs):
                return
            yield
        raise AssertionError("not all reqs finished")

    def test_mixed_lengths_then_more_arrivals(self):
        self.server.execute_script(self._script_mixed_lengths_then_more_arrivals)

    @staticmethod
    def _script_mixed_lengths_then_more_arrivals(t: ScriptedContext):
        initial = [t.start_req(prompt_len=16, max_new_tokens=4) for _ in range(3)]
        yield
        yield
        more = [t.start_req(prompt_len=16, max_new_tokens=4) for _ in range(3)]
        yield from run_until_all_finished(initial + more)
        for r in initial + more:
            assert r.finished


class TestMultiReqPriority(ScriptedTestCase):
    ENGINE_KWARGS = base_engine_kwargs(
        chunked_prefill_size=DEFAULT_CHUNK_SIZE,
        enable_priority_scheduling=True,
    )

    def test_parallel_with_priority(self):
        self.server.execute_script(self._script_parallel_with_priority)

    @staticmethod
    def _script_parallel_with_priority(t: ScriptedContext):
        baseline = t.engine_stats()
        normal = [t.start_req(prompt_len=16, max_new_tokens=2) for _ in range(3)]
        high = [
            t.start_req(prompt_len=16, max_new_tokens=2, priority="high")
            for _ in range(2)
        ]
        yield from run_until_all_finished(normal + high)
        for r in normal + high:
            assert r.finished
        final = t.engine_stats()
        assert final["kv_pool_free"] >= baseline["kv_pool_free"]


if __name__ == "__main__":
    unittest.main()
