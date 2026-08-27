import unittest

from sglang.srt.managers.schedule_batch import FINISH_ABORT
from sglang.test.scripted_runtime.context import ScriptedContext
from sglang.test.scripted_runtime.test_case import ScriptedTestCase
from sglang.test.scripted_runtime_chunked_helpers import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_MAX_STEPS,
    VERY_LONG_PROMPT_LEN,
    base_engine_kwargs,
    run_until,
    run_until_finished,
)


def _drain_until_released(t, *handles):
    for _ in range(12):
        if all(
            h.kv_pages == 0
            and h.lock_refs == 0
            and (h.req is None or h.req.req_pool_idx is None)
            for h in handles
        ):
            return
        yield


class TestLifecycleBasic(ScriptedTestCase):
    ENGINE_KWARGS = base_engine_kwargs(chunked_prefill_size=DEFAULT_CHUNK_SIZE)

    def test_small_prompt_short_decode(self):
        self.server.execute_script(self._script_small_prompt_short_decode)

    @staticmethod
    def _script_small_prompt_short_decode(t: ScriptedContext):
        r = t.start_req(prompt_len=8, max_new_tokens=2, ignore_eos=True)
        yield from run_until_finished(r)
        assert r.finished
        assert r.chunks_done == 0
        assert len(r.req.output_ids) == 2

    def test_medium_prompt_medium_decode(self):
        self.server.execute_script(self._script_medium_prompt_medium_decode)

    @staticmethod
    def _script_medium_prompt_medium_decode(t: ScriptedContext):
        r = t.start_req(
            prompt_len=DEFAULT_CHUNK_SIZE, max_new_tokens=16, ignore_eos=True
        )
        yield from run_until_finished(r)
        assert r.finished
        assert r.chunks_done == 0
        assert len(r.req.output_ids) == 16

    def test_long_prompt_short_decode(self):
        self.server.execute_script(self._script_long_prompt_short_decode)

    @staticmethod
    def _script_long_prompt_short_decode(t: ScriptedContext):
        r = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2, ignore_eos=True
        )
        yield from run_until_finished(r)
        assert r.finished
        assert r.chunks_done == 8
        assert len(r.req.output_ids) == 2

    def test_long_prompt_long_decode(self):
        self.server.execute_script(self._script_long_prompt_long_decode)

    @staticmethod
    def _script_long_prompt_long_decode(t: ScriptedContext):
        r = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=64, ignore_eos=True
        )
        yield from run_until(r, lambda h: h.finished, max_steps=1000)
        assert r.finished
        assert r.chunks_done == 8
        assert len(r.req.output_ids) == 64

    def test_tiny_prompt_long_decode(self):
        self.server.execute_script(self._script_tiny_prompt_long_decode)

    @staticmethod
    def _script_tiny_prompt_long_decode(t: ScriptedContext):
        r = t.start_req(prompt_len=1, max_new_tokens=64, ignore_eos=True)
        yield from run_until(r, lambda h: h.finished, max_steps=500)
        assert r.finished
        assert r.chunks_done == 0
        assert len(r.req.output_ids) == 64

    def test_chunk_size_minus_one_prompt(self):
        self.server.execute_script(self._script_chunk_size_minus_one_prompt)

    @staticmethod
    def _script_chunk_size_minus_one_prompt(t: ScriptedContext):
        r = t.start_req(
            prompt_len=DEFAULT_CHUNK_SIZE - 1, max_new_tokens=4, ignore_eos=True
        )
        yield from run_until_finished(r)
        assert r.finished
        assert r.chunks_done == 0
        assert len(r.req.output_ids) == 4

    def test_chunk_size_plus_two_prompt(self):
        self.server.execute_script(self._script_chunk_size_plus_two_prompt)

    @staticmethod
    def _script_chunk_size_plus_two_prompt(t: ScriptedContext):
        r = t.start_req(
            prompt_len=DEFAULT_CHUNK_SIZE + 2, max_new_tokens=4, ignore_eos=True
        )
        yield from run_until_finished(r)
        assert r.finished
        assert r.chunks_done == 2
        assert len(r.req.output_ids) == 4

    def test_just_over_2x_chunk_size(self):
        self.server.execute_script(self._script_just_over_2x_chunk_size)

    @staticmethod
    def _script_just_over_2x_chunk_size(t: ScriptedContext):
        r = t.start_req(
            prompt_len=2 * DEFAULT_CHUNK_SIZE + 1, max_new_tokens=4, ignore_eos=True
        )
        yield from run_until_finished(r)
        assert r.finished
        assert r.chunks_done == 3
        assert len(r.req.output_ids) == 4

    def test_five_x_chunk_size(self):
        self.server.execute_script(self._script_five_x_chunk_size)

    @staticmethod
    def _script_five_x_chunk_size(t: ScriptedContext):
        r = t.start_req(
            prompt_len=5 * DEFAULT_CHUNK_SIZE, max_new_tokens=4, ignore_eos=True
        )
        yield from run_until_finished(r)
        assert r.finished
        assert r.chunks_done == 5
        assert len(r.req.output_ids) == 4

    def test_ten_x_chunk_size(self):
        self.server.execute_script(self._script_ten_x_chunk_size)

    @staticmethod
    def _script_ten_x_chunk_size(t: ScriptedContext):
        r = t.start_req(
            prompt_len=10 * DEFAULT_CHUNK_SIZE, max_new_tokens=2, ignore_eos=True
        )
        yield from run_until(r, lambda h: h.finished, max_steps=1000)
        assert r.finished
        assert r.chunks_done == 10
        assert len(r.req.output_ids) == 2

    def test_status_progression_happy_path(self):
        self.server.execute_script(self._script_status_progression_happy_path)

    @staticmethod
    def _script_status_progression_happy_path(t: ScriptedContext):
        r = t.start_req(prompt_len=16, max_new_tokens=2, ignore_eos=True)
        seen = []
        for _ in range(DEFAULT_MAX_STEPS):
            seen.append(r.status)
            if r.finished:
                break
            yield
        else:
            raise AssertionError("req did not finish within DEFAULT_MAX_STEPS")
        assert "running" in seen, f"never observed running status; seen={seen}"
        assert seen[-1] == "finished", f"final status must be finished; seen={seen}"
        finished_idx = seen.index("finished")
        assert all(
            s in ("finished",) for s in seen[finished_idx:]
        ), f"status regressed after finish; seen={seen}"

    def test_long_prompt_only_one_decode(self):
        self.server.execute_script(self._script_long_prompt_only_one_decode)

    @staticmethod
    def _script_long_prompt_only_one_decode(t: ScriptedContext):
        r = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=1, ignore_eos=True
        )
        yield from run_until_finished(r)
        assert r.finished
        assert r.chunks_done == 8
        assert len(r.req.output_ids) == 1

    def test_kv_pages_consistent_during_run(self):
        self.server.execute_script(self._script_kv_pages_consistent_during_run)

    @staticmethod
    def _script_kv_pages_consistent_during_run(t: ScriptedContext):
        r = t.start_req(
            prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=4, ignore_eos=True
        )
        saw_positive = False
        for _ in range(DEFAULT_MAX_STEPS):
            pages = r.kv_pages
            if pages > 0:
                saw_positive = True
            elif saw_positive and not r.finished:
                raise AssertionError(
                    f"kv_pages collapsed to 0 mid-run before finish; "
                    f"saw_positive={saw_positive}, status={r.status!r}"
                )
            if r.finished:
                break
            yield
        else:
            raise AssertionError("req did not finish within DEFAULT_MAX_STEPS")
        assert saw_positive
        assert r.kv_pages == 0
        assert len(r.req.output_ids) == 4

    def test_row_idx_recycled_after_finish(self):
        self.server.execute_script(self._script_row_idx_recycled_after_finish)

    @staticmethod
    def _script_row_idx_recycled_after_finish(t: ScriptedContext):
        r = t.start_req(prompt_len=16, max_new_tokens=2, ignore_eos=True)
        yield from run_until_finished(r)
        assert r.finished
        assert r.req.req_pool_idx is None
        assert r.kv_pages == 0
        assert r.lock_refs == 0

    def test_two_seq_clean_handoff(self):
        self.server.execute_script(self._script_two_seq_clean_handoff)

    @staticmethod
    def _script_two_seq_clean_handoff(t: ScriptedContext):
        r1 = t.start_req(prompt_len=16, max_new_tokens=2, ignore_eos=True)
        yield from run_until_finished(r1)
        yield from _drain_until_released(t, r1)
        assert r1.req.req_pool_idx is None and r1.kv_pages == 0 and r1.lock_refs == 0
        r1_output_len = len(r1.req.output_ids)

        r2 = t.start_req(prompt_len=16, max_new_tokens=2, ignore_eos=True)
        yield from run_until_finished(r2)
        yield from _drain_until_released(t, r2)
        assert r1.finished and r2.finished
        assert r1_output_len == 2 and len(r2.req.output_ids) == 2
        assert r2.req.req_pool_idx is None and r2.kv_pages == 0 and r2.lock_refs == 0

    def test_five_seq_clean(self):
        self.server.execute_script(self._script_five_seq_clean)

    @staticmethod
    def _script_five_seq_clean(t: ScriptedContext):
        reqs = []
        for _ in range(5):
            r = t.start_req(prompt_len=16, max_new_tokens=2, ignore_eos=True)
            yield from run_until_finished(r)
            assert r.finished
            assert len(r.req.output_ids) == 2
            assert r.req.req_pool_idx is None
            assert r.kv_pages == 0
            assert r.lock_refs == 0
            reqs.append(r)
        for r in reqs:
            assert r.finished

    def test_radix_partial_seq(self):
        self.server.execute_script(self._script_radix_partial_seq)

    @staticmethod
    def _script_radix_partial_seq(t: ScriptedContext):
        r1 = t.start_req(
            prompt_len=DEFAULT_CHUNK_SIZE, max_new_tokens=1, ignore_eos=True
        )
        yield from run_until_finished(r1)
        r2 = t.start_req(
            prompt_len=DEFAULT_CHUNK_SIZE + 32, max_new_tokens=2, ignore_eos=True
        )
        yield from run_until_finished(r2)
        assert r1.finished and r2.finished
        assert r2.chunks_done == 0
        assert r2.req.cached_tokens > 0, (
            f"r2 must hit r1's radix prefix; got cached_tokens="
            f"{r2.req.cached_tokens}"
        )
        assert len(r2.req.output_ids) == 2

    def test_alternating_short_long_seq(self):
        self.server.execute_script(self._script_alternating_short_long_seq)

    @staticmethod
    def _script_alternating_short_long_seq(t: ScriptedContext):
        for i in range(6):
            prompt = 8 if i % 2 == 0 else VERY_LONG_PROMPT_LEN
            r = t.start_req(
                prompt_len=prompt,
                max_new_tokens=2,
                ignore_eos=True,
                prompt_token=10 + i,
            )
            yield from run_until_finished(r)
            assert r.finished
            assert len(r.req.output_ids) == 2
            assert r.req.req_pool_idx is None and r.kv_pages == 0 and r.lock_refs == 0
            if prompt == VERY_LONG_PROMPT_LEN:
                assert r.chunks_done == 8
            else:
                assert r.chunks_done == 0

    def test_seq_with_growing_prompt(self):
        self.server.execute_script(self._script_seq_with_growing_prompt)

    @staticmethod
    def _script_seq_with_growing_prompt(t: ScriptedContext):
        for idx, L in enumerate([8, 32, 128, 512, 1024]):
            r = t.start_req(
                prompt_len=L, max_new_tokens=1, ignore_eos=True, prompt_token=10 + idx
            )
            yield from run_until_finished(r)
            assert r.finished
            assert len(r.req.output_ids) == 1
            yield from _drain_until_released(t, r)
            assert r.req is None or r.req.req_pool_idx is None
            assert r.kv_pages == 0 and r.lock_refs == 0
            if L > DEFAULT_CHUNK_SIZE:
                assert (
                    r.chunks_done == (L + DEFAULT_CHUNK_SIZE - 1) // DEFAULT_CHUNK_SIZE
                )
            else:
                assert r.chunks_done == 0

    def test_seq_with_shrinking_prompt(self):
        self.server.execute_script(self._script_seq_with_shrinking_prompt)

    @staticmethod
    def _script_seq_with_shrinking_prompt(t: ScriptedContext):
        for idx, L in enumerate([1024, 512, 128, 32, 8]):
            r = t.start_req(
                prompt_len=L, max_new_tokens=1, ignore_eos=True, prompt_token=10 + idx
            )
            yield from run_until_finished(r)
            assert r.finished
            assert len(r.req.output_ids) == 1
            yield from _drain_until_released(t, r)
            assert r.req is None or r.req.req_pool_idx is None
            assert r.kv_pages == 0 and r.lock_refs == 0
            if L > DEFAULT_CHUNK_SIZE:
                assert (
                    r.chunks_done == (L + DEFAULT_CHUNK_SIZE - 1) // DEFAULT_CHUNK_SIZE
                )
            else:
                assert r.chunks_done == 0

    def test_seq_with_idle_yields_between(self):
        self.server.execute_script(self._script_seq_with_idle_yields_between)

    @staticmethod
    def _script_seq_with_idle_yields_between(t: ScriptedContext):
        for _ in range(4):
            r = t.start_req(prompt_len=16, max_new_tokens=2, ignore_eos=True)
            yield from run_until_finished(r)
            assert r.finished
            assert len(r.req.output_ids) == 2
            assert r.req.req_pool_idx is None and r.kv_pages == 0 and r.lock_refs == 0
            for _ in range(20):
                yield

    def test_chunked_then_short_seq(self):
        self.server.execute_script(self._script_chunked_then_short_seq)

    @staticmethod
    def _script_chunked_then_short_seq(t: ScriptedContext):
        seq = [VERY_LONG_PROMPT_LEN, 8, VERY_LONG_PROMPT_LEN, 8]
        for idx, L in enumerate(seq):
            r = t.start_req(
                prompt_len=L, max_new_tokens=2, ignore_eos=True, prompt_token=10 + idx
            )
            yield from run_until_finished(r)
            assert r.finished
            assert len(r.req.output_ids) == 2
            assert r.req.req_pool_idx is None and r.kv_pages == 0 and r.lock_refs == 0
            if L == VERY_LONG_PROMPT_LEN:
                assert r.chunks_done == 8
            else:
                assert r.chunks_done == 0

    def test_seq_engine_stats_stable(self):
        self.server.execute_script(self._script_seq_engine_stats_stable)

    @staticmethod
    def _script_seq_engine_stats_stable(t: ScriptedContext):
        baseline = t.engine_stats()["kv_pool_free"]
        for _ in range(5):
            r = t.start_req(prompt_len=16, max_new_tokens=2, ignore_eos=True)
            yield from run_until_finished(r)
            assert r.finished
            assert len(r.req.output_ids) == 2
            assert r.req.req_pool_idx is None and r.kv_pages == 0 and r.lock_refs == 0
        for _ in range(5):
            yield
        t.flush_cache()
        yield
        final = t.engine_stats()["kv_pool_free"]
        assert (
            final >= baseline - 1
        ), f"KV pool drift: baseline={baseline}, final={final}"

    def test_abort_all_during_chunked(self):
        self.server.execute_script(self._script_abort_all_during_chunked)

    @staticmethod
    def _script_abort_all_during_chunked(t: ScriptedContext):
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=4)
        yield from run_until(r, lambda h: h.is_chunking and h.chunks_done >= 1)
        t.abort_all()

        def _error_message(h):
            if h.req is None:
                return None
            return (
                h.req.finished_reason.message
                if isinstance(h.req.finished_reason, FINISH_ABORT)
                else None
            )

        for _ in range(DEFAULT_MAX_STEPS):
            if r.finished or _error_message(r) is not None:
                break
            yield
        else:
            raise AssertionError(
                "chunked req did not terminate after abort_all within DEFAULT_MAX_STEPS"
            )
        assert r.finished or _error_message(r) is not None
        assert r.kv_pages == 0
        assert r.req is None or r.req.req_pool_idx is None
        assert r.lock_refs == 0


if __name__ == "__main__":
    unittest.main()
