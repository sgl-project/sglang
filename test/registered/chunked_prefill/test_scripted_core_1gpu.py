import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.scripted_runtime.runtime import ScriptedRuntime
from sglang.test.scripted_runtime.testcase import ScriptedRuntimeTestCase
from sglang.test.scripted_runtime_chunked_helpers import (
    DEFAULT_MAX_STEPS,
    base_engine_kwargs,
    run_until,
    run_until_finished,
)

# Single-GPU scripted chunked-prefill core tests. The PP sweep, which
# needs 4 GPUs, lives in test_scripted_core_4gpu.py.
register_cuda_ci(est_time=300, stage="extra-a", runner_config="1-gpu-small")


_CHUNK_SIZE = 64
_PROMPT_LEN = 256


class TestScriptedCore(ScriptedRuntimeTestCase):
    ENGINE_KWARGS = base_engine_kwargs(chunked_prefill_size=_CHUNK_SIZE)

    def test_chunked_prefill_smoke(self):
        """Engine boots with small chunk_size and a multi-chunk req finishes cleanly."""
        self.runtime.run(self._script_chunked_prefill_smoke)

    @staticmethod
    def _script_chunked_prefill_smoke(t: ScriptedRuntime):
        r = t.start_req(prompt_len=_PROMPT_LEN, max_new_tokens=3)
        yield from run_until_finished(r)
        assert r.finished, f"req did not finish, status={r.status!r}"

    def test_chunked_req_scheduled_last_iter_observed_true_then_false(self):
        """While chunking, the scheduler flag flips True at least once; after finish it clears to False."""
        self.runtime.run(
            self._script_chunked_req_scheduled_last_iter_observed_true_then_false
        )

    @staticmethod
    def _script_chunked_req_scheduled_last_iter_observed_true_then_false(
        t: ScriptedRuntime,
    ):
        r = t.start_req(prompt_len=2 * _CHUNK_SIZE, max_new_tokens=2)
        saw_true = False
        for _ in range(DEFAULT_MAX_STEPS):
            if t.last_chunked_req_scheduled_iter_flag():
                saw_true = True
            if r.finished:
                break
            yield
        assert r.finished, f"req did not finish, status={r.status!r}"
        assert saw_true, "scheduler flag should be True at least once mid-chunk"
        assert t.last_chunked_req_scheduled_iter_flag() is False, (
            f"flag must clear to False after last chunk; "
            f"got {t.last_chunked_req_scheduled_iter_flag()!r}"
        )

    def test_pause_generation_retract_clears_chunked_req(self):
        """Mid-chunk pause_generation(retract) drops the req back to waiting and clears the chunked slot."""
        self.runtime.run(self._script_pause_generation_retract_clears_chunked_req)

    @staticmethod
    def _script_pause_generation_retract_clears_chunked_req(t: ScriptedRuntime):
        r = t.start_req(prompt_len=4 * _CHUNK_SIZE, max_new_tokens=2)
        yield from run_until(r, lambda h: h.is_chunking)

        t.pause_generation(mode="retract")
        # Retract state is reflected in the scheduler on the next event-loop iter.
        yield

        assert r.status == "waiting", (
            f"after pause(retract) the req should be back in waiting_queue; "
            f"got status={r.status!r}"
        )
        assert t.last_chunked_req_scheduled_iter_flag() is False, (
            f"pause(retract) must clear chunked_req; "
            f"flag={t.last_chunked_req_scheduled_iter_flag()!r}"
        )

        t.continue_generation()
        yield from run_until_finished(r)
        assert r.finished, f"req did not resume to finished, status={r.status!r}"

    def test_abort_all_during_chunked_prefill_clears_chunked_req(self):
        """Mid-chunk abort_all() terminates the req; scheduler clears the chunked slot within a few yields."""
        self.runtime.run(
            self._script_abort_all_during_chunked_prefill_clears_chunked_req
        )

    @staticmethod
    def _script_abort_all_during_chunked_prefill_clears_chunked_req(t: ScriptedRuntime):
        r = t.start_req(prompt_len=4 * _CHUNK_SIZE, max_new_tokens=2)
        yield from run_until(r, lambda h: h.is_chunking)

        t.abort_all()
        # abort_request marks the req FINISH_ABORT but the chunked_req slot
        # is only cleared on the next normal-cleanup iter; give a few yields.
        for _ in range(8):
            yield
            if r.finished:
                break

        assert r.finished, f"req did not finish after abort_all, status={r.status!r}"
        assert t.last_chunked_req_scheduled_iter_flag() is False, (
            f"chunked slot must be cleared after abort; "
            f"flag={t.last_chunked_req_scheduled_iter_flag()!r}"
        )

    def test_chunked_req_prefill_only_finishes(self):
        """max_new_tokens=0 with chunked prefill: req finishes after the last chunk, no decode."""
        self.runtime.run(self._script_chunked_req_prefill_only_finishes)

    @staticmethod
    def _script_chunked_req_prefill_only_finishes(t: ScriptedRuntime):
        r = t.start_req(prompt_len=4 * _CHUNK_SIZE, max_new_tokens=0)
        yield from run_until_finished(r)
        assert r.finished, f"prefill-only req did not finish, status={r.status!r}"
        assert t.last_chunked_req_scheduled_iter_flag() is False, (
            f"chunked slot must be cleared after prefill-only finish; "
            f"flag={t.last_chunked_req_scheduled_iter_flag()!r}"
        )

    def test_chunked_prefill_does_not_inflate_radix_hit_count(self):
        """Chunked inserts skip hit_count, so the whole-tree total equals the node count, not the inflated per-chunk sum."""
        self.runtime.run(self._script_chunked_prefill_does_not_inflate_radix_hit_count)

    @staticmethod
    def _script_chunked_prefill_does_not_inflate_radix_hit_count(t: ScriptedRuntime):
        # Reset first: the shared engine and the all-ones token prefix mean
        # earlier scripts leave a radix chain whose hit counts would otherwise
        # make this total history-dependent.
        t.reset_radix_cache()
        before = t.get_all_node_hit_counts()
        assert before == 0, f"reset must empty the radix tree; got {before}"

        r = t.start_req(prompt_len=_PROMPT_LEN, max_new_tokens=2)
        yield from run_until_finished(r)
        assert r.finished

        # _PROMPT_LEN / _CHUNK_SIZE = 4 chunks -> 3 chunked cache_unfinished_req
        # inserts (all hit_count-skipped) build prefix nodes [0:64], [64:128],
        # [128:192]; the single non-chunked cache_finished_req insert then bumps
        # those three plus the [192:] tail node exactly once each. Total == 4.
        # Without the guard the first node alone would reach 4 (one bump per
        # chunked insert + the finish insert) and the total would be 10.
        after = t.get_all_node_hit_counts()
        assert after - before == 4, (
            f"chunked prefill must bump each radix node exactly once; "
            f"expected total 4, got {after - before}"
        )


if __name__ == "__main__":
    unittest.main()
