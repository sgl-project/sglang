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

# Single-file registration covers both the 1-GPU classes below and the
# PP class in this same file; PP needs 4 GPUs so the whole file runs on
# the 4-gpu-h100 runner / stage-c slot.
register_cuda_ci(est_time=900, stage="base-c", runner_config="4-gpu-h100")


_CHUNK_SIZE = 64
_PROMPT_LEN = 256


class TestScriptedCoreSmoke(ScriptedRuntimeTestCase):
    ENGINE_KWARGS = base_engine_kwargs(chunked_prefill_size=_CHUNK_SIZE)

    def test_chunked_prefill_smoke(self):
        """Engine boots with small chunk_size and a multi-chunk req finishes cleanly."""
        self.runtime.run(self._script_chunked_prefill_smoke)

    @staticmethod
    def _script_chunked_prefill_smoke(t: ScriptedRuntime):
        r = t.start_req(prompt_len=_PROMPT_LEN, max_new_tokens=3)
        yield from run_until_finished(r)
        assert r.finished, f"req did not finish, status={r.status!r}"


class TestScriptedCoreFlag(ScriptedRuntimeTestCase):
    ENGINE_KWARGS = base_engine_kwargs(chunked_prefill_size=_CHUNK_SIZE)

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


class TestScriptedCorePauseRetract(ScriptedRuntimeTestCase):
    ENGINE_KWARGS = base_engine_kwargs(chunked_prefill_size=_CHUNK_SIZE)

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


class TestScriptedCoreAbortAll(ScriptedRuntimeTestCase):
    ENGINE_KWARGS = base_engine_kwargs(chunked_prefill_size=_CHUNK_SIZE)

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


class TestScriptedCorePrefillOnly(ScriptedRuntimeTestCase):
    ENGINE_KWARGS = base_engine_kwargs(chunked_prefill_size=_CHUNK_SIZE)

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


class TestScriptedCoreRadixHitCountSkip(ScriptedRuntimeTestCase):
    ENGINE_KWARGS = base_engine_kwargs(
        chunked_prefill_size=_CHUNK_SIZE,
        disable_radix_cache=False,
    )

    def test_chunked_req_skips_radix_hit_count_increment(self):
        """A chunked prefill triggers at least one _inc_hit_count(chunked=True) skip on radix insert."""
        self.runtime.run(self._script_chunked_req_skips_radix_hit_count_increment)

    @staticmethod
    def _script_chunked_req_skips_radix_hit_count_increment(t: ScriptedRuntime):
        before = t.radix_chunked_hit_count_skip_count()
        r = t.start_req(prompt_len=4 * _CHUNK_SIZE, max_new_tokens=2)
        yield from run_until_finished(r)
        after = t.radix_chunked_hit_count_skip_count()
        assert r.finished, f"req did not finish, status={r.status!r}"
        assert after - before >= 1, (
            f"chunked _inc_hit_count skip branch must be exercised at least "
            f"once; before={before}, after={after}"
        )


if __name__ == "__main__":
    unittest.main()
