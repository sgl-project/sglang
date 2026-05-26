import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.scripted_runtime.runtime import ScriptedRuntime
from sglang.test.scripted_runtime.testcase import ScriptedRuntimeTestCase
from sglang.test.scripted_runtime_chunked_helpers import (
    DEFAULT_MAX_STEPS,
    SMALL_MODEL,
    base_engine_kwargs,
    run_until,
    run_until_all_finished,
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


class TestScriptedPpChunkSweep(ScriptedRuntimeTestCase):
    # pp_async_batch_depth=2 makes pp_loop_size = pp_size + depth = 6,
    # so the loop size strictly exceeds pp_size — the regime where the
    # PP queue depth is larger than the GPU count.
    ENGINE_KWARGS = base_engine_kwargs(
        model_path=SMALL_MODEL,
        tp_size=1,
        dp_size=1,
        pp_size=4,
        pp_async_batch_depth=2,
        chunked_prefill_size=_CHUNK_SIZE,
    )

    _PP_LOOP_SIZE = 6
    _NUM_CHUNKS_VALUES = (2, 4, 6, 8)
    _NUM_CONC_REQS_VALUES = (1, 2, 4)

    def test_pp_chunk_sweep(self):
        """Sweep num_chunks x num_conc_reqs under PP=4; every combo reaches the expected concurrency latch and drains clean."""
        self.runtime.run(self._script_pp_chunk_sweep)

    @staticmethod
    def _script_pp_chunk_sweep(t: ScriptedRuntime):
        for num_chunks in TestScriptedPpChunkSweep._NUM_CHUNKS_VALUES:
            for num_conc_reqs in TestScriptedPpChunkSweep._NUM_CONC_REQS_VALUES:
                yield from TestScriptedPpChunkSweep._script_one_combo(
                    t, num_chunks=num_chunks, num_conc_reqs=num_conc_reqs
                )

    @staticmethod
    def _script_one_combo(t: ScriptedRuntime, *, num_chunks: int, num_conc_reqs: int):
        prompt_len = num_chunks * _CHUNK_SIZE
        reqs = [
            t.start_req(prompt_len=prompt_len, max_new_tokens=2)
            for _ in range(num_conc_reqs)
        ]

        # Per-req latch: running_batch is per-microbatch under PP, so a
        # plain "sum status==running" each yield systematically undercounts.
        # Latch over >= pp_loop_size yields to catch each req on the
        # microbatch where it actually appears.
        ever_running = {r.rid: False for r in reqs}
        pp_loop_size = TestScriptedPpChunkSweep._PP_LOOP_SIZE
        for _ in range(pp_loop_size):
            for r in reqs:
                if r.status == "running":
                    ever_running[r.rid] = True
            yield

        yield from run_until_all_finished(reqs, max_steps=DEFAULT_MAX_STEPS)

        # Final concurrency latch pass to absorb any req that only
        # appeared in a microbatch after the initial spin.
        for r in reqs:
            if r.status == "running":
                ever_running[r.rid] = True

        num_ever_running = sum(ever_running.values())
        expected_concurrency = min(num_conc_reqs, pp_loop_size)
        assert num_ever_running >= expected_concurrency, (
            f"combo num_chunks={num_chunks}, num_conc_reqs={num_conc_reqs}: "
            f"expected at least {expected_concurrency} reqs to be observed "
            f"running across the pp_loop_size={pp_loop_size} latch window; "
            f"got num_ever_running={num_ever_running}"
        )
        for r in reqs:
            assert r.finished, (
                f"combo num_chunks={num_chunks}, num_conc_reqs={num_conc_reqs}: "
                f"req {r.rid!r} did not finish, status={r.status!r}"
            )

        # Between-combo cleanup: flag only reflects current microbatch's
        # chunked slot, so spin >= pp_loop_size to verify no other mb
        # still holds a residual chunked_req.
        for _ in range(pp_loop_size):
            yield
        assert t.last_chunked_req_scheduled_iter_flag() is False, (
            f"combo num_chunks={num_chunks}, num_conc_reqs={num_conc_reqs}: "
            f"chunked slot must be empty after drain; "
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
