import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.scripted_runtime.runtime import ScriptedRuntime
from sglang.test.scripted_runtime.testcase import ScriptedRuntimeTestCase
from sglang.test.scripted_runtime_chunked_helpers import (
    DEFAULT_MAX_STEPS,
    SMALL_MODEL,
    base_engine_kwargs,
    run_until_all_finished,
)

# PP scripted chunked-prefill sweep. PP needs 4 GPUs, so this runs on the
# 4-gpu-h100 runner. The single-GPU core tests live in
# test_scripted_core_1gpu.py.
register_cuda_ci(est_time=900, stage="extra-b", runner_config="4-gpu-h100")


_CHUNK_SIZE = 64


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


if __name__ == "__main__":
    unittest.main()
