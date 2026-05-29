import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.scripted_runtime.runtime import ScriptedRuntime
from sglang.test.scripted_runtime.test_case import ScriptedRuntimeTestCase
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

    _PP_LOOP_SIZE = ENGINE_KWARGS["pp_size"] + ENGINE_KWARGS["pp_async_batch_depth"]
    assert _PP_LOOP_SIZE == 6

    _NUM_CHUNKS_VALUES = (2, 4, 6, 8)
    _NUM_CONC_REQS_VALUES = (1, 2, 4)

    def test_pp_chunk_sweep(self):
        """Every (num_chunks, num_conc_reqs) combo under PP=4 drains to finished."""
        for num_chunks in self._NUM_CHUNKS_VALUES:
            for num_conc_reqs in self._NUM_CONC_REQS_VALUES:
                with self.subTest(num_chunks=num_chunks, num_conc_reqs=num_conc_reqs):
                    self.runtime.run(
                        self._script_pp_one_combo,
                        args=(num_chunks, num_conc_reqs),
                    )

    @staticmethod
    def _script_pp_one_combo(t: ScriptedRuntime, num_chunks: int, num_conc_reqs: int):
        # A few tokens short of a clean chunk multiple so the last chunk is
        # partial — exercises the off-by-one path under PP.
        prompt_len = num_chunks * _CHUNK_SIZE - 3
        reqs = [
            t.start_req(prompt_len=prompt_len, max_new_tokens=2)
            for _ in range(num_conc_reqs)
        ]
        yield from run_until_all_finished(reqs, max_steps=DEFAULT_MAX_STEPS)
        for r in reqs:
            assert r.finished, (
                f"combo num_chunks={num_chunks}, num_conc_reqs={num_conc_reqs}: "
                f"req {r.rid!r} did not finish"
            )


if __name__ == "__main__":
    unittest.main()
