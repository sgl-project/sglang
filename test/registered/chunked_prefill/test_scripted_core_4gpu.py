import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.scripted_runtime.context import ScriptedContext
from sglang.test.scripted_runtime.test_case import ScriptedTestCase
from sglang.test.scripted_runtime_chunked_helpers import (
    DEFAULT_MAX_STEPS,
    SMALL_MODEL,
    base_engine_kwargs,
    run_until_all_finished,
    run_until_finished,
)

register_cuda_ci(est_time=115, stage="extra-b", runner_config="4-gpu-h100")


_CHUNK_SIZE = 64


class TestScriptedPpChunkSweep(ScriptedTestCase):
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
        for num_chunks in self._NUM_CHUNKS_VALUES:
            for num_conc_reqs in self._NUM_CONC_REQS_VALUES:
                with self.subTest(num_chunks=num_chunks, num_conc_reqs=num_conc_reqs):
                    self.server.execute_script(
                        self._script_pp_one_combo,
                        args=(num_chunks, num_conc_reqs),
                    )

    @staticmethod
    def _script_pp_one_combo(t: ScriptedContext, num_chunks: int, num_conc_reqs: int):
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

    def test_pp_flush_cache_during_inflight_chunk_results(self):
        """flush_cache landing after the last chunk dispatch but before its batch result is processed must not corrupt the radix tree."""
        self.server.execute_script(self._script_flush_during_inflight_chunk_results)

    @staticmethod
    def _script_flush_during_inflight_chunk_results(t: ScriptedContext):
        scheduler = t.scheduler
        r = t.start_req(prompt_len=2 * _CHUNK_SIZE - 3, max_new_tokens=2)

        # Wait for the window where the queues and the current microbatch slot
        # bindings are all clear, yet dispatched chunk batch results are still
        # in flight in the PP pipeline, then post flush_cache into that window.
        flushed = False
        for _ in range(DEFAULT_MAX_STEPS):
            in_flight = any(
                mb is not None and not mb.is_empty() for mb in scheduler.mbs
            )
            queues_clear = (
                scheduler.chunked_req is None
                and len(scheduler.waiting_queue) == 0
                and all(x.is_empty() for x in scheduler.running_mbs)
                and (scheduler.cur_batch is None or scheduler.cur_batch.is_empty())
                and (scheduler.last_batch is None or scheduler.last_batch.is_empty())
            )
            if in_flight and queues_clear:
                t.flush_cache()
                flushed = True
                break
            yield
        assert flushed, "never observed in-flight chunk results with clear queues"

        yield from run_until_finished(r, max_steps=DEFAULT_MAX_STEPS)
        assert r.finished, f"req {r.rid!r} did not finish after flush_cache"


if __name__ == "__main__":
    unittest.main()
