import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.scripted_runtime.runtime import ScriptedRuntime
from sglang.test.scripted_runtime.testcase import ScriptedRuntimeTestCase
from sglang.test.scripted_runtime_chunked_helpers import (
    DEFAULT_CHUNK_SIZE,
    VERY_LONG_PROMPT_LEN,
    base_engine_kwargs,
    run_until,
)

# Single-GPU scripted chunked-prefill test for the hybrid-SWA
# add_chunked_req early-return gate. gpt-oss-20b is a hybrid-SWA model
# and needs a single 80 GB GPU, so this runs on 1-gpu-large rather than
# the 1-gpu-small runner the rest of the core suite uses.
register_cuda_ci(est_time=400, stage="extra-a", runner_config="1-gpu-large")


_SWA_MODEL = "openai/gpt-oss-20b"


class TestScriptedSwaChunkedReqScheduledLastIter(ScriptedRuntimeTestCase):
    ENGINE_KWARGS = base_engine_kwargs(
        model_path=_SWA_MODEL,
        chunked_prefill_size=DEFAULT_CHUNK_SIZE,
        mem_fraction_static=0.70,
        disable_piecewise_cuda_graph=True,
    )

    def test_swa_chunked_req_early_return_no_double_free(self):
        """SWA add_chunked_req early-return must flip _chunked_req_scheduled_last_iter cleanly."""
        self.runtime.run(self._script_swa_chunked_req_early_return_no_double_free)

    # SWA pool critical + add_chunked_req forced early-return —
    # _chunked_req_scheduled_last_iter must flip correctly; stash must not
    # be double-freed.
    @staticmethod
    def _script_swa_chunked_req_early_return_no_double_free(t: ScriptedRuntime):
        # Long competitor consumes SWA budget so the chunked req hits
        # the add_chunked_req SWA-early-return branch at least once.
        competitor = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=8)
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
        yield from run_until(r, lambda h: h.is_chunking)
        # Snapshot counters across iters: once the req leaves the
        # scheduler's structures the rid lookup returns 0, so we must
        # capture peaks during the chunked-active window.
        observed_true = False
        observed_false_after_early_return = False
        peak_early_return_count = 0
        peak_double_free_count = 0
        for _ in range(800):
            flag = r.chunked_req_scheduled_last_iter
            er = r.swa_chunked_early_return_count
            df = r.swa_stash_double_free_count
            peak_early_return_count = max(peak_early_return_count, er)
            peak_double_free_count = max(peak_double_free_count, df)
            if flag is True:
                observed_true = True
            if flag is False and er > 0:
                observed_false_after_early_return = True
            if r.finished and competitor.finished:
                break
            yield
        assert r.finished
        assert (
            observed_true
        ), "_chunked_req_scheduled_last_iter must be True at least once"
        assert peak_early_return_count > 0, (
            "test must exercise the SWA early-return branch; "
            "peak swa_chunked_early_return_count stayed 0 across the run"
        )
        assert observed_false_after_early_return, (
            "_chunked_req_scheduled_last_iter must observe False after a "
            "SWA early-return (gate must close on this iter)"
        )
        assert peak_double_free_count == 0, (
            f"SWA stash double-free observed {peak_double_free_count} times "
            "(gate at scheduler.get_next_batch_to_run is broken)"
        )


if __name__ == "__main__":
    unittest.main()
