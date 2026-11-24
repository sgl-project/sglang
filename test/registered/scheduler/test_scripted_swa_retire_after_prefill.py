import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.scripted_runtime.context import ScriptedContext
from sglang.test.scripted_runtime.test_case import ScriptedTestCase
from sglang.test.scripted_runtime_chunked_helpers import base_engine_kwargs

register_cuda_ci(est_time=240, stage="extra-a", runner_config="1-gpu-small")


_SWA_MODEL = "openai/gpt-oss-20b"

# One extend covers the whole prompt, so the extend-time SWA eviction has
# nothing to retire and only the post-prefill retire can move the cursor.
_PROMPT_LEN = 2048
_CHUNK_SIZE = 4096
_MAX_NEW_TOKENS = 16
_MAX_STEPS = 400


class TestScriptedSwaRetireAfterPrefill(ScriptedTestCase):
    ENGINE_KWARGS = base_engine_kwargs(
        model_path=_SWA_MODEL,
        chunked_prefill_size=_CHUNK_SIZE,
        disable_radix_cache=True,
        page_size=1,
        mem_fraction_static=0.70,
    )

    def test_out_of_window_rows_retired_when_prefill_completes(self):
        self.server.execute_script(
            self._script_out_of_window_rows_retired_when_prefill_completes
        )

    @staticmethod
    def _script_out_of_window_rows_retired_when_prefill_completes(
        t: ScriptedContext,
    ):
        window = t.scheduler.tree_cache.swa_retain_window
        assert window is not None and window < _PROMPT_LEN // 4, window

        h = t.start_req(
            prompt_len=_PROMPT_LEN,
            max_new_tokens=_MAX_NEW_TOKENS,
            ignore_eos=True,
        )
        # Stop at the first iteration after the prefill result is processed,
        # before the decode path (decode_batch_idx >= 1) retires anything.
        for _ in range(_MAX_STEPS):
            req = h.req
            if req is not None and len(req.output_ids) > 0:
                break
            yield
        else:
            raise AssertionError(f"prefill did not complete in {_MAX_STEPS} steps")

        assert len(req.output_ids) == 1, len(req.output_ids)
        evicted = req.kv.max_evicted_seqlen
        assert evicted >= _PROMPT_LEN - 2 * window, (
            f"SWA cursor at {evicted} right after prefill of {_PROMPT_LEN} tokens "
            f"(window {window}); the out-of-window rows were not retired"
        )

        for _ in range(_MAX_STEPS):
            if h.finished:
                break
            yield
        else:
            raise AssertionError(f"request did not finish in {_MAX_STEPS} steps")


class TestScriptedSwaRetireAfterPrefillNoOverlap(TestScriptedSwaRetireAfterPrefill):
    ENGINE_KWARGS = {
        **TestScriptedSwaRetireAfterPrefill.ENGINE_KWARGS,
        "disable_overlap_schedule": True,
    }


if __name__ == "__main__":
    unittest.main()
