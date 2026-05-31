import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.scripted_runtime.context import ScriptedContext
from sglang.test.scripted_runtime.test_case import ScriptedTestCase
from sglang.test.scripted_runtime_chunked_helpers import (
    base_engine_kwargs,
    run_until,
)

register_cuda_ci(est_time=400, stage="extra-a", runner_config="1-gpu-large")


_SWA_MODEL = "openai/gpt-oss-20b"

# To reach add_chunked_req's early-return the full KV pool must run out while the
# chunked req r is mid-prefill. The admission gate already reserves the resident
# competitor's future decode tokens, so r is only admitted when competitor +
# r_prompt fits the pool; the pool can therefore only overflow via the resident
# req decoding *more* than its (0.7x) reservation. enable_mixed_chunk lets the
# competitor decode alongside r's prefill chunks, and the prompt/decode sizes are
# tuned so that extra growth pushes the pool to 1.00 on r's final chunk while the
# competitor (ignore_eos) is still resident -- parking r until it frees its KV.
_MAX_TOTAL_TOKENS = 8192
_SWA_FULL_TOKENS_RATIO = 0.8
_CHUNK_SIZE = 64
_RESIDENT_PROMPT = 4032
_RESIDENT_DECODE = 82
_CHUNKED_PROMPT = 4096


class TestScriptedSwaChunkedReqEarlyReturn(ScriptedTestCase):
    ENGINE_KWARGS = base_engine_kwargs(
        model_path=_SWA_MODEL,
        chunked_prefill_size=_CHUNK_SIZE,
        max_total_tokens=_MAX_TOTAL_TOKENS,
        swa_full_tokens_ratio=_SWA_FULL_TOKENS_RATIO,
        page_size=1,
        mem_fraction_static=0.70,
        disable_piecewise_cuda_graph=True,
        enable_mixed_chunk=True,
    )

    def test_swa_chunked_req_early_return_no_double_free(self):
        self.server.execute_script(
            self._script_swa_chunked_req_early_return_no_double_free
        )

    @staticmethod
    def _script_swa_chunked_req_early_return_no_double_free(t: ScriptedContext):
        s = t._scheduler

        competitor = t.start_req(
            prompt_len=_RESIDENT_PROMPT,
            max_new_tokens=_RESIDENT_DECODE,
            ignore_eos=True,
        )
        yield from run_until(competitor, lambda h: h.is_chunking)
        yield from run_until(competitor, lambda h: not h.is_chunking)

        r = t.start_req(prompt_len=_CHUNKED_PROMPT, max_new_tokens=2, prompt_token=2)
        yield from run_until(r, lambda h: h.is_chunking)

        observed_early_return = False
        for _ in range(800):
            chunked = s.chunked_req
            last_batch = s.last_batch
            if (
                chunked is not None
                and chunked.rid == r.rid
                and last_batch is not None
                and chunked not in last_batch.reqs
            ):
                observed_early_return = True
            if r.finished and competitor.finished:
                break
            yield

        assert (
            r.finished
        ), "chunked req did not finish after the competitor freed its KV"
        assert observed_early_return, (
            "test must exercise the add_chunked_req early-return branch: no iter "
            "observed r parked as scheduler.chunked_req while absent from the "
            "forward batch — the KV pool was never tight enough"
        )

        yield from run_until(
            r,
            lambda _h: s.chunked_req is None
            and len(s.waiting_queue) == 0
            and s.running_batch.is_empty(),
        )
        locked = {nid: lr for nid, lr in t.get_all_node_lock_refs().items() if lr != 0}
        assert not locked, (
            f"radix nodes left locked after drain {locked} — stash gate let "
            "an un-scheduled chunked req commit partial KV"
        )


if __name__ == "__main__":
    unittest.main()
