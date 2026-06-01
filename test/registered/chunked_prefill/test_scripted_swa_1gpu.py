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

# add_chunked_req's hybrid-SWA early-return (schedule_policy.py) fires when the
# full KV pool runs out (rem_total_tokens <= 0) while a request r is mid-chunked-
# prefill: r is kept as scheduler.chunked_req but is *not* added to the forward
# batch, so the scheduler must not stash its partial KV (which would leak radix
# lock refs / double-free).
#
# Reaching that condition through ordinary request traffic requires a second KV
# consumer to grow *during* r's chunked prefill -- but without enable_mixed_chunk
# a running request never decodes while a chunked req is being prefilled (prefill
# preempts decode), and no second request can be admitted while the chunked req
# holds the per-batch chunk budget. The only such consumer is enable_mixed_chunk
# decode kept alive by an ignore_eos competitor.
#
# So instead of enable_mixed_chunk + ignore_eos, we drive the exact pool state
# directly: admit r on an empty pool, then drain the full-attention allocator
# from the script so r's next chunk cannot be admitted -> the real scheduler
# takes the early-return and parks r. Releasing the reserved KV lets r resume and
# finish; the test then asserts no radix lock refs were leaked.
_MAX_TOTAL_TOKENS = 8192
_SWA_FULL_TOKENS_RATIO = 0.8
_CHUNK_SIZE = 64
_CHUNKED_PROMPT = 512


class TestScriptedSwaChunkedReqEarlyReturn(ScriptedTestCase):
    ENGINE_KWARGS = base_engine_kwargs(
        model_path=_SWA_MODEL,
        chunked_prefill_size=_CHUNK_SIZE,
        max_total_tokens=_MAX_TOTAL_TOKENS,
        swa_full_tokens_ratio=_SWA_FULL_TOKENS_RATIO,
        page_size=1,
        mem_fraction_static=0.70,
        disable_piecewise_cuda_graph=True,
    )

    def test_swa_chunked_req_early_return_no_double_free(self):
        self.server.execute_script(
            self._script_swa_chunked_req_early_return_no_double_free
        )

    @staticmethod
    def _script_swa_chunked_req_early_return_no_double_free(t: ScriptedContext):
        s = t._scheduler

        r = t.start_req(prompt_len=_CHUNKED_PROMPT, max_new_tokens=2, prompt_token=2)
        yield from run_until(r, lambda h: h.is_chunking)

        # Drain the full-attention pool so rem_total_tokens <= 0 on r's next
        # chunk; the scheduler then early-returns and parks r without scheduling
        # it (_chunked_req_scheduled_last_iter stays False).
        reservation = t.reserve_full_attention_kv(t.full_attention_available_size())

        observed_early_return = False
        for _ in range(200):
            if (
                s.chunked_req is not None
                and s.chunked_req.rid == r.rid
                and not s._chunked_req_scheduled_last_iter
            ):
                observed_early_return = True
                break
            yield

        assert observed_early_return, (
            "test must exercise the add_chunked_req early-return branch: r was "
            "never parked as scheduler.chunked_req while unscheduled despite the "
            "full KV pool being drained"
        )
        assert not r.finished, "r must still be mid-chunked-prefill while parked"

        # Releasing the reserved KV lets the parked chunk be admitted again.
        reservation.release()

        yield from run_until(
            r,
            lambda _h: s.chunked_req is None
            and len(s.waiting_queue) == 0
            and s.running_batch.is_empty(),
        )
        assert r.finished, "chunked req did not finish after KV pressure was released"

        locked = {nid: lr for nid, lr in t.get_all_node_lock_refs().items() if lr != 0}
        assert not locked, (
            f"radix nodes left locked after drain {locked} — stash gate let "
            "an un-scheduled chunked req commit partial KV"
        )


if __name__ == "__main__":
    unittest.main()
