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

# To reach add_chunked_req's hybrid-SWA early-return the full KV pool must run
# out while the chunked req r is mid-prefill. We arrange this WITHOUT
# enable_mixed_chunk and WITHOUT an ignore_eos competitor:
#
# A resident competitor (bounded by max_new_tokens, EOS allowed) finishes its
# own prefill first, so its entire prompt occupies the full-attention KV pool.
# While r then prefills chunk by chunk, prefill batches preempt decode, so the
# competitor never gets a decode step -- it neither advances toward EOS nor
# grows, staying pinned at its full prompt footprint. The competitor prompt +
# r prompt deliberately exceed the full pool (4096 + 4608 > 8192), so r's
# per-chunk add_chunked_req drives rem_total_tokens <= 0 partway through its
# prefill -- the early-return parks r (kept as scheduler.chunked_req but absent
# from the forward batch). Only once r is parked does the decode batch run,
# letting the competitor decode to max_new_tokens, finish, and free its KV so r
# can resume and complete.
_MAX_TOTAL_TOKENS = 8192
_SWA_FULL_TOKENS_RATIO = 0.8
_CHUNK_SIZE = 64
_RESIDENT_PROMPT = 4096
_RESIDENT_DECODE = 82
_CHUNKED_PROMPT = 4608


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

        competitor = t.start_req(
            prompt_len=_RESIDENT_PROMPT,
            max_new_tokens=_RESIDENT_DECODE,
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
