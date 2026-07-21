import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.scripted_runtime.context import ScriptedContext
from sglang.test.scripted_runtime.test_case import ScriptedTestCase
from sglang.test.scripted_runtime_chunked_helpers import base_engine_kwargs

register_cuda_ci(est_time=124, stage="extra-a", runner_config="1-gpu-large")


_SWA_MODEL = "openai/gpt-oss-20b"

_MAX_TOTAL_TOKENS = 4096
_SWA_FULL_TOKENS_RATIO = 0.1
_CHUNK_SIZE = 64

_N_DECODERS = 6
_DECODER_PROMPT = 64
_DECODER_MAX_NEW = 512
_DECODER_WARMUP_RUNNING = 3

_CHUNKED_PROMPT = 384
_CHUNKED_MAX_NEW = 2
_N_CANDIDATES = 24
_STEPS_PER_CANDIDATE = 120
_DECODER_WARMUP_STEPS = 60
_DRAIN_STEPS = 400


class TestScriptedSwaChunkedReqEarlyReturn(ScriptedTestCase):
    ENGINE_KWARGS = base_engine_kwargs(
        model_path=_SWA_MODEL,
        chunked_prefill_size=_CHUNK_SIZE,
        max_total_tokens=_MAX_TOTAL_TOKENS,
        swa_full_tokens_ratio=_SWA_FULL_TOKENS_RATIO,
        page_size=1,
        mem_fraction_static=0.70,
    )

    def test_swa_chunked_req_early_return_no_double_free(self):
        self.server.execute_script(
            self._script_swa_chunked_req_early_return_no_double_free
        )

    @staticmethod
    def _script_swa_chunked_req_early_return_no_double_free(t: ScriptedContext):
        s = t.scheduler

        for i in range(_N_DECODERS):
            t.start_req(
                prompt_len=_DECODER_PROMPT,
                max_new_tokens=_DECODER_MAX_NEW,
                ignore_eos=True,
                prompt_token=10 + i,
            )
        for _ in range(_DECODER_WARMUP_STEPS):
            if len(s.running_batch.reqs) >= _DECODER_WARMUP_RUNNING:
                break
            yield

        candidates = []
        parked = False
        for _ in range(_N_CANDIDATES):
            candidates.append(
                t.start_req(
                    prompt_len=_CHUNKED_PROMPT,
                    max_new_tokens=_CHUNKED_MAX_NEW,
                    prompt_token=2,
                )
            )
            for _ in range(_STEPS_PER_CANDIDATE):
                if any(t.chunked_parks(c.rid) > 0 for c in candidates):
                    parked = True
                    break
                if candidates[-1].finished:
                    break
                yield
            if parked:
                break

        parked = parked or any(t.chunked_parks(c.rid) > 0 for c in candidates)
        assert parked, (
            "no chunked candidate was ever parked by add_chunked_req's hybrid-SWA "
            "early-return; the test never exercised the stash gate"
        )

        t.abort_all()
        for _ in range(_DRAIN_STEPS):
            if (
                s.chunked_req is None
                and len(s.waiting_queue) == 0
                and s.running_batch.is_empty()
            ):
                break
            yield
        for _ in range(20):
            yield

        locked = {nid: lr for nid, lr in t.get_all_node_lock_refs().items() if lr != 0}
        assert not locked, (
            f"radix nodes left locked after drain {locked} -- stash gate let an "
            "un-scheduled chunked req commit partial KV"
        )


if __name__ == "__main__":
    unittest.main()
