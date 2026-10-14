import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.scripted_runtime.context import ScriptedContext
from sglang.test.scripted_runtime.test_case import ScriptedTestCase
from sglang.test.scripted_runtime_chunked_helpers import base_engine_kwargs

register_cuda_ci(est_time=330, stage="extra-a", runner_config="1-gpu-large")


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

_MIXED_DECODER_MAX_NEW = 256
_MIXED_N_CANDIDATES = 4
_MIXED_MAX_STEPS = 4000


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


class TestScriptedSwaMixedChunkDecodeFit(ScriptedTestCase):
    # A mixed step whose admitted prefill leaves no SWA room for the running
    # rows must hold them back for a decode step, not run out of the pool in
    # alloc_for_decode.
    ENGINE_KWARGS = base_engine_kwargs(
        model_path=_SWA_MODEL,
        chunked_prefill_size=_CHUNK_SIZE,
        max_total_tokens=_MAX_TOTAL_TOKENS,
        swa_full_tokens_ratio=_SWA_FULL_TOKENS_RATIO,
        page_size=1,
        mem_fraction_static=0.70,
        enable_mixed_chunk=True,
    )

    def test_mixed_step_holds_back_decode_rows_that_do_not_fit(self):
        self.server.execute_script(
            self._script_mixed_step_holds_back_decode_rows_that_do_not_fit
        )

    @staticmethod
    def _script_mixed_step_holds_back_decode_rows_that_do_not_fit(
        t: ScriptedContext,
    ):
        s = t.scheduler

        decoders = [
            t.start_req(
                prompt_len=_DECODER_PROMPT,
                max_new_tokens=_MIXED_DECODER_MAX_NEW,
                ignore_eos=True,
                prompt_token=10 + i,
            )
            for i in range(_N_DECODERS)
        ]
        for _ in range(_DECODER_WARMUP_STEPS):
            if len(s.running_batch.reqs) >= _DECODER_WARMUP_RUNNING:
                break
            yield

        candidates = [
            t.start_req(
                prompt_len=_CHUNKED_PROMPT,
                max_new_tokens=_CHUNKED_MAX_NEW,
                ignore_eos=True,
                prompt_token=100 + i,
            )
            for i in range(_MIXED_N_CANDIDATES)
        ]
        handles = decoders + candidates

        decode_held_back = False
        decoder_rids = {h.rid for h in decoders}
        for _ in range(_MIXED_MAX_STEPS):
            batch = s.last_batch
            if (
                batch is not None
                and batch.forward_mode is not None
                and batch.forward_mode.is_extend()
                and not batch.forward_mode.is_mixed()
                and any(
                    r.rid in decoder_rids and not r.finished()
                    for r in s.running_batch.reqs
                )
            ):
                decode_held_back = True
            if all(h.finished for h in handles):
                break
            yield
        else:
            raise AssertionError(
                f"requests still running after {_MIXED_MAX_STEPS} steps: "
                f"{[h.rid for h in handles if not h.finished]}"
            )

        assert decode_held_back, (
            "no prefill ran while running decode rows sat out the step; the "
            "test never put the SWA pool under pressure on a mixed step"
        )


if __name__ == "__main__":
    unittest.main()
