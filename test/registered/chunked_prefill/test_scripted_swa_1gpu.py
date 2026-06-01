import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.scripted_runtime.context import ScriptedContext
from sglang.test.scripted_runtime.test_case import ScriptedTestCase
from sglang.test.scripted_runtime_chunked_helpers import base_engine_kwargs

register_cuda_ci(est_time=400, stage="extra-a", runner_config="1-gpu-large")


_SWA_MODEL = "openai/gpt-oss-20b"

# Regression for issue #24252 / PR #24318: a chunked req that is deferred by
# add_chunked_req's hybrid-SWA early-return must not be stash-cached, otherwise
# its already-freed req_pool_idx is double-freed and prefix_indices corrupted,
# producing an empty/garbage micro-batch that crashes the extend path.
#
# This reproduces the production trigger directly, with no internal-state pokes.
# With the overlap scheduler (the production default), decode iterations are
# pipelined alongside an in-flight chunked prefill, so decode -- and therefore
# the retraction it can trigger -- runs while a req is mid-chunk. We saturate a
# small hybrid-SWA pool (swa_full_tokens_ratio=0.1, as in the issue) with a pool
# of ignore_eos decoders. They drive repeated real "KV cache pool is full.
# Retract requests." retractions; while a chunked req is mid-prefill, the SWA
# pool exhausts (and the retract's new_token_ratio jump inflates the running
# batch's reserved-decode offset), so the chunked req's next add_chunked_req
# early-returns -- parking it (the scheduler keeps it as chunked_req but does not
# run it in the batch). We detect the park purely from observed batch composition
# (t.chunked_parks: an iteration whose chunked_rid is this req yet the req is
# absent from the batch that ran), so the test does not depend on any internal
# scheduler scheduling flag. Which candidate gets parked depends on the exact
# churn, so we feed chunked candidates until one is observed parked, then assert
# the stash gate left no radix lock refs after the engine drains. On the un-gated
# (buggy) code the spurious stash double-frees and crashes the scheduler
# (KV-canary).
_MAX_TOTAL_TOKENS = 4096
_SWA_FULL_TOKENS_RATIO = 0.1
_CHUNK_SIZE = 64

# Decoders that fill the small SWA pool and drive the retraction churn.
_N_DECODERS = 6
_DECODER_PROMPT = 64
_DECODER_MAX_NEW = 512
_DECODER_WARMUP_RUNNING = 3

# Chunked candidate: long enough to span several chunks (so it is mid-prefill
# when a retraction fires), short enough to be admitted under memory pressure.
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
        s = t._scheduler

        # Sustained memory pressure: ignore_eos decoders that keep the small SWA
        # pool saturated and trigger repeated real retractions.
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

        # Feed chunked candidates until one is parked by the add_chunked_req
        # hybrid-SWA early-return.
        parked = False
        for _ in range(_N_CANDIDATES):
            r = t.start_req(
                prompt_len=_CHUNKED_PROMPT,
                max_new_tokens=_CHUNKED_MAX_NEW,
                prompt_token=2,
            )
            for _ in range(_STEPS_PER_CANDIDATE):
                if t.chunked_parks(r.rid) > 0:
                    parked = True
                    break
                if r.finished:
                    break
                yield
            if parked:
                break

        assert parked, (
            "no chunked candidate was ever parked by add_chunked_req's hybrid-SWA "
            "early-return; the test never exercised the stash gate"
        )

        # Drain. On the buggy (un-gated) code the spurious stash of the deferred
        # chunked req has already double-freed/corrupted state (KV-canary crash)
        # before we get here; on fixed code the drain must leave no locked nodes.
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
