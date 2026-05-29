import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.scripted_runtime.runtime import ScriptedRuntime
from sglang.test.scripted_runtime.test_case import ScriptedRuntimeTestCase
from sglang.test.scripted_runtime_chunked_helpers import (
    base_engine_kwargs,
    run_until,
)

# Single-GPU scripted test for the hybrid-SWA add_chunked_req early-return
# gate. gpt-oss-20b is a hybrid-SWA model and needs an 80 GB GPU for its
# weights, so this runs on 1-gpu-large rather than the 1-gpu-small runner
# the rest of the core suite uses.
register_cuda_ci(est_time=400, stage="extra-a", runner_config="1-gpu-large")


_SWA_MODEL = "openai/gpt-oss-20b"

# ---------------------------------------------------------------------------
# Why the chunked req early-returns (page_size == 1 so token accounting is
# exact; the engine kwargs pin it).
#
# Pools (pool_configurator.py) with max_total_tokens = M, swa_full_tokens_ratio
# = R:
#     full pool  F = M            = 8192 slots   (full-attention layers)
#     swa  pool  S = floor(R*M)   = 819 slots    (sliding-window layers)
#
# gpt-oss-20b sliding window W = sliding_window(128) - 1 = 127. The two layer
# families consume their pools very differently for a req of committed length
# L:
#     full-attention layers keep ALL tokens  -> full footprint  = L
#     sliding-window layers keep only last W -> swa  footprint  = min(L, W) ~ 127
# So a single req only ever ties up ~127 SWA slots: S=819 is never the binding
# budget here. The FULL pool is. (Set swa_full_tokens_ratio low anyway so S is
# explicit and we can show it stays slack rather than relying on the 0.8
# default.) add_chunked_req early-returns the moment EITHER budget is exhausted
# under hybrid SWA, so starving the full pool is enough.
#
# Timeline:
#   1. Competitor (prompt 6144) prefills then stays resident decoding. Its
#      full-pool footprint is the whole 6144 (full layers keep everything),
#      plus a small decode reserve -> free_full ~ 8192 - 6144 - ~64 ~ 1984.
#   2. Chunked r (prompt 4096, chunk 512) resumes. add_chunked_req computes
#         _rem = min(rem_chunk_tokens=512, rem_total_tokens, rem_swa - page)
#      rem_total_tokens walks 1984 -> 1472 -> 960 -> 448 as r commits 512-token
#      chunks; the 4th chunk truncates to 448 and drives free_full to 0, so the
#      next resume sees rem_total_tokens <= 0 -> _rem <= 0 -> is_hybrid_swa ->
#      `return req`: the early-return. r (4096) far exceeds the ~1984 free, so
#      this is guaranteed, not marginal. r stays parked as scheduler.chunked_req
#      without entering the forward batch.
#   3. Competitor finishes its decode tokens and frees 6144 slots -> r resumes
#      and drains (4096 <= F=8192, swa footprint ~127 <= S=819).
# ---------------------------------------------------------------------------
_MAX_TOTAL_TOKENS = 8192
_SWA_FULL_TOKENS_RATIO = 0.1  # SWA pool = 819 slots; stays slack (window-capped)
_CHUNK_SIZE = 512
_RESIDENT_PROMPT = 6144  # competitor: pins ~6144 of the 8192 full pool
_RESIDENT_DECODE = 64  # keep it resident while r overflows and early-returns
_CHUNKED_PROMPT = 4096  # >> the ~1984 free full slots -> guaranteed overflow


class TestScriptedSwaChunkedReqEarlyReturn(ScriptedRuntimeTestCase):
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
        """KV-cache starvation makes add_chunked_req early-return; the stash gate keeps the un-scheduled chunked req's partial KV out of the tree."""
        self.runtime.run(self._script_swa_chunked_req_early_return_no_double_free)

    # A resident competitor pins most of the KV pool, so the chunked req hits
    # the add_chunked_req early-return: it stays parked as scheduler.chunked_req
    # without running, and the stash gate must keep its partial KV out of the
    # tree so no radix node is left locked after drain.
    @staticmethod
    def _script_swa_chunked_req_early_return_no_double_free(t: ScriptedRuntime):
        s = t._scheduler

        # Bring the competitor up first and let it finish its own chunked
        # prefill, so it is resident (decoding) and pinning ~_RESIDENT_PROMPT
        # KV slots before the chunked req under test starts. Only one req may
        # be chunking at a time, so starting them together would just serialize
        # them with no pressure.
        competitor = t.start_req(
            prompt_len=_RESIDENT_PROMPT, max_new_tokens=_RESIDENT_DECODE
        )
        yield from run_until(competitor, lambda h: h.is_chunking)
        yield from run_until(competitor, lambda h: not h.is_chunking)

        r = t.start_req(prompt_len=_CHUNKED_PROMPT, max_new_tokens=2)
        yield from run_until(r, lambda h: h.is_chunking)

        # "assert the scenario exists" derived purely from real batch state,
        # no scheduler instrumentation: an early-return leaves r parked as
        # scheduler.chunked_req yet absent from the iter's forward batch
        # (add_chunked_req returned it without admitting it into can_run_list,
        # so it never reached last_batch). A normally-scheduled chunk would
        # appear in last_batch.reqs the next yield.
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

        # Gate held: drain to a fully idle engine, then assert no radix node
        # stayed locked. A stash on the un-scheduled chunked req (broken gate)
        # inc_lock_refs a path with no matching release, surfacing here as a
        # nonzero terminal lock_ref.
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
