import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.scripted_runtime.runtime import ScriptedRuntime
from sglang.test.scripted_runtime.testcase import ScriptedRuntimeTestCase
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
# Sizing so the SWA budget — not the full pool — is what starves the chunked
# req, i.e. the genuine hybrid-SWA early-return path. The arithmetic below
# assumes page_size == 1 (set explicitly) so token accounting is exact.
#
# Pools (pool_configurator.py): with max_total_tokens = M and
# swa_full_tokens_ratio = R,
#     full pool  F = M                 = 512 slots
#     swa  pool  S = floor(R * M)      = floor(0.25 * 512) = 128 slots
#
# gpt-oss-20b sliding window W = sliding_window(128) - 1 = 127
# (models/gpt_oss.py:get_attention_sliding_window_size). Every prompt here is
# <= W, so NO token is ever beyond the window => SWA keeps all of them =>
# a req's SWA footprint == its committed length, identical to its full-pool
# footprint. Two facts follow:
#   * The full pool F=512 holds ~5 reqs of 96 -> rem_total_tokens never hits 0.
#   * The SWA pool S=128 is the smaller budget for the same footprint -> it is
#     the binding constraint. This is the whole reason swa_full_tokens_ratio
#     must be set: it pins S small enough to starve before F.
#
# Timeline (rem_swa = swa_available + swa_evictable; resident reqs are not
# evictable):
#   1. Competitor (prompt 96) prefills then stays resident decoding, holding
#      96 SWA slots -> rem_swa = 128 - 96 = 32.
#   2. Chunked r (prompt 96, chunk 32) resumes. add_chunked_req computes
#         _rem = min(rem_chunk_tokens=32, rem_total_tokens(~400),
#                    rem_swa_tokens - page_size)
#      First resume: rem_swa 32 -> 32-1 = 31 -> admits 31 (truncated chunk);
#      rem_swa -> 1. Next resume: rem_swa 1 -> 1-1 = 0 <= 0 -> is_hybrid_swa
#      -> `return req`: the early-return. r stays parked as scheduler.chunked_req
#      without entering the forward batch.
#   3. Competitor finishes its 32 decode tokens and frees its 96 slots ->
#      rem_swa jumps back above a chunk -> r drains to completion (96 <= F, S).
# ---------------------------------------------------------------------------
_MAX_TOTAL_TOKENS = 512
_SWA_FULL_TOKENS_RATIO = 0.25  # -> SWA pool = 128 slots, the binding budget
_CHUNK_SIZE = 32
_SWA_WINDOW = 127  # gpt-oss-20b: sliding_window(128) - 1
_RESIDENT_PROMPT = 96  # <= _SWA_WINDOW so SWA footprint == length
_RESIDENT_DECODE = 32
_CHUNKED_PROMPT = 96  # <= _SWA_WINDOW; > _CHUNK_SIZE so it actually chunks

assert _RESIDENT_PROMPT <= _SWA_WINDOW and _CHUNKED_PROMPT <= _SWA_WINDOW
assert _CHUNKED_PROMPT > _CHUNK_SIZE


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
        """SWA-budget starvation makes add_chunked_req early-return; the stash gate keeps the un-scheduled chunked req's partial KV out of the tree."""
        self.runtime.run(self._script_swa_chunked_req_early_return_no_double_free)

    # A resident competitor occupies most of the tiny SWA pool, so the chunked
    # req hits the add_chunked_req SWA early-return: it stays parked as
    # scheduler.chunked_req without running, and the stash gate must keep its
    # partial KV out of the tree so no radix node is left locked after drain.
    @staticmethod
    def _script_swa_chunked_req_early_return_no_double_free(t: ScriptedRuntime):
        s = t._scheduler

        # Bring the competitor up first and let it finish its own chunked
        # prefill, so it is resident (decoding) and holding ~_RESIDENT_PROMPT
        # SWA slots before the chunked req under test starts. Only one req may
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
            "forward batch — the SWA budget was never tight enough"
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
