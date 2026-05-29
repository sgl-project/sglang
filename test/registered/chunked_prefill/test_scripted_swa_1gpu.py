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

# The whole point of the test is to starve the KV cache so the chunked req
# cannot allocate its next chunk. We pin the pool to an explicit, tiny size
# and pick prompts around it:
#
#   full KV pool == _MAX_TOTAL_TOKENS slots.
#   A resident competitor holds ~_RESIDENT_PROMPT slots; that leaves
#   _MAX_TOTAL_TOKENS - _RESIDENT_PROMPT (< _CHUNKED_PROMPT) for the chunked
#   req. So once the chunked req has committed a couple of chunks it can no
#   longer extend -> add_chunked_req returns it without admitting it (the
#   early-return; valid for any exhausted budget under hybrid SWA). Once the
#   competitor finishes and frees its slots, the chunked req (which fits the
#   pool on its own) drains to completion.
_MAX_TOTAL_TOKENS = 2048
_CHUNK_SIZE = 256
_RESIDENT_PROMPT = 1280
_RESIDENT_DECODE = 32
_CHUNKED_PROMPT = 1280


class TestScriptedSwaChunkedReqEarlyReturn(ScriptedRuntimeTestCase):
    ENGINE_KWARGS = base_engine_kwargs(
        model_path=_SWA_MODEL,
        chunked_prefill_size=_CHUNK_SIZE,
        max_total_tokens=_MAX_TOTAL_TOKENS,
        mem_fraction_static=0.70,
        disable_piecewise_cuda_graph=True,
    )

    def test_swa_chunked_req_early_return_no_double_free(self):
        """Under KV-cache starvation the chunked req's add_chunked_req early-returns; the stash gate keeps its partial KV out of the tree."""
        self.runtime.run(self._script_swa_chunked_req_early_return_no_double_free)

    # A resident competitor occupies most of a deliberately tiny KV pool, so
    # the chunked req hits the add_chunked_req early-return: it stays parked as
    # scheduler.chunked_req without running, and the stash gate must keep its
    # partial KV out of the tree so no radix node is left locked after drain.
    @staticmethod
    def _script_swa_chunked_req_early_return_no_double_free(t: ScriptedRuntime):
        s = t._scheduler

        # Bring the competitor up first and let it finish its own chunked
        # prefill, so it is resident (decoding) and holding ~_RESIDENT_PROMPT
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
