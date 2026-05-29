import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.scripted_runtime.runtime import ScriptedRuntime
from sglang.test.scripted_runtime.testcase import ScriptedRuntimeTestCase
from sglang.test.scripted_runtime_chunked_helpers import (
    DEFAULT_CHUNK_SIZE,
    VERY_LONG_PROMPT_LEN,
    base_engine_kwargs,
    run_until,
)

# Single-GPU scripted chunked-prefill test for the hybrid-SWA
# add_chunked_req early-return gate. gpt-oss-20b is a hybrid-SWA model
# and needs a single 80 GB GPU, so this runs on 1-gpu-large rather than
# the 1-gpu-small runner the rest of the core suite uses.
register_cuda_ci(est_time=400, stage="extra-a", runner_config="1-gpu-large")


_SWA_MODEL = "openai/gpt-oss-20b"


class TestScriptedSwaChunkedReqScheduledLastIter(ScriptedRuntimeTestCase):
    ENGINE_KWARGS = base_engine_kwargs(
        model_path=_SWA_MODEL,
        chunked_prefill_size=DEFAULT_CHUNK_SIZE,
        mem_fraction_static=0.70,
        disable_piecewise_cuda_graph=True,
    )

    def test_swa_chunked_req_early_return_no_double_free(self):
        """SWA add_chunked_req early-return must not stash the un-scheduled chunked req."""
        self.runtime.run(self._script_swa_chunked_req_early_return_no_double_free)

    # SWA pool critical + add_chunked_req forced early-return: the chunked
    # req stays parked as scheduler.chunked_req without running, and the
    # stash gate must keep its partial KV out of the tree so no radix node
    # is left locked once the engine drains.
    @staticmethod
    def _script_swa_chunked_req_early_return_no_double_free(t: ScriptedRuntime):
        s = t._scheduler
        # Long competitor consumes SWA budget so the chunked req hits
        # the add_chunked_req SWA-early-return branch at least once.
        competitor = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=8)
        r = t.start_req(prompt_len=VERY_LONG_PROMPT_LEN, max_new_tokens=2)
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

        assert r.finished, "chunked req under SWA pressure did not finish"
        assert observed_early_return, (
            "test must exercise the SWA add_chunked_req early-return branch: "
            "no iter observed r parked as scheduler.chunked_req while absent "
            "from the forward batch — SWA budget was never tight enough"
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
        locked = {
            nid: lr for nid, lr in t.get_all_node_lock_refs().items() if lr != 0
        }
        assert not locked, (
            f"radix nodes left locked after drain {locked} — stash gate let "
            "an un-scheduled chunked req commit partial KV"
        )


if __name__ == "__main__":
    unittest.main()
