import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.scripted_runtime.context import ScriptedContext
from sglang.test.scripted_runtime.test_case import ScriptedTestCase
from sglang.test.scripted_runtime_chunked_helpers import (
    base_engine_kwargs,
    run_until,
    run_until_finished,
)

register_cuda_ci(est_time=400, stage="extra-a", runner_config="1-gpu-large")


_SWA_MODEL = "openai/gpt-oss-20b"

# Regression for issue #24252 / PR #24318: a chunked req that is deferred by
# add_chunked_req's hybrid-SWA early-return must not be stash-cached, otherwise
# its already-freed req_pool_idx is double-freed and prefix_indices corrupted,
# producing an empty/garbage micro-batch that crashes the extend path.
#
# The production trigger is a KV-pool-full retraction that jumps new_token_ratio
# (issue log: "Retract requests. #new_token_ratio: 0.0980 -> 0.7589"). The
# running batch's reserved-decode offset is sum_running (max_new - output) *
# new_token_ratio, so a higher ratio inflates that reservation and drives
# rem_total_tokens <= 0 -- which parks a mid-prefill chunked req.
#
# A retraction can only fire on the decode path, and the scripted runtime steps
# deterministically with overlap scheduling disabled, so decode (hence the
# retraction and its ratio jump) cannot run while a chunked req is being
# prefilled. We therefore reproduce the retraction's effect directly: a resident
# request with a large max_new holds the reservation, r is admitted while the
# ratio is low (as after decay), then the ratio is jumped up exactly as
# retract_decode would -- parking r. Restoring the decayed ratio lets r resume,
# and the test asserts the stash gate left no radix lock refs behind.
_MAX_TOTAL_TOKENS = 4096
_SWA_FULL_TOKENS_RATIO = 0.5
_CHUNK_SIZE = 64
# CLIP_MAX_NEW_TOKENS: the holder's reserved decode then ~= the full pool at
# ratio 1.0, so the jumped ratio alone drives rem_total_tokens <= 0.
_RESERVATION_HOLDER_PROMPT = 64
_RESERVATION_HOLDER_MAX_NEW = 4096
_CHUNKED_PROMPT = 512
_DECAYED_RATIO = 0.01
_RETRACT_JUMP_RATIO = 1.0


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
        tracker = s.new_token_ratio_tracker

        # Resident reservation holder: a large remaining max_new means its
        # reserved-decode offset dominates the prefill budget once the ratio is
        # high. It never decodes (prefill preempts decode while r is chunked).
        holder = t.start_req(
            prompt_len=_RESERVATION_HOLDER_PROMPT,
            max_new_tokens=_RESERVATION_HOLDER_MAX_NEW,
            prompt_token=9,
        )
        yield from run_until(holder, lambda h: not h.is_chunking)

        # Admit r while new_token_ratio is low (as after decay): the holder's
        # reserved decode is small, so rem_total_tokens is large and r fits.
        tracker.current = _DECAYED_RATIO
        r = t.start_req(prompt_len=_CHUNKED_PROMPT, max_new_tokens=2, prompt_token=2)
        yield from run_until(r, lambda h: h.is_chunking)

        # The retraction's ratio jump: the holder's reservation balloons and
        # rem_total_tokens goes <= 0, so r's next add_chunked_req early-returns
        # and parks r (chunked_req set, _chunked_req_scheduled_last_iter False).
        tracker.current = _RETRACT_JUMP_RATIO

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
            "never parked as scheduler.chunked_req while unscheduled after the "
            "new_token_ratio jump inflated the reserved-decode offset"
        )
        assert not r.finished, "r must still be mid-chunked-prefill while parked"

        # As the ratio decays back after the retraction, the reservation shrinks
        # and the parked chunk is admitted again.
        tracker.current = _DECAYED_RATIO
        yield from run_until_finished(r)

        # Drain the holder so any remaining radix lock ref is a genuine leak.
        t.abort_all()
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
