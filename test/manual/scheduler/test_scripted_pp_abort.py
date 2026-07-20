import unittest

from sglang.test.scripted_runtime.context import ScriptedContext
from sglang.test.scripted_runtime.req_handle import ScriptedReqHandle
from sglang.test.scripted_runtime.test_case import ScriptedTestCase
from sglang.test.scripted_runtime_chunked_helpers import (
    DEFAULT_CHUNK_SIZE,
    base_engine_kwargs,
)


def _drain_until_released(t: ScriptedContext, *handles: ScriptedReqHandle):
    for _ in range(12):
        if all(
            h.kv_pages == 0
            and h.lock_refs == 0
            and (h.req is None or h.req.req_pool_idx is None)
            for h in handles
        ):
            return
        yield


def _advance_until_in_running_mbs(
    t: ScriptedContext, *handles: ScriptedReqHandle, max_steps: int = 800
):
    rids = {h.rid for h in handles}
    present: set[str] = set()
    for _ in range(max_steps):
        present = {
            req.rid
            for mb in t.scheduler.running_mbs
            for req in mb.reqs
            if req.rid in rids
        }
        if present == rids:
            return
        yield
    raise AssertionError(
        f"reqs did not all reach a running_mbs decode slot within {max_steps} "
        f"steps; present={present!r} wanted={rids!r}"
    )


class TestAbortPPCrossSlot(ScriptedTestCase):
    ENGINE_KWARGS = base_engine_kwargs(
        chunked_prefill_size=DEFAULT_CHUNK_SIZE,
        pp_size=4,
        pp_max_micro_batch_size=1,
    )

    def test_abort_all_reaches_running_reqs_in_all_microbatch_slots(self):
        """abort_all must abort running reqs in every PP microbatch slot, not just the current one (needs >2 slots)."""
        self.server.execute_script(
            self._script_abort_all_reaches_running_reqs_in_all_microbatch_slots
        )

    @staticmethod
    def _script_abort_all_reaches_running_reqs_in_all_microbatch_slots(
        t: ScriptedContext,
    ):
        reqs = [
            t.start_req(
                prompt_len=16,
                max_new_tokens=512,
                ignore_eos=True,
                prompt_token=310 + i,
            )
            for i in range(4)
        ]
        yield from _advance_until_in_running_mbs(t, *reqs)

        slot_of = {}
        for slot_id, mb in enumerate(t.scheduler.running_mbs):
            for req in mb.reqs:
                slot_of[req.rid] = slot_id
        slots = {slot_of[r.rid] for r in reqs}
        assert len(slots) == len(reqs), (
            f"setup invalid: reqs must each occupy a distinct mb slot to exercise "
            f"the cross-slot abort scan; slot_of={slot_of!r}"
        )

        t.abort_all()
        yield from _drain_until_released(t, *reqs)

        alive = {r.rid: r.kv_pages for r in reqs if r.kv_pages != 0}
        assert not alive, (
            f"abort_all left running reqs alive in non-current mb slots (only the "
            f"current slot + stale cur_batch were scanned): still_holding_kv={alive!r}"
        )


if __name__ == "__main__":
    unittest.main()
