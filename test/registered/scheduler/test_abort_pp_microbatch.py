"""
Unit tests for scheduler.abort_request traversing all microbatches in PP mode.

Background: In PP mode, running_mbs is a list of ScheduleBatch slots (length = pp_size).
When abort_request is called, self.running_batch only points to the slot of the current
mb_id. The original code only checked that slot, missing requests in other slots.
The fix adds a traversal over all running_mbs entries.

This is a pure CPU unit test with no GPU / real model dependency. Run directly with:
    python3 test_abort_pp_microbatch.py
"""

import unittest
from unittest.mock import MagicMock

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=5, suite="stage-a-test-cpu")

# ---------------------------------------------------------------------------
# Lightweight mocks: minimal interfaces for Req / ScheduleBatch / AbortReq
# ---------------------------------------------------------------------------


class _Req:
    def __init__(self, rid: str, already_finished: bool = False):
        self.rid = rid
        # finished_reason is None means not yet finished
        self.finished_reason = object() if already_finished else None
        self.to_finish = None

    def finished(self) -> bool:
        return self.finished_reason is not None


class _Batch:
    def __init__(self, reqs=None):
        self.reqs = reqs or []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_scheduler_instance(pp_size: int, current_mb_id: int):
    """
    Import the real Scheduler class and return an instance with __init__ bypassed.
    All attributes accessed by abort_request before the running_batch block are
    stubbed out so the method can reach the code under test.
    """
    # Import here so that the module-level import of scheduler.py (which pulls in
    # torch, zmq, etc.) only happens when the test actually runs, not at collection time.
    from sglang.srt.managers.scheduler import Scheduler

    # Bypass __init__ entirely; we will set every attribute manually.
    instance = object.__new__(Scheduler)

    # --- Stubs for the early parts of abort_request (waiting_queue, grammar, disagg) ---
    instance.waiting_queue = []
    instance.enable_hicache_storage = False
    instance.disaggregation_mode = MagicMock()
    # Make all disaggregation_mode comparisons return False so those branches are skipped.
    instance.disaggregation_mode.__eq__ = lambda self, other: False
    instance.grammar_manager = MagicMock()

    # --- PP microbatch state ---
    instance.running_mbs = [_Batch() for _ in range(pp_size)]
    instance.running_batch = instance.running_mbs[current_mb_id]
    instance.cur_batch = instance.running_batch

    return instance


def _make_abort_req(rid: str, abort_all: bool = False):
    """Build a real AbortReq dataclass instance."""
    from sglang.srt.managers.io_struct import AbortReq

    return AbortReq(rid=rid, abort_all=abort_all)


def _is_finish_abort(obj) -> bool:
    """Return True if obj is an instance of the real FINISH_ABORT class."""
    from sglang.srt.managers.schedule_batch import FINISH_ABORT

    return isinstance(obj, FINISH_ABORT)


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


class TestAbortRequestPPMicrobatch(CustomTestCase):
    """Verify that abort_request correctly handles all microbatches in PP mode."""

    def test_abort_req_in_current_microbatch(self):
        """Target request is in the current running_batch slot and should be aborted."""
        s = _make_scheduler_instance(pp_size=4, current_mb_id=0)
        target = _Req("req-001")
        other = _Req("req-002")
        s.running_mbs[0].reqs = [target, other]

        s.abort_request(_make_abort_req("req-001"))

        self.assertTrue(_is_finish_abort(target.to_finish))
        self.assertIsNone(other.to_finish)

    def test_abort_req_in_other_microbatch(self):
        """
        Core regression: AbortReq is consumed at mb_id=0, but the target request
        lives in mb_id=2. Before the fix it was silently skipped; after the fix
        it must be marked FINISH_ABORT.
        """
        s = _make_scheduler_instance(pp_size=4, current_mb_id=0)
        target = _Req("req-target")
        bystander = _Req("req-bystander")
        s.running_mbs[2].reqs = [target, bystander]

        s.abort_request(_make_abort_req("req-target"))

        self.assertTrue(
            _is_finish_abort(target.to_finish),
            "Target request in another microbatch must be marked FINISH_ABORT",
        )
        self.assertIsNone(bystander.to_finish)

    def test_abort_req_spread_across_microbatches(self):
        """Target requests spread across multiple microbatches must all be aborted."""
        s = _make_scheduler_instance(pp_size=4, current_mb_id=1)
        req0 = _Req("session-abc-0")
        req2 = _Req("session-abc-1")
        req3 = _Req("session-abc-2")
        unrelated = _Req("session-xyz-0")

        s.running_mbs[0].reqs = [req0]
        # mb_id=1 is the current running_batch and is intentionally left empty
        s.running_mbs[2].reqs = [req2, unrelated]
        s.running_mbs[3].reqs = [req3]

        s.abort_request(_make_abort_req("session-abc"))

        self.assertTrue(_is_finish_abort(req0.to_finish))
        self.assertTrue(_is_finish_abort(req2.to_finish))
        self.assertTrue(_is_finish_abort(req3.to_finish))
        self.assertIsNone(
            unrelated.to_finish,
            "Requests that do not match the prefix must not be aborted",
        )

    def test_abort_all_clears_all_microbatches(self):
        """With abort_all=True, every unfinished request across all microbatches must be aborted."""
        s = _make_scheduler_instance(pp_size=4, current_mb_id=0)
        reqs = [_Req(f"req-{i}") for i in range(8)]
        for i in range(4):
            s.running_mbs[i].reqs = [reqs[i * 2], reqs[i * 2 + 1]]

        s.abort_request(_make_abort_req("", abort_all=True))

        for req in reqs:
            self.assertTrue(_is_finish_abort(req.to_finish))

    def test_already_finished_req_not_overwritten(self):
        """Requests that are already finished must not have to_finish set again."""
        s = _make_scheduler_instance(pp_size=2, current_mb_id=0)
        done = _Req("req-done", already_finished=True)
        active = _Req("req-active")
        s.running_mbs[1].reqs = [done, active]

        s.abort_request(_make_abort_req("", abort_all=True))

        self.assertIsNone(
            done.to_finish,
            "An already-finished request must not have to_finish overwritten",
        )
        self.assertTrue(_is_finish_abort(active.to_finish))


if __name__ == "__main__":
    unittest.main(verbosity=2)
