"""What a draining retiree counts as its own work, and the shrink gate's scope."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import asyncio
import unittest
from types import MethodType, SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from sglang.srt.managers import tokenizer_manager
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.test.test_utils import CustomTestCase

IDLE = SimpleNamespace(is_empty=lambda: True)
BUSY = SimpleNamespace(is_empty=lambda: False)


def _gate(*, closed, retiring):
    event = asyncio.Event()
    if not closed:
        event.set()
    return SimpleNamespace(
        _elastic_shrink_pause_event=event, _elastic_shrink_retiring=set(retiring)
    )


class TestPendingResults(CustomTestCase):
    CASES = [
        # (case, batches, ignore_idle_batches, expected)
        ("strict count includes idle batches", (IDLE, BUSY), False, 2),
        ("idle batches discounted", (IDLE, BUSY), True, 1),
        ("all idle reads as drained", (IDLE,) * 4, True, 0),
        ("all idle, strict", (IDLE,) * 4, False, 4),
        ("real work still blocks", (BUSY,), True, 1),
        ("empty queue, strict", (), False, 0),
        ("empty queue, discounted", (), True, 0),
    ]

    def test_pending_results(self):
        for case, batches, ignore, expected in self.CASES:
            with self.subTest(case):
                sched = SimpleNamespace(result_queue=[(b, object()) for b in batches])
                self.assertEqual(Scheduler._pending_results(sched, ignore), expected)


class TestShrinkGateScope(CustomTestCase):
    """Only a pin onto a departing slot waits. Everything else goes through."""

    CASES = [
        # (case, closed, retiring, request, admitted)
        ("open gate admits everything", False, (1, 2), {"routed_dp_rank": 1}, True),
        ("closed gate admits unpinned", True, (1, 2), {"routed_dp_rank": None}, True),
        ("closed gate admits a live pin", True, (2, 3), {"routed_dp_rank": 0}, True),
        ("closed gate holds retiring pin", True, (2, 3), {"routed_dp_rank": 3}, False),
        ("no pin attribute is admitted", True, (1, 2), {}, True),
    ]

    def test_gate_scope(self):
        for case, closed, retiring, request, admitted in self.CASES:
            with self.subTest(case):
                tm = _gate(closed=closed, retiring=retiring)
                wait = TokenizerManager._await_elastic_shrink_gate(
                    tm, SimpleNamespace(**request)
                )
                loop = asyncio.new_event_loop()
                try:
                    loop.run_until_complete(asyncio.wait_for(wait, 0.05))
                    got = True
                except asyncio.TimeoutError:
                    got = False
                self.assertEqual(got, admitted)

    def test_close_records_the_retiring_slots_and_open_clears_them(self):
        tm = _gate(closed=False, retiring=())
        TokenizerManager._close_elastic_shrink_gate(tm, range(2, 4))
        self.assertEqual(tm._elastic_shrink_retiring, {2, 3})
        self.assertFalse(tm._elastic_shrink_pause_event.is_set())
        TokenizerManager._open_elastic_shrink_gate(tm)
        self.assertEqual(tm._elastic_shrink_retiring, set())
        self.assertTrue(tm._elastic_shrink_pause_event.is_set())


class TestRejectedStageReleasesDrainingSlots(CustomTestCase):
    """The DPC stops routing to the tail on a shrink, so an unstarted stage frees it."""

    def test_rejected_stage(self):
        reject = SimpleNamespace(
            success=False,
            old_ep_size=8,
            scale_phase="serving",
            pending_ep_size=None,
            message="rejected",
        )
        exec_ctx = SimpleNamespace(moe=SimpleNamespace(elastic_ep_scale_timeout=60))
        for target, clears in ((6, 1), (10, 0)):
            with self.subTest(target=target):
                tm = _gate(closed=False, retiring=())
                vars(tm).update(
                    _elastic_ep_size=8,
                    attn_replica_size=1,
                    elastic_worker_count=8,
                    elastic_operation_id=None,
                    scale_elastic_ep_communicator=AsyncMock(return_value=[reject]),
                    _clear_staged_scale=Mock(),
                    _clear_elastic_draining=Mock(),
                )
                for name in ("_close_elastic_shrink_gate", "_open_elastic_shrink_gate"):
                    setattr(tm, name, MethodType(getattr(TokenizerManager, name), tm))
                with patch.object(tokenizer_manager, "get_exec", return_value=exec_ctx):
                    asyncio.new_event_loop().run_until_complete(
                        TokenizerManager._drive_next_elastic_stage(tm, target)
                    )
                self.assertEqual(tm._clear_elastic_draining.call_count, clears)
                self.assertEqual(tm.elastic_scale_phase, "failed")
                self.assertTrue(tm._elastic_shrink_pause_event.is_set())


if __name__ == "__main__":
    unittest.main()
