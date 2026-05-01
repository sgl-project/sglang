"""Regression hybrid SWA batch full reset.

Adds is_hybrid_swa to the predicate gating batch_is_full reset in
_get_new_batch_prefill_raw, so a stale True doesn't keep prefills out
after SWA cache pressure eases.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.scheduler import Scheduler


def _make_scheduler_stub(
    *,
    enable_priority_preemption: bool,
    is_hybrid_swa: bool,
    initial_batch_is_full: bool = True,
):
    # Stubs only the attrs touched before the early-return at
    # `batch_is_full or len(waiting_queue)==0`. Any new attr access added
    # before that point will surface as AttributeError, not a silent pass.
    sched = Scheduler.__new__(Scheduler)
    sched.enable_priority_preemption = enable_priority_preemption
    sched.is_hybrid_swa = is_hybrid_swa
    sched.enable_hierarchical_cache = False
    sched.grammar_manager = MagicMock()
    sched.grammar_manager.has_waiting_grammars = MagicMock(return_value=False)
    sched.running_batch = SimpleNamespace(batch_is_full=initial_batch_is_full)
    sched.waiting_queue = []
    sched.chunked_req = None
    return sched


def _invoke_prefill_raw(sched):
    return Scheduler._get_new_batch_prefill_raw(sched, prefill_delayer_single_pass=None)


class TestSchedulerHybridSWABatchFullReset(CustomTestCase):
    def test_neither_keeps_batch_is_full_true(self):
        sched = _make_scheduler_stub(
            enable_priority_preemption=False, is_hybrid_swa=False
        )
        self.assertIsNone(_invoke_prefill_raw(sched))
        self.assertTrue(sched.running_batch.batch_is_full)

    def test_hybrid_swa_resets_batch_is_full(self):
        sched = _make_scheduler_stub(
            enable_priority_preemption=False, is_hybrid_swa=True
        )
        _invoke_prefill_raw(sched)
        self.assertFalse(sched.running_batch.batch_is_full)

    def test_priority_preemption_alone_still_resets(self):
        sched = _make_scheduler_stub(
            enable_priority_preemption=True, is_hybrid_swa=False
        )
        _invoke_prefill_raw(sched)
        self.assertFalse(sched.running_batch.batch_is_full)

    def test_both_flags_set_resets(self):
        sched = _make_scheduler_stub(
            enable_priority_preemption=True, is_hybrid_swa=True
        )
        _invoke_prefill_raw(sched)
        self.assertFalse(sched.running_batch.batch_is_full)

    def test_initial_false_unchanged(self):
        sched = _make_scheduler_stub(
            enable_priority_preemption=False,
            is_hybrid_swa=True,
            initial_batch_is_full=False,
        )
        _invoke_prefill_raw(sched)
        self.assertFalse(sched.running_batch.batch_is_full)


if __name__ == "__main__":
    unittest.main()
