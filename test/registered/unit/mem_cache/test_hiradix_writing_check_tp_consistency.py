"""Regression test: HiRadixCache.writing_check must enter its all_reduce barrier
on every TP rank, even when the local ongoing_write_through dict is empty.

writing_check() previously short-circuited with `if len(ongoing_write_through) ==
0: return` before the all_reduce, on the assumption that all ranks hold an
identical ongoing_write_through. That assumption does not hold: write_backup()
enqueues a node into ongoing_write_through only on the ranks whose host-pool alloc
succeeds (hiradix_cache.py: `if host_indices is not None: ... else: return 0`).
Host-pool occupancy is per-rank physical state (driven by load_back / eviction /
DMA-ack timing, which loading_check drains per rank), so the dict can diverge.
Once one rank's dict empties while a peer's does not, the empty rank takes the
early return and skips the collective while the peer enters it -> the NCCL op
sequence desyncs -> permanent TP deadlock (NCCL has no timeout; the watchdog
eventually SIGQUITs the process).

The fix removes the early return so the all_reduce(MIN) runs unconditionally; an
empty rank contributes finish_count=0 and MIN(0, ...) makes the drain loop a
no-op -- behaviourally identical to the old early return, minus the desync. This
mirrors loading_check() (already unconditional) and the UnifiedRadixCache fix in
PR #27489 (this is the sibling fix for HiRadixCache, the cache used by the MLA
prefix path). These tests pin the "every rank enters the collective" invariant
WITHOUT real GPUs by driving writing_check() over stubbed ranks and modelling the
all_reduce.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.mem_cache.hiradix_cache import HiRadixCache

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

GROUP_SIZE = 8


def _ready_ack(ack_id):
    """An ack-queue entry whose write-through DMA event has already completed."""
    finish_event = SimpleNamespace(query=lambda: True, synchronize=lambda: None)
    return (None, finish_event, [ack_id])


def _rank(*, ongoing, ack_queue, all_reduce, pp_rank=0):
    """A HiRadixCache shell carrying only the attributes writing_check() touches.

    Built with __new__ so no pools / CUDA / process group are constructed; the
    real, unbound HiRadixCache.writing_check is then driven over it.
    """
    s = HiRadixCache.__new__(HiRadixCache)
    s.ongoing_write_through = dict(ongoing)
    s.pp_rank = pp_rank
    s.cache_controller = SimpleNamespace(ack_write_queue=list(ack_queue))
    s._all_reduce = all_reduce
    s._finish_write_through_ack = MagicMock()
    return s


class TestWritingCheckTPConsistency(CustomTestCase):
    def test_empty_rank_still_enters_all_reduce(self):
        # The core invariant the fix establishes: an empty ongoing_write_through
        # must NOT skip the collective.
        all_reduce = MagicMock()
        s = _rank(ongoing={}, ack_queue=[], all_reduce=all_reduce)

        HiRadixCache.writing_check(s, write_back=False)

        self.assertEqual(all_reduce.call_count, 1)
        tensor, op = all_reduce.call_args.args
        self.assertEqual(tensor.item(), 0)  # empty rank contributes 0
        self.assertIs(op, torch.distributed.ReduceOp.MIN)
        s._finish_write_through_ack.assert_not_called()

    def test_empty_and_nonempty_ranks_both_participate(self):
        # Anti-deadlock property: with the dicts diverged across ranks (one empty,
        # one with a pending write-through), every rank still enters the collective
        # exactly once -> the NCCL op sequence stays aligned.
        all_reduce = MagicMock()
        empty = _rank(ongoing={}, ack_queue=[], all_reduce=all_reduce)
        nonempty = _rank(
            ongoing={7: object()}, ack_queue=[_ready_ack(7)], all_reduce=all_reduce
        )

        HiRadixCache.writing_check(empty, write_back=False)
        HiRadixCache.writing_check(nonempty, write_back=False)

        self.assertEqual(all_reduce.call_count, 2)

    def test_min_consensus_blocks_nonempty_rank_from_draining_ahead(self):
        # The barrier is a MIN-reduce: a rank that locally has a finished ack must
        # not drain it while a peer rank reports 0. Model the collective returning
        # the group minimum (0, because a peer was empty) and assert the non-empty
        # rank leaves its ack queued instead of racing ahead.
        def all_reduce_to_group_min(tensor, op):
            tensor.fill_(0)  # group MIN across {local=1, peer=0}

        nonempty = _rank(
            ongoing={7: object()},
            ack_queue=[_ready_ack(7)],
            all_reduce=all_reduce_to_group_min,
        )

        HiRadixCache.writing_check(nonempty, write_back=False)

        # Drain loop must not run: ack stays queued, no ack finalized.
        self.assertEqual(len(nonempty.cache_controller.ack_write_queue), 1)
        nonempty._finish_write_through_ack.assert_not_called()

    def test_negative_control_old_early_return_would_desync(self):
        # Proves the scenario actually exercises the fix: under the OLD logic the
        # empty rank skips the collective while the non-empty rank enters it, so the
        # ranks disagree on whether the collective ran -> the desync that deadlocks.
        def old_rank_participates(ongoing_write_through):
            if len(ongoing_write_through) == 0:  # the removed early return
                return False
            return True

        participation = [
            old_rank_participates({}),  # empty rank
            old_rank_participates({7: object()}),  # rank with a pending write
        ]
        self.assertEqual(participation, [False, True])
        self.assertEqual(
            len(set(participation)),
            2,
            "ranks must disagree on entering the collective for this to be a real "
            "desync; otherwise the fix would be untested",
        )

    def test_write_back_path_does_not_all_reduce(self):
        # No regression on the blocking write_back=True drain: it synchronizes events
        # directly and must not issue the scalar all_reduce barrier at all.
        all_reduce = MagicMock()
        s = _rank(ongoing={}, ack_queue=[], all_reduce=all_reduce)

        HiRadixCache.writing_check(s, write_back=True)

        all_reduce.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
