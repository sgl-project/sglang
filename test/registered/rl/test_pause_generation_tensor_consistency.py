"""
Unit test for the pause_generation.
"""

import unittest

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")


# ---------------------------------------------------------------------------
# Minimal stand-alone simulation of the relevant ScheduleBatch logic.
# We do NOT import ScheduleBatch directly because that pulls in heavy
# GPU-extension dependencies (deep_gemm, etc.).  Instead we replicate the
# exact behaviour of filter_batch / merge_batch / is_empty that matters for
# this bug.
# ---------------------------------------------------------------------------


class _FakeReq:
    def __init__(self, finished: bool = False):
        self._finished = finished

    def finished(self) -> bool:
        return self._finished


class _FakeBatch:
    """Minimal simulation of the scheduler-side fields touched by this bug."""

    def __init__(self, n: int, all_finished: bool = False):
        self.reqs = [_FakeReq(finished=all_finished) for _ in range(n)]
        self.seq_lens = torch.ones(n, dtype=torch.int32)
        self.seq_lens_cpu = torch.ones(n, dtype=torch.int32)
        self.orig_seq_lens = torch.ones(n, dtype=torch.int32)
        self.req_pool_indices = torch.zeros(n, dtype=torch.int64)
        self.output_ids = torch.zeros(n, dtype=torch.int64)
        self.seq_lens_sum = n

    def is_empty(self) -> bool:
        return len(self.reqs) == 0

    def filter_batch(self):
        """Simplified filter_batch: identical early-return logic to ScheduleBatch."""
        keep_indices = [i for i in range(len(self.reqs)) if not self.reqs[i].finished()]

        # Early-return paths — tensors are NOT updated.
        if len(keep_indices) == 0:
            self.reqs = []
            return
        if len(keep_indices) == len(self.reqs):
            return

        # Full filter path (not needed for this test but included for completeness).
        self.reqs = [self.reqs[i] for i in keep_indices]
        idx = torch.tensor(keep_indices, dtype=torch.int64)
        self.seq_lens = self.seq_lens[idx]
        self.seq_lens_cpu = self.seq_lens_cpu[idx]
        self.orig_seq_lens = self.orig_seq_lens[idx]
        self.req_pool_indices = self.req_pool_indices[idx]
        if self.output_ids is not None:
            self.output_ids = self.output_ids[idx]
        self.seq_lens_sum = int(self.seq_lens.sum().item())

    def merge_batch(self, other: "_FakeBatch"):
        """Simplified merge_batch: replicates the tensor-cat logic."""
        self.seq_lens = torch.cat([self.seq_lens, other.seq_lens])
        self.seq_lens_cpu = torch.cat([self.seq_lens_cpu, other.seq_lens_cpu])
        self.orig_seq_lens = torch.cat([self.orig_seq_lens, other.orig_seq_lens])
        self.req_pool_indices = torch.cat(
            [self.req_pool_indices, other.req_pool_indices]
        )
        if self.output_ids is not None and other.output_ids is not None:
            self.output_ids = torch.cat([self.output_ids, other.output_ids])
        self.seq_lens_sum += other.seq_lens_sum
        self.reqs.extend(other.reqs)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPauseGenerationTensorConsistency(CustomTestCase):
    """Verify pause_generation does not corrupt the running_batch tensors."""

    # ------------------------------------------------------------------
    # Bug reproduction
    # ------------------------------------------------------------------

    def test_buggy_merge_violates_invariant(self):
        """Without the fix, merging an all-finished extend batch breaks the
        invariant ``len(reqs) == seq_lens.shape[0]``."""
        N = 651
        running_batch = _FakeBatch(N)
        last_batch = _FakeBatch(1, all_finished=True)

        # Pre-fix pause_generation path:
        # filter_batch -> reqs=[], tensors unchanged (early return)
        last_batch.filter_batch()
        self.assertTrue(last_batch.is_empty())
        # Tensors still have M=1 element each despite reqs being empty.
        self.assertEqual(last_batch.seq_lens.shape[0], 1)

        # BUG: unconditional merge
        running_batch.merge_batch(last_batch)

        # Invariant is now violated.
        self.assertEqual(len(running_batch.reqs), N)
        self.assertEqual(running_batch.seq_lens.shape[0], N + 1)
        self.assertNotEqual(
            len(running_batch.reqs),
            running_batch.seq_lens.shape[0],
            "len(reqs) != seq_lens.shape[0] — invariant broken",
        )

    # ------------------------------------------------------------------
    # Fix verification
    # ------------------------------------------------------------------

    def test_fix_preserves_invariant_when_all_reqs_finished(self):
        """With the is_empty() guard the merge is skipped and invariant holds."""
        N = 651
        running_batch = _FakeBatch(N)
        last_batch = _FakeBatch(1, all_finished=True)

        last_batch.filter_batch()  # reqs=[], tensors untouched

        # FIX: mirror get_next_batch_to_run's is_empty() guard
        if not last_batch.is_empty():
            if running_batch.is_empty():
                running_batch = last_batch
            else:
                running_batch.merge_batch(last_batch)

        self.assertEqual(
            len(running_batch.reqs),
            running_batch.seq_lens.shape[0],
            "Invariant preserved: len(reqs) == seq_lens.shape[0]",
        )
        self.assertEqual(len(running_batch.reqs), N)
        self.assertEqual(running_batch.seq_lens.shape[0], N)

    def test_fix_still_merges_partial_extend_batch(self):
        """The fix must not skip a merge when some extend requests survive."""
        N = 651
        running_batch = _FakeBatch(N)

        # 3-req extend batch: 1 finished, 2 still running
        last_batch = _FakeBatch(3, all_finished=False)
        last_batch.reqs[0] = _FakeReq(finished=True)

        last_batch.filter_batch()  # keeps 2 running reqs

        self.assertEqual(len(last_batch.reqs), 2)
        self.assertFalse(last_batch.is_empty())

        if not last_batch.is_empty():
            if running_batch.is_empty():
                running_batch = last_batch
            else:
                running_batch.merge_batch(last_batch)

        self.assertEqual(len(running_batch.reqs), N + 2)
        self.assertEqual(running_batch.seq_lens.shape[0], N + 2)

    def test_fix_handles_empty_running_batch(self):
        """When running_batch is empty and last_batch has live reqs, the fix
        replaces running_batch (matches get_next_batch_to_run semantics)."""
        running_batch = _FakeBatch(0)
        last_batch = _FakeBatch(3, all_finished=False)

        last_batch.filter_batch()  # all 3 alive -> no-op

        if not last_batch.is_empty():
            if running_batch.is_empty():
                running_batch = last_batch
            else:
                running_batch.merge_batch(last_batch)

        self.assertEqual(len(running_batch.reqs), 3)
        self.assertEqual(running_batch.seq_lens.shape[0], 3)

    def test_next_filter_batch_early_return_preserves_inconsistency(self):
        """After the buggy merge, the next filter_batch call returns early
        (because keep_indices covers all N reqs), leaving N+1 tensors behind."""
        N = 651
        running_batch = _FakeBatch(N)
        last_batch = _FakeBatch(1, all_finished=True)

        last_batch.filter_batch()
        running_batch.merge_batch(last_batch)  # BUG path

        # Simulate update_running_batch -> filter_batch: all N reqs still alive
        running_batch.filter_batch()

        # Early return: tensors NOT trimmed
        self.assertEqual(len(running_batch.reqs), N)
        self.assertEqual(
            running_batch.seq_lens.shape[0],
            N + 1,
            "seq_lens is still N+1 after the second filter_batch early-return",
        )
        self.assertNotEqual(len(running_batch.reqs), running_batch.seq_lens.shape[0])


if __name__ == "__main__":
    unittest.main()
