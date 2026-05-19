"""Regression test for the CpuPageTracker tensor-view pickle bug.

Background: ``CpuPageTracker.free`` queues freed slot-index tensors for the
GPU worker via ``ModelWorkerBatch.indices_to_free``. The callers
(radix_cache cache_finished_req / cache_unfinished_req) pass *views* of
``req_to_token_pool.req_to_token`` — a [4096, 32772] int32 tensor (~512 MB).

Before the fix, ``free`` stored the view as-is; pickling later serialized
the *full backing storage*, adding 524 MB per prefill batch and pushing
TTFT from ~600 ms to >2 s. The fix clones the slice so it owns a small
private storage.

This test simulates that exact bug. If anyone reintroduces the regression
(e.g. removes the .clone() or .to("cpu", copy=True) flag), this test fails.
"""

import pickle
import unittest

import torch

from sglang.srt.managers.scheduler_cpu import CpuPageTracker


class TestCpuPageTrackerPickleSize(unittest.TestCase):
    def test_freed_indices_do_not_drag_parent_storage(self):
        # Simulate the production layout: a large parent tensor (req_to_token
        # at production sizes) whose individual rows are sliced into views and
        # then passed to free().
        parent = torch.zeros(4096, 32768, dtype=torch.int32)
        parent_storage_bytes = parent.numel() * parent.element_size()
        # Sanity: the parent really is ~512 MB. If this ever shrinks below
        # ~16 MB the test loses its meaning.
        self.assertGreater(parent_storage_bytes, 16 * 1024 * 1024)

        tracker = CpuPageTracker(total_pages=parent.shape[0], page_size=16)

        # Free a SINGLE row-slice view — exactly what radix_cache does on a
        # typical decode step. The view shares storage with ``parent``:
        # ``view.storage().nbytes()`` reflects the full parent (~512 MB), and
        # naive pickling would carry all of it.
        #
        # Multiple-tensor case is automatically safe (drain calls torch.cat,
        # which materializes a fresh tensor); the single-tensor case is the
        # actual hot path AND the path that was broken.
        row_view = parent[0, :256]
        self.assertGreater(
            row_view.untyped_storage().nbytes(),
            parent_storage_bytes // 2,
            msg="row_view should still reference the parent's full storage",
        )
        tracker.free(row_view)

        drained = tracker.drain_pending_free()
        self.assertIsNotNone(drained)

        pickle_size = len(pickle.dumps(drained))

        # The freed indices total 256 int32 elements ≈ 1 KB plus pickle
        # overhead. Anything above 64 KB means the parent storage came
        # along for the ride.
        self.assertLess(
            pickle_size,
            64 * 1024,
            msg=(
                f"drain_pending_free pickled to {pickle_size:,} bytes — likely "
                f"a tensor-view storage leak. Parent storage is "
                f"{parent_storage_bytes:,} bytes."
            ),
        )


if __name__ == "__main__":
    unittest.main()
