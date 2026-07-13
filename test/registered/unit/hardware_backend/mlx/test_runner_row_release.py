"""Regression guard: MLX runner must never write a finished request's KV into
a reused ``ReqToTokenPool`` row.

The bug: :class:`MlxModelRunner` keeps a ``req_id -> req_pool_idx`` mapping and
reads slot IDs out of ``req_to_token[req_pool_idx, ...]`` when flushing decode
KV to the shared attention pool. The scheduler frees and *reuses* that row once
a request finishes, but the runner used to read it from two late paths:

  * ``remove_request`` — runs during stale-rid cleanup, one batch after the row
    was freed;
  * ``flush_decode_kv_for_slots`` — iterates every live cache, including a
    finished-but-not-yet-removed request whose row was already reused.

Concretely (reviewer's repro, one-row pool): request A has ``synced_offset=2``
and cache offset 4; the scheduler frees A's row and request C reuses it with
slots ``[20, 21, 22, 23, 24]``; a flush for C's prefix then read ``req_to_token``
under A's stale ``req_pool_idx`` and synced A's KV into C's slots ``[22, 23]`` —
corrupting C's prefix right before C reads it.

The fix makes the pre-release hook (``sync_and_release_request``) the final sync
point while the row is still valid, marks the request released, and turns
``remove_request`` into a pure discard. These tests pin all three behaviours.

Model-free: the runner is built with ``__new__`` and ``_sync_new_kv_to_pool``
(the actual pool writer) is spied, so no model or GPU is needed. Apple-Silicon /
mlx gated because the module imports ``mlx.core`` at load.
"""

from __future__ import annotations

import importlib.util
import platform
import unittest
from types import SimpleNamespace

import torch

from sglang.test.ci.ci_register import register_cpu_ci, register_mlx_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
register_mlx_ci(est_time=5, suite="stage-a-unit-test-mlx")

_IS_APPLE_SILICON = platform.system() == "Darwin" and platform.machine() == "arm64"
_HAS_MLX = importlib.util.find_spec("mlx") is not None
_SKIP_REASON = "Apple-Silicon-only (model_runner imports mlx.core at module load)"


@unittest.skipUnless(_IS_APPLE_SILICON and _HAS_MLX, _SKIP_REASON)
class TestMlxRunnerRowRelease(unittest.TestCase):
    """Lifecycle: finish/release A, reallocate its row to C, never corrupt C."""

    def _runner(self, row_slots):
        """Build a bare runner over a one-row pool holding ``row_slots``.

        ``_sync_new_kv_to_pool`` (the actual writer) is replaced with a spy that
        records ``(cache_start, slot_ids)`` so tests can assert exactly which
        pool slots would have been written.
        """
        from sglang.srt.hardware_backend.mlx.model_runner import MlxModelRunner

        runner = MlxModelRunner.__new__(MlxModelRunner)
        runner.disable_radix_cache = False
        runner._attention_kv_pool = object()  # non-None: enables sync paths
        runner._req_to_token_pool = SimpleNamespace(
            req_to_token=torch.tensor([list(row_slots)], dtype=torch.long)
        )
        runner._cache_layout = SimpleNamespace(
            first_attention_layer_index=0, attention_layer_indices=[0]
        )
        runner._req_caches = {}
        runner._req_token_ids = {}
        runner._req_pool_idx = {}
        runner._req_synced_offset = {}
        runner._req_released = set()
        runner._cache_pool = []
        runner._cache_layout.has_auxiliary_state = False

        writes: list[tuple[int, list[int]]] = []
        runner._sync_new_kv_to_pool = (
            lambda cache, cache_start, slot_ids: writes.append(
                (cache_start, list(slot_ids))
            )
        )
        return runner, writes

    @staticmethod
    def _activate(runner, req_id, req_pool_idx, offset, synced_offset):
        runner._req_caches[req_id] = [SimpleNamespace(offset=offset)]
        runner._req_token_ids[req_id] = [0]
        runner._req_pool_idx[req_id] = req_pool_idx
        runner._req_synced_offset[req_id] = synced_offset

    def _reuse_row(self, runner, req_pool_idx, new_slots):
        """Simulate the scheduler reassigning a freed row to a new request."""
        runner._req_to_token_pool.req_to_token[req_pool_idx] = torch.tensor(
            new_slots, dtype=torch.long
        )

    def test_pre_release_flushes_own_slots_and_marks_released(self):
        """The pre-release hook flushes A's own un-synced slots and releases it."""
        # Row 0 holds A's slots [10, 11, 12, 13]; A synced through offset 2.
        runner, writes = self._runner(row_slots=[10, 11, 12, 13, 99])
        self._activate(runner, "A", req_pool_idx=0, offset=4, synced_offset=2)

        runner.sync_and_release_request("A")

        # Flushed exactly A's un-synced tail [12, 13] at cache_start 2.
        self.assertEqual(writes, [(2, [12, 13])])
        self.assertEqual(runner._req_synced_offset["A"], 4)
        self.assertIn("A", runner._req_released)

    def test_flush_after_row_reuse_never_writes_into_reused_slots(self):
        """Flushing for C's prefix must not sync A's KV into C's reused row."""
        runner, writes = self._runner(row_slots=[10, 11, 12, 13, 99])
        self._activate(runner, "A", req_pool_idx=0, offset=4, synced_offset=2)

        # Correct lifecycle: final sync while the row still belongs to A.
        runner.sync_and_release_request("A")
        writes.clear()

        # Scheduler frees A's row; C reuses it. A is still in _req_caches until
        # stale-rid cleanup runs on a later batch.
        self._reuse_row(runner, 0, [20, 21, 22, 23, 24])

        runner.flush_decode_kv_for_slots({20, 21, 22, 23})

        # A is released -> the flush skips it. Nothing is written into C's row.
        self.assertEqual(writes, [])

    def test_remove_request_never_reads_reused_row(self):
        """remove_request on a request whose row was reused writes nothing.

        Covers the retraction shape too: A is *not* released (no pre-release
        hook), its row is already reused by C, and cleanup calls remove_request.
        Pre-fix code synced here and corrupted C's slots [22, 23]; the fix makes
        remove_request a pure discard.
        """
        runner, writes = self._runner(row_slots=[10, 11, 12, 13, 99])
        self._activate(runner, "A", req_pool_idx=0, offset=4, synced_offset=2)

        self._reuse_row(runner, 0, [20, 21, 22, 23, 24])

        runner.remove_request("A")

        self.assertEqual(writes, [])
        # State fully discarded.
        self.assertNotIn("A", runner._req_caches)
        self.assertNotIn("A", runner._req_pool_idx)
        self.assertNotIn("A", runner._req_synced_offset)
        self.assertNotIn("A", runner._req_released)

    def test_released_request_removed_after_flush_leaves_c_intact(self):
        """End-to-end: A can never write into C's slots across its whole teardown."""
        runner, writes = self._runner(row_slots=[10, 11, 12, 13, 99])
        self._activate(runner, "A", req_pool_idx=0, offset=4, synced_offset=2)
        # C is a live request on the same row after reuse.
        self.assertIsNotNone(runner)

        runner.sync_and_release_request("A")  # final sync (row still A's)
        self._reuse_row(runner, 0, [20, 21, 22, 23, 24])  # C reuses the row
        self._activate(runner, "C", req_pool_idx=0, offset=5, synced_offset=5)

        runner.flush_decode_kv_for_slots({20, 21, 22, 23})  # C prefix flush
        runner.remove_request("A")  # stale-rid cleanup for A

        c_slots = {20, 21, 22, 23, 24}
        for _cache_start, slot_ids in writes:
            self.assertTrue(
                c_slots.isdisjoint(slot_ids),
                msg=f"A wrote into C's slots: {slot_ids}",
            )


if __name__ == "__main__":
    unittest.main()
