"""Regression guard: MLX runner must never write a request's KV into pool
slots it does not own — neither an uncommitted (chained-ahead) row tail nor a
reused ``ReqToTokenPool`` row.

Background. :class:`MlxModelRunner` keeps a ``req_id -> req_pool_idx`` mapping
and reads slot IDs out of ``req_to_token[req_pool_idx, synced:end]`` when
flushing decode KV to the shared attention pool. Two independent failure modes
were reported on this PR:

1. **Uncommitted tail (chained decode).** ``end`` used to be ``cache.offset``,
   the private-cache write cursor. Overlap-chained decode advances that cursor
   for a step the scheduler has not yet committed to ``req_to_token``, so the
   tail cells are the padding slot ``0`` (or, on a reused row, another
   request's slots). The reviewer reproduced row ``[1, 2, 0, 0, 0]`` with
   ``synced=2`` and cache offset ``5`` writing into pool slot ``0``. Fix: clamp
   the read to the scheduler-committed length (``_req_committed_len``).

2. **Reused row (retraction).** Retraction frees the row via
   ``release_kv_cache(..., is_insert=False)`` without the pre-release hook, so
   the runner keeps the request with a stale ``req_pool_idx``. Before stale-rid
   cleanup runs (next decode batch), a prefill/extend batch's
   ``flush_decode_kv_for_slots`` reads the reused row and syncs the old tail
   into the new owner's slots. Fix: a row-generation ownership check
   (``ReqToTokenPool.req_generation`` bumps on realloc) rejects a reused row.

Model-free: the runner is built with ``__new__`` and ``_sync_new_kv_to_pool``
(the actual pool writer) is spied. Apple-Silicon / mlx gated because the module
imports ``mlx.core`` at load.
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
    """Ownership: A must never sync KV into slots it does not own."""

    def _runner(self, row_slots):
        """Bare runner over a one-row pool holding ``row_slots``.

        ``_sync_new_kv_to_pool`` (the actual writer) is replaced with a spy
        recording ``(cache_start, slot_ids)`` so tests assert exactly which pool
        slots would be written. The pool exposes ``req_generation`` (row 0) so
        the ownership check is exercised for real.
        """
        from sglang.srt.hardware_backend.mlx.model_runner import MlxModelRunner

        runner = MlxModelRunner.__new__(MlxModelRunner)
        runner.disable_radix_cache = False
        runner._attention_kv_pool = object()  # non-None: enables sync paths
        runner._req_to_token_pool = SimpleNamespace(
            req_to_token=torch.tensor([list(row_slots)], dtype=torch.long),
            req_generation=torch.zeros(1, dtype=torch.int64),
        )
        runner._cache_layout = SimpleNamespace(
            first_attention_layer_index=0, attention_layer_indices=[0]
        )
        runner._req_caches = {}
        runner._req_token_ids = {}
        runner._req_pool_idx = {}
        runner._req_synced_offset = {}
        runner._req_committed_len = {}
        runner._req_row_generation = {}
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
    def _activate(runner, req_id, *, offset, synced, committed, pool_idx=0):
        """Register an active request, recording the row's current generation."""
        runner._req_caches[req_id] = [SimpleNamespace(offset=offset)]
        runner._req_token_ids[req_id] = [0]
        runner._req_pool_idx[req_id] = pool_idx
        runner._req_synced_offset[req_id] = synced
        runner._req_committed_len[req_id] = committed
        runner._req_row_generation[req_id] = int(
            runner._req_to_token_pool.req_generation[pool_idx].item()
        )

    def _reuse_row(self, runner, new_slots, *, pool_idx=0):
        """Simulate the scheduler reallocating a freed row to a new request.

        Writes the new owner's slots and bumps the row generation, exactly as
        ``ReqToTokenPool.alloc`` does on realloc.
        """
        runner._req_to_token_pool.req_to_token[pool_idx] = torch.tensor(
            new_slots, dtype=torch.long
        )
        runner._req_to_token_pool.req_generation[pool_idx] += 1

    # ---------- comment 1: uncommitted (chained-ahead) tail ----------

    def test_pre_release_flushes_only_committed_positions(self):
        """The pre-release flush syncs A's committed, un-synced tail only."""
        # Row committed only up to 4; A synced through 2.
        runner, writes = self._runner(row_slots=[10, 11, 12, 13, 99])
        self._activate(runner, "A", offset=4, synced=2, committed=4)

        runner.sync_and_release_request("A")

        self.assertEqual(writes, [(2, [12, 13])])
        self.assertEqual(runner._req_synced_offset["A"], 4)

    def test_chained_offset_past_committed_never_writes_padding_slot(self):
        """cache.offset ahead of the committed row must not sync uncommitted cells.

        Reviewer's repro: row [1..2, 0, 0, 0] (committed 2), synced 2, cache
        offset 5. Pre-fix this read positions [2:5] = [0, 0, 0] and wrote into
        padding slot 0. The committed clamp makes it a no-op.
        """
        runner, writes = self._runner(row_slots=[1, 2, 0, 0, 0])
        self._activate(runner, "A", offset=5, synced=2, committed=2)

        runner.sync_and_release_request("A")

        # Nothing beyond the committed length (2) is read, so slot 0 is untouched.
        self.assertEqual(writes, [])

    def test_chained_offset_syncs_committed_but_not_uncommitted_tail(self):
        """Committed positions still sync; the chained-ahead tail never does."""
        # Committed 2 (slots 1, 2); positions 2..4 are the padding slot 0.
        runner, writes = self._runner(row_slots=[1, 2, 0, 0, 0])
        self._activate(runner, "A", offset=5, synced=0, committed=2)

        runner.sync_and_release_request("A")

        self.assertEqual(writes, [(0, [1, 2])])
        for _start, slot_ids in writes:
            self.assertNotIn(0, slot_ids)  # padding slot never written

    # ---------- comment 2: reused row (retraction) ----------

    def test_retraction_flush_before_removal_skips_reused_row(self):
        """flush after retraction + row reuse, BEFORE remove_request, writes nothing.

        This is the exact window the reviewer flagged: retraction bypasses the
        pre-release hook, so A is still active with a stale req_pool_idx when a
        later prefill batch flushes. The row generation no longer matches, so
        the ownership check rejects the read.
        """
        runner, writes = self._runner(row_slots=[10, 11, 12, 13, 99])
        # A retracted: NOT released (no pre-release hook), still has un-synced KV.
        self._activate(runner, "A", offset=4, synced=2, committed=4)

        # Scheduler frees A's row and reuses it for C.
        self._reuse_row(runner, [20, 21, 22, 23, 24])

        # Prefill batch flushes for C's prefix, before stale-rid cleanup removes A.
        runner.flush_decode_kv_for_slots({20, 21, 22, 23})

        self.assertEqual(writes, [])  # generation mismatch -> A skipped

        # Cleanup on the next decode batch discards A without reading the row.
        runner.remove_request("A")
        self.assertEqual(writes, [])
        self.assertNotIn("A", runner._req_caches)

    def test_finish_then_reuse_then_flush_leaves_c_intact(self):
        """End-to-end finish path: A can never write into C's reused slots."""
        runner, writes = self._runner(row_slots=[10, 11, 12, 13, 99])
        self._activate(runner, "A", offset=4, synced=2, committed=4)

        runner.sync_and_release_request("A")  # final sync while row is A's
        writes.clear()

        self._reuse_row(runner, [20, 21, 22, 23, 24])  # C reuses the row
        self._activate(runner, "C", offset=5, synced=5, committed=5)

        runner.flush_decode_kv_for_slots({20, 21, 22, 23})  # C prefix flush
        runner.remove_request("A")  # stale-rid cleanup for A

        c_slots = {20, 21, 22, 23, 24}
        for _start, slot_ids in writes:
            self.assertTrue(
                c_slots.isdisjoint(slot_ids),
                msg=f"A wrote into C's slots: {slot_ids}",
            )

    def test_remove_request_never_reads_the_row(self):
        """remove_request is a pure discard: it never syncs, even on a reused row."""
        runner, writes = self._runner(row_slots=[10, 11, 12, 13, 99])
        self._activate(runner, "A", offset=4, synced=2, committed=4)
        self._reuse_row(runner, [20, 21, 22, 23, 24])

        runner.remove_request("A")

        self.assertEqual(writes, [])
        self.assertNotIn("A", runner._req_pool_idx)
        self.assertNotIn("A", runner._req_committed_len)
        self.assertNotIn("A", runner._req_row_generation)


if __name__ == "__main__":
    unittest.main()
