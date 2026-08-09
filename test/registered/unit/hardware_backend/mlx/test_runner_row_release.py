"""Regression guard: MLX runner must never write a request's KV into pool
slots it does not own — neither an uncommitted (chained-ahead) row tail nor a
row/KV slot the scheduler freed and handed to another request.
"""

from __future__ import annotations

import importlib.util
import platform
import unittest
from types import SimpleNamespace

import torch

from sglang.test.ci.ci_register import register_cpu_ci, register_mlx_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
register_mlx_ci(est_time=5, suite="stage-a-unit-test-mlx")

_IS_APPLE_SILICON = platform.system() == "Darwin" and platform.machine() == "arm64"
_HAS_MLX = importlib.util.find_spec("mlx") is not None
_SKIP_REASON = "Apple-Silicon-only (model_runner imports mlx.core at module load)"


@unittest.skipUnless(_IS_APPLE_SILICON and _HAS_MLX, _SKIP_REASON)
class TestMlxRunnerRowRelease(CustomTestCase):
    """Ownership: A must never sync KV into slots it does not own."""

    # ---------------- real-object fixtures ----------------

    @staticmethod
    def _new_req():
        """A minimal request object accepted by ReqToTokenPool.alloc/free."""
        return SimpleNamespace(
            req_pool_idx=None, inflight_middle_chunks=0, kv_committed_len=0
        )

    @staticmethod
    def _pools(*, rows: int, kv: int):
        """Build a real ReqToTokenPool + TokenToKVPoolAllocator.

        ``rows=1`` forces a freed row to be reused by the next request (the
        allocator has no other row to hand out); larger pools let a retracted
        request keep its old row while a new request takes a different one.
        """
        from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator
        from sglang.srt.mem_cache.memory_pool import ReqToTokenPool

        req_pool = ReqToTokenPool(
            size=rows, max_context_len=8, device="cpu", enable_memory_saver=False
        )
        kv_allocator = TokenToKVPoolAllocator(
            size=kv, dtype=torch.float32, device="cpu", kvcache=object(), need_sort=True
        )
        return req_pool, kv_allocator

    def _runner(self, req_pool):
        """Bare runner bound to ``req_pool``; ``_sync_new_kv_to_pool`` is spied.

        The spy records ``(cache_start, slot_ids)`` for each flush write so tests
        can assert exactly which pool slots would be touched.
        """
        from sglang.srt.hardware_backend.mlx.model_runner import MlxModelRunner

        runner = MlxModelRunner.__new__(MlxModelRunner)
        runner.disable_radix_cache = False
        runner._attention_kv_pool = object()  # non-None: enables sync paths
        runner._req_to_token_pool = req_pool
        runner._cache_layout = SimpleNamespace(
            first_attention_layer_index=0,
            attention_layer_indices=[0],
            has_auxiliary_state=False,
        )
        runner._req_caches = {}
        runner._req_token_ids = {}
        runner._req_pool_idx = {}
        runner._req_synced_offset = {}
        runner._req_committed_len = {}
        runner._req_row_generation = {}
        runner._cache_pool = []

        writes: list[tuple[int, list[int]]] = []
        runner._sync_new_kv_to_pool = (
            lambda cache, cache_start, slot_ids: writes.append(
                (cache_start, list(slot_ids))
            )
        )
        return runner, writes

    def _alloc(self, req_pool, kv_allocator, n_slots):
        """Allocate a request row + ``n_slots`` KV slots, wiring req_to_token.

        Returns ``(req, row, slots)``. Uncommitted positions in the row stay 0
        (the padding slot), matching a real request before those tokens decode.
        """
        req = self._new_req()
        [row] = req_pool.alloc([req])
        slots = kv_allocator.alloc(n_slots)
        req_pool.req_to_token[row, :n_slots] = slots.to(torch.int32)
        return req, row, slots

    def _activate(self, runner, req_id, *, row, offset, synced, committed):
        """Register an active request, snapshotting the row's live generation.

        ``offset`` (private cache cursor) may exceed ``committed`` to model a
        chained decode that ran ahead of the scheduler's req_to_token commit.
        """
        runner._req_caches[req_id] = [SimpleNamespace(offset=offset)]
        runner._req_token_ids[req_id] = [0]
        runner._req_pool_idx[req_id] = row
        runner._req_synced_offset[req_id] = synced
        runner._req_committed_len[req_id] = committed
        runner._req_row_generation[req_id] = int(
            runner._req_to_token_pool.req_generation[row].item()
        )

    def _worker(self, runner):
        """Bare MlxTpModelWorker bound to ``runner``.

        Only the fields the retraction/routing hooks touch are wired, so the
        tests exercise the real ``prepare_for_retraction`` /
        ``_route_extend_request`` / ``_gather_prefill_prefix_slots`` code rather
        than a stand-in.
        """
        from sglang.srt.hardware_backend.mlx.tp_worker import MlxTpModelWorker

        worker = MlxTpModelWorker.__new__(MlxTpModelWorker)
        worker._mlx_runner = runner
        return worker

    def test_pre_release_flushes_only_committed_positions(self):
        """The pre-release flush syncs A's committed, un-synced tail only."""
        req_pool, kv = self._pools(rows=1, kv=8)
        runner, writes = self._runner(req_pool)
        _req, row, slots = self._alloc(req_pool, kv, 4)  # slots [1, 2, 3, 4]
        self._activate(runner, "A", row=row, offset=4, synced=2, committed=4)

        runner.sync_and_release_request("A")

        # Only the committed, un-synced positions [2:4] -> slots [3, 4].
        self.assertEqual(writes, [(2, slots[2:4].tolist())])
        self.assertEqual(runner._req_synced_offset["A"], 4)

    def test_chained_offset_past_committed_never_writes_padding_slot(self):
        """cache.offset ahead of the committed row must not sync uncommitted cells.

        A committed only 2 positions but a chained decode pushed cache.offset to
        5. Pre-fix this read positions [2:5] (all the padding slot 0) and wrote
        into pool slot 0. The committed clamp makes it a no-op.
        """
        req_pool, kv = self._pools(rows=1, kv=8)
        runner, writes = self._runner(req_pool)
        self._alloc(req_pool, kv, 2)  # committed positions -> slots [1, 2]
        self._activate(runner, "A", row=1, offset=5, synced=2, committed=2)

        runner.sync_and_release_request("A")

        self.assertEqual(writes, [])  # nothing past committed -> slot 0 untouched

    def test_chained_offset_syncs_committed_but_not_uncommitted_tail(self):
        """Committed positions still sync; the chained-ahead tail never does."""
        req_pool, kv = self._pools(rows=1, kv=8)
        runner, writes = self._runner(req_pool)
        _req, row, slots = self._alloc(req_pool, kv, 2)  # slots [1, 2]
        self._activate(runner, "A", row=row, offset=5, synced=0, committed=2)

        runner.sync_and_release_request("A")

        self.assertEqual(writes, [(0, slots.tolist())])
        for _start, slot_ids in writes:
            self.assertNotIn(0, slot_ids)  # padding slot never written

    def test_retraction_flush_before_removal_skips_reused_row(self):
        """Flush after retraction + row reuse, BEFORE remove_request, writes nothing.

        Retraction bypasses the pre-release hook, so A is still active with a
        stale req_pool_idx when a later prefill batch flushes. A single-row pool
        forces the freed row to be reused by C, bumping its generation, so the
        ownership check rejects the read.
        """
        req_pool, kv = self._pools(rows=1, kv=8)
        runner, writes = self._runner(req_pool)
        req_a, row, a_slots = self._alloc(req_pool, kv, 4)
        # A retracted: NOT released (no pre-release hook), still has un-synced KV.
        self._activate(runner, "A", row=row, offset=4, synced=2, committed=4)

        # Scheduler frees A's row + KV, then reuses the same row for C.
        req_pool.free(req_a)
        kv.free(a_slots)
        _req_c, c_row, c_slots = self._alloc(req_pool, kv, 4)
        self.assertEqual(c_row, row)  # single-row pool -> same row, new generation

        runner.flush_decode_kv_for_slots(set(c_slots.tolist()))
        self.assertEqual(writes, [])  # generation mismatch -> A skipped

        # Cleanup on the next decode batch discards A without reading the row.
        runner.remove_request("A")
        self.assertEqual(writes, [])
        self.assertNotIn("A", runner._req_caches)

    def test_retraction_hook_prevents_freed_kv_slot_write(self):
        """The retraction hook closes the different-row / freed-KV-slot gap.

        Retracting A (row 1, slots [1,2,3,4]) frees its row and its un-synced KV
        tail [3,4]; C then takes a *different* row (so A's row generation is
        unchanged) while the KV allocator recycles [3,4]. The row-generation
        check alone cannot see this. ``prepare_for_retraction`` drops A from the
        runner at retraction time, so the later flush never touches its freed
        slots.
        """
        req_pool, kv = self._pools(rows=2, kv=4)
        runner, writes = self._runner(req_pool)
        worker = self._worker(runner)

        req_a, a_row, a_slots = self._alloc(req_pool, kv, 4)
        self._activate(runner, "A", row=a_row, offset=4, synced=2, committed=4)

        # Retraction: the scheduler invokes the hook, then frees the row + the
        # un-synced KV tail [3, 4] (release_kv_cache(..., is_insert=False)).
        worker.prepare_for_retraction(SimpleNamespace(rid="A"))
        req_pool.free(req_a)
        kv.free(a_slots[2:])
        self.assertFalse(runner.has_request("A"))

        # C gets a different row while the KV allocator reuses [3, 4].
        _req_c, c_row, c_slots = self._alloc(req_pool, kv, 2)
        self.assertNotEqual(a_row, c_row)
        self.assertEqual(c_slots.tolist(), [3, 4])

        runner.flush_decode_kv_for_slots(set(c_slots.tolist()))
        self.assertEqual(
            writes, [], "retracted A must not write into C's reused KV slots"
        )

    def test_retraction_reschedules_as_fresh_prefill(self):
        """After the retraction hook, a re-scheduled A routes as prefill.

        Dropping A's runner state flips ``has_request`` to False, so the worker
        classifies a re-scheduled A as ``"prefill"`` (not ``"continuation"``)
        and gathers its prefix slots — i.e. A re-reads its cached prefix from the
        pool instead of extending a now-invalid private cache.
        """
        req_pool, kv = self._pools(rows=2, kv=8)
        runner, _writes = self._runner(req_pool)
        worker = self._worker(runner)

        req_a, a_row, a_slots = self._alloc(req_pool, kv, 4)
        self._activate(runner, "A", row=a_row, offset=4, synced=2, committed=4)

        # Retract A.
        worker.prepare_for_retraction(SimpleNamespace(rid="A"))
        req_pool.free(req_a)
        kv.free(a_slots)
        self.assertFalse(runner.has_request("A"))

        # A is re-scheduled onto a fresh row with a matched prefix.
        _req_a2, a_row2, prefix_slots = self._alloc(req_pool, kv, 2)
        rescheduled = SimpleNamespace(
            rid="A",
            req_pool_idx=a_row2,
            prefix_indices=prefix_slots,
        )

        # Unseen -> routed as prefill, and its prefix slots are gathered.
        self.assertEqual(worker._route_extend_request("A", set()), "prefill")
        self.assertEqual(
            worker._gather_prefill_prefix_slots([rescheduled]),
            set(prefix_slots.tolist()),
        )

    def test_finish_then_reuse_then_flush_leaves_c_intact(self):
        """End-to-end finish path: A can never write into C's reused slots."""
        req_pool, kv = self._pools(rows=1, kv=8)
        runner, writes = self._runner(req_pool)
        req_a, row, a_slots = self._alloc(req_pool, kv, 4)
        self._activate(runner, "A", row=row, offset=4, synced=2, committed=4)

        runner.sync_and_release_request("A")  # final sync while row is A's
        self.assertEqual(writes, [(2, a_slots[2:4].tolist())])
        writes.clear()

        # Scheduler frees A, C reuses the same row.
        req_pool.free(req_a)
        kv.free(a_slots)
        _req_c, c_row, c_slots = self._alloc(req_pool, kv, 4)
        self.assertEqual(c_row, row)
        self._activate(runner, "C", row=c_row, offset=4, synced=4, committed=4)

        runner.flush_decode_kv_for_slots(set(c_slots.tolist()))
        runner.remove_request("A")

        c_set = set(c_slots.tolist())
        for _start, slot_ids in writes:
            self.assertTrue(
                c_set.isdisjoint(slot_ids), msg=f"A wrote into C's slots: {slot_ids}"
            )

    def test_remove_request_never_reads_the_row(self):
        """remove_request is a pure discard: it never syncs, even on a reused row."""
        req_pool, kv = self._pools(rows=1, kv=8)
        runner, writes = self._runner(req_pool)
        req_a, row, a_slots = self._alloc(req_pool, kv, 4)
        self._activate(runner, "A", row=row, offset=4, synced=2, committed=4)

        req_pool.free(req_a)
        kv.free(a_slots)
        self._alloc(req_pool, kv, 4)  # C reuses the row

        runner.remove_request("A")

        self.assertEqual(writes, [])
        self.assertNotIn("A", runner._req_pool_idx)
        self.assertNotIn("A", runner._req_committed_len)
        self.assertNotIn("A", runner._req_row_generation)


if __name__ == "__main__":
    unittest.main()
