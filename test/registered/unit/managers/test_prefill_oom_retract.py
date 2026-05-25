"""Tests for graceful prefill KV-OOM handling.

Covers two surfaces introduced when the scheduler stopped fail-stopping
on transient prefill KV-pool exhaustion:

1. ``alloc_for_extend`` raises ``KVPoolOOMError`` on KV exhaustion and
   releases the freshly-allocated req_pool slots before doing so, so
   the scheduler's catch path starts from a clean state.
2. ``Scheduler._retract_prefill_admission`` rolls back lock refs,
   chunked_req state, and the waiting queue so the next scheduler tick
   can retry cleanly.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.mem_cache.base_prefix_cache import DecLockRefParams
from sglang.srt.mem_cache.common import KVPoolOOMError

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _make_admitted_req(rid: str, swa_uuid: object = None) -> SimpleNamespace:
    """Minimal stand-in for a Req that admission has already locked."""
    return SimpleNamespace(
        rid=rid,
        last_node=SimpleNamespace(node_id=rid),
        swa_uuid_for_lock=swa_uuid,
    )


def _scheduler_for_retract(*, is_hybrid_swa: bool, chunked_req=None) -> Scheduler:
    s = Scheduler.__new__(Scheduler)
    s.is_hybrid_swa = is_hybrid_swa
    s.tree_cache = MagicMock()
    s.chunked_req = chunked_req
    s._chunked_req_scheduled_last_iter = False
    s.waiting_queue = []
    s.running_batch = MagicMock()
    s.running_batch.batch_is_full = True
    return s


class TestRetractPrefillAdmissionNonSWA(CustomTestCase):
    def test_dec_lock_ref_called_per_req_without_swa_params(self):
        s = _scheduler_for_retract(is_hybrid_swa=False)
        req_a = _make_admitted_req("a")
        req_b = _make_admitted_req("b")

        s._retract_prefill_admission(
            admitted_reqs=[req_a, req_b],
            pre_admit_chunked_req=None,
            pre_admit_chunked_req_scheduled_last_iter=False,
        )

        # Non-SWA path: dec_lock_ref(node) with no params kwarg.
        self.assertEqual(s.tree_cache.dec_lock_ref.call_count, 2)
        for call in s.tree_cache.dec_lock_ref.call_args_list:
            args, kwargs = call
            self.assertEqual(len(args), 1)  # only the node, no DecLockRefParams
            self.assertEqual(kwargs, {})

    def test_admitted_reqs_returned_to_waiting_queue_head(self):
        s = _scheduler_for_retract(is_hybrid_swa=False)
        existing = SimpleNamespace(rid="existing")
        s.waiting_queue = [existing]
        req_a = _make_admitted_req("a")
        req_b = _make_admitted_req("b")

        s._retract_prefill_admission(
            admitted_reqs=[req_a, req_b],
            pre_admit_chunked_req=None,
            pre_admit_chunked_req_scheduled_last_iter=False,
        )

        self.assertEqual(s.waiting_queue, [req_a, req_b, existing])

    def test_batch_is_full_cleared(self):
        s = _scheduler_for_retract(is_hybrid_swa=False)
        s.running_batch.batch_is_full = True

        s._retract_prefill_admission(
            admitted_reqs=[],
            pre_admit_chunked_req=None,
            pre_admit_chunked_req_scheduled_last_iter=False,
        )

        self.assertFalse(s.running_batch.batch_is_full)


class TestRetractPrefillAdmissionSWA(CustomTestCase):
    def test_dec_lock_ref_passes_swa_uuid_per_req(self):
        s = _scheduler_for_retract(is_hybrid_swa=True)
        req_a = _make_admitted_req("a", swa_uuid="uuid-a")
        req_b = _make_admitted_req("b", swa_uuid="uuid-b")

        s._retract_prefill_admission(
            admitted_reqs=[req_a, req_b],
            pre_admit_chunked_req=None,
            pre_admit_chunked_req_scheduled_last_iter=False,
        )

        self.assertEqual(s.tree_cache.dec_lock_ref.call_count, 2)
        seen_uuids = []
        for call in s.tree_cache.dec_lock_ref.call_args_list:
            args, _ = call
            self.assertEqual(len(args), 2)
            params = args[1]
            self.assertIsInstance(params, DecLockRefParams)
            seen_uuids.append(params.swa_uuid_for_lock)
        self.assertEqual(sorted(seen_uuids), ["uuid-a", "uuid-b"])


class TestRetractPrefillAdmissionChunkedReqRollback(CustomTestCase):
    def test_inflight_decrement_and_chunked_req_restored(self):
        original = SimpleNamespace(rid="original", inflight_middle_chunks=3)
        new_chunked = SimpleNamespace(rid="new", inflight_middle_chunks=1)
        # Mid-cycle state: scheduler swapped chunked_req to new and bumped its
        # inflight_middle_chunks. Retract must walk this back.
        s = _scheduler_for_retract(is_hybrid_swa=False, chunked_req=new_chunked)
        s._chunked_req_scheduled_last_iter = True

        s._retract_prefill_admission(
            admitted_reqs=[],
            pre_admit_chunked_req=original,
            pre_admit_chunked_req_scheduled_last_iter=False,
        )

        # The req that was current at retract time gets its counter rolled back.
        self.assertEqual(new_chunked.inflight_middle_chunks, 0)
        # Original inflight count untouched (we never bumped it this cycle).
        self.assertEqual(original.inflight_middle_chunks, 3)
        # chunked_req identity restored.
        self.assertIs(s.chunked_req, original)
        self.assertFalse(s._chunked_req_scheduled_last_iter)

    def test_no_chunked_req_means_no_decrement(self):
        s = _scheduler_for_retract(is_hybrid_swa=False, chunked_req=None)

        # Should not crash trying to decrement a None chunked_req.
        s._retract_prefill_admission(
            admitted_reqs=[],
            pre_admit_chunked_req=None,
            pre_admit_chunked_req_scheduled_last_iter=False,
        )

        self.assertIsNone(s.chunked_req)

    def test_pre_existing_chunked_req_not_duplicated_in_waiting_queue(self):
        # A pre-existing chunked req is in ``can_run_list`` (admission
        # called ``add_chunked_req`` which appends it). Retract restores
        # ``self.chunked_req`` to that same req. If we also re-queued it
        # at the head of ``waiting_queue``, the next tick would process
        # it through both the chunked-req branch and the waiting-queue
        # loop, causing double-admission.
        chunked = _make_admitted_req("chunked")
        regular = _make_admitted_req("regular")
        s = _scheduler_for_retract(is_hybrid_swa=False, chunked_req=chunked)

        s._retract_prefill_admission(
            admitted_reqs=[chunked, regular],
            pre_admit_chunked_req=chunked,
            pre_admit_chunked_req_scheduled_last_iter=True,
        )

        self.assertIs(s.chunked_req, chunked)
        self.assertEqual(s.waiting_queue, [regular])
        self.assertNotIn(chunked, s.waiting_queue)


class TestAllocForExtendOOMReleasesFreshSlots(CustomTestCase):
    """``alloc_for_extend`` raises ``KVPoolOOMError`` on KV exhaustion
    and releases the freshly-allocated req_pool slots before raising,
    so the scheduler's catch path starts from a clean state."""

    def _make_batch_returning_oom(self):
        import torch

        from sglang.srt.mem_cache.common import alloc_for_extend

        allocator = MagicMock()
        allocator.alloc.return_value = None
        allocator.page_size = 1
        allocator.available_size.return_value = 0

        tree_cache = MagicMock()
        tree_cache.token_to_kv_pool_allocator = allocator
        tree_cache.page_size = 1
        tree_cache.is_chunk_cache.return_value = False
        tree_cache.evict = MagicMock()

        req_to_token_pool = MagicMock()
        req_to_token_pool.alloc.return_value = [10, 11]
        req_to_token_pool.available_size.return_value = 64

        # ``req_pool_idx is None`` marks the req as freshly allocated
        # this cycle; the rollback path frees only those.
        req_a = SimpleNamespace(
            prefix_indices=torch.empty(0, dtype=torch.int64),
            req_pool_idx=None,
            inflight_middle_chunks=0,
            kv_committed_len=0,
        )
        req_b = SimpleNamespace(
            prefix_indices=torch.empty(0, dtype=torch.int64),
            req_pool_idx=None,
            inflight_middle_chunks=0,
            kv_committed_len=0,
        )

        batch = SimpleNamespace(
            reqs=[req_a, req_b],
            tree_cache=tree_cache,
            req_to_token_pool=req_to_token_pool,
            device="cpu",
            prefix_lens=[0, 0],
            extend_lens=[4, 4],
            seq_lens=torch.tensor([4, 4], dtype=torch.int64),
            seq_lens_cpu=torch.tensor([4, 4], dtype=torch.int64),
            extend_num_tokens=8,
            maybe_evict_swa=lambda: None,
        )
        return batch, alloc_for_extend, req_to_token_pool

    def test_raises_kv_pool_oom_and_frees_fresh_req_slots(self):
        batch, alloc_for_extend, req_to_token_pool = self._make_batch_returning_oom()

        with self.assertRaises(KVPoolOOMError):
            alloc_for_extend(batch)

        self.assertEqual(req_to_token_pool.free.call_count, 2)
        freed_reqs = [c.args[0] for c in req_to_token_pool.free.call_args_list]
        self.assertIs(freed_reqs[0], batch.reqs[0])
        self.assertIs(freed_reqs[1], batch.reqs[1])

    def test_kv_pool_oom_is_a_runtime_error(self):
        # Existing ``except RuntimeError`` callers must keep working.
        batch, alloc_for_extend, _ = self._make_batch_returning_oom()

        with self.assertRaises(RuntimeError):
            alloc_for_extend(batch)


if __name__ == "__main__":
    unittest.main()
