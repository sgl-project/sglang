import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.disaggregation.decode import (
    DecodePreallocQueue,
    DecodeReqToTokenPool,
    HybridMambaDecodeReqToTokenPool,
)
from sglang.srt.mem_cache.allocation import alloc_req_slots, ensure_mamba_capacity
from sglang.srt.mem_cache.allocation_sizing import get_mamba_tracking_slots
from sglang.srt.mem_cache.allocator.mamba import MambaSlotAllocator
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool, ReqToTokenPool
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def make_pool(free, *, lazy=False, extra_buffer=True, track_buffer_size=2):
    pool = object.__new__(HybridReqToTokenPool)
    pool.mamba_allocator = MambaSlotAllocator(5, "cpu")
    pool.enable_mamba_extra_buffer = extra_buffer
    pool.enable_mamba_extra_buffer_lazy = lazy
    pool.mamba_ping_pong_track_buffer_size = track_buffer_size
    pool.mamba_initial_tracking_slots = get_mamba_tracking_slots(
        extra_buffer=extra_buffer, overlap=track_buffer_size == 2, lazy=lazy
    )
    held = pool.mamba_allocator.alloc(5 - free)
    pool.available_size = lambda: 1
    pool.alloc = MagicMock(return_value=[1])
    return pool, held


def make_req(*, holds_mamba=False, has_buffer=False):
    return SimpleNamespace(
        kv=SimpleNamespace(
            holds_mamba=holds_mamba,
            mamba_ping_pong_track_buffer=object() if has_buffer else None,
            req_pool_idx=None,
            mamba_pool_idx=object() if holds_mamba else None,
        )
    )


def make_cache(pool, held, release):
    reclaimed = False

    def evict(_):
        nonlocal reclaimed
        if release and not reclaimed:
            pool.mamba_allocator.free(held[:release])
            reclaimed = True

    return SimpleNamespace(
        supports_mamba=lambda: True,
        evict_for_alloc=MagicMock(side_effect=evict),
    )


class TestMambaCapacity(unittest.TestCase):
    def test_admission_matches_tracking_allocation(self):
        for extra_buffer, overlap, lazy, expected in [
            (False, False, False, 0),
            (False, True, False, 0),
            (True, False, False, 1),
            (True, True, False, 2),
            (True, True, True, 1),
        ]:
            with self.subTest(extra_buffer=extra_buffer, overlap=overlap, lazy=lazy):
                # Run the real constructor's sizing, without allocating model tensors.
                with (
                    patch.object(ReqToTokenPool, "__init__", return_value=None),
                    patch.object(HybridReqToTokenPool, "_init_mamba_pool"),
                ):
                    pool = HybridReqToTokenPool(
                        size=1,
                        mamba_size=5,
                        mamba_spec_state_size=1,
                        max_context_len=16,
                        device="cpu",
                        enable_memory_saver=False,
                        cache_params=None,
                        mamba_layer_ids=[],
                        enable_mamba_extra_buffer=extra_buffer,
                        enable_overlap_schedule=overlap,
                        enable_mamba_extra_buffer_lazy=lazy,
                    )
                self.assertEqual(pool.mamba_initial_tracking_slots, expected)
                pool.mamba_allocator = MambaSlotAllocator(1 + expected, "cpu")
                req = make_req()
                held = pool.mamba_allocator.alloc(1)
                self.assertFalse(ensure_mamba_capacity(pool, [req], None))
                pool.mamba_allocator.free(held)
                self.assertTrue(ensure_mamba_capacity(pool, [req], None))

                # Consume the live slot, then allocate tracking buffers using production code.
                pool.mamba_allocator.alloc(1)
                if extra_buffer:
                    pool._alloc_ping_pong_buffer(req)
                    buf = req.kv.mamba_ping_pong_track_buffer
                    self.assertEqual(int((buf >= 0).sum()), expected)
                    if lazy:
                        self.assertEqual(buf.tolist()[1], -1)
                self.assertEqual(pool.mamba_allocator.available_size(), 0)

    def test_pd_pool_sizing_includes_tracking_and_preallocated_requests(self):
        for extra_buffer, overlap, expected_slots_per_req in [
            (False, False, 1),
            (False, True, 1),
            (True, False, 2),
            (True, True, 3),
        ]:
            with self.subTest(extra_buffer=extra_buffer, overlap=overlap):
                with (
                    patch.object(DecodeReqToTokenPool, "__init__", return_value=None),
                    patch.object(HybridReqToTokenPool, "_init_mamba_pool") as init_pool,
                ):
                    pool = HybridMambaDecodeReqToTokenPool(
                        size=2,
                        pre_alloc_size=1,
                        mamba_size=1,
                        max_context_len=16,
                        device="cpu",
                        enable_memory_saver=False,
                        cache_params=None,
                        mamba_layer_ids=[],
                        speculative_num_draft_tokens=None,
                        enable_mamba_extra_buffer=extra_buffer,
                        enable_overlap_schedule=overlap,
                    )
                self.assertEqual(
                    1 + pool.mamba_initial_tracking_slots, expected_slots_per_req
                )
                self.assertEqual(
                    init_pool.call_args.kwargs["mamba_size"],
                    3 * expected_slots_per_req,
                )

    def test_reclaims_full_shortfall_before_allocation(self):
        for free, expected_eviction in [(1, 2), (2, 1), (3, 0)]:
            with self.subTest(free=free):
                pool, held = make_pool(free)
                cache = make_cache(pool, held, expected_eviction)
                self.assertTrue(ensure_mamba_capacity(pool, [make_req()], cache))
                self.assertGreaterEqual(pool.mamba_allocator.available_size(), 3)
                if expected_eviction:
                    self.assertEqual(
                        cache.evict_for_alloc.call_args.args[0].mamba_num,
                        expected_eviction,
                    )
                else:
                    cache.evict_for_alloc.assert_not_called()

    def test_partial_eviction_defers_before_mutating_request(self):
        pool, held = make_pool(1)
        cache = make_cache(pool, held, 1)
        req = make_req()
        self.assertFalse(ensure_mamba_capacity(pool, [req], cache))
        self.assertEqual(pool.mamba_allocator.available_size(), 2)
        self.assertIsNone(req.kv.req_pool_idx)
        self.assertIsNone(req.kv.mamba_pool_idx)
        self.assertEqual(alloc_req_slots(pool, [req], cache), [1])
        pool.alloc.assert_called_once_with([req])

    def test_uses_only_missing_slots_for_each_request(self):
        pool, held = make_pool(2)
        cache = make_cache(pool, held, 1)
        reqs = [make_req(), make_req(holds_mamba=True, has_buffer=True)]
        self.assertTrue(ensure_mamba_capacity(pool, reqs, cache))
        self.assertEqual(cache.evict_for_alloc.call_args.args[0].mamba_num, 1)

        for holds_mamba, has_buffer, free, release, expected_eviction in [
            (True, False, 1, 1, 1),
            (False, True, 1, 0, 0),
            (True, True, 0, 0, 0),
        ]:
            with self.subTest(holds_mamba=holds_mamba, has_buffer=has_buffer):
                pool, held = make_pool(free)
                cache = make_cache(pool, held, release)
                req = make_req(holds_mamba=holds_mamba, has_buffer=has_buffer)
                self.assertTrue(ensure_mamba_capacity(pool, [req], cache))
                if expected_eviction:
                    self.assertEqual(
                        cache.evict_for_alloc.call_args.args[0].mamba_num,
                        expected_eviction,
                    )
                else:
                    cache.evict_for_alloc.assert_not_called()

    def test_tracking_buffer_settings(self):
        pool, held = make_pool(1, lazy=True)
        cache = make_cache(pool, held, 1)
        self.assertTrue(ensure_mamba_capacity(pool, [make_req()], cache))
        self.assertEqual(cache.evict_for_alloc.call_args.args[0].mamba_num, 1)

        pool, held = make_pool(1, track_buffer_size=1)
        cache = make_cache(pool, held, 1)
        self.assertTrue(ensure_mamba_capacity(pool, [make_req()], cache))
        self.assertEqual(cache.evict_for_alloc.call_args.args[0].mamba_num, 1)

        pool, held = make_pool(1, extra_buffer=False)
        cache = make_cache(pool, held, 0)
        cache.supports_mamba = lambda: False
        self.assertTrue(ensure_mamba_capacity(pool, [make_req()], cache))
        cache.evict_for_alloc.assert_not_called()

        pool, held = make_pool(1)
        cache = make_cache(pool, held, 0)
        cache.supports_mamba = lambda: False
        self.assertFalse(ensure_mamba_capacity(pool, [make_req()], cache))
        cache.evict_for_alloc.assert_not_called()

    def test_pd_preallocation_leaves_blocked_request_queued(self):
        pool, held = make_pool(1)
        cache = make_cache(pool, held, 0)
        req = SimpleNamespace(
            rid="blocked",
            origin_input_ids=[1],
            output_ids=[],
            finished_reason=None,
            is_retracted=True,
            kv=make_req().kv,
        )
        decode_req = SimpleNamespace(
            req=req, waiting_for_input=True, is_rebootstrap=False
        )
        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.pp_size = 1
        queue.queue = [decode_req]
        queue.pending_reqs = []
        queue.retracted_queue = [req]
        queue.req_to_token_pool = pool
        queue.req_to_metadata_buffer_idx_allocator = SimpleNamespace(
            available_size=lambda: 1
        )
        queue.tree_cache = cache
        queue.scheduler = SimpleNamespace(
            running_batch=SimpleNamespace(reqs=[]),
            enable_priority_scheduling=False,
            enable_hisparse=False,
            enable_lora=False,
        )
        queue._resolve_pending_reqs = MagicMock()
        queue._update_handshake_waiters = MagicMock()
        queue._uses_swa_tail_prealloc = MagicMock(return_value=False)
        queue._uses_swa_reservation = MagicMock(return_value=False)
        queue._allocatable_token_budgets = MagicMock(return_value=4096)
        queue._hicache_pending_restore_tokens = MagicMock(return_value=0)
        queue._match_prefix_and_lock = MagicMock()
        queue._pre_alloc = MagicMock()

        with patch(
            "sglang.srt.disaggregation.decode.get_disagg",
            return_value=SimpleNamespace(
                disaggregation_decode_host_receive_threshold=1
            ),
        ):
            self.assertEqual(queue.pop_preallocated(), ([], []))
        self.assertEqual(queue.queue, [decode_req])
        queue._match_prefix_and_lock.assert_not_called()
        queue._pre_alloc.assert_not_called()
        self.assertIsNone(req.kv.req_pool_idx)

        queue._prealloc_required_tokens = MagicMock(return_value=(1, 0))
        queue._prealloc_reservation_fits = MagicMock(return_value=True)
        queue.token_to_kv_pool_allocator = SimpleNamespace(
            prealloc_fits_assumes_reclaim=lambda: False
        )
        with patch(
            "sglang.srt.disaggregation.decode.get_disagg",
            return_value=SimpleNamespace(
                disaggregation_decode_retraction_backup="cpu_tensor"
            ),
        ):
            self.assertEqual(queue.resume_retracted_reqs(), [])
        self.assertEqual(queue.retracted_queue, [req])
        self.assertTrue(req.is_retracted)
        queue._pre_alloc.assert_not_called()


if __name__ == "__main__":
    unittest.main()
