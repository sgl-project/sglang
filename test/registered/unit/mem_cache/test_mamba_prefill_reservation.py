"""CPU regression tests for Mamba checkpoint slots held across prefill."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.managers.schedule_policy import PrefillAdder
from sglang.srt.mem_cache.common import checkpoint_kv_cache
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool, ReqToTokenPool
from sglang.srt.mem_cache.unified_cache.components.mamba import (
    MambaComponent,
    MambaSlotExhausted,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _Allocator:
    def __init__(self, size):
        self.free_ids = list(range(size))

    def alloc(self, n):
        if len(self.free_ids) < n:
            return None
        return torch.tensor([self.free_ids.pop() for _ in range(n)])

    def free(self, slots):
        self.free_ids.extend(slots.reshape(-1).tolist())

    def available_size(self):
        return len(self.free_ids)


def _make_pool_and_req():
    pool = object.__new__(HybridReqToTokenPool)
    pool.mamba_allocator = _Allocator(3)
    pool.enable_mamba_extra_buffer = True
    pool.enable_mamba_extra_buffer_lazy = False
    pool.mamba_ping_pong_track_buffer_size = 1
    pool.mamba_ckpt_pool = None
    pool.mamba_pool = SimpleNamespace(
        replayssm_spec_write_pos=None, replayssm_write_pos=None
    )
    pool.req_index_to_mamba_index_mapping = torch.zeros(1, dtype=torch.int32)
    pool.req_index_to_mamba_ping_pong_track_buffer_mapping = torch.zeros(
        (1, 1), dtype=torch.int64
    )
    live = pool.mamba_allocator.alloc(1)
    req = SimpleNamespace(
        skip_radix_cache_insert=False,
        kv=SimpleNamespace(
            req_pool_idx=0,
            holds_mamba=True,
            mamba_pool_idx=live[0],
            mamba_prefill_live_slot=None,
            mamba_prefill_ping_pong_slots=None,
            mamba_ping_pong_track_buffer=None,
            mamba_cache_reserve_slot=None,
            mamba_last_track_seqlen=16,
        ),
    )
    return pool, req


class TestMambaPrefillReservation(unittest.TestCase):
    def test_continuing_chunk_defers_before_being_added_without_reservation(self):
        adder = object.__new__(PrefillAdder)
        adder.dllm_config = None
        adder.memory_budget = SimpleNamespace(
            available_chunk_tokens=Mock(return_value=16),
            fit_chunk=Mock(return_value=16),
        )
        adder.rem_chunk_tokens = 16
        adder.kv_shard_granule = 0
        adder.prefill_delayer_single_pass = None
        adder.chunked_req_limit = None
        adder.can_run_list = []
        adder.tree_cache = SimpleNamespace(req_to_token_pool=None)
        adder._mamba_slot_cost = 0
        adder._has_mamba_slots_for_req = lambda req: False
        adder._kv_shard_reserve_scratch = lambda **kwargs: True
        req = SimpleNamespace(
            prefix_indices=[0] * 16,
            full_untruncated_fill_ids=[0] * 32,
            sampling_params=SimpleNamespace(max_new_tokens=16),
            output_ids=[],
        )
        with self.assertRaises(MambaSlotExhausted):
            adder.add_chunked_req(req)
        self.assertEqual(adder.can_run_list, [])

    def test_unfinished_prefill_can_donate_when_no_free_slot_remains(self):
        pool, req = _make_pool_and_req()
        self.assertEqual(
            pool.mamba_slots_needed_for_extend(req, reserve_cache_slot=True), 2
        )
        self.assertTrue(pool.reserve_mamba_prefill_slots(req, reserve_cache_slot=True))
        with patch.object(ReqToTokenPool, "alloc", return_value=[0]):
            with patch.object(
                pool.mamba_allocator,
                "alloc",
                side_effect=AssertionError("late Mamba allocation"),
            ):
                pool.alloc([req], reserve_mamba_cache_slot=True)
        self.assertEqual(pool.mamba_allocator.available_size(), 0)
        self.assertIsNotNone(req.kv.mamba_cache_reserve_slot)

        component = object.__new__(MambaComponent)
        component.cache = SimpleNamespace(
            req_to_token_pool=pool, enable_mamba_extra_buffer=True
        )
        insert_params = SimpleNamespace(mamba_value=None)
        self.assertEqual(
            component.prepare_for_caching_req(
                req, insert_params, token_ids_len=16, is_finished=False
            ),
            16,
        )
        self.assertIsNone(req.kv.mamba_cache_reserve_slot)
        self.assertIsNotNone(insert_params.mamba_value)
        self.assertEqual(pool.mamba_allocator.available_size(), 0)

        component.cleanup_after_caching_req(
            req,
            is_finished=False,
            insert_result=SimpleNamespace(mamba_exist=False),
            insert_params=insert_params,
        )
        pool.free_mamba_cache(req)
        self.assertEqual(pool.mamba_allocator.available_size(), 2)

    def test_aborted_prefill_releases_unconsumed_reservation(self):
        pool, req = _make_pool_and_req()
        self.assertTrue(pool.reserve_mamba_prefill_slots(req, reserve_cache_slot=True))
        with patch.object(ReqToTokenPool, "alloc", return_value=[0]):
            pool.alloc([req], reserve_mamba_cache_slot=True)
        self.assertEqual(pool.mamba_allocator.available_size(), 0)
        pool.free_mamba_cache(req)
        self.assertIsNone(req.kv.mamba_cache_reserve_slot)
        self.assertEqual(pool.mamba_allocator.available_size(), 3)

    def test_uncacheable_prefill_releases_unconsumed_reservation(self):
        pool, req = _make_pool_and_req()
        self.assertTrue(pool.reserve_mamba_prefill_slots(req, reserve_cache_slot=True))
        with patch.object(ReqToTokenPool, "alloc", return_value=[0]):
            pool.alloc([req], reserve_mamba_cache_slot=True)
        req.skip_radix_cache_insert = True
        req.finished = lambda: False
        cache = SimpleNamespace(req_to_token_pool=pool)
        checkpoint_kv_cache(req, cache)
        self.assertIsNone(req.kv.mamba_cache_reserve_slot)
        self.assertEqual(pool.mamba_allocator.available_size(), 1)
        pool.free_mamba_cache(req)
        self.assertEqual(pool.mamba_allocator.available_size(), 3)

    def test_partial_admission_reservation_rolls_back(self):
        pool, req = _make_pool_and_req()
        pool.mamba_allocator.alloc(1)  # Leave room for only one of two slots.
        self.assertFalse(pool.reserve_mamba_prefill_slots(req, reserve_cache_slot=True))
        self.assertEqual(pool.mamba_allocator.available_size(), 1)
        self.assertIsNone(req.kv.mamba_prefill_ping_pong_slots)
        self.assertIsNone(req.kv.mamba_cache_reserve_slot)

    def test_fresh_request_consumes_all_admission_slots_without_late_alloc(self):
        pool, req = _make_pool_and_req()
        pool.mamba_allocator.free(req.kv.mamba_pool_idx.unsqueeze(0))
        req.kv.mamba_pool_idx = None
        req.kv.holds_mamba = False
        self.assertEqual(
            pool.mamba_slots_needed_for_extend(req, reserve_cache_slot=True), 3
        )
        self.assertTrue(pool.reserve_mamba_prefill_slots(req, reserve_cache_slot=True))
        self.assertEqual(pool.mamba_allocator.available_size(), 0)
        with patch.object(ReqToTokenPool, "alloc", return_value=[0]):
            with patch.object(
                pool.mamba_allocator,
                "alloc",
                side_effect=AssertionError("late Mamba allocation"),
            ):
                pool.alloc([req], reserve_mamba_cache_slot=True)
        self.assertIsNone(req.kv.mamba_prefill_live_slot)
        self.assertIsNone(req.kv.mamba_prefill_ping_pong_slots)
        pool.free_mamba_cache(req)
        self.assertEqual(pool.mamba_allocator.available_size(), 3)

    def test_hicache_load_back_uses_reserved_live_slot(self):
        pool, req = _make_pool_and_req()
        pool.mamba_allocator.free(req.kv.mamba_pool_idx.unsqueeze(0))
        req.kv.mamba_pool_idx = None
        req.kv.holds_mamba = False
        self.assertTrue(pool.reserve_mamba_prefill_slots(req, reserve_cache_slot=True))
        component = object.__new__(MambaComponent)
        component.cache = SimpleNamespace(req_to_token_pool=pool)
        component.tree_core = SimpleNamespace(
            component_has_host_value_only=lambda *args: True
        )
        with patch.object(
            pool.mamba_allocator,
            "alloc",
            side_effect=AssertionError("load-back allocated another Mamba slot"),
        ):
            prep = component.prepare_load_back(1, req=req)
        self.assertIsNone(req.kv.mamba_prefill_live_slot)
        self.assertEqual(int(prep.allocated_mamba_slot[0]), int(req.kv.mamba_pool_idx))
        component.finalize_load_back(req, prep, False)
        pool.release_mamba_prefill_slots(req)
        self.assertEqual(pool.mamba_allocator.available_size(), 3)


if __name__ == "__main__":
    unittest.main()
