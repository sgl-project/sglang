"""PDMux cache pumps and SWA overlap must preserve device-page ownership."""

import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import Mock

import torch
from test_swa_locked_full_recover_unified import _build_swa_composite

from sglang.srt.mem_cache.base_prefix_cache import InsertParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestSWAProtectedPrefixOwnership(unittest.TestCase):
    def _tombstone(self):
        allocator = _build_swa_composite()
        cache = UnifiedRadixCache(
            CacheInitParams(
                disable=False,
                req_to_token_pool=None,
                token_to_kv_pool_allocator=allocator,
                page_size=1,
                sliding_window_size=8,
                tree_components=(ComponentType.FULL, ComponentType.SWA),
                tree_core_backend="python",
            )
        )
        value = allocator.alloc(8)
        allocator.free_swa(value)
        key = RadixKey(array("q", range(8)))
        cache.insert(
            InsertParams(
                key=key,
                value=value,
                component_evicted_seqlens={ComponentType.SWA: 8},
            )
        )
        node = next(iter(cache.tree_core.root_node.children.values()))
        return cache, allocator, key, value, node

    def test_partial_protected_prefix_never_frees_cache_owned_full_pages(self):
        cache, allocator, key, kept, node = self._tombstone()
        original_mapping = allocator.full_attn_allocator.virtual_to_physical[
            kept.to(torch.int64)
        ].clone()

        # Repeated duplicate chunks exercise subsequent reuse of the freed ids.
        for _ in range(3):
            full_available = allocator.full_attn_allocator.available_size()
            swa_available = allocator.swa_attn_allocator.available_size()
            fresh = allocator.alloc(4)
            result = cache.insert(
                InsertParams(
                    key=key,
                    value=torch.cat((kept[:4], fresh)),
                    prev_prefix_len=4,
                    track_adopted_ranges=True,
                )
            )

            torch.testing.assert_close(
                node.component_data[ComponentType.FULL].value, kept
            )
            torch.testing.assert_close(
                allocator.full_attn_allocator.virtual_to_physical[kept.to(torch.int64)],
                original_mapping,
            )
            self.assertTrue(bool((original_mapping > 0).all()))
            self.assertIsNone(node.component_data[ComponentType.SWA].value)
            self.assertEqual(allocator.full_attn_allocator.allocated_count(), 8)
            self.assertEqual(allocator.swa_attn_allocator.allocated_count(), 0)
            self.assertEqual(
                allocator.full_attn_allocator.available_size(), full_available
            )
            self.assertEqual(
                allocator.swa_attn_allocator.available_size(), swa_available
            )
            self.assertNotIn(ComponentType.FULL, result.adopted_ranges)
            self.assertNotIn(ComponentType.SWA, result.adopted_ranges)
            self.assertEqual(allocator.verify_byte_accounting(), [])

    def test_fresh_range_still_recovers_tombstone(self):
        cache, allocator, key, kept, node = self._tombstone()
        incoming = allocator.alloc(8)
        result = cache.insert(
            InsertParams(key=key, value=incoming, track_adopted_ranges=True)
        )

        torch.testing.assert_close(
            node.component_data[ComponentType.FULL].value, incoming
        )
        self.assertTrue(bool((node.component_data[ComponentType.SWA].value > 0).all()))
        self.assertTrue(
            bool(
                (
                    allocator.full_attn_allocator.virtual_to_physical[
                        incoming.to(torch.int64)
                    ]
                    > 0
                ).all()
            )
        )
        self.assertTrue(
            bool(
                (
                    allocator.full_attn_allocator.virtual_to_physical[
                        kept.to(torch.int64)
                    ]
                    == -1
                ).all()
            )
        )
        self.assertEqual(allocator.full_attn_allocator.allocated_count(), 8)
        self.assertEqual(allocator.swa_attn_allocator.allocated_count(), 8)
        self.assertEqual(result.adopted_ranges[ComponentType.FULL], [(0, 8)])
        self.assertEqual(result.adopted_ranges[ComponentType.SWA], [(0, 8)])


class TestHiCacheDeviceWorkReporting(unittest.TestCase):
    def _cache(self, *, write_policy="write_through", writes=0, loads=0):
        cache = object.__new__(UnifiedRadixCache)
        cache.tree_core = SimpleNamespace(enable_storage=False)
        cache.linker = None
        cache.buffer_pipeline = None
        cache.cache_controller = SimpleNamespace(write_policy=write_policy)
        cache.enable_storage_metrics = False
        cache.storage_metrics_collector = None
        cache._drain_async_work = Mock()
        cache.flush_pending_backups = Mock()
        cache._sync_hicache_ready_counts = Mock(
            return_value=(writes, loads, (0, 0, 0, 0), ())
        )
        cache.writing_check = Mock()
        cache.loading_check = Mock()
        cache._drain_storage_control_queues_impl = Mock()
        return cache

    def test_host_only_write_through_acks_do_not_publish_device_work(self):
        cache = self._cache(writes=2, loads=1)
        self.assertFalse(cache.check_hicache_events())
        cache.writing_check.assert_called_once_with(finish_count=2)
        cache.loading_check.assert_called_once_with(finish_count=1)

    def test_write_back_ack_reports_possible_device_work(self):
        cache = self._cache(write_policy="write_back", writes=1)
        self.assertTrue(cache.check_hicache_events())
        cache = self._cache(write_policy="write_back")
        self.assertFalse(cache.check_hicache_events())

    def test_storage_queue_progress_reports_possible_device_work(self):
        cache = self._cache()
        cache.enable_storage = True
        cache._sync_hicache_ready_counts.return_value = (
            0,
            0,
            (0, 1, 0, 0, 2),
            ("swa",),
        )
        self.assertTrue(cache.check_hicache_events())
        cache._drain_storage_control_queues_impl.assert_called_once_with(
            n_storage_hit=0,
            n_ack_prefetch=1,
            n_backup=0,
            n_release=0,
            extra_release_counts={"swa": 2},
            log_metrics=True,
        )

    def test_buffer_load_ack_reports_auxiliary_device_slot_releases(self):
        cache = self._cache(loads=1)
        cache.buffer_pipeline = SimpleNamespace(flush_pending_writes=Mock())
        self.assertTrue(cache.check_hicache_events())
        cache.buffer_pipeline.flush_pending_writes.assert_called_once_with()
        cache = self._cache(writes=1)
        cache.buffer_pipeline = SimpleNamespace(flush_pending_writes=Mock())
        self.assertFalse(cache.check_hicache_events())

    def test_linker_acks_only_drop_locks(self):
        cache = self._cache()
        cache.linker = SimpleNamespace(
            num_completed_loads=Mock(return_value=1),
            num_completed_offloads=Mock(return_value=1),
            drain_loads=Mock(),
            take_completed_offloads=Mock(return_value=[True]),
            commit_completed_offloads=Mock(),
        )
        cache._all_reduce_attn_groups = Mock()
        self.assertFalse(cache.check_hicache_events())
        cache.linker.drain_loads.assert_called_once_with(1)
        cache.linker.commit_completed_offloads.assert_called_once_with([True])
        cache._sync_hicache_ready_counts.assert_not_called()


if __name__ == "__main__":
    unittest.main()
