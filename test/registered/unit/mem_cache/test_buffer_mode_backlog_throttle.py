"""Buffer-only write-backlog throttle: the queued-but-unstaged backlog is
withheld from prefill admission only when opted in, released as writes stage
or sweep, and lapses (latched) while the write path makes no progress."""

import logging
import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.srt.mem_cache.buffer_mode import pipeline as pipeline_module
from sglang.srt.mem_cache.buffer_mode.pipeline import BufferModePipeline
from sglang.srt.mem_cache.buffer_mode.storage_existence_cache import (
    StorageExistenceCache,
)
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.components.base import ComponentType
from sglang.srt.mem_cache.unified_cache.unified_tree_core_interface import (
    BufferBackupSnapshot,
    BufferBackupState,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

PAGE_SIZE = 2
HOST_POOL_TOKENS = 64
DEVICE_POOL_TOKENS = 16
MAX_CONTEXT_LEN = 6


class TestBufferModeBacklogThrottle(CustomTestCase):
    def setUp(self):
        self.snapshots: dict[int, BufferBackupSnapshot] = {}
        self.states: dict[int, BufferBackupState] = {}

    def _make_pipeline(self, *, throttle: bool) -> BufferModePipeline:
        controller = MagicMock()
        controller.mem_pool_host.size = HOST_POOL_TOKENS
        # Not a HostKVCache: the pipeline takes the per-pool (non-shared) path.
        controller.mem_pool_host.anchor_entry.host_pool = SimpleNamespace()
        controller.mem_pool_host.entry_map = {}
        controller.prefetch_tokens_occupied = 0
        controller.write.side_effect = (
            lambda device_value, node_id, extra_pools, flush: torch.arange(
                len(device_value), dtype=torch.int64
            )
        )
        controller.write_storage.side_effect = iter(range(1000, 2000))

        cache = MagicMock()
        cache.cache_controller = controller
        cache.page_size = PAGE_SIZE
        cache.enable_storage = True
        cache.enable_storage_metrics = False
        cache.hicache_storage_pass_prefix_keys = False
        cache.components = {ComponentType.FULL: SimpleNamespace()}
        cache.sidecar_pool_specs = []
        cache._build_backup_sidecar.return_value = []
        cache.storage_existence_cache = StorageExistenceCache()
        cache.token_to_kv_pool_allocator.get_kvcache.return_value = SimpleNamespace(
            size=DEVICE_POOL_TOKENS
        )
        tree = cache.tree_core
        tree.snapshot_buffer_backup.side_effect = lambda node_id, _: self.snapshots.get(
            node_id
        )
        tree.validate_buffer_backup.side_effect = lambda node_id, _: self.states.get(
            node_id
        )
        tree.buffer_backup_pool_keys.return_value = {}

        pipeline = BufferModePipeline(
            cache=cache,
            swa_window_pages=0,
            write_backlog_cap=1 << 20,
            max_context_len=MAX_CONTEXT_LEN,
            write_backlog_throttle=throttle,
        )
        pipeline._build_backup_transfers = lambda node_id, *, kv_only=False: (
            torch.arange(len(self.snapshots[node_id].key), dtype=torch.int64),
            {},
            None,
        )
        return pipeline

    def _add_node(self, node_id: int, pages: int = 2) -> int:
        self.snapshots[node_id] = BufferBackupSnapshot(
            node_id=node_id,
            parent_node_id=0,
            parent_is_root=True,
            parent_last_hash=None,
            hash_values=[f"n{node_id}-p{i}" for i in range(pages)],
            key=RadixKey(
                array("q", range(node_id * 1000, node_id * 1000 + pages * PAGE_SIZE))
            ),
            prefix_keys=None,
        )
        self.states[node_id] = BufferBackupState(0, True, None)
        return pages * PAGE_SIZE

    @staticmethod
    def _age_past_timeout(pipeline: BufferModePipeline) -> None:
        pipeline._write_progress_at -= pipeline.write_backlog_stall_timeout_s

    def test_default_off_holds_nothing(self):
        pipeline = self._make_pipeline(throttle=False)
        tokens = self._add_node(1)
        pipeline.enqueue_backup_intent(1)
        self.assertEqual(pipeline.write_backlog_tokens_, tokens)
        self.assertEqual(pipeline.write_backlog_hold_tokens(), 0)

    def test_hold_is_the_whole_backlog_and_released_as_writes_stage_or_sweep(self):
        """Pacing needs the whole backlog held: once a backlog exceeds any
        fixed cap, each admission re-queues the spans it evicted and the cap
        stops pacing. The hold shrinks as intents launch or are swept."""
        pipeline = self._make_pipeline(throttle=True)
        tokens = self._add_node(1)
        tokens += self._add_node(2, pages=DEVICE_POOL_TOKENS // PAGE_SIZE)
        for node_id in (1, 2):
            pipeline.enqueue_backup_intent(node_id)
        self.assertEqual(pipeline.write_backlog_tokens_, tokens)
        self.assertEqual(pipeline.write_backlog_hold_tokens(), tokens)
        # Node 2 is evicted before its launch (stale-swept), node 1 stages.
        del self.states[2]
        del self.snapshots[2]
        pipeline.flush_pending_writes()
        self.assertFalse(pipeline.pending_write_queue)
        self.assertEqual(len(pipeline.ongoing_write_through), 1)
        self.assertEqual(pipeline.write_backlog_hold_tokens(), 0)

    def test_hold_lapses_while_the_write_path_is_stalled(self):
        """A store that neither launches nor acks within the stall timeout
        must not keep admission blocked: the local verdict latches stalled,
        the rank-agreed flag releases the hold, and only a launch or an ack
        re-engages it. A dead store with a held window must stay lapsed
        across sweeps and new backlogs instead of re-arming every timeout."""
        pipeline = self._make_pipeline(throttle=True)
        # Twelve tokens fill the write window once loads occupy the host pool.
        self._add_node(1, pages=6)
        pipeline.enqueue_backup_intent(1)
        self.assertTrue(pipeline.write_path_alive())
        self._age_past_timeout(pipeline)
        self.assertFalse(pipeline.write_path_alive())
        with self.assertLogs(pipeline_module.logger, level=logging.WARNING):
            pipeline.set_write_path_alive(False)
        self.assertEqual(pipeline.write_backlog_hold_tokens(), 0)

        # A launch is progress: the local verdict recovers and the agreed
        # flag re-engages the hold for whatever is still queued.
        self._add_node(2)
        pipeline.enqueue_backup_intent(2)
        pipeline._cache.cache_controller.prefetch_tokens_occupied = HOST_POOL_TOKENS
        pipeline.flush_pending_writes()
        self.assertEqual(len(pipeline.pending_write_queue), 1)
        self.assertTrue(pipeline.write_path_alive())
        pipeline.set_write_path_alive(True)
        self.assertEqual(
            pipeline.write_backlog_hold_tokens(), pipeline.write_backlog_tokens_
        )
        self.assertGreater(pipeline.write_backlog_hold_tokens(), 0)

        # Dead store from here: the launched write never acks. Stalled latches
        # through the sweep that empties the queue and through a new backlog
        # (the window is still held, so a new backlog is not a fresh start).
        self._age_past_timeout(pipeline)
        self.assertFalse(pipeline.write_path_alive())
        pipeline.set_write_path_alive(False)
        del self.states[2]
        del self.snapshots[2]
        pipeline.flush_pending_writes()
        self.assertFalse(pipeline.pending_write_queue)
        self.assertFalse(pipeline.write_path_alive())
        self._add_node(3)
        pipeline.enqueue_backup_intent(3)
        self.assertFalse(pipeline.write_path_alive())
        self.assertEqual(pipeline.write_backlog_hold_tokens(), 0)
        # The storage ack is the progress that clears the latch.
        (ack_id,) = pipeline.ongoing_write_through
        pipeline.finish_backup_ack(ack_id)
        self.assertFalse(pipeline.write_path_alive())
        (operation_id,) = pipeline.ongoing_backup
        pipeline.finish_storage_write_ack(operation_id)
        self.assertTrue(pipeline.write_path_alive())
        pipeline.set_write_path_alive(True)
        self.assertGreater(pipeline.write_backlog_hold_tokens(), 0)
        # An empty backlog is never stalled, and with nothing staged or in
        # flight the next backlog starts the timeout afresh.
        pipeline.pending_write_queue.clear()
        self._age_past_timeout(pipeline)
        self.assertTrue(pipeline.write_path_alive())
        self._add_node(4)
        pipeline.enqueue_backup_intent(4)
        self.assertTrue(pipeline.write_path_alive())

    def test_extra_config_key_is_consumed_and_type_checked(self):
        (
            extra_config,
            prefetch_threshold,
            _timeout_base,
            _timeout_per_ki_token,
            _pass_prefix_keys,
            write_backlog_throttle,
        ) = HybridCacheController.parse_storage_backend_extra_config(
            '{"write_backlog_throttle": true, "prefetch_threshold": 64}'
        )
        self.assertEqual(
            (extra_config, prefetch_threshold, write_backlog_throttle), ({}, 64, True)
        )
        self.assertFalse(
            HybridCacheController.parse_storage_backend_extra_config(None)[5]
        )
        with self.assertRaisesRegex(ValueError, "write_backlog_throttle must be bool"):
            HybridCacheController.parse_storage_backend_extra_config(
                '{"write_backlog_throttle": 1}'
            )


if __name__ == "__main__":
    unittest.main()
