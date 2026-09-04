import unittest
from unittest.mock import MagicMock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.io_struct import FlushCacheReqInput
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.flush_wrapper import (
    SchedulerFlushWrapper,
)
from sglang.srt.mem_cache.base_prefix_cache import EvictParams, EvictResult
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

register_cpu_ci(est_time=14, suite="base-a-test-cpu")
register_cpu_ci(est_time=8, suite="stage-b-test-cpu-intel")


class TestSchedulerFlushCache(unittest.TestCase):
    def _new_scheduler(self) -> Scheduler:
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.ipc_channels = MagicMock()
        scheduler.flush_cache = MagicMock(return_value=True)
        scheduler.is_fully_idle = MagicMock(return_value=False)
        scheduler.flush_wrapper = SchedulerFlushWrapper(
            flush_cache=scheduler.flush_cache,
            is_fully_idle=scheduler.is_fully_idle,
            ipc_channels=scheduler.ipc_channels,
        )
        return scheduler

    def test_immediate_flush_no_timeout(self):
        """No timeout → flush immediately regardless of idle state."""
        scheduler = self._new_scheduler()
        scheduler.flush_cache.return_value = False

        output = scheduler.flush_wrapper.handle(FlushCacheReqInput(timeout_s=None))

        self.assertFalse(output.success)
        scheduler.flush_cache.assert_called_once_with(False)

    def test_immediate_flush_when_idle(self):
        """Positive timeout but already idle → flush immediately."""
        scheduler = self._new_scheduler()
        scheduler.is_fully_idle.return_value = True

        output = scheduler.flush_wrapper.handle(FlushCacheReqInput(timeout_s=5.0))

        self.assertTrue(output.success)
        scheduler.flush_cache.assert_called_once_with(False)

    def test_preserve_hicache_storage_is_forwarded(self):
        scheduler = self._new_scheduler()
        scheduler.is_fully_idle.return_value = True

        output = scheduler.flush_wrapper.handle(
            FlushCacheReqInput(
                timeout_s=5.0,
                preserve_hicache_storage=True,
            )
        )

        self.assertTrue(output.success)
        scheduler.flush_cache.assert_called_once_with(True)

    def test_defers_when_busy(self):
        """Positive timeout + busy → defers, returns None."""
        scheduler = self._new_scheduler()
        req = FlushCacheReqInput(timeout_s=3.0)

        with patch(
            "sglang.srt.managers.scheduler_components.flush_wrapper.time.monotonic",
            return_value=10.0,
        ):
            output = scheduler.flush_wrapper.handle(req)

        self.assertIsNone(output)
        pending_req, deadline = scheduler.flush_wrapper._pending
        self.assertIs(pending_req, req)
        self.assertEqual(deadline, 13.0)

    def test_rejects_when_already_pending(self):
        """Any new request is rejected while another is pending."""
        scheduler = self._new_scheduler()
        scheduler.flush_wrapper._pending = (FlushCacheReqInput(timeout_s=10.0), 999.0)

        for timeout in [None, 5.0]:
            output = scheduler.flush_wrapper.handle(
                FlushCacheReqInput(timeout_s=timeout)
            )
            self.assertFalse(output.success)
            self.assertIn("already in progress", output.message)

        scheduler.flush_cache.assert_not_called()

    def test_pending_flush_completes_on_idle(self):
        scheduler = self._new_scheduler()
        scheduler.is_fully_idle.return_value = True
        req = FlushCacheReqInput(
            timeout_s=1.0,
            preserve_hicache_storage=True,
        )
        scheduler.flush_wrapper._pending = (req, 111.0)

        scheduler.flush_wrapper.check_pending()

        self.assertIsNone(scheduler.flush_wrapper._pending)
        scheduler.flush_cache.assert_called_once_with(True)
        out = scheduler.ipc_channels.send_to_tokenizer.send_output.call_args.args[0]
        self.assertTrue(out.success)

    def test_pending_flush_expires_on_timeout(self):
        scheduler = self._new_scheduler()
        req = FlushCacheReqInput(timeout_s=1.0)
        scheduler.flush_wrapper._pending = (req, 99.0)

        with patch(
            "sglang.srt.managers.scheduler_components.flush_wrapper.time.monotonic",
            return_value=100.0,
        ):
            scheduler.flush_wrapper.check_pending()

        self.assertIsNone(scheduler.flush_wrapper._pending)
        scheduler.flush_cache.assert_not_called()
        out = scheduler.ipc_channels.send_to_tokenizer.send_output.call_args.args[0]
        self.assertFalse(out.success)

    def test_pending_flush_survives_before_deadline(self):
        scheduler = self._new_scheduler()
        req = FlushCacheReqInput(timeout_s=5.0)
        scheduler.flush_wrapper._pending = (req, 101.0)

        with patch(
            "sglang.srt.managers.scheduler_components.flush_wrapper.time.monotonic",
            return_value=100.0,
        ):
            scheduler.flush_wrapper.check_pending()

        self.assertIsNotNone(scheduler.flush_wrapper._pending)
        scheduler.ipc_channels.send_to_tokenizer.send_output.assert_not_called()

    def test_checkpoint_hicache_storage_drains_all_components(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.enable_hierarchical_cache = True
        scheduler.enable_hicache_storage = True
        scheduler.server_args = MagicMock(hicache_host_memory_mode="cache")
        scheduler.tree_cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
        scheduler.tree_cache.full_evictable_size = MagicMock(side_effect=[10, 0])
        scheduler.tree_cache.swa_evictable_size = MagicMock(side_effect=[4, 0])
        scheduler.tree_cache.mamba_evictable_size = MagicMock(side_effect=[2, 0])
        scheduler.tree_cache.full_protected_size = MagicMock(return_value=0)
        scheduler.tree_cache.swa_protected_size = MagicMock(return_value=0)
        scheduler.tree_cache.mamba_protected_size = MagicMock(return_value=0)
        scheduler.tree_cache.evict = MagicMock(
            return_value=EvictResult(
                num_tokens_evicted=10,
                swa_num_tokens_evicted=4,
                mamba_num_evicted=2,
            )
        )
        scheduler.tree_cache.writing_check = MagicMock()
        scheduler.tree_cache.check_hicache_events = MagicMock()
        scheduler.tree_cache.ongoing_backup = {1: object()}

        def drain_backup():
            scheduler.tree_cache.ongoing_backup.clear()

        scheduler.tree_cache.check_hicache_events.side_effect = drain_backup

        self.assertTrue(scheduler._checkpoint_hicache_storage())
        scheduler.tree_cache.evict.assert_called_once_with(
            EvictParams(num_tokens=10, swa_num_tokens=4, mamba_num=2)
        )
        scheduler.tree_cache.writing_check.assert_called_once_with(write_back=True)
        scheduler.tree_cache.check_hicache_events.assert_called_once()

    def test_checkpoint_hicache_storage_requires_storage_backend(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.enable_hierarchical_cache = True
        scheduler.enable_hicache_storage = False
        scheduler.server_args = MagicMock(hicache_host_memory_mode="cache")
        scheduler.tree_cache = MagicMock()

        self.assertFalse(scheduler._checkpoint_hicache_storage())
        scheduler.tree_cache.evict.assert_not_called()

    def test_checkpoint_hicache_storage_rejects_buffer_only_mode(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.enable_hierarchical_cache = True
        scheduler.enable_hicache_storage = True
        scheduler.server_args = MagicMock(hicache_host_memory_mode="buffer_only")
        scheduler.tree_cache = MagicMock()

        self.assertFalse(scheduler._checkpoint_hicache_storage())
        scheduler.tree_cache.evict.assert_not_called()

    def test_checkpoint_hicache_storage_rejects_protected_entries(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.enable_hierarchical_cache = True
        scheduler.enable_hicache_storage = True
        scheduler.server_args = MagicMock(hicache_host_memory_mode="cache")
        scheduler.tree_cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
        scheduler.tree_cache.full_protected_size = MagicMock(return_value=1)
        scheduler.tree_cache.swa_protected_size = MagicMock(return_value=0)
        scheduler.tree_cache.mamba_protected_size = MagicMock(return_value=0)
        scheduler.tree_cache.evict = MagicMock()

        self.assertFalse(scheduler._checkpoint_hicache_storage())
        scheduler.tree_cache.evict.assert_not_called()


if __name__ == "__main__":
    unittest.main()
