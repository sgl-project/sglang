"""Async host-backup regression tests for HiSparse spec decode.

The trailing ``wait_for_pending_backup`` in ``_backup_device_locs_to_host``
synchronized every ``commit_accept_tokens`` call with the device→host copy.
Removing that wait lets the scheduler continue on the CPU while the copy
runs on ``decode_backup_stream``; correctness is preserved by calling
``wait_for_pending_backup`` in ``swap_in_selected_pages`` and
``swap_in_selected_blocks`` before any host-pool read.
"""

import inspect
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.managers.hisparse_coordinator import HiSparseSpecSwapManager
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _patch_device_module():
    """Patch device_module so the backup function runs on CPU."""
    return patch("sglang.srt.managers.hisparse_coordinator.device_module")


class TestAsyncHostBackup(CustomTestCase):
    """Verify the backup event ordering without a GPU."""

    def _make_manager(self):
        coordinator = MagicMock()
        coordinator.wait_for_pending_backup = MagicMock()
        coordinator.decode_backup_stream = MagicMock()
        coordinator.decode_producer_stream = None
        coordinator.mem_pool_host = MagicMock()
        coordinator.mem_pool_device = MagicMock()
        coordinator._backup_done_event = MagicMock()
        coordinator._has_pending_backup = False

        manager = object.__new__(HiSparseSpecSwapManager)
        manager._coordinator = coordinator
        return manager, coordinator

    def _run_backup(self, manager, coordinator, numel=4):
        host_locs = SimpleNamespace(numel=lambda: numel, is_cuda=False)
        device_locs = SimpleNamespace(numel=lambda: numel, is_cuda=False)
        device_locs.contiguous = lambda: device_locs

        with _patch_device_module() as mock_device:
            mock_device.current_stream.return_value = MagicMock()
            mock_device.stream.return_value.__enter__ = lambda self: None
            mock_device.stream.return_value.__exit__ = lambda *args: None
            manager._backup_device_locs_to_host(host_locs, device_locs)

    def test_backup_returns_without_trailing_wait(self):
        """_backup_device_locs_to_host must not wait for the copy to finish."""
        manager, coordinator = self._make_manager()
        self._run_backup(manager, coordinator, numel=4)

        # Exactly one wait: the pre-copy drain of the previous backup.
        # A trailing wait would make this two calls.
        self.assertEqual(
            coordinator.wait_for_pending_backup.call_count,
            1,
            "backup must not block on the trailing copy completion "
            f"(got {coordinator.wait_for_pending_backup.call_count} calls, expected 1)",
        )

    def test_backup_sets_pending_flag(self):
        """After the async copy, the pending flag must be set for later sync."""
        manager, coordinator = self._make_manager()
        self._run_backup(manager, coordinator, numel=2)

        self.assertTrue(
            coordinator._has_pending_backup,
            "the pending flag must be set after launching the backup copy",
        )

    def test_empty_backup_is_noop(self):
        """Zero-element backup should skip all stream work and waits."""
        manager, coordinator = self._make_manager()
        self._run_backup(manager, coordinator, numel=0)

        coordinator.wait_for_pending_backup.assert_not_called()
        self.assertFalse(coordinator._has_pending_backup)

    def test_swap_in_selected_pages_contains_sync(self):
        """swap_in_selected_pages must synchronize before reading host data."""
        from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator

        source = inspect.getsource(HiSparseCoordinator.swap_in_selected_pages)
        self.assertIn(
            "wait_for_pending_backup",
            source,
            "swap_in_selected_pages must call wait_for_pending_backup "
            "before reading from the host pool",
        )

    def test_swap_in_selected_blocks_contains_sync(self):
        """swap_in_selected_blocks must synchronize before reading host data."""
        from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator

        source = inspect.getsource(HiSparseCoordinator.swap_in_selected_blocks)
        self.assertIn(
            "wait_for_pending_backup",
            source,
            "swap_in_selected_blocks must call wait_for_pending_backup "
            "before reading from the host pool",
        )


if __name__ == "__main__":
    unittest.main()
