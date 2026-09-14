"""Unit tests for HybridCacheController._page_backup sidecar_ok logic.

Regression tests for issue #33656: when a pure-MLA model (e.g. DSV4) runs
with TP>1, follower ranks have no rank-sharded sidecars to back up.
Pre-fix, ``sidecar_ok = bool(backup_transfers)`` evaluated to ``False``
when ``backup_transfers`` was empty, yielding ``completed_tokens=0`` and
preventing L3 restore from ever triggering on follower ranks — causing
NaN on DSpark speculative hits.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
    StorageOperation,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def _indices(start: int, end: int) -> torch.Tensor:
    return torch.arange(start, end, dtype=torch.int64)


class TestPageBackupSidecarOk(unittest.TestCase):
    """Guard the sidecar_ok default and validation loop in _page_backup."""

    def _make_controller(self, *, backup_skip: bool) -> HybridCacheController:
        ctrl = HybridCacheController.__new__(HybridCacheController)
        ctrl.backup_skip = backup_skip
        ctrl.page_size = 1
        ctrl.storage_backend_type = "mooncake"
        ctrl.mem_pool_host = SimpleNamespace(entry_map={})
        ctrl.storage_backend = mock.Mock()
        return ctrl

    def _make_operation(self, *, pool_transfers=None, num_pages=4):
        return StorageOperation(
            host_indices=_indices(0, num_pages),
            token_ids=list(range(num_pages)),
            hash_value=[f"h{i}" for i in range(num_pages)],
            pool_transfers=pool_transfers,
        )

    def test_pure_mla_follower_reports_success_with_no_sidecars(self):
        """Follower rank with all-replicated pools (empty backup_transfers)
        must report completed_tokens > 0 — TP0 already persisted everything.

        Pre-fix: sidecar_ok = bool([]) = False -> completed_tokens = 0 (bug).
        Post-fix: sidecar_ok = True -> completed_tokens = full.
        """
        ctrl = self._make_controller(backup_skip=True)
        op = self._make_operation(pool_transfers=[])

        ctrl._page_backup(op)

        self.assertEqual(op.completed_tokens, len(op.hash_value) * ctrl.page_size)

    def test_follower_sidecar_write_failure_reports_zero(self):
        """When rank-sharded sidecar writes fail, completed_tokens must be 0
        so the corrupted backup is not considered restorable."""
        ctrl = self._make_controller(backup_skip=True)
        transfer = PoolTransfer(
            name=PoolName.MAMBA,
            host_indices=_indices(0, 4),
            keys=None,
            indices_from_pool=None,
        )
        op = self._make_operation(pool_transfers=[transfer])

        ctrl.storage_backend.batch_set_v2.return_value = {
            PoolName.MAMBA: [True, False, True, True]
        }

        ctrl._page_backup(op)

        self.assertEqual(op.completed_tokens, 0)

    def test_follower_sidecar_write_success_reports_full(self):
        """When rank-sharded sidecar writes succeed, completed_tokens must
        equal the full page count."""
        ctrl = self._make_controller(backup_skip=True)
        transfer = PoolTransfer(
            name=PoolName.MAMBA,
            host_indices=_indices(0, 4),
            keys=None,
            indices_from_pool=None,
        )
        op = self._make_operation(pool_transfers=[transfer])

        ctrl.storage_backend.batch_set_v2.return_value = {
            PoolName.MAMBA: [True, True, True, True]
        }

        ctrl._page_backup(op)

        self.assertEqual(op.completed_tokens, len(op.hash_value) * ctrl.page_size)


if __name__ == "__main__":
    unittest.main()
