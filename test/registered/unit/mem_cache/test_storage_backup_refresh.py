"""Rank-agreed age tracking for HiCache storage refresh submissions."""

import unittest
from unittest.mock import patch

import torch

from sglang.srt.mem_cache.storage_backup_refresh import StorageBackupRefresh
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestStorageBackupRefresh(CustomTestCase):
    def make_refresh(self, agreed_time):
        # The collective is the boundary: simulate ranks with different clocks
        # receiving the same reduced timestamp, while exercising real tensors.
        def all_reduce(clock, op):
            self.assertEqual(op, torch.distributed.ReduceOp.MIN)
            clock.fill_(agreed_time[0])

        return StorageBackupRefresh(20, all_reduce)

    def sync(self, refresh, agreed_time, now):
        agreed_time[0] = now
        for _ in range(refresh._CLOCK_SYNC_STEPS):
            refresh.step()

    def test_new_write_near_sync_does_not_expire_early(self):
        agreed_time = [1000]
        refresh = self.make_refresh(agreed_time)
        refresh.step()
        with patch(
            "sglang.srt.mem_cache.storage_backup_refresh.time.time", return_value=1255
        ):
            refresh.stamp("prefix")
        self.assertTrue(refresh.is_fresh("prefix"))
        self.sync(refresh, agreed_time, 1256)
        self.assertTrue(refresh.is_fresh("prefix"))
        self.sync(refresh, agreed_time, 1275)
        self.assertTrue(refresh.is_fresh("prefix"))
        self.sync(refresh, agreed_time, 1276)
        self.assertFalse(refresh.is_fresh("prefix"))

    def test_restamp_preserves_other_keys_expiry_order(self):
        agreed_time = [1000]
        refresh = self.make_refresh(agreed_time)
        refresh.stamp("renewed")
        refresh.stamp("old")
        refresh.step()
        refresh.stamp("renewed")
        self.sync(refresh, agreed_time, 1019)
        self.sync(refresh, agreed_time, 1020)
        self.assertFalse(refresh.is_fresh("old"))
        self.assertTrue(refresh.is_fresh("renewed"))
        self.sync(refresh, agreed_time, 1039)
        self.assertFalse(refresh.is_fresh("renewed"))

    def test_ranks_with_different_local_clocks_expire_together(self):
        agreed_time = [1000]
        ranks = [self.make_refresh(agreed_time) for _ in range(2)]
        for rank, refresh in enumerate(ranks):
            with patch(
                "sglang.srt.mem_cache.storage_backup_refresh.time.time",
                return_value=1000 + rank * 100,
            ):
                refresh.step()
                refresh.stamp("prefix")
        for now, fresh in ((1256, True), (1275, True), (1276, False)):
            for rank, refresh in enumerate(ranks):
                with patch(
                    "sglang.srt.mem_cache.storage_backup_refresh.time.time",
                    return_value=now + rank * 100,
                ):
                    self.sync(refresh, agreed_time, now)
                self.assertEqual(refresh.is_fresh("prefix"), fresh)


if __name__ == "__main__":
    unittest.main()
