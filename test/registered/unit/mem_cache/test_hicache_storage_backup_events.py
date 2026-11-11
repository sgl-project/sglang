"""Which settled HiCache storage backups are announced on the storage tier."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestStorageBackupAckEvents(unittest.TestCase):
    def _settle(self, *, requested_tokens, agreed_tokens):
        cache = object.__new__(UnifiedRadixCache)
        cache.buffer_pipeline = None
        cache.enable_storage_metrics = False
        cache.storage_metrics_collector = None
        cache.ongoing_backup = {7: ("node", "lock-params")}
        cache._backup_requested_tokens = {7: requested_tokens}
        calls = MagicMock()
        cache.tree_core = calls.tree_core
        cache.dec_host_lock_ref = calls.dec_host_lock_ref
        cache._finish_backup_ack(
            SimpleNamespace(id=7, completed_tokens=agreed_tokens),
            requested_tokens=requested_tokens,
            agreed_tokens=agreed_tokens,
            log_metrics=False,
        )
        return [call[0] for call in calls.mock_calls]

    def test_whole_write_is_announced_while_the_node_is_still_locked(self):
        self.assertEqual(
            self._settle(requested_tokens=4, agreed_tokens=4),
            ["tree_core.record_storage_backup_event", "dec_host_lock_ref"],
        )

    def test_short_or_untracked_write_is_not_announced(self):
        """Storage may hold only a prefix of a short write, so an indexer must not
        list the node there; an untracked write carries no span to announce."""
        for requested, agreed in ((4, 2), (4, 0), (None, 0)):
            with self.subTest(requested_tokens=requested, agreed_tokens=agreed):
                self.assertEqual(
                    self._settle(requested_tokens=requested, agreed_tokens=agreed),
                    ["dec_host_lock_ref"],
                )


if __name__ == "__main__":
    unittest.main()
