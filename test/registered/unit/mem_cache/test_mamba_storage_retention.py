"""Only the newest Mamba state of a radix path stays in external storage."""

import gc
import threading
import time
import unittest
import weakref
from collections import Counter
from types import SimpleNamespace

from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.unified_cache.mamba_storage_retention import (
    MambaStateRemover,
    MambaStorageRetention,
)
from sglang.srt.mem_cache.unified_cache.unified_tree_core_interface import (
    TreePathNodeInfo,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _cache(path_hashes, path_info, own_hashes=()):
    return SimpleNamespace(
        page_size=2,
        tree_core=SimpleNamespace(
            collect_path_info=lambda node_ids: (path_info,),
            get_hash_values=lambda node_id: list(own_hashes),
        ),
        get_prefix_hash_values=lambda node_id: list(path_hashes),
    )


class _Storage:
    def __init__(self, *codes):
        self.codes = list(codes)
        self.calls = []

    def batch_remove_pool_keys(self, pool_name, page_keys):
        self.calls.append((pool_name, list(page_keys)))
        codes = self.codes.pop(0)
        if isinstance(codes, Exception):
            raise codes
        return codes


def _wait_for(predicate):
    deadline = time.monotonic() + 5
    while not predicate() and time.monotonic() < deadline:
        time.sleep(0.01)
    return predicate()


class TestMambaStorageRetention(CustomTestCase):
    def _remover(self, stats, **kwargs):
        remover = MambaStateRemover(stats, **kwargs)
        self.addCleanup(remover.close)
        return remover

    def test_close_cancels_retry_and_releases_backend(self):
        """Detach must release even a backend retained for a leased-key retry."""
        storage = _Storage([-706])
        backend_ref = weakref.ref(storage)
        stats = Counter()
        remover = self._remover(stats, retry_delay_s=65)
        remover.submit(storage, ["old"])
        self.assertTrue(_wait_for(lambda: stats["mamba_remove_deferred"] == 1))
        remover.close()
        self.assertFalse(remover._thread.is_alive())
        self.assertEqual(len(storage.calls), 1)
        remover.submit(storage, ["late"])
        del storage
        gc.collect()
        self.assertIsNone(backend_ref())

    def test_close_waits_for_active_remove_and_cancels_queued_work(self):
        """A backend cannot close while its removal RPC is still in progress."""
        entered, release = threading.Event(), threading.Event()
        calls = []

        def remove(pool, keys):
            calls.append(list(keys))
            entered.set()
            if not release.wait(5):
                raise RuntimeError("test did not release removal")
            return [0]

        remover = self._remover(Counter())
        self.addCleanup(release.set)
        storage = SimpleNamespace(batch_remove_pool_keys=remove)
        remover.submit(storage, ["active"])
        self.assertTrue(entered.wait(5))
        remover.submit(storage, ["queued"])
        closer = threading.Thread(target=remover.close)
        closer.start()
        try:
            self.assertTrue(remover._stop.wait(5))
            self.assertTrue(closer.is_alive())
        finally:
            release.set()
            closer.join(5)
        self.assertFalse(closer.is_alive())
        self.assertFalse(remover._thread.is_alive())
        self.assertEqual(calls, [["active"]])

    def test_partial_rank_write_keeps_backup_beliefs_identical(self):
        """Local Mamba write failure must not split subsequent TP admission."""
        from sglang.srt.mem_cache.buffer_mode.storage_existence_cache import (
            StorageExistenceCache,
        )
        from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

        coverage = []
        deletes = []
        for written in (1, 0):
            beliefs = StorageExistenceCache()
            beliefs.pool(PoolName.MAMBA).add(["old"])
            submitted = []
            entry = SimpleNamespace(superseded_mamba_keys=["old"])
            cache = SimpleNamespace(
                page_size=64,
                buffer_pipeline=SimpleNamespace(
                    finish_storage_write_ack=lambda *a, **kw: entry
                ),
                storage_existence_cache=beliefs,
                mamba_storage_retention=SimpleNamespace(
                    remover=SimpleNamespace(
                        submit=lambda storage, keys: submitted.extend(keys)
                    )
                ),
                cache_controller=SimpleNamespace(storage_backend=object()),
            )
            operation = SimpleNamespace(
                id=1,
                pool_storage_result=SimpleNamespace(
                    extra_pool_hit_pages={PoolName.MAMBA: written}
                ),
            )
            UnifiedRadixCache._finish_backup_ack(
                cache,
                operation,
                requested_tokens=None,
                agreed_tokens=0,
                log_metrics=False,
            )
            coverage.append(beliefs.pool(PoolName.MAMBA).covered(["old"]))
            deletes.append(submitted)
        self.assertEqual(coverage, [False, False])
        self.assertEqual(deletes, [["old"], []])

    def test_supersede_keeps_forks_and_unknown_states(self):
        path_info = (
            TreePathNodeInfo(1, 0, 0, 4, 4, 2),
            TreePathNodeInfo(2, 1, 4, 6, 2, 1),
            TreePathNodeInfo(3, 2, 6, 10, 4, 0),
        )
        retention = MambaStorageRetention(_cache(["a", "b", "c"], path_info))
        for key in ("a", "b", "c", "elsewhere"):
            retention.remember(key)

        self.assertEqual(retention.supersede(3, "e"), ["a", "c"])
        self.assertEqual(list(retention.keys), ["b", "elsewhere", "e"])
        self.assertEqual(retention.supersede(3, "e"), [])

    def test_supersede_finds_states_inside_a_whole_node(self):
        path_info = (TreePathNodeInfo(1, 0, 0, 6, 6, 0),)
        retention = MambaStorageRetention(_cache([], path_info, ["a", "b", "c"]))
        retention.remember("b")
        self.assertEqual(retention.supersede(1, "c"), ["b"])

    def test_remembered_keys_are_capped_oldest_first(self):
        retention = MambaStorageRetention(_cache([], ()), capacity=2)
        for key in ("a", "b", "a", "c"):
            retention.remember(key)
        self.assertEqual(list(retention.keys), ["a", "c"])

    def test_leased_page_is_retried_once(self):
        storage = _Storage([0, 0, -706, 0], [-706, -706])
        stats = Counter()
        self._remover(stats, retry_delay_s=0.01).submit(storage, ["a", "b"])
        self.assertTrue(_wait_for(lambda: sum(stats.values()) == 6))
        self.assertEqual(
            storage.calls, [(PoolName.MAMBA, ["a", "b"]), (PoolName.MAMBA, ["b"])]
        )
        self.assertEqual(
            stats,
            Counter(mamba_removed=3, mamba_remove_deferred=1, mamba_remove_leased=2),
        )

    def test_remove_counts_each_outcome(self):
        storage = _Storage([0, -704, -1], RuntimeError("down"), None)
        stats = {"offload": 3}
        remover = self._remover(stats, retry_delay_s=0.01)
        remover.submit(storage, ["a", "b", "c"])
        remover.submit(storage, ["d", "e"])
        remover.submit(storage, ["f"])
        self.assertTrue(_wait_for(lambda: len(storage.calls) == 3))
        self.assertTrue(_wait_for(lambda: stats.get("mamba_remove_failed") == 3))
        self.assertEqual(
            stats,
            {
                "offload": 3,
                "mamba_removed": 1,
                "mamba_remove_not_found": 1,
                "mamba_remove_failed": 3,
            },
        )

    def test_store_without_batch_remove_leaves_states_to_eviction(self):
        storage = _Storage(None, None)
        stats = Counter()
        remover = self._remover(stats, retry_delay_s=0.01)
        with self.assertLogs(
            "sglang.srt.mem_cache.unified_cache.mamba_storage_retention", "INFO"
        ) as logs:
            remover.submit(storage, ["a"])
            remover.submit(storage, ["b"])
            self.assertTrue(_wait_for(lambda: len(storage.calls) == 2))
            time.sleep(0.05)
        self.assertEqual(len(logs.records), 1)
        self.assertEqual(stats, Counter())


if __name__ == "__main__":
    unittest.main()
