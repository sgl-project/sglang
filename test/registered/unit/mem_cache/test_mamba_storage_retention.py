"""Only the newest Mamba state of a radix path stays in external storage."""

import time
import unittest
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


class TestMambaStorageRetention(unittest.TestCase):
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
        MambaStateRemover(stats, retry_delay_s=0.01).submit(storage, ["a", "b"])
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
        remover = MambaStateRemover(stats, retry_delay_s=0.01)
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


if __name__ == "__main__":
    unittest.main()
