"""Unit tests for removing superseded Mamba states from the external store."""

import unittest
from collections import OrderedDict, deque
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.storage.mooncake_store.mooncake_direct_linker import (
    MooncakeDirectLinker,
)
from sglang.srt.mem_cache.unified_cache.unified_cache_linker import (
    UnifiedCacheLinkerWrapper,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

PAGE = 4


def _wrapper(hashes, forks_at_pages=()):
    path = [
        SimpleNamespace(path_end=pages * PAGE, num_children=2)
        for pages in forks_at_pages
    ]
    cache = SimpleNamespace(
        page_size=PAGE,
        tree_core=SimpleNamespace(
            is_eagle=False, collect_path_info=lambda node_ids: [path]
        ),
        get_prefix_hash_values=lambda node_id: hashes,
    )
    wrapper = object.__new__(UnifiedCacheLinkerWrapper)
    wrapper.cache = cache
    wrapper._drop_superseded_mamba = True
    wrapper._known_mamba_keys = OrderedDict()
    return wrapper


def _mamba(key):
    return PoolTransfer(name=PoolName.MAMBA, keys=[key])


class TestSupersededMambaSelection(CustomTestCase):
    def test_loaded_key_is_superseded_but_fork_and_unknown_keys_stay(self):
        wrapper = _wrapper(["h1", "h2", "h3", "h4"], forks_at_pages=(1,))
        # h1 sits at a fork, h2 was loaded (stored by another rank), h3 is unknown.
        wrapper._remember_mamba_key("h1")
        wrapper._remember_loaded_mamba_key([_mamba("h2")])

        self.assertEqual(wrapper._superseded_mamba_keys(0, [_mamba("h4")]), ["h2"])
        self.assertEqual(list(wrapper._known_mamba_keys), ["h1", "h4"])
        # The next turn supersedes the state just stored.
        wrapper.cache.get_prefix_hash_values = lambda node_id: [
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
        ]
        self.assertEqual(wrapper._superseded_mamba_keys(0, [_mamba("h5")]), ["h4"])


class _Storage:
    def __init__(self, codes):
        self.codes = codes
        self.calls = []

    def batch_remove_pool_keys(self, pool_name, keys):
        self.calls.append(list(keys))
        return [self.codes[key].pop(0) for key in keys]


class TestSupersededMambaRemovalRetry(CustomTestCase):
    def _linker(self, codes):
        linker = object.__new__(MooncakeDirectLinker)
        linker.storage = _Storage(codes)
        linker.stats = {}
        linker.mamba_remove_retry_secs = 65.0
        linker.deferred_mamba_removals = deque()
        return linker

    def test_leased_key_is_retried_after_the_lease_and_then_given_up(self):
        leased, not_found = -706, -704
        linker = self._linker({"a": [leased, 0], "b": [not_found], "c": [leased] * 4})
        now = [100.0]
        with patch(
            "sglang.srt.mem_cache.storage.mooncake_store."
            "mooncake_direct_linker.time.monotonic",
            lambda: now[0],
        ):
            linker._remove_mamba_states(["a", "b", "c"])
            linker._retry_mamba_removals()
            self.assertEqual(linker.storage.calls, [["a", "b", "c"]])
            self.assertEqual(linker._mamba_retry_timeout(), 65.0)

            for _ in range(3):
                now[0] += 65.0
                linker._retry_mamba_removals()

        self.assertEqual(
            linker.storage.calls, [["a", "b", "c"], ["a", "c"], ["c"], ["c"]]
        )
        self.assertFalse(linker.deferred_mamba_removals)
        self.assertEqual(
            linker.stats,
            {"mamba_removed": 1, "mamba_remove_not_found": 1, "mamba_remove_leased": 5},
        )


if __name__ == "__main__":
    unittest.main()
