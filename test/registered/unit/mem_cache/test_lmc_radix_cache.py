"""CPU-only tests for the LMCache MP radix-cache fast path."""

import importlib
import sys
import types
import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.mem_cache.base_prefix_cache import MatchResult
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _load_lmcache_radix_module():
    """Import the integration without requiring LMCache in CPU CI."""
    module_name = "sglang.srt.mem_cache.storage.lmcache.lmc_radix_cache"
    if module_name in sys.modules:
        return sys.modules[module_name]

    lmcache = types.ModuleType("lmcache")
    lmcache.__path__ = []
    integration = types.ModuleType("lmcache.integration")
    integration.__path__ = []
    sglang_integration = types.ModuleType("lmcache.integration.sglang")
    sglang_integration.__path__ = []
    mp_adapter = types.ModuleType("lmcache.integration.sglang.multi_process_adapter")
    sglang_adapter = types.ModuleType("lmcache.integration.sglang.sglang_adapter")
    utils = types.ModuleType("lmcache.integration.sglang.utils")

    mp_adapter.LMCacheMPConnector = MagicMock
    sglang_adapter.LMCacheLayerwiseConnector = MagicMock
    sglang_adapter.LoadMetadata = MagicMock
    sglang_adapter.StoreMetadata = MagicMock
    utils.lmcache_get_config = MagicMock

    lmcache.integration = integration
    integration.sglang = sglang_integration
    sglang_integration.multi_process_adapter = mp_adapter
    sglang_integration.sglang_adapter = sglang_adapter
    sglang_integration.utils = utils

    stubs = {
        "lmcache": lmcache,
        "lmcache.integration": integration,
        "lmcache.integration.sglang": sglang_integration,
        "lmcache.integration.sglang.multi_process_adapter": mp_adapter,
        "lmcache.integration.sglang.sglang_adapter": sglang_adapter,
        "lmcache.integration.sglang.utils": utils,
    }
    with patch.dict(sys.modules, stubs):
        return importlib.import_module(module_name)


_lmc_radix_cache = _load_lmcache_radix_module()
LMCRadixCache = _lmc_radix_cache.LMCRadixCache


class TestLMCacheMPMatchPrefix(unittest.TestCase):
    CHUNK_SIZE = 64
    RID = "request-0"

    def setUp(self):
        self.connector = MagicMock()
        self.connector.chunk_size.return_value = self.CHUNK_SIZE
        self.cache = object.__new__(LMCRadixCache)
        self.cache.lmcache_connector = self.connector
        self.cache._mp_load_back_markers = {}
        self.req = SimpleNamespace(rid=self.RID)
        self.last_node = object()

    def _match(
        self, key: RadixKey, device_tokens: int
    ) -> tuple[MatchResult, MatchResult]:
        value = torch.arange(device_tokens, dtype=torch.int64)
        base_result = MatchResult(
            device_indices=value,
            last_device_node=self.last_node,
            last_host_node=self.last_node,
            best_match_node=self.last_node,
        )
        result = self.cache._mp_match_prefix(
            key,
            base_result,
            value,
            self.last_node,
            self.req,
        )
        return result, base_result

    def test_skips_lookup_when_device_covers_aligned_maximum(self):
        key = RadixKey(array("q", range(130)))
        self.connector.lookup_kv.side_effect = AssertionError(
            "blocking lookup must not run"
        )

        result, base_result = self._match(key, device_tokens=128)

        self.assertIs(result, base_result)
        self.connector.lookup_kv.assert_not_called()
        self.connector.release_pending.assert_called_once_with(self.RID)
        self.assertNotIn(self.RID, self.cache._mp_load_back_markers)

    def test_lookup_runs_when_one_aligned_token_can_be_loaded(self):
        key = RadixKey(array("q", range(128)))
        self.connector.lookup_kv.return_value = 128

        result, _ = self._match(key, device_tokens=127)

        self.connector.lookup_kv.assert_called_once_with(key.raw_token_ids(), self.RID)
        self.connector.release_pending.assert_not_called()
        self.assertEqual(result.host_hit_length, 1)
        self.assertEqual(
            self.cache._mp_load_back_markers[self.RID].value_numel,
            127,
        )

    def test_effective_capped_key_uses_no_lookup_for_sub_chunk_input(self):
        key = RadixKey(array("q", range(130)), limit=63)
        self.connector.lookup_kv.side_effect = AssertionError(
            "sub-chunk lookup must not run"
        )

        result, base_result = self._match(key, device_tokens=0)

        self.assertIs(result, base_result)
        self.connector.lookup_kv.assert_not_called()
        self.connector.release_pending.assert_called_once_with(self.RID)

    def test_lookup_can_extend_across_multiple_chunks(self):
        key = RadixKey(array("q", range(130)))
        self.connector.lookup_kv.return_value = 128

        result, _ = self._match(key, device_tokens=64)

        self.connector.lookup_kv.assert_called_once_with(key.raw_token_ids(), self.RID)
        self.connector.release_pending.assert_not_called()
        self.assertEqual(result.host_hit_length, 64)

    def test_skip_releases_stale_marker_without_second_lookup(self):
        key = RadixKey(array("q", range(130)))
        self.connector.lookup_kv.return_value = 128
        first_result, _ = self._match(key, device_tokens=0)
        self.assertEqual(first_result.host_hit_length, 128)
        self.assertIn(self.RID, self.cache._mp_load_back_markers)

        second_result, second_base_result = self._match(key, device_tokens=128)

        self.assertIs(second_result, second_base_result)
        self.connector.lookup_kv.assert_called_once()
        self.connector.release_pending.assert_called_once_with(self.RID)
        self.assertNotIn(self.RID, self.cache._mp_load_back_markers)


if __name__ == "__main__":
    unittest.main(verbosity=2)
