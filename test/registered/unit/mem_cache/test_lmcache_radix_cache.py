"""CPU-only contract tests for the SGLang LMCache MP integration."""

from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

_MODULE_PATH = (
    Path(__file__).resolve().parents[4]
    / "python/sglang/srt/mem_cache/storage/lmcache/lmc_radix_cache.py"
)
_MODULE_NAME = "_sglang_lmc_radix_cache_under_test"


class _RadixKey:
    def __init__(
        self,
        token_ids,
        extra_key=None,
        is_bigram=False,
        limit=None,
        cache_salt=None,
    ):
        self.token_ids = token_ids
        self.extra_key = extra_key
        self.is_bigram = is_bigram
        self.limit = limit
        self.cache_salt = cache_salt

    def raw_token_ids(self):
        return self.token_ids

    def __len__(self):
        return len(self.token_ids)

    def __getitem__(self, item):
        return type(self)(
            self.token_ids[item],
            self.extra_key,
            self.is_bigram,
            cache_salt=self.cache_salt,
        )


class _TreeNode:
    def __init__(self, priority=0):
        self.priority = priority
        self.children = {}


@dataclass
class _MatchPrefixParams:
    key: _RadixKey
    req: object = None


@dataclass
class _MatchResult:
    device_indices: torch.Tensor
    last_device_node: object
    last_host_node: object = None
    best_match_node: object = None
    host_hit_length: int = 0


@dataclass
class _StoreMetadata:
    last_node: object
    token_ids: list[int]
    kv_indices: torch.Tensor
    offset: int
    request_id: str = ""
    cache_salt: str = ""


@dataclass
class _LoadMetadata:
    token_ids: list[int]
    slot_mapping: torch.Tensor
    offset: int
    prefix_pad: int = 0
    request_id: str = ""


class _RadixCache:
    def match_prefix(self, params):
        return self._base_match_result

    def cache_finished_req(self, req, is_insert=True, *, kv_len_to_handle):
        self._base_finished_calls.append((req, is_insert, kv_len_to_handle))


def _package(name):
    module = types.ModuleType(name)
    module.__path__ = []
    return module


def _load_target_module():
    base_prefix_cache = types.ModuleType("sglang.srt.mem_cache.base_prefix_cache")
    for name in (
        "EvictParams",
        "EvictResult",
        "InitLoadBackParams",
    ):
        setattr(base_prefix_cache, name, type(name, (), {}))
    base_prefix_cache.MatchPrefixParams = _MatchPrefixParams
    base_prefix_cache.MatchResult = _MatchResult

    radix_cache = types.ModuleType("sglang.srt.mem_cache.radix_cache")
    radix_cache.RadixCache = _RadixCache
    radix_cache.RadixKey = _RadixKey
    radix_cache.TreeNode = _TreeNode

    runtime_context = types.ModuleType("sglang.srt.runtime_context")
    runtime_context.get_memory = lambda: SimpleNamespace(lmcache_config_file="")
    runtime_context.get_spec = lambda: SimpleNamespace(speculative_eagle_topk=None)

    utils = types.ModuleType("sglang.srt.utils")
    utils.create_device_stream = Mock()
    utils.device_stream_context = lambda stream: nullcontext()

    mp_adapter = types.ModuleType("lmcache.integration.sglang.multi_process_adapter")
    mp_adapter.LMCacheMPConnector = type("LMCacheMPConnector", (), {})

    sglang_adapter = types.ModuleType("lmcache.integration.sglang.sglang_adapter")
    sglang_adapter.LMCacheLayerwiseConnector = type("LMCacheLayerwiseConnector", (), {})
    sglang_adapter.LoadMetadata = _LoadMetadata
    sglang_adapter.StoreMetadata = _StoreMetadata

    lmcache_utils = types.ModuleType("lmcache.integration.sglang.utils")
    lmcache_utils.lmcache_get_config = Mock()

    stubs = {
        "sglang.srt": _package("sglang.srt"),
        "sglang.srt.mem_cache": _package("sglang.srt.mem_cache"),
        "sglang.srt.mem_cache.storage": _package("sglang.srt.mem_cache.storage"),
        "sglang.srt.mem_cache.storage.lmcache": _package(
            "sglang.srt.mem_cache.storage.lmcache"
        ),
        base_prefix_cache.__name__: base_prefix_cache,
        radix_cache.__name__: radix_cache,
        runtime_context.__name__: runtime_context,
        utils.__name__: utils,
        "lmcache": _package("lmcache"),
        "lmcache.integration": _package("lmcache.integration"),
        "lmcache.integration.sglang": _package("lmcache.integration.sglang"),
        mp_adapter.__name__: mp_adapter,
        sglang_adapter.__name__: sglang_adapter,
        lmcache_utils.__name__: lmcache_utils,
    }

    spec = importlib.util.spec_from_file_location(_MODULE_NAME, _MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, stubs):
        sys.modules[_MODULE_NAME] = module
        spec.loader.exec_module(module)
    return module


class TestLMCacheMPCacheSalt(CustomTestCase):
    def setUp(self):
        self.module = _load_target_module()

    def _lookup_tree(self, connector):
        tree = object.__new__(self.module.LMCRadixCache)
        tree.lmcache_connector = connector
        tree._mp_load_back_markers = {}
        return tree

    def _store_tree(self, connector):
        tree = object.__new__(self.module.LMCRadixCache)
        tree._mode = self.module.LMCacheMode.MP
        tree.lmcache_connector = connector
        tree.req_to_token_pool = SimpleNamespace(
            req_to_token=torch.arange(8, dtype=torch.int64).reshape(1, 8)
        )
        tree._base_match_result = _MatchResult(
            device_indices=torch.empty(0, dtype=torch.int64),
            last_device_node=_TreeNode(),
        )
        tree._base_finished_calls = []
        tree._mp_load_back_markers = {}
        tree.inc_lock_ref = Mock()
        tree.dec_lock_ref = Mock()
        return tree

    @staticmethod
    def _base_match_result():
        node = _TreeNode()
        return _MatchResult(
            device_indices=torch.empty(0, dtype=torch.int64),
            last_device_node=node,
            last_host_node=node,
            best_match_node=node,
        )

    @staticmethod
    def _req(cache_salt):
        return SimpleNamespace(
            rid="request-1",
            origin_input_ids=[11, 12, 13],
            output_ids=[],
            extra_key=None,
            cache_salt=cache_salt,
            kv=SimpleNamespace(req_pool_idx=0, kv_committed_len=3),
        )

    def test_lookup_forwards_cache_salt_when_connector_supports_it(self):
        connector = SimpleNamespace(
            supports_cache_salt=True,
            lookup_kv=Mock(return_value=0),
            release_pending=Mock(),
        )
        tree = self._lookup_tree(connector)
        key = _RadixKey([11, 12, 13], cache_salt="tenant-a")

        tree._mp_match_prefix(
            key,
            self._base_match_result(),
            torch.empty(0, dtype=torch.int64),
            _TreeNode(),
            self._req("tenant-a"),
        )

        connector.lookup_kv.assert_called_once_with(
            [11, 12, 13], "request-1", cache_salt="tenant-a"
        )

    def test_store_forwards_cache_salt_when_connector_supports_it(self):
        connector = SimpleNamespace(
            supports_cache_salt=True,
            store_kv=Mock(),
            end_session=Mock(),
        )
        tree = self._store_tree(connector)

        tree.cache_finished_req(
            self._req("tenant-a"), is_insert=True, kv_len_to_handle=3
        )

        store_metadata = connector.store_kv.call_args.args[0]
        self.assertEqual(store_metadata.cache_salt, "tenant-a")

    def test_legacy_connector_keeps_unsalted_lookup_compatible(self):
        connector = SimpleNamespace(
            lookup_kv=Mock(return_value=0),
            release_pending=Mock(),
        )
        tree = self._lookup_tree(connector)

        tree._mp_match_prefix(
            _RadixKey([11, 12, 13]),
            self._base_match_result(),
            torch.empty(0, dtype=torch.int64),
            _TreeNode(),
            self._req(None),
        )

        connector.lookup_kv.assert_called_once_with([11, 12, 13], "request-1")

    def test_legacy_connector_skips_salted_lookup_before_external_access(self):
        connector = SimpleNamespace(
            lookup_kv=Mock(return_value=0),
            release_pending=Mock(),
        )
        tree = self._lookup_tree(connector)
        base_res = self._base_match_result()

        with self.assertLogs(self.module.logger, level="WARNING") as logs:
            result = tree._mp_match_prefix(
                _RadixKey([11, 12, 13], cache_salt="tenant-a"),
                base_res,
                torch.empty(0, dtype=torch.int64),
                _TreeNode(),
                self._req("tenant-a"),
            )

        self.assertIs(result, base_res)
        self.assertIn("Skipping LMCache MP lookup", logs.output[0])
        connector.lookup_kv.assert_not_called()

    def test_legacy_connector_keeps_unsalted_store_compatible(self):
        class LegacyStoreMetadata:
            def __init__(self, last_node, token_ids, kv_indices, offset, request_id=""):
                self.last_node = last_node
                self.token_ids = token_ids
                self.kv_indices = kv_indices
                self.offset = offset
                self.request_id = request_id

        connector = SimpleNamespace(
            store_kv=Mock(),
            end_session=Mock(),
        )
        tree = self._store_tree(connector)
        self.module.StoreMetadata = LegacyStoreMetadata

        tree.cache_finished_req(self._req(None), is_insert=True, kv_len_to_handle=3)

        connector.store_kv.assert_called_once()

    def test_legacy_connector_skips_salted_store_after_local_cleanup(self):
        connector = SimpleNamespace(
            store_kv=Mock(),
            end_session=Mock(),
        )
        tree = self._store_tree(connector)

        with self.assertLogs(self.module.logger, level="WARNING") as logs:
            tree.cache_finished_req(
                self._req("tenant-a"), is_insert=True, kv_len_to_handle=3
            )

        self.assertEqual(len(tree._base_finished_calls), 1)
        self.assertIn("Skipping LMCache MP store", logs.output[0])
        connector.store_kv.assert_not_called()
        connector.end_session.assert_called_once_with("request-1")

    def test_mp_store_failure_always_releases_local_and_external_state(self):
        connector = SimpleNamespace(
            supports_cache_salt=True,
            store_kv=Mock(side_effect=ValueError("invalid cache_salt")),
            end_session=Mock(),
        )
        tree = self._store_tree(connector)
        tree._mp_load_back_markers["request-1"] = object()

        with self.assertRaisesRegex(ValueError, "invalid cache_salt"):
            tree.cache_finished_req(
                self._req("tenant-a"), is_insert=True, kv_len_to_handle=3
            )

        self.assertNotIn("request-1", tree._mp_load_back_markers)
        tree.dec_lock_ref.assert_called_once()
        connector.end_session.assert_called_once_with("request-1")


if __name__ == "__main__":
    unittest.main()
