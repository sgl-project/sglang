import importlib
import sys
import types
import unittest
from array import array
from contextlib import contextmanager
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Iterator
from unittest.mock import MagicMock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


_MODULE_NAME = "sglang.srt.mem_cache.storage.lmcache.lmc_radix_cache"
_PARENT_MODULE_NAME, _, _MODULE_ATTRIBUTE = _MODULE_NAME.rpartition(".")
_MISSING = object()


def _fake_lmcache_modules() -> dict[str, types.ModuleType]:
    lmcache = types.ModuleType("lmcache")
    integration = types.ModuleType("lmcache.integration")
    sglang = types.ModuleType("lmcache.integration.sglang")
    multi_process_adapter = types.ModuleType(
        "lmcache.integration.sglang.multi_process_adapter"
    )
    sglang_adapter = types.ModuleType("lmcache.integration.sglang.sglang_adapter")
    utils = types.ModuleType("lmcache.integration.sglang.utils")

    lmcache.__path__ = []
    integration.__path__ = []
    sglang.__path__ = []
    multi_process_adapter.LMCacheMPConnector = object
    sglang_adapter.LMCacheLayerwiseConnector = object
    sglang_adapter.LoadMetadata = SimpleNamespace
    sglang_adapter.StoreMetadata = SimpleNamespace
    utils.lmcache_get_config = MagicMock()

    return {
        "lmcache": lmcache,
        "lmcache.integration": integration,
        "lmcache.integration.sglang": sglang,
        "lmcache.integration.sglang.multi_process_adapter": multi_process_adapter,
        "lmcache.integration.sglang.sglang_adapter": sglang_adapter,
        "lmcache.integration.sglang.utils": utils,
    }


@contextmanager
def _import_lmc_module() -> Iterator[types.ModuleType]:
    existing_module = sys.modules.pop(_MODULE_NAME, None)
    parent_module = sys.modules.get(_PARENT_MODULE_NAME)
    existing_parent_attribute = (
        getattr(parent_module, _MODULE_ATTRIBUTE, _MISSING)
        if parent_module is not None
        else _MISSING
    )
    try:
        with patch.dict("sys.modules", _fake_lmcache_modules()):
            yield importlib.import_module(_MODULE_NAME)
    finally:
        sys.modules.pop(_MODULE_NAME, None)
        if existing_module is not None:
            sys.modules[_MODULE_NAME] = existing_module
        parent_module = sys.modules.get(_PARENT_MODULE_NAME)
        if parent_module is not None:
            if existing_parent_attribute is _MISSING:
                parent_module.__dict__.pop(_MODULE_ATTRIBUTE, None)
            else:
                setattr(parent_module, _MODULE_ATTRIBUTE, existing_parent_attribute)


class _FakeAllocator:
    def __init__(self, page_size: int):
        self.page_size = page_size
        self.alloc_requests: list[int] = []
        self.free_calls: list[torch.Tensor] = []

    def available_size(self) -> int:
        return 1 << 20

    def alloc(self, num_tokens: int) -> torch.Tensor:
        self.alloc_requests.append(num_tokens)
        return torch.arange(100, 100 + num_tokens, dtype=torch.int64)

    def free(self, slots: torch.Tensor) -> None:
        self.free_calls.append(slots.clone())


@dataclass
class _LoadBackRun:
    result: Any
    allocator: _FakeAllocator
    last_node: Any
    mappings: list[torch.Tensor]


def _make_cache(
    module: types.ModuleType,
    *,
    allocator_page_size: int,
    storage_page_size: int,
    chunk_size: int,
    mode: Any,
) -> tuple[Any, _FakeAllocator]:
    allocator = _FakeAllocator(page_size=allocator_page_size)
    cache = object.__new__(module.LMCRadixCache)
    cache.token_to_kv_pool_allocator = allocator
    cache.lmcache_connector = SimpleNamespace(chunk_size=lambda: chunk_size)
    cache.device = torch.device("cpu")
    cache.page_size = storage_page_size
    cache._mode = mode
    cache.evictable_size_ = 0
    cache.evict = MagicMock()
    cache._update_leaf_status = MagicMock()
    cache._record_store_event = MagicMock()
    return cache, allocator


def _load_back(
    module: types.ModuleType,
    *,
    allocator_page_size: int,
    storage_page_size: int,
    value_numel: int,
    uncached_len: int,
    num_retrieved: int,
) -> _LoadBackRun:
    cache, allocator = _make_cache(
        module,
        allocator_page_size=allocator_page_size,
        storage_page_size=storage_page_size,
        chunk_size=4,
        mode=module.LMCacheMode.MP,
    )
    last_node = module.TreeNode(priority=7)
    key = module.RadixKey(array("q", range(value_numel + uncached_len)))
    mappings: list[torch.Tensor] = []

    def load_fn(slot_mapping: torch.Tensor, _prefix_pad: int) -> int:
        mappings.append(slot_mapping.clone())
        return num_retrieved

    result = cache._load_back(
        key=key,
        value_numel=value_numel,
        uncached_len=uncached_len,
        last_node=last_node,
        load_fn=load_fn,
    )
    return _LoadBackRun(
        result=result,
        allocator=allocator,
        last_node=last_node,
        mappings=mappings,
    )


class TestLMCachePageAlignedLoadBack(unittest.TestCase):
    def test_partial_hit_publishes_complete_pages_and_frees_tail_once(self):
        """A partial trailing page remains allocator-owned."""
        with _import_lmc_module() as module:
            run = _load_back(
                module,
                allocator_page_size=4,
                storage_page_size=2,
                value_numel=4,
                uncached_len=7,
                num_retrieved=6,
            )

            self.assertIsNotNone(run.result)
            published_slots, new_node = run.result
            self.assertEqual(run.allocator.alloc_requests, [8])
            self.assertEqual(
                run.mappings[0].tolist(), [-1] * 4 + list(range(100, 107))
            )
            self.assertEqual(published_slots.tolist(), list(range(100, 104)))
            self.assertEqual(new_node.value.tolist(), list(range(100, 104)))
            self.assertEqual(list(new_node.key), list(range(4, 8)))
            self.assertEqual(list(run.last_node.children.values()), [new_node])
            self.assertEqual(
                [slots.tolist() for slots in run.allocator.free_calls],
                [list(range(104, 108))],
            )

    def test_subpage_hit_releases_the_complete_allocation(self):
        """A hit shorter than one allocator page creates no radix child."""
        with _import_lmc_module() as module:
            run = _load_back(
                module,
                allocator_page_size=4,
                storage_page_size=2,
                value_numel=4,
                uncached_len=7,
                num_retrieved=3,
            )

            self.assertIsNone(run.result)
            self.assertEqual(len(run.last_node.children), 0)
            self.assertEqual(
                [slots.tolist() for slots in run.allocator.free_calls],
                [list(range(100, 108))],
            )

    def test_invalid_retrieval_count_releases_before_raising(self):
        """An oversized connector result cannot retain allocated pages."""
        with _import_lmc_module() as module:
            cache, allocator = _make_cache(
                module,
                allocator_page_size=4,
                storage_page_size=2,
                chunk_size=4,
                mode=module.LMCacheMode.MP,
            )
            key = module.RadixKey(array("q", range(11)))

            with self.assertRaisesRegex(ValueError, "more uncached tokens"):
                cache._load_back(
                    key=key,
                    value_numel=4,
                    uncached_len=7,
                    last_node=module.TreeNode(priority=7),
                    load_fn=lambda _slot_mapping, _prefix_pad: 8,
                )

            self.assertEqual(
                [slots.tolist() for slots in allocator.free_calls],
                [list(range(100, 108))],
            )

    def test_layerwise_large_page_rejects_before_external_setup(self):
        """Layerwise mode rejects multi-token pages during construction."""
        with _import_lmc_module() as module:
            allocator = MagicMock(page_size=4)

            def initialize_radix(cache: Any, _params: Any) -> None:
                cache.token_to_kv_pool_allocator = allocator

            class IPLMCRadixCache(module.LMCRadixCache):
                _mode = module.LMCacheMode.IP

            with (
                patch.object(module.RadixCache, "__init__", new=initialize_radix),
                patch.object(module, "get_server_args") as get_server_args,
                patch.object(module.torch.cuda, "Stream") as stream,
                patch.object(module, "LMCacheMPConnector") as mp_connector,
                patch.object(
                    module, "LMCacheLayerwiseConnector"
                ) as layerwise_connector,
            ):
                with self.assertRaisesRegex(
                    ValueError, "requires allocator page size 1"
                ):
                    IPLMCRadixCache(object())

            get_server_args.assert_not_called()
            allocator.get_kvcache.assert_not_called()
            stream.assert_not_called()
            mp_connector.assert_not_called()
            layerwise_connector.assert_not_called()


if __name__ == "__main__":
    unittest.main()
