import importlib
import sys
import types
import unittest
from array import array
from contextlib import contextmanager
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Iterator, Optional
from unittest.mock import MagicMock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

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
    def __init__(self, page_size: int, events: list[tuple[str, list[int]]]):
        self.page_size = page_size
        self.events = events
        self.alloc_requests: list[int] = []
        self.free_calls: list[torch.Tensor] = []

    def available_size(self) -> int:
        return 1 << 20

    def alloc(self, num_tokens: int) -> torch.Tensor:
        self.alloc_requests.append(num_tokens)
        return torch.arange(100, 100 + num_tokens, dtype=torch.int64)

    def free(self, slots: torch.Tensor) -> None:
        released = slots.clone()
        self.free_calls.append(released)
        self.events.append(("free", released.tolist()))


@dataclass
class _CacheHarness:
    cache: Any
    allocator: _FakeAllocator
    events: list[tuple[str, list[int]]]


@dataclass
class _LoadBackRun:
    cache: Any
    allocator: _FakeAllocator
    last_node: Any
    result: Optional[tuple[torch.Tensor, Any]]
    slot_mapping: torch.Tensor
    prefix_pad: int
    events: list[tuple[str, list[int]]]


def _make_load_back_cache(
    module: types.ModuleType,
    *,
    allocator_page_size: int,
    configured_page_size: int,
    chunk_size: int,
) -> _CacheHarness:
    events: list[tuple[str, list[int]]] = []
    allocator = _FakeAllocator(page_size=allocator_page_size, events=events)
    cache = object.__new__(module.LMCRadixCache)
    cache.token_to_kv_pool_allocator = allocator
    cache.lmcache_connector = SimpleNamespace(chunk_size=lambda: chunk_size)
    cache.device = torch.device("cpu")
    cache.page_size = configured_page_size
    cache.evictable_size_ = 0
    cache.evict = MagicMock()
    cache._update_leaf_status = MagicMock()
    cache._record_store_event = MagicMock()
    return _CacheHarness(cache=cache, allocator=allocator, events=events)


def _load_back(
    module: types.ModuleType,
    *,
    allocator_page_size: int,
    configured_page_size: int,
    chunk_size: int,
    value_numel: int,
    uncached_len: int,
    num_retrieved: int,
) -> _LoadBackRun:
    harness = _make_load_back_cache(
        module,
        allocator_page_size=allocator_page_size,
        configured_page_size=configured_page_size,
        chunk_size=chunk_size,
    )
    last_node = module.TreeNode(priority=7)
    key = module.RadixKey(array("q", range(value_numel + uncached_len)))
    captured_mapping: list[torch.Tensor] = []
    captured_prefix_pad: list[int] = []

    def load_fn(slot_mapping: torch.Tensor, prefix_pad: int) -> int:
        captured_mapping.append(slot_mapping.clone())
        captured_prefix_pad.append(prefix_pad)
        harness.events.append(("load", slot_mapping.tolist()))
        return num_retrieved

    result = harness.cache._load_back(
        key=key,
        value_numel=value_numel,
        uncached_len=uncached_len,
        last_node=last_node,
        load_fn=load_fn,
    )
    return _LoadBackRun(
        cache=harness.cache,
        allocator=harness.allocator,
        last_node=last_node,
        result=result,
        slot_mapping=captured_mapping[0],
        prefix_pad=captured_prefix_pad[0],
        events=harness.events,
    )


class TestLMCachePageAlignedLoadBack(CustomTestCase):
    def test_partial_hit_publishes_only_complete_allocator_pages(self):
        """A partial trailing page stays out of every radix ownership field."""
        with _import_lmc_module() as module:
            run = _load_back(
                module,
                allocator_page_size=4,
                configured_page_size=1,
                chunk_size=4,
                value_numel=4,
                uncached_len=7,
                num_retrieved=6,
            )

            self.assertIsNotNone(run.result)
            published_slots, new_node = run.result
            self.assertEqual(run.allocator.alloc_requests, [8])
            self.assertEqual(
                run.slot_mapping.tolist(),
                [-1] * 4 + list(range(100, 107)),
            )
            self.assertEqual(run.prefix_pad, 0)
            self.assertEqual(published_slots.tolist(), list(range(100, 104)))
            self.assertEqual(new_node.value.tolist(), list(range(100, 104)))
            self.assertEqual(list(new_node.key), list(range(4, 8)))
            self.assertIs(new_node.parent, run.last_node)
            self.assertEqual(list(run.last_node.children.values()), [new_node])
            self.assertEqual(run.cache.evictable_size_, 4)
            self.assertEqual(
                [slots.tolist() for slots in run.allocator.free_calls],
                [list(range(104, 108))],
            )
            self.assertEqual(
                [event[0] for event in run.events],
                ["load", "free"],
            )

    def test_subpage_hit_does_not_create_empty_child(self):
        """A hit shorter than one allocator page releases the complete lease."""
        with _import_lmc_module() as module:
            run = _load_back(
                module,
                allocator_page_size=4,
                configured_page_size=1,
                chunk_size=4,
                value_numel=4,
                uncached_len=7,
                num_retrieved=3,
            )

            self.assertIsNone(run.result)
            self.assertEqual(len(run.last_node.children), 0)
            self.assertEqual(run.cache.evictable_size_, 0)
            self.assertEqual(
                [slots.tolist() for slots in run.allocator.free_calls],
                [list(range(100, 108))],
            )
            run.cache._update_leaf_status.assert_not_called()
            run.cache._record_store_event.assert_not_called()

    def test_exact_page_and_full_allocation_hits_keep_complete_pages(self):
        """Exact page boundaries neither drop live slots nor free empty tails."""
        with _import_lmc_module() as module:
            cases = ((7, 4, [list(range(104, 108))]), (8, 8, []))
            for uncached_len, num_retrieved, expected_frees in cases:
                with self.subTest(
                    uncached_len=uncached_len,
                    num_retrieved=num_retrieved,
                ):
                    run = _load_back(
                        module,
                        allocator_page_size=4,
                        configured_page_size=1,
                        chunk_size=4,
                        value_numel=4,
                        uncached_len=uncached_len,
                        num_retrieved=num_retrieved,
                    )

                    self.assertIsNotNone(run.result)
                    published_slots, new_node = run.result
                    self.assertEqual(
                        published_slots.numel(),
                        num_retrieved,
                    )
                    self.assertEqual(len(new_node.key), num_retrieved)
                    self.assertEqual(run.cache.evictable_size_, num_retrieved)
                    self.assertEqual(
                        [slots.tolist() for slots in run.allocator.free_calls],
                        expected_frees,
                    )

    def test_zero_retrieval_precedes_prefix_pad_validation(self):
        """A zero-first miss with prefix padding releases the lease normally."""
        with _import_lmc_module() as module:
            run = _load_back(
                module,
                allocator_page_size=4,
                configured_page_size=1,
                chunk_size=6,
                value_numel=4,
                uncached_len=7,
                num_retrieved=0,
            )

            self.assertEqual(run.prefix_pad, 4)
            self.assertIsNone(run.result)
            self.assertEqual(len(run.last_node.children), 0)
            self.assertEqual(run.cache.evictable_size_, 0)
            self.assertEqual(
                [slots.tolist() for slots in run.allocator.free_calls],
                [list(range(100, 108))],
            )

    def test_invalid_positive_counts_release_lease_before_failing(self):
        """Positive connector contract violations clean the full allocation."""
        with _import_lmc_module() as module:
            for num_retrieved, message in (
                (3, "smaller than its prefix pad"),
                (12, "more uncached tokens"),
            ):
                with self.subTest(num_retrieved=num_retrieved):
                    harness = _make_load_back_cache(
                        module,
                        allocator_page_size=4,
                        configured_page_size=1,
                        chunk_size=6,
                    )
                    last_node = module.TreeNode(priority=0)
                    key = module.RadixKey(array("q", range(11)))

                    with self.assertRaisesRegex(ValueError, message):
                        harness.cache._load_back(
                            key=key,
                            value_numel=4,
                            uncached_len=7,
                            last_node=last_node,
                            load_fn=lambda _mapping, _pad: num_retrieved,
                        )

                    self.assertEqual(len(last_node.children), 0)
                    self.assertEqual(harness.cache.evictable_size_, 0)
                    self.assertEqual(
                        [slots.tolist() for slots in harness.allocator.free_calls],
                        [list(range(100, 108))],
                    )

    def test_node_eviction_and_tail_release_have_disjoint_ownership(self):
        """Evicting a published node never frees its discarded tail twice."""
        with _import_lmc_module() as module:
            run = _load_back(
                module,
                allocator_page_size=4,
                configured_page_size=1,
                chunk_size=4,
                value_numel=4,
                uncached_len=7,
                num_retrieved=6,
            )

            self.assertIsNotNone(run.result)
            _, new_node = run.result
            tail_slots = set(run.allocator.free_calls[0].tolist())
            run.allocator.free(new_node.value)
            node_slots = set(run.allocator.free_calls[1].tolist())
            self.assertTrue(tail_slots.isdisjoint(node_slots))
            self.assertEqual(tail_slots | node_slots, set(range(100, 108)))

    def test_actual_allocator_page_controls_dcp_style_load_back(self):
        """Allocation and publication use actual pages over configured pages."""
        with _import_lmc_module() as module:
            run = _load_back(
                module,
                allocator_page_size=8,
                configured_page_size=2,
                chunk_size=8,
                value_numel=8,
                uncached_len=9,
                num_retrieved=9,
            )

            self.assertIsNotNone(run.result)
            published_slots, new_node = run.result
            self.assertEqual(run.allocator.alloc_requests, [16])
            self.assertEqual(run.slot_mapping.numel(), 17)
            self.assertEqual(
                run.slot_mapping[8:].tolist(),
                list(range(100, 109)),
            )
            self.assertEqual(published_slots.numel(), 8)
            self.assertEqual(len(new_node.key), 8)
            self.assertEqual(run.cache.evictable_size_, 8)
            self.assertEqual(
                [slots.tolist() for slots in run.allocator.free_calls],
                [list(range(108, 116))],
            )

    def test_layerwise_page_gate_precedes_external_side_effects(self):
        """Unsupported IP pages fail before global, CUDA, or connector work."""
        with _import_lmc_module() as module:
            allocator = MagicMock(page_size=4)

            def initialize_radix(cache, _params) -> None:
                cache.token_to_kv_pool_allocator = allocator

            class IPLMCRadixCache(module.LMCRadixCache):
                _mode = module.LMCacheMode.IP

            with (
                patch.object(module.RadixCache, "__init__", new=initialize_radix),
                patch.object(module, "get_global_server_args") as get_global_args,
                patch.object(module, "lmcache_get_config") as get_config,
                patch.object(module.torch.cuda, "Stream") as stream,
                patch.object(module, "LMCacheLayerwiseConnector") as connector,
                patch.object(module, "LayerTransferCounter") as counter,
            ):
                with self.assertRaisesRegex(
                    ValueError, "requires allocator page size 1"
                ):
                    IPLMCRadixCache(object())

            get_global_args.assert_not_called()
            get_config.assert_not_called()
            allocator.get_kvcache.assert_not_called()
            stream.assert_not_called()
            connector.assert_not_called()
            counter.assert_not_called()

    def test_mp_accepts_large_pages_and_ip_preserves_page_one_route(self):
        """MP supports actual pages while page-one IP keeps its legacy setup."""
        with _import_lmc_module() as module:
            for mode, allocator_page_size in (
                (module.LMCacheMode.MP, 8),
                (module.LMCacheMode.IP, 1),
            ):
                with self.subTest(mode=mode, allocator_page_size=allocator_page_size):
                    kvcache = SimpleNamespace(
                        k_buffer=object(),
                        v_buffer=object(),
                        register_layer_transfer_counter=MagicMock(),
                    )
                    allocator = SimpleNamespace(
                        page_size=allocator_page_size,
                        _kvcache=kvcache,
                        get_kvcache=MagicMock(return_value=kvcache),
                    )

                    def initialize_radix(cache, _params) -> None:
                        cache.token_to_kv_pool_allocator = allocator

                    class SelectedLMCRadixCache(module.LMCRadixCache):
                        _mode = mode

                    params = SimpleNamespace(page_size=2)
                    model_config = SimpleNamespace(num_hidden_layers=3)
                    with (
                        patch.object(
                            module.RadixCache,
                            "__init__",
                            new=initialize_radix,
                        ),
                        patch.object(
                            module,
                            "get_global_server_args",
                            return_value=SimpleNamespace(
                                lmcache_config_file="cfg.yaml"
                            ),
                        ),
                        patch.object(
                            module,
                            "lmcache_get_config",
                            return_value=SimpleNamespace(mp_host="host", mp_port=1234),
                        ),
                        patch.object(module.torch.cuda, "Stream"),
                        patch.object(module, "LMCacheMPConnector") as mp_connector,
                        patch.object(
                            module,
                            "LMCacheLayerwiseConnector",
                        ) as layerwise_connector,
                    ):
                        SelectedLMCRadixCache(
                            params,
                            model_config=model_config,
                        )

                    if mode is module.LMCacheMode.MP:
                        mp_connector.assert_called_once()
                        layerwise_connector.assert_not_called()
                        kvcache.register_layer_transfer_counter.assert_not_called()
                    else:
                        layerwise_connector.assert_called_once()
                        mp_connector.assert_not_called()
                        kvcache.register_layer_transfer_counter.assert_called_once()


if __name__ == "__main__":
    unittest.main(verbosity=3)
