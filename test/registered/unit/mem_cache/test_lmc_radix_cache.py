import importlib
import sys
import threading
import types
import unittest
from array import array
from contextlib import contextmanager
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Iterator
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.mem_cache.base_prefix_cache import InsertParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
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
    cache.page_size = allocator_page_size
    cache._storage_page_size = storage_page_size
    cache._allocator_page_size = allocator_page_size
    cache._tp_size = 1
    cache._tp_cpu_group = None
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
            self.assertEqual(run.mappings[0].tolist(), [-1] * 4 + list(range(100, 107)))
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
            params = SimpleNamespace(
                page_size=2,
                token_to_kv_pool_allocator=allocator,
            )

            class IPLMCRadixCache(module.LMCRadixCache):
                _mode = module.LMCacheMode.IP

            with (
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
                    IPLMCRadixCache(params)

            get_server_args.assert_not_called()
            allocator.get_kvcache.assert_not_called()
            stream.assert_not_called()
            mp_connector.assert_not_called()
            layerwise_connector.assert_not_called()

    def test_incompatible_page_sizes_reject_before_external_setup(self):
        """A storage page that cannot divide allocator pages fails early."""
        with _import_lmc_module() as module:
            allocator = MagicMock(page_size=4)
            params = SimpleNamespace(
                page_size=3,
                token_to_kv_pool_allocator=allocator,
            )

            with (
                patch.object(module.RadixCache, "__init__") as initialize_radix,
                patch.object(module, "get_server_args") as get_server_args,
                patch.object(module.torch.cuda, "Stream") as stream,
            ):
                with self.assertRaisesRegex(ValueError, "storage page size to divide"):
                    module.LMCRadixCache(params)

            initialize_radix.assert_not_called()
            get_server_args.assert_not_called()
            stream.assert_not_called()

    def test_constructor_splits_allocator_and_storage_page_authority(self):
        """The base radix owns allocator pages while the connector keeps storage pages."""
        with _import_lmc_module() as module:
            kvcache = SimpleNamespace(
                k_buffer=[torch.empty(0)], v_buffer=[torch.empty(0)]
            )
            allocator = MagicMock(page_size=4, device="cpu", _kvcache=kvcache)
            allocator.get_kvcache.return_value = kvcache
            params = CacheInitParams(
                disable=False,
                req_to_token_pool=MagicMock(),
                token_to_kv_pool_allocator=allocator,
                page_size=2,
            )
            base_page_sizes: list[int] = []

            def initialize_base(cache: Any, base_params: CacheInitParams) -> None:
                base_page_sizes.append(base_params.page_size)
                cache.token_to_kv_pool_allocator = allocator
                cache.page_size = base_params.page_size

            tp_group = SimpleNamespace(device_group="device", cpu_group="cpu")
            mp_connector = MagicMock()
            with (
                patch.object(module.RadixCache, "__init__", new=initialize_base),
                patch.object(
                    module,
                    "get_server_args",
                    return_value=SimpleNamespace(lmcache_config_file="config.yaml"),
                ),
                patch.object(
                    module,
                    "lmcache_get_config",
                    return_value=SimpleNamespace(mp_host="host", mp_port=1),
                ),
                patch.object(module, "LMCacheMPConnector", mp_connector),
                patch.object(module.torch.cuda, "Stream", return_value=MagicMock()),
            ):
                cache = module.LMCRadixCache(
                    params,
                    tp_size=2,
                    tp_group=tp_group,
                )

            self.assertEqual(base_page_sizes, [4])
            self.assertEqual(cache.page_size, 4)
            self.assertEqual(cache._storage_page_size, 2)
            self.assertEqual(cache._allocator_page_size, 4)
            self.assertEqual(cache._tp_cpu_group, "cpu")
            self.assertEqual(mp_connector.call_args.kwargs["page_size"], 2)
            self.assertEqual(mp_connector.call_args.kwargs["tp_group"], "device")


class TestLMCacheCanonicalStore(unittest.TestCase):
    def _make_store_cache(
        self,
        module: types.ModuleType,
        *,
        is_eagle: bool = False,
    ) -> tuple[Any, Any, Any]:
        req_to_token_pool = MagicMock()
        req_to_token_pool.req_to_token = torch.tensor(
            [[200, 201, 202, 203, 204, 205, 206, 207]], dtype=torch.int64
        )
        allocator = MagicMock(page_size=4, device="cpu")
        cache = object.__new__(module.LMCRadixCache)
        module.RadixCache.__init__(
            cache,
            CacheInitParams(
                disable=False,
                req_to_token_pool=req_to_token_pool,
                token_to_kv_pool_allocator=allocator,
                page_size=4,
                is_eagle=is_eagle,
            ),
        )
        cache._mode = module.LMCacheMode.MP
        cache._storage_page_size = 2
        cache._allocator_page_size = 4
        cache._tp_size = 1
        cache._tp_cpu_group = None
        cache._mp_load_back_markers = {"request": MagicMock()}
        cache._in_flight_nodes = []
        cache._node_lock = threading.Lock()
        cache.store_stream = MagicMock()
        cache.lmcache_connector = MagicMock()
        canonical_indices = torch.tensor([100, 101, 102, 103], dtype=torch.int64)
        canonical_token_ids = (
            [10, 11, 12, 13, 14] if is_eagle else [10, 11, 12, 13]
        )
        cache.insert(
            InsertParams(
                key=module.RadixKey(
                    array("q", canonical_token_ids),
                    is_bigram=is_eagle,
                ),
                value=canonical_indices,
            )
        )
        req = SimpleNamespace(
            rid="request",
            origin_input_ids=[10, 11, 12, 13],
            output_ids=[14, 15, 16, 17],
            extra_key=None,
            req_pool_idx=0,
            cache_protected_len=0,
            last_node=cache.root_node,
            priority=0,
            kv_committed_len=8,
        )
        return cache, allocator, req

    def test_duplicate_slots_store_from_canonical_handoff_boundary(self):
        """Duplicate request slots are freed before canonical slots back the store."""
        with _import_lmc_module() as module:
            cache, allocator, req = self._make_store_cache(module)
            with patch.object(module.torch.cuda, "stream", return_value=MagicMock()):
                result = cache.cache_finished_req(
                    req,
                    kv_len_to_handle=4,
                )

            store_metadata = cache.lmcache_connector.store_kv.call_args.args[0]
            self.assertEqual(result.unhandled_kv_start, 4)
            self.assertEqual(store_metadata.token_ids, [10, 11, 12, 13])
            self.assertEqual(store_metadata.kv_indices.tolist(), [100, 101, 102, 103])
            self.assertEqual(
                allocator.free.call_args.args[0].tolist(), [200, 201, 202, 203]
            )
            cache.lmcache_connector.end_session.assert_called_once_with("request")
            self.assertNotIn("request", cache._mp_load_back_markers)

    def test_eagle_store_rematches_with_boundary_token(self):
        """Eagle canonical rematch retains its final raw boundary token."""
        with _import_lmc_module() as module:
            cache, allocator, req = self._make_store_cache(module, is_eagle=True)
            with patch.object(module.torch.cuda, "stream", return_value=MagicMock()):
                result = cache.cache_finished_req(
                    req,
                    kv_len_to_handle=5,
                )

            store_metadata = cache.lmcache_connector.store_kv.call_args.args[0]
            self.assertEqual(result.unhandled_kv_start, 4)
            self.assertEqual(store_metadata.token_ids, [10, 11, 12, 13])
            self.assertEqual(store_metadata.kv_indices.tolist(), [100, 101, 102, 103])
            self.assertEqual(
                allocator.free.call_args.args[0].tolist(), [200, 201, 202, 203]
            )
            cache.lmcache_connector.end_session.assert_called_once_with("request")

    def test_zero_handoff_ends_mp_session_without_store(self):
        """A subpage base handoff closes its MP session without storing."""
        with _import_lmc_module() as module:
            cache, _allocator, req = self._make_store_cache(module)

            result = cache.cache_finished_req(req, kv_len_to_handle=2)

            self.assertEqual(result.unhandled_kv_start, 0)
            cache.lmcache_connector.store_kv.assert_not_called()
            cache.lmcache_connector.end_session.assert_called_once_with("request")
            self.assertNotIn("request", cache._mp_load_back_markers)

    def test_asymmetric_canonical_failure_rejects_before_store(self):
        """A remote preparation failure preserves fixed collectives and skips store."""
        with _import_lmc_module() as module:
            cache, _allocator, req = self._make_store_cache(module)
            cache._tp_size = 2
            cache._tp_cpu_group = "cpu"
            cache.inc_lock_ref = MagicMock()
            reduced_values = iter([0, 4, -4])

            def reduce_scalar(tensor: torch.Tensor, **_kwargs: Any) -> None:
                tensor.fill_(next(reduced_values))

            with (
                patch.object(
                    module.torch.distributed, "all_reduce", side_effect=reduce_scalar
                ) as reduce,
                patch.object(module.torch.cuda, "stream", return_value=MagicMock()),
            ):
                result = cache.cache_finished_req(req, kv_len_to_handle=4)

            self.assertEqual(result.unhandled_kv_start, 4)
            self.assertEqual(reduce.call_count, 3)
            cache.inc_lock_ref.assert_not_called()
            cache.lmcache_connector.store_kv.assert_not_called()
            cache.lmcache_connector.end_session.assert_called_once_with("request")

    def test_asymmetric_handoff_boundaries_reject_after_three_collectives(self):
        """Different TP handoff boundaries are rejected after the fixed protocol."""
        with _import_lmc_module() as module:
            cache, _allocator, _req = self._make_store_cache(module)
            cache._tp_size = 2
            cache._tp_cpu_group = "cpu"
            reduced_values = iter([1, 4, -8])

            def reduce_scalar(tensor: torch.Tensor, **_kwargs: Any) -> None:
                tensor.fill_(next(reduced_values))

            with patch.object(
                module.torch.distributed,
                "all_reduce",
                side_effect=reduce_scalar,
            ) as reduce:
                coordinated = cache._coordinate_store_preparation(
                    local_status=1,
                    boundary=4,
                )

            self.assertFalse(coordinated)
            self.assertEqual(reduce.call_count, 3)


if __name__ == "__main__":
    unittest.main()
