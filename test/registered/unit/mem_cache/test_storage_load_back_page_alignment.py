import sys
import unittest
from array import array
from types import SimpleNamespace
from typing import Any, List, Optional
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.mem_cache.base_prefix_cache import InitLoadBackParams, MatchResult
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.radix_cache import RadixKey, TreeNode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_LMCACHE_MODULES = [
    "lmcache",
    "lmcache.integration",
    "lmcache.integration.sglang",
    "lmcache.integration.sglang.multi_process_adapter",
    "lmcache.integration.sglang.sglang_adapter",
    "lmcache.integration.sglang.utils",
]

_FLEXKV_MODULES = [
    "flexkv",
    "flexkv.common",
    "flexkv.common.request",
    "flexkv.common.storage",
    "flexkv.integration",
    "flexkv.integration.config",
    "flexkv.kvmanager",
    "flexkv.server",
    "flexkv.server.client",
    "flexkv.transfer",
    "flexkv.transfer.layerwise",
    "flexkv.transfer_manager",
]


def _import_under_stub_modules(stub_names: List[str], module_name: str):
    with patch.dict(sys.modules, {name: MagicMock() for name in stub_names}):
        __import__(module_name)
        return sys.modules[module_name]


_LMC_MODULE = _import_under_stub_modules(
    _LMCACHE_MODULES, "sglang.srt.mem_cache.storage.lmcache.lmc_radix_cache"
)
_FLEXKV_MODULE = _import_under_stub_modules(
    _FLEXKV_MODULES, "sglang.srt.mem_cache.storage.flexkv.flexkv_radix_cache"
)


class _LoadMetadata:
    def __init__(
        self,
        *,
        token_ids: Any,
        slot_mapping: torch.Tensor,
        offset: int,
        prefix_pad: int = 0,
        request_id: Optional[str] = None,
    ) -> None:
        self.token_ids = token_ids
        self.slot_mapping = slot_mapping
        self.offset = offset
        self.prefix_pad = prefix_pad
        self.request_id = request_id


class _FakeAllocator:
    def __init__(self, page_size: int, slots: torch.Tensor) -> None:
        self.page_size = page_size
        self.slots = slots
        self.freed: List[List[int]] = []

    def available_size(self) -> int:
        return self.slots.numel()

    def alloc(self, need_size: int) -> torch.Tensor:
        return self.slots[:need_size]

    def free(self, free_index: torch.Tensor) -> None:
        self.freed.append(free_index.tolist())

    def freed_slots(self) -> set:
        return {slot for batch in self.freed for slot in batch}


class _FakeLayerwiseConnector:
    def __init__(self, chunk_size: int, num_retrieved: int) -> None:
        self._chunk_size = chunk_size
        self._num_retrieved = num_retrieved
        self.pending_writes: set = set()

    def chunk_size(self) -> int:
        return self._chunk_size

    def start_load_kv(self, load_metadata: _LoadMetadata) -> int:
        offset = load_metadata.offset
        written = load_metadata.slot_mapping[offset : offset + self._num_retrieved]
        self.pending_writes = {int(s) for s in written.tolist() if s != -1}
        return self._num_retrieved


class _FakeBlockingConnector:
    def __init__(self, chunk_size: int, num_retrieved: int) -> None:
        self._chunk_size = chunk_size
        self._num_retrieved = num_retrieved
        self.released: List[str] = []

    def chunk_size(self) -> int:
        return self._chunk_size

    def retrieve_kv(self, load_metadata: _LoadMetadata) -> int:
        return self._num_retrieved

    def release_pending(self, rid: str) -> None:
        self.released.append(rid)


class _FakeReq:
    rid = "req-a"


def _make_lmc_cache(
    *,
    alloc_page_size: int,
    cache_page_size: int,
    slots: torch.Tensor,
    connector: Any,
):
    cache = object.__new__(_LMC_MODULE.LMCRadixCache)
    cache.token_to_kv_pool_allocator = _FakeAllocator(alloc_page_size, slots)
    cache.lmcache_connector = connector
    cache.device = "cpu"
    cache.page_size = cache_page_size
    cache.evictable_size_ = 0
    cache.evictable_leaves = set()
    cache.enable_kv_cache_events = False
    cache.load_stream = None
    return cache


def _run_ip_match_prefix(cache, *, value_numel: int, uncached_len: int) -> MatchResult:
    key = RadixKey(array("q", list(range(value_numel + uncached_len))))
    value = torch.arange(value_numel, dtype=torch.int64)
    last_node = TreeNode()
    base_res = MatchResult(
        device_indices=value,
        last_device_node=last_node,
        last_host_node=last_node,
        best_match_node=last_node,
    )
    with patch.object(_LMC_MODULE, "LoadMetadata", _LoadMetadata):
        return cache._ip_match_prefix(key, base_res, value, last_node)


def _run_mp_init_load_back(cache, *, value_numel: int, uncached_len: int):
    module = _LMC_MODULE
    key = RadixKey(array("q", list(range(value_numel + uncached_len))))
    req = _FakeReq()
    cache._mp_load_back_markers = {
        req.rid: module._LMCacheLoadBackMarker(key=key, value_numel=value_numel)
    }
    with patch.object(module, "LoadMetadata", _LoadMetadata):
        with patch.object(
            module.LMCRadixCache,
            "_mp_load_back",
            lambda self, **kwargs: self.lmcache_connector.retrieve_kv(
                _LoadMetadata(
                    token_ids=None,
                    slot_mapping=kwargs["slot_mapping"],
                    offset=0,
                )
            ),
        ):
            return cache.init_load_back(
                InitLoadBackParams(
                    best_match_node=TreeNode(),
                    host_hit_length=uncached_len,
                    req=req,
                )
            )


class TestLMCacheLayerwiseLoadBackOwnership(CustomTestCase):
    def test_discarded_tail_never_overlaps_the_layerwise_load_pending_writes(self):
        """Releasing a sub-page tail of a layerwise load hands slots it still writes back to the allocator."""
        slots = torch.arange(100, 108, dtype=torch.int64)
        connector = _FakeLayerwiseConnector(chunk_size=4, num_retrieved=8)
        cache = _make_lmc_cache(
            alloc_page_size=4, cache_page_size=4, slots=slots, connector=connector
        )

        _run_ip_match_prefix(cache, value_numel=6, uncached_len=8)

        prefix_pad = 6 % 4
        raw_fetched = 8 - prefix_pad
        slots_the_later_layers_still_write = set(range(100, 100 + raw_fetched))
        slots_a_page_floor_would_have_released = set(range(104, 108))

        self.assertEqual(connector.pending_writes, slots_the_later_layers_still_write)
        self.assertEqual(
            slots_the_later_layers_still_write & slots_a_page_floor_would_have_released,
            {104, 105},
        )
        self.assertEqual(
            connector.pending_writes & cache.token_to_kv_pool_allocator.freed_slots(),
            set(),
        )

    def test_layerwise_load_keeps_every_written_slot_in_the_node(self):
        """A layerwise load's written slot dropped from the node would be leaked, not just unaligned."""
        slots = torch.arange(100, 108, dtype=torch.int64)
        connector = _FakeLayerwiseConnector(chunk_size=4, num_retrieved=8)
        cache = _make_lmc_cache(
            alloc_page_size=4, cache_page_size=4, slots=slots, connector=connector
        )

        result = _run_ip_match_prefix(cache, value_numel=6, uncached_len=8)

        self.assertEqual(result.best_match_node.value.tolist(), list(range(100, 106)))


class TestLMCacheBlockingLoadBackPageAlignment(CustomTestCase):
    def _run(
        self,
        *,
        alloc_page_size: int,
        cache_page_size: int,
        chunk_size: int,
        num_retrieved: int,
        slots: torch.Tensor,
        value_numel: int,
        uncached_len: int,
    ):
        connector = _FakeBlockingConnector(
            chunk_size=chunk_size, num_retrieved=num_retrieved
        )
        cache = _make_lmc_cache(
            alloc_page_size=alloc_page_size,
            cache_page_size=cache_page_size,
            slots=slots,
            connector=connector,
        )
        result = _run_mp_init_load_back(
            cache, value_numel=value_numel, uncached_len=uncached_len
        )
        return cache, result

    def test_blocking_load_floors_an_unaligned_fetch_to_the_allocator_page(self):
        """A blocking load is settled on return, so its sub-page remainder is released rather than half-owning a page."""
        slots = torch.arange(100, 116, dtype=torch.int64)
        cache, result = self._run(
            alloc_page_size=4,
            cache_page_size=4,
            chunk_size=4,
            num_retrieved=6,
            slots=slots,
            value_numel=8,
            uncached_len=16,
        )

        slot_values, node = result
        self.assertEqual(slot_values.tolist(), list(range(100, 104)))
        self.assertEqual(node.value.tolist(), list(range(100, 104)))
        self.assertEqual(len(node.key), 4)
        self.assertEqual(cache.evictable_size_, 4)
        self.assertEqual(cache.token_to_kv_pool_allocator.freed, [list(range(104, 116))])

    def test_blocking_load_accepts_a_chunk_size_that_does_not_divide_the_page(self):
        """A chunk size that does not divide the page can still land an aligned fetch, and must load."""
        slots = torch.arange(100, 118, dtype=torch.int64)
        cache, result = self._run(
            alloc_page_size=6,
            cache_page_size=6,
            chunk_size=4,
            num_retrieved=8,
            slots=slots,
            value_numel=6,
            uncached_len=12,
        )

        slot_values, _ = result
        self.assertEqual(slot_values.tolist(), list(range(100, 106)))
        self.assertEqual(cache.evictable_size_, 6)

    def test_blocking_load_gives_up_when_the_fetch_floors_to_zero(self):
        """A fetch shorter than one page owns nothing, so every slot goes back."""
        slots = torch.arange(100, 116, dtype=torch.int64)
        cache, result = self._run(
            alloc_page_size=8,
            cache_page_size=8,
            chunk_size=8,
            num_retrieved=6,
            slots=slots,
            value_numel=8,
            uncached_len=16,
        )

        slot_values, _ = result
        self.assertEqual(slot_values.numel(), 0)
        self.assertEqual(cache.evictable_size_, 0)
        self.assertEqual(cache.token_to_kv_pool_allocator.freed, [list(range(100, 116))])

    def test_blocking_load_handles_a_fetch_shorter_than_the_prefix_pad(self):
        """A fetch below prefix_pad must not produce a negative slice, size or node."""
        slots = torch.arange(100, 116, dtype=torch.int64)
        cache, result = self._run(
            alloc_page_size=4,
            cache_page_size=4,
            chunk_size=8,
            num_retrieved=2,
            slots=slots,
            value_numel=6,
            uncached_len=16,
        )

        slot_values, _ = result
        self.assertEqual(slot_values.numel(), 0)
        self.assertEqual(cache.evictable_size_, 0)
        self.assertEqual(cache.token_to_kv_pool_allocator.freed, [list(range(100, 116))])

    def test_blocking_load_floors_on_the_allocator_page_not_the_cache_page(self):
        """The floor serves the alloc/free contract, so it reads the allocator's page, not the cache's."""
        slots = torch.arange(100, 116, dtype=torch.int64)
        cache, result = self._run(
            alloc_page_size=4,
            cache_page_size=1,
            chunk_size=4,
            num_retrieved=6,
            slots=slots,
            value_numel=8,
            uncached_len=16,
        )

        slot_values, _ = result
        self.assertEqual(slot_values.tolist(), list(range(100, 104)))


class _FakeConstructedConnector:
    def __init__(self, page_size: int, **kwargs: Any) -> None:
        self.page_size = page_size

    def chunk_size(self) -> int:
        return 4


class _FakeKVCache:
    k_buffer: List[Any] = []
    v_buffer: List[Any] = []


class _FakeConstructionAllocator:
    device = "cpu"
    page_size = 6
    _kvcache = _FakeKVCache()

    def get_kvcache(self) -> _FakeKVCache:
        return self._kvcache


class TestLMCacheConstruction(CustomTestCase):
    def test_construction_accepts_a_chunk_size_that_does_not_divide_the_page(self):
        """Startup must not reject page=6 with chunk=4: it is a working configuration."""
        allocator = _FakeConstructionAllocator()
        params = CacheInitParams(
            disable=False,
            req_to_token_pool=None,
            token_to_kv_pool_allocator=allocator,
            page_size=6,
        )

        with patch.object(_LMC_MODULE, "LMCacheMPConnector", _FakeConstructedConnector):
            with patch.object(_LMC_MODULE.torch.cuda, "Stream", lambda: None):
                with patch.object(
                    _LMC_MODULE,
                    "get_server_args",
                    lambda: SimpleNamespace(lmcache_config_file="lmcache.yaml"),
                ):
                    cache = _LMC_MODULE.LMCRadixCache(params=params)

        self.assertEqual(cache.page_size, 6)
        self.assertEqual(cache.lmcache_connector.chunk_size(), 4)


def _make_flexkv_cache(*, alloc_page_size: int, slots: torch.Tensor):
    cache = object.__new__(_FLEXKV_MODULE.FlexKVRadixCache)
    cache.token_to_kv_pool_allocator = _FakeAllocator(alloc_page_size, slots)
    cache.device = "cpu"
    cache.page_size = alloc_page_size
    cache.evictable_size_ = 0
    cache.evictable_leaves = set()
    cache.enable_kv_cache_events = False
    return cache


def _run_allocate_and_load(
    cache, *, value_numel: int, uncached_len: int, num_retrieved: int
):
    key = RadixKey(array("q", list(range(value_numel + uncached_len))))
    return cache._allocate_and_load(
        key=key,
        value_numel=value_numel,
        uncached_len=uncached_len,
        last_node=TreeNode(),
        load_fn=lambda token_slots: num_retrieved,
    )


class TestFlexKVLoadBackPageAlignment(CustomTestCase):
    def test_load_rejects_an_unaligned_retrieval(self):
        """FlexKV blocks are sglang pages, so an unaligned retrieval means that guarantee broke."""
        slots = torch.arange(100, 116, dtype=torch.int64)
        cache = _make_flexkv_cache(alloc_page_size=4, slots=slots)

        with self.assertRaises(AssertionError):
            _run_allocate_and_load(
                cache, value_numel=8, uncached_len=16, num_retrieved=6
            )


if __name__ == "__main__":
    unittest.main()
