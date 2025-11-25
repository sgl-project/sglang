import importlib
import sys
import types
from contextlib import nullcontext
from pathlib import Path

import torch


class DummyStream:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def synchronize(self):
        return None


class DummyReqToTokenPool:
    def __init__(self, mapping: torch.Tensor):
        self.req_to_token = mapping
        self.freed = []

    def free(self, idx, free_mamba_cache=True):
        self.freed.append(idx)


class DummyKVCache:
    def __init__(self):
        self.k_buffer = [torch.zeros((8, 1, 1))]
        self.v_buffer = [torch.zeros((8, 1, 1))]
        self.registered = []

    def register_layer_transfer_counter(self, counter):
        self.registered.append(counter)


class DummyTokenToKVPoolAllocator:
    def __init__(self):
        self.device = torch.device("cpu")
        self._kvcache = DummyKVCache()
        self.freed = []

    def get_kvcache(self):
        return self._kvcache

    def available_size(self):
        return 1024

    def alloc(self, num_tokens: int):
        return torch.arange(num_tokens, dtype=torch.int64, device=self.device)

    def free(self, indices):
        self.freed.append(indices.clone() if torch.is_tensor(indices) else indices)


class DummyReq:
    def __init__(self, token_ids: list[int]):
        self.req_pool_idx = 0
        self.origin_input_ids = token_ids[:-1]
        self.output_ids = token_ids[-1:]
        self.fill_ids = token_ids.copy()
        self.extra_key = "ekey"
        self.cache_protected_len = 0
        self.priority = None
        self.last_node = None
        self._kv_len = len(token_ids)

    def pop_committed_kv_cache(self):
        return self._kv_len


class FakeStoreMetadata:
    def __init__(self, last_node, token_ids, kv_indices, offset):
        self.last_node = last_node
        self.token_ids = token_ids
        self.kv_indices = kv_indices
        self.offset = offset


class FakeLoadMetadata:
    def __init__(self, token_ids, slot_mapping, offset):
        self.token_ids = token_ids
        self.slot_mapping = slot_mapping
        self.offset = offset


class FakeConnector:
    def __init__(self, *args, **kwargs):
        self.store_calls = []

    def chunk_size(self):
        return 1

    def start_load_kv(self, *args, **kwargs):
        return 0

    def store_kv(self, metadata):
        self.store_calls.append(metadata)

    def load_kv_layerwise(self, layer_id):
        return None


def _install_fake_lmcache(monkeypatch):
    adapter = types.ModuleType("lmcache.integration.sglang.sglang_adapter")
    adapter.LMCacheLayerwiseConnector = FakeConnector
    adapter.LoadMetadata = FakeLoadMetadata
    adapter.StoreMetadata = FakeStoreMetadata

    monkeypatch.setitem(sys.modules, "lmcache", types.ModuleType("lmcache"))
    monkeypatch.setitem(
        sys.modules, "lmcache.integration", types.ModuleType("lmcache.integration")
    )
    monkeypatch.setitem(
        sys.modules,
        "lmcache.integration.sglang",
        types.ModuleType("lmcache.integration.sglang"),
    )
    monkeypatch.setitem(
        sys.modules, "lmcache.integration.sglang.sglang_adapter", adapter
    )


def _patch_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "Stream", DummyStream)
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())


def test_lmc_cache_finished_req_stores_and_protects(monkeypatch):
    _install_fake_lmcache(monkeypatch)
    _patch_cuda(monkeypatch)

    repo_root = Path(__file__).resolve().parents[2]
    sglang_path = repo_root / "python" / "sglang"
    srt_path = sglang_path / "srt"
    sglang_pkg = types.ModuleType("sglang")
    sglang_pkg.__path__ = [str(sglang_path)]
    sys.modules["sglang"] = sglang_pkg
    srt_pkg = types.ModuleType("sglang.srt")
    srt_pkg.__path__ = [str(srt_path)]
    sys.modules["sglang.srt"] = srt_pkg
    mem_cache_pkg = types.ModuleType("sglang.srt.mem_cache")
    mem_cache_pkg.__path__ = [str(srt_path / "mem_cache")]
    sys.modules["sglang.srt.mem_cache"] = mem_cache_pkg
    storage_pkg = types.ModuleType("sglang.srt.mem_cache.storage")
    storage_pkg.__path__ = [str(srt_path / "mem_cache" / "storage")]
    sys.modules["sglang.srt.mem_cache.storage"] = storage_pkg
    utils_module = types.ModuleType("sglang.srt.mem_cache.utils")

    def convert_to_bigram_key(tokens):
        if tokens and isinstance(tokens[0], tuple):
            return tokens
        if len(tokens) < 2:
            return []
        return [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]

    utils_module.convert_to_bigram_key = convert_to_bigram_key
    sys.modules["sglang.srt.mem_cache.utils"] = utils_module

    allocator_module = types.ModuleType("sglang.srt.mem_cache.allocator")

    class BaseTokenToKVPoolAllocator:  # pragma: no cover - stub
        pass

    allocator_module.BaseTokenToKVPoolAllocator = BaseTokenToKVPoolAllocator
    sys.modules["sglang.srt.mem_cache.allocator"] = allocator_module

    memory_pool_module = types.ModuleType("sglang.srt.mem_cache.memory_pool")

    class ReqToTokenPool:  # pragma: no cover - stub
        pass

    memory_pool_module.ReqToTokenPool = ReqToTokenPool
    sys.modules["sglang.srt.mem_cache.memory_pool"] = memory_pool_module

    metrics_pkg = types.ModuleType("sglang.srt.metrics")
    metrics_pkg.__path__ = [str(srt_path / "metrics")]
    sys.modules["sglang.srt.metrics"] = metrics_pkg
    metrics_module = types.ModuleType("sglang.srt.metrics.collector")

    class RadixCacheMetricsCollector:  # pragma: no cover - stub
        def __init__(self, labels=None):
            self.labels = labels or {}

        def observe_eviction_duration(self, *_args, **_kwargs):
            return None

        def increment_eviction_num_tokens(self, *_args, **_kwargs):
            return None

    metrics_module.RadixCacheMetricsCollector = RadixCacheMetricsCollector
    sys.modules["sglang.srt.metrics.collector"] = metrics_module

    disagg_pkg = types.ModuleType("sglang.srt.disaggregation")
    disagg_pkg.__path__ = [str(srt_path / "disaggregation")]
    sys.modules["sglang.srt.disaggregation"] = disagg_pkg
    kv_events_module = types.ModuleType("sglang.srt.disaggregation.kv_events")

    class BlockStored:  # pragma: no cover - stub
        pass

    class BlockRemoved:  # pragma: no cover - stub
        pass

    class AllBlocksCleared:  # pragma: no cover - stub
        pass

    kv_events_module.BlockStored = BlockStored
    kv_events_module.BlockRemoved = BlockRemoved
    kv_events_module.AllBlocksCleared = AllBlocksCleared
    sys.modules["sglang.srt.disaggregation.kv_events"] = kv_events_module
    triton_module = types.ModuleType("triton")
    triton_language = types.ModuleType("triton.language")

    def _jit(fn=None, **_kwargs):
        if fn is None:
            return lambda f: f
        return fn

    triton_module.jit = _jit
    triton_module.language = triton_language
    sys.modules["triton"] = triton_module
    sys.modules["triton.language"] = triton_language

    sys.modules.pop("sglang.srt.mem_cache.cache_init_params", None)
    cache_init_params = importlib.import_module(
        "sglang.srt.mem_cache.cache_init_params"
    )
    CacheInitParams = cache_init_params.CacheInitParams

    module_name = "sglang.srt.mem_cache.storage.lmcache.lmc_radix_cache"
    sys.modules.pop(module_name, None)
    lmc_radix_cache = importlib.import_module(module_name)
    LMCRadixCache = lmc_radix_cache.LMCRadixCache

    allocator = DummyTokenToKVPoolAllocator()
    req_to_token_pool = DummyReqToTokenPool(
        torch.tensor([[10, 11, 12]], dtype=torch.int64)
    )
    params = CacheInitParams(
        disable=False,
        req_to_token_pool=req_to_token_pool,
        token_to_kv_pool_allocator=allocator,
        page_size=1,
        enable_kv_cache_events=False,
    )
    cache = LMCRadixCache(params=params, model_config=None, tp_size=1, rank=0)

    req = DummyReq([1, 2, 3])
    req.last_node = cache.root_node

    cache.cache_finished_req(req)

    assert cache.lmcache_connector.store_calls, "store_kv should be invoked"
    stored_md = cache.lmcache_connector.store_calls[0]
    assert stored_md.token_ids == [1, 2, 3]
    torch.testing.assert_close(
        stored_md.kv_indices, torch.tensor([10, 11, 12], dtype=torch.int64)
    )
    assert cache._in_flight_nodes and cache._in_flight_nodes[0] is stored_md.last_node
    assert req_to_token_pool.freed == [0]
    sys.modules.pop(module_name, None)
    for name in [
        "sglang.srt.mem_cache.cache_init_params",
        "sglang.srt.mem_cache.storage",
        "sglang.srt.disaggregation.kv_events",
        "sglang.srt.disaggregation",
        "sglang.srt.metrics.collector",
        "sglang.srt.metrics",
        "sglang.srt.mem_cache.memory_pool",
        "sglang.srt.mem_cache.allocator",
        "sglang.srt.mem_cache.utils",
        "sglang.srt.mem_cache",
        "sglang.srt",
        "sglang",
        "triton.language",
        "triton",
    ]:
        sys.modules.pop(name, None)
