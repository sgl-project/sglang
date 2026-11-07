import importlib.util
import sys
from contextlib import contextmanager
from types import ModuleType, SimpleNamespace
from unittest import mock

import pytest
import torch

from sglang.srt.mem_cache.base_prefix_cache import MatchResult
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


@pytest.fixture
def cache_class():
    metadata = ModuleType("lmcache.integration.sglang.lmcache_mp_metadata")
    metadata.LMCacheExternalFlow = mock.Mock()
    metadata.LMCachePendingStore = mock.Mock()
    connector = ModuleType("lmcache.integration.sglang.unified_lmcache_mp_connector")
    connector.UnifiedLMCacheMPConnector = mock.Mock()
    with mock.patch.dict(
        sys.modules, {metadata.__name__: metadata, connector.__name__: connector}
    ):
        spec = importlib.util.find_spec(
            "sglang.srt.mem_cache.storage.lmcache.lmcache_unified_radix_cache"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module.LMCacheUnifiedRadixCache


@pytest.mark.parametrize("store_raises", [False, True])
@pytest.mark.parametrize("device_type", ["cuda", "xpu"])
def test_store_orders_indices_and_producer_event_on_forward_stream(
    cache_class, store_raises, device_type
):
    cache = cache_class.__new__(cache_class)
    caller = object()
    forward = SimpleNamespace(
        device=torch.device(f"{device_type}:0"), wait_stream=mock.Mock()
    )
    active = [caller]
    sequence = []
    forward.wait_stream.side_effect = lambda stream: sequence.append(("wait", stream))

    @contextmanager
    def select(stream):
        sequence.append(("enter", stream))
        active.append(stream)
        try:
            yield
        finally:
            active.pop()
            sequence.append(("exit", stream))

    def indices(_):
        assert active[-1] is forward, "GPU indices prepared outside producer stream"
        sequence.append(("indices", forward))
        return torch.arange(256)

    def submit(*args, **kwargs):
        # The connector records the producer event on the current stream.
        assert device_module.current_stream() is forward
        sequence.append(("event", forward))
        if store_raises:
            raise RuntimeError("store failed")
        return object()

    cache._forward_stream = forward
    cache.tree_core = SimpleNamespace(page_size=1, is_eagle=False)
    cache._mamba_component = None
    cache._pending_stores = []
    cache._pending_store_counts = {}
    cache.path_device_indices = mock.Mock(side_effect=indices)
    cache.inc_lock_ref = mock.Mock(
        return_value=SimpleNamespace(to_dec_params=lambda: "lock")
    )
    cache.dec_lock_ref = mock.Mock()
    cache.lmcache_connector = mock.Mock(chunk_size=256)
    cache.lmcache_connector.get_store_start.return_value = 0
    cache.lmcache_connector.submit_store.side_effect = submit
    device_module = SimpleNamespace(
        current_stream=mock.Mock(side_effect=lambda device=None: active[-1]),
        stream=mock.Mock(side_effect=select),
    )
    req = SimpleNamespace(rid="request", extra_key=None, cache_salt="salt")
    matched = MatchResult(
        device_prefix_len=256,
        last_device_node=7,
        last_host_node=None,
        best_match_node=7,
    )
    with (
        mock.patch.object(UnifiedRadixCache, "match_prefix", return_value=matched),
        mock.patch.object(
            torch, "get_device_module", return_value=device_module
        ) as get_device_module,
        mock.patch.object(
            torch.cuda,
            "current_stream",
            side_effect=AssertionError("Store must use the producer device backend"),
        ),
    ):
        cache._submit_store(req, list(range(256)))

    forward.wait_stream.assert_called_once_with(caller)
    get_device_module.assert_called_once_with(forward.device)
    assert active == [caller]
    assert [name for name, _ in sequence] == [
        "wait",
        "enter",
        "indices",
        "event",
        "exit",
    ]
    if store_raises:
        cache.dec_lock_ref.assert_called_once_with(7, "lock")
        assert not cache._pending_stores
        assert not cache._pending_store_counts
    else:
        cache.dec_lock_ref.assert_not_called()
        assert len(cache._pending_stores) == 1
        assert cache._pending_store_counts == {"request": 1}
