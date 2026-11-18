"""LMCache CPU KV events: tree page hashes as identity, tree events kept in order."""

import importlib.util
import sys
from array import array
from types import ModuleType, SimpleNamespace
from unittest import mock

import pytest
import torch

from sglang.srt.disaggregation.kv_events import (
    BlockRemoved,
    BlockStored,
    StorageMedium,
)
from sglang.srt.managers.req_types import MM_PAD_SHIFT_VALUE
from sglang.srt.mem_cache.base_prefix_cache import (
    IncLockRefResult,
    InsertParams,
    MatchPrefixParams,
    MatchResult,
)
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.kits.unified_radix_cache.config import CacheConfig
from sglang.test.kits.unified_radix_cache.fixtures import build_fixture

register_cuda_ci(est_time=6, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=6, suite="base-b-test-1-gpu-small-amd")

PAGE = 2
CFG = CacheConfig(page_size=PAGE, kv_size=64, max_context_len=64)


@pytest.fixture
def cache_module():
    # Exercise the real cache module without installing the optional MP service.
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
    return module


def _facade(cache_module, *, publisher=True, tree=None):
    cache_class = cache_module.LMCacheUnifiedRadixCache
    cache = cache_class.__new__(cache_class)
    cache.tree_core = (
        tree.tree_core
        if tree is not None
        else SimpleNamespace(page_size=PAGE, is_eagle=False)
    )
    cache._lmcache_kv_publisher = publisher
    cache._external_flows = {}
    cache.lmcache_connector = mock.Mock()
    cache.lmcache_connector.build_cache_salt.return_value = ""
    return cache


def _insert(tree, allocator, tokens):
    value = allocator.alloc(len(tokens))
    assert value is not None
    tree.insert(InsertParams(key=RadixKey(array("q", tokens)), value=value))


def _published_pages(tree):
    """page hash -> (parent hash, page tokens) for every GPU page stored so far."""
    pages = {}
    for event in tree.take_events():
        if not isinstance(event, BlockStored) or event.medium != StorageMedium.GPU:
            continue
        parent = event.parent_block_hash
        for index, block_hash in enumerate(event.block_hashes):
            start = index * event.block_size
            tokens = list(event.token_ids[start : start + event.block_size])
            pages[block_hash] = (parent, tokens)
            parent = block_hash
    return pages


def _hashes_of_matched_prefix(facade, tree, tokens):
    """What a store of ``tokens`` passes: the hashes of the prefix it matched."""
    key = RadixKey(array("q", tokens))
    node = tree.match_prefix(MatchPrefixParams(key=key)).last_device_node
    return facade._native_block_hashes(
        key, node_id=node, path_tokens=len(tree.path_device_indices(node))
    )


@pytest.mark.parametrize("backend", ["python", "rust"])
def test_native_hashes_are_the_tree_event_hashes_of_the_same_prefix(
    cache_module, backend
):
    """A CPU block must carry the id and parent the tree published for that page,
    whichever node holds it: an unsplit node, both halves of a split one, a
    suffix behind a cached prefix, and a prefix that ends inside a node."""
    tree, allocator, _ = build_fixture(
        CFG, enable_kv_cache_events=True, tree_core_backend=backend
    )
    first = [1, 2, 3, 4, 5, 6, 7, 8]
    branch = [1, 2, 3, 4, 9, 10]
    longer = first + [11, 12, 13, 14]
    pages = {}
    for tokens in (first, branch, longer):
        _insert(tree, allocator, tokens)
        pages.update(_published_pages(tree))
    facade = _facade(cache_module, tree=tree)

    for tokens in (first, branch, longer, longer[:10]):
        hashes = _hashes_of_matched_prefix(facade, tree, tokens)
        assert len(hashes) == len(tokens) // PAGE
        parent = None
        for index, block_hash in enumerate(hashes):
            page_tokens = tokens[index * PAGE : (index + 1) * PAGE]
            assert pages[block_hash] == (parent, page_tokens)
            parent = block_hash


@pytest.mark.parametrize("backend", ["python", "rust"])
def test_a_path_the_tree_did_not_hash_gets_no_identity(cache_module, backend):
    """Hashes that do not cover the whole path would name the wrong pages, so a
    tree without page hashes (a rank that publishes no events) binds nothing."""
    tree, allocator, _ = build_fixture(
        CFG, enable_kv_cache_events=False, tree_core_backend=backend
    )
    tokens = [1, 2, 3, 4, 5, 6]
    _insert(tree, allocator, tokens)
    facade = _facade(cache_module, tree=tree)

    assert _hashes_of_matched_prefix(facade, tree, tokens) is None


@pytest.mark.parametrize(
    "namespace, tokens",
    [
        ({"extra_key": "lora-or-tenant"}, [1, 2, 3, 4]),
        ({"cache_salt": "tenant"}, [1, 2, 3, 4]),
        ({}, [1, 2, MM_PAD_SHIFT_VALUE + 7, 4]),
    ],
)
def test_a_request_outside_the_event_scope_has_no_native_identity(
    cache_module, namespace, tokens
):
    """A salted, extra-key or multimodal request must not get the hashes of the
    unsalted text prefix with the same tokens, which a router would match."""
    key = RadixKey(array("q", tokens), **namespace)
    facade = _facade(cache_module)

    assert facade._native_block_hashes(key, node_id=0, path_tokens=4) is None


def _store(facade, tokens):
    """Run _submit_store on a rank that holds 8 of the requested tokens."""
    connector = facade.lmcache_connector
    connector.chunk_size = 4
    connector.get_store_start.return_value = 0
    facade._mamba_component = None
    facade._pending_stores = []
    facade._pending_store_counts = {}
    facade.inc_lock_ref = mock.Mock(return_value=IncLockRefResult(node_id=7))
    facade._native_block_hashes = mock.Mock(return_value=[11, 12, 13, 14])

    def match(self, params):
        return MatchResult(
            device_prefix_len=min(len(params.key), 8),
            last_device_node=7,
            last_host_node=None,
            best_match_node=None,
        )

    with (
        mock.patch.object(UnifiedRadixCache, "match_prefix", match),
        mock.patch.object(facade, "path_device_indices", return_value=torch.arange(8)),
    ):
        facade._submit_store(
            SimpleNamespace(rid="request", extra_key=None, cache_salt=None), tokens
        )
    return connector.submit_store.call_args


def test_store_binds_the_prefix_it_stores(cache_module):
    """A store limited to the prefix every rank holds sends the tokens and the
    hashes of that shortened prefix, not of the request."""
    facade = _facade(cache_module)
    tokens = list(range(1, 13))

    store = _store(facade, tokens)

    assert list(store.args[1]) == tokens[:8]
    assert store.kwargs["native_block_hashes"] == [11, 12, 13, 14]
    hashed = facade._native_block_hashes.call_args
    assert list(hashed.args[0].raw_token_ids()) == tokens[:8]
    assert hashed.kwargs == {"node_id": 7, "path_tokens": 8}


def test_a_rank_that_does_not_publish_stores_exactly_as_before(cache_module):
    """Other ranks, and every rank without --kv-events-config, must not read
    the tree's hashes or pass an argument older LMCache connectors lack."""
    facade = _facade(cache_module, publisher=False)

    store = _store(facade, list(range(1, 13)))

    assert "native_block_hashes" not in store.kwargs
    facade._native_block_hashes.assert_not_called()


def _gpu_events():
    return [
        BlockStored(
            block_hashes=[11],
            parent_block_hash=None,
            token_ids=[1, 2],
            block_size=PAGE,
            lora_id=None,
            medium=StorageMedium.GPU,
        ),
        BlockRemoved(block_hashes=[11], medium=StorageMedium.GPU),
    ]


def _cpu_events():
    return [BlockRemoved(block_hashes=[21], medium=StorageMedium.CPU)]


def test_cpu_events_follow_the_tree_events_without_replacing_them(cache_module):
    facade = _facade(cache_module)
    facade.lmcache_connector.take_events.return_value = _cpu_events()

    with mock.patch.object(
        UnifiedRadixCache, "take_events", return_value=_gpu_events()
    ):
        events = facade.take_events()

    assert events == _gpu_events() + _cpu_events()


def test_a_rank_that_does_not_publish_never_drains_cpu_events(cache_module):
    facade = _facade(cache_module, publisher=False)

    with mock.patch.object(
        UnifiedRadixCache, "take_events", return_value=_gpu_events()
    ):
        assert facade.take_events() == _gpu_events()

    facade.lmcache_connector.take_events.assert_not_called()


@pytest.mark.parametrize(
    "configured, tree_publishes", [(True, True), (True, False), (False, False)]
)
def test_construction_asks_for_cpu_events_only_with_kv_events_configured(
    cache_module, configured, tree_publishes
):
    """CPU events have no flag of their own. Without --kv-events-config the
    connector gets the arguments it always got; with it, every rank registers
    for events and only the tree's publisher rank drains."""
    cache_class = cache_module.LMCacheUnifiedRadixCache
    connector_class = cache_module.UnifiedLMCacheMPConnector
    connector_class.reset_mock()
    params = SimpleNamespace(
        tp_cache_group=None,
        pp_size=1,
        pp_rank=0,
        pp_cache_group=None,
        enable_kv_cache_events=tree_publishes,
        dynamo_kv_event_format=False,
    )

    def parent_init(self, params):
        self._components_tuple = ()
        self.tree_core = SimpleNamespace(page_size=PAGE)
        self.token_to_kv_pool_allocator = self.req_to_token_pool = None
        self.tree_components = ()

    with (
        mock.patch.object(UnifiedRadixCache, "__init__", parent_init),
        mock.patch.object(UnifiedRadixCache, "sliding_window_size", None),
        mock.patch.object(cache_module.atexit, "register"),
    ):
        cache = cache_class(
            params,
            model_config=None,
            tp_size=1,
            tp_rank=0,
            lmcache_config_file=None,
            forward_stream=None,
            kv_events_configured=configured,
        )

    event_args = {
        name: value
        for name, value in connector_class.call_args.kwargs.items()
        if name in ("enable_kv_events", "kv_event_publisher", "dynamo_format")
    }
    assert event_args == (
        {
            "enable_kv_events": True,
            "kv_event_publisher": tree_publishes,
            "dynamo_format": False,
        }
        if configured
        else {}
    )
    assert cache._lmcache_kv_publisher is tree_publishes


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
