"""FlexKV host KV is isolated by the request's (extra_key, cache_salt) namespace."""

import importlib
import sys
import threading
from array import array
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import torch

from sglang.srt.managers.req_types import FINISH_LENGTH
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import InitLoadBackParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixCache
from sglang.srt.runtime_context import publish
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

TOKENS = list(range(101, 133))
KV_MARKER = 0xC0FFEE
STORED_NAMESPACE = ("lora-a", "tenant-a")

_FLEXKV_MODULES = (
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
)


@pytest.fixture(scope="module")
def flexkv():
    # Exercise the real cache and connector without the out-of-tree flexkv wheel.
    stub = mock.MagicMock()
    with mock.patch.dict(sys.modules, {name: stub for name in _FLEXKV_MODULES}):
        radix = importlib.import_module(
            "sglang.srt.mem_cache.storage.flexkv.flexkv_radix_cache"
        )
        connector = importlib.import_module(
            "sglang.srt.mem_cache.storage.flexkv.flexkv_connector"
        )
        publish(ServerArgs(model_path="dummy"), role="test")
        yield SimpleNamespace(
            cache_class=radix.FlexKVRadixCache,
            mode=radix.FlexKVMode,
            connector_class=connector.FlexKVConnector,
            success=connector.KVResponseStatus.SUCCESS,
        )


class _FakeKVManager:
    """FlexKV's matching contract: a block is keyed by its namespace and token prefix."""

    def __init__(self, store, key_buffer, success):
        self.store = store
        self.key_buffer = key_buffer
        self.success = success
        self.tasks = {}

    @staticmethod
    def _keys(token_ids, namespace):
        seed = tuple(namespace) if namespace else None
        tokens = [int(token) for token in token_ids]
        return [(seed, tuple(tokens[: end + 1])) for end in range(len(tokens))]

    def get_match(self, token_ids, token_mask, namespace=None):
        keys = self._keys(token_ids, namespace)
        present = np.cumprod([key in self.store for key in keys]).astype(bool)
        hit = present & np.asarray(token_mask, dtype=bool)
        task_id = len(self.tasks)
        self.tasks[task_id] = ("get", [key for key, h in zip(keys, hit) if h])
        return task_id, hit

    def put_match(self, token_ids, token_mask=None, namespace=None):
        keys = self._keys(token_ids, namespace)
        task_id = len(self.tasks)
        self.tasks[task_id] = ("put", keys)
        return task_id, torch.ones(len(keys), dtype=torch.bool)

    def launch(self, task_ids, slot_mappings, **kwargs):
        op, keys = self.tasks[task_ids[0]]
        for key, slot in zip(keys, slot_mappings[0].tolist()):
            if op == "put":
                self.store[key] = self.key_buffer[slot].clone()
            else:
                self.key_buffer[slot] = self.store[key]

    def wait(self, task_ids, timeout=None):
        return {task_id: SimpleNamespace(status=self.success) for task_id in task_ids}

    def cancel(self, task_ids):
        pass


def _instance(flexkv, store):
    """One SGLang instance attached to the FlexKV store shared across instances."""
    pool = MHATokenToKVPool(
        size=128,
        page_size=1,
        dtype=torch.float32,
        head_num=1,
        head_dim=1,
        layer_num=1,
        device="cpu",
        enable_memory_saver=False,
    )
    allocator = TokenToKVPoolAllocator(
        size=128, dtype=torch.float32, device="cpu", kvcache=pool, need_sort=False
    )
    rows = ReqToTokenPool(
        size=4, max_context_len=64, device="cpu", enable_memory_saver=False
    )
    cache = flexkv.cache_class.__new__(flexkv.cache_class)
    RadixCache.__init__(
        cache,
        CacheInitParams(
            disable=False,
            req_to_token_pool=rows,
            token_to_kv_pool_allocator=allocator,
            page_size=1,
        ),
    )
    connector = flexkv.connector_class.__new__(flexkv.connector_class)
    connector.page_size = 1
    connector.kv_manager = _FakeKVManager(
        store=store, key_buffer=pool.get_key_buffer(0), success=flexkv.success
    )
    connector._sync_ctx = SimpleNamespace(
        is_sync_leader=True,
        needs_sync=False,
        is_pp_active=False,
        is_pp_sender=False,
        is_pp_receiver=False,
        should_send_slot_mapping_to_remote=False,
    )
    connector._pending_lookups = {}
    connector._inflight_stores = {}
    cache.flexkv_connector = connector
    cache._mode = flexkv.mode.MP
    cache.load_stream = cache.store_stream = None
    cache._load_markers = {}
    cache._inflight_store_nodes = {}
    cache._node_lock = threading.Lock()
    return SimpleNamespace(cache=cache, pool=pool, rows=rows, allocator=allocator)


def _admit(instance, namespace):
    extra_key, cache_salt = namespace
    req = Req(
        rid="req",
        origin_input_text=None,
        origin_input_ids=array("q", TOKENS),
        sampling_params=SamplingParams(),
        extra_key=extra_key,
        cache_salt=cache_salt,
    )
    req.init_next_round_input(instance.cache)
    return req


def _store_finished_request(flexkv, store, namespace):
    instance = _instance(flexkv, store)
    req = _admit(instance, namespace)
    instance.rows.alloc([req])
    slots = instance.allocator.alloc(len(TOKENS))
    instance.rows.write(
        (req.kv.req_pool_idx, slice(0, len(TOKENS))), slots.to(torch.int32)
    )
    req.kv.set_row_len(len(TOKENS))
    markers = KV_MARKER + torch.arange(len(TOKENS)).view(-1, 1, 1).float()
    instance.pool.get_key_buffer(0)[slots.long()] = markers
    req.lock = instance.cache.lock(req.last_node)
    req.finished_reason = FINISH_LENGTH(length=0)
    instance.cache.checkpoint(req, up_to=len(TOKENS))


def _load_from_fresh_instance(flexkv, store, namespace):
    """Returns the host hit length and the key-buffer values FlexKV loaded back."""
    instance = _instance(flexkv, store)
    req = _admit(instance, namespace)
    loaded_len, node = instance.cache.init_load_back(
        InitLoadBackParams(
            best_match_node=req.best_match_node,
            host_hit_length=req.host_hit_length,
            req=req,
        )
    )
    if loaded_len == 0:
        return req.host_hit_length, []
    return (
        req.host_hit_length,
        instance.pool.get_key_buffer(0)[node.value.long(), 0, 0].tolist(),
    )


@pytest.mark.parametrize(
    "reader_namespace",
    [
        (None, "tenant-b"),
        ("lora-b", "tenant-a"),
        ("lora-a", None),
        (None, None),
    ],
    ids=["other-salt", "other-adapter", "adapter-without-salt", "unnamespaced"],
)
def test_other_namespace_does_not_hit_stored_kv(flexkv, reader_namespace):
    """KV stored under one namespace is never served to a request in another."""
    store = {}
    _store_finished_request(flexkv, store, STORED_NAMESPACE)
    assert store

    host_hit_length, loaded = _load_from_fresh_instance(flexkv, store, reader_namespace)

    assert host_hit_length == 0
    assert loaded == []


def test_same_namespace_loads_stored_kv(flexkv):
    store = {}
    _store_finished_request(flexkv, store, STORED_NAMESPACE)

    host_hit_length, loaded = _load_from_fresh_instance(flexkv, store, STORED_NAMESPACE)

    assert host_hit_length > 0
    assert loaded == [float(KV_MARKER + i) for i in range(host_hit_length)]


def test_unnamespaced_request_keeps_unseeded_flexkv_keys(flexkv):
    """Entries stored before namespacing existed stay reachable by unnamespaced requests."""
    store = {}
    _store_finished_request(flexkv, store, (None, None))

    assert {seed for seed, _ in store} == {None}
    host_hit_length, _ = _load_from_fresh_instance(flexkv, store, (None, None))
    assert host_hit_length > 0
