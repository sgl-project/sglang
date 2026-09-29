"""DCP storage must preserve request bytes across unrelated physical allocations.

The native-client boundary below copies bytes and checks both registration and
request ownership. Host pools, key construction and v1/v2 adapters are real.
"""

import ctypes
import unittest
from dataclasses import replace
from itertools import product
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorageConfig,
    HiCacheStorageExtraInfo,
    PoolName,
    PoolTransfer,
)
from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost
from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import MooncakeStore
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

SOURCE_PAGES = (7, 2, 11)
TARGET_PAGES = (3, 9, 1)
KEYS = ["prefix_0_0", "prefix_1_0", "prefix_2_0"]


def _config(tp=4, dcp=2, rank=0, layout="page_first", page=4, **changes):
    config = HiCacheStorageConfig(
        tp_rank=rank,
        tp_size=tp,
        pp_rank=0,
        pp_size=1,
        attn_cp_rank=0,
        attn_cp_size=1,
        is_mla_model=True,
        enable_storage_metrics=False,
        is_page_first_layout=layout == "page_first",
        model_name="test/model",
        dcp_size=dcp,
        dcp_rank=rank % dcp,
        logical_page_size=page * dcp,
        kv_cache_dtype=torch.float8_e4m3fn,
        host_layout=layout,
        extra_config={
            "master_server_address": "127.0.0.1:50051",
            "global_segment_size": 64 << 20,
            "protocol": "tcp",
            "check_server": False,
        },
    )
    return replace(config, **changes)


def _pool(config):
    page = config.logical_page_size // config.dcp_size
    device = SimpleNamespace(
        size=16 * page,
        host_capacity_tokens=None,
        store_dtype=(
            torch.uint8
            if config.kv_cache_dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
            else config.kv_cache_dtype
        ),
        kv_lora_rank=8,
        qk_rope_head_dim=4,
        layer_num=3,
        start_layer=0,
        end_layer=2,
        device="cpu",
        layers_to_capture=None,
        layer_shard_enabled=False,
    )
    return MLATokenToKVPoolHost(
        device,
        host_to_device_ratio=2,
        host_size=0,
        page_size=config.logical_page_size,
        layout=config.host_layout,
        pin_memory=False,
        device="cpu",
        dcp_size=config.dcp_size,
        dcp_rank=config.dcp_rank,
    )


def _page_segments(pool, buffer, page):
    """Independent physical slices, never production index/metadata helpers."""
    start, end = page * pool.page_size, (page + 1) * pool.page_size
    if pool.layout == "layer_first":
        return [buffer[layer, start:end] for layer in range(pool.layer_num)]
    if pool.layout == "page_first":
        return [buffer[start:end]]
    return [buffer[page : page + 1]]


def _indices(config, pages):
    return torch.tensor(
        [
            p * config.logical_page_size + i
            for p in pages
            for i in range(config.logical_page_size)
        ],
        dtype=torch.int64,
    )


class _ByteStore:
    def __init__(self, objects):
        self.objects = objects
        self.registered = []
        self.owned = []

    def setup(self, *args, **kwargs):
        self.segment_bytes = args[2]
        return 0

    def register_buffer(self, ptr, size):
        self.registered.append((ptr, size))
        return 0

    def own(self, pool, pages):
        self.owned = [
            (part.data_ptr(), part.numel() * part.element_size())
            for page in pages
            for part in _page_segments(pool, pool.kv_buffer, page)
        ]

    def _check(self, ptr, size):
        for ranges in (self.registered, self.owned):
            assert any(
                base <= ptr and ptr + size <= base + length for base, length in ranges
            ), (ptr, size, ranges)

    def put(self, key, value):
        self.objects[key] = value
        return 0

    def get(self, key):
        return self.objects.get(key)

    def is_exist(self, key):
        return int(key in self.objects)

    def batch_is_exist(self, keys):
        return [self.is_exist(key) for key in keys]

    def batch_put_from(self, keys, ptrs, sizes, *args):
        return self.batch_put_from_multi_buffers(
            keys, [[p] for p in ptrs], [[n] for n in sizes]
        )

    def batch_put_from_multi_buffers(self, keys, ptrs, sizes, *args):
        assert len(keys) == len(ptrs) == len(sizes)
        for key, addresses, lengths in zip(keys, ptrs, sizes):
            assert len(addresses) == len(lengths)
            for ptr, size in zip(addresses, lengths):
                self._check(ptr, size)
            self.objects[key] = b"".join(
                ctypes.string_at(ptr, size) for ptr, size in zip(addresses, lengths)
            )
        return [0] * len(keys)

    def batch_get_into(self, keys, ptrs, sizes):
        return self.batch_get_into_multi_buffers(
            keys, [[p] for p in ptrs], [[n] for n in sizes]
        )

    def batch_get_into_multi_buffers(self, keys, ptrs, sizes):
        assert len(keys) == len(ptrs) == len(sizes)
        results = []
        for key, addresses, lengths in zip(keys, ptrs, sizes):
            value = self.objects.get(key)
            if value is None:
                results.append(-1)
                continue
            assert len(value) <= sum(lengths)
            offset = 0
            for ptr, size in zip(addresses, lengths):
                self._check(ptr, size)
                data = value[offset : offset + size]
                ctypes.memmove(ptr, data, len(data))
                offset += size
            results.append(len(value))
        return results


def _store(config, pool, objects):
    client = _ByteStore(objects)
    with (
        patch.object(
            MooncakeStore, "_import_mooncake_store", return_value=lambda: client
        ),
        patch.object(
            MooncakeStore,
            "_import_mooncake_group_semantics",
            return_value=(SimpleNamespace, False),
        ),
    ):
        store = MooncakeStore(config)
    store.register_mem_pool_host(pool)
    # The controller supplies the anchor mapping; registering KV twice must not
    # register its physical buffer twice.
    store.registered_pools[PoolName.KV] = pool
    store.register_mem_host_pool_v2(pool, PoolName.KV)
    return store


def _io(store, config, pages, api, write):
    indices = _indices(config, pages)
    if api == 1:
        method = store.batch_set_v1 if write else store.batch_get_v1
        return method(KEYS, indices)
    method = store.batch_set_v2 if write else store.batch_get_v2
    return method([PoolTransfer(PoolName.KV, host_indices=indices, keys=KEYS)])[
        PoolName.KV
    ]


class TestMooncakeDcpStorage(CustomTestCase):
    def test_native_failures_are_explicit_per_object_failures(self):
        """A failed native call must not kill workers or acknowledge a write.

        A short result vector also cannot prove that every requested object
        completed, even when all returned statuses indicate success.
        """
        for layout, api, operation, failure in product(
            ("page_first", "layer_first"),
            (1, 2),
            ("get", "put", "exists"),
            ("exception", "short", "extra", "none"),
        ):
            with self.subTest(
                layout=layout, api=api, operation=operation, failure=failure
            ):
                config = _config(layout=layout)
                pool = _pool(config)
                store = _store(config, pool, {})
                native_method = (
                    "batch_is_exist"
                    if operation == "exists"
                    else f"batch_{operation}_{'into' if operation == 'get' else 'from'}"
                    + ("_multi_buffers" if layout == "layer_first" else "")
                )
                outcomes = {
                    "short": [],
                    "extra": [1 if operation == "exists" else 0] * 4,
                    "none": None,
                }
                fault = (
                    {"side_effect": RuntimeError("injected native I/O failure")}
                    if failure == "exception"
                    else {"return_value": outcomes[failure]}
                )
                with patch.object(store.store, native_method, **fault):
                    if operation == "exists":
                        hits = (
                            store.batch_exists(KEYS)
                            if api == 1
                            else store.batch_exists_v2(KEYS).kv_hit_pages
                        )
                        self.assertEqual(hits, 0)
                    else:
                        self.assertEqual(
                            _io(store, config, SOURCE_PAGES, api, operation == "put"),
                            [False] * 3,
                        )

    def test_round_trip_has_no_cross_request_or_cross_shard_writes(self):
        for dcp, layout, api, dtype in product(
            (1, 2, 4, 8, 16),
            ("layer_first", "page_first", "page_first_direct"),
            (1, 2),
            (torch.float8_e4m3fn, torch.bfloat16),
        ):
            objects = {}
            for rank in range(dcp):
                config = _config(
                    tp=dcp, dcp=dcp, rank=rank, layout=layout, kv_cache_dtype=dtype
                )
                with self.subTest(
                    dcp=dcp, rank=rank, layout=layout, api=api, dtype=dtype
                ):
                    source, target = _pool(config), _pool(config)
                    # Multibyte tags distinguish layers, rows, channels and rank.
                    words = source.kv_buffer.view(torch.int32)
                    words.copy_(
                        (
                            torch.arange(words.numel(), dtype=torch.int32)
                            + (rank + 1) * 100003
                        ).reshape(words.shape)
                    )
                    target.kv_buffer.fill_(165)
                    expected = target.kv_buffer.clone()
                    for src, dst in zip(SOURCE_PAGES, TARGET_PAGES):
                        for a, b in zip(
                            _page_segments(source, source.kv_buffer, src),
                            _page_segments(target, expected, dst),
                        ):
                            b.copy_(a)
                    writer = _store(config, source, objects)
                    reader = _store(config, target, objects)
                    writer.store.own(source, SOURCE_PAGES)
                    reader.store.own(target, TARGET_PAGES)
                    self.assertEqual(
                        _io(writer, config, SOURCE_PAGES, api, True), [True] * 3
                    )
                    self.assertEqual(reader.batch_exists(KEYS), 3)
                    self.assertEqual(
                        _io(reader, config, TARGET_PAGES, api, False), [True] * 3
                    )
                    torch.testing.assert_close(
                        target.kv_buffer.view(torch.uint8),
                        expected.view(torch.uint8),
                        rtol=0,
                        atol=0,
                    )
                    self.assertEqual(
                        writer.store.registered,
                        [
                            (
                                source.kv_buffer.data_ptr(),
                                source.kv_buffer.numel()
                                * source.kv_buffer.element_size(),
                            )
                        ],
                    )
                    self.assertEqual(writer.store.segment_bytes, (64 << 20) // dcp)

    def test_incompatible_namespaces_miss_and_replicas_hit(self):
        config = _config()
        pool, objects = _pool(config), {}
        writer = _store(config, pool, objects)
        writer.store.own(pool, SOURCE_PAGES)
        self.assertEqual(_io(writer, config, SOURCE_PAGES, 1, True), [True] * 3)
        for changes in (
            dict(tp_size=8),
            dict(dcp_size=4),
            dict(pp_size=2),
            dict(attn_cp_size=2),
            dict(logical_page_size=16),
            dict(kv_cache_dtype=torch.float8_e5m2),
            dict(host_layout="layer_first"),
            dict(tp_rank=1, dcp_rank=1),
            dict(model_name="other/model"),
        ):
            with self.subTest(changes=changes):
                other = replace(config, **changes)
                self.assertEqual(
                    _store(other, _pool(other), objects).batch_exists(KEYS), 0
                )
        replica = replace(config, tp_rank=2)
        self.assertEqual(_store(replica, _pool(replica), objects).batch_exists(KEYS), 3)

    def test_pp_lookup_changes_only_the_pp_partition(self):
        config = _config(pp_size=2, pp_rank=1)
        source, objects = _pool(config), {}
        writer = _store(config, source, objects)
        writer.store.own(source, SOURCE_PAGES)
        self.assertEqual(_io(writer, config, SOURCE_PAGES, 1, True), [True] * 3)
        other = replace(config, pp_rank=0)
        reader = _store(other, _pool(other), objects)
        self.assertEqual(reader.batch_exists(KEYS), 0)
        self.assertEqual(
            reader.batch_exists(
                KEYS, HiCacheStorageExtraInfo(extra_info={"pp_rank": 1})
            ),
            3,
        )

    def test_short_get_is_a_failure_for_each_api_and_layout(self):
        for layout, api in product(
            ("layer_first", "page_first", "page_first_direct"), (1, 2)
        ):
            with self.subTest(layout=layout, api=api):
                config = _config(layout=layout)
                source, target, objects = _pool(config), _pool(config), {}
                writer, reader = (
                    _store(config, source, objects),
                    _store(config, target, objects),
                )
                writer.store.own(source, SOURCE_PAGES)
                reader.store.own(target, TARGET_PAGES)
                self.assertEqual(
                    _io(writer, config, SOURCE_PAGES, api, True), [True] * 3
                )
                key = next(k for k in objects if KEYS[1] in k)
                objects[key] = objects[key][:-1]
                self.assertEqual(
                    _io(reader, config, TARGET_PAGES, api, False), [True, False, True]
                )

    def test_malformed_logical_pages_are_rejected_before_pointer_io(self):
        config = _config()
        pool = _pool(config)
        good = _indices(config, SOURCE_PAGES)
        hole = good.clone()
        hole[2] += 1
        for indices in (
            good[:-1],
            good + 1,
            hole,
            good - 1000,
            good.float(),
            good + pool.logical_size,
        ):
            with (
                self.subTest(indices=indices.tolist()),
                self.assertRaises((ValueError, IndexError)),
            ):
                pool.get_page_buffer_meta(indices)

    def test_get_result_count_mismatch_cannot_publish_a_prefix(self):
        config = _config()
        pool = _pool(config)
        reader = _store(config, pool, {})
        for api, results in product((1, 2), ([], [144], [144] * 4)):
            with (
                self.subTest(api=api, results=results),
                patch.object(reader.store, "batch_get_into", return_value=results),
            ):
                self.assertEqual(
                    _io(reader, config, TARGET_PAGES, api, False), [False] * 3
                )


if __name__ == "__main__":
    unittest.main()
