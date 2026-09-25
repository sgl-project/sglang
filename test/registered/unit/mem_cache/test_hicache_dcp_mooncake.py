"""DCP Mooncake keys and zero-copy transfers using real CPU host buffers."""

import ctypes
import unittest
from dataclasses import replace
from itertools import product
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorageConfig,
    HiCacheStorageExtraInfo,
)
from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost
from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import MooncakeStore
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class _ReplicateConfig:
    def __init__(self):
        self.group_ids = None


class _ByteStore:
    """Replace only Mooncake transport; copy bytes through registered pointers."""

    def __init__(self):
        self.objects = {}
        self.regions = []
        self.put_calls = []
        self.queries = []
        self.failed_gets = set()
        self.failed_puts = set()

    def setup(self, *args, **kwargs):
        return 0

    def register_buffer(self, ptr, size):
        self.regions.append((ptr, ptr + size))
        return 0

    def _check_region(self, ptr, size):
        assert any(start <= ptr and ptr + size <= end for start, end in self.regions)

    def batch_is_exist(self, keys):
        self.queries.append(list(keys))
        return [int(key in self.objects) for key in keys]

    def batch_put_from(self, keys, ptrs, sizes, config=None):
        return self.batch_put_from_multi_buffers(
            keys, [[ptr] for ptr in ptrs], [[size] for size in sizes], config
        )

    def batch_put_from_multi_buffers(self, keys, ptrs, sizes, config=None):
        self.put_calls.append(
            (list(keys), config.group_ids if config is not None else None)
        )
        results = []
        for key, pointers, lengths in zip(keys, ptrs, sizes):
            if key in self.failed_puts:
                results.append(-1)
                continue
            for ptr, size in zip(pointers, lengths):
                self._check_region(ptr, size)
            self.objects[key] = b"".join(
                ctypes.string_at(ptr, size) for ptr, size in zip(pointers, lengths)
            )
            results.append(0)
        return results

    def batch_get_into(self, keys, ptrs, sizes):
        return self.batch_get_into_multi_buffers(
            keys, [[ptr] for ptr in ptrs], [[size] for size in sizes]
        )

    def batch_get_into_multi_buffers(self, keys, ptrs, sizes):
        results = []
        for key, pointers, lengths in zip(keys, ptrs, sizes):
            if key not in self.objects or key in self.failed_gets:
                results.append(-1)
                continue
            payload = self.objects[key]
            assert len(payload) == sum(lengths)
            offset = 0
            for ptr, size in zip(pointers, lengths):
                self._check_region(ptr, size)
                ctypes.memmove(ptr, payload[offset : offset + size], size)
                offset += size
            results.append(len(payload))
        return results


def _config(rank=0, tp=4, dcp=2, **overrides):
    cfg = HiCacheStorageConfig(
        tp_rank=rank,
        tp_size=tp,
        pp_rank=0,
        pp_size=1,
        attn_cp_rank=0,
        attn_cp_size=1,
        is_mla_model=True,
        enable_storage_metrics=False,
        is_page_first_layout=True,
        model_name="test/model",
        dcp_size=dcp,
        dcp_rank=rank % dcp,
        logical_page_size=64 * dcp,
        extra_config={
            "master_server_address": "127.0.0.1:50051",
            "check_server": False,
            "global_segment_size": 1024 * 1024,
            "extra_backend_tag": "test",
            "enable_group_semantics": True,
        },
    )
    return replace(cfg, **overrides)


def _pool(config, layout="page_first", dtype=torch.bfloat16):
    device = SimpleNamespace(
        size=512,
        host_capacity_tokens=None,
        store_dtype=dtype,
        kv_lora_rank=8,
        qk_rope_head_dim=4,
        layer_num=2,
        start_layer=0,
        end_layer=1,
        device="cpu",
        layers_to_capture=None,
        layer_shard_enabled=False,
    )
    return MLATokenToKVPoolHost(
        device,
        host_to_device_ratio=2,
        host_size=0,
        page_size=config.logical_page_size,
        layout=layout,
        pin_memory=False,
        device="cpu",
        dcp_size=config.dcp_size,
        dcp_rank=config.dcp_rank,
    )


def _backend(config, pool, transport):
    with (
        mock.patch.object(
            MooncakeStore, "_import_mooncake_store", return_value=lambda: transport
        ),
        mock.patch.object(
            MooncakeStore,
            "_import_mooncake_group_semantics",
            return_value=(_ReplicateConfig, True),
        ),
        mock.patch.object(MooncakeStore, "warmup"),
    ):
        backend = MooncakeStore(config, pool)
    backend.register_mem_pool_host(pool)
    return backend


def _indices(pool, pages):
    page_size = pool.logical_page_size
    return torch.cat(
        [torch.arange(page * page_size, (page + 1) * page_size) for page in pages]
    )


def _page(buffer, layout, page):
    if layout == "layer_first":
        return buffer[:, page * 64 : (page + 1) * 64]
    if layout == "page_first":
        return buffer[page * 64 : (page + 1) * 64]
    return buffer[page]


class TestDcpMooncake(CustomTestCase):
    def test_shard_round_trip_and_group_ids(self):
        for (tp, dcp), layout, dtype, groups in product(
            ((2, 2), (4, 2), (4, 4)),
            ("layer_first", "page_first", "page_first_direct"),
            (torch.bfloat16, torch.float16, torch.uint8),
            (False, True),
        ):
            with self.subTest(
                tp=tp, dcp=dcp, layout=layout, dtype=dtype, groups=groups
            ):
                transport = _ByteStore()
                sources = {}
                keys = ["page-a", "page-b"]
                configs = [_config(rank, tp, dcp) for rank in range(tp)]
                for cfg in configs:
                    cfg.extra_config["enable_group_semantics"] = groups
                    if not cfg.is_storage_writer:
                        continue
                    pool = _pool(cfg, layout, dtype)
                    values = (
                        torch.arange(pool.kv_buffer.numel()) * 13 + cfg.dcp_rank * 17
                    ) % 251
                    pool.kv_buffer.copy_(values.reshape(pool.kv_buffer.shape))
                    sources[cfg.dcp_rank] = pool
                    backend = _backend(cfg, pool, transport)
                    self.assertEqual(
                        backend.batch_set_v1(keys, _indices(pool, [3, 0])), [True, True]
                    )
                self.assertEqual(len(transport.put_calls), dcp)
                self.assertEqual(len(transport.objects), len(keys) * dcp)
                self.assertEqual(
                    {len(data) for data in transport.objects.values()},
                    {64 * 2 * 12 * dtype.itemsize},
                )
                group_ids = [group_ids for _, group_ids in transport.put_calls]
                if groups:
                    self.assertEqual(len(set(group_ids[0])), len(keys))
                    self.assertTrue(all(ids == group_ids[0] for ids in group_ids))
                else:
                    self.assertEqual(group_ids, [None] * dcp)
                for cfg in configs:
                    target = _pool(cfg, layout, dtype)
                    target.kv_buffer.fill_(255)
                    expected = target.kv_buffer.clone()
                    reader = _backend(cfg, target, transport)
                    self.assertEqual(reader.batch_exists(keys + ["missing"]), 2)
                    self.assertEqual(
                        reader.batch_get_v1(keys, _indices(target, [5, 1])),
                        [True, True],
                    )
                    for src_page, dst_page in ((3, 5), (0, 1)):
                        _page(expected, layout, dst_page).copy_(
                            _page(sources[cfg.dcp_rank].kv_buffer, layout, src_page)
                        )
                    torch.testing.assert_close(
                        target.kv_buffer, expected, rtol=0, atol=0
                    )
                    # A second writer sees the same objects and skips upload.
                    self.assertEqual(
                        reader.batch_set_v1(keys, _indices(target, [5, 1])),
                        [True, True],
                    )
                self.assertEqual(len(transport.put_calls), dcp)

    def test_topologies_and_partitions_do_not_alias(self):
        base = _config()
        variants = [
            base,
            _config(rank=1),
            _config(tp=2),
            _config(dcp=4),
            _config(dcp=1),
            replace(base, logical_page_size=256),
            replace(base, model_name="other/model"),
            replace(base, pp_size=2),
            replace(base, pp_size=2, pp_rank=1),
            replace(base, attn_cp_size=2),
            replace(base, attn_cp_size=2, attn_cp_rank=1),
        ]
        object_keys = []
        group_ids = []
        for cfg in variants:
            transport = _ByteStore()
            pool = _pool(cfg)
            backend = _backend(cfg, pool, transport)
            backend.batch_set_v1(["page"], _indices(pool, [0]))
            object_keys.append(next(iter(transport.objects)))
            group_ids.append(transport.put_calls[0][1][0])
        self.assertEqual(len(set(object_keys)), len(variants))
        # One group per logical page, shared by all its shard/stage objects.
        self.assertEqual(group_ids[0], group_ids[1])
        self.assertEqual(group_ids[7], group_ids[8])
        self.assertEqual(group_ids[9], group_ids[10])
        self.assertNotEqual(group_ids[0], group_ids[2])
        self.assertNotEqual(group_ids[0], group_ids[3])
        self.assertEqual(object_keys[4], "test_test-model_page__k")

    def test_pp_query_keeps_shard_and_prefix(self):
        transport = _ByteStore()
        for shard in (0, 1):
            readers = []
            written_key = None
            for stage in (0, 1):
                cfg = _config(shard, pp_size=2, pp_rank=stage)
                pool = _pool(cfg)
                backend = _backend(cfg, pool, transport)
                readers.append(backend)
                if stage == 1:
                    backend.batch_set_v1(["page_0_hash"], _indices(pool, [0]))
                    written_key = transport.put_calls[-1][0][0]
            self.assertEqual(readers[0].batch_exists(["page_0_hash"]), 0)
            self.assertEqual(
                readers[0].batch_exists(
                    ["page_0_hash"], HiCacheStorageExtraInfo(extra_info={"pp_rank": 1})
                ),
                1,
            )
            self.assertEqual(transport.queries[-1], [written_key])

    def test_missing_and_failed_pages_report_per_page_results(self):
        transport = _ByteStore()
        cfg = _config()
        pool = _pool(cfg)
        pool.kv_buffer.fill_(1)
        backend = _backend(cfg, pool, transport)
        keys = ["first", "middle", "last"]
        indices = _indices(pool, [3, 0, 5])
        self.assertEqual(backend.batch_set_v1(keys, indices), [True] * 3)
        middle = transport.put_calls[-1][0][1]
        del transport.objects[middle]
        self.assertEqual(backend.batch_exists(keys), 1)
        self.assertEqual(backend.batch_get_v1(keys, indices), [True, False, True])
        transport.failed_puts.add(middle)
        self.assertEqual(backend.batch_set_v1(keys, indices), [True, False, True])
        transport.failed_puts.clear()
        self.assertEqual(backend.batch_set_v1(keys, indices), [True] * 3)
        transport.failed_gets.add(middle)
        self.assertEqual(backend.batch_exists(keys), 3)
        self.assertEqual(backend.batch_get_v1(keys, indices), [True, False, True])


if __name__ == "__main__":
    unittest.main()
