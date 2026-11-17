"""Unified file objects retain their byte identity across TP/PP reshaping."""

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.hicache_key_scheme import UnifiedPoolPlan, plan_unified_kv
from sglang.srt.mem_cache.hicache_storage import (
    HiCacheFile,
    HiCacheStorageConfig,
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
)
from sglang.srt.mem_cache.pool_host.page_unified import PageUnifiedLayout
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_C4, _STATE = PoolName.DEEPSEEK_V4_C4, PoolName.DEEPSEEK_V4_C4_STATE


class _HostPool:
    def __init__(self, layout, page):
        self.page_unified_layout = layout
        self.page_size = layout.page_size
        self.page = page

    def get_data_page(self, offset, flat=False):
        assert offset == 0
        return self.page.flatten() if flat else self.page

    def get_dummy_flat_data_page(self):
        return torch.empty_like(self.page).flatten()

    def set_from_flat_data_page(self, offset, page):
        assert offset == 0
        self.page.copy_(page.reshape_as(self.page))


def _rank(tp_size=1, pp_size=1, tp_rank=0, pp_rank=0):
    # Every scalar identifies an absolute head/layer/component/token coordinate.
    full = torch.arange(4 * 4 * 2 * 4 * 4, dtype=torch.int32).reshape(4, 4, 2, 4, 1, 4)
    heads, layers = 4 // tp_size, 4 // pp_size
    start_layer = pp_rank * layers
    page = full[
        tp_rank * heads : (tp_rank + 1) * heads,
        start_layer : start_layer + layers,
    ].contiguous()
    layout = PageUnifiedLayout(
        page_size=4,
        layer_num=layers,
        head_num=heads,
        head_group_num=heads,
        head_dim=4,
        itemsize=4,
    )
    plan = plan_unified_kv(
        model_id="test/model",
        dtype="int32",
        page_size=layout.page_size,
        rank_replicated=False,
        local_kv_heads=heads,
        attn_tp_rank=tp_rank,
        attn_tp_size=tp_size,
        attn_cp_size=1,
        start_layer=start_layer,
        end_layer=start_layer + layers,
        is_final_stage=pp_rank == pp_size - 1,
        head_group=1,
        layer_partition=2,
    )
    config = HiCacheStorageConfig(
        tp_rank=tp_rank,
        tp_size=tp_size,
        pp_rank=pp_rank,
        pp_size=pp_size,
        attn_cp_rank=0,
        attn_cp_size=1,
        is_mla_model=False,
        enable_storage_metrics=False,
        is_page_first_layout=False,
        model_name="test/model",
        unified_suffixes=plan.suffixes,
        unified_chunk_spans=[
            layout.chunk_span(layer_range, head_group)
            for layer_range in plan.layer_ranges
            for head_group in range(plan.head_group_num)
        ],
        extra_config={
            "enable_metadata_cache": True,
            "metadata_ttl": -1,
            "max_size": 0,
            "min_free_space": 0,
        },
    )
    return config, layout, page


class TestUnifiedFileChunks(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        context = envs.SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR.override(
            self.directory.name
        )
        context.__enter__()
        self.addCleanup(context.__exit__, None, None, None)

    def _backend(self, config, layout=None, page=None):
        backend = HiCacheFile(config)
        if layout is not None:
            pool = _HostPool(layout, page)
            backend.register_mem_pool_host(pool)
            backend.register_mem_host_pool_v2(pool, PoolName.KV)
        return backend

    def _path(self, backend, key, suffix):
        return Path(backend.file_path) / f"{key}_{suffix}.bin"

    def test_reshard_tp_and_pp_in_both_directions(self):
        for writer, reader in (((2, 1), (1, 2)), ((1, 2), (2, 1))):
            with self.subTest(writer=writer, reader=reader):
                key = f"page-{writer}"
                writer_files = set()
                for pp_rank in range(writer[1]):
                    for tp_rank in range(writer[0]):
                        config, layout, page = _rank(*writer, tp_rank, pp_rank)
                        backend = self._backend(config, layout, page)
                        transfer = PoolTransfer(
                            name=PoolName.KV,
                            keys=[key],
                            host_indices=torch.arange(layout.page_size),
                        )
                        self.assertEqual(
                            backend.batch_set_v2([transfer]), {PoolName.KV: [True]}
                        )
                        writer_files.update(backend._get_component_keys(key))
                reader_files = set()
                for pp_rank in range(reader[1]):
                    for tp_rank in range(reader[0]):
                        config, layout, expected = _rank(*reader, tp_rank, pp_rank)
                        target = torch.full_like(expected, -1)
                        backend = self._backend(config, layout, target)
                        self.assertEqual(backend.batch_exists_v2([key]).kv_hit_pages, 1)
                        transfer = PoolTransfer(
                            name=PoolName.KV,
                            keys=[key],
                            host_indices=torch.arange(layout.page_size),
                        )
                        self.assertEqual(
                            backend.batch_get_v2([transfer]), {PoolName.KV: [True]}
                        )
                        torch.testing.assert_close(target, expected, rtol=0, atol=0)
                        reader_files.update(backend._get_component_keys(key))
                self.assertEqual(reader_files, writer_files)

    def test_key_order_does_not_define_byte_order(self):
        config, layout, page = _rank()
        self.assertNotEqual(
            config.unified_chunk_spans, sorted(config.unified_chunk_spans)
        )
        writer = self._backend(config)
        self.assertTrue(writer.set("page", page))
        permutation = [3, 0, 6, 1, 7, 4, 2, 5]
        reader = self._backend(
            replace(
                config,
                unified_suffixes=[config.unified_suffixes[i] for i in permutation],
                unified_chunk_spans=[
                    config.unified_chunk_spans[i] for i in permutation
                ],
            )
        )
        target = torch.zeros_like(page)
        self.assertIs(reader.get("page", target), target)
        torch.testing.assert_close(target, page, rtol=0, atol=0)

    def test_missing_or_truncated_chunk_is_a_miss_with_cached_metadata(self):
        for truncate in (False, True):
            with self.subTest(truncate=truncate):
                config, _, page = _rank()
                writer = self._backend(config)
                keys = [f"{truncate}-{i}" for i in range(3)]
                self.assertTrue(writer.batch_set(keys, [page] * 3))
                reader = self._backend(config)
                stems = reader._get_component_keys(keys[1])
                self.assertTrue(
                    all(reader.metadata_cache.contains(stem) for stem in stems)
                )
                path = self._path(reader, keys[1], config.unified_suffixes[-1])
                if truncate:
                    path.write_bytes(b"short")
                else:
                    path.unlink()
                self.assertFalse(reader.exists(keys[1]))
                self.assertEqual(reader.batch_exists_v2(keys).kv_hit_pages, 1)
                target = torch.full_like(page, -1)
                self.assertIsNone(reader.get(keys[1], target))
                self.assertTrue(torch.all(target == -1))
                self.assertFalse(reader.metadata_cache.contains(stems[-1]))
                # A later backup repairs the missing or malformed object.
                self.assertTrue(writer.set(keys[1], page))
                self.assertEqual(reader.batch_exists_v2(keys).kv_hit_pages, 3)

    def test_read_failure_does_not_publish_partial_page(self):
        config, _, page = _rank()
        backend = self._backend(config)
        self.assertTrue(backend.set("page", page))
        read_file = backend._get_file
        failed_key = backend._get_component_keys("page")[1]

        def read(key, target):
            return None if key == failed_key else read_file(key, target)

        target = torch.full_like(page, -1)
        with patch.object(backend, "_get_file", side_effect=read):
            self.assertIsNone(backend.get("page", target))
        self.assertTrue(torch.all(target == -1))

    def test_lru_tracks_all_chunks_and_never_acknowledges_partial_page(self):
        config, _, page = _rank()
        config = replace(
            config,
            extra_config={
                **config.extra_config,
                "max_size": page.numel() * page.element_size(),
            },
        )
        writer = self._backend(config)
        self.assertTrue(writer.set("old", page))
        self.assertEqual(len(writer._evictor._lru), len(config.unified_suffixes))
        peer = self._backend(replace(config, attn_cp_rank=1, attn_cp_size=2))
        self.assertEqual(peer._evictor._owned_suffixes, ())
        self.assertEqual(len(peer._evictor._lru), 0)
        self.assertTrue(peer.exists("old"))
        self.assertFalse(peer.set("peer", page))
        self.assertTrue(writer.set("new", page))
        self.assertFalse(peer.exists("old"))
        self.assertTrue(peer.exists("new"))
        too_small = self._backend(
            replace(config, extra_config={**config.extra_config, "max_size": 256})
        )
        self.assertFalse(too_small.set("cannot-fit", page))
        self.assertFalse(too_small.exists("cannot-fit"))

    def test_invalid_spans_and_page_sizes_are_rejected(self):
        config, layout, page = _rank()
        for spans in (None, [], [(0, 10)] * 8, [(1 + i * 256, 256) for i in range(8)]):
            with self.subTest(spans=spans), self.assertRaises(ValueError):
                self._backend(replace(config, unified_chunk_spans=spans))
        backend = self._backend(config)
        with self.assertRaisesRegex(ValueError, "page has"):
            backend.set("wrong-size", torch.empty(1, dtype=torch.uint8))
        with self.assertRaisesRegex(ValueError, "page has"):
            backend.register_mem_pool_host(
                SimpleNamespace(page_unified_layout=SimpleNamespace(bytes_per_page=1))
            )
        with self.assertRaisesRegex(ValueError, "unique"):
            self._backend(replace(config, unified_suffixes=["same"] * 8))

    def test_single_chunk_and_legacy_keys_remain_compatible(self):
        config, _, _ = _rank()
        value = torch.arange(64, dtype=torch.uint8)
        single = replace(
            config, unified_suffixes=["namespace_L0-4_H0"], unified_chunk_spans=None
        )
        writer = self._backend(single)
        self.assertTrue(writer.set("page", value))
        self.assertTrue(self._path(writer, "page", single.unified_suffixes[0]).exists())
        reader = self._backend(replace(single, unified_chunk_spans=[(0, 64)]))
        target = torch.empty_like(value)
        self.assertIs(reader.get("page", target), target)
        torch.testing.assert_close(target, value)
        legacy = self._backend(
            replace(config, unified_suffixes=None, unified_chunk_spans=None)
        )
        self.assertEqual(legacy.config_suffix, "_test-model_0_1")
        self.assertTrue(legacy.set("legacy", value))
        self.assertIs(legacy.get("legacy", target), target)
        torch.testing.assert_close(target, value)

    def _native_backend(self, layers=(2, 6), tp_size=1, tp_rank=0, max_size=0):
        config, _, _ = _rank(tp_size=tp_size, tp_rank=tp_rank)
        geometry = SimpleNamespace(page_size=4)
        plans = {PoolName.KV: UnifiedPoolPlan(("marker",), ((0, 0),))}
        pools = {PoolName.KV: _HostPool(geometry, torch.empty(0, dtype=torch.uint8))}
        for name, size in ((_C4, 6), (_STATE, 5)):
            plans[name] = UnifiedPoolPlan(
                tuple(f"{name.value}_L{layer}" for layer in layers),
                tuple((i * size, size) for i in range(len(layers))),
            )
            page = [torch.full((size,), layer + size) for layer in layers]
            pools[name] = _HostPool(geometry, torch.cat(page).to(torch.uint8))
        backend = HiCacheFile(
            replace(
                config,
                is_mla_model=True,
                unified_suffixes=None,
                unified_chunk_spans=None,
                unified_pool_plans=plans,
                extra_config={**config.extra_config, "max_size": max_size},
            )
        )
        backend.register_mem_pool_host(pools[PoolName.KV])
        for name, pool in pools.items():
            backend.register_mem_host_pool_v2(pool, name)
        return backend, pools

    @staticmethod
    def _transfer(name, keys, **kwargs):
        indices = torch.zeros(4 * len(keys), dtype=torch.int64)
        return PoolTransfer(name=name, keys=keys, host_indices=indices, **kwargs)

    def test_native_pools_reuse_global_layers_across_tp_and_pp(self):
        writer, pools = self._native_backend(tp_size=2, max_size=1024)
        keys = ["n0", "n1"]
        written = writer.batch_set_v2([self._transfer(name, keys) for name in pools])
        self.assertTrue(all(all(ok) for ok in written.values()))
        self.assertEqual(self._path(writer, "n0", "marker").stat().st_size, 0)
        self.assertEqual(len(writer._evictor._lru), 10)
        for layer in (2, 6):
            # A wider TP rank holding one PP layer reads the same objects.
            reader, targets = self._native_backend((layer,), tp_size=4, tp_rank=1)
            self.assertEqual(reader._evictor._owned_suffixes, ())
            for target in targets.values():
                target.page.fill_(255)
            selected = [self._transfer(name, keys) for name in targets]
            self.assertEqual(reader.batch_exists_v2(keys, selected).kv_hit_pages, 2)
            read = reader.batch_get_v2(selected)
            self.assertTrue(all(all(ok) for ok in read.values()))
            self.assertTrue(torch.all(targets[_C4].page == layer + 6))
            self.assertTrue(torch.all(targets[_STATE].page == layer + 5))
        with self.assertRaisesRegex(ValueError, "page has"):
            writer.register_mem_host_pool_v2(
                _HostPool(SimpleNamespace(page_size=4), torch.empty(1)), _C4
            )

    def test_corrupt_side_object_gates_hit_and_preserves_host_page(self):
        writer, pools = self._native_backend()
        keys = ["c0", "c1", "c2"]
        writer.batch_set_v2([self._transfer(name, keys) for name in pools])
        reader, targets = self._native_backend(tp_size=2, tp_rank=1)
        sidecars = [
            self._transfer(_C4, keys),
            self._transfer(_STATE, keys[-1:], hit_policy=PoolHitPolicy.TRAILING_PAGES),
        ]
        self.assertEqual(reader.batch_exists_v2(keys, sidecars).kv_hit_pages, 3)
        self._path(reader, keys[-1], f"{_STATE.value}_L6").write_bytes(b"bad")
        self.assertTrue(reader.exists(keys[-1]))  # The zero-byte anchor remains.
        self.assertEqual(reader.batch_exists_v2(keys, sidecars).kv_hit_pages, 2)
        state = targets[_STATE].page.fill_(255)
        self.assertEqual(reader.batch_get_v2([sidecars[-1]]), {_STATE: [False]})
        self.assertTrue(torch.all(state == 255))


if __name__ == "__main__":
    unittest.main()
