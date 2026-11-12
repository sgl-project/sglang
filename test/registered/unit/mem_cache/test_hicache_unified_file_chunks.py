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
from sglang.srt.mem_cache.hybrid_cache import hybrid_pool_assembler
from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool
from sglang.srt.mem_cache.pool_host.dsa import DSAIndexerPoolHost
from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost
from sglang.srt.mem_cache.pool_host.page_unified import PageUnifiedLayout
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


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


class _NativeHostPool:
    """CPU page-byte fixture with the same flat-page contract as DSV4 pools."""

    def __init__(self, page_size, page):
        self.page_size = page_size
        self.page = page

    def get_data_page(self, offset, flat=False):
        return self.page.flatten()

    def get_dummy_flat_data_page(self):
        return torch.empty_like(self.page).flatten()

    def set_from_flat_data_page(self, offset, page):
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


class TestUnifiedFileChunks(CustomTestCase):
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

    def _native_backend(self, *, tp_size=1, tp_rank=0, layers=(2, 6), max_size=0):
        config, _, _ = _rank(tp_size=tp_size, tp_rank=tp_rank)
        plans = {
            PoolName.KV: UnifiedPoolPlan(("native_marker",), ((0, 0),)),
            PoolName.DEEPSEEK_V4_C4: UnifiedPoolPlan(
                tuple(f"native_c4_L{layer}" for layer in layers),
                tuple((i * 6, 6) for i in range(len(layers))),
            ),
            PoolName.DEEPSEEK_V4_C4_STATE: UnifiedPoolPlan(
                tuple(f"native_state_L{layer}" for layer in layers),
                tuple((i * 5, 5) for i in range(len(layers))),
            ),
        }
        backend = self._backend(
            replace(
                config,
                is_mla_model=True,
                unified_suffixes=None,
                unified_chunk_spans=None,
                unified_pool_plans=plans,
                extra_config={**config.extra_config, "max_size": max_size},
            )
        )
        pools = {
            PoolName.KV: _NativeHostPool(4, torch.empty(0, dtype=torch.uint8)),
            PoolName.DEEPSEEK_V4_C4: _NativeHostPool(
                4,
                torch.cat(
                    [torch.full((6,), layer, dtype=torch.uint8) for layer in layers]
                ),
            ),
            PoolName.DEEPSEEK_V4_C4_STATE: _NativeHostPool(
                2, torch.empty(len(layers) * 5, dtype=torch.uint8)
            ),
        }
        for i, layer in enumerate(layers):
            pools[PoolName.DEEPSEEK_V4_C4_STATE].page[i * 5 : (i + 1) * 5] = layer + 10
        backend.register_mem_pool_host(pools[PoolName.KV])
        for name, pool in pools.items():
            backend.register_mem_host_pool_v2(pool, name)
        return backend, pools

    def _native_transfer(self, pools, name, keys, **kwargs):
        return PoolTransfer(
            name=name,
            keys=keys,
            host_indices=torch.zeros(
                len(keys) * pools[name].page_size, dtype=torch.int64
            ),
            **kwargs,
        )

    def test_native_pools_reuse_layers_across_tp_and_pp(self):
        writer, pools = self._native_backend(tp_size=2)
        transfers = [self._native_transfer(pools, name, ["native"]) for name in pools]
        self.assertTrue(all(all(ok) for ok in writer.batch_set_v2(transfers).values()))
        self.assertEqual(
            self._path(writer, "native", "native_marker").stat().st_size, 0
        )
        self.assertEqual(
            self._path(writer, "native", "native_state_L6").stat().st_size, 5
        )
        for layer in (2, 6):
            # A different TP deployment and a one-layer PP stage use the same
            # objects with their own pool-specific page sizes.
            reader, targets = self._native_backend(
                tp_size=4, tp_rank=1, layers=(layer,)
            )
            self.assertEqual(reader._evictor._owned_suffixes, ())
            for target in targets.values():
                target.page.fill_(255)
            selected = [
                self._native_transfer(targets, name, ["native"]) for name in targets
            ]
            self.assertEqual(
                reader.batch_exists_v2(["native"], selected).kv_hit_pages, 1
            )
            self.assertTrue(
                all(all(ok) for ok in reader.batch_get_v2(selected).values())
            )
            self.assertTrue(torch.all(targets[PoolName.DEEPSEEK_V4_C4].page == layer))
            state = targets[PoolName.DEEPSEEK_V4_C4_STATE].page
            self.assertTrue(torch.all(state == layer + 10))

    def test_native_sidecar_corruption_gates_hit_and_preserves_host_page(self):
        writer, pools = self._native_backend()
        keys = ["native-0", "native-1", "native-2"]
        for name in pools:
            transfer = self._native_transfer(pools, name, keys)
            self.assertEqual(writer.batch_set_v2([transfer])[name], [True] * 3)
        reader, targets = self._native_backend(tp_size=2, tp_rank=1)
        sidecars = [
            self._native_transfer(targets, PoolName.DEEPSEEK_V4_C4, keys),
            self._native_transfer(
                targets,
                PoolName.DEEPSEEK_V4_C4_STATE,
                [keys[-1]],
                hit_policy=PoolHitPolicy.TRAILING_PAGES,
            ),
        ]
        self.assertEqual(reader.batch_exists_v2(keys, sidecars).kv_hit_pages, 3)
        self._path(reader, keys[-1], "native_state_L6").write_bytes(b"bad")
        self.assertTrue(reader.exists(keys[-1]))  # The logical marker still exists.
        self.assertEqual(reader.batch_exists_v2(keys, sidecars).kv_hit_pages, 2)
        state = targets[PoolName.DEEPSEEK_V4_C4_STATE].page
        state.fill_(255)
        self.assertEqual(
            reader.batch_get_v2([sidecars[-1]]),
            {PoolName.DEEPSEEK_V4_C4_STATE: [False]},
        )
        self.assertTrue(torch.all(state == 255))

    def test_native_plan_registration_and_lru_account_for_each_pool(self):
        backend, pools = self._native_backend(max_size=1024)
        transfers = [
            self._native_transfer(pools, name, ["registered"]) for name in pools
        ]
        self.assertTrue(all(all(ok) for ok in backend.batch_set_v2(transfers).values()))
        self.assertEqual(len(backend._evictor._lru), 5)
        with self.assertRaisesRegex(ValueError, "page has"):
            backend.register_mem_host_pool_v2(
                _NativeHostPool(4, torch.empty(1, dtype=torch.uint8)),
                PoolName.DEEPSEEK_V4_C4,
            )
        with self.assertRaisesRegex(ValueError, "Missing unified storage plan"):
            backend.register_mem_host_pool_v2(pools[PoolName.KV], PoolName.INDEXER)

    def _dsa_backend(self, *, start=2, end=8, tp_size=8, tp_rank=0):
        # CPU file I/O is independent of whether the host running this unit
        # test can compile optional GPU transfer kernels.
        with (
            get_parallel().override(attn_cp_size=1, attn_dcp_size=1),
            patch(
                "sglang.srt.mem_cache.pool_host.mla.can_use_hicache_jit_kernel",
                return_value=False,
            ),
            patch(
                "sglang.srt.mem_cache.pool_host.dsa.can_use_write_back_jit_kernel",
                return_value=False,
            ),
        ):
            root = DSATokenToKVPool(
                size=64,
                page_size=64,
                kv_lora_rank=512,
                dtype=torch.float8_e4m3fn,
                qk_rope_head_dim=64,
                layer_num=end - start,
                device="cpu",
                index_head_dim=128,
                enable_memory_saver=False,
                kv_cache_dim=656,
                start_layer=start,
                end_layer=end,
                skip_topk_layers=[layer not in (2, 6) for layer in range(start, end)],
            )
            kv_info, index_info = root.get_device_pool_infos()
            kv = MLATokenToKVPoolHost(
                kv_info.buffer_info,
                host_to_device_ratio=2,
                host_size=0,
                page_size=64,
                layout="page_unified",
                pin_memory=False,
                override_kv_cache_dim=root.kv_cache_dim,
                device_capacity_tokens=root.size,
                start_layer=start,
                end_layer=end,
                target_layer_num=root.layer_num,
            )
            index = DSAIndexerPoolHost(
                index_info.buffer_info,
                layer_mapping={
                    layer - start: i for i, layer in enumerate(index_info.layer_ids)
                },
                target_device_layer_num=root.layer_num,
                num_host_pages=kv.page_num,
                transfer_page_size=64,
                layout="page_unified",
                buffer_device="cpu",
                dtype=root.store_dtype,
                pin_memory=False,
            )
            self.addCleanup(kv.destroy)
            self.addCleanup(index.destroy)
            pools = {PoolName.KV: kv, PoolName.INDEXER: index}
            group = SimpleNamespace(
                entries=[
                    SimpleNamespace(
                        name=name, host_pool=pool, packed_draft_device_pools=()
                    )
                    for name, pool in pools.items()
                ]
            )
            with patch.object(
                hybrid_pool_assembler, "get_memory", return_value=SimpleNamespace()
            ):
                plans = hybrid_pool_assembler.build_dsa_unified_pool_plans(
                    "glm/test", root, group, 64
                )
        config, _, _ = _rank()
        backend = self._backend(
            replace(
                config,
                tp_size=tp_size,
                tp_rank=tp_rank,
                is_mla_model=True,
                unified_suffixes=None,
                unified_chunk_spans=None,
                unified_pool_plans=plans,
            )
        )
        backend.register_mem_pool_host(kv)
        for info, pool in ((kv_info, kv), (index_info, index)):
            backend.register_mem_host_pool_v2(pool, info.pool_name)
            # Distinct native bytes, including the index keys and FP32 scale
            # region, identify each global layer through both PP subsets.
            page = torch.stack(
                [
                    (
                        (torch.arange(plans[info.pool_name].chunk_spans[0][1]) + layer)
                        % 251
                    ).to(torch.uint8)
                    for layer in info.layer_ids
                ]
            )
            pool.set_from_flat_data_page(0, page.flatten())
        return backend, pools

    def test_dsa_real_mla_anchor_and_sparse_native_indexer_survive_tp_pp(self):
        """An MLA anchor carries KV bytes; a sparse indexer is required on every page."""
        writer, sources = self._dsa_backend()
        keys = ["dsa-0", "dsa-1"]
        transfers = [self._native_transfer(sources, name, keys) for name in sources]
        self.assertTrue(all(all(ok) for ok in writer.batch_set_v2(transfers).values()))
        for name, layer_bytes in (
            (PoolName.KV, 64 * 656),
            (PoolName.INDEXER, 64 * 132),
        ):
            plan = writer._unified_pool_plans[name]
            self.assertEqual(
                self._path(writer, keys[0], plan.suffixes[0]).stat().st_size,
                layer_bytes,
            )
        for start, end in ((2, 6), (6, 8)):
            reader, targets = self._dsa_backend(
                start=start, end=end, tp_size=4, tp_rank=1
            )
            expected = {
                name: pool.get_data_page(0).clone() for name, pool in targets.items()
            }
            for pool in targets.values():
                pool.set_from_flat_data_page(
                    0, torch.full_like(pool.get_data_page(0), 255)
                )
            selected = [self._native_transfer(targets, name, keys) for name in targets]
            self.assertEqual(reader.batch_exists_v2(keys, selected).kv_hit_pages, 2)
            self.assertTrue(
                all(all(ok) for ok in reader.batch_get_v2(selected).values())
            )
            for name, pool in targets.items():
                self.assertTrue(torch.equal(pool.get_data_page(0), expected[name]))
            pointers, lengths = targets[PoolName.INDEXER].get_page_buffer_meta(
                torch.arange(128)
            )
            self.assertEqual(lengths, [64 * 132] * 2)
            self.assertEqual(pointers[1] - pointers[0], 64 * 132)

        # A different native FP8 encoding must miss even with all MLA chunks
        # present, instead of decoding the writer's index bytes incorrectly.
        with patch(
            "sglang.kernels.ops.quantization.fp8_kernel.fp8_dtype",
            torch.float8_e4m3fnuz,
        ):
            wrong_format, targets = self._dsa_backend(
                start=6, end=8, tp_size=4, tp_rank=1
            )
        selected = [self._native_transfer(targets, name, keys) for name in targets]
        self.assertTrue(wrong_format.exists(keys[0]))
        self.assertEqual(wrong_format.batch_exists_v2(keys, selected).kv_hit_pages, 0)

        reader, targets = self._dsa_backend(start=6, end=8, tp_size=4, tp_rank=1)
        suffix = reader._unified_pool_plans[PoolName.INDEXER].suffixes[0]
        self._path(reader, keys[1], suffix).write_bytes(b"truncated")
        selected = [self._native_transfer(targets, name, keys) for name in targets]
        self.assertEqual(reader.batch_exists_v2(keys, selected).kv_hit_pages, 1)
        index = targets[PoolName.INDEXER]
        before = index.get_data_page(0).clone()
        transfer = self._native_transfer(targets, PoolName.INDEXER, [keys[1]])
        self.assertEqual(reader.batch_get_v2([transfer])[PoolName.INDEXER], [False])
        self.assertTrue(torch.equal(index.get_data_page(0), before))


if __name__ == "__main__":
    unittest.main()
