"""DCP L3 shard identity and writer selection, without a server or model.

Run: python test/registered/unit/mem_cache/test_hicache_dcp_storage_identity.py -v
"""

import tempfile
import unittest
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path

import torch
from test_mooncake_dcp_storage import _indices, _mamba_pool, _page_segments, _pool

from sglang.srt.environ import envs
from sglang.srt.mem_cache.hicache_storage import (
    HiCacheFile,
    HiCacheStorageConfig,
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def make_config(tp_rank=0, tp_size=4, dcp_size=2, **overrides):
    fields = dict(
        tp_rank=tp_rank,
        tp_size=tp_size,
        pp_rank=0,
        pp_size=1,
        attn_cp_rank=0,
        attn_cp_size=1,
        is_mla_model=True,
        enable_storage_metrics=False,
        is_page_first_layout=True,
        model_name="test/model",
        dcp_size=dcp_size,
        dcp_rank=tp_rank % dcp_size,
        logical_page_size=64 * dcp_size,
        kv_cache_dtype=torch.bfloat16,
        host_layout="page_first",
        extra_config={
            "enable_metadata_cache": True,
            "metadata_ttl": -1,
            "max_size": "0",
            "min_free_space": "0",
        },
    )
    fields.update(overrides)
    return HiCacheStorageConfig(**fields)


class TestDcpStorageIdentity(CustomTestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.directory = self.stack.enter_context(tempfile.TemporaryDirectory())
        self.stack.enter_context(
            envs.SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR.override(self.directory)
        )

    def backend(self, config):
        return HiCacheFile(config)

    def test_replica_keys_and_single_writer_per_shard(self):
        for tp_size, dcp_size, expected_writers in (
            (2, 2, [0, 1]),
            (4, 2, [0, 1]),
            (4, 4, [0, 1, 2, 3]),
        ):
            with self.subTest(tp_size=tp_size, dcp_size=dcp_size):
                configs = [
                    make_config(rank, tp_size, dcp_size) for rank in range(tp_size)
                ]
                keys = [
                    self.backend(config)._get_suffixed_key("prefix")
                    for config in configs
                ]
                self.assertEqual(len(set(keys)), dcp_size)
                self.assertEqual(
                    [c.tp_rank for c in configs if c.is_storage_writer],
                    expected_writers,
                )
                for rank, config in enumerate(configs):
                    self.assertEqual(keys[rank], keys[config.dcp_rank])
                    self.assertEqual(
                        sum(
                            c.is_storage_writer
                            for c, key in zip(configs, keys)
                            if key == keys[rank]
                        ),
                        1,
                    )

    def test_layouts_and_topologies_cannot_alias(self):
        config = make_config()
        variants = [
            config,
            make_config(tp_rank=1),
            make_config(tp_size=2),
            make_config(dcp_size=4),
            make_config(dcp_size=1),
            replace(config, logical_page_size=256),
            replace(config, kv_cache_dtype=torch.float16),
            replace(config, kv_cache_dtype=torch.float8_e4m3fn),
            replace(config, kv_cache_dtype=torch.float8_e5m2),
            replace(config, pp_size=2, pp_rank=0),
            replace(config, pp_size=2, pp_rank=1),
            replace(config, attn_cp_size=2, attn_cp_rank=0),
            replace(config, attn_cp_size=2, attn_cp_rank=1),
            replace(
                config, host_layout="page_first_direct", is_page_first_layout=False
            ),
            replace(config, host_layout="layer_first", is_page_first_layout=False),
            replace(config, model_name="different/model"),
        ]
        keys = [self.backend(c)._get_suffixed_key("prefix") for c in variants]
        self.assertEqual(len(set(keys)), len(variants))

    def test_fresh_replicas_resolve_same_files_and_metadata(self):
        # Exercise actual key consumers with small byte payloads, not model KV.
        for rank in (0, 1):
            self.assertTrue(
                self.backend(make_config(rank)).set(
                    "prefix", torch.tensor([rank, rank + 10], dtype=torch.uint8)
                )
            )
        self.assertEqual(len(list(Path(self.directory).glob("*.bin"))), 2)

        # Newly constructed backends model another engine's independent state.
        for rank in range(4):
            backend = self.backend(make_config(rank))
            shard = rank % 2
            with self.subTest(rank=rank):
                self.assertTrue(backend.exists("prefix"))
                self.assertEqual(backend.batch_exists(["prefix", "missing"]), 1)
                self.assertTrue(
                    backend.metadata_cache.contains(backend._get_suffixed_key("prefix"))
                )
                result = backend.get("prefix", torch.empty(2, dtype=torch.uint8))
                torch.testing.assert_close(
                    result, torch.tensor([shard, shard + 10], dtype=torch.uint8)
                )
                self.assertEqual(backend._evictor.config_suffix, backend.config_suffix)

    def test_dcp_one_retains_mla_and_gqa_keys(self):
        for rank in range(4):
            with self.subTest(rank=rank):
                mla = make_config(rank, dcp_size=1)
                gqa = replace(mla, is_mla_model=False)
                self.assertEqual(
                    self.backend(mla)._get_suffixed_key("prefix"), "prefix_test-model"
                )
                self.assertEqual(
                    self.backend(gqa)._get_suffixed_key("prefix"),
                    f"prefix_test-model_{rank}_4",
                )
                self.assertEqual(mla.is_storage_writer, rank == 0)
                self.assertTrue(gqa.is_storage_writer)

    def test_file_state_shards_do_not_alias_mla_replicas(self):
        """TP0 and TP2 share MLA, but must restore different recurrent state."""
        for layout in ("page_first", "page_first_direct"):
            expected = {}
            for rank in (0, 2):
                state = _mamba_pool(layout=layout)
                for i, buffer in enumerate(state.get_hybrid_pool_buffer()):
                    buffer.view(torch.uint8).fill_(17 + rank + i)
                expected[rank] = state.get_data_page(3).clone()
                writer = self.backend(make_config(rank, dcp_size=2))
                writer.register_mem_host_pool_v2(state, PoolName.MAMBA)
                transfer = PoolTransfer(
                    PoolName.MAMBA,
                    host_indices=torch.tensor([3]),
                    keys=["checkpoint"],
                )
                self.assertEqual(
                    writer.batch_set_v2([transfer])[PoolName.MAMBA], [True]
                )

            for rank in (0, 2):
                # Cache capacity is not part of a compatible tensor schema.
                state = _mamba_pool(capacity=16, layout=layout)
                reader = self.backend(make_config(rank, dcp_size=2))
                reader.register_mem_host_pool_v2(state, PoolName.MAMBA)
                transfer = PoolTransfer(
                    PoolName.MAMBA,
                    host_indices=torch.tensor([7]),
                    keys=["checkpoint"],
                )
                self.assertEqual(
                    reader.batch_get_v2([transfer])[PoolName.MAMBA], [True]
                )
                torch.testing.assert_close(state.get_data_page(7), expected[rank])

    def test_bounded_mla_replica_can_write_and_evict_only_its_state(self):
        """An MLA replica still owns its TP-sharded recurrent checkpoint."""
        for model in ("bounded/model", "模型" * 70):
            with self.subTest(model=model):
                state = _mamba_pool()
                size = state.get_data_page(3).nbytes
                config = make_config(
                    dcp_size=2,
                    model_name=model,
                    extra_config=dict(
                        max_size=size,
                        min_free_space=0,
                        eviction_ratio=1.0,
                        enable_metadata_cache=False,
                    ),
                )
                owner = self.backend(config)
                self.assertTrue(owner.set("kv", torch.ones(size, dtype=torch.uint8)))
                replica_config = replace(config, tp_rank=2)
                replica = self.backend(replica_config)
                replica.register_mem_host_pool_v2(state, PoolName.MAMBA)
                self.assertFalse(replica.set("unowned-kv", torch.ones(1)))
                for key in ("old", "new"):
                    transfer = PoolTransfer(
                        PoolName.MAMBA,
                        host_indices=torch.tensor([3]),
                        keys=[key],
                    )
                    self.assertEqual(
                        replica.batch_set_v2([transfer]), {PoolName.MAMBA: [True]}
                    )
                self.assertFalse(
                    replica.exists(replica._log_key(PoolName.MAMBA, "old"))
                )
                self.assertTrue(owner.exists("kv"))
                # Fresh scans must not adopt another rank's files.
                fresh = self.backend(replica_config)
                fresh.register_mem_host_pool_v2(state, PoolName.MAMBA)
                self.assertTrue(fresh.exists(fresh._log_key(PoolName.MAMBA, "new")))
                self.assertTrue(owner.exists("kv"))
                self.assertEqual(fresh._evictor._total_bytes, size)
                # The primary writer shares one cap across its KV and state,
                # without adopting a replica's independently owned state.
                owner.register_mem_host_pool_v2(state, PoolName.MAMBA)
                self.assertEqual(
                    owner.batch_set_v2([transfer]), {PoolName.MAMBA: [True]}
                )
                self.assertFalse(owner.exists("kv"))
                self.assertTrue(fresh.exists(fresh._log_key(PoolName.MAMBA, "new")))
                self.assertEqual(owner._evictor._total_bytes, size)

    def test_long_model_identity_restores_kv_and_state_from_fresh_backend(self):
        """Snapshot paths plus state fingerprints exceeded NAME_MAX in Kimi serving."""
        for model in ("/cache/models--org--model/snapshots/" + "a" * 100, "模型" * 70):
            with self.subTest(model=model):
                config = make_config(model_name=model)
                key = "b" * 64
                state = _mamba_pool()
                for buffer in state.get_hybrid_pool_buffer():
                    buffer.view(torch.uint8).fill_(37)
                expected = state.get_data_page(3).clone()
                writer = self.backend(config)
                writer.register_mem_host_pool_v2(state, PoolName.MAMBA)
                self.assertTrue(writer.set(key, torch.tensor([19], dtype=torch.uint8)))
                transfer = PoolTransfer(
                    PoolName.MAMBA,
                    host_indices=torch.tensor([3]),
                    keys=[key],
                    hit_policy=PoolHitPolicy.TRAILING_PAGES,
                )
                self.assertEqual(
                    writer.batch_set_v2([transfer])[PoolName.MAMBA], [True]
                )
                reader = self.backend(config)
                target = _mamba_pool(capacity=16)
                reader.register_mem_host_pool_v2(target, PoolName.MAMBA)
                self.assertEqual(
                    reader.batch_exists_v2([key], [transfer]).kv_hit_pages, 1
                )
                transfer.host_indices = torch.tensor([7])
                self.assertEqual(
                    reader.batch_get_v2([transfer])[PoolName.MAMBA], [True]
                )
                torch.testing.assert_close(target.get_data_page(7), expected)
                torch.testing.assert_close(
                    reader.get(key, torch.empty(1, dtype=torch.uint8)),
                    torch.tensor([19], dtype=torch.uint8),
                )
                other = self.backend(replace(config, model_name=model + "-other"))
                other.register_mem_host_pool_v2(target, PoolName.MAMBA)
                self.assertEqual(
                    other.batch_exists_v2([key], [transfer]).kv_hit_pages, 0
                )
                self.assertTrue(
                    all(
                        len(p.name.encode()) <= 255
                        for p in Path(self.directory).glob("*.bin")
                    )
                )

    def test_file_v2_uses_logical_mla_indices_and_physical_page_bytes(self):
        """DCP v2 must consume logical index runs, then restore unrelated pages."""
        config = make_config()
        pool = _pool(config)
        store = self.backend(config)
        store.register_mem_host_pool_v2(pool, PoolName.KV)
        pool.kv_buffer.zero_()
        for page in (2, 4):
            for segment in _page_segments(pool, pool.kv_buffer, page):
                segment.view(torch.uint8).fill_(23 + page)
        source = _indices(config, (2, 4))
        keys = ["first", "second"]
        self.assertEqual(
            store.batch_set_v2(
                [PoolTransfer(PoolName.KV, host_indices=source, keys=keys)]
            )[PoolName.KV],
            [True, True],
        )
        pool.kv_buffer.zero_()
        target = _indices(config, (7, 9))
        self.assertEqual(
            store.batch_get_v2(
                [PoolTransfer(PoolName.KV, host_indices=target, keys=keys)]
            )[PoolName.KV],
            [True, True],
        )
        expected = torch.zeros_like(pool.kv_buffer)
        for source_page, target_page in ((2, 7), (4, 9)):
            for segment in _page_segments(pool, expected, target_page):
                segment.view(torch.uint8).fill_(23 + source_page)
        torch.testing.assert_close(
            pool.kv_buffer.view(torch.uint8), expected.view(torch.uint8)
        )

    def test_file_query_preserves_sparse_legal_checkpoints(self):
        """Taking min of two maxima can select an absent recurrent checkpoint."""
        store = self.backend(make_config())
        store.register_mem_host_pool_v2(_mamba_pool(), PoolName.MAMBA)
        keys = ["a", "b", "c", "d"]
        for key in keys:
            self.assertTrue(store.set(key, torch.tensor([1], dtype=torch.uint8)))
        for name, pages in ((PoolName.MAMBA, (1, 4)), (PoolName.SWA, (1, 3))):
            for page in pages:
                self.assertTrue(
                    store.set(
                        store._log_key(name, keys[page - 1]),
                        torch.tensor([2], dtype=torch.uint8),
                    )
                )
        transfers = [
            PoolTransfer(name, hit_policy=PoolHitPolicy.TRAILING_PAGES)
            for name in (PoolName.MAMBA, PoolName.SWA)
        ]
        result = store.batch_exists_v2(keys, transfers)
        self.assertEqual(result.kv_hit_pages, 1)
        self.assertEqual(result.restorable_prefix_pages, [1])

    def test_incomplete_or_inconsistent_dcp_identity_is_rejected(self):
        config = make_config()
        invalid_fields = [
            {"dcp_size": 0},
            {"dcp_rank": 2},
            {"tp_rank": 1},
            {"tp_rank": 4},
            {"tp_size": 3},
            {"is_mla_model": False},
            {"logical_page_size": None},
            {"logical_page_size": 0},
            {"logical_page_size": 127},
            {"kv_cache_dtype": None},
            {"host_layout": None},
        ]
        for fields in invalid_fields:
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                replace(config, **fields)


if __name__ == "__main__":
    unittest.main()
