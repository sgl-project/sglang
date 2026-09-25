"""DCP L3 shard identity and writer selection, without a server or model.

Run: python test/registered/unit/mem_cache/test_hicache_dcp_storage_identity.py -v
"""

import tempfile
import unittest
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path

import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.hicache_storage import HiCacheFile, HiCacheStorageConfig
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
