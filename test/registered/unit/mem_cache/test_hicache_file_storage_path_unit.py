"""
Unit test for --file-storage-path routing to the file HiCache backend, plus the
regression guard that an UNSET flag keeps HiCacheFile on its /tmp/hicache default
instead of routing the arg's default value into the backend.

Pure CPU test; no server, no CUDA.
Run with:
    python3 -m pytest test/registered/unit/mem_cache/test_hicache_file_storage_path_unit.py -v
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import os
import unittest

from sglang.srt.mem_cache.hicache_storage import HiCacheFile, HiCacheStorageConfig
from sglang.test.test_utils import CustomTestCase

_ENV = "SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR"


def _make_config(extra_config):
    # tp_rank=1 so HiCacheFile does not create the storage dir as a side effect.
    return HiCacheStorageConfig(
        tp_rank=1,
        tp_size=1,
        pp_rank=0,
        pp_size=1,
        attn_cp_rank=0,
        attn_cp_size=1,
        is_mla_model=True,
        enable_storage_metrics=False,
        is_page_first_layout=True,
        model_name="testmodel",
        extra_config=extra_config,
    )


def _inject(file_storage_path, extra_config=None):
    # Mirrors the HiRadixCache / UnifiedRadixCache routing: only route the arg
    # when it is set, so an unset (default) value is left out of extra_config.
    extra_config = dict(extra_config or {})
    if file_storage_path:
        extra_config.setdefault("file_storage_path", file_storage_path)
    return extra_config


class TestFileStoragePathRouting(CustomTestCase):
    def setUp(self):
        self._saved = os.environ.pop(_ENV, None)

    def tearDown(self):
        if self._saved is not None:
            os.environ[_ENV] = self._saved
        else:
            os.environ.pop(_ENV, None)

    def test_unset_flag_keeps_tmp_hicache(self):
        extra = _inject(None)
        self.assertNotIn("file_storage_path", extra)
        self.assertEqual(HiCacheFile(_make_config(extra)).file_path, "/tmp/hicache")

    def test_set_flag_is_routed(self):
        extra = _inject("/mnt/nvme/hicache")
        self.assertEqual(
            HiCacheFile(_make_config(extra)).file_path, "/mnt/nvme/hicache"
        )

    def test_env_var_wins_over_flag(self):
        os.environ[_ENV] = "/env/hicache"
        extra = _inject("/mnt/nvme/hicache")
        self.assertEqual(HiCacheFile(_make_config(extra)).file_path, "/env/hicache")


class TestHybridFilePoolKeys(CustomTestCase):
    def test_bounded_shard_writes_and_eviction_preserve_peer_files(self):
        import tempfile
        from dataclasses import replace
        from types import SimpleNamespace

        import torch

        from sglang.srt.mem_cache.hicache_storage import PoolName

        with tempfile.TemporaryDirectory() as root:
            configs = [
                replace(
                    _make_config(
                        {
                            "enable_metadata_cache": False,
                            "max_size": 16,
                            "eviction_ratio": 1.0,
                            "min_free_space": 0,
                        }
                    ),
                    tp_rank=rank,
                    tp_size=2,
                )
                for rank in range(2)
            ]
            stores = [HiCacheFile(config, file_path=root) for config in configs]
            for store in stores:
                store.register_mem_host_pool_v2(SimpleNamespace(), PoolName.MAMBA)
            value = torch.tensor([42], dtype=torch.int64)
            shared = "shared"
            shard0 = stores[0]._log_key(PoolName.MAMBA, "a")
            self.assertTrue(stores[0].set(shared, value))
            self.assertTrue(stores[0].set(shard0, value))
            # Reads of replicated KV must not put it into rank 1's LRU.
            self.assertIsNotNone(stores[1].get(shared, torch.empty_like(value)))
            for key in ("a", "b", "c"):
                self.assertTrue(
                    stores[1].set(stores[1]._log_key(PoolName.MAMBA, key), value)
                )
            self.assertFalse(stores[1].exists(stores[1]._log_key(PoolName.MAMBA, "a")))
            self.assertTrue(stores[0].exists(shared))
            self.assertTrue(stores[0].exists(shard0))
            # Startup scans must also leave other ranks' files out of the cap.
            restarted = HiCacheFile(configs[0], file_path=root)
            self.assertTrue(restarted.exists(shared))
            self.assertTrue(stores[1].exists(stores[1]._log_key(PoolName.MAMBA, "b")))
            self.assertTrue(stores[1].exists(stores[1]._log_key(PoolName.MAMBA, "c")))

    def test_rank_sharded_state_roundtrip_and_existence(self):
        import tempfile
        from dataclasses import replace

        import torch

        from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer

        class Pool:
            page_size = 1

            def __init__(self, value):
                self.value = torch.tensor([value], dtype=torch.int64)

            def get_data_page(self, index, flat=False):
                return self.value

            def get_dummy_flat_data_page(self):
                return torch.empty_like(self.value)

            def set_from_flat_data_page(self, index, value):
                self.value = value.clone()

        with tempfile.TemporaryDirectory() as root:
            stores = []
            pools = []
            transfer = PoolTransfer(
                name=PoolName.MAMBA, keys=["page"], host_indices=torch.tensor([0])
            )
            for rank in range(2):
                config = replace(
                    _make_config({"enable_metadata_cache": False}),
                    tp_rank=rank,
                    tp_size=2,
                )
                store = HiCacheFile(config, file_path=root)
                pool = Pool(100 + rank)
                store.register_mem_host_pool_v2(pool, PoolName.MAMBA)
                self.assertTrue(store.set("page", torch.tensor([7])))
                if rank == 0:
                    self.assertEqual(
                        store.batch_set_v2([transfer])[PoolName.MAMBA], [True]
                    )
                else:
                    from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
                        HybridCacheController,
                        StorageOperation,
                    )

                    controller = HybridCacheController.__new__(HybridCacheController)
                    controller.backup_skip = True
                    controller.page_size = 1
                    controller.storage_backend_type = "file"
                    controller.storage_backend = store
                    operation = StorageOperation(
                        torch.tensor([0]),
                        [1],
                        hash_value=["page"],
                        pool_transfers=[transfer, PoolTransfer(name=PoolName.INDEXER)],
                    )
                    controller._page_backup_with_stable_layout(operation)
                    self.assertEqual(operation.completed_tokens, 1)
                    self.assertFalse(store.exists("page.indexer"))
                stores.append(store)
                pools.append(pool)
            self.assertEqual(
                stores[0]._get_component_key("page"),
                stores[1]._get_component_key("page"),
            )
            self.assertEqual(
                stores[0]._get_component_key("page", PoolName.INDEXER),
                stores[1]._get_component_key("page", PoolName.INDEXER),
            )
            for rank, (store, pool) in enumerate(zip(stores, pools)):
                self.assertEqual(
                    store.batch_exists_v2(["page"], [transfer]).kv_hit_pages, 1
                )
                pool.value.zero_()
                self.assertEqual(store.batch_get_v2([transfer])[PoolName.MAMBA], [True])
                self.assertEqual(pool.value.item(), 100 + rank)
            # Removing one shard must not be masked by the other rank's file.
            from pathlib import Path

            (
                Path(root)
                / (stores[1]._get_component_key("page", PoolName.MAMBA) + ".bin")
            ).unlink()
            self.assertEqual(
                stores[1].batch_exists_v2(["page"], [transfer]).kv_hit_pages, 0
            )
            self.assertEqual(
                stores[0].batch_exists_v2(["page"], [transfer]).kv_hit_pages, 1
            )

    def test_file_backend_mha_draft_is_backed_up_per_rank(self):
        # Regression for an MLA target with an MHA draft (e.g. Kimi-K2-Instruct
        # + Kimi-K2-Instruct-eagle3) on the file backend. A nonzero TP rank has
        # backup_skip=True, and should_backup() used to allow MHA draft/draft-SWA
        # only for mooncake, so the file backend never wrote rank>0's draft
        # shard. Because HiCacheFile keys that shard per rank, the missing file
        # zeroed rank>0's storage hit and the cross-rank MIN collapsed the whole
        # prefix hit to zero.
        import tempfile
        from dataclasses import replace
        from types import SimpleNamespace
        from unittest.mock import MagicMock

        import torch

        from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
        from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
            HybridCacheController,
        )
        from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost

        # (1) should_backup now lets every rank back up an MHA draft on the file
        #     backend (matching mooncake), while keeping the TP0-only rules for
        #     non-MHA draft hosts and for backends that do not namespace draft
        #     keys per rank.
        def controller_for(backend_type, host_pool):
            controller = HybridCacheController.__new__(HybridCacheController)
            controller.backup_skip = True  # nonzero MLA TP rank
            controller.storage_backend_type = backend_type
            controller.mem_pool_host = SimpleNamespace(
                entry_map={PoolName.DRAFT: SimpleNamespace(host_pool=host_pool)}
            )
            return controller

        mha_draft = MagicMock(spec=MHATokenToKVPoolHost)
        draft_transfer = SimpleNamespace(name=PoolName.DRAFT)
        self.assertTrue(controller_for("file", mha_draft).should_backup(draft_transfer))
        self.assertTrue(
            controller_for("mooncake", mha_draft).should_backup(draft_transfer)
        )
        self.assertFalse(
            controller_for("file", SimpleNamespace()).should_backup(draft_transfer)
        )
        self.assertFalse(
            controller_for("hf3fs", mha_draft).should_backup(draft_transfer)
        )

        # (2) The shard each rank writes is rank-specific, so rank1's draft is
        #     independently addressable and is never masked by rank0's file.
        class FakeMHAHost(MHATokenToKVPoolHost):
            page_size = 1

            def __init__(self, value):
                self.value = torch.tensor([value], dtype=torch.int64)

            def get_data_page(self, index, flat=False):
                return self.value

            def get_dummy_flat_data_page(self):
                return torch.empty_like(self.value)

            def set_from_flat_data_page(self, index, value):
                self.value = value.clone()

        with tempfile.TemporaryDirectory() as root:
            transfer = PoolTransfer(
                name=PoolName.DRAFT, keys=["page"], host_indices=torch.tensor([0])
            )
            stores, pools = [], []
            for rank in range(2):
                config = replace(
                    _make_config({"enable_metadata_cache": False}),
                    tp_rank=rank,
                    tp_size=2,
                )
                store = HiCacheFile(config, file_path=root)
                pool = FakeMHAHost(100 + rank)
                store.register_mem_host_pool_v2(pool, PoolName.DRAFT)
                self.assertTrue(store.set("page", torch.tensor([7])))
                self.assertEqual(store.batch_set_v2([transfer])[PoolName.DRAFT], [True])
                stores.append(store)
                pools.append(pool)
            draft0 = stores[0]._get_component_key("page", PoolName.DRAFT)
            draft1 = stores[1]._get_component_key("page", PoolName.DRAFT)
            self.assertNotEqual(draft0, draft1)
            self.assertIn(".tp0_2", draft0)
            self.assertIn(".tp1_2", draft1)
            for rank, (store, pool) in enumerate(zip(stores, pools)):
                self.assertEqual(
                    store.batch_exists_v2(["page"], [transfer]).kv_hit_pages, 1
                )
                pool.value.zero_()
                self.assertEqual(store.batch_get_v2([transfer])[PoolName.DRAFT], [True])
                self.assertEqual(pool.value.item(), 100 + rank)
            # Removing rank1's draft shard must not be masked by rank0's file.
            from pathlib import Path

            (Path(root) / (draft1 + ".bin")).unlink()
            self.assertEqual(
                stores[1].batch_exists_v2(["page"], [transfer]).kv_hit_pages, 0
            )
            self.assertEqual(
                stores[0].batch_exists_v2(["page"], [transfer]).kv_hit_pages, 1
            )


if __name__ == "__main__":
    unittest.main()
