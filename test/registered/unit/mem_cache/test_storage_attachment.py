"""Storage existence state across runtime backend changes."""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.buffer_mode.pipeline import BufferModePipeline
from sglang.srt.mem_cache.buffer_mode.storage_existence_cache import (
    StorageExistenceCache,
)
from sglang.srt.mem_cache.hicache_storage import (
    HiCacheFile,
    HiCacheStorageConfig,
    PoolName,
)
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.storage_prefetch import StoragePrefetchRetries
from sglang.srt.mem_cache.unified_cache.storage_attachment import StorageAttachment
from sglang.srt.mem_cache.unified_cache.unified_tree_core_interface import (
    BufferBackupSnapshot,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


class TestStorageAttachment(CustomTestCase):
    def setUp(self):
        super().setUp()
        self.controller = Mock(
            storage_backend_type="file",
            write_policy="write_through",
            mem_pool_host=SimpleNamespace(entries=[]),
        )
        self.cache = SimpleNamespace(
            cache_controller=self.controller,
            is_write_back=False,
            host_memory_mode="buffer_only",
            enable_storage=False,
            prefetch_stop_policy="best_effort",
            write_through_threshold=1,
            sidecar_pool_specs=[],
            host_pool_group=SimpleNamespace(entry_map={}),
            components={},
            _enable_metrics_flag=False,
            extra_metric_labels=None,
            page_size=4,
            tree_core=SimpleNamespace(
                backfill_missing_hash_values=lambda: 0,
                buffer_backup_pool_keys=lambda *_: {},
            ),
            ongoing_prefetch={},
            ongoing_backup={},
            _storage_prefetch_hit_remaining_by_reqid={},
            prefetch_loaded_tokens_by_reqid={},
            prefetch_loaded_storage_start_by_reqid={},
            storage_prefetch_retries=StoragePrefetchRetries(),
            storage_existence_cache=StorageExistenceCache(),
            drain_storage_control_queues_local=Mock(),
        )
        self.attachment = StorageAttachment(self.cache)

    def _attach(self, **kwargs):
        ok, message = self.attachment.attach("file", **kwargs)
        self.assertTrue(ok, message)

    def _detach(self):
        ok, message = self.attachment.detach()
        self.assertTrue(ok, message)

    def test_reattach_admits_backup_to_empty_file_backend(self):
        backends = []

        def attach_file(**kwargs):
            backends.append(
                HiCacheFile(
                    HiCacheStorageConfig(
                        tp_rank=0,
                        tp_size=1,
                        pp_rank=0,
                        pp_size=1,
                        attn_cp_rank=0,
                        attn_cp_size=1,
                        is_mla_model=False,
                        enable_storage_metrics=False,
                        is_page_first_layout=True,
                        model_name="test",
                        extra_config=kwargs["storage_backend_extra_config"],
                    )
                )
            )

        self.controller.attach_storage_backend.side_effect = attach_file
        pipeline = BufferModePipeline.__new__(BufferModePipeline)
        pipeline._cache = self.cache
        snapshot = BufferBackupSnapshot(
            node_id=1,
            parent_node_id=0,
            parent_is_root=True,
            parent_last_hash=None,
            hash_values=["page"],
            key=RadixKey([1, 2, 3, 4]),
            prefix_keys=None,
        )
        value = torch.arange(16, dtype=torch.float32)
        with (
            TemporaryDirectory() as directory,
            envs.SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR.override(None),
        ):
            self._attach(
                storage_backend_extra_config_json=json.dumps(
                    {"file_storage_path": str(Path(directory) / "a")}
                )
            )
            self.assertTrue(backends[0].set("page", value))
            self.cache.storage_existence_cache.pool(PoolName.KV).add(["page"])
            self.assertEqual(pipeline._write_intents(snapshot), [])

            self._detach()
            self._attach(
                storage_backend_extra_config_json=json.dumps(
                    {"file_storage_path": str(Path(directory) / "b")}
                )
            )
            self.assertFalse(backends[1].exists("page"))
            self.assertEqual(
                pipeline._write_intents(snapshot), [(PoolName.KV, ["page"])]
            )
            self.assertTrue(backends[1].set("page", value))
            restored = backends[1].get("page", torch.empty_like(value))
            torch.testing.assert_close(restored, value, rtol=0, atol=0)

    def test_reattach_invalidates_every_pool(self):
        self._attach()
        pools = (PoolName.KV, PoolName.SWA, PoolName.INDEXER)
        for pool in pools:
            beliefs = self.cache.storage_existence_cache.pool(pool)
            beliefs.add(["stored"])
        self._detach()
        self._attach()
        for pool in pools:
            with self.subTest(pool=pool):
                beliefs = self.cache.storage_existence_cache.pool(pool)
                self.assertFalse(beliefs.contains("stored"))

    def test_policy_update_preserves_presence(self):
        self._attach()
        beliefs = self.cache.storage_existence_cache.pool(PoolName.KV)
        beliefs.add(["stored"])
        self._attach(hicache_storage_prefetch_policy="timeout")
        self.assertEqual(self.cache.prefetch_stop_policy, "timeout")
        self.assertTrue(beliefs.contains("stored"))
        self.controller.attach_storage_backend.assert_called_once()

    def test_failed_attach_preserves_presence(self):
        self._attach()
        beliefs = self.cache.storage_existence_cache.pool(PoolName.KV)
        beliefs.add(["stored"])
        self._detach()
        self.controller.attach_storage_backend.side_effect = RuntimeError("offline")
        with self.assertLogs(StorageAttachment.__module__, level="ERROR"):
            ok, _ = self.attachment.attach("file")
        self.assertFalse(ok)
        self.assertFalse(self.cache.enable_storage)
        self.assertTrue(beliefs.contains("stored"))

    def test_rejected_backend_change_preserves_presence(self):
        self._attach()
        beliefs = self.cache.storage_existence_cache.pool(PoolName.KV)
        beliefs.add(["stored"])
        ok, _ = self.attachment.attach("mooncake")
        self.assertFalse(ok)
        self.assertTrue(self.cache.enable_storage)
        self.assertTrue(beliefs.contains("stored"))
        self.controller.attach_storage_backend.assert_called_once()


if __name__ == "__main__":
    unittest.main()
