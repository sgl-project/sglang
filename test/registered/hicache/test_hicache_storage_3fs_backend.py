"""
Benchmark tests for HiCache Storage with 3FS backend.
Usage:
    python3 -m pytest test/registered/hicache/test_hicache_storage_3fs_backend.py -v
"""

import json
import os
import tempfile
import unittest

import torch
from test_hicache_storage_file_backend import HiCacheStorageBaseMixin

from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.storage.hf3fs.mini_3fs_metadata_server import (
    Hf3fsLocalMetadataClient,
)
from sglang.srt.mem_cache.storage.hf3fs.storage_hf3fs import HiCacheHF3FS
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=206, stage="base-b", runner_config="2-gpu-large")
register_amd_ci(est_time=300, suite="base-b-test-2-gpu-large")


PAGE_BYTES = 16


class HiCacheStorage3FSBackendBaseMixin(HiCacheStorageBaseMixin):
    """Base mixin class with common setup and utilities"""

    @classmethod
    def _get_additional_server_args_and_env(cls):
        """Get additional server arguments specific to configuration - override in subclasses"""
        # Create a temporary JSON config file for HF3FS
        hf3fs_config = {
            "file_path_prefix": os.path.join(cls.temp_dir, "hicache"),
            "file_size": 1024 * 1024 * 1024 * 2,
            "numjobs": 2,
            "entries": 8,
            "use_mock_hf3fs_client": True,
            "hicache_storage_pass_prefix_keys": True,
        }

        # Write config to temporary file
        config_file = os.path.join(cls.temp_dir, "hf3fs_config.json")
        with open(config_file, "w") as f:
            json.dump(hf3fs_config, f, indent=2)

        server_args = {
            "--tp-size": 1,
            "--hicache-ratio": 1.2,
            "--hicache-storage-backend": "hf3fs",
            "--hicache-storage-backend-extra-config": json.dumps(hf3fs_config),
        }

        # Set the environment variable to point to our config file
        env_vars = {
            "SGLANG_HICACHE_HF3FS_CONFIG_PATH": config_file,
        }

        return server_args, env_vars


class TestHf3fsBackendLayerFirstLayout(
    HiCacheStorage3FSBackendBaseMixin, CustomTestCase
):
    """Layer first layout tests for HiCache-Hf3fs backend"""

    @classmethod
    def _get_additional_server_args_and_env(cls):
        """Get additional server arguments specific to configuration - override in subclasses"""
        server_args, env_vars = super()._get_additional_server_args_and_env()
        server_args["--hicache-mem-layout"] = "layer_first"
        server_args["--hicache-io-backend"] = "direct"
        server_args["--tp-size"] = 2
        return server_args, env_vars


class TestHf3fsBackendAccuracy(HiCacheStorage3FSBackendBaseMixin, CustomTestCase):
    """Accuracy tests for HiCache-Hf3fs backend"""

    @classmethod
    def _get_additional_server_args_and_env(cls):
        """Get additional server arguments specific to configuration - override in subclasses"""
        server_args, env_vars = super()._get_additional_server_args_and_env()
        server_args["--hicache-ratio"] = 1.5
        server_args["--tp-size"] = 2
        server_args["--hicache-mem-layout"] = "page_first_direct"
        server_args["--hicache-io-backend"] = "direct"
        return server_args, env_vars

    def test_eval_accuracy(self):
        """Test eval accuracy with cache persistence across cache flushes"""
        from test_hicache_storage_file_backend import run_eval_accuracy_test

        run_eval_accuracy_test(self)


class _StatePool:
    page_size = 1

    def __init__(self, value):
        self.buffer = torch.full((1, PAGE_BYTES), value, dtype=torch.uint8)

    def get_ksize_per_token(self):
        return PAGE_BYTES

    def get_dummy_flat_data_page(self):
        return torch.zeros(PAGE_BYTES, dtype=torch.uint8)

    def get_data_page(self, index, flat=True):
        return self.buffer[index].clone()

    def set_from_flat_data_page(self, index, data_page):
        self.buffer[index].copy_(data_page)


class TestHf3fsMambaKeys(CustomTestCase):
    def test_mla_mamba_round_trip_keeps_each_tp_rank(self):
        """MLA shares KV keys, but Mamba checkpoints must not overwrite another rank."""
        tmp = tempfile.mkdtemp()
        metadata = Hf3fsLocalMetadataClient()
        transfer = PoolTransfer(
            name=PoolName.MAMBA, keys=["page"], host_indices=torch.tensor([0])
        )
        ranks = []
        for rank in range(2):
            store = HiCacheHF3FS(
                rank=rank,
                file_path=os.path.join(tmp, "hicache.bin"),
                file_size=PAGE_BYTES * 16,
                numjobs=1,
                bytes_per_page=PAGE_BYTES,
                entries=8,
                client_timeout=5,
                dtype=torch.uint8,
                metadata_client=metadata,
                is_mla_model=True,
                use_mock_client=True,
                tp_size=2,
            )
            self.addCleanup(store.close)
            pool = _StatePool(rank + 1)
            store.register_mem_host_pool_v2(pool, PoolName.MAMBA)
            self.assertEqual(store.batch_set_v2([transfer])[PoolName.MAMBA], [True])
            ranks.append((store, pool))

        for rank, (store, pool) in enumerate(ranks):
            pool.buffer.zero_()
            self.assertEqual(store.batch_get_v2([transfer])[PoolName.MAMBA], [True])
            self.assertTrue(torch.all(pool.buffer == rank + 1))


if __name__ == "__main__":
    unittest.main(verbosity=2)
