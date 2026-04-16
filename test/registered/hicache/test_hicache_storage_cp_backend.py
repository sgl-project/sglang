"""
E2E tests for HiCache storage with context parallelism and tensor parallelism.

Usage:
    python3 -m pytest test/registered/hicache/test_hicache_storage_cp_backend.py -v
"""

import unittest
from typing import Dict

import torch
from test_hicache_storage_file_backend import HiCacheStorageBaseMixin
from test_hicache_storage_mooncake_backend import HiCacheStorageMooncakeBackendBaseMixin

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=900, suite="stage-c-test-4-gpu-h100")

QWEN3_30B_MODEL_PATH = "Qwen/Qwen3-30B-A3B-FP8"


def has_at_least_4_cuda_gpus() -> bool:
    return torch.cuda.is_available() and torch.cuda.device_count() >= 4


def get_qwen3_30b_cp_tp_hicache_args() -> Dict:
    return {
        "--tp-size": 4,
        "--moe-dp-size": 2,
        "--ep-size": 2,
        "--attn-cp-size": 2,
        "--enable-prefill-context-parallel": True,
        "--trust-remote-code": True,
        "--cuda-graph-max-bs": 32,
        "--max-running-requests": 32,
        "--max-total-tokens": 8192,
        "--model-loader-extra-config": (
            '{"enable_multithread_load": true, "num_threads": 64}'
        ),
        "--hicache-mem-layout": "page_first_direct",
        "--hicache-io-backend": "direct",
    }


@unittest.skipUnless(
    has_at_least_4_cuda_gpus(), "Requires at least 4 CUDA GPUs for TP4+CP2"
)
class TestHiCacheStorageFileQwen330BCPAndTP(HiCacheStorageBaseMixin, CustomTestCase):
    """Qwen3-30B with file HiCache storage, context parallelism, and tensor parallelism."""

    @classmethod
    def _get_model_name(cls):
        return QWEN3_30B_MODEL_PATH

    @classmethod
    def _get_additional_server_args_and_env(cls):
        server_args, env_vars = super()._get_additional_server_args_and_env()
        server_args.update(get_qwen3_30b_cp_tp_hicache_args())
        return server_args, env_vars


@unittest.skipUnless(
    has_at_least_4_cuda_gpus(), "Requires at least 4 CUDA GPUs for TP4+CP2"
)
class TestHiCacheStorageMooncakeQwen330BCPAndTP(
    HiCacheStorageMooncakeBackendBaseMixin, CustomTestCase
):
    """Qwen3-30B with Mooncake HiCache storage, context parallelism, and tensor parallelism."""

    @classmethod
    def _get_model_name(cls):
        return QWEN3_30B_MODEL_PATH

    @classmethod
    def _get_additional_server_args_and_env(cls):
        server_args, env_vars = super()._get_additional_server_args_and_env()
        server_args.update(get_qwen3_30b_cp_tp_hicache_args())
        return server_args, env_vars


if __name__ == "__main__":
    unittest.main(verbosity=2)
