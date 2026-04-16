"""
E2E tests for DeepSeek V3.2 HiCache storage with NSA context parallelism.

Usage:
    python3 -m pytest test/registered/hicache/test_hicache_storage_dsv32_cp_backend.py -v
"""

import unittest
from typing import Dict

import torch
from test_hicache_storage_file_backend import HiCacheStorageBaseMixin
from test_hicache_storage_mooncake_backend import HiCacheStorageMooncakeBackendBaseMixin

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=1200, suite="stage-c-test-deepep-8-gpu-h200")

DEEPSEEK_V32_MODEL_PATH = "deepseek-ai/DeepSeek-V3.2"


def has_at_least_8_cuda_gpus() -> bool:
    return torch.cuda.is_available() and torch.cuda.device_count() >= 8


def get_deepseek_v32_cp_hicache_args() -> Dict:
    return {
        "--tp": 8,
        "--attn-cp-size": 8,
        "--enable-nsa-prefill-context-parallel": True,
        "--nsa-prefill-cp-mode": "round-robin-split",
        "--trust-remote-code": True,
        "--mem-fraction-static": 0.7,
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
    has_at_least_8_cuda_gpus(), "Requires at least 8 CUDA GPUs for TP8+CP8"
)
class TestHiCacheStorageFileDeepseekV32CP(HiCacheStorageBaseMixin, CustomTestCase):
    """DeepSeek V3.2 with file HiCache storage and NSA context parallelism."""

    @classmethod
    def _get_model_name(cls):
        return DEEPSEEK_V32_MODEL_PATH

    @classmethod
    def _get_additional_server_args_and_env(cls):
        server_args, env_vars = super()._get_additional_server_args_and_env()
        server_args.update(get_deepseek_v32_cp_hicache_args())
        return server_args, env_vars


@unittest.skipUnless(
    has_at_least_8_cuda_gpus(), "Requires at least 8 CUDA GPUs for TP8+CP8"
)
class TestHiCacheStorageMooncakeDeepseekV32CP(
    HiCacheStorageMooncakeBackendBaseMixin, CustomTestCase
):
    """DeepSeek V3.2 with Mooncake HiCache storage and NSA context parallelism."""

    @classmethod
    def _get_model_name(cls):
        return DEEPSEEK_V32_MODEL_PATH

    @classmethod
    def _get_additional_server_args_and_env(cls):
        server_args, env_vars = super()._get_additional_server_args_and_env()
        server_args.update(get_deepseek_v32_cp_hicache_args())
        return server_args, env_vars


if __name__ == "__main__":
    unittest.main(verbosity=2)
