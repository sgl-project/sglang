"""
Benchmark tests for HiCache Storage with 3FS backend.
Usage:
    python3 -m pytest test/registered/hicache/test_hicache_storage_3fs_backend.py -v
"""

import json
import os
import unittest

from test_hicache_storage_file_backend import HiCacheStorageBaseMixin

from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=150, suite="stage-b-test-2-gpu-large")
register_amd_ci(est_time=300, suite="stage-b-test-2-gpu-large")


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


# ---------------------------------------------------------------------------
# Hybrid (v2) end-to-end test — PLAN.md §5 #10
# ---------------------------------------------------------------------------


try:
    from sglang.test.test_utils import DEFAULT_HYBRID_MAMBA_MODEL_NAME_FOR_TEST

    _HYBRID_MODEL = DEFAULT_HYBRID_MAMBA_MODEL_NAME_FOR_TEST
except Exception:  # pragma: no cover
    _HYBRID_MODEL = None


@unittest.skipIf(
    _HYBRID_MODEL is None,
    "No hybrid (Mamba/linear-attention) test model registered in sglang.test.test_utils",
)
class TestHf3fsBackendHybrid(HiCacheStorage3FSBackendBaseMixin, CustomTestCase):
    """End-to-end HiCache-Hf3fs hybrid test (KV + MAMBA pools).

    Launches a hybrid/linear-attention model with the hf3fs storage backend
    and confirms that the second run after a flush reports a non-zero cache
    hit, proving the v2 path (batch_exists_v2 / batch_get_v2 / batch_set_v2)
    is wired up end-to-end.
    """

    @classmethod
    def _get_model_name(cls):
        return _HYBRID_MODEL

    @classmethod
    def _get_additional_server_args_and_env(cls):
        server_args, env_vars = super()._get_additional_server_args_and_env()
        server_args["--tp-size"] = 1
        server_args["--hicache-storage-prefetch-policy"] = "wait_complete"
        return server_args, env_vars

    def test_hybrid_cache_hit_after_flush(self):
        """Prime the cache, flush, re-run the same prompt, expect a hit."""
        import time

        prompt = self.gen_prompt(768)

        # Prime.
        r1 = self.send_request(prompt, max_tokens=32)
        self.assertIsNotNone(r1)

        # Force eviction to disk and flush the device cache.
        self.trigger_offloading_and_flush()

        start = time.time()
        r2 = self.send_request(prompt, max_tokens=32)
        elapsed = time.time() - start

        cached_tokens = self.get_cached_tokens(r2)
        print(
            f"[hybrid] second-run cached_tokens={cached_tokens} "
            f"latency={elapsed:.3f}s"
        )
        # For a hybrid model, a non-trivial number of cached tokens on the
        # second run proves that both KV and the auxiliary (mamba) pool
        # were restored from 3FS.
        self.assertGreater(
            cached_tokens,
            700,
            "Expected a significant remote cache hit for the hybrid model; "
            "this indicates the v2 path (KV + MAMBA) restored correctly.",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
