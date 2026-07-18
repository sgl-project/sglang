"""Unit tests for the kv_size_mb / kv_size_thres helper."""

import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.memory_threshold import (
    gpu_family_from_text,
    kv_size_mb_from_server_info,
    resolve_kv_size_thres,
)
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestKvSizeThreshold(CustomTestCase):
    def test_kv_size_mb_from_server_info(self):
        info = {
            "internal_states": [
                {
                    "memory_usage": {
                        "weight": 10.0,
                        "kvcache": 2.0,
                        "graph": 3.0,
                        "mamba": 0.5,
                        "kv_size_mb": 2560.0,
                    }
                }
            ]
        }
        self.assertEqual(kv_size_mb_from_server_info(info), 2560.0)

    def test_kv_size_mb_fallback_excludes_weight_graph(self):
        info = {
            "memory_usage": {
                "weight": 10.0,
                "kvcache": 1.0,
                "graph": 5.0,
                "mamba": 1.0,
            }
        }
        # Only kvcache + mamba = 2 GB * 1024
        self.assertEqual(kv_size_mb_from_server_info(info), 2048.0)

    def test_gpu_family(self):
        self.assertEqual(gpu_family_from_text("NVIDIA H200"), "h200")
        self.assertEqual(gpu_family_from_text("8-gpu-b200"), "b200")

    def test_resolve_kv_size_thres(self):
        self.assertEqual(resolve_kv_size_thres(100), 100.0)
        self.assertEqual(
            resolve_kv_size_thres({"h200": 10, "b200": 20}, gpu_family="b200"),
            20.0,
        )
        self.assertIsNone(resolve_kv_size_thres({"h200": 10}, gpu_family="b200"))


if __name__ == "__main__":
    unittest.main()
