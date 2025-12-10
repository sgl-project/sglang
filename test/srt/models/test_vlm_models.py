import argparse
import random
import sys
import unittest
from types import SimpleNamespace

from sglang.srt.utils import is_hip
from sglang.test.mmmu_vlm_mixin import DEFAULT_MEM_FRACTION_STATIC, MMMUVLMMixin
from sglang.test.test_utils import CustomTestCase, is_in_ci

_is_hip = is_hip()
# VLM models for testing
if _is_hip:
    MODELS = [SimpleNamespace(model="openbmb/MiniCPM-V-2_6", mmmu_accuracy=0.4)]
else:
    MODELS = [
        SimpleNamespace(model="google/gemma-3-4b-it", mmmu_accuracy=0.38),
        SimpleNamespace(model="Qwen/Qwen2.5-VL-3B-Instruct", mmmu_accuracy=0.4),
        SimpleNamespace(model="openbmb/MiniCPM-V-2_6", mmmu_accuracy=0.4),
    ]


class TestVLMModels(MMMUVLMMixin, CustomTestCase):
    def _detect_eviction_in_logs(self, log_output: str) -> tuple[bool, int]:
        """Detect if eviction events occurred in the log output."""
        eviction_keyword = "Cache eviction"

        eviction_detected = False
        eviction_count = 0

        for line in log_output.split("\n"):
            if eviction_keyword in line:
                eviction_detected = True
                eviction_count += 1
                print(f"Eviction detected: {line.strip()}")

        return eviction_detected, eviction_count

    def test_vlm_mmmu_benchmark(self):
        """Test VLM models against MMMU benchmark."""
        models_to_test = MODELS

        if is_in_ci():
            models_to_test = [random.choice(MODELS)]

        for model in models_to_test:
            self._run_vlm_mmmu_test(model, "./logs")

    def test_vlm_mmmu_benchmark_with_small_cache(self):
        """Test VLM models with a tiny embedding cache to exercise eviction logic."""
        models_to_test = MODELS

        if is_in_ci():
            models_to_test = [random.choice(MODELS)]

        for model in models_to_test:
            custom_env = {"SGLANG_VLM_CACHE_SIZE_MB": "5"}
            server_output = self._run_vlm_mmmu_test(
                model,
                "./logs_small_cache",
                test_name=" with small embedding cache (evict test)",
                custom_env=custom_env,
                log_level="debug",
                capture_output=True,
            )
            print("Server output:\n", server_output)

            eviction_detected, eviction_count = self._detect_eviction_in_logs(
                server_output
            )

            self.assertTrue(
                eviction_detected,
                (
                    "Expected eviction events to be detected with small cache (5MB), "
                    "but none found. Cache size may be too large for the workload or "
                    "eviction logic may not be working."
                ),
            )

            print(
                f"Eviction detection summary: {eviction_count} eviction events detected"
            )

            if eviction_detected:
                print("✅ Eviction logic successfully triggered and detected!")


if __name__ == "__main__":
    # Define and parse arguments here, before unittest.main
    parser = argparse.ArgumentParser(description="Test VLM models")
    parser.add_argument(
        "--mem-fraction-static",
        type=float,
        help="Static memory fraction for the model",
        default=DEFAULT_MEM_FRACTION_STATIC,
    )

    # Parse args intended for unittest
    args = parser.parse_args()

    # Store the parsed args object on the class
    TestVLMModels.parsed_args = args

    # Pass args to unittest
    unittest.main(argv=[sys.argv[0]])
