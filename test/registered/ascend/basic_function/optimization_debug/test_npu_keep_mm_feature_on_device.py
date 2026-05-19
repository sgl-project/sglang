import time
import unittest

from sglang.test.ascend.test_ascend_utils import PHI_4_MULTIMODAL_INSTRUCT_WEIGHTS_PATH
from sglang.test.ascend.vlm_utils import TestVLMModels
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=800, suite="nightly-1-npu-a3", nightly=True)


class TestPhi4MultimodalLatencyCompare(TestVLMModels):
    """Testcase: Verify that enabling --keep-mm-feature-on-device improves inference speed for multimodal models.

    [Test Category] Parameter
    [Test Target] --keep-mm-feature-on-device
    """

    model = PHI_4_MULTIMODAL_INSTRUCT_WEIGHTS_PATH
    mmmu_accuracy = 0.2

    base_args = [
        "--trust-remote-code",
        "--mem-fraction-static",
        "0.8",
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
    ]

    def _run_and_measure_time(self, extra_args):
        self.other_args = self.base_args + extra_args
        start = time.perf_counter()
        metrics = self._run_vlm_mmmu_test()
        duration = time.perf_counter() - start
        print(f"Duration: {duration:.2f}s | metrics: {metrics}")
        return duration

    def test_vlm_mmmu_latency_compare(self):
        # Without --keep-mm-feature-on-device
        time_off = self._run_and_measure_time([])

        # With --keep-mm-feature-on-device enabled
        time_on = self._run_and_measure_time(["--keep-mm-feature-on-device"])

        # Latency should decrease after enabling --keep-mm-feature-on-device
        self.assertLess(
            time_on,
            time_off,
            "Latency did not decrease after enabling --keep-mm-feature-on-device",
        )


if __name__ == "__main__":
    unittest.main()
