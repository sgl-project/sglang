import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.gpt_oss_common import BaseTestGptOss

register_cuda_ci(est_time=345, stage="extra-a", runner_config="1-gpu-small")


@unittest.skipIf(not torch.cuda.is_available(), "CUDA is not available")
class TestGptOssSm120(BaseTestGptOss):
    @classmethod
    def setUpClass(cls):
        compute_capability = torch.cuda.get_device_capability()
        if compute_capability != (12, 0):
            raise unittest.SkipTest(
                f"GPT-OSS SM120 test requires SM 12.0, but found {compute_capability[0]}.{compute_capability[1]}"
            )

    def test_mxfp4_20b(self):
        # The shared 20B helper captures CUDA graphs up to bs=600, which OOMs
        # on the 1-gpu-small SM120 runner during startup.
        self._run_test_raw(
            model="openai/gpt-oss-20b",
            expected_score_of_reasoning_effort={
                "low": 0.34,
                "medium": 0.34,
                "high": 0.27,
            },
            other_args=[
                "--cuda-graph-max-bs",
                "198",
                "--max-running-requests",
                "198",
            ],
        )


if __name__ == "__main__":
    unittest.main()
