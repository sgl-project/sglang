"""Four-rank colocated target-KV serving, capture and natural retraction."""

import unittest

import test_training_capture_dspark_pp as pipeline_cases
import torch
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=600, stage="extra-b", runner_config="4-gpu-h100")


@unittest.skipUnless(torch.cuda.device_count() >= 4, "four CUDA devices required")
class TestDSparkCombinedCapture(pipeline_cases.TestDSparkPipelineCapture):
    tp_size = 2


if __name__ == "__main__":
    unittest.main()
