"""Pause, resume and fenced abort across all combined TP/PP P/D owners."""

import unittest

import test_training_capture_pd_control as control
import torch
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=900, stage="extra-b", runner_config="4-gpu-h100")


@unittest.skipUnless(torch.cuda.device_count() >= 4, "four CUDA devices required")
class TestCombinedPDCaptureControl(control.TestPDCaptureControl):
    tp_size = 2
    pp_size = 2


class TestDSparkCombinedPDCaptureControl(TestCombinedPDCaptureControl):
    draft_kind = "target_kv"


if __name__ == "__main__":
    unittest.main()
