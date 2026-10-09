"""Real TP2 and PP2 capture control across separate P/D HTTP endpoints."""

import unittest

import test_training_capture_pd_control as control
import torch
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=1200, stage="base-b", runner_config="2-gpu-large")


@unittest.skipUnless(torch.cuda.device_count() >= 2, "two CUDA devices required")
class TestTensorPDCaptureControl(control.TestPDCaptureControl):
    tp_size = 2


@unittest.skipUnless(torch.cuda.device_count() >= 2, "two CUDA devices required")
class TestPipelinePDCaptureControl(control.TestPDCaptureControl):
    pp_size = 2


class TestDSparkTensorPDCaptureControl(TestTensorPDCaptureControl):
    draft_kind = "target_kv"


class TestDSparkPipelinePDCaptureControl(TestPipelinePDCaptureControl):
    draft_kind = "target_kv"


if __name__ == "__main__":
    unittest.main()
