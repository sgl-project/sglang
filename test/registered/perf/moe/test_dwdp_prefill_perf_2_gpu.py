"""DWDP prefill throughput against TP on the same 2 GPUs (gpt-oss-20b, prefill only)."""

import unittest

from sglang.srt.utils import is_cuda, is_xpu
from sglang.test.ci.ci_register import register_cuda_ci, register_xpu_ci
from sglang.test.dwdp_test_utils import assert_prefill_perf_vs_tp
from sglang.test.test_utils import CustomTestCase

register_xpu_ci(est_time=2400, suite="nightly-xpu-2-gpu", nightly=True)
register_cuda_ci(est_time=900, stage="nightly", runner_config="2-gpu-large")

NUM_GPUS = 2


@unittest.skipUnless(
    is_cuda() or is_xpu(), "DWDP runs only on NVIDIA CUDA and Intel XPU"
)
class TestDwdpPrefillPerf2Gpu(CustomTestCase):
    def test_prefill_throughput_vs_tp(self):
        assert_prefill_perf_vs_tp(self, NUM_GPUS)


if __name__ == "__main__":
    unittest.main()
