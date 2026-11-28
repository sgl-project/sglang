"""DWDP prefill throughput against TP on the same 4 GPUs (gpt-oss-20b, prefill only)."""

import unittest

from sglang.srt.utils import is_xpu
from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.dwdp_test_utils import assert_prefill_perf_vs_tp
from sglang.test.test_utils import CustomTestCase

register_xpu_ci(est_time=2400, suite="nightly-kernel-main-test-4-gpu-xpu", nightly=True)

NUM_GPUS = 4


@unittest.skipUnless(is_xpu(), "requires Intel XPU")
class TestDwdpPrefillPerf4Gpu(CustomTestCase):
    def test_prefill_throughput_vs_tp(self):
        assert_prefill_perf_vs_tp(self, NUM_GPUS)


if __name__ == "__main__":
    unittest.main()
