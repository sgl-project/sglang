"""gpt-oss-20b (mxfp4) GSM8K accuracy at TP=2, the control arm for DWDP.

Same model and same threshold as test_dwdp_gpt_oss_20b.py, with expert weights
sharded on the TP axis instead of prefetched from peers. A drop in both files is
a plain mxfp4 regression on this backend; a drop only in the DWDP file is a
prefetch bug. Without this arm a DWDP failure cannot be attributed.
"""

import unittest

from sglang.srt.utils import is_cuda, is_xpu, kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci, register_xpu_ci
from sglang.test.dwdp_test_utils import (
    GPT_OSS_20B_GSM8K_FLOOR,
    assert_gsm8k_accuracy,
    launch_dwdp_server,
)
from sglang.test.test_utils import DEFAULT_URL_FOR_TEST, CustomTestCase

register_xpu_ci(est_time=600, suite="nightly-xpu-2-gpu", nightly=True)
register_cuda_ci(est_time=600, stage="extra-b", runner_config="2-gpu-large")

MODEL = "openai/gpt-oss-20b"
TP_SIZE = 2
LAUNCH_TIMEOUT = 1800


@unittest.skipUnless(is_cuda() or is_xpu(), "requires a CUDA or Intel XPU device")
class TestDwdpGptOss20BTpControl(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = launch_dwdp_server(
            MODEL,
            cls.base_url,
            extra_args=["--mem-fraction-static", "0.85"],
            timeout=LAUNCH_TIMEOUT,
            tp_size=TP_SIZE,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        # 8 threads: the XPU default is single-stream because intel_xpu attention
        # wedged the driver on concurrent prefill, and 8 measured clean here
        # across four full gpt-oss-20b runs at TP=2.
        assert_gsm8k_accuracy(
            self,
            model=MODEL,
            base_url=self.base_url,
            accuracy=GPT_OSS_20B_GSM8K_FLOOR,
            num_examples=100,
            num_threads=8,
        )


if __name__ == "__main__":
    unittest.main()
