"""gpt-oss-20b (mxfp4) GSM8K accuracy under DWDP at dwdp_size=2.

Accuracy is the only usable gate: a byte-diff against the TP control cannot serve
as one, because DWDP collapses the layer to moe_tp_size=1 while plain TP splits
the intermediate axis, so the two sum expert outputs in a different order and
greedy argmax flips at near-ties even when both arms are correct.

This is also the only test of the mxfp4 weight/scale pairing: DWDP manages
w13_weight and w2_weight only, so the *_weight_scale companions stay replicated
per rank while the packed weights arrive from peers.

Runs on CUDA or Intel XPU, whichever the host has; the TP control arm is
test_dwdp_gpt_oss_20b_tp_control.py.
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

register_xpu_ci(est_time=1800, suite="nightly-xpu-2-gpu", nightly=True)
register_cuda_ci(est_time=900, stage="extra-b", runner_config="4-gpu-h100")

MODEL = "openai/gpt-oss-20b"
DWDP_SIZE = 2
LAUNCH_TIMEOUT = 1800


@unittest.skipUnless(is_cuda() or is_xpu(), "requires a CUDA or Intel XPU device")
class TestDwdpGptOss20B(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = launch_dwdp_server(
            MODEL,
            cls.base_url,
            extra_args=[
                # DWDP asserts dwdp_size == tp_size and forces dp_size, ep_size,
                # moe_a2a_backend=none and disable_cuda_graph on its own.
                "--dwdp-size",
                str(DWDP_SIZE),
                # DWDP's two staging slots cost 2x the per-layer expert bytes on
                # top of the owned shard; the TP control fits under 0.75, this
                # arm does not.
                "--mem-fraction-static",
                "0.85",
            ],
            timeout=LAUNCH_TIMEOUT,
            tp_size=DWDP_SIZE,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        # Both arms assert the same floor: the claim under test is that prefetching
        # experts does not change what the model answers.
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
