import unittest

import torch

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.test_utils import (
    DEFAULT_HYBRID_GDN_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_xpu_ci(est_time=900, suite="nightly-xpu-kernel-main-2-gpu", nightly=True)


@unittest.skipUnless(torch.xpu.is_available(), "needs Intel XPU")
class TestXPUDCPGSM8K(GSM8KMixin, CustomTestCase):
    model = DEFAULT_HYBRID_GDN_SMALL_MODEL_NAME_FOR_TEST
    base_url = DEFAULT_URL_FOR_TEST
    gsm8k_score_threshold = 0.80
    gsm8k_num_examples = 200
    gsm8k_num_threads = 32

    @classmethod
    def setUpClass(cls):
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--device",
                "xpu",
                "--attention-backend",
                "intel_xpu",
                "--tp-size",
                "2",
                "--dcp-size",
                "2",
                "--disable-radix-cache",
                "--chunked-prefill-size",
                "256",
                "--mem-fraction-static",
                "0.60",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)


if __name__ == "__main__":
    unittest.main()
