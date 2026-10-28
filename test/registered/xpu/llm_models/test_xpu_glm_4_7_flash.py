"""GLM-4.7-Flash GSM8K accuracy on Intel XPU (TP=4).

Scored by sgl-eval with the GLM-4.7-Flash cookbook sampling settings.
"""

import unittest

import torch

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_xpu_ci(est_time=2400, suite="nightly-xpu-kernel-main-4-gpu", nightly=True)


@unittest.skipUnless(
    torch.xpu.is_available(),
    "Intel XPU not available (torch.xpu.is_available() returned False)",
)
class TestGLM4_7FlashXPU(GSM8KMixin, CustomTestCase):
    model = "zai-org/GLM-4.7-Flash"
    base_url = DEFAULT_URL_FOR_TEST
    gsm8k_backend = "sgl_eval"
    gsm8k_score_threshold = 0.80
    gsm8k_num_examples = 50
    gsm8k_num_threads = 4
    # Cookbook sampling; the chat template thinks by default.
    gsm8k_thinking = True
    gsm8k_temperature = 0.7
    gsm8k_top_p = 0.95
    gsm8k_max_tokens = 2048

    @classmethod
    def setUpClass(cls):
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=3600,
            other_args=[
                "--device",
                "xpu",
                "--attention-backend",
                "intel_xpu",
                "--dtype",
                "bfloat16",
                "--tp-size",
                "4",
                "--trust-remote-code",
                "--reasoning-parser",
                "glm45",
                "--mem-fraction-static",
                "0.85",
                # More concurrent requests lose the device on Arc Pro B60.
                "--max-running-requests",
                "4",
                "--disable-overlap-schedule",
                "--disable-radix-cache",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)


if __name__ == "__main__":
    unittest.main()
