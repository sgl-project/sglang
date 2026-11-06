"""GLM-4.7-Flash GSM8K accuracy on Intel XPU (TP=4).

Scored by sgl-eval's ``gsm8k`` benchmark with the GLM-4.7-Flash cookbook
sampling settings.
"""

import unittest

from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang.test.xpu.simple_eval_gsm8k_xpu_mixin import SimpleEvalGSM8KXPUMixin

register_xpu_ci(est_time=2400, suite="nightly-xpu-kernel-main-4-gpu", nightly=True)


class TestGLM4_7FlashXPU(SimpleEvalGSM8KXPUMixin, CustomTestCase):
    model = "zai-org/GLM-4.7-Flash"
    tp_size = 4
    # 3 runs on 4x B60 scored 0.80-0.90; one question of margin below the lowest.
    accuracy = 0.75
    timeout_for_server_launch = 3600
    num_examples = 20
    num_threads = 4
    # Cookbook sampling; the chat template thinks by default.
    temperature = 0.7
    top_p = 0.95
    max_tokens = 2048

    other_args = SimpleEvalGSM8KXPUMixin.other_args + [
        "--reasoning-parser",
        "glm45",
        "--mem-fraction-static",
        "0.85",
        # More concurrent requests lose the device on Arc Pro B60.
        "--max-running-requests",
        "4",
    ]


if __name__ == "__main__":
    unittest.main()
