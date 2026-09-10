"""MI35x Qwen 3.5 GSM8K sgl-eval Evaluation Test (8-GPU)

Tests Qwen/Qwen3.5-397B-A17B (MoE, Hybrid Attention with Gated Delta Networks)
with sgl-eval GSM8K benchmark on MI35x; historical thresholds require recalibration.

Registry: nightly-amd-accuracy-8-gpu-mi35x-qwen35 suite
"""

import os
import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_amd_ci(
    est_time=3600, suite="nightly-amd-accuracy-8-gpu-mi35x-qwen35", nightly=True
)

QWEN35_MODEL_PATH = "Qwen/Qwen3.5-397B-A17B"
SERVER_LAUNCH_TIMEOUT = 3600
TP_SIZE = 8


class TestQwen35EvalMI35x(GSM8KMixin, CustomTestCase):
    """Qwen 3.5 GSM8K sgl-eval Test for AMD MI35x."""

    gsm8k_score_threshold = 0.9704 * (1 - 0.05)
    gsm8k_num_examples = 1319
    gsm8k_num_threads = 256
    gsm8k_max_tokens = 2048
    gsm8k_thinking = True

    @classmethod
    def setUpClass(cls):
        cls.model = QWEN35_MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST

    def test_gsm8k(self):
        """Override to handle server lifecycle and write results to summary."""
        other_args = [
            "--tp",
            str(TP_SIZE),
            "--attention-backend",
            "aiter",
            "--trust-remote-code",
            "--model-loader-extra-config",
            '{"enable_multithread_load": true}',
            "--watchdog-timeout",
            "1200",
        ]
        env = os.environ.copy()
        env["SGLANG_USE_AITER"] = "1"

        process = popen_launch_server(
            QWEN35_MODEL_PATH,
            self.base_url,
            timeout=SERVER_LAUNCH_TIMEOUT,
            other_args=other_args,
            env=env,
        )

        try:
            super().test_gsm8k()
        finally:
            kill_process_tree(process.pid)


if __name__ == "__main__":
    unittest.main()
