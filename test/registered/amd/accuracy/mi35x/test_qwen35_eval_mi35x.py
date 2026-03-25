"""MI35x Qwen 3.5 GSM8K lm-eval Evaluation Test (8-GPU)

Tests Qwen/Qwen3.5-397B-A17B (MoE, Hybrid Attention with Gated Delta Networks)
with lm-eval GSM8K benchmark on MI35x, matching the AMD Day 0 article.

Registry: nightly-amd-accuracy-8-gpu-mi35x-qwen35 suite
"""

import os
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.kits.lm_eval_kit import LMEvalMixin
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


class TestQwen35EvalMI35x(LMEvalMixin, CustomTestCase):
    """Qwen 3.5 GSM8K lm-eval Test for AMD MI35x."""

    model_config_name = "lm_eval_configs/Qwen3.5-397B-A17B.yaml"

    @classmethod
    def setUpClass(cls):
        cls.model = QWEN35_MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST

    def test_lm_eval(self):
        """Override to handle server lifecycle within test method (MI35x pattern)."""
        other_args = [
            "--tp",
            str(TP_SIZE),
            "--attention-backend",
            "triton",
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
            requests.get(self.base_url + "/flush_cache")
            super().test_lm_eval()
        finally:
            kill_process_tree(process.pid)


if __name__ == "__main__":
    unittest.main()
