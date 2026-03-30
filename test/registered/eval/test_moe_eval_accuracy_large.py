"""
Usage:
python -m unittest test_moe_eval_accuracy_large.TestMoEEvalAccuracyLarge.test_mmlu
"""

import os
import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.kits.eval_accuracy_kit import HumanEvalMixin, MGSMEnMixin, MMLUMixin
from sglang.test.test_utils import (
    DEFAULT_MOE_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_amd_ci,
    popen_launch_server,
)

register_cuda_ci(est_time=500, suite="stage-b-test-2-gpu-large")
register_amd_ci(est_time=500, suite="stage-b-test-2-gpu-large-amd")


class TestMoEEvalAccuracyLarge(CustomTestCase, MMLUMixin, HumanEvalMixin, MGSMEnMixin):
    mmlu_score_threshold = 0.62
    humaneval_score_threshold = 0.40
    mgsm_en_score_threshold = 0.61

    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_MOE_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST

        # Disable AITER for AMD CI to ensure consistent results
        env = None
        if is_in_amd_ci():
            env = os.environ.copy()
            env["SGLANG_USE_AITER"] = "0"
            env["SGLANG_USE_AITER_AR"] = "0"
            env["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--log-level-http",
                "warning",
                "--tp",
                "2",
            ],
            env=env,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)


if __name__ == "__main__":
    unittest.main()
