"""PPU Flash Attention 3 wrapper verification.

Launches a server with FA3 attention backend on PPU and verifies
generation correctness via GSM8K evaluation.
"""

import os
import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_ppu_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_ppu_ci(est_time=106, suite="stage-b-test-1-gpu-ppu")

PPU_SMALL_MODEL = os.environ.get(
    "PPU_SMALL_MODEL_PATH",
    "Qwen/Qwen2.5-0.5B-Instruct",
)


class TestPPUFlashAttention(CustomTestCase, GSM8KMixin):
    gsm8k_accuracy_thres = 0.28
    gsm8k_num_questions = 200
    gsm8k_num_threads = 64

    @classmethod
    def setUpClass(cls):
        cls.model = PPU_SMALL_MODEL
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--attention-backend",
                "fa3",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)


if __name__ == "__main__":
    unittest.main(verbosity=3)
