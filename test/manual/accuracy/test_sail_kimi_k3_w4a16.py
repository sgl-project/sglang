"""Kimi-K3 W4A16 MMA accuracy check on SAIL TP16; needs a local checkpoint."""

import os
import unittest

import torch

from sglang.srt.environ import envs
from sglang.srt.utils import is_sail
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.test_utils import (
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

# Not registered in CI: it needs a dedicated checkpoint and 16 GPUs, so the
# manual-run contract is kept.
MODEL_PATH = os.environ.get("MODEL_PATH", "Kimi-K3-MXFP4-W4A16")
SERVED_MODEL_NAME = "Kimi-K3-MXFP4-W4A16-MMA"
BASE_URL = "http://127.0.0.1:9000"


class TestSailKimiK3W4a16Mma(CustomTestCase, GSM8KMixin):
    model = SERVED_MODEL_NAME
    base_url = BASE_URL
    gsm8k_score_threshold = 0.50
    gsm8k_num_examples = 200
    gsm8k_accept_length_thres = 1.0
    # New chat accuracy tests use the unified sgl-eval; gsm8k_data_path is not
    # read by the mixin.
    gsm8k_backend = "sgl_eval"
    gsm8k_max_tokens = 512

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if not is_sail() or not torch.cuda.is_available():
            raise unittest.SkipTest("Requires a SAIL device.")
        if torch.cuda.device_count() < 16:
            raise unittest.SkipTest("Requires 16 SAIL devices to run TP16.")
        with (
            envs.SGLANG_SAIL_DEEPGEMM_MXFP4_W4A16.override(False),
            envs.SGLANG_SAIL_DEEPGEMM_MXFP4_W4A16_MMA.override(True),
            envs.SGLANG_SAIL_DEEPGEMM_MOE_TP_FUSED.override(True),
        ):
            cls.process = popen_launch_server(
                MODEL_PATH,
                cls.base_url,
                timeout=24000,
                other_args=[
                    "--tp-size",
                    "16",
                    "--trust-remote-code",
                    "--watchdog-timeout",
                    "600",
                    "--served-model-name",
                    cls.model,
                    "--max-total-tokens",
                    "32768",
                    "--max-running-requests",
                    "1",
                    "--mem-fraction-static",
                    "0.85",
                    "--chunked-prefill-size",
                    "-1",
                    "--disable-radix-cache",
                    "--attention-backend",
                    "triton",
                    "--moe-runner-backend",
                    "deep_gemm",
                ],
            )

    @classmethod
    def tearDownClass(cls):
        try:
            if hasattr(cls, "process") and cls.process:
                terminate_and_kill_process_tree(cls.process)
        finally:
            super().tearDownClass()


if __name__ == "__main__":
    unittest.main()
