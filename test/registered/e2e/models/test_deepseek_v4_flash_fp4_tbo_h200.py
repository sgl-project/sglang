"""H200 extra coverage: DeepSeek-V4-Flash FP4 with non-EP two-batch overlap.

TP 8 with attention DP 2 runs the overlapped DP gather and combine at attention
TP 4.
"""

import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    try_cached_model,
)

register_cuda_ci(est_time=300, stage="extra-b", runner_config="8-gpu-h200")

MODEL = "deepseek-ai/DeepSeek-V4-Flash"
SERVER_LAUNCH_TIMEOUT = 3600


class TestDSV4FlashFP4TboH200(GSM8KMixin, CustomTestCase):
    gsm8k_score_threshold = 0.93

    @classmethod
    def setUpClass(cls):
        cls.model = try_cached_model(MODEL)
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=SERVER_LAUNCH_TIMEOUT,
            other_args=[
                "--trust-remote-code",
                "--tp",
                "8",
                "--attn-dp-size",
                "2",
                "--enable-prefill-delayer",
                "--enable-two-batch-overlap",
                "--moe-a2a-backend",
                "none",
                "--moe-runner-backend",
                "marlin",
                "--disable-shared-experts-fusion",
                "--disable-radix-cache",
                "--watchdog-timeout",
                "900",
            ],
            env={
                "SGLANG_SHARED_EXPERT_TP1": "1",
                "SGLANG_DP_USE_GATHERV": "1",
                "SGLANG_DP_USE_REDUCE_SCATTER": "1",
            },
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)


if __name__ == "__main__":
    unittest.main()
