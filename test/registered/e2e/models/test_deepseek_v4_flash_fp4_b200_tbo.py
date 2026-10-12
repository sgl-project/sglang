"""B200 nightly CI: DeepSeek-V4-Flash FP4 with two-batch overlap.

Launches TP=4, DP attention 4 and DeepEP with --enable-two-batch-overlap, so
prefill runs DeepSeek-V4's own two-batch-overlap layer loop. Runs a GSM8K
accuracy gate.

Registry: nightly-4-gpu-b200 (nightly, 4x B200)
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

register_cuda_ci(est_time=600, stage="nightly", runner_config="4-gpu-b200")

MODEL = "deepseek-ai/DeepSeek-V4-Flash-0731"
SERVER_LAUNCH_TIMEOUT = 3600
DEEPEP_CONFIG = '{"normal_dispatch":{"num_sms":96},"normal_combine":{"num_sms":96}}'


class TestDSV4FlashFP4TboB200(GSM8KMixin, CustomTestCase):
    gsm8k_accuracy_thres = 0.93

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
                "4",
                "--attn-dp-size",
                "4",
                "--moe-a2a-backend",
                "deepep",
                "--deepep-config",
                DEEPEP_CONFIG,
                "--enable-two-batch-overlap",
            ],
            env={"SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "1024"},
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)


if __name__ == "__main__":
    unittest.main()
