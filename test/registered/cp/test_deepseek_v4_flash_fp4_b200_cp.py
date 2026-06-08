"""B200 extra CI: DeepSeek-V4-Flash FP4 with attn-CP (DSA prefill CP).

Balanced recipe (TP=4, DeepEP, EAGLE) plus --attn-cp-size=4 with the
DSA prefill-CP round-robin-split mode. Split out of
models_e2e/test_deepseek_v4_flash_fp4_b200.py so the `cp` group covers
all context-parallel tests.

Registry: extra-b-test-4-gpu-b200 (label-gated extra CI, 4x B200)
"""

import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.basic_decode_correctness_kit import BasicDecodeCorrectnessMixin
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    try_cached_model,
)

register_cuda_ci(est_time=387, stage="extra-b", runner_config="deepep-4-gpu-b200")

MODEL = "deepseek-ai/DeepSeek-V4-Flash"
SERVER_LAUNCH_TIMEOUT = 3600
DEEPEP_CONFIG = '{"normal_dispatch":{"num_sms":96},"normal_combine":{"num_sms":96}}'

_DEEPEP_ENV = {
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "1024",
}


class TestDSV4FlashFP4B200Balanced_CP(
    BasicDecodeCorrectnessMixin,
    GSM8KMixin,
    CustomTestCase,
):
    """Balanced recipe: TP=4, DP=4, DeepEP, EAGLE (1-step spec)."""

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
                "--attn-cp-size",
                "4",
                "--enable-dp-attention",
                "--moe-a2a-backend",
                "deepep",
                "--speculative-algorithm",
                "EAGLE",
                "--speculative-num-steps",
                "1",
                "--speculative-eagle-topk",
                "1",
                "--speculative-num-draft-tokens",
                "2",
                "--enable-dsa-prefill-context-parallel",
                "--dsa-prefill-cp-mode",
                "round-robin-split",
                "--deepep-config",
                DEEPEP_CONFIG,
            ],
            env=_DEEPEP_ENV,
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)


class TestDSV4FlashFP4B200Balanced_CP_NonDeepEP(
    BasicDecodeCorrectnessMixin,
    GSM8KMixin,
    CustomTestCase,
):
    """Balanced recipe: TP=4, DP=4, EAGLE (1-step spec)."""

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
                "--attn-cp-size",
                "4",
                "--speculative-algorithm",
                "EAGLE",
                "--speculative-num-steps",
                "1",
                "--speculative-eagle-topk",
                "1",
                "--speculative-num-draft-tokens",
                "2",
                "--enable-dsa-prefill-context-parallel",
                "--dsa-prefill-cp-mode",
                "round-robin-split",
                "--moe-runner-backend",  # for fp4 checkpoint
                "flashinfer_mxfp4",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)


if __name__ == "__main__":
    unittest.main()
