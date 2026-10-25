"""B200 extra CI: DeepSeek-V4-Flash FP4 with attn-CP.

Balanced recipe (TP=4, DeepEP, EAGLE) plus --attn-cp-size=4 with the
DSA prefill-CP interleave strategy. Split out of
e2e/models/test_deepseek_v4_flash_fp4_b200.py so the `cp` group covers
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

register_cuda_ci(est_time=689, stage="extra-b", runner_config="4-gpu-b200")

MODEL = "deepseek-ai/DeepSeek-V4-Flash"
SERVER_LAUNCH_TIMEOUT = 3600
DSPARK_MODEL = "deepseek-ai/DeepSeek-V4-Flash-DSpark"
DEEPEP_CONFIG = '{"normal_dispatch":{"num_sms":96},"normal_combine":{"num_sms":96}}'

_MEGAMOE_ENV = {
    "SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK": "8320",
}


class TestDSV4FlashFP4B200Balanced_CP_Megamoe(
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
                "--moe-a2a-backend",
                "megamoe",
                "--enable-w4a4-mxfp4-megamoe",
                "--speculative-algorithm",
                "EAGLE",
                "--speculative-num-steps",
                "1",
                "--speculative-eagle-topk",
                "1",
                "--speculative-num-draft-tokens",
                "2",
                "--enable-prefill-cp",
                "--cp-strategy",
                "interleave",
                "--deepep-config",
                DEEPEP_CONFIG,
            ],
            env=_MEGAMOE_ENV,
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)


class TestDSV4FlashFP4B200_CP_DSpark(
    BasicDecodeCorrectnessMixin,
    GSM8KMixin,
    CustomTestCase,
):
    """DSPARK speculation + prefill CP (interleave, CP, attn_cp=tp)."""

    gsm8k_accuracy_thres = 0.90

    @classmethod
    def setUpClass(cls):
        cls.model = try_cached_model(DSPARK_MODEL)
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
                "DSPARK",
                "--enable-prefill-cp",
                "--cp-strategy",
                "interleave",
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
