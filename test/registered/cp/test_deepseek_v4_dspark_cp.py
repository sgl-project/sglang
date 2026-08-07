"""E2E: DeepSeek-V4-Flash DSPARK speculation together with DSA prefill CP.

Covers the interaction between DSpark aux hidden-state capture and the
round-robin-split DSA prefill CP path: the model must gather each aux
tensor on the same CP token split as hidden_states, and DSpark's
markov_w2 TP-shard must pick the lm_head shard group (which is the full
TP group when attn_tp_group degenerates to size 1 under CP).

Registry: extra-b-test-4-gpu-b200 (label-gated extra CI, 4x B200).
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

register_cuda_ci(est_time=260, stage="extra-b", runner_config="deepep-4-gpu-b200")

# The Flash-DSpark checkpoint bundles the DSpark draft head with the target,
# so --speculative-draft-model-path is not required.
MODEL = "deepseek-ai/DeepSeek-V4-Flash-DSpark"
SERVER_LAUNCH_TIMEOUT = 3600


class TestDSV4FlashDSparkCP(
    BasicDecodeCorrectnessMixin,
    GSM8KMixin,
    CustomTestCase,
):
    """TP=4 with DSPARK + DSA prefill CP (round-robin-split, attn_cp=tp)."""

    gsm8k_accuracy_thres = 0.90

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
                "DSPARK",
                "--enable-dsa-prefill-context-parallel",
                "--dsa-prefill-cp-mode",
                "round-robin-split",
                "--moe-runner-backend",
                "flashinfer_mxfp4",
                "--mem-fraction-static",
                "0.78",
                "--page-size",
                "256",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)


if __name__ == "__main__":
    unittest.main()
