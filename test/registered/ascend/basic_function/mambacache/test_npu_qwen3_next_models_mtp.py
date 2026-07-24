"""
NPU regression test for Qwen3-Next-80B-A3B MambaCache + MTP (Multi-Token Prediction).

This test validates two MTP (speculative decoding) modes on NPU for the
Qwen3-Next-80B-A3B-Instruct model (Mamba + MoE hybrid architecture):

1. TestQwen3NextMTPTopk (TP=8):
   - NEXTN speculative decoding with topk=1, 3 steps, 4 draft tokens
   - MambaCache with extra_buffer scheduler strategy
   - Verifies: GSM8K accuracy >= 0.93 AND prefix cache branching KL divergence < 0.008
   - Inherits PrefixCacheBranchingMixin to test that speculative decoding does not
     break prefix cache consistency across branching requests

2. TestQwen3NextMTPV2 (TP=4):
   - NEXTN speculative decoding V2 (SGLANG_ENABLE_SPEC_V2=1)
   - Lower KL divergence threshold (0.0035) for stricter accuracy
   - Verifies: GSM8K accuracy >= 0.93
   - Overrides setUpClass to inject SGLANG_ENABLE_SPEC_V2=1 env var

NPU-specific configurations:
  - attention_backend="ascend"
  - mamba_ssm_dtype="bfloat16" (Mamba state dtype)
  - mamba_scheduler_strategy="extra_buffer" (required for Mamba + MTP on NPU)

Usage:
    python -m unittest test_npu_qwen3_next_models_mtp
"""

import unittest

from sglang.test.ascend.test_ascend_utils import (
    QWEN3_NEXT_80B_A3B_INSTRUCT_WEIGHTS_FOR_TEST,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.kits.prefix_cache_branching_kit import PrefixCacheBranchingMixin
from sglang.test.server_fixtures.default_fixture import (
    DefaultServerBase,
    openai_api_env,
)
from sglang.test.test_utils import popen_launch_server

register_npu_ci(est_time=600, suite="full-8-npu-a3", nightly=True)

QWEN3_NEXT_MODEL = QWEN3_NEXT_80B_A3B_INSTRUCT_WEIGHTS_FOR_TEST.model_path


class TestQwen3NextMTPTopk(GSM8KMixin, PrefixCacheBranchingMixin, DefaultServerBase):
    """MTP speculative decoding with topk=1 (TP=8).

    Validates that NEXTN speculative decoding with topk=1 and MambaCache
    extra_buffer strategy produces accurate results and does not break
    prefix cache consistency under branching requests.
    """

    model = QWEN3_NEXT_MODEL
    cache_chunk_size = 64
    gsm8k_accuracy_thres = 0.93
    kl_div_thres = 0.008
    other_args = [
        "--trust-remote-code",
        "--speculative-algorithm",
        "NEXTN",
        "--speculative-num-steps",
        "3",
        "--speculative-eagle-topk",
        "1",
        "--speculative-num-draft-tokens",
        "4",
        "--mem-fraction-static",
        "0.7",
        "--tp",
        "8",
        "--chunked-prefill-size",
        "2048",
        "--mamba-scheduler-strategy",
        "extra_buffer",
        "--mamba-track-interval",
        "128",
        "--attention-backend",
        "ascend",
        "--mamba-ssm-dtype",
        "bfloat16",
    ]


class TestQwen3NextMTPV2(GSM8KMixin, DefaultServerBase):
    """MTP speculative decoding V2 (TP=4).

    Validates NEXTN speculative decoding V2 with stricter accuracy
    requirements (KL threshold 0.0035). Enables SGLANG_ENABLE_SPEC_V2=1
    to use the V2 speculative decoding path.
    """

    model = QWEN3_NEXT_MODEL
    gsm8k_accuracy_thres = 0.93
    kl_div_thres = 0.0035
    other_args = [
        "--trust-remote-code",
        "--speculative-algorithm",
        "NEXTN",
        "--speculative-num-steps",
        "3",
        "--speculative-eagle-topk",
        "1",
        "--speculative-num-draft-tokens",
        "4",
        "--mem-fraction-static",
        "0.7",
        "--tp",
        "4",
        "--chunked-prefill-size",
        "2048",
        "--mamba-scheduler-strategy",
        "extra_buffer",
        "--mamba-track-interval",
        "128",
        "--attention-backend",
        "ascend",
        "--mamba-ssm-dtype",
        "bfloat16",
    ]

    @classmethod
    def setUpClass(cls):
        assert cls.model is not None, "Please set cls.model in subclass"
        with openai_api_env(cls.api_key):
            cls.process = popen_launch_server(
                cls.model,
                cls.base_url,
                timeout=cls.timeout,
                other_args=cls.other_args,
                env={
                    "SGLANG_ENABLE_SPEC_V2": "1",
                },
            )


if __name__ == "__main__":
    unittest.main()
