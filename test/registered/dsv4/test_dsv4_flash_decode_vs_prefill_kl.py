"""Decode-vs-prefill KL match across parallelism configs (DSv4-Flash-FP8).

For each config, generate ``Q`` tokens by greedy-decode after a 512-token
prefix ``P``, then re-prefill the full ``P + Q`` and compare logprobs at the
same Q positions. Cache-write and decode-vs-prefill attention kernels should
agree modulo small FP noise (KL ≈ 0.005-0.006).

Three configs exercise the same model under different attention parallelism
schemes (TP, DP, CP). The DP and CP paths skip post-attention allreduce, so
any per-rank non-determinism in K-cache writes that TP-allreduce would mask
becomes visible here.
"""

import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kl_multiturn_utils import (
    get_input_ids,
    test_input_output_logprobs_match_helper,
)
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=900, stage="stage-c", runner_config="dsv4-8-gpu-h200")

MODEL = "sgl-project/DeepSeek-V4-Flash-FP8"
LAUNCH_TIMEOUT = 3600
DEEPEP_CONFIG = '{"normal_dispatch":{"num_sms":96},"normal_combine":{"num_sms":96}}'

PREFIX_LEN = 512
DECODE_LEN = 512
NUM_SAMPLES = 4


class _DecodeVsPrefillKLMixin:
    model = MODEL
    kl_threshold: float = 0.015

    def test_decode_vs_prefill_logprobs_match(self):
        raw_ids = get_input_ids(self.model, num_samples=NUM_SAMPLES * 4)
        prompts = [ids[:PREFIX_LEN] for ids in raw_ids if len(ids) >= PREFIX_LEN][
            :NUM_SAMPLES
        ]
        assert len(prompts) == NUM_SAMPLES
        test_input_output_logprobs_match_helper(
            self.base_url,
            self.model,
            self.kl_threshold,
            prompts,
            label=f"{self.__class__.__name__}.decode_vs_prefill",
            max_new_tokens=DECODE_LEN,
            sampling_temperature=0,
        )


class TestDSV4FlashDecodeVsPrefillKL_TP4(_DecodeVsPrefillKLMixin, CustomTestCase):
    """TP=4, no DP-attention, no CP. Baseline: TP allreduce after attention
    smooths per-rank FP noise across ranks."""

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=LAUNCH_TIMEOUT,
            other_args=[
                "--trust-remote-code",
                "--tp",
                "4",
                "--chunked-prefill-size",
                "16384",
                "--mem-fraction-static",
                "0.85",
                "--max-running-requests",
                "4",
            ],
            env={
                "SGLANG_DSV4_FP4_EXPERTS": "0",
                "SGLANG_ENABLE_UNIFIED_RADIX_TREE": "1",
            },
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)


class TestDSV4FlashDecodeVsPrefillKL_DP4EP4(_DecodeVsPrefillKLMixin, CustomTestCase):
    """TP=4 + DP=4 + EP=4 + DeepEP, no CP. DP-attention path: each rank owns
    a request slice, no post-attention allreduce, so per-rank K-write noise is
    not averaged out."""

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=LAUNCH_TIMEOUT,
            other_args=[
                "--trust-remote-code",
                "--tp",
                "4",
                "--dp",
                "4",
                "--enable-dp-attention",
                "--moe-a2a-backend",
                "deepep",
                "--deepep-config",
                DEEPEP_CONFIG,
                "--chunked-prefill-size",
                "8192",
                "--mem-fraction-static",
                "0.6",
                "--max-running-requests",
                "4",
                "--cuda-graph-max-bs",
                "8",
            ],
            env={
                "SGLANG_DSV4_FP4_EXPERTS": "0",
                "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "1024",
                "SGLANG_ENABLE_UNIFIED_RADIX_TREE": "1",
            },
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)


class TestDSV4FlashDecodeVsPrefillKL_CP4(_DecodeVsPrefillKLMixin, CustomTestCase):
    """TP=4 + NSA prefill context-parallel (round-robin). validate_deepseek_v4_cp
    forces enable_dp_attention=True and attn_cp_size=4. Prefill takes the CP
    path (gather raw bf16 across CP ranks + fused JIT write); decode takes the
    normal multi-stream path. Both must produce K-cache bit-equivalent to the
    eager / re-prefill reference."""

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=LAUNCH_TIMEOUT,
            other_args=[
                "--trust-remote-code",
                "--tp",
                "4",
                "--moe-a2a-backend",
                "deepep",
                "--deepep-config",
                DEEPEP_CONFIG,
                "--enable-nsa-prefill-context-parallel",
                "--nsa-prefill-cp-mode",
                "round-robin-split",
                "--chunked-prefill-size",
                "16384",
                "--mem-fraction-static",
                "0.78",
                "--swa-full-tokens-ratio",
                "0.25",
                "--max-total-tokens",
                "20000",
                "--max-running-requests",
                "4",
            ],
            env={
                "SGLANG_DSV4_FP4_EXPERTS": "0",
                "SGLANG_OPT_USE_JIT_INDEXER_METADATA": "1",
                "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "1024",
                "SGLANG_ENABLE_UNIFIED_RADIX_TREE": "1",
                "SGLANG_OPT_USE_FUSED_STORE_CACHE": "1",
            },
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)


if __name__ == "__main__":
    unittest.main()
