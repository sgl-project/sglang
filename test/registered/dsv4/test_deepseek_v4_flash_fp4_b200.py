"""B200 per-commit CI: DeepSeek-V4-Flash FP4 (LowLatency recipe).

Launches TP=4 with flashinfer_mxfp4 MoE runner + EAGLE speculative decoding.
Runs 12 ServerSanity probes (correctness, streaming, concurrency, determinism)
plus a GSM8K accuracy gate.

Registry: stage-c-test-dsv4-4-gpu-b200 (per-commit, 4x B200)
"""

import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.server_sanity_kit import ServerSanityMixin
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    try_cached_model,
)

register_cuda_ci(est_time=1800, stage="stage-c", runner_config="dsv4-4-gpu-b200")

MODEL = "deepseek-ai/DeepSeek-V4-Flash"
SERVER_LAUNCH_TIMEOUT = 3600
DEEPEP_CONFIG = '{"normal_dispatch":{"num_sms":96},"normal_combine":{"num_sms":96}}'

_DEEPEP_ENV = {
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "1024",
}

_MEGAMOE_ENV = {
    "SGLANG_OPT_USE_DEEPGEMM_MEGA_MOE": "1",
    "SGLANG_OPT_FIX_MEGA_MOE_MEMORY": "1",
    "SGLANG_OPT_FIX_NEXTN_MEGA_MOE": "1",
    "SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK": "4096",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "0",
}


def _gsm8k_check(test_case):
    args = SimpleNamespace(
        base_url=test_case.base_url,
        model=test_case.model,
        eval_name="gsm8k",
        api="completion",
        max_tokens=512,
        num_examples=200,
        num_threads=128,
    )
    metrics = run_eval(args)
    print(f"[{type(test_case).__name__}] GSM8K {metrics=}")
    test_case.assertGreater(metrics["score"], 0.93)


class TestDSV4FlashFP4B200(ServerSanityMixin, CustomTestCase):
    """LowLatency recipe: TP=4, FP4 (mxfp4), EAGLE spec decoding."""

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
                "--moe-runner-backend",
                "flashinfer_mxfp4",
                "--speculative-algorithm",
                "EAGLE",
                "--speculative-num-steps",
                "3",
                "--speculative-eagle-topk",
                "1",
                "--speculative-num-draft-tokens",
                "4",
                "--chunked-prefill-size",
                "4096",
                "--disable-flashinfer-autotune",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        _gsm8k_check(self)


class TestDSV4FlashFP4B200Balanced(ServerSanityMixin, CustomTestCase):
    """Balanced recipe: TP=4, DP=4, DeepEP, EAGLE (1-step spec)."""

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
                "--dp",
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
                "--deepep-config",
                DEEPEP_CONFIG,
            ],
            env=_DEEPEP_ENV,
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        _gsm8k_check(self)


class TestDSV4FlashFP4B200MegaMoE(ServerSanityMixin, CustomTestCase):
    """Balanced recipe: TP=4, DP=4, MegaMoE."""

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
                "--dp",
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
            ],
            env=_MEGAMOE_ENV,
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        _gsm8k_check(self)


if __name__ == "__main__":
    unittest.main()
