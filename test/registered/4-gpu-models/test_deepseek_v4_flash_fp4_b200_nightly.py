"""B200 nightly CI: DeepSeek-V4-Flash FP4 (Balanced + MaxThroughput recipes).

Two server configurations exercise the DeepEP all-to-all + DP-attention path
that the per-commit LowLatency test does not cover.

  Balanced:       TP=4, DP=4, DeepEP, EAGLE (1 step)
  MaxThroughput:  TP=4, DP=4, DeepEP, no speculation

Each class inherits 12 ServerSanity probes plus a GSM8K accuracy gate.

Registry: nightly-4-gpu-b200
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

register_cuda_ci(est_time=3600, suite="nightly-4-gpu-b200", nightly=True)

MODEL = "deepseek-ai/DeepSeek-V4-Flash"
SERVER_LAUNCH_TIMEOUT = 3600
DEEPEP_CONFIG = '{"normal_dispatch":{"num_sms":96},"normal_combine":{"num_sms":96}}'

_DEEPEP_ENV = {
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "1024",
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


class TestDSV4FlashFP4B200MaxThroughput(ServerSanityMixin, CustomTestCase):
    """MaxThroughput recipe: TP=4, DP=4, DeepEP, no speculation."""

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


if __name__ == "__main__":
    unittest.main()
