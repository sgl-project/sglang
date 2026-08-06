"""B200 nightly: DeepSeek-V4-Pro AIME25 accuracy with reasoning (Think Max).

Two recipes:
  1. Low-Latency: TP=8, FP4 MoE (flashinfer_mxfp4), EAGLE
  2. MegaMoE: TP=8, DP=8, dp-attention, DeepEP + MegaMoE, EAGLE (1-step)

Both run AIME25 in thinking mode with reasoning_effort="max".

Registry: nightly-8-gpu-b200 (nightly, 8x B200)
"""

import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    try_cached_model,
)

register_cuda_ci(est_time=7200, suite="nightly-8-gpu-b200", nightly=True)

MODEL = "deepseek-ai/DeepSeek-V4-Pro"
SERVER_LAUNCH_TIMEOUT = 3600

_MEGAMOE_ENV = {
    "SGLANG_OPT_USE_DEEPGEMM_MEGA_MOE": "1",
    "SGLANG_OPT_FIX_MEGA_MOE_MEMORY": "1",
    "SGLANG_OPT_FIX_NEXTN_MEGA_MOE": "1",
    "SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK": "4096",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "0",
}

DEEPEP_CONFIG = '{"normal_dispatch":{"num_sms":96},"normal_combine":{"num_sms":96}}'


def _run_aime25(test_case):
    """AIME25 with Think Max reasoning (repeat=4, majority vote)."""
    args = SimpleNamespace(
        base_url=test_case.base_url,
        model=test_case.model,
        eval_name="aime25",
        num_examples=None,  # full dataset
        num_threads=128,
        max_tokens=400000,
        temperature=1.0,
        top_p=1.0,
        thinking_mode="deepseek-v3",
        reasoning_effort="max",
        repeat=4,
    )
    metrics = run_eval(args)
    print(f"[{type(test_case).__name__}] AIME25 {metrics=}")
    test_case.assertGreater(metrics["mean_score"], 0.95)


class TestDSV4ProAIME25LowLatency(CustomTestCase):
    """DeepSeek-V4-Pro Low-Latency + Think Max on AIME25."""

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
                "--mem-fraction-static",
                "0.88",
                "--reasoning-parser",
                "deepseek-v4",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)

    def test_aime25(self):
        _run_aime25(self)


class TestDSV4ProAIME25MegaMoE(CustomTestCase):
    """DeepSeek-V4-Pro MegaMoE + Think Max on AIME25."""

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
                "--dp",
                "8",
                "--enable-dp-attention",
                "--moe-a2a-backend",
                "deepep",
                "--deepep-config",
                DEEPEP_CONFIG,
                "--speculative-algorithm",
                "EAGLE",
                "--speculative-num-steps",
                "1",
                "--speculative-eagle-topk",
                "1",
                "--speculative-num-draft-tokens",
                "2",
                "--mem-fraction-static",
                "0.82",
                "--cuda-graph-max-bs",
                "64",
                "--max-running-requests",
                "128",
                "--reasoning-parser",
                "deepseek-v4",
            ],
            env=_MEGAMOE_ENV,
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)

    def test_aime25(self):
        _run_aime25(self)


if __name__ == "__main__":
    unittest.main()
