"""H200 per-commit CI: DeepSeek-V4-Flash FP8 (LowLatency recipe).

Launches TP=4 with DeepEP a2a backend + EAGLE speculative decoding,
with FP4 experts disabled via SGLANG_DSV4_FP4_EXPERTS=0.
Runs 12 ServerSanity probes (correctness, streaming, concurrency, determinism)
plus a GSM8K accuracy gate.

Registry: stage-c-test-dsv4-8-gpu-h200 (per-commit, 8x H200 — only 4 used by TP=4)
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

register_cuda_ci(est_time=900, suite="stage-c-test-dsv4-8-gpu-h200")

MODEL_FP8 = "sgl-project/DeepSeek-V4-Flash-FP8"
SERVER_LAUNCH_TIMEOUT = 3600
DEEPEP_CONFIG = '{"normal_dispatch":{"num_sms":96},"normal_combine":{"num_sms":96}}'


class TestDSV4FlashFP8H200(ServerSanityMixin, CustomTestCase):
    """LowLatency recipe: TP=4, Marlin FP4, EAGLE spec decoding."""

    @classmethod
    def setUpClass(cls):
        cls.model = try_cached_model(MODEL_FP8)
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
                "--cuda-graph-max-bs",
                "128",
                "--max-running-requests",
                "128",
                "--deepep-config",
                DEEPEP_CONFIG,
            ],
            env={
                "SGLANG_DSV4_FP4_EXPERTS": "0",
                "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "256",
            },
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="gsm8k",
            api="completion",
            max_tokens=512,
            num_examples=200,
            num_threads=128,
        )
        metrics = run_eval(args)
        print(f"[DSV4 Flash FP4 Marlin H200] GSM8K {metrics=}")
        self.assertGreater(metrics["score"], 0.93)


if __name__ == "__main__":
    unittest.main()
