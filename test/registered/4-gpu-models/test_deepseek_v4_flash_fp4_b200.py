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

register_cuda_ci(est_time=900, suite="stage-c-test-dsv4-4-gpu-b200")

MODEL = "deepseek-ai/DeepSeek-V4-Flash"
SERVER_LAUNCH_TIMEOUT = 3600


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
        print(f"[DSV4 Flash FP4 B200] GSM8K {metrics=}")
        self.assertGreater(metrics["score"], 0.93)


if __name__ == "__main__":
    unittest.main()
