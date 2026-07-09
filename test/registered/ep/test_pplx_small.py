"""End-to-end accuracy tests for the ``pplx`` MoE all-to-all backend.

Besides the throughput GSM8K run, each server also runs a **single-stream**
GSM8K eval (``num_threads=1``). That is deliberate: the pplx EP combine already
reduces each token's expert outputs back to the source rank, so the model must
skip the post-experts all-reduce (see
``should_skip_post_experts_all_reduce`` / ``is_pplx()``). When it does not, the
extra all-reduce folds *idle* DP ranks' fabricated outputs into the real tokens
-- which only happens when fewer requests are in flight than ``dp_size``. The
high-concurrency run keeps every rank busy and hides that bug; the single-stream
run leaves ranks idle and is the regression guard for it.
"""

import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST_MLA,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=420, stage="base-c", runner_config="deepep-4-gpu-h100")


class TestPureDP(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST_MLA
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--tp",
                "4",
                "--enable-dp-attention",
                "--dp",
                "4",
                "--moe-a2a-backend",
                "pplx",
                "--deepep-mode",
                "low_latency",
                "--cuda-graph-max-bs-decode",
                "128",
                "--max-running-requests",
                "512",
                "--mem-fraction-static",
                "0.5",
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
        print(metrics)

        self.assertGreater(metrics["score"], 0.60)

    def test_gsm8k_single_stream(self):
        # Regression guard for the post-experts all-reduce double-count: with
        # num_threads=1 only one DP rank has a real request at a time, leaving
        # the others idle. If pplx does not skip the post-experts all-reduce,
        # those idle ranks' outputs corrupt the answer and the score collapses
        # (~0). Keep this serial + low example count so it stays cheap.
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="gsm8k",
            api="completion",
            max_tokens=512,
            num_examples=40,
            num_threads=1,
        )
        metrics = run_eval(args)
        print(metrics)

        self.assertGreater(metrics["score"], 0.50)


class TestHybridDPTP(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST_MLA
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--tp",
                "4",
                "--enable-dp-attention",
                "--dp",
                "2",
                "--moe-a2a-backend",
                "pplx",
                "--deepep-mode",
                "low_latency",
                "--cuda-graph-max-bs-decode",
                "128",
                "--max-running-requests",
                "256",
                "--mem-fraction-static",
                "0.5",
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
        print(metrics)

        self.assertGreater(metrics["score"], 0.60)


if __name__ == "__main__":
    unittest.main()
