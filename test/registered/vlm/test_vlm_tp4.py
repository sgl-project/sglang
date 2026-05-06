"""
VLM TP=4 per-commit test using Qwen3.5-27B with MMMU evaluation.
"""

import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=200, suite="nightly-4-gpu", nightly=True)

QWEN35_27B_MODEL = "Qwen/Qwen3.5-27B"
MMMU_ACCURACY_THRESHOLD = 0.65
MMMU_NUM_EXAMPLES = 32


class TestVLMTP4(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = QWEN35_27B_MODEL
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--tp-size",
                "4",
                "--cuda-graph-max-bs",
                "32",
                "--mem-fraction-static",
                "0.8",
                "--trust-remote-code",
                "--mamba-scheduler-strategy",
                "extra_buffer",
                "--mamba-track-interval",
                "128",
                "--mamba-ssm-dtype",
                "bfloat16",
                "--chunked-prefill-size",
                "2048",
                "--max-running-requests",
                "128",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)

    def test_mmmu_accuracy(self):
        args = SimpleNamespace(
            model=self.model,
            eval_name="mmmu",
            num_examples=MMMU_NUM_EXAMPLES,
            num_threads=16,
            max_tokens=2048,
            chat_template_kwargs={"enable_thinking": False},
            base_url=self.base_url,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )
        metrics = run_eval(args)
        print(f"MMMU score: {metrics['score']}")
        self.assertGreaterEqual(
            metrics["score"],
            MMMU_ACCURACY_THRESHOLD,
            f"MMMU accuracy {metrics['score']:.4f} below threshold {MMMU_ACCURACY_THRESHOLD}",
        )


if __name__ == "__main__":
    unittest.main()
