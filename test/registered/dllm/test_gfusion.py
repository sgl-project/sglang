from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=500, stage="base-b", runner_config="1-gpu-large")

import unittest
from types import SimpleNamespace
from typing import List, Optional

from sglang.srt.utils import kill_process_tree
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


class TestGFusion(CustomTestCase):
    extra_args: Optional[List[str]] = None  # None marks the abstract base

    @classmethod
    def setUpClass(cls):
        if cls.extra_args is None:
            raise unittest.SkipTest("Skip the abstract base test class")

        cls.model = "ai-sage/GFusion-10B-A1.8B"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--attention-backend",
                "fa3",
                "--dllm-algorithm",
                "EBSampling",
                "--mem-fraction-static",
                "0.85",
                "--max-running-requests",
                "32",
                "--cuda-graph-bs",
                "1",
                "2",
                "4",
                "8",
                "16",
                "24",
                "32",
                *cls.extra_args,
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_mmlu(self):
        metrics = run_eval(
            SimpleNamespace(
                base_url=self.base_url,
                model=self.model,
                eval_name="mmlu",
                api="chat",
                max_tokens=512,
                num_examples=200,
                num_threads=64,
            )
        )
        print(f"{type(self).__name__} mmlu={metrics['score']:.4f}")
        self.assertGreater(metrics["score"], 0.65)


class TestGFusionPageSize1NoRadix(TestGFusion):
    extra_args = ["--page-size", "1", "--disable-radix-cache"]


class TestGFusionPageSizeBlockNoRadix(TestGFusion):
    extra_args = ["--page-size", "32", "--disable-radix-cache"]


class TestGFusionRadixCache(TestGFusion):
    extra_args = []


if __name__ == "__main__":
    unittest.main()
