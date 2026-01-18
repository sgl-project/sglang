"""
Qwen3 Next piecewise CUDA graph tests.

DISABLED: See https://github.com/sgl-project/sglang/issues/17039
PCG tests for Qwen3 Next have intermittent failures (5-10% probability).
Investigation ongoing by @YuweiAn.
"""

import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.few_shot_gsm8k import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

QWEN3_NEXT_MODEL = "Qwen/Qwen3-Next-80B-A3B-Instruct"

ACC_THRESHOLDS = {
    QWEN3_NEXT_MODEL: {"kl_div": 0.0025, "gsm8k": 0.93},
}


@unittest.skip("Disabled: intermittent failures, see #17039")
class TestQwen3NextPiecewiseCudaGraph(CustomTestCase):

    @classmethod
    def setUpClass(cls):
        cls.model = QWEN3_NEXT_MODEL
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--tp",
                "4",
                "--enable-piecewise-cuda-graph",
                "--piecewise-cuda-graph-compiler",
                "eager",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=200,
            max_new_tokens=512,
            parallel=128,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )
        metrics = run_eval(args)
        print(f"{metrics=}")
        self.assertGreaterEqual(
            metrics["accuracy"], ACC_THRESHOLDS[self.model]["gsm8k"]
        )


if __name__ == "__main__":
    unittest.main()
