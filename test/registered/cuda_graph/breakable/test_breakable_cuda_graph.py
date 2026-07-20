"""Tests for the breakable CUDA graph (BCG) runner.

TestBreakableCudaGraph: integration test — spin up Qwen3-8B with
--cuda-graph-backend-prefill=breakable and check mgsm_en accuracy.

The capture/replay mechanism itself now lives in the upstream
piecewise_cuda_graphs library (CUDAGraphSequence / piecewise_graph / no_graph);
its internals are covered by that library's own tests, not here.
"""

import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    SimpleNamespace,
    popen_launch_server,
)

# CI Registration — large suite to fit the integration test's server startup.
register_cuda_ci(est_time=79, stage="base-b", runner_config="1-gpu-large")
register_amd_ci(est_time=200, suite="stage-c-test-large-8-gpu-amd-mi35x")


class TestBreakableCudaGraph(CustomTestCase):
    """Integration: Qwen3-8B with --enable-breakable-cuda-graph on mgsm_en."""

    @classmethod
    def setUpClass(cls):
        cls.model = "Qwen/Qwen3-8B"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--cuda-graph-backend-prefill=breakable",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k_accuracy(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="mgsm_en",
            num_examples=1319,
            num_threads=1024,
        )

        metrics = run_eval(args)
        score = metrics["score"]
        print(f"mgsm_en accuracy with breakable CUDA graph: {score:.3f}")

        self.assertGreaterEqual(score, 0.80)


if __name__ == "__main__":
    unittest.main()
