"""Archived test classes split out of test/registered/piecewise_cuda_graph/test_piecewise_cuda_graph_support_1_gpu.py.

Originally registered with `register_cuda_ci(...)`. Moved here as part of
the per-commit pruning effort to keep the code reachable manually.
Run with `python3 test/manual/piecewise_cuda_graph/test_piecewise_cuda_graph_support_1_gpu_archived.py`.
"""

import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    SimpleNamespace,
    popen_launch_server,
)


# CI Registration
class TestPiecewiseCudaGraphInternVL25(CustomTestCase):
    """Test piecewise CUDA graph with InternVL2.5-8B model"""

    @classmethod
    def setUpClass(cls):
        cls.model = "OpenGVLab/InternVL2_5-8B"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--enforce-piecewise-cuda-graph",
                "--disable-radix-cache",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k_accuracy(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="gsm8k",
            num_examples=None,
            num_threads=1024,
        )

        metrics = run_eval(args)
        print(f"GSM8K Accuracy: {metrics['score']:.3f}")

        # Baseline (no piecewise CUDA graph): 0.571 — this eval uses 5-shot
        # concatenated text via chat API, which scores lower than reported
        # benchmarks (~77.8%) that use proper CoT chat format. The threshold
        # is set 5% below observed to catch catastrophic regressions.
        self.assertGreaterEqual(metrics["score"], 0.54)


if __name__ == "__main__":
    unittest.main()
