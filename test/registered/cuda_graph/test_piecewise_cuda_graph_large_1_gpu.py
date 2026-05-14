import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    SimpleNamespace,
    popen_launch_server,
)

# CI Registration - Large 1-GPU tests (80GB GPU required)
register_cuda_ci(est_time=480, suite="stage-b-test-large-1-gpu")


class TestPiecewiseCudaGraphQwen3MoE(CustomTestCase):
    """Test piecewise CUDA graph with Qwen3-Coder-30B-A3B-Instruct MoE model"""

    @classmethod
    def setUpClass(cls):
        cls.model = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--enable-piecewise-cuda-graph",
                "--piecewise-cuda-graph-compiler",
                "eager",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k_accuracy(self):
        """Test GSM8K accuracy with 8-shot setting"""
        num_examples = 2000

        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="mgsm_en",
            num_examples=num_examples,
            num_threads=min(num_examples, 1024),
        )

        metrics = run_eval(args)
        print(f"GSM8K Accuracy: {metrics['score']:.3f}")

        self.assertGreaterEqual(metrics["score"], 0.90)


class TestPiecewiseCudaGraphGPTQ(CustomTestCase):

    @classmethod
    def setUpClass(cls):
        cls.model = "Qwen/Qwen3-30B-A3B-GPTQ-Int4"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=["--enable-piecewise-cuda-graph"],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_mgsm_accuracy(self):
        num_examples = 1319

        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="mgsm_en",
            num_examples=num_examples,
            num_threads=min(num_examples, 1024),
        )

        metrics = run_eval(args)
        print(f"MGSM Accuracy: {metrics['score']:.3f}")

        # Expected accuracy: 0.948, allow some variance
        self.assertGreaterEqual(metrics["score"], 0.92)


class TestPiecewiseCudaGraphAWQ(CustomTestCase):
    """Test piecewise CUDA graph with AWQ quantized model"""

    @classmethod
    def setUpClass(cls):
        cls.model = "Qwen/QwQ-32B-AWQ"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=["--enable-piecewise-cuda-graph"],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_mgsm_accuracy(self):
        """Test MGSM accuracy with AWQ model"""
        num_examples = 1319

        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="mgsm_en",
            num_examples=num_examples,
            num_threads=min(num_examples, 1024),
        )

        metrics = run_eval(args)
        print(f"MGSM Accuracy: {metrics['score']:.3f}")
        print(f"Output throughput: {metrics.get('throughput', 'N/A')} token/s")

        # Expected accuracy: 0.680, allow some variance
        self.assertGreaterEqual(metrics["score"], 0.65)


if __name__ == "__main__":
    unittest.main()
