import unittest

from sglang.srt.utils import get_device_count, kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    SimpleNamespace,
    popen_launch_server,
)

# CI Registration - 2-GPU tests (80GB GPUs required)
register_cuda_ci(est_time=160, suite="stage-b-test-large-2-gpu")


class TestPiecewiseCudaGraphFusedMoE(CustomTestCase):
    """Test piecewise CUDA graph with FusedMoE Backend"""

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
                "--tp",
                "2",
                "--ep-size",
                "2",
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


@unittest.skipIf(
    get_device_count() < 2,
    f"Test requires at least 2 GPUs, but only {get_device_count()} GPU(s) available",
)
class TestPiecewiseCudaGraphWithPP(CustomTestCase):
    """Test piecewise CUDA graph with Pipeline Parallelism (PP) support"""

    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--enable-piecewise-cuda-graph",
                "--pp-size",
                "2",
                "--tp-size",
                "1",
                "--chunked-prefill-size",
                "256",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k_accuracy(self):
        """Test GSM8K accuracy with PP and piecewise CUDA graph"""
        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=200,
            max_new_tokens=512,
            parallel=128,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )
        metrics = run_eval_few_shot_gsm8k(args)
        print(f"GSM8K Accuracy with PP+Piecewise CUDA Graph: {metrics['accuracy']:.3f}")
        self.assertGreater(metrics["accuracy"], 0.74)


if __name__ == "__main__":
    unittest.main()
