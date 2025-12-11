import unittest

import torch

from sglang import Engine
from sglang.lang.chat_template import get_chat_template_by_model_path
from sglang.srt.utils import get_device_sm, kill_process_tree
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_IMAGE_URL,
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_MODEL_NAME_FOR_TEST_MLA,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    SimpleNamespace,
    popen_launch_server,
    run_bench_one_batch,
)


class TestPiecewiseCudaGraphCorrectness(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST
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

    def test_mmlu(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="mmlu",
            num_examples=64,
            num_threads=32,
        )

        metrics = run_eval(args)
        self.assertGreaterEqual(metrics["score"], 0.65)


class TestPiecewiseCudaGraphBenchmark(CustomTestCase):

    def test_latency(self):
        prefill_latency, _, _ = run_bench_one_batch(
            DEFAULT_MODEL_NAME_FOR_TEST,
            other_args=["--enable-piecewise-cuda-graph"],
        )
        self.assertLess(prefill_latency, 0.015)


@unittest.skipIf(get_device_sm() < 100, "Test requires CUDA SM 100 or higher")
class TestPiecewiseCudaGraphLlama31FP4(CustomTestCase):
    """MGSM test: piecewise CUDA graph with NVFP4 Llama3.1 8B on Blackwell."""

    @classmethod
    def setUpClass(cls):
        cls.model = "nvidia/Llama-3.1-8B-Instruct-FP4"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--enable-piecewise-cuda-graph",
                "--quantization",
                "modelopt_fp4",
                "--mem-fraction-static",
                "0.8",
            ],
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
        self.assertGreaterEqual(metrics["score"], 0.78)


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


class TestPiecewiseCudaGraphDeepSeek(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST_MLA
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--enable-piecewise-cuda-graph",
                "--piecewise-cuda-graph-compiler",
                "eager",
                "--piecewise-cuda-graph-max-tokens",
                "4096",  # should less than max_context_len
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
        metrics = run_eval_few_shot_gsm8k(args)
        print(metrics)

        self.assertGreater(metrics["accuracy"], 0.62)


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


class TestPiecewiseCudaGraphFP8(CustomTestCase):
    """Test piecewise CUDA graph with FP8 quantized model"""

    @classmethod
    def setUpClass(cls):
        cls.model = "nvidia/Llama-3.1-8B-Instruct-FP8"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--enable-piecewise-cuda-graph",
                "--quantization",
                "modelopt_fp8",
                "--kv-cache-dtype",
                "bfloat16",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_mgsm_accuracy(self):
        """Test MGSM accuracy with FP8 model"""
        num_examples = 1319
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="mgsm_en",
            num_examples=num_examples,
            num_threads=min(num_examples, 1024),
        )
        metrics = run_eval(args)
        self.assertGreaterEqual(metrics["score"], 0.85)
        print(f"MGSM Accuracy: {metrics['score']:.3f}")


class TestPiecewiseCudaGraphQwen25VL(CustomTestCase):
    """Test piecewise CUDA graph with Qwen2.5-VL-7B-Instruct model"""

    @classmethod
    def setUpClass(cls):
        cls.model = "Qwen/Qwen2.5-VL-7B-Instruct"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--enable-piecewise-cuda-graph",
                "--piecewise-cuda-graph-compiler",
                "eager",
                "--disable-radix-cache",
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

        self.assertGreaterEqual(metrics["score"], 0.70)


class TestPiecewiseCudaGraphInternVL25(CustomTestCase):
    """Test piecewise CUDA graph with InternVL2.5-8B-Instruct model"""

    @classmethod
    def setUpClass(cls):
        cls.model = "OpenGVLab/InternVL2_5-8B"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--enable-piecewise-cuda-graph",
                "--piecewise-cuda-graph-compiler",
                "eager",
                "--disable-radix-cache",
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

        self.assertGreaterEqual(metrics["score"], 0.70)


class TestPiecewiseCudaGraphQwen25VLEmbedding(CustomTestCase):
    """Test piecewise CUDA graph with Qwen2.5-VL-3B-Instruct embedding model"""

    def test_embedding(self):
        model_path = "Qwen/Qwen2.5-VL-3B-Instruct"
        chat_template = get_chat_template_by_model_path(model_path)
        text = f"{chat_template.image_token}What is in this picture? Answer: "

        engine = Engine(
            model_path=model_path,
            enable_multimodal=True,
            is_embedding=True,
            enable_piecewise_cuda_graph=True,
            piecewise_cuda_graph_compiler="eager",
        )
        out = engine.encode([text], image_data=[DEFAULT_IMAGE_URL])[0]["embedding"]
        engine.shutdown()
        self.assertGreater(len(out), 0)

        engine = Engine(
            model_path=model_path,
            enable_multimodal=True,
            is_embedding=True,
            enable_piecewise_cuda_graph=False,
        )
        out_without_pcg = engine.encode([text], image_data=[DEFAULT_IMAGE_URL])[0][
            "embedding"
        ]
        engine.shutdown()
        self.assertGreater(len(out_without_pcg), 0)

        self.assertTrue(
            torch.allclose(torch.tensor(out), torch.tensor(out_without_pcg))
        )


class TestPiecewiseCudaGraphQwen3OmniMOE(CustomTestCase):
    """Test piecewise CUDA graph with Qwen3-Omni-30B-A3B-Instruct  model"""

    @classmethod
    def setUpClass(cls):
        cls.model = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--enable-piecewise-cuda-graph",
                "--piecewise-cuda-graph-compiler",
                "eager",
                "--disable-radix-cache",
                "--tp=4",
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

        self.assertGreaterEqual(metrics["score"], 0.70)


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

        # Verify accuracy is reasonable (should be similar to non-PP case)
        self.assertGreater(metrics["accuracy"], 0.74)

    def test_basic_generation(self):
        """Test basic text generation with PP and piecewise CUDA graph"""
        import requests

        response = requests.post(
            f"{self.base_url}/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 16,
                },
            },
        )
        self.assertEqual(response.status_code, 200)
        response_json = response.json()
        self.assertIn("text", response_json)
        self.assertGreater(len(response_json["text"]), 0)


class TestPiecewiseCudaGraphPPConsistency(CustomTestCase):
    """Test consistency between PP with and without piecewise CUDA graph"""

    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST
        cls.base_url_pp_only = "http://127.0.0.1:23337"
        cls.base_url_pp_pcg = "http://127.0.0.1:23338"

    def run_gsm8k_test(self, base_url, enable_piecewise_cuda_graph=False):
        """Helper method to run GSM8K test with given configuration"""
        other_args = [
            "--pp-size",
            "2",
            "--tp-size",
            "1",
            "--chunked-prefill-size",
            "256",
        ]
        if enable_piecewise_cuda_graph:
            other_args.append("--enable-piecewise-cuda-graph")

        process = popen_launch_server(
            self.__class__.model,
            base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

        try:
            args = SimpleNamespace(
                num_shots=5,
                data_path=None,
                num_questions=100,  # Use fewer questions for consistency test
                max_new_tokens=256,
                parallel=64,
                host="http://127.0.0.1",
                port=int(base_url.split(":")[-1]),
            )
            metrics = run_eval_few_shot_gsm8k(args)
            import time

            time.sleep(2)  # Wait for cleanup
            return metrics
        finally:
            kill_process_tree(process.pid)

    def test_pp_consistency_with_piecewise_cuda_graph(self):
        """Test that PP with piecewise CUDA graph produces similar results to PP without it"""
        pp_only_metrics = self.run_gsm8k_test(
            self.base_url_pp_only, enable_piecewise_cuda_graph=False
        )
        pp_pcg_metrics = self.run_gsm8k_test(
            self.base_url_pp_pcg, enable_piecewise_cuda_graph=True
        )

        print(
            f"[PP Consistency Test] PP only: {pp_only_metrics} | PP+Piecewise CUDA Graph: {pp_pcg_metrics}"
        )

        # Both should have reasonable accuracy
        self.assertGreaterEqual(pp_only_metrics["accuracy"], 0.70)
        self.assertGreaterEqual(pp_pcg_metrics["accuracy"], 0.70)

        # Accuracy difference should be small (within 3%)
        accuracy_diff = abs(pp_only_metrics["accuracy"] - pp_pcg_metrics["accuracy"])
        self.assertLess(
            accuracy_diff,
            0.03,
            msg=(
                f"Accuracy difference too large: {accuracy_diff:.3f}. "
                f"PP only: {pp_only_metrics['accuracy']:.3f}, "
                f"PP+PCG: {pp_pcg_metrics['accuracy']:.3f}"
            ),
        )


if __name__ == "__main__":
    unittest.main()
