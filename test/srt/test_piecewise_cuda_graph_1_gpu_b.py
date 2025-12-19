import unittest

import torch

from sglang import Engine
from sglang.lang.chat_template import get_chat_template_by_model_path
from sglang.srt.utils import kill_process_tree
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_IMAGE_URL,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    SimpleNamespace,
    popen_launch_server,
)


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
