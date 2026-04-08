import unittest

import torch

from sglang import Engine
from sglang.lang.chat_template import get_chat_template_by_model_path
from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_IMAGE_URL,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    SimpleNamespace,
    popen_launch_server,
)

# CI Registration
register_cuda_ci(est_time=220, suite="stage-b-test-1-gpu-large")


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

        self.assertGreaterEqual(metrics["score"], 0.82)


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
            enforce_piecewise_cuda_graph=True,
        )
        out = engine.encode([text], image_data=[DEFAULT_IMAGE_URL])[0]["embedding"]
        engine.shutdown()
        self.assertGreater(len(out), 0)

        engine = Engine(
            model_path=model_path,
            enable_multimodal=True,
            is_embedding=True,
            disable_piecewise_cuda_graph=True,
        )
        out_without_pcg = engine.encode([text], image_data=[DEFAULT_IMAGE_URL])[0][
            "embedding"
        ]
        engine.shutdown()
        self.assertGreater(len(out_without_pcg), 0)

        t_out = torch.tensor(out)
        t_out_without_pcg = torch.tensor(out_without_pcg)
        max_abs_diff = (t_out - t_out_without_pcg).abs().max().item()
        max_rel_diff = (
            ((t_out - t_out_without_pcg).abs() / (t_out_without_pcg.abs() + 1e-8))
            .max()
            .item()
        )
        print(
            f"PCG embedding diff: max_abs={max_abs_diff:.6f}, max_rel={max_rel_diff:.6f}"
        )
        self.assertTrue(
            torch.allclose(
                t_out,
                t_out_without_pcg,
                atol=1e-2,
                rtol=1e-2,
            ),
            f"Piecewise CUDA graph embedding mismatch: max_abs_diff={max_abs_diff}, max_rel_diff={max_rel_diff}",
        )


if __name__ == "__main__":
    unittest.main()
