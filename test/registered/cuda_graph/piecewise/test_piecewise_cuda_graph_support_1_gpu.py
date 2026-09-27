import unittest

import torch
from transformers import AutoProcessor

from sglang import Engine
from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_IMAGE_URL,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    SimpleNamespace,
    build_vlm_image_prompt,
    is_in_amd_ci,
    popen_launch_server,
)

# CI Registration
register_cuda_ci(est_time=250, stage="nightly", runner_config="1-gpu-large")
register_amd_ci(est_time=180, suite="stage-b-test-1-gpu-large-amd")


# The 192GB mi300x runners have less headroom than the 256GB mi325x ones they
# replaced: the auto-derived fraction left too little room for the ViT
# activations plus the piecewise graph private pools, and the server died under
# the 1024-thread gsm8k load.
AMD_MEM_FRACTION_STATIC = 0.6


class TestPiecewiseCudaGraphQwen25VL(CustomTestCase):
    """Test piecewise CUDA graph with Qwen2.5-VL-7B-Instruct model"""

    @classmethod
    def setUpClass(cls):
        cls.model = "Qwen/Qwen2.5-VL-7B-Instruct"
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = [
            "--cuda-graph-backend-prefill=tc_piecewise",
            "--disable-radix-cache",
        ]
        if is_in_amd_ci():
            other_args += ["--mem-fraction-static", str(AMD_MEM_FRACTION_STATIC)]
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
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

        self.assertGreaterEqual(metrics["score"], 0.80)


class TestPiecewiseCudaGraphQwen25VLEmbedding(CustomTestCase):
    """Test piecewise CUDA graph with Qwen2.5-VL-3B-Instruct embedding model"""

    def test_embedding(self):
        model_path = "Qwen/Qwen2.5-VL-3B-Instruct"
        text = build_vlm_image_prompt(
            AutoProcessor.from_pretrained(model_path), "What is in this picture?"
        )
        extra_args = (
            {"mem_fraction_static": AMD_MEM_FRACTION_STATIC} if is_in_amd_ci() else {}
        )

        engine = Engine(
            model_path=model_path,
            enable_multimodal=True,
            is_embedding=True,
            cuda_graph_backend_prefill="tc_piecewise",
            **extra_args,
        )
        out = engine.encode([text], image_data=[DEFAULT_IMAGE_URL])[0]["embedding"]
        engine.shutdown()
        self.assertGreater(len(out), 0)

        engine = Engine(
            model_path=model_path,
            enable_multimodal=True,
            is_embedding=True,
            cuda_graph_backend_prefill="disabled",
            **extra_args,
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
