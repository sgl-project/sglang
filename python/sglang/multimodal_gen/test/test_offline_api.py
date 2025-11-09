# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

"""
    Testing the performance of generate command of sgl_diffusion' CLI
"""

import unittest

import torch

from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import DiffGenerator
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class TestGeneratorAPIBase(unittest.TestCase):
    # server args
    server_kwargs = {}

    # sampling
    output_path: str = "outputs"

    results = []

    @classmethod
    def setUpClass(cls):
        cls.results = []

    def verify_single_generation_result(self, result):
        self.assertIsNotNone(result, "Generation failed")
        self.assertTrue(
            "samples" in result and isinstance(result["samples"], torch.Tensor),
            f"Incorrect Generation result",
        )

    def _run_test(self, name, server_kwargs, test_key: str):
        generator = DiffGenerator.from_pretrained(**server_kwargs)
        result = generator.generate(prompt="A curious raccoon")
        self.verify_single_generation_result(result)

    def test_single_gpu(self):
        self._run_test(
            name=self.server_kwargs["model_path"],
            server_kwargs=self.server_kwargs | dict(num_gpus=1),
            test_key="test_single_gpu",
        )

    def test_cfg_parallel(self):
        self._run_test(
            name=self.server_kwargs["model_path"],
            server_kwargs=self.server_kwargs
            | dict(num_gpus=2, enable_cfg_parallel=True),
            test_key="test_cfg_parallel",
        )

    def test_multiple_prompts(self):
        generator = DiffGenerator.from_pretrained(
            **self.server_kwargs | dict(num_gpus=2, enable_cfg_parallel=True)
        )
        prompts = ["A curious raccoon", "A curious cat"]
        results = generator.generate(prompt=prompts)

        self.assertEqual(len(results), len(prompts), "Some generation tasks fail")
        for result in results:
            self.verify_single_generation_result(result)


class TestWan2_1_T2V(TestGeneratorAPIBase):
    server_kwargs = {"model_path": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"}


if __name__ == "__main__":
    del TestGeneratorAPIBase
    unittest.main()
