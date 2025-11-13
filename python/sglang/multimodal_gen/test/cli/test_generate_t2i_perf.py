# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

import unittest
from pathlib import Path

from sglang.multimodal_gen.configs.sample.base import DataType
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.test.test_utils import TestGenerateBase

logger = init_logger(__name__)


class TestFlux_T2V(TestGenerateBase):
    model_path = "black-forest-labs/FLUX.1-dev"
    extra_args = []
    data_type: DataType = DataType.IMAGE
    thresholds = {
        "test_single_gpu": 6.90 * 1.05,
    }


class TestQwenImage(TestGenerateBase):
    model_path = "Qwen/Qwen-Image"
    extra_args = []
    data_type: DataType = DataType.IMAGE
    thresholds = {
        "test_single_gpu": 11.7 * 1.05,
    }


class TestQwenImageEdit(TestGenerateBase):
    model_path = "Qwen/Qwen-Image-Edit"
    extra_args = []
    data_type: DataType = DataType.IMAGE
    thresholds = {
        "test_single_gpu": 43.5 * 1.05,
    }

    prompt: str | None = (
        "Change the rabbit's color to purple, with a flash light background."
    )

    def setUp(self):
        test_dir = Path(__file__).parent
        img_path = (test_dir / ".." / "test_files" / "rabbit.jpg").resolve().as_posix()
        self.base_command = [
            "sglang",
            "generate",
            "--text-encoder-cpu-offload",
            "--pin-cpu-memory",
            f"--prompt",
            f"{self.prompt}",
            "--save-output",
            "--log-level=debug",
            f"--width={self.width}",
            f"--height={self.height}",
            f"--output-path={self.output_path}",
        ] + [f"--image-path={img_path}"]

    def test_single_gpu(self):
        self._run_test(
            name=f"{self.model_name()}_single_gpu",
            args=None,
            model_path=self.model_path,
            test_key="test_single_gpu",
        )


if __name__ == "__main__":
    del TestGenerateBase
    unittest.main()
