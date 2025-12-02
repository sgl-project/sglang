# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

import unittest
from pathlib import Path

from sglang.multimodal_gen.configs.sample.sampling_params import DataType
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.test.test_utils import TestGenerateBase

logger = init_logger(__name__)


class TestFlux_T2V(TestGenerateBase):
    model_path = "black-forest-labs/FLUX.1-dev"
    extra_args = []
    data_type: DataType = DataType.IMAGE
    thresholds = {
        "test_single_gpu": 6.5 * 1.05,
        "test_usp": 8.3 * 1.05,
    }

    def test_cfg_parallel(self):
        pass

    def test_mixed(self):
        pass


class TestQwenImage(TestGenerateBase):
    model_path = "Qwen/Qwen-Image"
    extra_args = []
    data_type: DataType = DataType.IMAGE
    thresholds = {
        "test_single_gpu": 10.4 * 1.05,
        "test_usp": 20.2 * 1.05,
    }

    def test_cfg_parallel(self):
        pass

    def test_mixed(self):
        pass


class TestQwenImageEdit(TestGenerateBase):
    model_path = "Qwen/Qwen-Image-Edit"
    extra_args = []
    data_type: DataType = DataType.IMAGE
    thresholds = {
        "test_single_gpu": 33.4 * 1.05,
        "test_usp": 26.9 * 1.05,
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

    def test_cfg_parallel(self):
        pass

    def test_mixed(self):
        pass


if __name__ == "__main__":
    del TestGenerateBase
    unittest.main()
