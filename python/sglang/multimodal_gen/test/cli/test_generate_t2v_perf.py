# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

import os.path
import unittest
from pathlib import Path

from PIL import Image

from sglang.multimodal_gen.configs.sample.base import DataType
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.test.test_utils import (
    TestCLIBase,
    TestGenerateBase,
    check_image_size,
)

logger = init_logger(__name__)


class TestFastWan2_1_T2V(TestGenerateBase):
    model_path = "FastVideo/FastWan2.1-T2V-1.3B-Diffusers"
    extra_args = ["--attention-backend=video_sparse_attn"]
    data_type: DataType = DataType.VIDEO
    thresholds = {
        "test_single_gpu": 13.0,
        "test_cfg_parallel": 15.0,
        "test_usp": 15.0,
        "test_mixed": 15.0,
    }


class TestFastWan2_2_T2V(TestGenerateBase):
    model_path = "FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers"
    extra_args = []
    data_type: DataType = DataType.VIDEO
    thresholds = {
        "test_single_gpu": 25.0,
        "test_cfg_parallel": 30.0,
        "test_usp": 30.0,
        "test_mixed": 30.0,
    }


class TestWan2_1_T2V(TestGenerateBase):
    model_path = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    extra_args = []
    data_type: DataType = DataType.VIDEO
    thresholds = {
        "test_single_gpu": 76.0,
        "test_cfg_parallel": 46.5 * 1.05,
        "test_usp": 22.5,
        "test_mixed": 26.5,
    }


class TestWan2_2_T2V(TestGenerateBase):
    model_path = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
    extra_args = []
    data_type: DataType = DataType.VIDEO
    thresholds = {
        "test_single_gpu": 865,
        "test_cfg_parallel": 446,
        "test_usp": 124,
        "test_mixed": 159,
    }

    def test_mixed(self):
        pass

    def test_cfg_parallel(self):
        pass


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
        "test_single_gpu": 10.0 * 1.05,
    }


class TestQwenImageEdit(TestGenerateBase):
    model_path = "Qwen/Qwen-Image-Edit"
    extra_args = []
    data_type: DataType = DataType.IMAGE
    thresholds = {
        "test_single_gpu": 40.5 * 1.05,
    }

    prompt: str | None = (
        "Change the rabbit's color to purple, with a flash light background."
    )

    def test_cfg_parallel(self):
        pass

    def test_mixed(self):
        pass

    def test_usp(self):
        pass

    def test_single_gpu(self):
        test_dir = Path(__file__).parent
        img_path = (test_dir / ".." / "test_files" / "rabbit.jpg").resolve().as_posix()
        self.base_command = [
            "sglang",
            "generate",
            "--text-encoder-cpu-offload",
            "--pin-cpu-memory",
            f"--prompt='{self.prompt}'",
            "--save-output",
            "--log-level=debug",
            f"--width={self.width}",
            f"--height={self.height}",
            f"--output-path={self.output_path}",
        ] + [f"--image-path={img_path}"]

        self._run_test(
            name=f"{self.model_name()}, single gpu",
            args=None,
            model_path=self.model_path,
            test_key="test_single_gpu",
        )


if __name__ == "__main__":
    del TestGenerateBase
    unittest.main()
