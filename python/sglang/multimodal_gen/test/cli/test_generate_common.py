# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

"""
    Common generate cli test, one test for image and video each
"""

import os
import unittest
from pathlib import Path

from PIL import Image

from sglang.multimodal_gen.test.test_utils import (
    TestCLIBase,
    check_image_size,
    is_mp4,
    run_command,
)


class TestGenerate(TestCLIBase):
    model_path = "black-forest-labs/FLUX.1-dev"
    launch_file_name = "launch_flux.json"
    output_name = "FLUX.1-dev, single gpu"
    ext = "jpg"

    def test_generate_with_config(self):
        test_dir = Path(__file__).parent
        config_path = (
            (test_dir / ".." / "test_files" / self.launch_file_name)
            .resolve()
            .as_posix()
        )
        command = [
            "sgl_diffusion",
            "generate",
            f"--config={config_path}",
        ]
        duration = run_command(command)

        self.assertIsNotNone(duration, f"Run command failed: {command}")

        # verify
        self.verify_image(self.output_name)

    def test_generate_multiple_outputs(self):
        command = [
            "sglang",
            "generate",
            "--prompt='A curious raccoon'",
            "--output-path=outputs",
            f"--model-path={self.model_path}",
            "--save-output",
            f"--output-file-name={self.output_name}",
            "--num-outputs-per-prompt=2",
            "--width=720",
            "--height=720",
        ]
        duration = run_command(command)
        self.assertIsNotNone(duration, f"Run command failed: {command}")

        self.verify_image(f"{self.output_name}_0.{self.ext}")
        self.verify_image(f"{self.output_name}_1.{self.ext}")

    def verify_image(self, output_name):
        path = os.path.join("outputs", output_name)
        with Image.open(path) as image:
            check_image_size(self, image, 720, 720)

    def verify_video(self, output_name):
        path = os.path.join("outputs", output_name)
        with open(path, "rb") as f:
            header = f.read(12)
            assert is_mp4(header)


class TestWanGenerate(TestGenerate):
    model_path = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    launch_file_name = "launch_wan.json"
    output_name = "Wan2.1-T2V-1.3B-Diffusers, single gpu"
    ext = "mp4"

    def test_generate_multiple_outputs(self):
        command = [
            "sglang",
            "generate",
            "--prompt='A curious raccoon'",
            "--output-path=outputs",
            f"--model-path={self.model_path}",
            "--save-output",
            f"--output-file-name={self.output_name}",
            "--num-outputs-per-prompt=2",
            "--width=720",
            "--height=720",
        ]
        duration = run_command(command)
        self.assertIsNotNone(duration, f"Run command failed: {command}")

        self.verify_video(f"{self.output_name}_0.{self.ext}")
        # FIXME: second video is a meaningless output
        self.verify_video(f"{self.output_name}_1.{self.ext}")


if __name__ == "__main__":
    unittest.main()
