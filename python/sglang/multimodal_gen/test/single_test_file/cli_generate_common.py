# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

"""
Common generate cli test, one test for image and video each
"""

import os
import shlex
import tempfile
import unittest
from typing import Any

from PIL import Image


def run_command(command: list[str]) -> bool:
    from sglang.multimodal_gen.test.test_utils import run_command as _run_command

    return _run_command(command)


def check_image_size(ut, image, width, height):
    ut.assertEqual(image.size, (width, height))


class CLIBase(unittest.TestCase):
    model_path: str = None
    extra_args: tuple[str, ...] = ()
    data_type: Any = None

    log_level: str = "info"
    width: int = 720
    height: int = 720
    output_path: str | None = None

    def setUp(self):
        super().setUp()
        self._temp_output_dir = None
        if self.output_path is None:
            self._temp_output_dir = tempfile.TemporaryDirectory(
                prefix="sglang_cli_test_"
            )
            self.output_path = self._temp_output_dir.name
        else:
            os.makedirs(self.output_path, exist_ok=True)
            self._clear_output_files()

    def tearDown(self):
        try:
            if self._temp_output_dir is not None:
                self._temp_output_dir.cleanup()
            elif self.output_path and os.path.exists(self.output_path):
                self._clear_output_files()
        finally:
            super().tearDown()

    def _clear_output_files(self):
        for filename in os.listdir(self.output_path):
            path = os.path.join(self.output_path, filename)
            if os.path.isfile(path):
                os.remove(path)

    def get_base_command(self):
        return [
            "sglang",
            "generate",
            "--prompt",
            "A red cube on a white table",
            "--save-output",
            f"--log-level={self.log_level}",
            f"--width={self.width}",
            f"--height={self.height}",
            f"--output-path={self.output_path}",
        ]

    def _run_command(self, name: str, model_path: str, args: str | None = None):
        command = (
            self.get_base_command()
            + [f"--model-path={model_path}"]
            + shlex.split(args or "")
            + ["--output-file-name", f"{name}"]
            + list(self.extra_args)
        )
        succeed = run_command(command)
        status = "Success" if succeed else "Failed"

        return name, status

    def _run_test(self, name: str, args, model_path: str, test_key: str):
        name, status = self._run_command(name, args=args, model_path=model_path)
        self.verify(status, name)

    def verify(self, status, name):
        print("-" * 80)
        print("\n" * 3)

        # test task status
        self.assertEqual(status, "Success", f"{name} command failed")

        # test output file
        path = os.path.join(
            self.output_path, f"{name}.{self.data_type.get_default_extension()}"
        )
        self.assertTrue(os.path.exists(path), f"Output file not exist for {path}")
        if self.data_type.get_default_extension() in ("png", "jpg", "jpeg", "webp"):
            with Image.open(path) as image:
                check_image_size(self, image, self.width, self.height)

    def model_name(self):
        return self.model_path.split("/")[-1]

    def test_single_gpu(self):
        """single gpu"""
        self._run_test(
            name=f"{self.model_name()}_single_gpu",
            args=None,
            model_path=self.model_path,
            test_key="test_single_gpu",
        )
