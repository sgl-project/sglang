# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

"""
    Common generate cli test, one test for image and video each
"""
import dataclasses
import os
import shlex
import subprocess
import sys
import unittest
from typing import Optional

from PIL import Image

from sglang.multimodal_gen.configs.sample.sampling_params import DataType
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.test.test_utils import check_image_size

logger = init_logger(__name__)


@dataclasses.dataclass
class TestResult:
    name: str
    key: str
    succeed: bool


def run_command(command) -> Optional[float]:
    """Runs a command and returns the execution time and status."""
    print(f"Running command: {shlex.join(command)}")

    with subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
    ) as process:
        for line in process.stdout:
            sys.stdout.write(line)
        process.wait()
        if process.returncode == 0:
            return True
        print(f"Command failed with exit code {process.returncode}")
    return False


class CLIBase(unittest.TestCase):
    model_path: str = None
    extra_args = []
    data_type: DataType = None
    # tested on h100

    width: int = 720
    height: int = 720
    output_path: str = "test_outputs"

    def get_base_command(self):
        return [
            "sglang",
            "generate",
            "--prompt",
            "A curious raccoon",
            "--save-output",
            "--log-level=debug",
            f"--width={self.width}",
            f"--height={self.height}",
            f"--output-path={self.output_path}",
        ]

    def _run_command(self, name: str, model_path: str, args=[]):
        command = (
            self.get_base_command()
            + [f"--model-path={model_path}"]
            + shlex.split(args or "")
            + ["--output-file-name", f"{name}"]
            + self.extra_args
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
        if self.data_type == DataType.IMAGE:
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
