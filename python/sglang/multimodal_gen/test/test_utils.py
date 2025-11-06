# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo
import os
import shlex
import socket
import subprocess
import sys
import time
import unittest

from PIL import Image

from sglang.multimodal_gen.configs.sample.base import DataType
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


def run_command(command):
    """Runs a command and returns the execution time and status."""
    print(f"Running command: {' '.join(command)}")

    duration = None
    with subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
    ) as process:
        for line in process.stdout:
            sys.stdout.write(line)
            if "Pixel data generated" in line:
                words = line.split(" ")
                duration = float(words[-2])

    if process.returncode == 0:
        return duration
    else:
        print(f"Command failed with exit code {process.returncode}")
        return None


def probe_port(host="127.0.0.1", port=30010, timeout=2.0) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        try:
            s.connect((host, port))
            return True
        except OSError:
            return False


def is_mp4(data):
    idx = data.find(b"ftyp")
    return 0 <= idx <= 32


def is_png(data):
    # PNG files start with: 89 50 4E 47 0D 0A 1A 0A
    return data.startswith(b"\x89PNG\r\n\x1a\n")


def wait_for_port(host="127.0.0.1", port=30010, deadline=300.0, interval=0.5):
    end = time.time() + deadline
    last_err = None
    while time.time() < end:
        if probe_port(host, port, timeout=interval):
            return True
        time.sleep(interval)
    raise TimeoutError(f"Port {host}:{port} not ready. Last error: {last_err}")


def check_image_size(ut, image, width, height):
    # check image size
    ut.assertEqual(image.size, (width, height))


class TestCLIBase(unittest.TestCase):
    model_path: str = None
    extra_args = []
    data_type: DataType = None
    # tested on h100
    thresholds = {}

    width: int = 720
    height: int = 720
    output_path: str = "outputs"

    base_command = [
        "sglang",
        "generate",
        "--text-encoder-cpu-offload",
        "--pin-cpu-memory",
        "--prompt='A curious raccoon'",
        "--save-output",
        "--log-level=debug",
        f"--width={width}",
        f"--height={height}",
        f"--output-path={output_path}",
    ]

    results = []

    @classmethod
    def setUpClass(cls):
        cls.results = []

    def _run_command(self, name, model_path: str, test_key: str = "", args=[]):
        command = (
            self.base_command
            + [f"--model-path={model_path}"]
            + shlex.split(args or "")
            + [f"--output-file-name={name}"]
            + self.extra_args
        )
        duration = run_command(command)
        status = "Success" if duration else "Failed"

        duration_str = f"{duration:.4f}s" if duration else "NA"
        self.__class__.results.append(
            {"name": name, "key": test_key, "duration": duration_str, "status": status}
        )

        return name, duration, status


class TestGenerateBase(TestCLIBase):
    model_path: str = None
    extra_args = []
    data_type: DataType = None
    # tested on h100
    thresholds = {}

    width: int = 720
    height: int = 720
    output_path: str = "outputs"
    image_path: str | None = None
    prompt: str | None = "A curious raccoon"

    base_command = [
        "sglang",
        "generate",
        # "--text-encoder-cpu-offload",
        # "--pin-cpu-memory",
        f"--prompt='{prompt}'",
        "--save-output",
        "--log-level=debug",
        f"--width={width}",
        f"--height={height}",
        f"--output-path={output_path}",
    ]

    results = []

    @classmethod
    def setUpClass(cls):
        cls.results = []

    @classmethod
    def tearDownClass(cls):
        # Print markdown table
        print("\n## Test Results\n")
        print("| Test Case                      | Duration | Status  |")
        print("|--------------------------------|----------|---------|")
        test_keys = ["test_single_gpu", "test_cfg_parallel", "test_usp", "test_mixed"]
        test_key_to_order = {
            test_key: order for order, test_key in enumerate(test_keys)
        }

        ordered_results: list[dict] = [{}] * len(test_keys)

        for result in cls.results:
            order = test_key_to_order[result["key"]]
            ordered_results[order] = result

        for result in ordered_results:
            if not result:
                continue
            status = (
                result["status"] and result["duration"] <= cls.thresholds[result["key"]]
            )
            print(f"| {result['name']:<30} | {result['duration']:<8} | {status:<7} |")
        print()
        durations = [result["duration"] for result in cls.results]
        print(" | ".join([""] + durations + [""]))

    def _run_test(self, name, args, model_path: str, test_key: str):
        time_threshold = self.thresholds[test_key]
        name, duration, status = self._run_command(
            name, args=args, model_path=model_path, test_key=test_key
        )
        self.verify(status, name, duration, time_threshold)

    def verify(self, status, name, duration, time_threshold):
        print("-" * 80)
        print("\n" * 3)

        # test task status
        self.assertEqual(status, "Success", f"{name} command failed")
        self.assertIsNotNone(duration, f"Could not parse duration for {name}")
        self.assertLessEqual(
            duration,
            time_threshold,
            f"{name} failed with {duration:.4f}s > {time_threshold}s",
        )

        # test output file
        path = os.path.join(
            self.output_path, f"{name}.{self.data_type.get_default_extension()}"
        )
        self.assertTrue(os.path.exists(path), f"Output file not exist for {path}")
        if self.data_type == DataType.IMAGE:
            with Image.open(path) as image:
                check_image_size(self, image, self.width, self.height)
        logger.info(f"{name} passed in {duration:.4f}s (threshold: {time_threshold}s)")

    def model_name(self):
        return self.model_path.split("/")[-1]

    def test_single_gpu(self):
        """single gpu"""
        self._run_test(
            name=f"{self.model_name()}, single gpu",
            args=None,
            model_path=self.model_path,
            test_key="test_single_gpu",
        )

    def test_cfg_parallel(self):
        """cfg parallel"""
        if self.data_type == DataType.IMAGE:
            return
        self._run_test(
            name=f"{self.model_name()}, cfg parallel",
            args="--num-gpus 2 --enable-cfg-parallel",
            model_path=self.model_path,
            test_key="test_cfg_parallel",
        )

    def test_usp(self):
        """usp"""
        if self.data_type == DataType.IMAGE:
            return
        self._run_test(
            name=f"{self.model_name()}, usp",
            args="--num-gpus 4 --ulysses-degree=2 --ring-degree=2",
            model_path=self.model_path,
            test_key="test_usp",
        )

    def test_mixed(self):
        """mixed"""
        if self.data_type == DataType.IMAGE:
            return
        self._run_test(
            name=f"{self.model_name()}, mixed",
            args="--num-gpus 4 --ulysses-degree=2 --ring-degree=1 --enable-cfg-parallel",
            model_path=self.model_path,
            test_key="test_mixed",
        )
