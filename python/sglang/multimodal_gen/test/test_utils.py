import shlex
import socket
import subprocess
import sys
import time
import unittest

from sglang.multimodal_gen.api.configs.sample.base import DataType


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
            if "Pixel data generated successfully in " in line:
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
        "sgl-diffusion",
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
