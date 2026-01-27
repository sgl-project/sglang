"""AMD performance benchmark for Z-Image-Turbo diffusion model.

This test benchmarks Z-Image-Turbo text-to-image generation on AMD GPUs.

Example usage:
    python -m pytest test_zimage_turbo_perf_amd.py -v
"""

import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from urllib.parse import urlparse

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    write_github_step_summary,
)

register_amd_ci(est_time=600, suite="stage-b-test-small-1-gpu-diffusion-amd")

ZIMAGE_MODEL = "Tongyi-MAI/Z-Image-Turbo"

# Performance thresholds (with margin for variance)
# Based on baseline: throughput=0.07 req/s, latency_mean=14.57s
MIN_THROUGHPUT_QPS = 0.02  # Minimum acceptable throughput (req/s)
MAX_LATENCY_MEAN_S = 10.0  # Maximum acceptable mean latency (seconds)


def launch_diffusion_server(
    model: str,
    base_url: str,
    timeout: float,
    other_args: list[str] = None,
) -> subprocess.Popen:
    """Launch a diffusion server."""
    parsed = urlparse(base_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 30000

    cmd = [
        sys.executable,
        "-m",
        "sglang.multimodal_gen.runtime.launch_server",
        "--model-path",
        model,
        "--host",
        host,
        "--port",
        str(port),
    ]

    if other_args:
        cmd.extend(other_args)

    print(f"Launching diffusion server: {' '.join(cmd)}")

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env={**os.environ},
    )

    # Wait for server to be ready
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            import requests

            response = requests.get(f"{base_url}/health", timeout=5)
            if response.status_code == 200:
                print(f"Server ready after {time.time() - start_time:.1f}s")
                return process
        except Exception:
            pass
        time.sleep(5)

    process.kill()
    raise RuntimeError(f"Diffusion server failed to start within {timeout}s")


def run_diffusion_benchmark(
    base_url: str,
    model: str,
    num_prompts: int = 20,
    width: int = 1024,
    height: int = 1024,
) -> dict:
    """Run diffusion benchmark and return metrics."""
    parsed = urlparse(base_url)
    port = parsed.port or 30000

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        output_file = f.name

    cmd = [
        sys.executable,
        "-m",
        "sglang.multimodal_gen.benchmarks.bench_serving",
        "--port",
        str(port),
        "--model",
        model,
        "--dataset",
        "random",
        "--num-prompts",
        str(num_prompts),
        "--width",
        str(width),
        "--height",
        str(height),
        "--task",
        "text-to-image",
        "--output-file",
        output_file,
    ]

    print(f"Running benchmark: {' '.join(cmd)}")

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env={**os.environ},
    )

    print(f"Benchmark stdout:\n{result.stdout}")
    if result.stderr:
        print(f"Benchmark stderr:\n{result.stderr}")

    if result.returncode != 0:
        raise RuntimeError(f"Benchmark failed with return code {result.returncode}")

    with open(output_file, "r") as f:
        metrics = json.load(f)

    os.unlink(output_file)
    return metrics


class TestZImageTurboPerfAMD(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = ZIMAGE_MODEL
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = [
            "--log-level",
            "info",
            "--enable-torch-compile",
            "--warmup",
            "--dit-cpu-offload",
            "false",
            "--text-encoder-cpu-offload",
            "false",
        ]
        cls.process = launch_diffusion_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_t2i_throughput_latency(self):
        metrics = run_diffusion_benchmark(
            base_url=self.base_url,
            model=self.model,
            num_prompts=8,
            width=1024,
            height=1024,
        )

        throughput_qps = metrics.get("throughput_qps", 0)
        latency_mean = metrics.get("latency_mean", float("inf"))

        print(f"{throughput_qps=:.4f} req/s")
        print(f"{latency_mean=:.4f} s")

        if is_in_ci():
            write_github_step_summary(
                f"### test_t2i_throughput_latency (Z-Image-Turbo)\n"
                f"{throughput_qps=:.4f} req/s\n"
                f"{latency_mean=:.4f} s\n"
            )
            self.assertGreaterEqual(throughput_qps, MIN_THROUGHPUT_QPS)
            self.assertLessEqual(latency_mean, MAX_LATENCY_MEAN_S)


if __name__ == "__main__":
    unittest.main()
