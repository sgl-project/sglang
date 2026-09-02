"""NPU smoke test for the GLM-Image external-AR distributed topology."""

from __future__ import annotations

import base64
import os
import signal
import subprocess
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
import torch

from sglang.multimodal_gen.test.server.ascend.testcase_configs_npu import (
    GLM_IMAGE_WEIGHTS_PATH,
)
from sglang.multimodal_gen.test.test_utils import find_free_port, wait_for_server_health
from sglang.test.test_utils import CustomTestCase

HOST = "127.0.0.1"
_LOG_DIR = Path(os.environ.get("SGLANG_TEST_LOG_DIR", "/tmp"))
_STARTUP_TIMEOUT_S = float(os.environ.get("SGLANG_GLM_AR_STARTUP_TIMEOUT", "600"))
# A3 warm AR (26.711s) plus six sequential denoiser outputs (24.1945s each).
_EXPECTED_MAKESPAN_S = 172.0
_PERFORMANCE_TOLERANCE = 0.25


def _kill_process_tree(proc: subprocess.Popen) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def _tail_log(path: Path, lines: int = 80) -> str:
    if not path.exists():
        return f"<no log at {path}>"
    try:
        return "\n".join(path.read_text(errors="ignore").splitlines()[-lines:])
    except OSError as error:
        return f"<log read failed: {error}>"


def _wait_for_log(path: Path, message: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            try:
                if message in path.read_text(errors="ignore"):
                    return
            except OSError:
                pass
        time.sleep(2)
    raise TimeoutError(f"Missing {message!r} in {path}:\n{_tail_log(path)}")


class _GlmDistributedCluster:
    def __init__(self) -> None:
        self.model_path = Path(GLM_IMAGE_WEIGHTS_PATH)
        self.ar_port = find_free_port(HOST)
        self.denoiser_port = find_free_port(HOST)
        self.head_port = find_free_port(HOST)
        self.head_scheduler_port = find_free_port(HOST)
        self.master_port = find_free_port(HOST)
        self.processes: list[subprocess.Popen] = []
        self.log_paths = {
            "ar": _LOG_DIR / "glm_image_distributed_ar.log",
            "denoiser": _LOG_DIR / "glm_image_distributed_denoiser.log",
            "head": _LOG_DIR / "glm_image_distributed_head.log",
        }
        self._log_handles: list = []

    def __enter__(self) -> _GlmDistributedCluster:
        if not self.model_path.is_dir():
            raise RuntimeError(
                f"GLM-Image ModelScope cache is missing: {self.model_path}"
            )
        try:
            self._start_ar()
            self._start_denoiser()
            self._start_head()
        except Exception:
            self.stop()
            raise
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    def _start_process(self, command: list[str], log_name: str) -> None:
        log_handle = open(self.log_paths[log_name], "w")
        self._log_handles.append(log_handle)
        self.processes.append(
            subprocess.Popen(
                command,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setsid,
                env=os.environ.copy(),
            )
        )

    def _start_ar(self) -> None:
        self._start_process(
            [
                "sglang",
                "serve",
                "--model-path",
                str(self.model_path / "vision_language_encoder"),
                "--tokenizer-path",
                str(self.model_path / "processor"),
                "--enable-multimodal",
                "--device",
                "npu",
                "--attention-backend",
                "ascend",
                "--disable-fast-image-processor",
                "--tp-size",
                "1",
                "--cuda-graph-bs",
                "2",
                "--base-gpu-id",
                "0",
                "--host",
                HOST,
                "--port",
                str(self.ar_port),
            ],
            "ar",
        )
        self._wait_for_health("ar", self.ar_port)

    def _start_denoiser(self) -> None:
        self._start_process(
            [
                "sglang",
                "serve",
                "--model-path",
                str(self.model_path),
                "--disagg-role",
                "denoiser",
                "--disagg-server-addr",
                f"tcp://{HOST}:{self.head_scheduler_port}",
                "--srt-encoder-url",
                f"http://{HOST}:{self.ar_port}",
                "--scheduler-port",
                str(self.denoiser_port),
                "--master-port",
                str(self.master_port),
                "--num-gpus",
                "1",
                "--base-gpu-id",
                "1",
                "--denoiser-sp",
                "1",
                "--cfg-parallel-size",
                "1",
                "--batching-max-size",
                "1",
                "--dit-cpu-offload",
                "false",
                "--attention-backend",
                "fa",
            ],
            "denoiser",
        )

    def _start_head(self) -> None:
        self._start_process(
            [
                "sglang",
                "serve",
                "--model-path",
                str(self.model_path),
                "--disagg-role",
                "server",
                "--denoiser-urls",
                f"tcp://{HOST}:{self.denoiser_port}",
                "--srt-encoder-url",
                f"http://{HOST}:{self.ar_port}",
                "--batching-mode",
                "dynamic",
                "--batching-max-size",
                "2",
                "--batching-delay-ms",
                "100",
                "--scheduler-port",
                str(self.head_scheduler_port),
                "--host",
                HOST,
                "--port",
                str(self.head_port),
            ],
            "head",
        )
        _wait_for_log(
            self.log_paths["denoiser"], "Role DENOISER ready", _STARTUP_TIMEOUT_S
        )
        self._wait_for_health("head", self.head_port)

    def _wait_for_health(self, name: str, port: int) -> None:
        try:
            wait_for_server_health(
                f"http://{HOST}:{port}", path="/v1/models", timeout=_STARTUP_TIMEOUT_S
            )
        except Exception as error:
            raise RuntimeError(
                f"{name} failed to become healthy:\n{_tail_log(self.log_paths[name])}"
            ) from error

    def stop(self) -> None:
        for process in self.processes:
            _kill_process_tree(process)
        for log_handle in self._log_handles:
            log_handle.close()
        self.processes.clear()
        self._log_handles.clear()


class TestGlmImageDistributedNpu(CustomTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        if not hasattr(torch, "npu") or torch.npu.device_count() < 2:
            raise unittest.SkipTest("requires two Ascend NPUs")
        cls.cluster = _GlmDistributedCluster()
        cls.cluster.__enter__()

    @classmethod
    def tearDownClass(cls) -> None:
        if hasattr(cls, "cluster"):
            for name, path in cls.cluster.log_paths.items():
                print(f"\n=== [glm-image-distributed] {name} log tail ===")
                print(_tail_log(path))
            cls.cluster.stop()
        super().tearDownClass()

    def _generate(self, prompt: str, n: int = 1) -> list[bytes]:
        response = requests.post(
            f"http://{HOST}:{self.cluster.head_port}/v1/images/generations",
            json={
                "model": str(self.cluster.model_path),
                "prompt": prompt,
                "n": n,
                "size": "1024x1024",
                "response_format": "b64_json",
            },
            timeout=600,
        )
        response.raise_for_status()
        images = response.json()["data"]
        self.assertEqual(len(images), n)
        return [base64.b64decode(image["b64_json"]) for image in images]

    def test_external_ar_batching_multi_output_disaggregation_performance(
        self,
    ) -> None:
        self._generate("A warmup landscape")
        requests_to_generate = [
            ("A mountain sunrise", 2),
            ("A city at night", 1),
            ("A forest lake", 2),
            ("A desert sunset", 1),
        ]

        def generate(request: tuple[str, int]) -> tuple[list[bytes], float]:
            prompt, output_count = request
            start_time = time.perf_counter()
            images = self._generate(prompt, n=output_count)
            return images, time.perf_counter() - start_time

        start_time = time.perf_counter()
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(generate, requests_to_generate))
        makespan_s = time.perf_counter() - start_time
        request_latencies_s = [latency_s for _, latency_s in results]
        images = [image for request_images, _ in results for image in request_images]

        print(
            "GLM distributed performance: "
            f"makespan={makespan_s:.2f}s, "
            f"request_latencies={[round(value, 2) for value in request_latencies_s]}"
        )
        self.assertLessEqual(
            makespan_s,
            _EXPECTED_MAKESPAN_S * (1 + _PERFORMANCE_TOLERANCE),
            "GLM external-AR and DiT overlap performance regressed",
        )

        self.assertTrue(
            all(images),
            "each requested output must produce a decoded image",
        )
        head_log = self.cluster.log_paths["head"].read_text(errors="ignore")
        self.assertGreaterEqual(
            head_log.count(
                "GLM distributed AR dispatched batch size=2 requests, 2 outputs"
            ),
            1,
            "the two n=1 requests must share an external-AR batch",
        )
        self.assertGreaterEqual(
            head_log.count(
                "GLM distributed AR dispatched batch size=1 requests, 2 outputs"
            ),
            2,
            "each n=2 request must preserve both outputs through disaggregation",
        )


if __name__ == "__main__":
    unittest.main()
