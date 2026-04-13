# SPDX-License-Identifier: Apache-2.0
"""End-to-end tests for disaggregated diffusion server.

Launches 4 CLI processes (encoder, denoiser, decoder, server head) via
``sglang serve --disagg-role ...``, sends generation requests via
the OpenAI-compatible API, and validates the output.

Usage:
    # Run all disagg tests (requires 2+ GPUs)
    python -m pytest python/sglang/multimodal_gen/test/server/test_disagg_server.py -v -s

    # Run only the video (Wan) test
    python -m pytest ... -k wan

    # Run only the image (Z-Image-Turbo) test
    python -m pytest ... -k zimage

    # With custom GPU assignment
    DISAGG_ENCODER_GPU=0 DISAGG_DENOISER_GPU=1 DISAGG_DECODER_GPU=2 \
        python -m pytest python/sglang/multimodal_gen/test/server/test_disagg_server.py -v -s
"""

from __future__ import annotations

import base64
import os
import shlex
import subprocess
import sys
import tempfile
import threading
import time
from collections import OrderedDict
from pathlib import Path

import pytest

from sglang.multimodal_gen.runtime.utils.common import kill_process_tree
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.test.test_utils import find_free_port

logger = init_logger(__name__)

HOST = "127.0.0.1"

# ---------------------------------------------------------------------------
# Model configs
# ---------------------------------------------------------------------------

# Video model (Wan T2V)
WAN_MODEL = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"

# Image model (Z-Image-Turbo)
ZIMAGE_MODEL = "Tongyi-MAI/Z-Image-Turbo"


# ---------------------------------------------------------------------------
# Cluster launch helpers
# ---------------------------------------------------------------------------


def _start_process(command: list[str], log_path: Path, env: dict) -> subprocess.Popen:
    """Start a subprocess and stream its output to a log file."""
    logger.info("Running: %s", shlex.join(command))
    logger.info("  Log: %s", log_path)

    fh = log_path.open("w", encoding="utf-8", buffering=1)
    proc = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )

    def _drain(pipe, file):
        try:
            with pipe:
                for line in iter(pipe.readline, ""):
                    sys.stdout.write(line)
                    sys.stdout.flush()
                    file.write(line)
                    file.flush()
        except Exception:
            pass

    t = threading.Thread(target=_drain, args=(proc.stdout, fh), daemon=True)
    t.start()
    return proc


def _wait_for_log_message(
    log_path: Path,
    proc: subprocess.Popen,
    message: str,
    timeout: float,
    label: str = "",
) -> None:
    """Poll log file until a specific message appears."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"{label or 'Process'} exited early (code {proc.returncode}). "
                f"See log: {log_path}"
            )
        if log_path.exists():
            try:
                content = log_path.read_text(encoding="utf-8", errors="ignore")
                if message in content:
                    return
            except Exception:
                pass
        time.sleep(1)
    raise TimeoutError(
        f"{label or 'Process'} not ready within {timeout}s. See log: {log_path}"
    )


def _launch_disagg_cluster(model: str, extra_args: list[str] | None = None):
    """Launch a disaggregated diffusion cluster (4 CLI processes).

    Args:
        model: HuggingFace model path.
        extra_args: Additional CLI args appended to each role instance command.
    """
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if torch.cuda.device_count() < 2:
        pytest.skip("Requires 2+ GPUs for disaggregated mode")

    encoder_gpu = int(os.environ.get("DISAGG_ENCODER_GPU", "0"))
    denoiser_gpu = int(os.environ.get("DISAGG_DENOISER_GPU", "1"))
    decoder_gpu = int(os.environ.get("DISAGG_DECODER_GPU", "0"))

    # Allocate ports dynamically
    http_port = find_free_port(HOST)
    server_scheduler_port = find_free_port(HOST)
    encoder_port = find_free_port(HOST)
    denoiser_port = find_free_port(HOST)
    decoder_port = find_free_port(HOST)

    ds_addr = f"tcp://{HOST}:{server_scheduler_port}"
    log_dir = Path(tempfile.mkdtemp(prefix="sglang_disagg_test_"))
    env = os.environ.copy()

    wait_timeout = float(os.environ.get("SGLANG_TEST_WAIT_SECS", "600"))

    processes: list[tuple[str, subprocess.Popen]] = []

    def _cleanup():
        for name, proc in processes:
            if proc.poll() is None:
                logger.info("Killing %s (pid=%d)...", name, proc.pid)
                kill_process_tree(proc.pid)
        for name, proc in processes:
            proc.wait(timeout=10)

    try:
        # 1-3. Launch role instances.
        roles = [
            ("encoder", encoder_port, encoder_gpu),
            ("denoiser", denoiser_port, denoiser_gpu),
            ("decoder", decoder_port, decoder_gpu),
        ]

        gpu_groups: dict[int, list] = OrderedDict()
        for role_name, port, gpu_id in roles:
            gpu_groups.setdefault(gpu_id, []).append((role_name, port, gpu_id))

        def _launch_and_wait(role_name, port, gpu_id):
            cmd = [
                "sglang",
                "serve",
                "--model-path",
                model,
                "--disagg-role",
                role_name,
                "--disagg-server-addr",
                ds_addr,
                "--scheduler-port",
                str(port),
                "--num-gpus",
                "1",
                "--base-gpu-id",
                str(gpu_id),
                "--log-level",
                "debug",
            ]
            if extra_args:
                cmd += extra_args
            log_path = log_dir / f"{role_name}.log"
            proc = _start_process(cmd, log_path, env)
            processes.append((role_name, proc))

            ready_msg = f"Role {role_name.upper()} ready"
            logger.info("Waiting for %s to be ready...", role_name)
            try:
                _wait_for_log_message(
                    log_dir / f"{role_name}.log",
                    proc,
                    ready_msg,
                    wait_timeout,
                    role_name,
                )
            except (RuntimeError, TimeoutError) as e:
                _cleanup()
                pytest.fail(str(e))
            logger.info("%s is ready.", role_name)

        for gpu_id, group in gpu_groups.items():
            for role_name, port, gid in group:
                _launch_and_wait(role_name, port, gid)

        # 4. Launch DiffusionServer head
        server_cmd = [
            "sglang",
            "serve",
            "--model-path",
            model,
            "--disagg-role",
            "server",
            "--encoder-urls",
            f"tcp://{HOST}:{encoder_port}",
            "--denoiser-urls",
            f"tcp://{HOST}:{denoiser_port}",
            "--decoder-urls",
            f"tcp://{HOST}:{decoder_port}",
            "--scheduler-port",
            str(server_scheduler_port),
            "--port",
            str(http_port),
            "--host",
            HOST,
            "--log-level",
            "debug",
        ]
        server_log = log_dir / "server.log"
        server_proc = _start_process(server_cmd, server_log, env)
        processes.append(("server", server_proc))

        # Wait for HTTP server to be ready
        logger.info("Waiting for HTTP server at port %d...", http_port)
        _wait_for_log_message(
            server_log,
            server_proc,
            "Application startup complete.",
            wait_timeout,
            "server",
        )
        logger.info("All components ready!")

        yield {"port": http_port, "model": model, "log_dir": log_dir}

    finally:
        _cleanup()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def disagg_wan_server():
    """Disaggregated cluster for Wan T2V (video generation)."""
    yield from _launch_disagg_cluster(WAN_MODEL)


@pytest.fixture(scope="module")
def disagg_zimage_server():
    """Disaggregated cluster for Z-Image-Turbo (image generation)."""
    yield from _launch_disagg_cluster(ZIMAGE_MODEL)


# ---------------------------------------------------------------------------
# Shared test logic
# ---------------------------------------------------------------------------


def _test_health_check(server_info):
    import requests

    port = server_info["port"]
    resp = requests.get(f"http://{HOST}:{port}/health")
    assert resp.status_code == 200


def _test_video_generation(server_info):
    from openai import OpenAI

    port = server_info["port"]
    model = server_info["model"]

    client = OpenAI(
        base_url=f"http://{HOST}:{port}/v1",
        api_key="unused",
    )

    prompt = "A curious raccoon exploring a garden"

    job = client.videos.create(
        model=model,
        prompt=prompt,
        size="832x480",
    )
    video_id = job.id
    logger.info("Created video job: %s", video_id)

    # Poll for completion
    timeout = 300
    deadline = time.time() + timeout
    completed = False

    while time.time() < deadline:
        page = client.videos.list()
        item = next((v for v in page.data if v.id == video_id), None)
        if item and getattr(item, "status", None) == "completed":
            completed = True
            break
        time.sleep(2)

    assert completed, f"Video job {video_id} did not complete within {timeout}s"

    resp = client.videos.download_content(video_id=video_id)
    content = resp.read()

    assert len(content) > 0, "Empty video content"
    logger.info(
        "Video generation completed: job=%s, size=%d bytes",
        video_id,
        len(content),
    )


def _test_image_generation(server_info):
    from openai import OpenAI

    port = server_info["port"]
    model = server_info["model"]

    client = OpenAI(
        base_url=f"http://{HOST}:{port}/v1",
        api_key="unused",
    )

    prompt = "A beautiful sunset over a mountain lake"

    response = client.images.generate(
        model=model,
        prompt=prompt,
        n=1,
        size="1024x1024",
        response_format="b64_json",
    )

    assert len(response.data) > 0, "No images returned"
    b64_data = response.data[0].b64_json
    assert b64_data is not None, "No b64_json in response"

    img_bytes = base64.b64decode(b64_data)
    assert len(img_bytes) > 1000, f"Image too small ({len(img_bytes)} bytes)"

    logger.info(
        "Image generation completed: size=%d bytes",
        len(img_bytes),
    )


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------


class TestDisaggWan:
    """E2E tests for disaggregated Wan T2V (video generation)."""

    def test_health_check(self, disagg_wan_server):
        _test_health_check(disagg_wan_server)

    def test_video_generation(self, disagg_wan_server):
        _test_video_generation(disagg_wan_server)


class TestDisaggZImage:
    """E2E tests for disaggregated Z-Image-Turbo (image generation)."""

    def test_health_check(self, disagg_zimage_server):
        _test_health_check(disagg_zimage_server)

    def test_image_generation(self, disagg_zimage_server):
        _test_image_generation(disagg_zimage_server)
