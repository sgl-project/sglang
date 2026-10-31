"""
Test request logging for diffusion models.

Tests the --log-requests CLI flags for diffusion model serving,
verifying that request logs are correctly written to stdout and files.
"""

import json
import os
import shutil
import tempfile
import time
from pathlib import Path

import pytest
from openai import OpenAI

from sglang.multimodal_gen.test.server.test_server_utils import ServerManager
from sglang.multimodal_gen.test.test_utils import get_dynamic_server_port

# Test models and prompts
IMAGE_MODEL = "Efficient-Large-Model/Sana_600M_512px_diffusers"
IMAGE_PROMPT = "A beautiful sunset over mountains, oil painting style"
VIDEO_MODEL = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
VIDEO_PROMPT = "A cat playing with a ball"

# Timeout settings
BASE_TIMEOUT = float(os.environ.get("SGLANG_TEST_OPENAI_REQUEST_TIMEOUT_SECS", "600"))
POLL_INTERVAL = 1.0


def _start_server(model: str, log_format: str):
    """Start server with request logging enabled."""
    temp_dir = tempfile.mkdtemp()
    port = get_dynamic_server_port()
    extra_args = (
        f"--log-requests "
        f"--log-requests-level 2 "
        f"--log-requests-format {log_format} "
        f"--log-requests-target stdout {temp_dir} "
        f"--strict-ports"
    )
    wait_deadline = float(os.environ.get("SGLANG_TEST_WAIT_SECS", "1200"))
    manager = ServerManager(
        model=model,
        port=port,
        wait_deadline=wait_deadline,
        extra_args=extra_args,
    )
    ctx = manager.start()
    ctx.temp_dir = temp_dir
    return ctx


def _cleanup_server(ctx):
    """Cleanup server and temp directory."""
    ctx.cleanup()
    shutil.rmtree(ctx.temp_dir, ignore_errors=True)


def _create_client(ctx) -> OpenAI:
    """Create OpenAI client for the server."""
    return OpenAI(
        api_key="test",
        base_url=f"http://localhost:{ctx.port}/v1",
        timeout=BASE_TIMEOUT,
    )


def _wait_for_video_completion(client: OpenAI, video_id: str, timeout: float):
    """Poll video job until completion."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        page = client.videos.list()
        item = next((v for v in page.data if v.id == video_id), None)
        status = getattr(item, "status", None) if item else None

        if status == "completed":
            return True
        if status in ("failed", "cancelled", "deleted"):
            pytest.fail(f"Video job {video_id} ended with status={status}")

        time.sleep(POLL_INTERVAL)

    pytest.fail(f"Video job {video_id} did not complete in {timeout}s")


def _verify_json_logs(content: str):
    """Verify JSON logs contain request.received and request.finished events."""
    has_received = False
    has_finished = False

    for line in content.splitlines():
        idx = line.find("{")
        if idx == -1:
            continue
        try:
            data = json.loads(line[idx:])
        except json.JSONDecodeError:
            continue

        if data.get("event") == "request.received":
            has_received = True
        elif data.get("event") == "request.finished":
            has_finished = True

    assert has_received, "request.received event not found"
    assert has_finished, "request.finished event not found"


def _verify_text_logs(content: str, prompt: str):
    """Verify text logs contain Receive, prompt, and Finish markers."""
    assert "Receive:" in content, "'Receive:' not found"
    assert prompt in content, f"Prompt '{prompt}' not found"
    assert "Finish:" in content, "'Finish:' not found"


@pytest.fixture(scope="class")
def image_text_server():
    """Server with text-format logging for image model."""
    ctx = _start_server(IMAGE_MODEL, "text")
    yield ctx
    _cleanup_server(ctx)


@pytest.fixture(scope="class")
def image_json_server():
    """Server with JSON-format logging for image model."""
    ctx = _start_server(IMAGE_MODEL, "json")
    yield ctx
    _cleanup_server(ctx)


@pytest.fixture(scope="class")
def video_text_server():
    """Server with text-format logging for video model."""
    ctx = _start_server(VIDEO_MODEL, "text")
    yield ctx
    _cleanup_server(ctx)


@pytest.fixture(scope="class")
def video_json_server():
    """Server with JSON-format logging for video model."""
    ctx = _start_server(VIDEO_MODEL, "json")
    yield ctx
    _cleanup_server(ctx)


class TestImageRequestLoggerText:
    """Test text-format request logging for image models."""

    def test_request_logging(self, image_text_server):
        ctx = image_text_server
        client = _create_client(ctx)

        # Image generation is synchronous, waits for completion
        client.images.generate(prompt=IMAGE_PROMPT, size="256x256", n=1)

        # Verify stdout and file logs
        stdout = ctx.log_tail(lines=500)
        _verify_text_logs(stdout, IMAGE_PROMPT[:30])

        logs = list(Path(ctx.temp_dir).glob("*.log"))
        assert logs, "No log files found"
        _verify_text_logs("".join(f.read_text() for f in logs), IMAGE_PROMPT[:30])


class TestImageRequestLoggerJson:
    """Test JSON-format request logging for image models."""

    def test_request_logging(self, image_json_server):
        ctx = image_json_server
        client = _create_client(ctx)

        # Image generation is synchronous, waits for completion
        client.images.generate(prompt=IMAGE_PROMPT, size="256x256", n=1)

        # Verify stdout and file logs
        stdout = ctx.log_tail(lines=500)
        _verify_json_logs(stdout)

        logs = list(Path(ctx.temp_dir).glob("*.log"))
        assert logs, "No log files found"
        _verify_json_logs("".join(f.read_text() for f in logs))


class TestVideoRequestLoggerText:
    """Test text-format request logging for video models."""

    def test_request_logging(self, video_text_server):
        ctx = video_text_server
        client = _create_client(ctx)

        # Video generation is async - create job and poll until completion
        job = client.videos.create(
            prompt=VIDEO_PROMPT,
            size="832x480",
            extra_body={"num_frames": 5, "num_inference_steps": 10},
        )
        _wait_for_video_completion(client, job.id, BASE_TIMEOUT * 2)

        # Verify stdout and file logs
        stdout = ctx.log_tail(lines=500)
        _verify_text_logs(stdout, VIDEO_PROMPT[:20])

        logs = list(Path(ctx.temp_dir).glob("*.log"))
        assert logs, "No log files found"
        _verify_text_logs("".join(f.read_text() for f in logs), VIDEO_PROMPT[:20])


class TestVideoRequestLoggerJson:
    """Test JSON-format request logging for video models."""

    def test_request_logging(self, video_json_server):
        ctx = video_json_server
        client = _create_client(ctx)

        # Video generation is async - create job and poll until completion
        job = client.videos.create(
            prompt=VIDEO_PROMPT,
            size="832x480",
            extra_body={"num_frames": 5, "num_inference_steps": 10},
        )
        _wait_for_video_completion(client, job.id, BASE_TIMEOUT * 2)

        # Verify stdout and file logs
        stdout = ctx.log_tail(lines=500)
        _verify_json_logs(stdout)

        logs = list(Path(ctx.temp_dir).glob("*.log"))
        assert logs, "No log files found"
        _verify_json_logs("".join(f.read_text() for f in logs))
