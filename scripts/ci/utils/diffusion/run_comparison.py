"""Cross-framework comparison benchmark for diffusion serving.

Launches servers (SGLang, vLLM-Omni, LightX2V) for each test case, sends a
single request and/or bench_serving traffic, then writes comparison-results.json.

Usage:
    # Full run (requires GPU)
    python3 scripts/ci/utils/diffusion/run_comparison.py

    # Dry-run (config parsing + command preview only)
    python3 scripts/ci/utils/diffusion/run_comparison.py --dry-run

    # Run only specific case(s)
    python3 scripts/ci/utils/diffusion/run_comparison.py --case-ids flux1_dev_t2i_1024

    # Run only specific framework(s)
    python3 scripts/ci/utils/diffusion/run_comparison.py --frameworks sglang

    # Run single-request E2E plus high-pressure throughput
    python3 scripts/ci/utils/diffusion/run_comparison.py --modes single_e2e throughput
"""

import argparse
import base64
import io
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CONFIGS_PATH = Path(__file__).parent / "comparison_configs.json"
INSTALL_SCRIPT = Path(__file__).parents[1] / "install_comparison_frameworks.sh"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 30000
HEALTH_TIMEOUT = (
    2400  # seconds (40 min — FLUX.2-dev needs ~10 min download + torch.compile)
)
REQUEST_TIMEOUT = 1200  # seconds
GPU_CLEAR_WAIT = 15  # seconds between framework runs
MODE_SINGLE_E2E = "single_e2e"
MODE_THROUGHPUT = "throughput"
DEFAULT_BENCHMARK = {
    "warmup": {"num_requests": 2, "num_inference_steps": 3},
    "throughput": {"num_requests": 4, "max_concurrency": 2},
}

# Frameworks that need separate installation (conflict with sglang's deps)
INSTALLABLE_FRAMEWORKS = {"vllm-omni", "lightx2v"}
FRAMEWORK_ORDER = ["sglang", "vllm-omni", "lightx2v"]

# Cached reference image (downloaded once)
_cached_ref_image: bytes | None = None
_cached_ref_image_path: str | None = None


# ---------------------------------------------------------------------------
# Server lifecycle — command builders
# ---------------------------------------------------------------------------


def _build_sglang_cmd(case: dict, fw_cfg: dict, port: int) -> list[str]:
    cmd = [
        "sglang",
        "serve",
        "--model-path",
        case["model"],
        "--port",
        str(port),
        "--host",
        DEFAULT_HOST,
    ]
    if case["num_gpus"] > 1:
        cmd += ["--num-gpus", str(case["num_gpus"])]
    if fw_cfg.get("serve_args", "").strip():
        cmd += fw_cfg["serve_args"].strip().split()
    return cmd


def _build_vllm_cmd(case: dict, fw_cfg: dict, port: int) -> list[str]:
    cmd = [
        "vllm",
        "serve",
        case["model"],
        "--omni",
        "--port",
        str(port),
        "--host",
        DEFAULT_HOST,
    ]
    if fw_cfg.get("serve_args", "").strip():
        cmd += fw_cfg["serve_args"].strip().split()
    return cmd


def _resolve_hf_model_path(model_id: str) -> str:
    """Resolve a HuggingFace model ID to a local cache path, or return as-is."""
    if os.path.isdir(model_id):
        return model_id
    try:
        from huggingface_hub import snapshot_download

        path = snapshot_download(model_id)
        print(f"  Resolved {model_id} -> {path}")
        return path
    except Exception:
        return model_id


def _write_lightx2v_config(case: dict, fw_cfg: dict) -> str:
    """Write a minimal LightX2V config JSON and return its path."""
    cfg = {
        "infer_steps": case.get("num_inference_steps", 50),
        "guidance_scale": case.get("guidance_scale", 4.0),
        "seed": case.get("seed", 42),
    }
    if "num_frames" in case:
        cfg["target_video_length"] = case["num_frames"]
    if "height" in case:
        cfg["height"] = case["height"]
        cfg["target_height"] = case["height"]
    if "width" in case:
        cfg["width"] = case["width"]
        cfg["target_width"] = case["width"]
    cfg.update(fw_cfg.get("lightx2v_config", {}))

    config_path = os.path.join(
        tempfile.gettempdir(), f"lightx2v_config_{case['id']}.json"
    )
    with open(config_path, "w") as f:
        json.dump(cfg, f)
    return config_path


def _build_lightx2v_cmd(case: dict, fw_cfg: dict, port: int) -> list[str]:
    """Build LightX2V server launch command.

    Single GPU:  python -m lightx2v.server --model_path ... --model_cls ... --task ... --port ...
    Multi GPU:   torchrun --nproc_per_node=N -m lightx2v.server ...

    LightX2V requires a local model path and a config JSON with infer params.
    """
    model_cls = fw_cfg["model_cls"]
    task = fw_cfg["lightx2v_task"]
    num_gpus = case["num_gpus"]
    model_path = _resolve_hf_model_path(case["model"])
    config_path = _write_lightx2v_config(case, fw_cfg)

    server_args = [
        "--model_path",
        model_path,
        "--model_cls",
        model_cls,
        "--task",
        task,
        "--config_json",
        config_path,
        "--host",
        DEFAULT_HOST,
        "--port",
        str(port),
    ]
    if fw_cfg.get("serve_args", "").strip():
        server_args += fw_cfg["serve_args"].strip().split()

    if num_gpus > 1:
        cmd = [
            "torchrun",
            f"--nproc_per_node={num_gpus}",
            "-m",
            "lightx2v.server",
        ] + server_args
    else:
        cmd = ["python3", "-m", "lightx2v.server"] + server_args

    return cmd


def build_server_cmd(framework: str, case: dict, fw_cfg: dict, port: int) -> list[str]:
    builders = {
        "sglang": _build_sglang_cmd,
        "vllm-omni": _build_vllm_cmd,
        "lightx2v": _build_lightx2v_cmd,
    }
    builder = builders.get(framework)
    if builder is None:
        raise ValueError(f"Unknown framework: {framework}")
    return builder(case, fw_cfg, port)


# ---------------------------------------------------------------------------
# Server lifecycle — health check & cleanup
# ---------------------------------------------------------------------------

# Health check endpoints per framework
HEALTH_ENDPOINTS = {
    "sglang": "/health",
    "vllm-omni": "/health",
    "lightx2v": "/v1/service/status",
}


def wait_for_health(
    base_url: str, framework: str = "sglang", timeout: int = HEALTH_TIMEOUT
) -> None:
    """Poll health endpoint until 200, then verify model is loaded."""
    endpoint = HEALTH_ENDPOINTS.get(framework, "/health")
    health_url = f"{base_url}{endpoint}"
    print(f"  Waiting for server at {health_url} ...")
    start = time.time()
    while True:
        try:
            resp = requests.get(health_url, timeout=2)
            if resp.status_code == 200:
                break
        except requests.exceptions.RequestException:
            pass
        if time.time() - start > timeout:
            raise TimeoutError(
                f"Server at {health_url} did not start within {timeout}s"
            )
        time.sleep(2)

    # For SGLang, /health can return 200 before model routes are registered.
    # Poll /v1/models to confirm the model is fully loaded.
    if framework == "sglang":
        models_url = f"{base_url}/v1/models"
        while True:
            try:
                resp = requests.get(models_url, timeout=5)
                if resp.status_code == 200:
                    break
            except requests.exceptions.RequestException:
                pass
            if time.time() - start > timeout:
                raise TimeoutError(f"Model at {models_url} not ready within {timeout}s")
            time.sleep(2)

    elapsed = time.time() - start
    print(f"  Server ready in {elapsed:.1f}s")


KILLALL_SCRIPT = Path(__file__).parents[3] / "killall_sglang.sh"


def kill_server(proc: subprocess.Popen) -> None:
    """Kill server process tree and clean up GPU processes."""
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        pass
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        proc.wait(timeout=10)
    # Use killall_sglang.sh for thorough cleanup (esp. multi-GPU workers)
    if KILLALL_SCRIPT.exists():
        subprocess.run(
            ["bash", str(KILLALL_SCRIPT)],
            timeout=30,
            capture_output=True,
        )


# ---------------------------------------------------------------------------
# Reference image helpers
# ---------------------------------------------------------------------------


def _get_ref_image_bytes(config: dict) -> bytes:
    """Download and cache the shared test reference image."""
    global _cached_ref_image
    if _cached_ref_image is not None:
        return _cached_ref_image
    url = config.get("test_image_url", "")
    if not url:
        raise RuntimeError("No test_image_url in config for image-conditioned case")
    print(f"  Downloading reference image from {url} ...")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    _cached_ref_image = resp.content
    return _cached_ref_image


def _get_ref_image_b64(config: dict) -> str:
    """Get reference image as base64 string."""
    return base64.b64encode(_get_ref_image_bytes(config)).decode("utf-8")


def _get_ref_image_path(config: dict) -> str:
    """Save reference image to a temp file and return path."""
    global _cached_ref_image_path
    if _cached_ref_image_path and os.path.exists(_cached_ref_image_path):
        return _cached_ref_image_path
    data = _get_ref_image_bytes(config)
    fd, path = tempfile.mkstemp(suffix=".png")
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    _cached_ref_image_path = path
    return path


# ---------------------------------------------------------------------------
# Request helpers — SGLang (OpenAI-compatible)
# ---------------------------------------------------------------------------


def _build_sglang_payload(case: dict) -> dict:
    """Build common SGLang request payload."""
    payload = {
        "model": case["model"],
        "prompt": case["prompt"],
        "size": f"{case['width']}x{case['height']}",
        "n": 1,
        "response_format": "b64_json",
    }
    for key in (
        "num_inference_steps",
        "guidance_scale",
        "seed",
        "num_frames",
        "fps",
        "negative_prompt",
    ):
        if key in case:
            payload[key] = case[key]
    return payload


def _read_perf_dump(perf_dump_path: str, timeout: float = 10.0) -> float | None:
    """Read total_duration_ms from a perf dump JSON written by the server.

    The server writes the file asynchronously after the HTTP response,
    so we poll briefly.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with open(perf_dump_path) as f:
                data = json.load(f)
            total_ms = data.get("total_duration_ms")
            if total_ms is not None:
                return total_ms / 1000.0
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        time.sleep(0.5)
    return None


def send_image_request_sglang(
    base_url: str, case: dict, perf_dump_path: str | None = None
) -> float:
    """Send a single T2I request via SGLang's /v1/images/generations."""
    payload = _build_sglang_payload(case)
    if perf_dump_path:
        payload["perf_dump_path"] = perf_dump_path

    start = time.time()
    resp = requests.post(
        f"{base_url}/v1/images/generations",
        json=payload,
        timeout=REQUEST_TIMEOUT,
    )
    client_latency = time.time() - start
    resp.raise_for_status()
    data = resp.json()
    if "data" not in data or len(data["data"]) == 0:
        raise RuntimeError(f"Image request returned no data: {data}")

    if perf_dump_path:
        server_latency = _read_perf_dump(perf_dump_path)
        if server_latency is not None:
            print(
                f"  Image generated in {server_latency:.2f}s (server-side), "
                f"client={client_latency:.2f}s"
            )
            return server_latency
    print(f"  Image generated in {client_latency:.2f}s")
    return client_latency


def send_video_request_sglang(
    base_url: str, case: dict, perf_dump_path: str | None = None
) -> float:
    """Send a single T2V request via SGLang's /v1/videos (async)."""
    payload = _build_sglang_payload(case)
    if perf_dump_path:
        payload["perf_dump_path"] = perf_dump_path

    start = time.time()

    # Submit job
    resp = requests.post(
        f"{base_url}/v1/videos",
        json=payload,
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    job = resp.json()
    job_id = job.get("id")
    if not job_id:
        raise RuntimeError(f"Video submit returned no job id: {job}")

    # Poll for completion
    poll_url = f"{base_url}/v1/videos/{job_id}"
    while True:
        time.sleep(1)
        poll_resp = requests.get(poll_url, timeout=30)
        poll_resp.raise_for_status()
        poll_data = poll_resp.json()
        status = poll_data.get("status")
        if status == "completed":
            break
        elif status == "failed":
            raise RuntimeError(f"Video generation failed: {poll_data}")
        if time.time() - start > REQUEST_TIMEOUT:
            raise TimeoutError(f"Video generation timed out after {REQUEST_TIMEOUT}s")

    client_latency = time.time() - start

    if perf_dump_path:
        server_latency = _read_perf_dump(perf_dump_path)
        if server_latency is not None:
            print(
                f"  Video generated in {server_latency:.2f}s (server-side), "
                f"client={client_latency:.2f}s"
            )
            return server_latency
    print(f"  Video generated in {client_latency:.2f}s")
    return client_latency


def send_image_conditioned_request_sglang(
    base_url: str, case: dict, config: dict, perf_dump_path: str | None = None
) -> float:
    """Send an image-conditioned request (edit/I2V/TI2V) via SGLang multipart API."""
    task = case["task"]
    ref_bytes = _get_ref_image_bytes(config)

    # Build multipart form — field name depends on endpoint:
    # image edits use "image", video (I2V/TI2V) uses "input_reference"
    if task in ("image-to-video", "text-image-to-video"):
        file_field = "input_reference"
    else:
        file_field = "image"
    files = {file_field: ("ref.png", io.BytesIO(ref_bytes), "image/png")}
    data = {
        "model": case["model"],
        "prompt": case["prompt"],
        "size": f"{case['width']}x{case['height']}",
        "n": "1",
        "response_format": "b64_json",
    }
    for key in (
        "num_inference_steps",
        "guidance_scale",
        "seed",
        "num_frames",
        "fps",
        "negative_prompt",
    ):
        if key in case:
            data[key] = str(case[key])
    if perf_dump_path:
        data["perf_dump_path"] = perf_dump_path
    # Choose endpoint based on task
    if task in ("image-edit", "image-to-image"):
        endpoint = "/v1/images/edits"
    elif task in ("image-to-video", "text-image-to-video"):
        endpoint = "/v1/videos"
    else:
        endpoint = "/v1/images/generations"

    start = time.time()
    resp = requests.post(
        f"{base_url}{endpoint}",
        files=files,
        data=data,
        timeout=REQUEST_TIMEOUT,
    )

    # For video endpoints, need to poll
    if task in ("image-to-video", "text-image-to-video"):
        resp.raise_for_status()
        job = resp.json()
        job_id = job.get("id")
        if not job_id:
            raise RuntimeError(f"Video submit returned no job id: {job}")
        poll_url = f"{base_url}/v1/videos/{job_id}"
        while True:
            time.sleep(1)
            poll_resp = requests.get(poll_url, timeout=30)
            poll_resp.raise_for_status()
            poll_data = poll_resp.json()
            status = poll_data.get("status")
            if status == "completed":
                break
            elif status == "failed":
                raise RuntimeError(f"Video generation failed: {poll_data}")
            if time.time() - start > REQUEST_TIMEOUT:
                raise TimeoutError(f"Timed out after {REQUEST_TIMEOUT}s")
    else:
        resp.raise_for_status()

    client_latency = time.time() - start

    if perf_dump_path:
        server_latency = _read_perf_dump(perf_dump_path)
        if server_latency is not None:
            print(
                f"  Generated in {server_latency:.2f}s (server-side), "
                f"client={client_latency:.2f}s"
            )
            return server_latency
    print(f"  Generated in {client_latency:.2f}s (sglang, image-conditioned)")
    return client_latency


# ---------------------------------------------------------------------------
# Request helpers — vLLM-Omni
# ---------------------------------------------------------------------------


def send_request_vllm_omni(base_url: str, case: dict, config: dict) -> float:
    """Send request via vLLM-Omni's OpenAI-compatible diffusion endpoints."""
    task = case["task"]
    if task in ("text-to-video", "image-to-video", "text-image-to-video"):
        data = {
            "prompt": case["prompt"],
            "size": f"{case['width']}x{case['height']}",
            "width": str(case["width"]),
            "height": str(case["height"]),
        }
        for key in (
            "num_inference_steps",
            "guidance_scale",
            "guidance_scale_2",
            "seed",
            "num_frames",
            "fps",
            "negative_prompt",
        ):
            if key in case:
                data[key] = str(case[key])
        files = None
        if case.get("reference_image"):
            files = {
                "input_reference": (
                    "ref.png",
                    io.BytesIO(_get_ref_image_bytes(config)),
                    "image/png",
                )
            }

        start = time.time()
        resp = requests.post(
            f"{base_url}/v1/videos",
            data=data,
            files=files,
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        job = resp.json()
        job_id = job.get("id")
        if not job_id:
            raise RuntimeError(f"vLLM-Omni video submit returned no id: {job}")
        poll_url = f"{base_url}/v1/videos/{job_id}"
        while True:
            time.sleep(1)
            poll_resp = requests.get(poll_url, timeout=30)
            poll_resp.raise_for_status()
            poll_data = poll_resp.json()
            status = poll_data.get("status")
            if status == "completed":
                break
            if status == "failed":
                raise RuntimeError(f"vLLM-Omni video generation failed: {poll_data}")
            if time.time() - start > REQUEST_TIMEOUT:
                raise TimeoutError(
                    f"vLLM-Omni video timed out after {REQUEST_TIMEOUT}s"
                )
        latency = time.time() - start
        print(f"  Generated in {latency:.2f}s (vllm-omni)")
        return latency

    if task == "text-to-image":
        payload = _build_sglang_payload(case)
        start = time.time()
        resp = requests.post(
            f"{base_url}/v1/images/generations",
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )
        latency = time.time() - start
        resp.raise_for_status()
        return latency
    if task in ("image-edit", "image-to-image"):
        files = {
            "image": ("ref.png", io.BytesIO(_get_ref_image_bytes(config)), "image/png")
        }
        data = {
            "model": case["model"],
            "prompt": case["prompt"],
            "size": f"{case['width']}x{case['height']}",
            "n": "1",
            "response_format": "b64_json",
        }
        for key in (
            "num_inference_steps",
            "guidance_scale",
            "seed",
            "negative_prompt",
        ):
            if key in case:
                data[key] = str(case[key])
        start = time.time()
        resp = requests.post(
            f"{base_url}/v1/images/edits",
            files=files,
            data=data,
            timeout=REQUEST_TIMEOUT,
        )
        latency = time.time() - start
        resp.raise_for_status()
        return latency

    extra_body = {
        "height": case["height"],
        "width": case["width"],
        "num_inference_steps": case.get("num_inference_steps", 50),
        "guidance_scale": case.get("guidance_scale", 4.0),
        "seed": case.get("seed", 42),
    }
    if "num_frames" in case:
        extra_body["num_frames"] = case["num_frames"]
    if "fps" in case:
        extra_body["fps"] = case["fps"]
    if "negative_prompt" in case:
        extra_body["negative_prompt"] = case["negative_prompt"]

    # Build message content (text or text+image)
    content: list[dict] | str = case["prompt"]
    if case.get("reference_image"):
        ref_b64 = _get_ref_image_b64(config)
        content = [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{ref_b64}"},
            },
            {"type": "text", "text": case["prompt"]},
        ]

    payload = {
        "model": case["model"],
        "messages": [{"role": "user", "content": content}],
        "extra_body": extra_body,
    }

    start = time.time()
    resp = requests.post(
        f"{base_url}/v1/chat/completions",
        json=payload,
        timeout=REQUEST_TIMEOUT,
    )
    latency = time.time() - start
    resp.raise_for_status()
    data = resp.json()
    choices = data.get("choices", [])
    if not choices:
        raise RuntimeError(f"vLLM-Omni request returned no choices: {data}")
    print(f"  Generated in {latency:.2f}s (vllm-omni)")
    return latency


# ---------------------------------------------------------------------------
# Request helpers — LightX2V
# ---------------------------------------------------------------------------


def send_request_lightx2v(base_url: str, case: dict, config: dict) -> float:
    """Send request via LightX2V's async task API."""
    endpoint = "/v1/tasks/"

    payload = {
        "prompt": case["prompt"],
        "seed": case.get("seed", 42),
        "infer_steps": case.get("num_inference_steps", 50),
    }
    # LightX2V uses target_video_length for frames, height/width directly
    if "num_frames" in case:
        payload["target_video_length"] = case["num_frames"]
    if "height" in case:
        payload["height"] = case["height"]
    if "width" in case:
        payload["width"] = case["width"]
    if "height" in case and "width" in case:
        payload["target_shape"] = [case["height"], case["width"]]
    if "guidance_scale" in case:
        payload["guidance_scale"] = case["guidance_scale"]
    if "fps" in case:
        payload["fps"] = case["fps"]
    if "negative_prompt" in case:
        payload["negative_prompt"] = case["negative_prompt"]
    if case.get("reference_image"):
        payload["image_path"] = _get_ref_image_path(config)

    start = time.time()

    # Submit task
    resp = requests.post(
        f"{base_url}{endpoint}",
        json=payload,
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    task_data = resp.json()
    task_id = task_data.get("task_id")
    if not task_id:
        raise RuntimeError(f"LightX2V submit returned no task_id: {task_data}")

    # Poll for completion
    poll_url = f"{base_url}/v1/tasks/{task_id}/status"
    while True:
        time.sleep(1)
        poll_resp = requests.get(poll_url, timeout=30)
        poll_resp.raise_for_status()
        poll_data = poll_resp.json()
        status = (
            poll_data.get("task_status") or poll_data.get("status") or ""
        ).upper()
        if status == "COMPLETED":
            break
        elif status in ("FAILED", "CANCELLED"):
            raise RuntimeError(f"LightX2V task {status}: {poll_data}")
        if time.time() - start > REQUEST_TIMEOUT:
            raise TimeoutError(f"LightX2V task timed out after {REQUEST_TIMEOUT}s")

    latency = time.time() - start
    print(f"  Generated in {latency:.2f}s (lightx2v)")
    return latency


# ---------------------------------------------------------------------------
# Unified request dispatcher
# ---------------------------------------------------------------------------


def send_request(
    base_url: str,
    case: dict,
    framework: str = "sglang",
    config: dict | None = None,
    perf_dump_path: str | None = None,
) -> float:
    config = config or {}
    if framework == "vllm-omni":
        return send_request_vllm_omni(base_url, case, config)
    elif framework == "lightx2v":
        return send_request_lightx2v(base_url, case, config)
    # SGLang — use OpenAI-compatible endpoints with optional perf log
    task = case["task"]
    if case.get("reference_image"):
        return send_image_conditioned_request_sglang(
            base_url, case, config, perf_dump_path
        )
    elif task == "text-to-image":
        return send_image_request_sglang(base_url, case, perf_dump_path)
    elif task == "text-to-video":
        return send_video_request_sglang(base_url, case, perf_dump_path)
    else:
        raise ValueError(f"Unknown task type: {task}")


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


def _merge_nested(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_nested(merged[key], value)
        else:
            merged[key] = value
    return merged


def _benchmark_config(config: dict | None, case: dict) -> dict:
    cfg = _merge_nested(DEFAULT_BENCHMARK, (config or {}).get("benchmark_defaults", {}))
    return _merge_nested(cfg, case.get("benchmark", {}))


def _base_result(case: dict, framework: str, mode: str) -> dict:
    return {
        "case_id": case["id"],
        "framework": framework,
        "mode": mode,
        "model": case["model"],
        "task": case["task"],
        "width": case.get("width"),
        "height": case.get("height"),
        "num_frames": case.get("num_frames"),
        "num_gpus": case.get("num_gpus"),
        "latency_s": None,
        "error": None,
    }


def _case_for_framework(case: dict, fw_cfg: dict) -> dict:
    overrides = {key: fw_cfg[key] for key in ("model", "num_gpus") if key in fw_cfg}
    if not overrides:
        return case
    return {**case, **overrides}


def _current_commit_sha() -> str:
    try:
        ret = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if ret.returncode == 0:
            return ret.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return os.environ.get("GITHUB_SHA", "unknown")


def _collect_hardware_metadata() -> dict:
    metadata = {
        "runner_labels": os.environ.get("RUNNER_LABELS"),
        "gpu_config": os.environ.get("GPU_CONFIG"),
    }
    try:
        query = "name,memory.total,driver_version"
        ret = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={query}",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if ret.returncode == 0:
            metadata["gpus"] = [
                line.strip() for line in ret.stdout.splitlines() if line.strip()
            ]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return metadata


def _run_warmups(
    base_url: str,
    case: dict,
    framework: str,
    config: dict | None,
    bench_cfg: dict,
) -> None:
    warmup_cfg = bench_cfg.get("warmup", {})
    warmup_requests = int(warmup_cfg.get("num_requests", 0) or 0)
    warmup_steps = warmup_cfg.get("num_inference_steps")
    if warmup_requests <= 0:
        return
    warmup_case = dict(case)
    if warmup_steps is not None:
        warmup_case["num_inference_steps"] = int(warmup_steps)
    for wi in range(1, warmup_requests + 1):
        print(f"  Sending warmup request ({wi}/{warmup_requests})...")
        try:
            send_request(base_url, warmup_case, framework, config)
        except Exception as e:
            print(f"  Warmup request {wi} failed (non-fatal): {e}")


def run_single_request(
    base_url: str,
    case: dict,
    framework: str,
    log_dir: Path,
    config: dict | None = None,
) -> dict:
    result = _base_result(case, framework, MODE_SINGLE_E2E)

    perf_dump_path = None
    if framework == "sglang":
        perf_dump_path = os.path.join(str(log_dir), f"perf_{case['id']}_measured.json")
    if perf_dump_path and os.path.exists(perf_dump_path):
        os.remove(perf_dump_path)
    print("  Sending measured single request...")
    latency = send_request(
        base_url, case, framework, config, perf_dump_path=perf_dump_path
    )
    result["latency_s"] = round(latency, 3)
    return result


def _bench_serving_task(task: str) -> str:
    return {
        "image-edit": "image-to-image",
        "text-image-to-video": "image-to-video",
    }.get(task, task)


def _bench_extra_body(case: dict) -> dict:
    extra_body = {}
    for key in (
        "guidance_scale",
        "guidance_scale_2",
        "true_cfg_scale",
        "negative_prompt",
        "seed",
    ):
        if key in case:
            extra_body[key] = case[key]
    return extra_body


def run_throughput(
    base_url: str,
    case: dict,
    framework: str,
    config: dict | None,
    bench_cfg: dict,
    log_dir: Path,
) -> dict:
    result = _base_result(case, framework, MODE_THROUGHPUT)
    throughput_cfg = bench_cfg.get("throughput", {})
    num_requests = int(throughput_cfg.get("num_requests", 4) or 4)
    max_concurrency = int(throughput_cfg.get("max_concurrency", 2) or 2)
    max_concurrency = max(1, min(max_concurrency, num_requests))

    metrics_path = log_dir / f"bench_serving_{case['id']}_{framework}.json"
    cmd = [
        sys.executable,
        "-m",
        "sglang.multimodal_gen.benchmarks.bench_serving",
        "--backend",
        framework,
        "--base-url",
        base_url,
        "--dataset",
        "fixed",
        "--prompt",
        case["prompt"],
        "--model",
        case["model"],
        "--task",
        _bench_serving_task(case["task"]),
        "--num-prompts",
        str(num_requests),
        "--max-concurrency",
        str(max_concurrency),
        "--request-rate",
        str(throughput_cfg.get("request_rate", "inf")),
        "--output-file",
        str(metrics_path),
        "--disable-tqdm",
    ]
    for key in ("width", "height", "num_frames", "fps", "num_inference_steps"):
        if key in case:
            cmd.extend([f"--{key.replace('_', '-')}", str(case[key])])
    extra_body = _bench_extra_body(case)
    if extra_body:
        cmd.extend(["--extra-body", json.dumps(extra_body)])
    if case.get("reference_image"):
        cmd.extend(["--image-path", _get_ref_image_path(config or {})])

    print(
        f"  Running bench_serving throughput: requests={num_requests}, "
        f"max_concurrency={max_concurrency}"
    )
    ret = subprocess.run(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=REQUEST_TIMEOUT * max(1, num_requests),
    )
    bench_log_path = log_dir / f"bench_serving_{case['id']}_{framework}.log"
    with open(bench_log_path, "w", encoding="utf-8") as f:
        f.write(ret.stdout)
    if ret.returncode != 0:
        result["error"] = f"bench_serving failed (exit {ret.returncode})"
        return result

    with open(metrics_path) as f:
        metrics = json.load(f)

    result.update(
        {
            "latency_s": round(metrics.get("latency_p50", 0), 3),
            "metrics": {
                "duration_s": round(metrics.get("duration", 0), 3),
                "num_requests": metrics.get("completed_requests", 0)
                + metrics.get("failed_requests", 0),
                "completed_requests": metrics.get("completed_requests", 0),
                "failed_requests": metrics.get("failed_requests", 0),
                "max_concurrency": max_concurrency,
                "throughput_rps": round(metrics.get("throughput_qps", 0), 4),
                "output_throughput_ops": round(
                    metrics.get("output_throughput_ops", 0), 4
                ),
                "latency_mean_s": round(metrics.get("latency_mean", 0), 3),
                "latency_p50_s": round(metrics.get("latency_p50", 0), 3),
                "latency_p90_s": round(metrics.get("latency_p90", 0), 3),
                "latency_p95_s": round(metrics.get("latency_p95", 0), 3),
                "latency_p99_s": round(metrics.get("latency_p99", 0), 3),
            },
        }
    )
    return result


def run_case_framework(
    case: dict,
    framework: str,
    fw_cfg: dict,
    modes: list[str],
    port: int,
    log_dir: Path,
    config: dict | None = None,
) -> tuple[dict | None, dict | None]:
    """Run one server lifecycle and collect requested benchmark modes."""
    case = _case_for_framework(case, fw_cfg)
    single_result = None
    throughput_result = None
    cmd = build_server_cmd(framework, case, fw_cfg, port)
    print(f"\n  Command: {' '.join(cmd)}")

    env = os.environ.copy()
    env.update(fw_cfg.get("extra_env", {}))
    env = _framework_env(framework, env)

    log_file = log_dir / f"{case['id']}_{framework}.log"
    log_fh = open(log_file, "w", encoding="utf-8", buffering=1)
    log_thread = None

    proc = None
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            preexec_fn=os.setsid,
            text=True,
            bufsize=1,
        )

        # Tee server output to both log file and stdout (like test_server_utils)
        def _log_pipe(pipe, fh):
            try:
                for line in iter(pipe.readline, ""):
                    sys.stdout.write(f"  [server] {line}")
                    sys.stdout.flush()
                    fh.write(line)
            except ValueError:
                pass  # pipe closed

        log_thread = threading.Thread(target=_log_pipe, args=(proc.stdout, log_fh))
        log_thread.daemon = True
        log_thread.start()

        base_url = f"http://{DEFAULT_HOST}:{port}"
        wait_for_health(base_url, framework)
        bench_cfg = _merge_nested(
            _benchmark_config(config, case), fw_cfg.get("benchmark", {})
        )
        _run_warmups(base_url, case, framework, config, bench_cfg)

        if MODE_SINGLE_E2E in modes:
            single_result = run_single_request(
                base_url, case, framework, log_dir, config
            )
        if MODE_THROUGHPUT in modes:
            throughput_result = run_throughput(
                base_url, case, framework, config, bench_cfg, log_dir
            )

    except Exception as e:
        print(f"  ERROR: {e}")
        if MODE_SINGLE_E2E in modes:
            single_result = _base_result(case, framework, MODE_SINGLE_E2E)
            single_result["error"] = str(e)
        if MODE_THROUGHPUT in modes:
            throughput_result = _base_result(case, framework, MODE_THROUGHPUT)
            throughput_result["error"] = str(e)
    finally:
        if proc:
            kill_server(proc)
        if log_thread:
            log_thread.join(timeout=5)
        log_fh.close()

    return single_result, throughput_result


def _install_framework(fw_name: str, dry_run: bool = False) -> bool:
    """Install a comparison framework via the install script. Returns True on success."""
    if fw_name not in INSTALLABLE_FRAMEWORKS:
        return True
    if not INSTALL_SCRIPT.exists():
        print(f"  WARNING: Install script not found at {INSTALL_SCRIPT}")
        return False
    if dry_run:
        print(f"  [DRY-RUN] Would install: bash {INSTALL_SCRIPT} {fw_name}")
        return True
    print(f"\n{'='*60}")
    print(f"Installing framework: {fw_name}")
    print(f"{'='*60}")
    ret = subprocess.run(
        ["bash", str(INSTALL_SCRIPT), fw_name],
        timeout=600,
    )
    if ret.returncode != 0:
        print(f"  WARNING: {fw_name} installation failed (exit {ret.returncode})")
        return False
    return True


def _framework_venv_path(fw_name: str) -> str:
    root = os.environ.get(
        "SGLANG_DIFFUSION_FRAMEWORK_VENV_ROOT",
        "/tmp/sglang-diffusion-framework-venvs",
    )
    return os.path.join(root, fw_name)


def _framework_env(fw_name: str, env: dict[str, str]) -> dict[str, str]:
    if fw_name not in INSTALLABLE_FRAMEWORKS:
        return env
    venv_path = _framework_venv_path(fw_name)
    bin_path = os.path.join(venv_path, "bin")
    framework_env = dict(env)
    framework_env["VIRTUAL_ENV"] = venv_path
    framework_env["PATH"] = f"{bin_path}:{framework_env.get('PATH', '')}"
    return framework_env


def run_comparison(
    config: dict,
    case_ids: list[str] | None = None,
    frameworks: list[str] | None = None,
    modes: list[str] | None = None,
    port: int = DEFAULT_PORT,
    output: str = "comparison-results.json",
    dry_run: bool = False,
) -> dict:
    """Run all comparison cases, grouped by framework to minimize installs.

    Order: sglang first (already installed), then vllm-omni, then lightx2v.
    Each non-sglang framework is installed right before its cases run.
    """
    timestamp = datetime.now(timezone.utc).isoformat()
    commit_sha = _current_commit_sha()
    run_id = os.environ.get("GITHUB_RUN_ID", "local")

    log_dir = Path("comparison-logs")
    log_dir.mkdir(exist_ok=True)

    modes = modes or [MODE_SINGLE_E2E]
    invalid_modes = sorted(set(modes) - {MODE_SINGLE_E2E, MODE_THROUGHPUT})
    if invalid_modes:
        raise ValueError(f"Unknown benchmark mode(s): {invalid_modes}")

    fw_cases: dict[str, list[tuple[dict, dict]]] = {fw: [] for fw in FRAMEWORK_ORDER}

    for case in config["cases"]:
        if case_ids and case["id"] not in case_ids:
            continue
        for fw_name, fw_cfg in case["frameworks"].items():
            if frameworks and fw_name not in frameworks:
                continue
            if fw_name not in fw_cases:
                fw_cases[fw_name] = []
            fw_cases[fw_name].append((case, fw_cfg))

    results = []
    throughput_results = []
    installed_fws: set[str] = set()

    for fw_name in FRAMEWORK_ORDER:
        pairs = fw_cases.get(fw_name, [])
        if not pairs:
            continue

        # Install framework if needed (once per framework)
        if fw_name not in installed_fws and fw_name in INSTALLABLE_FRAMEWORKS:
            if not _install_framework(fw_name, dry_run):
                # Skip all cases for this framework
                for case, pair_fw_cfg in pairs:
                    case_for_fw = _case_for_framework(case, pair_fw_cfg)
                    if MODE_SINGLE_E2E in modes:
                        result = _base_result(case_for_fw, fw_name, MODE_SINGLE_E2E)
                        result["error"] = f"{fw_name} installation failed"
                        results.append(result)
                    if MODE_THROUGHPUT in modes:
                        result = _base_result(case_for_fw, fw_name, MODE_THROUGHPUT)
                        result["error"] = f"{fw_name} installation failed"
                        throughput_results.append(result)
                continue
            installed_fws.add(fw_name)

        for case, fw_cfg in pairs:
            print(f"\n{'='*60}")
            print(f"Case: {case['id']} | Model: {case['model']} | Framework: {fw_name}")
            print(f"{'='*60}")

            if dry_run:
                case_for_fw = _case_for_framework(case, fw_cfg)
                cmd = build_server_cmd(fw_name, case_for_fw, fw_cfg, port)
                print(f"  [DRY-RUN] Would run: {' '.join(cmd)}")
                if fw_name in INSTALLABLE_FRAMEWORKS:
                    print(f"  [DRY-RUN] venv: {_framework_venv_path(fw_name)}")
                if MODE_SINGLE_E2E in modes:
                    result = _base_result(case_for_fw, fw_name, MODE_SINGLE_E2E)
                    result["error"] = "dry-run"
                    results.append(result)
                if MODE_THROUGHPUT in modes:
                    result = _base_result(case_for_fw, fw_name, MODE_THROUGHPUT)
                    result["error"] = "dry-run"
                    throughput_results.append(result)
                continue

            single_result, throughput_result = run_case_framework(
                case, fw_name, fw_cfg, modes, port, log_dir, config
            )
            if single_result is not None:
                results.append(single_result)
            if throughput_result is not None:
                throughput_results.append(throughput_result)

            # Wait for GPU memory to clear
            print(f"  Waiting {GPU_CLEAR_WAIT}s for GPU memory to clear...")
            time.sleep(GPU_CLEAR_WAIT)

    output_data = {
        "timestamp": timestamp,
        "commit_sha": commit_sha,
        "run_id": run_id,
        "hardware": _collect_hardware_metadata(),
        "benchmark_modes": modes,
        "results": results,
        "throughput_results": throughput_results,
    }

    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with open(output, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults written to {output}")

    # Print summary table
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for r in results:
        lat = f"{r['latency_s']:.2f}s" if r["latency_s"] else r.get("error", "N/A")
        print(f"  {r['case_id']:30s} | {r['framework']:12s} | {lat}")
    if throughput_results:
        print("\nTHROUGHPUT")
        for r in throughput_results:
            metrics = r.get("metrics", {})
            throughput = metrics.get("throughput_rps")
            value = (
                f"{throughput:.4f} req/s"
                if isinstance(throughput, (float, int))
                else r.get("error", "N/A")
            )
            print(f"  {r['case_id']:30s} | {r['framework']:12s} | {value}")

    return output_data


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Cross-framework diffusion serving comparison benchmark"
    )
    parser.add_argument(
        "--config",
        default=str(CONFIGS_PATH),
        help="Path to comparison_configs.json",
    )
    parser.add_argument(
        "--case-ids",
        nargs="+",
        default=None,
        help="Only run specific case IDs",
    )
    parser.add_argument(
        "--frameworks",
        nargs="+",
        default=None,
        help="Only run specific frameworks (sglang, vllm-omni, lightx2v)",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=[MODE_SINGLE_E2E],
        choices=[MODE_SINGLE_E2E, MODE_THROUGHPUT],
        help="Benchmark modes to run",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help="Server port",
    )
    parser.add_argument(
        "--output",
        default="comparison-results.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse config and print commands without launching servers",
    )

    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    print(f"Loaded {len(config['cases'])} comparison case(s) from {args.config}")

    output_data = run_comparison(
        config=config,
        case_ids=args.case_ids,
        frameworks=args.frameworks,
        modes=args.modes,
        port=args.port,
        output=args.output,
        dry_run=args.dry_run,
    )

    # Exit with non-zero if any case had an error
    errors = [
        r
        for r in output_data.get("results", [])
        + output_data.get("throughput_results", [])
        if r.get("error")
    ]
    if errors and not args.dry_run:
        print(f"\n{len(errors)} case(s) had errors:")
        for e in errors:
            print(f"  {e['case_id']} ({e['framework']}): {e['error']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
