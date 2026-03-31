#!/usr/bin/env python3
"""Smoke tests for disaggregated diffusion: launch encoder/denoiser/decoder/server,
send one request, verify non-empty output.

Usage:
    python test_disagg_smoke.py                  # run all
    python test_disagg_smoke.py --model zimage   # run one
    pytest test_disagg_smoke.py -v -k wan21      # via pytest
"""

import argparse
import base64
import os
import re
import signal
import subprocess
import sys
import time

import requests

HOST = "127.0.0.1"


# ---------------------------------------------------------------------------
# Process management
# ---------------------------------------------------------------------------


def kill_tree(pid):
    try:
        os.killpg(os.getpgid(pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def wait_for_log(path, message, timeout=300):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if os.path.exists(path):
            with open(path, "r", errors="ignore") as f:
                if message in f.read():
                    return True
        time.sleep(2)
    return False


def start_proc(cmd, log_path):
    env = os.environ.copy()
    fh = open(log_path, "w")
    proc = subprocess.Popen(
        cmd,
        stdout=fh,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
        env=env,
    )
    return proc, fh


def get_actual_port(log_path):
    with open(log_path, "r", errors="ignore") as f:
        for line in f:
            if "Uvicorn running on" in line:
                m = re.search(r":(\d{4,5})\b", line)
                if m:
                    return int(m.group(1))
    return None


def tail_log(log_path, n=30):
    """Print last n lines of a log file for debugging."""
    if not os.path.exists(log_path):
        return
    with open(log_path, "r", errors="ignore") as f:
        lines = f.readlines()
    for line in lines[-n:]:
        print(f"  | {line.rstrip()}")


# ---------------------------------------------------------------------------
# DisaggCluster context manager
# ---------------------------------------------------------------------------


class DisaggCluster:
    """Launch a disaggregated diffusion cluster and tear it down on exit."""

    def __init__(
        self,
        model: str,
        name: str,
        enc_gpu: int,
        den_gpu: int,
        dec_gpu: int,
        base_port: int = 29500,
        timeout: int = 300,
    ):
        self.model = model
        self.name = name
        self.enc_gpu = enc_gpu
        self.den_gpu = den_gpu
        self.dec_gpu = dec_gpu
        self.base_port = base_port
        self.timeout = timeout
        self.api_port = base_port + 100
        self._procs = []
        self._fhs = []
        self._logs = {}

    def __enter__(self):
        bp = self.base_port
        enc_port = bp + 10
        den_port = bp + 20
        dec_port = bp + 30

        roles = [
            ("encoder", enc_port, self.enc_gpu, f"Role ENCODER ready"),
            ("denoiser", den_port, self.den_gpu, f"Role DENOISER ready"),
            ("decoder", dec_port, self.dec_gpu, f"Role DECODER ready"),
        ]

        for role, port, gpu, ready_msg in roles:
            log = f"/tmp/disagg_smoke_{self.name}_{role}.log"
            self._logs[role] = log
            p, fh = start_proc(
                [
                    "sglang",
                    "serve",
                    "--model-path",
                    self.model,
                    "--disagg-role",
                    role,
                    "--disagg-server-addr",
                    f"tcp://{HOST}:{bp}",
                    "--scheduler-port",
                    str(port),
                    "--num-gpus",
                    "1",
                    "--base-gpu-id",
                    str(gpu),
                    "--log-level",
                    "info",
                ],
                log,
            )
            self._procs.append(p)
            self._fhs.append(fh)
            if not wait_for_log(log, ready_msg, self.timeout):
                print(f"ERROR: {role} failed to start. Log tail:")
                tail_log(log)
                raise RuntimeError(f"{role} failed to start for {self.name}")
            print(f"  {role} ready", flush=True)

        # Server head
        log_s = f"/tmp/disagg_smoke_{self.name}_server.log"
        self._logs["server"] = log_s
        p, fh = start_proc(
            [
                "sglang",
                "serve",
                "--model-path",
                self.model,
                "--disagg-role",
                "server",
                "--encoder-urls",
                f"tcp://{HOST}:{enc_port}",
                "--denoiser-urls",
                f"tcp://{HOST}:{den_port}",
                "--decoder-urls",
                f"tcp://{HOST}:{dec_port}",
                "--scheduler-port",
                str(bp),
                "--port",
                str(self.api_port),
                "--host",
                HOST,
                "--log-level",
                "info",
            ],
            log_s,
        )
        self._procs.append(p)
        self._fhs.append(fh)
        if not wait_for_log(log_s, "Application startup complete", self.timeout):
            print("ERROR: server head failed to start. Log tail:")
            tail_log(log_s)
            raise RuntimeError(f"server head failed to start for {self.name}")

        actual = get_actual_port(log_s)
        if actual:
            self.api_port = actual
        print(f"  server head ready on port {self.api_port}", flush=True)
        return self

    def __exit__(self, *exc):
        for p, fh in zip(self._procs, self._fhs):
            kill_tree(p.pid)
            fh.close()
        time.sleep(3)


# ---------------------------------------------------------------------------
# Request helpers
# ---------------------------------------------------------------------------


def send_image_request(
    port,
    model,
    prompt="A sunset over mountains",
    size="1024x1024",
    num_inference_steps=4,
):
    from openai import OpenAI

    client = OpenAI(base_url=f"http://{HOST}:{port}/v1", api_key="unused")
    resp = client.images.generate(
        model=model,
        prompt=prompt,
        n=1,
        size=size,
        response_format="b64_json",
    )
    return base64.b64decode(resp.data[0].b64_json)


def send_video_request(
    port,
    model,
    prompt="A cat walking on the beach",
    size="480x832",
    num_inference_steps=2,
    poll_timeout=600,
):
    base_url = f"http://{HOST}:{port}/v1/videos"
    resp = requests.post(
        base_url,
        json={
            "model": model,
            "prompt": prompt,
            "size": size,
            "num_inference_steps": num_inference_steps,
            "seed": 42,
        },
    )
    assert (
        resp.status_code == 200
    ), f"Submit failed ({resp.status_code}): {resp.text[:200]}"

    job_id = resp.json().get("id", "")
    deadline = time.time() + poll_timeout
    while time.time() < deadline:
        time.sleep(2)
        poll = requests.get(f"{base_url}/{job_id}")
        assert poll.status_code == 200, f"Poll failed ({poll.status_code})"
        data = poll.json()
        status = data.get("status", "")
        if status == "completed":
            return data.get("file_path", "")
        elif status == "failed":
            err = data.get("error", {}).get("message", "unknown")
            raise RuntimeError(f"Video generation failed: {err}")
    raise TimeoutError(f"Video generation timed out after {poll_timeout}s")


# ---------------------------------------------------------------------------
# GPU check
# ---------------------------------------------------------------------------


def _gpu_count():
    try:
        import torch

        return torch.cuda.device_count()
    except Exception:
        return 0


def _skip_if_insufficient_gpus(n):
    count = _gpu_count()
    if count < n:
        try:
            import pytest

            pytest.skip(f"Need {n} GPUs, have {count}")
        except ImportError:
            print(f"SKIP: need {n} GPUs, have {count}")
            return False
    return True


# ---------------------------------------------------------------------------
# Test functions
# ---------------------------------------------------------------------------


def test_disagg_zimage():
    if not _skip_if_insufficient_gpus(2):
        return
    print("\n=== Z-Image-Turbo disagg smoke test ===", flush=True)
    model = "Tongyi-MAI/Z-Image-Turbo"
    with DisaggCluster(
        model, "zimage", enc_gpu=0, den_gpu=1, dec_gpu=0, base_port=29500, timeout=300
    ) as cluster:
        img = send_image_request(cluster.api_port, model)
        assert len(img) > 1000, f"Image too small: {len(img)} bytes"
        print(f"  PASS: got {len(img)} bytes image", flush=True)


def test_disagg_wan21():
    if not _skip_if_insufficient_gpus(2):
        return
    print("\n=== Wan2.1-T2V-1.3B disagg smoke test ===", flush=True)
    model = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    with DisaggCluster(
        model, "wan21", enc_gpu=0, den_gpu=1, dec_gpu=0, base_port=29600, timeout=600
    ) as cluster:
        path = send_video_request(cluster.api_port, model, num_inference_steps=2)
        assert path and os.path.exists(path), f"Video file not found: {path}"
        size = os.path.getsize(path)
        assert size > 1000, f"Video too small: {size} bytes"
        print(f"  PASS: got {size // 1024} KB video at {path}", flush=True)


def test_disagg_wan22():
    if not _skip_if_insufficient_gpus(3):
        return
    print("\n=== Wan2.2-T2V-A14B disagg smoke test ===", flush=True)
    model = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
    with DisaggCluster(
        model, "wan22", enc_gpu=0, den_gpu=1, dec_gpu=2, base_port=29700, timeout=600
    ) as cluster:
        path = send_video_request(cluster.api_port, model, num_inference_steps=2)
        assert path and os.path.exists(path), f"Video file not found: {path}"
        size = os.path.getsize(path)
        assert size > 1000, f"Video too small: {size} bytes"
        print(f"  PASS: got {size // 1024} KB video at {path}", flush=True)


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

TESTS = {
    "zimage": test_disagg_zimage,
    "wan21": test_disagg_wan21,
    "wan22": test_disagg_wan22,
}


def main():
    parser = argparse.ArgumentParser(description="Disaggregated diffusion smoke tests")
    parser.add_argument(
        "--model",
        choices=list(TESTS.keys()) + ["all"],
        default="all",
        help="Which model to test (default: all)",
    )
    args = parser.parse_args()

    selected = TESTS if args.model == "all" else {args.model: TESTS[args.model]}
    results = {}

    for name, fn in selected.items():
        try:
            fn()
            results[name] = "PASS"
        except Exception as e:
            results[name] = f"FAIL: {e}"
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 40)
    print("Results:")
    for name, result in results.items():
        print(f"  {name}: {result}")
    print("=" * 40)

    sys.exit(0 if all(v == "PASS" for v in results.values()) else 1)


if __name__ == "__main__":
    main()
