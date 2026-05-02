#!/usr/bin/env python3
"""Optional local-server smoke for RelayKV runtime observation.

This smoke is intentionally opt-in. It only runs when a local model path is
provided and server launch is explicitly enabled, so regular smoke runs do not
start a server or trigger model downloads.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

os.environ.setdefault("FLASHINFER_WORKSPACE_BASE", "/tmp/relaykv_flashinfer_cache")

MODEL_ENV = "RELAYKV_OPTIONAL_SERVER_SMOKE_MODEL"
RUN_ENV = "RELAYKV_OPTIONAL_SERVER_SMOKE_RUN"
TIMEOUT_ENV = "RELAYKV_OPTIONAL_SERVER_SMOKE_TIMEOUT"
OBSERVATION_ENV = "SGLANG_RELAYKV_RUNTIME_OBSERVATION"


def _print_result(result: dict[str, Any]) -> None:
    print(
        "relaykv_optional_server_observation_smoke_result="
        + json.dumps(result, sort_keys=True)
    )


def _clean_skip(reason: str, **extra: Any) -> int:
    result = {"skipped": True, "skip_reason": reason}
    result.update(extra)
    print("relaykv_optional_server_observation_smoke: skip")
    _print_result(result)
    return 0


def _reserve_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _request_json(
    url: str,
    payload: dict[str, Any] | None = None,
    timeout: float = 5.0,
) -> tuple[int, Any]:
    data = None
    method = "GET"
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        method = "POST"
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read()
        if not body:
            return response.status, None
        return response.status, json.loads(body.decode("utf-8"))


def _wait_for_health(proc: subprocess.Popen[str], base_url: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    last_error = "server did not become healthy"
    while time.monotonic() < deadline:
        return_code = proc.poll()
        if return_code is not None:
            raise RuntimeError(f"server exited before health check: {return_code}")
        try:
            status, _ = _request_json(f"{base_url}/health_generate", timeout=2.0)
            if status == 200:
                return
        except (OSError, urllib.error.URLError, TimeoutError) as exc:
            last_error = str(exc)
        time.sleep(1)
    raise TimeoutError(last_error)


def _terminate_process(proc: subprocess.Popen[str]) -> str:
    if proc.poll() is None:
        proc.terminate()
        try:
            stdout, _ = proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, _ = proc.communicate(timeout=10)
    else:
        stdout, _ = proc.communicate(timeout=10)
    return stdout or ""


def _tail(text: str, max_lines: int = 80) -> str:
    return "\n".join(text.splitlines()[-max_lines:])


def _run_server_case(model_path: str, observation_value: str, timeout: float) -> dict[str, Any]:
    port = _reserve_local_port()
    base_url = f"http://127.0.0.1:{port}"
    env = os.environ.copy()
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env[OBSERVATION_ENV] = observation_value
    env.setdefault("FLASHINFER_WORKSPACE_BASE", "/tmp/relaykv_flashinfer_cache")

    repo_python = str(Path(__file__).resolve().parents[1] / "python")
    if env.get("PYTHONPATH"):
        env["PYTHONPATH"] = repo_python + os.pathsep + env["PYTHONPATH"]
    else:
        env["PYTHONPATH"] = repo_python

    cmd = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        model_path,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        text=True,
        bufsize=1,
    )
    stdout = ""
    try:
        _wait_for_health(proc, base_url, timeout)
        payload = {
            "text": "RelayKV observation smoke.",
            "sampling_params": {"max_new_tokens": 1, "temperature": 0},
        }
        status, response = _request_json(f"{base_url}/generate", payload, timeout=30)
        if status != 200:
            raise RuntimeError(f"/generate returned HTTP {status}")
        stdout = _terminate_process(proc)
        return {
            "observation_env": observation_value,
            "forward_completed": True,
            "http_status": status,
            "has_response": response is not None,
            "relaykv_summary_logged": "relaykv_runtime_observation_summary" in stdout,
            "relaykv_skip_logged": "relaykv_runtime_observation_hook_skip" in stdout,
            "log_tail": _tail(stdout, max_lines=20),
        }
    except Exception as exc:
        stdout = _terminate_process(proc)
        raise RuntimeError(
            f"server observation case failed for {OBSERVATION_ENV}={observation_value}: "
            f"{exc}\nserver_log_tail:\n{_tail(stdout)}"
        ) from exc


def main() -> int:
    model_path = os.environ.get(MODEL_ENV)
    if not model_path:
        return _clean_skip("model_env_unset", model_env=MODEL_ENV)

    model_path_obj = Path(model_path).expanduser()
    if not model_path_obj.exists():
        return _clean_skip(
            "model_path_not_found",
            model_env=MODEL_ENV,
            model_path=model_path,
        )

    if os.environ.get(RUN_ENV) != "1":
        return _clean_skip(
            "explicit_run_env_not_enabled",
            model_env=MODEL_ENV,
            model_path=str(model_path_obj),
            run_env=RUN_ENV,
        )

    timeout = float(os.environ.get(TIMEOUT_ENV, "90"))
    cases = [
        _run_server_case(str(model_path_obj), observation_value="0", timeout=timeout),
        _run_server_case(str(model_path_obj), observation_value="1", timeout=timeout),
    ]
    result = {
        "skipped": False,
        "model_path": str(model_path_obj),
        "offline_env": {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        },
        "cases": cases,
    }
    _print_result(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
