#!/usr/bin/env python3
"""Optional local-server smoke for RelayKV runtime observation.

This smoke is intentionally opt-in. It only runs when a local model path is
provided and server launch is explicitly enabled, so regular smoke runs do not
start a server or trigger model downloads.
"""

from __future__ import annotations

import json
import os
import random
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
GENERATE_TIMEOUT_ENV = "RELAYKV_OPTIONAL_SERVER_SMOKE_GENERATE_TIMEOUT"
GENERATE_TIMEOUT_GRACE_ENV = "RELAYKV_OPTIONAL_SERVER_SMOKE_GENERATE_TIMEOUT_GRACE"
OBSERVATION_ENV = "SGLANG_RELAYKV_RUNTIME_OBSERVATION"
MIN_PORT = 1
MAX_PORT = 65535
OPTIONAL_PORT_BLOCK_SIZE = 3
OPTIONAL_PORT_MIN = 20000
OPTIONAL_PORT_MAX_START = 55000


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


def _check_port_range(port: int, name: str) -> None:
    if not (MIN_PORT <= port <= MAX_PORT):
        raise ValueError(f"{name} ({port}) must be between {MIN_PORT} and {MAX_PORT}")


def _try_bind_local_ports(
    ports: list[int],
) -> tuple[list[socket.socket] | None, str | None]:
    sockets = []
    try:
        for port in ports:
            _check_port_range(port, "relaykv_optional_server_smoke_port")
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("127.0.0.1", port))
            sockets.append(sock)
        return sockets, None
    except OSError as exc:
        for sock in sockets:
            sock.close()
        return None, str(exc)


def _reserve_local_port_block(count: int = OPTIONAL_PORT_BLOCK_SIZE) -> list[int]:
    if count < 1:
        raise ValueError("port block count must be positive")
    max_start = min(OPTIONAL_PORT_MAX_START, MAX_PORT - count + 1)
    candidates = list(range(OPTIONAL_PORT_MIN, max_start + 1))
    random.shuffle(candidates)
    last_error = None
    for start in candidates[:200]:
        ports = [start + offset for offset in range(count)]
        sockets, error = _try_bind_local_ports(ports)
        if sockets is None:
            last_error = error
            continue
        for sock in sockets:
            sock.close()
        return ports
    raise RuntimeError(
        "failed to reserve a local optional server smoke port block "
        f"in {OPTIONAL_PORT_MIN}..{max_start}; last_error={last_error}"
    )


def _port_info_from_ports(ports: list[int]) -> dict[str, int]:
    port_info = {
        "server_port": ports[0],
        "grpc_port": ports[1],
        "nccl_port": ports[2],
    }
    for name, port in port_info.items():
        _check_port_range(port, name)
    return port_info


def _check_port_selection_bounds_for_smoke() -> dict[str, int]:
    start = min(OPTIONAL_PORT_MAX_START, MAX_PORT - OPTIONAL_PORT_BLOCK_SIZE + 1)
    ports = [start + offset for offset in range(OPTIONAL_PORT_BLOCK_SIZE)]
    return _port_info_from_ports(ports)


def _reserve_port_selection_for_server() -> dict[str, int]:
    return _port_info_from_ports(_reserve_local_port_block())


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
            status, _ = _request_json(f"{base_url}/v1/models", timeout=2.0)
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


def _relaykv_observation_log_flags(stdout: str) -> dict[str, bool]:
    summary_logged = "relaykv_runtime_observation_summary" in stdout
    existing_metadata_summary_logged = (
        "relaykv_runtime_observation_forward_batch_existing_metadata_summary"
        in stdout
    )
    skip_logged = "relaykv_runtime_observation_skip" in stdout
    metadata_description_logged = "metadata_description" in stdout
    cpu_metadata_description_logged = (
        "forward_batch_cpu_metadata_description" in stdout
    )
    req_pool_idx_none_logged = '"req_pool_idx_none": true' in stdout
    cpu_tensor_value_source_logged = (
        '"seq_lens_cpu_value_source": "cpu_tensor_observation_only"' in stdout
    )
    return {
        "relaykv_summary_logged": summary_logged,
        "relaykv_existing_metadata_summary_logged": existing_metadata_summary_logged,
        "relaykv_skip_logged": skip_logged,
        "relaykv_metadata_description_logged": metadata_description_logged,
        "relaykv_cpu_metadata_description_logged": cpu_metadata_description_logged,
        "relaykv_req_pool_idx_none_logged": req_pool_idx_none_logged,
        "relaykv_cpu_tensor_value_source_logged": cpu_tensor_value_source_logged,
        "relaykv_observation_logged": (
            summary_logged or existing_metadata_summary_logged or skip_logged
        ),
    }


def _server_log_flags(stdout: str) -> dict[str, bool]:
    return {
        "generate_200_logged": (
            '/generate HTTP/1.1" 200 OK' in stdout
            or "POST /generate" in stdout
            and " 200 OK" in stdout
        ),
    }


def _validate_relaykv_observation_flags(
    *,
    observation_value: str,
    relaykv_log_flags: dict[str, bool],
) -> None:
    if observation_value == "0":
        if relaykv_log_flags["relaykv_observation_logged"]:
            raise RuntimeError("RelayKV observation log was emitted while env was off")
        return

    if not relaykv_log_flags["relaykv_observation_logged"]:
        raise RuntimeError("RelayKV observation hook log was not detected while env was on")
    if not relaykv_log_flags["relaykv_existing_metadata_summary_logged"]:
        raise RuntimeError("RelayKV existing metadata summary log was not detected")
    if not relaykv_log_flags["relaykv_cpu_tensor_value_source_logged"]:
        raise RuntimeError("RelayKV CPU tensor observation source log was not detected")
    if not relaykv_log_flags["relaykv_req_pool_idx_none_logged"]:
        raise RuntimeError("RelayKV req_pool_idx None observation log was not detected")


def _run_server_case(
    model_path: str,
    observation_value: str,
    timeout: float,
    generate_timeout: float,
    generate_timeout_grace: float,
) -> dict[str, Any]:
    port_info = _reserve_port_selection_for_server()
    port = port_info["server_port"]
    base_url = f"http://127.0.0.1:{port}"
    env = os.environ.copy()
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env[OBSERVATION_ENV] = observation_value
    env["SGLANG_GRPC_PORT"] = str(port_info["grpc_port"])
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
        "--nccl-port",
        str(port_info["nccl_port"]),
        "--disable-cuda-graph",
        "--disable-piecewise-cuda-graph",
        "--disable-overlap-schedule",
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
        generate_timeout_seen = False
        timeout_classification = ""
        response = None
        status = 0
        try:
            status, response = _request_json(
                f"{base_url}/generate",
                payload,
                timeout=generate_timeout,
            )
        except TimeoutError:
            generate_timeout_seen = True
            timeout_classification = "generate_request_timeout"
            if generate_timeout_grace > 0:
                time.sleep(generate_timeout_grace)
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, TimeoutError):
                generate_timeout_seen = True
                timeout_classification = "generate_request_timeout"
                if generate_timeout_grace > 0:
                    time.sleep(generate_timeout_grace)
            else:
                raise
        stdout = _terminate_process(proc)
        relaykv_log_flags = _relaykv_observation_log_flags(stdout)
        server_log_flags = _server_log_flags(stdout)
        if generate_timeout_seen and server_log_flags["generate_200_logged"]:
            status = 200
            timeout_classification = "generate_timeout_but_200_logged"
        if status != 200:
            raise RuntimeError(f"/generate returned HTTP {status}")
        _validate_relaykv_observation_flags(
            observation_value=observation_value,
            relaykv_log_flags=relaykv_log_flags,
        )
        return {
            "observation_env": observation_value,
            "forward_completed": True,
            "http_status": status,
            "has_response": response is not None,
            "generate_timeout": generate_timeout_seen,
            "timeout_classification": timeout_classification,
            "ports": port_info,
            **relaykv_log_flags,
            **server_log_flags,
            "log_tail": _tail(stdout, max_lines=80),
        }
    except Exception as exc:
        stdout = _terminate_process(proc)
        relaykv_log_flags = _relaykv_observation_log_flags(stdout)
        server_log_flags = _server_log_flags(stdout)
        generate_timeout_seen = isinstance(exc, TimeoutError)
        timeout_classification = "case_failed"
        if generate_timeout_seen and server_log_flags["generate_200_logged"]:
            timeout_classification = "generate_timeout_but_200_logged_case_failed"
        elif generate_timeout_seen:
            timeout_classification = "generate_request_timeout"
        return {
            "observation_env": observation_value,
            "case_failed": True,
            "error": (
                f"server observation case failed for "
                f"{OBSERVATION_ENV}={observation_value}: {exc}"
            ),
            "generate_timeout": generate_timeout_seen,
            "timeout_classification": timeout_classification,
            **relaykv_log_flags,
            **server_log_flags,
            "log_tail": _tail(stdout, max_lines=80),
        }


def main() -> int:
    port_bounds_check = _check_port_selection_bounds_for_smoke()
    model_path = os.environ.get(MODEL_ENV)
    if not model_path:
        return _clean_skip(
            "model_env_unset",
            model_env=MODEL_ENV,
            port_selection_bounds_check=port_bounds_check,
        )

    model_path_obj = Path(model_path).expanduser()
    if not model_path_obj.exists():
        return _clean_skip(
            "model_path_not_found",
            model_env=MODEL_ENV,
            model_path=model_path,
            port_selection_bounds_check=port_bounds_check,
        )

    if os.environ.get(RUN_ENV) != "1":
        return _clean_skip(
            "explicit_run_env_not_enabled",
            model_env=MODEL_ENV,
            model_path=str(model_path_obj),
            port_selection_bounds_check=port_bounds_check,
            run_env=RUN_ENV,
        )

    timeout = float(os.environ.get(TIMEOUT_ENV, "90"))
    generate_timeout = float(os.environ.get(GENERATE_TIMEOUT_ENV, "120"))
    generate_timeout_grace = float(os.environ.get(GENERATE_TIMEOUT_GRACE_ENV, "20"))
    cases = []
    for observation_value in ("0", "1"):
        cases.append(
            _run_server_case(
                str(model_path_obj),
                observation_value=observation_value,
                timeout=timeout,
                generate_timeout=generate_timeout,
                generate_timeout_grace=generate_timeout_grace,
            )
        )
    result = {
        "skipped": False,
        "model_path": str(model_path_obj),
        "offline_env": {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        },
        "timeouts": {
            "server_health_timeout": timeout,
            "generate_timeout": generate_timeout,
            "generate_timeout_grace": generate_timeout_grace,
        },
        "cases": cases,
    }
    _print_result(result)
    return 1 if any(case.get("case_failed") for case in cases) else 0


if __name__ == "__main__":
    raise SystemExit(main())
