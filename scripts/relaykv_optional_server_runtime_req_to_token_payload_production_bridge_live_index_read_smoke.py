#!/usr/bin/env python3
"""Optional local-server smoke for RelayKV producer + bridge + live index read."""

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

FLASHINFER_BASE = "/tmp/relaykv_flashinfer_cache"
os.environ.setdefault("FLASHINFER_WORKSPACE_BASE", FLASHINFER_BASE)

MODEL_ENV = "RELAYKV_OPTIONAL_SERVER_SMOKE_MODEL"
RUN_ENV = "RELAYKV_OPTIONAL_SERVER_SMOKE_RUN"
TIMEOUT_ENV = "RELAYKV_OPTIONAL_SERVER_SMOKE_TIMEOUT"
GENERATE_TIMEOUT_ENV = "RELAYKV_OPTIONAL_SERVER_SMOKE_GENERATE_TIMEOUT"
GENERATE_TIMEOUT_GRACE_ENV = "RELAYKV_OPTIONAL_SERVER_SMOKE_GENERATE_TIMEOUT_GRACE"
PRODUCTION_ENV = "SGLANG_RELAYKV_RUNTIME_REQ_TO_TOKEN_PAYLOAD_PRODUCTION"
BRIDGE_ENV = "SGLANG_RELAYKV_REQ_TO_TOKEN_RESOLUTION_BRIDGE"
INDEX_READ_ENV = "SGLANG_RELAYKV_TOKEN_TO_KV_POOL_INDEX_READ"
MIN_PORT = 1
MAX_PORT = 65535
OPTIONAL_PORT_BLOCK_SIZE = 3
OPTIONAL_PORT_MIN = 20000
OPTIONAL_PORT_MAX_START = 55000
PRODUCER_SAFETY_COUNTER_KEYS = (
    "req_to_token_read_count",
    "actual_req_to_token_pool_read_count",
    "token_to_kv_pool_read_count",
    "actual_token_to_kv_pool_read_count",
    "live_token_to_kv_pool_index_read_count",
    "kv_pool_read_count",
    "kv_snapshot_count",
    "tensor_read_count",
    "attention_comparison_executed_count",
    "attention_override_true_count",
    "runtime_writeback_true_count",
    "scheduler_policy_noop_false_count",
    "kv_cache_mutation_true_count",
    "source_mutated_true_count",
)
LIVE_SAFETY_COUNTER_KEYS = (
    "req_to_token_read_count",
    "actual_req_to_token_pool_read_count",
    "kv_pool_read_count",
    "kv_snapshot_count",
    "tensor_read_count",
    "attention_comparison_executed_count",
    "attention_override_true_count",
    "runtime_writeback_true_count",
    "scheduler_policy_noop_false_count",
    "kv_cache_mutation_true_count",
    "source_mutated_true_count",
)


def _print_result(result: dict[str, Any]) -> None:
    print(
        "relaykv_optional_server_runtime_req_to_token_payload_production_bridge_live_index_read_smoke_result="
        + json.dumps(result, sort_keys=True)
    )


def _clean_skip(reason: str, **extra: Any) -> int:
    result = {"skipped": True, "skip_reason": reason}
    result.update(extra)
    print(
        "relaykv_optional_server_runtime_req_to_token_payload_production_bridge_live_index_read_smoke: skip"
    )
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


def _server_log_flags(stdout: str) -> dict[str, bool]:
    return {
        "generate_200_logged": (
            '/generate HTTP/1.1" 200 OK' in stdout
            or "POST /generate" in stdout
            and " 200 OK" in stdout
        ),
    }


def _extract_summary_from_stdout(stdout: str, prefix: str) -> dict[str, Any] | None:
    for line in stdout.splitlines():
        marker = line.find(prefix)
        if marker < 0:
            continue
        payload = line[marker + len(prefix) :].strip()
        try:
            return json.loads(payload)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"invalid summary log for {prefix}: {exc}")
    return None


def _inspection_log_flags(stdout: str) -> dict[str, Any]:
    producer_summary = _extract_summary_from_stdout(
        stdout,
        "relaykv_runtime_req_to_token_payload_production_summary=",
    )
    live_summary = _extract_summary_from_stdout(
        stdout,
        "relaykv_live_token_to_kv_pool_index_read_summary=",
    )
    return {
        "relaykv_runtime_req_to_token_payload_production_logged": (
            producer_summary is not None
        ),
        "relaykv_runtime_req_to_token_payload_production_summary": producer_summary,
        "relaykv_live_token_to_kv_pool_index_read_logged": live_summary is not None,
        "relaykv_live_token_to_kv_pool_index_read_summary": live_summary,
    }


def _assert_zero_safety_counters(
    summary: dict[str, Any],
    keys: tuple[str, ...],
) -> None:
    for key in keys:
        if summary.get(key) != 0:
            raise RuntimeError(f"expected {key}=0, got {summary.get(key)}")


def _production_output_marker(response: Any) -> str:
    if isinstance(response, dict):
        text_value = response.get("text")
        if isinstance(text_value, str):
            return text_value
        if isinstance(text_value, list):
            return json.dumps(text_value, sort_keys=True, default=str)
    return json.dumps(response, sort_keys=True, default=str)


def _validate_producer_summary(
    *,
    producer_enabled: bool,
    inspection_log_flags: dict[str, Any],
) -> dict[str, Any] | None:
    summary = inspection_log_flags[
        "relaykv_runtime_req_to_token_payload_production_summary"
    ]
    if not producer_enabled:
        if inspection_log_flags[
            "relaykv_runtime_req_to_token_payload_production_logged"
        ]:
            raise RuntimeError(
                "RelayKV runtime req_to_token payload production summary was emitted while producer env was off"
            )
        return None

    if summary is None:
        raise RuntimeError(
            "RelayKV runtime req_to_token payload production summary log was not detected while producer env was on"
        )
    if summary.get("production_enabled") is not True:
        raise RuntimeError("producer summary did not report production_enabled=true")
    for key in (
        "payload_count",
        "resolved_count",
        "blocked_count",
        "error_count",
        "payload_attached",
        "relaykv_payload_attr_write_count",
    ):
        if key not in summary:
            raise RuntimeError(f"missing producer summary key: {key}")
    if "payload_attach_target" not in summary:
        raise RuntimeError("missing producer payload_attach_target")
    _assert_zero_safety_counters(summary, PRODUCER_SAFETY_COUNTER_KEYS)
    return summary


def _validate_live_summary(
    *,
    index_read_enabled: bool,
    bridge_enabled: bool,
    inspection_log_flags: dict[str, Any],
) -> dict[str, Any] | None:
    summary = inspection_log_flags["relaykv_live_token_to_kv_pool_index_read_summary"]
    if not index_read_enabled:
        if inspection_log_flags["relaykv_live_token_to_kv_pool_index_read_logged"]:
            raise RuntimeError(
                "RelayKV live token_to_kv_pool index read summary was emitted while index-read env was off"
            )
        return None

    if summary is None:
        raise RuntimeError(
            "RelayKV live token_to_kv_pool index read summary log was not detected while index-read env was on"
        )
    _assert_zero_safety_counters(summary, LIVE_SAFETY_COUNTER_KEYS)

    bridge_flag_value = summary.get("req_to_token_resolution_bridge_enabled")
    if bridge_enabled:
        if bridge_flag_value is not True:
            raise RuntimeError(
                "bridge-enabled case did not report req_to_token_resolution_bridge_enabled=true"
            )
        for key in (
            "req_to_token_resolution_bridge_state",
            "req_to_token_resolution_bridge_payload_count",
            "req_to_token_resolution_bridge_valid_count",
            "req_to_token_resolution_bridge_source_path",
            "req_to_token_resolution_bridge_blocked_reason",
        ):
            if key not in summary:
                raise RuntimeError(f"missing bridge metadata key: {key}")
    else:
        if bridge_flag_value not in (None, False):
            raise RuntimeError(
                "bridge-off case reported unexpected req_to_token_resolution_bridge_enabled value"
            )

    resolved_count = summary.get("physical_kv_index_resolved_count", 0)
    blocked_count = summary.get("blocked_count", 0)
    live_read_count = summary.get("live_token_to_kv_pool_index_read_count", 0)
    token_read_count = summary.get("token_to_kv_pool_read_count", 0)
    actual_token_read_count = summary.get("actual_token_to_kv_pool_read_count", 0)

    if resolved_count > 0:
        if live_read_count <= 0:
            raise RuntimeError(
                "resolved summary had zero live_token_to_kv_pool_index_read_count"
            )
        if token_read_count <= 0:
            raise RuntimeError("resolved summary had zero token_to_kv_pool_read_count")
        if actual_token_read_count <= 0:
            raise RuntimeError(
                "resolved summary had zero actual_token_to_kv_pool_read_count"
            )
        if bridge_enabled and summary.get(
            "req_to_token_resolution_bridge_valid_count", 0
        ) <= 0:
            raise RuntimeError(
                "bridge-enabled resolved summary had zero req_to_token_resolution_bridge_valid_count"
            )
        return summary

    if blocked_count > 0:
        if live_read_count != 0:
            raise RuntimeError(
                "blocked summary expected live_token_to_kv_pool_index_read_count=0"
            )
        return summary

    raise RuntimeError(
        "RelayKV live token_to_kv_pool index read summary did not report a resolved or blocked path"
    )


def _validate_case(
    *,
    producer_enabled: bool,
    bridge_enabled: bool,
    index_read_enabled: bool,
    response: Any,
    inspection_log_flags: dict[str, Any],
) -> dict[str, Any]:
    producer_summary = _validate_producer_summary(
        producer_enabled=producer_enabled,
        inspection_log_flags=inspection_log_flags,
    )
    live_summary = _validate_live_summary(
        index_read_enabled=index_read_enabled,
        bridge_enabled=bridge_enabled,
        inspection_log_flags=inspection_log_flags,
    )

    result = {
        "producer_expected": producer_enabled,
        "producer_ran": producer_summary is not None,
        "index_read_expected": index_read_enabled,
        "index_read_ran": live_summary is not None,
        "resolved": False,
        "clean_blocked": False,
        "response_marker": _production_output_marker(response),
    }
    if live_summary is not None:
        if live_summary.get("physical_kv_index_resolved_count", 0) > 0:
            result["resolved"] = True
        elif live_summary.get("blocked_count", 0) > 0:
            result["clean_blocked"] = True
    elif producer_summary is not None and producer_summary.get("blocked_count", 0) > 0:
        result["clean_blocked"] = True
    return result


def _run_server_case(
    model_path: str,
    *,
    producer_enabled: bool,
    bridge_enabled: bool,
    index_read_enabled: bool,
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
    if producer_enabled:
        env[PRODUCTION_ENV] = "1"
    else:
        env.pop(PRODUCTION_ENV, None)
    if bridge_enabled:
        env[BRIDGE_ENV] = "1"
    else:
        env.pop(BRIDGE_ENV, None)
    if index_read_enabled:
        env[INDEX_READ_ENV] = "1"
    else:
        env.pop(INDEX_READ_ENV, None)
    env["SGLANG_GRPC_PORT"] = str(port_info["grpc_port"])
    env.setdefault("FLASHINFER_WORKSPACE_BASE", FLASHINFER_BASE)
    Path(env["FLASHINFER_WORKSPACE_BASE"]).mkdir(parents=True, exist_ok=True)

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
        "--trust-remote-code",
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
            "text": (
                "RelayKV runtime req_to_token payload production bridge live index read smoke."
            ),
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
        inspection_log_flags = _inspection_log_flags(stdout)
        server_log_flags = _server_log_flags(stdout)
        if generate_timeout_seen and server_log_flags["generate_200_logged"]:
            status = 200
            timeout_classification = "generate_timeout_but_200_logged"
        if status != 200:
            raise RuntimeError(f"/generate returned HTTP {status}")
        if response is None:
            response = {
                "timeout_classification": timeout_classification or "response_missing"
            }
        validation = _validate_case(
            producer_enabled=producer_enabled,
            bridge_enabled=bridge_enabled,
            index_read_enabled=index_read_enabled,
            response=response,
            inspection_log_flags=inspection_log_flags,
        )
        return {
            "status": status,
            "response": response,
            "response_marker": validation["response_marker"],
            "producer_enabled": producer_enabled,
            "bridge_enabled": bridge_enabled,
            "index_read_enabled": index_read_enabled,
            "producer_summary": inspection_log_flags[
                "relaykv_runtime_req_to_token_payload_production_summary"
            ],
            "live_index_read_summary": inspection_log_flags[
                "relaykv_live_token_to_kv_pool_index_read_summary"
            ],
            "producer_ran": validation["producer_ran"],
            "index_read_ran": validation["index_read_ran"],
            "resolved": validation["resolved"],
            "clean_blocked": validation["clean_blocked"],
            "generate_timeout_seen": generate_timeout_seen,
            "timeout_classification": timeout_classification,
            "stdout_tail": _tail(stdout),
        }
    except Exception:
        if not stdout:
            stdout = _terminate_process(proc)
        raise RuntimeError(_tail(stdout)) from None


def main() -> int:
    model_path = os.environ.get(MODEL_ENV)
    run_flag = os.environ.get(RUN_ENV)
    port_bounds = _check_port_selection_bounds_for_smoke()
    if run_flag != "1":
        return _clean_skip(
            "run_env_unset",
            run_env=RUN_ENV,
            model_env=MODEL_ENV,
            port_selection_bounds_check=port_bounds,
        )
    if not model_path:
        return _clean_skip(
            "model_env_unset",
            model_env=MODEL_ENV,
            port_selection_bounds_check=port_bounds,
        )

    Path(os.environ.setdefault("FLASHINFER_WORKSPACE_BASE", FLASHINFER_BASE)).mkdir(
        parents=True,
        exist_ok=True,
    )
    timeout = float(os.environ.get(TIMEOUT_ENV, "240"))
    generate_timeout = float(os.environ.get(GENERATE_TIMEOUT_ENV, "90"))
    generate_timeout_grace = float(os.environ.get(GENERATE_TIMEOUT_GRACE_ENV, "15"))

    cases = [
        {
            "name": "all_off",
            "producer_enabled": False,
            "bridge_enabled": False,
            "index_read_enabled": False,
        },
        {
            "name": "producer_only",
            "producer_enabled": True,
            "bridge_enabled": False,
            "index_read_enabled": False,
        },
        {
            "name": "producer_bridge_index",
            "producer_enabled": True,
            "bridge_enabled": True,
            "index_read_enabled": True,
        },
    ]

    case_results = []
    for case in cases:
        result = _run_server_case(
            model_path,
            producer_enabled=case["producer_enabled"],
            bridge_enabled=case["bridge_enabled"],
            index_read_enabled=case["index_read_enabled"],
            timeout=timeout,
            generate_timeout=generate_timeout,
            generate_timeout_grace=generate_timeout_grace,
        )
        result["name"] = case["name"]
        case_results.append(result)

    response_markers = [result["response_marker"] for result in case_results]
    if len(set(response_markers)) != 1:
        raise RuntimeError(
            "response marker changed across cases: "
            + json.dumps(response_markers, sort_keys=True)
        )

    all_off = case_results[0]
    if all_off["producer_ran"] or all_off["index_read_ran"]:
        raise RuntimeError("all-off case unexpectedly emitted RelayKV summaries")

    producer_only = case_results[1]
    if not producer_only["producer_ran"]:
        raise RuntimeError("producer-only case did not emit producer summary")
    if producer_only["index_read_ran"]:
        raise RuntimeError("producer-only case unexpectedly emitted live-index summary")

    producer_bridge_index = case_results[2]
    if not producer_bridge_index["producer_ran"]:
        raise RuntimeError(
            "producer+bridge+index case did not emit producer summary"
        )
    if not producer_bridge_index["index_read_ran"]:
        raise RuntimeError(
            "producer+bridge+index case did not emit live-index summary"
        )

    result = {
        "skipped": False,
        "port_selection_bounds_check": port_bounds,
        "cases": case_results,
        "production_output_unaffected": True,
    }
    print(
        "relaykv_optional_server_runtime_req_to_token_payload_production_bridge_live_index_read_smoke=pass"
    )
    _print_result(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
