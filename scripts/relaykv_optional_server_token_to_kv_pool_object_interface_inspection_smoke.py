#!/usr/bin/env python3
"""Optional local-server smoke for token_to_kv_pool object interface inspection."""

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
OBSERVATION_ENV = "SGLANG_RELAYKV_RUNTIME_OBSERVATION"
REAL_REQ_TO_TOKEN_READ_ENV = "SGLANG_RELAYKV_REAL_REQ_TO_TOKEN_POOL_BOUNDED_READ"
SCALAR_CONVERSION_ENV = "SGLANG_RELAYKV_REAL_REQ_TO_TOKEN_POOL_SCALAR_ITEM_CONVERSION"
BRIDGE_ENV = "SGLANG_RELAYKV_REQ_TO_TOKEN_RESOLUTION_BRIDGE"
INDEX_READ_ENV = "SGLANG_RELAYKV_TOKEN_TO_KV_POOL_INDEX_READ"
INTERFACE_INSPECTION_ENV = "SGLANG_RELAYKV_TOKEN_TO_KV_POOL_OBJECT_INTERFACE_INSPECTION"
MIN_PORT = 1
MAX_PORT = 65535
OPTIONAL_PORT_BLOCK_SIZE = 3
OPTIONAL_PORT_MIN = 20000
OPTIONAL_PORT_MAX_START = 55000
REAL_READ_SAFETY_COUNTER_KEYS = (
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
INSPECTION_SAFETY_COUNTER_KEYS = (
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


def _print_result(result: dict[str, Any]) -> None:
    print(
        "relaykv_optional_server_token_to_kv_pool_object_interface_inspection_smoke_result="
        + json.dumps(result, sort_keys=True)
    )


def _clean_skip(reason: str, **extra: Any) -> int:
    result = {"skipped": True, "skip_reason": reason}
    result.update(extra)
    print(
        "relaykv_optional_server_token_to_kv_pool_object_interface_inspection_smoke: skip"
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


def _relaykv_observation_log_flags(stdout: str) -> dict[str, bool]:
    summary_logged = "relaykv_runtime_observation_summary" in stdout
    readonly_metadata_summary_logged = (
        "relaykv_runtime_observation_readonly_metadata_summary" in stdout
    )
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
    req_pool_idx_present_logged = '"req_pool_idx_none": false' in stdout
    cpu_tensor_value_source_logged = (
        '"seq_lens_cpu_value_source": "cpu_tensor_observation_only"' in stdout
    )
    return {
        "relaykv_summary_logged": summary_logged,
        "relaykv_readonly_metadata_summary_logged": readonly_metadata_summary_logged,
        "relaykv_existing_metadata_summary_logged": existing_metadata_summary_logged,
        "relaykv_skip_logged": skip_logged,
        "relaykv_metadata_description_logged": metadata_description_logged,
        "relaykv_cpu_metadata_description_logged": cpu_metadata_description_logged,
        "relaykv_req_pool_idx_none_logged": req_pool_idx_none_logged,
        "relaykv_req_pool_idx_present_logged": req_pool_idx_present_logged,
        "relaykv_cpu_tensor_value_source_logged": cpu_tensor_value_source_logged,
        "relaykv_observation_logged": (
            summary_logged
            or readonly_metadata_summary_logged
            or existing_metadata_summary_logged
            or skip_logged
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


def _inspection_hook_wired() -> bool:
    model_runner_path = (
        Path(__file__).resolve().parents[1]
        / "python"
        / "sglang"
        / "srt"
        / "model_executor"
        / "model_runner.py"
    )
    text = model_runner_path.read_text(encoding="utf-8")
    return (
        "SGLANG_RELAYKV_TOKEN_TO_KV_POOL_OBJECT_INTERFACE_INSPECTION" in text
        or "run_model_runner_token_to_kv_pool_object_interface_inspection_hook_for_smoke"
        in text
        or "relaykv_token_to_kv_pool_object_interface_inspection_summary"
        in text
    )


def _inspection_log_flags(stdout: str) -> dict[str, Any]:
    real_read_summary = _extract_summary_from_stdout(
        stdout,
        "relaykv_real_req_to_token_pool_bounded_read_summary=",
    )
    live_summary = _extract_summary_from_stdout(
        stdout,
        "relaykv_live_token_to_kv_pool_index_read_summary=",
    )
    interface_summary = _extract_summary_from_stdout(
        stdout,
        "relaykv_token_to_kv_pool_object_interface_inspection_summary=",
    )
    observation_flags = _relaykv_observation_log_flags(stdout)
    return {
        **observation_flags,
        "relaykv_real_req_to_token_pool_bounded_read_logged": (
            real_read_summary is not None
        ),
        "relaykv_real_req_to_token_pool_bounded_read_summary": real_read_summary,
        "relaykv_live_token_to_kv_pool_index_read_logged": live_summary is not None,
        "relaykv_live_token_to_kv_pool_index_read_summary": live_summary,
        "relaykv_token_to_kv_pool_object_interface_inspection_logged": (
            interface_summary is not None
        ),
        "relaykv_token_to_kv_pool_object_interface_inspection_summary": interface_summary,
    }


def _assert_zero_safety_counters(summary: dict[str, Any], keys: tuple[str, ...]) -> None:
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


def _validate_observation_flags(
    *,
    observation_enabled: bool,
    inspection_log_flags: dict[str, Any],
) -> None:
    if not observation_enabled:
        if inspection_log_flags["relaykv_observation_logged"]:
            raise RuntimeError(
                "RelayKV observation log was emitted while observation env was off"
            )
        return
    if not inspection_log_flags["relaykv_observation_logged"]:
        raise RuntimeError(
            "RelayKV observation hook log was not detected while observation env was on"
        )
    if inspection_log_flags["relaykv_readonly_metadata_summary_logged"]:
        if not inspection_log_flags["relaykv_req_pool_idx_present_logged"]:
            raise RuntimeError(
                "RelayKV req_pool_idx present observation log was not detected"
            )
        return
    if inspection_log_flags["relaykv_existing_metadata_summary_logged"]:
        if not inspection_log_flags["relaykv_cpu_tensor_value_source_logged"]:
            raise RuntimeError(
                "RelayKV CPU tensor observation source log was not detected"
            )
        if not inspection_log_flags["relaykv_req_pool_idx_none_logged"]:
            raise RuntimeError(
                "RelayKV req_pool_idx None observation log was not detected"
            )
        return
    raise RuntimeError("RelayKV observation summary log was not detected")


def _validate_live_summary(
    *,
    expected: bool,
    inspection_log_flags: dict[str, Any],
) -> dict[str, Any] | None:
    summary = inspection_log_flags["relaykv_live_token_to_kv_pool_index_read_summary"]
    if not expected:
        if inspection_log_flags["relaykv_live_token_to_kv_pool_index_read_logged"]:
            raise RuntimeError(
                "RelayKV live token_to_kv_pool index read summary was emitted while live-index env was off"
            )
        return None
    if summary is None:
        raise RuntimeError(
            "RelayKV live token_to_kv_pool index read summary log was not detected while live-index env was on"
        )
    _assert_zero_safety_counters(summary, LIVE_SAFETY_COUNTER_KEYS)
    for key in (
        "req_to_token_resolution_bridge_enabled",
        "req_to_token_resolution_bridge_state",
        "req_to_token_resolution_bridge_payload_count",
        "req_to_token_resolution_bridge_valid_count",
        "req_to_token_resolution_bridge_source_path",
        "token_to_kv_pool_source_path",
        "blocked_reason",
    ):
        if key not in summary:
            raise RuntimeError(f"missing live summary key: {key}")
    if summary.get("req_to_token_resolution_bridge_enabled") is not True:
        raise RuntimeError("live summary did not report bridge enabled")
    if not (
        int(summary.get("blocked_count") or 0) > 0
        or int(summary.get("physical_kv_index_resolved_count") or 0) > 0
    ):
        raise RuntimeError("live summary reported neither blocked nor resolved")
    if int(summary.get("blocked_count") or 0) > 0 and not summary.get("blocked_reason"):
        raise RuntimeError("blocked live summary lacked explicit blocked_reason")
    return summary


def _validate_real_read_summary(
    *,
    expected: bool,
    inspection_log_flags: dict[str, Any],
) -> dict[str, Any] | None:
    summary = inspection_log_flags["relaykv_real_req_to_token_pool_bounded_read_summary"]
    if not expected:
        if inspection_log_flags["relaykv_real_req_to_token_pool_bounded_read_logged"]:
            raise RuntimeError(
                "RelayKV real req_to_token bounded read summary was emitted while read env was off"
            )
        return None
    if summary is None:
        raise RuntimeError(
            "RelayKV real req_to_token bounded read summary log was not detected while read env was on"
        )
    _assert_zero_safety_counters(summary, REAL_READ_SAFETY_COUNTER_KEYS)
    for key in (
        "resolved_count",
        "blocked_count",
        "error_count",
        "payload_count",
        "req_to_token_payload_attached",
        "scalar_tensor_item_conversion_succeeded_count",
        "blocked_reason",
    ):
        if key not in summary:
            raise RuntimeError(f"missing real-read summary key: {key}")
    if not (
        int(summary.get("resolved_count") or 0) > 0
        or int(summary.get("blocked_count") or 0) > 0
        or int(summary.get("error_count") or 0) > 0
    ):
        raise RuntimeError("real-read summary reported neither resolved nor blocked")
    return summary


def _validate_interface_summary(
    *,
    inspection_requested: bool,
    hook_wired: bool,
    inspection_log_flags: dict[str, Any],
) -> tuple[dict[str, Any] | None, str | None]:
    summary = inspection_log_flags[
        "relaykv_token_to_kv_pool_object_interface_inspection_summary"
    ]
    if not inspection_requested:
        if inspection_log_flags[
            "relaykv_token_to_kv_pool_object_interface_inspection_logged"
        ]:
            raise RuntimeError(
                "RelayKV token_to_kv_pool interface inspection summary was emitted while inspection env was off"
            )
        return None, None

    if summary is None:
        if hook_wired:
            raise RuntimeError(
                "token_to_kv_pool object interface inspection env was on but no summary was emitted"
            )
        return None, "hook_not_wired"

    for key in (
        "inspection_enabled",
        "result_count",
        "inspected_count",
        "blocked_count",
        "error_count",
        "source_path",
        "observed_type_counts",
        "observed_shape_counts",
        "observed_dtype_counts",
        "observed_device_counts",
        "object_has_getitem_count",
        "known_method_presence_counts",
        "known_attr_presence_counts",
        "candidate_indexable_attr_names",
        "candidate_tensor_like_attr_names",
        "candidate_next_source_paths",
        "blocked_reason",
    ):
        if key not in summary:
            raise RuntimeError(f"missing interface inspection summary key: {key}")
    if summary.get("event_type") != "relaykv_token_to_kv_pool_object_interface_inspection_summary":
        raise RuntimeError("unexpected token_to_kv_pool interface inspection event_type")
    if summary.get("inspection_enabled") is not True:
        raise RuntimeError("inspection summary did not report inspection_enabled=true")
    _assert_zero_safety_counters(summary, INSPECTION_SAFETY_COUNTER_KEYS)
    inspected = int(summary.get("inspected_count") or 0)
    blocked = int(summary.get("blocked_count") or 0)
    error = int(summary.get("error_count") or 0)
    if inspected <= 0 and blocked <= 0 and error <= 0:
        raise RuntimeError(
            "interface inspection summary did not report inspected, blocked, or error"
        )
    if blocked > 0 and not summary.get("blocked_reason"):
        raise RuntimeError("blocked inspection summary lacked explicit blocked_reason")
    return summary, None


def _run_server_case(
    model_path: str,
    *,
    observation_enabled: bool,
    real_read_enabled: bool,
    scalar_conversion_enabled: bool,
    bridge_enabled: bool,
    index_read_enabled: bool,
    interface_inspection_enabled: bool,
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
    if observation_enabled:
        env[OBSERVATION_ENV] = "1"
    else:
        env.pop(OBSERVATION_ENV, None)
    if real_read_enabled:
        env[REAL_REQ_TO_TOKEN_READ_ENV] = "1"
    else:
        env.pop(REAL_REQ_TO_TOKEN_READ_ENV, None)
    if scalar_conversion_enabled:
        env[SCALAR_CONVERSION_ENV] = "1"
    else:
        env.pop(SCALAR_CONVERSION_ENV, None)
    if bridge_enabled:
        env[BRIDGE_ENV] = "1"
    else:
        env.pop(BRIDGE_ENV, None)
    if index_read_enabled:
        env[INDEX_READ_ENV] = "1"
    else:
        env.pop(INDEX_READ_ENV, None)
    if interface_inspection_enabled:
        env[INTERFACE_INSPECTION_ENV] = "1"
    else:
        env.pop(INTERFACE_INSPECTION_ENV, None)
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
                "RelayKV token_to_kv_pool object interface inspection optional server smoke."
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
        _validate_observation_flags(
            observation_enabled=observation_enabled,
            inspection_log_flags=inspection_log_flags,
        )
        real_read_summary = _validate_real_read_summary(
            expected=real_read_enabled,
            inspection_log_flags=inspection_log_flags,
        )
        live_summary = _validate_live_summary(
            expected=index_read_enabled,
            inspection_log_flags=inspection_log_flags,
        )
        return {
            "status": status,
            "response": response,
            "response_marker": _production_output_marker(response),
            "observation_logged": inspection_log_flags["relaykv_observation_logged"],
            "real_read_summary": real_read_summary,
            "live_summary": live_summary,
            "interface_summary": inspection_log_flags[
                "relaykv_token_to_kv_pool_object_interface_inspection_summary"
            ],
            "stdout_tail": _tail(stdout),
            "generate_timeout_seen": generate_timeout_seen,
            "timeout_classification": timeout_classification,
        }
    except Exception:
        if not stdout:
            stdout = _terminate_process(proc)
        raise RuntimeError(_tail(stdout)) from None


def main() -> int:
    model_path = os.environ.get(MODEL_ENV)
    run_flag = os.environ.get(RUN_ENV)
    port_bounds = _check_port_selection_bounds_for_smoke()
    hook_wired = _inspection_hook_wired()
    if run_flag != "1":
        return _clean_skip(
            "run_env_unset",
            run_env=RUN_ENV,
            model_env=MODEL_ENV,
            port_selection_bounds_check=port_bounds,
            inspection_hook_wired=hook_wired,
        )
    if not model_path:
        return _clean_skip(
            "model_env_unset",
            model_env=MODEL_ENV,
            port_selection_bounds_check=port_bounds,
            inspection_hook_wired=hook_wired,
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
            "observation_enabled": False,
            "real_read_enabled": False,
            "scalar_conversion_enabled": False,
            "bridge_enabled": False,
            "index_read_enabled": False,
            "interface_inspection_enabled": False,
        },
        {
            "name": "observation_only",
            "observation_enabled": True,
            "real_read_enabled": False,
            "scalar_conversion_enabled": False,
            "bridge_enabled": False,
            "index_read_enabled": False,
            "interface_inspection_enabled": False,
        },
        {
            "name": "full_chain_live_index",
            "observation_enabled": True,
            "real_read_enabled": True,
            "scalar_conversion_enabled": True,
            "bridge_enabled": True,
            "index_read_enabled": True,
            "interface_inspection_enabled": False,
        },
        {
            "name": "token_to_kv_interface_inspection",
            "observation_enabled": True,
            "real_read_enabled": True,
            "scalar_conversion_enabled": True,
            "bridge_enabled": True,
            "index_read_enabled": False,
            "interface_inspection_enabled": True,
        },
    ]

    raw_case_results = []
    for case in cases:
        result = _run_server_case(
            model_path,
            observation_enabled=case["observation_enabled"],
            real_read_enabled=case["real_read_enabled"],
            scalar_conversion_enabled=case["scalar_conversion_enabled"],
            bridge_enabled=case["bridge_enabled"],
            index_read_enabled=case["index_read_enabled"],
            interface_inspection_enabled=case["interface_inspection_enabled"],
            timeout=timeout,
            generate_timeout=generate_timeout,
            generate_timeout_grace=generate_timeout_grace,
        )
        result["name"] = case["name"]
        raw_case_results.append(result)

    response_markers = [result["response_marker"] for result in raw_case_results]
    if len(set(response_markers)) != 1:
        raise RuntimeError(
            "response marker changed across cases: "
            + json.dumps(response_markers, sort_keys=True)
        )

    case1 = raw_case_results[0]
    if case1["observation_logged"] or case1["live_summary"] is not None or case1["interface_summary"] is not None:
        raise RuntimeError("all-off case unexpectedly emitted RelayKV summaries")

    case2 = raw_case_results[1]
    if not case2["observation_logged"]:
        raise RuntimeError("observation-only case did not emit observation logs")
    if (
        case2["real_read_summary"] is not None
        or case2["live_summary"] is not None
        or case2["interface_summary"] is not None
    ):
        raise RuntimeError(
            "observation-only case unexpectedly emitted real-read, live-index, or interface-inspection summaries"
        )

    case3 = raw_case_results[2]
    case3_real = case3["real_read_summary"] or {}
    case3_live = case3["live_summary"] or {}
    if case3["real_read_summary"] is None:
        raise RuntimeError("case 3 did not emit real bounded-read summary")
    if int(case3_real.get("resolved_count") or 0) <= 0:
        raise RuntimeError(
            "case 3 did not resolve bounded read: "
            + json.dumps(
                {
                    "resolved_count": case3_real.get("resolved_count"),
                    "blocked_count": case3_real.get("blocked_count"),
                    "blocked_reason": case3_real.get("blocked_reason"),
                },
                sort_keys=True,
            )
        )
    if case3_real.get("req_to_token_payload_attached") is not True:
        raise RuntimeError("case 3 did not attach req_to_token payloads")
    if case3["live_summary"] is None:
        raise RuntimeError("case 3 did not emit live-index summary")
    if case3_live.get("req_to_token_resolution_bridge_state") != "bridged":
        raise RuntimeError("case 3 live summary did not report bridged state")
    if int(case3_live.get("req_to_token_resolution_bridge_valid_count") or 0) <= 0:
        raise RuntimeError("case 3 had zero bridge valid count")
    if case3["interface_summary"] is not None:
        raise RuntimeError("case 3 unexpectedly emitted interface inspection summary")

    case4 = raw_case_results[3]
    interface_summary, controlled_outcome = _validate_interface_summary(
        inspection_requested=True,
        hook_wired=hook_wired,
        inspection_log_flags={
            "relaykv_token_to_kv_pool_object_interface_inspection_summary": case4[
                "interface_summary"
            ],
            "relaykv_token_to_kv_pool_object_interface_inspection_logged": (
                case4["interface_summary"] is not None
            ),
        },
    )

    compact_cases = []
    for case in raw_case_results:
        live_summary = case["live_summary"] or {}
        interface_summary_case = case["interface_summary"] or {}
        compact_cases.append(
            {
                "name": case["name"],
                "response_marker": case["response_marker"],
                "real_read_resolved_count": (case["real_read_summary"] or {}).get("resolved_count"),
                "real_read_blocked_reason": (case["real_read_summary"] or {}).get("blocked_reason"),
                "payload_attached": (case["real_read_summary"] or {}).get("req_to_token_payload_attached"),
                "payload_count": (case["real_read_summary"] or {}).get("payload_count"),
                "live_index_blocked_reason": live_summary.get("blocked_reason"),
                "inspection_source_path": interface_summary_case.get("source_path"),
                "observed_type_counts": interface_summary_case.get("observed_type_counts"),
                "observed_shape_counts": interface_summary_case.get("observed_shape_counts"),
                "observed_dtype_counts": interface_summary_case.get("observed_dtype_counts"),
                "observed_device_counts": interface_summary_case.get("observed_device_counts"),
                "object_has_getitem_count": interface_summary_case.get("object_has_getitem_count"),
                "known_method_presence_counts": interface_summary_case.get("known_method_presence_counts"),
                "known_attr_presence_counts": interface_summary_case.get("known_attr_presence_counts"),
                "candidate_indexable_attr_names": interface_summary_case.get("candidate_indexable_attr_names"),
                "candidate_tensor_like_attr_names": interface_summary_case.get("candidate_tensor_like_attr_names"),
                "candidate_next_source_paths": interface_summary_case.get("candidate_next_source_paths"),
                "controlled_outcome": (
                    controlled_outcome if case["name"] == "token_to_kv_interface_inspection" else None
                ),
            }
        )

    result = {
        "skipped": False,
        "inspection_hook_wired": hook_wired,
        "case_results": compact_cases,
        "raw_case_results": raw_case_results,
        "response_marker": response_markers[0],
    }
    if interface_summary is not None:
        result["inspection_summary_state"] = (
            "inspected"
            if int(interface_summary.get("inspected_count") or 0) > 0
            else "blocked_or_error"
        )
    elif controlled_outcome is not None:
        result["inspection_summary_state"] = controlled_outcome
    else:
        raise RuntimeError("case 4 lacked both interface summary and controlled outcome")

    print("relaykv_optional_server_token_to_kv_pool_object_interface_inspection_smoke=pass")
    _print_result(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
