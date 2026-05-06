#!/usr/bin/env python3
"""Optional local-server smoke for RelayKV req_to_token_pool value-shape inspection."""

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
VALUE_SHAPE_INSPECTION_ENV = "SGLANG_RELAYKV_REQ_TO_TOKEN_POOL_VALUE_SHAPE_INSPECTION"
MIN_PORT = 1
MAX_PORT = 65535
OPTIONAL_PORT_BLOCK_SIZE = 3
OPTIONAL_PORT_MIN = 20000
OPTIONAL_PORT_MAX_START = 55000
INSPECTION_SAFETY_COUNTER_KEYS = (
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
        "relaykv_optional_server_req_to_token_pool_value_shape_inspection_smoke_result="
        + json.dumps(result, sort_keys=True)
    )


def _clean_skip(reason: str, **extra: Any) -> int:
    result = {"skipped": True, "skip_reason": reason}
    result.update(extra)
    print("relaykv_optional_server_req_to_token_pool_value_shape_inspection_smoke: skip")
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


def _inspection_log_flags(stdout: str) -> dict[str, Any]:
    inspection_summary = _extract_summary_from_stdout(
        stdout,
        "relaykv_req_to_token_pool_value_shape_inspection_summary=",
    )
    observation_flags = _relaykv_observation_log_flags(stdout)
    return {
        **observation_flags,
        "relaykv_req_to_token_pool_value_shape_inspection_logged": (
            inspection_summary is not None
        ),
        "relaykv_req_to_token_pool_value_shape_inspection_summary": inspection_summary,
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


def _validate_inspection_summary(
    *,
    inspection_enabled: bool,
    inspection_log_flags: dict[str, Any],
) -> dict[str, Any] | None:
    summary = inspection_log_flags["relaykv_req_to_token_pool_value_shape_inspection_summary"]
    if not inspection_enabled:
        if inspection_log_flags["relaykv_req_to_token_pool_value_shape_inspection_logged"]:
            raise RuntimeError(
                "RelayKV req_to_token_pool value-shape inspection summary was emitted while inspection env was off"
            )
        return None

    if summary is None:
        raise RuntimeError(
            "RelayKV req_to_token_pool value-shape inspection summary log was not detected while inspection env was on"
        )
    if summary.get("event_type") != "relaykv_req_to_token_pool_value_shape_inspection_summary":
        raise RuntimeError("unexpected value-shape inspection summary event_type")
    if summary.get("hook_enabled") is not True:
        raise RuntimeError("inspection summary did not report hook_enabled=true")
    if summary.get("inspection_enabled") is not True:
        raise RuntimeError("inspection summary did not report inspection_enabled=true")
    for key in (
        "runtime_observation_source_bridge_state",
        "runtime_observation_source_bridge_payload_count",
        "req_to_token_pool_source_path",
        "result_count",
        "inspected_count",
        "blocked_count",
        "error_count",
        "total_probed_value_count",
        "observed_type_counts",
        "observed_shape_counts",
        "observed_dtype_counts",
        "observed_device_counts",
        "observed_scalar_like_count",
        "observed_one_element_like_count",
        "blocked_reason",
    ):
        if key not in summary:
            raise RuntimeError(f"missing inspection summary key: {key}")
    _assert_zero_safety_counters(summary, INSPECTION_SAFETY_COUNTER_KEYS)

    inspected_count = int(summary.get("inspected_count") or 0)
    blocked_count = int(summary.get("blocked_count") or 0)
    error_count = int(summary.get("error_count") or 0)
    req_read_count = int(summary.get("req_to_token_read_count") or 0)
    actual_req_read_count = int(summary.get("actual_req_to_token_pool_read_count") or 0)
    inspection_count = int(summary.get("req_to_token_value_shape_inspection_count") or 0)
    if inspected_count > 0:
        if req_read_count <= 0:
            raise RuntimeError("inspected summary had zero req_to_token_read_count")
        if actual_req_read_count <= 0:
            raise RuntimeError(
                "inspected summary had zero actual_req_to_token_pool_read_count"
            )
        if inspection_count <= 0:
            raise RuntimeError(
                "inspected summary had zero req_to_token_value_shape_inspection_count"
            )
        return summary
    if blocked_count > 0 or error_count > 0:
        return summary
    raise RuntimeError(
        "RelayKV req_to_token_pool value-shape inspection summary did not report an inspected, blocked, or error path"
    )


def _validate_case(
    *,
    observation_enabled: bool,
    inspection_enabled: bool,
    response: Any,
    inspection_log_flags: dict[str, Any],
) -> dict[str, Any]:
    _validate_observation_flags(
        observation_enabled=observation_enabled,
        inspection_log_flags=inspection_log_flags,
    )
    inspection_summary = _validate_inspection_summary(
        inspection_enabled=inspection_enabled,
        inspection_log_flags=inspection_log_flags,
    )
    result = {
        "observation_expected": observation_enabled,
        "observation_ran": inspection_log_flags["relaykv_observation_logged"],
        "inspection_expected": inspection_enabled,
        "inspection_ran": inspection_summary is not None,
        "inspected": False,
        "clean_blocked": False,
        "response_marker": _production_output_marker(response),
    }
    if inspection_summary is not None:
        if int(inspection_summary.get("inspected_count") or 0) > 0:
            result["inspected"] = True
        elif int(inspection_summary.get("blocked_count") or 0) > 0 or int(
            inspection_summary.get("error_count") or 0
        ) > 0:
            result["clean_blocked"] = True
    return result


def _run_server_case(
    model_path: str,
    *,
    observation_enabled: bool,
    inspection_enabled: bool,
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
    if inspection_enabled:
        env[VALUE_SHAPE_INSPECTION_ENV] = "1"
    else:
        env.pop(VALUE_SHAPE_INSPECTION_ENV, None)
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
            "text": "RelayKV req_to_token_pool value-shape inspection optional server smoke.",
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
            observation_enabled=observation_enabled,
            inspection_enabled=inspection_enabled,
            response=response,
            inspection_log_flags=inspection_log_flags,
        )
        return {
            "status": status,
            "response": response,
            "response_marker": validation["response_marker"],
            "observation_enabled": observation_enabled,
            "inspection_enabled": inspection_enabled,
            "observation_logged": inspection_log_flags["relaykv_observation_logged"],
            "value_shape_inspection_summary": inspection_log_flags[
                "relaykv_req_to_token_pool_value_shape_inspection_summary"
            ],
            "inspection_ran": validation["inspection_ran"],
            "inspected": validation["inspected"],
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
            "observation_enabled": False,
            "inspection_enabled": False,
        },
        {
            "name": "observation_only",
            "observation_enabled": True,
            "inspection_enabled": False,
        },
        {
            "name": "observation_value_shape_inspection",
            "observation_enabled": True,
            "inspection_enabled": True,
        },
    ]

    case_results = []
    for case in cases:
        result = _run_server_case(
            model_path,
            observation_enabled=case["observation_enabled"],
            inspection_enabled=case["inspection_enabled"],
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

    case1 = case_results[0]
    if case1["observation_logged"] or case1["inspection_ran"]:
        raise RuntimeError("all-off case unexpectedly emitted RelayKV summaries")

    case2 = case_results[1]
    if not case2["observation_logged"]:
        raise RuntimeError("observation-only case did not emit observation logs")
    if case2["inspection_ran"]:
        raise RuntimeError(
            "observation-only case unexpectedly emitted value-shape inspection summary"
        )

    case3 = case_results[2]
    if not case3["observation_logged"]:
        raise RuntimeError(
            "observation+value-shape-inspection case did not emit observation logs"
        )
    if not case3["inspection_ran"]:
        raise RuntimeError(
            "observation+value-shape-inspection case did not emit inspection summary"
        )

    result = {
        "skipped": False,
        "port_selection_bounds_check": port_bounds,
        "cases": case_results,
        "production_output_unaffected": True,
    }
    print("relaykv_optional_server_req_to_token_pool_value_shape_inspection_smoke=pass")
    _print_result(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
