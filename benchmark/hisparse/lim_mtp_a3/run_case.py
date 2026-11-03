#!/usr/bin/env python3
"""Run one active GLM-5.2 A3 case with owned-process cleanup and evidence."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from replay import request_json, validate_server

HERE = Path(__file__).resolve().parent
LAUNCHER = HERE / "launch_server.sh"
REPLAY = HERE / "replay.py"
DEFAULT_INPUT_SHA256 = (
    "ecf2ddbf180fc07b82dd670ab53a54dc4d4f00c2028d79cc81164850c06b1383"
)
FORMAL_OUTPUT_TOKENS = 512
SMOKE_INPUT_TOKENS = 512
SMOKE_OUTPUT_TOKENS = 32
LONG_CORRECTNESS_INPUT_TOKENS = 16384
LONG_CORRECTNESS_OUTPUT_TOKENS = 32
FORMAL_ROUNDS = 3

ACTIVE_CASES = {
    "full_64k_b5": {
        "input_tokens": 64000,
        "batch_size": 5,
        "offload": 0,
        "copy_sfa": 0,
        "metadata_fusion": 0,
    },
    "offload_64k_b16_copysfa": {
        "input_tokens": 64000,
        "batch_size": 16,
        "offload": 1,
        "copy_sfa": 1,
        "metadata_fusion": 1,
    },
}

# Explicit allowlist: never serialize the whole environment, proxies, credentials,
# or arbitrary SGLANG_* values into experiment artifacts.
SAFE_ENV_KEYS = (
    "CASE_ID",
    "PORT",
    "MTP",
    "GRAPH",
    "OUTPUT_TOKENS",
    "METADATA_FUSION",
    "PRE_METADATA_FUSION",
    "SGLANG_ENABLE_NPU_LIM_MTP_FULL_VISIBLE",
    "SGLANG_ENABLE_ASYNC_ASSERT",
    "SGLANG_SOURCE_ROOT",
    "MODEL_PATH",
    "KERNEL_PYTHON_ROOT",
    "EXPECTED_KERNEL_SO_PATH",
    "EXPECTED_KERNEL_SO_SHA256",
    "TASK_ROOT",
    "ASCEND_RT_VISIBLE_DEVICES",
    "ASCEND_VISIBLE_DEVICES",
    "NPU_VISIBLE_DEVICES",
    "ASCEND_HOME_PATH",
    "ASCEND_TOOLKIT_HOME",
    "CANN_HOME",
    "HCCL_BUFFSIZE",
    "HCCL_OP_EXPANSION_MODE",
    "HCCL_SOCKET_IFNAME",
    "GLOO_SOCKET_IFNAME",
    "HCCL_CONNECT_TIMEOUT",
    "HCCL_EXEC_TIMEOUT",
    "SGLANG_SET_CPU_AFFINITY",
    "SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS",
    "STREAMS_PER_DEVICE",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM",
    "SGLANG_NPU_PROFILING",
    "TRANSFORMERS_VERBOSITY",
    "SGLANG_NPU_SPARSE_KV_INDEXER",
    "SGLANG_NPU_ENABLE_SPARSE_KV_OFFLOAD",
    "SGLANG_NPU_ENABLE_COPY_SFA",
    "SGLANG_NPU_SPARSE_KV_CACHE_SIZE",
    "SGLANG_NPU_FUSED_LIM_PRE_METADATA",
    "SGLANG_NPU_FUSED_LIM_COPY_SFA_METADATA",
    "DEEPEP_HYBRID_DEPLOYMENT",
    "DEEPEP_NORMAL_LONG_SEQ_ROUND",
    "DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS",
    "DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ",
    "DEEP_NORMAL_MODE_USE_INT8_QUANT",
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_proc(pid: int) -> dict[str, Any] | None:
    try:
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
        fields = raw.rsplit(")", 1)[1].split()
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ")
        return {
            "pid": pid,
            "state": fields[0],
            "group": int(fields[2]),
            "session": int(fields[3]),
            "start_ticks": int(fields[19]),
            "cmdline": cmdline.decode("utf-8", errors="replace").strip(),
        }
    except (OSError, ValueError, IndexError):
        return None


def read_owned_process_environment(owner: dict[str, Any]) -> dict[str, str]:
    current = read_proc(owner["pid"])
    if (
        current is None
        or current["session"] != owner["session"]
        or current["group"] != owner["group"]
        or current["start_ticks"] != owner["start_ticks"]
    ):
        raise RuntimeError("server PID ownership changed before environment capture")
    raw = Path(f"/proc/{owner['pid']}/environ").read_bytes()
    process_env = {}
    for item in raw.split(b"\0"):
        if b"=" not in item:
            continue
        key, value = item.split(b"=", 1)
        process_env[key.decode("utf-8", errors="replace")] = value.decode(
            "utf-8", errors="replace"
        )
    actual = safe_env_snapshot(process_env)
    actual["_capture_pid"] = str(owner["pid"])
    actual["_capture_start_ticks"] = str(owner["start_ticks"])
    return actual


def owned_group_members(owner: dict[str, Any]) -> list[dict[str, Any]]:
    members = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        info = read_proc(int(entry.name))
        if not info or info["group"] != owner["group"] or info["state"] == "Z":
            continue
        if (
            info["session"] != owner["session"]
            or info["start_ticks"] < owner["start_ticks"]
        ):
            raise RuntimeError(
                "process-session ownership check failed; refusing cleanup"
            )
        members.append(info)
    return members


def stop_owned_process(
    process: subprocess.Popen, owner: dict[str, Any]
) -> dict[str, Any]:
    initial = owned_group_members(owner)
    if initial:
        os.killpg(owner["group"], signal.SIGTERM)
    deadline = time.monotonic() + 30
    while owned_group_members(owner) and time.monotonic() < deadline:
        time.sleep(0.25)
    escalated = owned_group_members(owner)
    if escalated:
        os.killpg(owner["group"], signal.SIGKILL)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        # Recheck the exact process group and start-time ownership before another signal.
        remaining = owned_group_members(owner)
        if remaining:
            os.killpg(owner["group"], signal.SIGKILL)
        process.wait(timeout=30)
    deadline = time.monotonic() + 10
    while owned_group_members(owner) and time.monotonic() < deadline:
        time.sleep(0.25)
    remaining = owned_group_members(owner)
    return {
        "initial_members": initial,
        "kill_escalation_members": escalated,
        "remaining": remaining,
        "process_exitcode": process.returncode,
    }


def launch_owner(process: subprocess.Popen, name: str) -> dict[str, Any]:
    info = read_proc(process.pid)
    if info is None or info["group"] != process.pid or info["session"] != process.pid:
        raise RuntimeError(f"{name} did not establish its own process session")
    return info


def port_is_free(port: int) -> bool:
    with socket.socket() as probe:
        probe.settimeout(1)
        return probe.connect_ex(("127.0.0.1", port)) != 0


def safe_env_snapshot(env: dict[str, str]) -> dict[str, str]:
    return {key: env[key] for key in SAFE_ENV_KEYS if key in env}


def predicted_launcher_env_snapshot(
    env: dict[str, str], case_id: str, mtp: bool
) -> dict[str, str]:
    case = ACTIVE_CASES[case_id]
    snapshot = safe_env_snapshot(env)
    snapshot.update(
        {
            "HCCL_BUFFSIZE": "512",
            "HCCL_OP_EXPANSION_MODE": "AIV",
            "HCCL_SOCKET_IFNAME": "lo",
            "GLOO_SOCKET_IFNAME": "lo",
            "HCCL_CONNECT_TIMEOUT": "600",
            "HCCL_EXEC_TIMEOUT": "600",
            "SGLANG_SET_CPU_AFFINITY": "1",
            "SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS": "1",
            "STREAMS_PER_DEVICE": "32",
            "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
            "SGLANG_NPU_PROFILING": "0",
            "TRANSFORMERS_VERBOSITY": "error",
            "SGLANG_NPU_SPARSE_KV_INDEXER": "lim" if case["offload"] else "legacy",
            "SGLANG_NPU_ENABLE_SPARSE_KV_OFFLOAD": str(case["offload"]),
            "SGLANG_NPU_ENABLE_COPY_SFA": str(case["copy_sfa"]),
            "SGLANG_NPU_SPARSE_KV_CACHE_SIZE": "12288",
            "SGLANG_NPU_FUSED_LIM_PRE_METADATA": "0",
            "SGLANG_NPU_FUSED_LIM_COPY_SFA_METADATA": (
                str(case["metadata_fusion"]) if mtp else "0"
            ),
            "SGLANG_ENABLE_NPU_LIM_MTP_FULL_VISIBLE": str(
                int(bool(mtp and case["offload"]))
            ),
            "SGLANG_ENABLE_ASYNC_ASSERT": env["SGLANG_ENABLE_ASYNC_ASSERT"],
            "DEEPEP_HYBRID_DEPLOYMENT": "1",
            "DEEPEP_NORMAL_LONG_SEQ_ROUND": "72",
            "DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS": "1024",
            "DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ": "1",
            "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",
        }
    )
    return snapshot


def run_client(
    *,
    command: list[str],
    env: dict[str, str],
    log_path: Path,
    exit_path: Path,
    timeout_s: int,
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("wb") as log:
        process = subprocess.Popen(
            command,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            owner = launch_owner(process, "client")
        except BaseException:
            process.terminate()
            process.wait(timeout=10)
            raise
        timed_out = False
        cleanup = None
        try:
            returncode = process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            cleanup = stop_owned_process(process, owner)
            returncode = cleanup["process_exitcode"]
        except BaseException as exc:
            try:
                cleanup = stop_owned_process(process, owner)
            finally:
                write_json(
                    exit_path,
                    {
                        "command": command,
                        "pid": owner["pid"],
                        "process_group": owner["group"],
                        "session": owner["session"],
                        "start_ticks": owner["start_ticks"],
                        "cmdline": owner["cmdline"],
                        "timed_out": False,
                        "interrupted": f"{type(exc).__name__}: {exc}",
                        "returncode": process.poll(),
                        "cleanup": cleanup,
                    },
                )
            raise
        details = {
            "command": command,
            "pid": owner["pid"],
            "process_group": owner["group"],
            "session": owner["session"],
            "start_ticks": owner["start_ticks"],
            "cmdline": owner["cmdline"],
            "timed_out": timed_out,
            "returncode": returncode,
        }
        if cleanup is not None:
            details["cleanup"] = cleanup
        write_json(exit_path, details)
        return returncode if returncode is not None else 1


def verify_frozen_input(
    path: Path,
    expected_sha256: str,
    request_count: int,
    minimum_tokens: int,
) -> None:
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ValueError(
            f"frozen input SHA mismatch: expected {expected_sha256}, got {actual}"
        )
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        for row_index in range(request_count):
            line = stream.readline()
            if not line:
                raise ValueError(
                    f"frozen input has fewer than {request_count} request rows"
                )
            row = json.loads(line)
            token_ids = row.get("token_ids")
            if not isinstance(token_ids, list) or len(token_ids) < minimum_tokens:
                raise ValueError(
                    f"frozen input row {row_index} must contain at least "
                    f"{minimum_tokens} token_ids"
                )


def git_output(source_root: Path, *args: str) -> str:
    return (
        subprocess.check_output(
            ["git", "-C", str(source_root), *args], stderr=subprocess.STDOUT
        )
        .decode("utf-8", errors="replace")
        .strip()
    )


def provenance(args: argparse.Namespace) -> dict[str, Any]:
    source_root = args.source_root.resolve()
    model_config = args.model_config.resolve()
    kernel_so = args.kernel_so.resolve()
    diff = subprocess.check_output(
        ["git", "-C", str(source_root), "diff", "--binary", "HEAD"]
    )
    return {
        "source_root": str(source_root),
        "source_head": git_output(source_root, "rev-parse", "HEAD"),
        "source_branch": git_output(source_root, "branch", "--show-current"),
        "source_status_porcelain": git_output(source_root, "status", "--porcelain"),
        "source_tracked_diff_sha256": sha256_bytes(diff),
        "kernel_so_path": str(kernel_so),
        "kernel_so_sha256": sha256_file(kernel_so),
        "model_config_path": str(model_config),
        "model_config_sha256": sha256_file(model_config),
        "input_path": str(args.requests.resolve()),
        "input_sha256": sha256_file(args.requests),
        "scripts_sha256": {
            path.name: sha256_file(path)
            for path in (Path(__file__).resolve(), LAUNCHER, REPLAY)
        },
        "python": sys.version,
    }


def launch_env(
    *,
    args: argparse.Namespace,
    case_id: str,
    batch_size: int,
    mtp: bool,
    port: int,
) -> dict[str, str]:
    case = ACTIVE_CASES[case_id]
    env = os.environ.copy()
    env.update(
        CASE_ID=case_id,
        SGLANG_SOURCE_ROOT=str(args.source_root.resolve()),
        MODEL_PATH=str(args.model_path.resolve()),
        KERNEL_PYTHON_ROOT=str(args.kernel_python_root.resolve()),
        EXPECTED_KERNEL_SO_PATH=str(args.kernel_so.resolve()),
        EXPECTED_KERNEL_SO_SHA256=sha256_file(args.kernel_so),
        TASK_ROOT=str(args.task_root.resolve()),
        PORT=str(port),
        MTP="1" if mtp else "0",
        GRAPH="1",
        OUTPUT_TOKENS=str(FORMAL_OUTPUT_TOKENS),
        METADATA_FUSION=str(case["metadata_fusion"] if mtp else 0),
        PRE_METADATA_FUSION="0",
        SGLANG_ENABLE_NPU_LIM_MTP_FULL_VISIBLE=(
            "1" if mtp and case["offload"] else "0"
        ),
        SGLANG_ENABLE_ASYNC_ASSERT=str(args.async_assert),
    )
    return env


def replay_command(
    *,
    args: argparse.Namespace,
    case_id: str,
    base_url: str,
    output_path: Path,
    input_tokens: int,
    batch_size: int,
    output_tokens: int,
    mtp: bool,
) -> list[str]:
    command = [
        sys.executable,
        str(REPLAY),
        "--base-url",
        base_url,
        "--requests",
        str(args.requests.resolve()),
        "--input-sha256",
        args.input_sha256,
        "--output",
        str(output_path),
        "--case-id",
        case_id,
        "--input-tokens",
        str(input_tokens),
        "--batch-size",
        str(batch_size),
        "--output-tokens",
        str(output_tokens),
        "--timeout",
        str(args.client_timeout),
    ]
    if not mtp:
        command.append("--mtp-off-control")
    return command


def run_replay_step(
    *,
    args: argparse.Namespace,
    case_id: str,
    base_url: str,
    env: dict[str, str],
    run_dir: Path,
    label: str,
    input_tokens: int,
    batch_size: int,
    output_tokens: int,
    mtp: bool,
) -> dict[str, Any]:
    output_path = run_dir / f"{label}.json"
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite client result: {output_path}")
    command = replay_command(
        args=args,
        case_id=case_id,
        base_url=base_url,
        output_path=output_path,
        input_tokens=input_tokens,
        batch_size=batch_size,
        output_tokens=output_tokens,
        mtp=mtp,
    )
    returncode = run_client(
        command=command,
        env=env,
        log_path=run_dir / f"{label}.client.log",
        exit_path=run_dir / f"{label}.client_exitcode.json",
        timeout_s=args.client_timeout + 60,
    )
    result = {
        "label": label,
        "command": command,
        "returncode": returncode,
        "output_path": str(output_path),
        "log_path": str(run_dir / f"{label}.client.log"),
    }
    if output_path.exists():
        result_json = json.loads(output_path.read_text(encoding="utf-8"))
        result["output_sha256"] = sha256_file(output_path)
        result["summary"] = result_json.get("summary")
        if result_json.get("results"):
            result["output_text_sha256_by_request"] = [
                row.get("output_text_sha256") for row in result_json["results"]
            ]
            result["output_text_sha256"] = result_json["results"][0].get(
                "output_text_sha256"
            )
    if returncode != 0:
        raise RuntimeError(
            f"client step {label} failed with exit code {returncode}; evidence retained in {run_dir}"
        )
    return result


def start_server(
    *,
    args: argparse.Namespace,
    case_id: str,
    batch_size: int,
    mtp: bool,
    run_dir: Path,
    port: int,
    state: dict[str, Any],
) -> tuple[subprocess.Popen, dict[str, Any], dict[str, Any], dict[str, str]]:
    if not port_is_free(port):
        raise RuntimeError(f"127.0.0.1:{port} is already in use; refusing to touch it")
    env = launch_env(
        args=args, case_id=case_id, batch_size=batch_size, mtp=mtp, port=port
    )
    write_json(
        run_dir / "environment_predicted.json",
        predicted_launcher_env_snapshot(env, case_id, mtp),
    )
    launch_command = ["bash", str(LAUNCHER)]
    with (run_dir / "server.log").open("wb") as log:
        process = subprocess.Popen(
            launch_command,
            cwd=args.source_root,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    try:
        owner = launch_owner(process, "server")
    except BaseException:
        process.terminate()
        process.wait(timeout=10)
        raise
    server_record = {
        "mode": "mtp" if mtp else "mtp_off_control",
        "launch_command": launch_command,
        "pid": owner["pid"],
        "process_group": owner["group"],
        "session": owner["session"],
        "start_ticks": owner["start_ticks"],
        "cmdline": owner["cmdline"],
        "log_path": str(run_dir / "server.log"),
        "environment_predicted": str(run_dir / "environment_predicted.json"),
    }
    state.setdefault("servers", []).append(server_record)
    write_json(args.output_dir / "status.json", state)

    try:
        deadline = time.monotonic() + args.ready_timeout
        base_url = f"http://127.0.0.1:{port}"
        while True:
            if process.poll() is not None:
                raise RuntimeError(
                    f"server exited before readiness with code {process.returncode}; see {run_dir / 'server.log'}"
                )
            try:
                info = request_json(base_url + "/server_info", timeout=3)
                config = validate_server(info, "NEXTN" if mtp else None)
                if config.get("max_running_requests") != batch_size:
                    raise RuntimeError(
                        "resolved max_running_requests differs from global batch size"
                    )
                if mtp and config.get("speculative_algorithm") != "EAGLE":
                    raise RuntimeError(
                        "GLM-5.2 MTP resolved algorithm must be EAGLE; "
                        f"got {config.get('speculative_algorithm')!r}"
                    )
                actual_env = read_owned_process_environment(owner)
                write_json(run_dir / "environment_actual.json", actual_env)
                server_record["environment_actual"] = str(
                    run_dir / "environment_actual.json"
                )
                write_json(run_dir / "server-info.json", info)
                server_record["resolved_server_info"] = str(
                    run_dir / "server-info.json"
                )
                server_record["resolved_speculative_algorithm"] = config.get(
                    "speculative_algorithm"
                )
                server_record["topology_validated"] = True
                server_record["runtime_path_qualification"] = (
                    "not established by HTTP readiness or client smoke"
                )
                write_json(args.output_dir / "status.json", state)
                return process, owner, server_record, env
            except (
                OSError,
                TimeoutError,
                urllib.error.URLError,
                json.JSONDecodeError,
            ):
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        f"server readiness timeout; see {run_dir / 'server.log'}"
                    )
                time.sleep(2)
    except BaseException:
        cleanup = stop_owned_process(process, owner)
        write_json(run_dir / "cleanup.json", cleanup)
        state.setdefault("server_cleanups", []).append(
            {"run_dir": str(run_dir), **cleanup}
        )
        write_json(args.output_dir / "status.json", state)
        raise


def stop_server(
    process: subprocess.Popen,
    owner: dict[str, Any],
    run_dir: Path,
    state: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    cleanup = stop_owned_process(process, owner)
    write_json(run_dir / "cleanup.json", cleanup)
    state.setdefault("server_cleanups", []).append({"run_dir": str(run_dir), **cleanup})
    write_json(args.output_dir / "status.json", state)
    if cleanup["remaining"]:
        raise RuntimeError(
            f"owned server session remains alive: {cleanup['remaining']}"
        )
    return cleanup


def run_server_phase(
    *,
    args: argparse.Namespace,
    state: dict[str, Any],
    case_id: str,
    batch_size: int,
    mtp: bool,
    phase_dir: Path,
    port: int,
    work,
) -> Any:
    process = None
    owner = None
    try:
        process, owner, server_record, env = start_server(
            args=args,
            case_id=case_id,
            batch_size=batch_size,
            mtp=mtp,
            run_dir=phase_dir,
            port=port,
            state=state,
        )
        result = work(server_record, env)
        try:
            after = request_json(f"http://127.0.0.1:{port}/server_info", timeout=5)
            write_json(phase_dir / "server-info-after.json", after)
            server_record["server_info_after"] = str(
                phase_dir / "server-info-after.json"
            )
            write_json(args.output_dir / "status.json", state)
        except Exception as exc:
            write_json(phase_dir / "server-info-after-error.json", {"error": str(exc)})
        return result
    finally:
        if process is not None and owner is not None:
            stop_server(process, owner, phase_dir, state, args)


def plan_commands(args: argparse.Namespace, case_id: str, port: int) -> dict[str, Any]:
    case = ACTIVE_CASES[case_id]
    env = launch_env(
        args=args,
        case_id=case_id,
        batch_size=case["batch_size"],
        mtp=True,
        port=port,
    )
    dry_env = predicted_launcher_env_snapshot(env, case_id, mtp=True)
    dry_env.update(
        {
            "CASE_ID": case_id,
            "MTP": "1",
            "METADATA_FUSION": str(case["metadata_fusion"]),
            "PRE_METADATA_FUSION": "0",
        }
    )
    launch_previews = {}
    for mtp in (False, True):
        preview_env = launch_env(
            args=args,
            case_id=case_id,
            batch_size=case["batch_size"],
            mtp=mtp,
            port=port,
        )
        preview = subprocess.run(
            ["bash", str(LAUNCHER), "--dry-run"],
            cwd=args.source_root,
            env=preview_env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        if preview.returncode:
            raise RuntimeError(f"launcher dry-run failed: {preview.stdout}")
        launch_previews["mtp" if mtp else "mtp_off_control"] = preview.stdout
    commands = []
    phases = [
        ("correctness_mtp_off", 512, 1, SMOKE_OUTPUT_TOKENS, False),
    ]
    if not args.smoke_only:
        phases.append(
            (
                "long_correctness_mtp_off",
                LONG_CORRECTNESS_INPUT_TOKENS,
                case["batch_size"],
                LONG_CORRECTNESS_OUTPUT_TOKENS,
                False,
            )
        )
    phases.append(("correctness_mtp_on", 512, 1, SMOKE_OUTPUT_TOKENS, True))
    if not args.smoke_only:
        phases.append(
            (
                "long_correctness_mtp_on",
                LONG_CORRECTNESS_INPUT_TOKENS,
                case["batch_size"],
                LONG_CORRECTNESS_OUTPUT_TOKENS,
                True,
            )
        )
    if not (args.smoke_only or args.correctness_only):
        phases.append(
            (
                "warmup",
                args.warmup_input_tokens,
                case["batch_size"],
                args.warmup_output_tokens,
                True,
            )
        )
        phases.extend(
            (
                f"round-{round_id:02d}",
                64000,
                case["batch_size"],
                FORMAL_OUTPUT_TOKENS,
                True,
            )
            for round_id in range(1, FORMAL_ROUNDS + 1)
        )
    for label, input_tokens, batch_size, output_tokens, mtp in phases:
        commands.append(
            replay_command(
                args=args,
                case_id=case_id,
                base_url=f"http://127.0.0.1:{port}",
                output_path=args.output_dir / label / f"{label}.json",
                input_tokens=input_tokens,
                batch_size=batch_size,
                output_tokens=output_tokens,
                mtp=mtp,
            )
        )
    return {
        "case_id": case_id,
        "global_batch_size": case["batch_size"],
        "formal_rounds": 0
        if args.smoke_only or args.correctness_only
        else FORMAL_ROUNDS,
        "formal_output_tokens_assumption": FORMAL_OUTPUT_TOKENS,
        "warmup_input_tokens": None
        if args.smoke_only or args.correctness_only
        else args.warmup_input_tokens,
        "warmup_output_tokens": None
        if args.smoke_only or args.correctness_only
        else args.warmup_output_tokens,
        "launcher_dry_run_stdout": launch_previews,
        "selected_non_sensitive_environment": dry_env,
        "replay_commands": commands,
        "runtime_path_qualification": "not established by dry-run or HTTP client results",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-id", choices=sorted(ACTIVE_CASES), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--input-sha256", default=DEFAULT_INPUT_SHA256)
    parser.add_argument(
        "--source-root", type=Path, default=os.environ.get("SGLANG_SOURCE_ROOT")
    )
    parser.add_argument("--model-path", type=Path, default=os.environ.get("MODEL_PATH"))
    parser.add_argument("--model-config", type=Path)
    parser.add_argument(
        "--kernel-python-root", type=Path, default=os.environ.get("KERNEL_PYTHON_ROOT")
    )
    parser.add_argument("--kernel-so", type=Path, required=True)
    parser.add_argument(
        "--task-root",
        type=Path,
        default=os.environ.get(
            "TASK_ROOT", str(Path(tempfile.gettempdir()) / "sglang-lim-mtp-a3")
        ),
    )
    parser.add_argument(
        "--port", type=int, default=int(os.environ.get("PORT", "18898"))
    )
    parser.add_argument("--warmup-input-tokens", type=int, default=16384)
    parser.add_argument("--warmup-output-tokens", type=int, default=16)
    parser.add_argument("--async-assert", choices=("0", "1"), default="0")
    parser.add_argument("--ready-timeout", type=int, default=1800)
    parser.add_argument("--client-timeout", type=int, default=3600)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--smoke-only", action="store_true")
    mode.add_argument("--correctness-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.source_root is None:
        parser.error("--source-root or SGLANG_SOURCE_ROOT is required")
    if args.model_path is None:
        parser.error("--model-path or MODEL_PATH is required")
    if args.kernel_python_root is None:
        parser.error("--kernel-python-root or KERNEL_PYTHON_ROOT is required")
    args.source_root = Path(args.source_root)
    args.model_path = Path(args.model_path)
    args.kernel_python_root = Path(args.kernel_python_root)
    args.task_root = Path(args.task_root)
    args.requests = args.requests.resolve()
    args.output_dir = args.output_dir.resolve()
    args.model_config = (
        args.model_config.resolve()
        if args.model_config
        else (args.model_path / "config.json").resolve()
    )
    args.kernel_so = args.kernel_so.resolve()
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    if min(args.warmup_input_tokens, args.warmup_output_tokens) <= 0:
        parser.error("port and warmup token counts must be positive")
    if min(args.ready_timeout, args.client_timeout) <= 0:
        parser.error("timeouts must be positive")
    return args


def main() -> int:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(
            f"output directory exists; refusing overwrite: {args.output_dir}"
        )
    if (
        not args.requests.is_file()
        or not args.model_config.is_file()
        or not args.kernel_so.is_file()
    ):
        raise FileNotFoundError(
            "requests, model config, and kernel SO must be existing files"
        )
    case = ACTIVE_CASES[args.case_id]
    verify_frozen_input(
        args.requests,
        args.input_sha256,
        request_count=case["batch_size"],
        minimum_tokens=case["input_tokens"],
    )
    if args.dry_run:
        print(json.dumps(plan_commands(args, args.case_id, args.port), indent=2))
        return 0
    if not sys.platform.startswith("linux") or not Path("/proc").is_dir():
        raise RuntimeError(
            "real server execution requires Linux /proc process ownership checks"
        )

    args.output_dir.mkdir(parents=True)
    case = ACTIVE_CASES[args.case_id]
    state: dict[str, Any] = {
        "status": "running",
        "case_id": args.case_id,
        "started_unix": time.time(),
        "assumptions": {
            "active_cases_only": True,
            "global_batch_size": case["batch_size"],
            "input_tokens": 64000,
            "formal_output_tokens": FORMAL_OUTPUT_TOKENS,
            "formal_rounds": 0
            if args.smoke_only or args.correctness_only
            else FORMAL_ROUNDS,
            "warmup_input_tokens": None
            if args.smoke_only or args.correctness_only
            else args.warmup_input_tokens,
            "warmup_output_tokens": None
            if args.smoke_only or args.correctness_only
            else args.warmup_output_tokens,
            "async_assert": args.async_assert,
            "correctness_smoke": "one 512-token input, 32 output tokens, greedy MTP-off vs MTP-on text hash",
            "long_correctness": None
            if args.smoke_only
            else {
                "input_tokens": LONG_CORRECTNESS_INPUT_TOKENS,
                "output_tokens": LONG_CORRECTNESS_OUTPUT_TOKENS,
                "batch_size": case["batch_size"],
                "comparison": "per-request greedy MTP-off vs MTP-on output text SHA256",
                "native_path_qualification": "requires separate runtime path evidence",
            },
            "formal_performance_qualification": "requires manual native LIM/CopySFA path qualification; HTTP success alone is insufficient",
        },
        "provenance": provenance(args),
        "servers": [],
        "server_cleanups": [],
        "client_steps": [],
    }
    write_json(args.output_dir / "status.json", state)

    def interrupted(signum, _frame):
        raise KeyboardInterrupt(f"controlled stop signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    result_code = 0
    try:
        off_dir = args.output_dir / "correctness" / "mtp-off-control"
        off_dir.mkdir(parents=True)

        def mtp_off_work(_server, env):
            smoke = run_replay_step(
                args=args,
                case_id=args.case_id,
                base_url=f"http://127.0.0.1:{args.port}",
                env=env,
                run_dir=off_dir,
                label="smoke",
                input_tokens=SMOKE_INPUT_TOKENS,
                batch_size=1,
                output_tokens=SMOKE_OUTPUT_TOKENS,
                mtp=False,
            )
            result = {"smoke": smoke}
            if not args.smoke_only:
                result["long"] = run_replay_step(
                    args=args,
                    case_id=args.case_id,
                    base_url=f"http://127.0.0.1:{args.port}",
                    env=env,
                    run_dir=off_dir,
                    label="long-correctness",
                    input_tokens=LONG_CORRECTNESS_INPUT_TOKENS,
                    batch_size=case["batch_size"],
                    output_tokens=LONG_CORRECTNESS_OUTPUT_TOKENS,
                    mtp=False,
                )
            return result

        off_result = run_server_phase(
            args=args,
            state=state,
            case_id=args.case_id,
            batch_size=case["batch_size"],
            mtp=False,
            phase_dir=off_dir,
            port=args.port,
            work=mtp_off_work,
        )
        state["client_steps"].append({"phase": "correctness_mtp_off", **off_result})
        write_json(args.output_dir / "status.json", state)

        on_dir = args.output_dir / "correctness" / "mtp-on"
        on_dir.mkdir(parents=True)

        def mtp_target_work(server_record, env):
            on_result = run_replay_step(
                args=args,
                case_id=args.case_id,
                base_url=f"http://127.0.0.1:{args.port}",
                env=env,
                run_dir=on_dir,
                label="smoke",
                input_tokens=SMOKE_INPUT_TOKENS,
                batch_size=1,
                output_tokens=SMOKE_OUTPUT_TOKENS,
                mtp=True,
            )
            state["client_steps"].append({"phase": "correctness_mtp_on", **on_result})
            off_text_hash = off_result["smoke"]["output_text_sha256"]
            on_text_hash = on_result["output_text_sha256"]
            state["correctness_smoke"] = {
                "matched_greedy_output_hash": off_text_hash == on_text_hash,
                "mtp_off_sha256": off_text_hash,
                "mtp_on_sha256": on_text_hash,
                "native_path_qualified": False,
            }
            write_json(args.output_dir / "status.json", state)
            if off_text_hash != on_text_hash:
                raise RuntimeError(
                    "MTP-on correctness smoke text hash differs from MTP-off control"
                )
            if args.smoke_only:
                return {"correctness_smoke": state["correctness_smoke"]}

            long_result = run_replay_step(
                args=args,
                case_id=args.case_id,
                base_url=f"http://127.0.0.1:{args.port}",
                env=env,
                run_dir=on_dir,
                label="long-correctness",
                input_tokens=LONG_CORRECTNESS_INPUT_TOKENS,
                batch_size=case["batch_size"],
                output_tokens=LONG_CORRECTNESS_OUTPUT_TOKENS,
                mtp=True,
            )
            off_hashes = off_result["long"]["output_text_sha256_by_request"]
            on_hashes = long_result["output_text_sha256_by_request"]
            state["long_correctness"] = {
                "mtp_off_sha256_by_request": off_hashes,
                "mtp_on_sha256_by_request": on_hashes,
                "matched_all_requests": (
                    len(off_hashes) == len(on_hashes) == case["batch_size"]
                    and None not in off_hashes
                    and off_hashes == on_hashes
                ),
                "native_path_qualified": False,
            }
            state["client_steps"].append(
                {"phase": "long_correctness_mtp_on", **long_result}
            )
            write_json(args.output_dir / "status.json", state)
            if not state["long_correctness"]["matched_all_requests"]:
                raise RuntimeError(
                    "long-input MTP-on output differs from MTP-off control; formal rounds blocked"
                )
            if args.correctness_only:
                return {
                    "correctness_smoke": state["correctness_smoke"],
                    "long_correctness": state["long_correctness"],
                    "runtime_path_qualification": "requires separate runtime path evidence",
                }

            target_dir = args.output_dir / "target_mtp"
            target_dir.mkdir()
            warmup = run_replay_step(
                args=args,
                case_id=args.case_id,
                base_url=f"http://127.0.0.1:{args.port}",
                env=env,
                run_dir=target_dir,
                label="warmup",
                input_tokens=args.warmup_input_tokens,
                batch_size=case["batch_size"],
                output_tokens=args.warmup_output_tokens,
                mtp=True,
            )
            state["warmup"] = warmup
            write_json(args.output_dir / "status.json", state)
            rounds = []
            for round_id in range(1, FORMAL_ROUNDS + 1):
                round_dir = target_dir / f"round-{round_id:02d}"
                round_dir.mkdir()
                result = run_replay_step(
                    args=args,
                    case_id=args.case_id,
                    base_url=f"http://127.0.0.1:{args.port}",
                    env=env,
                    run_dir=round_dir,
                    label=f"formal-{round_id:02d}",
                    input_tokens=64000,
                    batch_size=case["batch_size"],
                    output_tokens=FORMAL_OUTPUT_TOKENS,
                    mtp=True,
                )
                rounds.append(result)
                state["formal_rounds"] = rounds
                write_json(args.output_dir / "status.json", state)
            state["runtime_path_qualification"] = (
                "not established by topology, HTTP readiness, acceptance metadata, or greedy smoke"
            )
            return {"warmup": warmup, "formal_rounds": rounds}

        mtp_results = run_server_phase(
            args=args,
            state=state,
            case_id=args.case_id,
            batch_size=case["batch_size"],
            mtp=True,
            phase_dir=on_dir,
            port=args.port,
            work=mtp_target_work,
        )
        state["target_results"] = mtp_results

        state["status"] = "complete"
    except BaseException as exc:
        state["status"] = "failed"
        state["error"] = f"{type(exc).__name__}: {exc}"
        result_code = 1
        raise
    finally:
        state["finished_unix"] = time.time()
        write_json(args.output_dir / "status.json", state)
    return result_code


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(130)
