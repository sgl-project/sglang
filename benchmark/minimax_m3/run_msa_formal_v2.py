#!/usr/bin/env python3
"""Public, fail-closed MiniMax-M3 accuracy and serving-speed validation runner."""

from __future__ import annotations

import argparse
import csv
import hashlib
import http.client
import json
import math
import os
import re
import shlex
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path


CONCURRENCIES = (1, 8, 32, 128)
NUM_PROMPTS = 256
SEED = 20260819
GPQA_TOTAL = 198
LONGBENCH_TOTAL = 100
GPQA_MARGIN_QUESTIONS = 1
LONGBENCH_MARGIN = 0.02
THERMAL_QUERY = (
    "timestamp,index,temperature.gpu,power.draw,clocks.current.sm,clocks.max.sm,"
    "clocks_throttle_reasons.sw_thermal_slowdown,"
    "clocks_throttle_reasons.hw_thermal_slowdown"
)
FAILURE_PATTERNS = {
    "errors": re.compile(
        r"Traceback \(most recent call last\)|CUDA error|RuntimeError:|Exception:|"
        r"(?:^|[\s:\[])FATAL(?:[\s:\]]|$)|(?:^|[\s:\[])ERROR(?:[\s:\]]|$)|"
        r"Segmentation fault|illegal memory|NCCL[^\n]*(?:error|abort)|"
        r"request[^\n]*(?:failed|failure)",
        re.IGNORECASE,
    ),
    "retries": re.compile(r"\bretr(?:y|ies|ied|ying)\b", re.IGNORECASE),
    "jit_or_compilation": re.compile(
        r"\b(?:JIT|NVRTC|ninja|compil(?:e|ed|ing|ation))\b", re.IGNORECASE
    ),
}
CLIENT_FAILURE_PATTERNS = {
    "errors": re.compile(
        r"Traceback \(most recent call last\)|\b(?:CUDA )?error\b|RuntimeError:|"
        r"\bException\b|Segmentation fault|illegal memory|NCCL[^\n]*(?:error|abort)|"
        r"\b(?:failed|failure|exhausted)\b|connection (?:reset|refused)",
        re.IGNORECASE,
    ),
    "retries": re.compile(r"\bretr(?:y|ies|ied|ying)\b", re.IGNORECASE),
    "timeouts": re.compile(r"\b(?:timed out|timeouts?)\b", re.IGNORECASE),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_sha256(path: Path, expected: str, label: str) -> None:
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(f"{label} SHA-256 mismatch: {actual} != {expected}")


def verify_file_manifest(
    root: Path,
    manifest_path: Path,
    expected_manifest_sha256: str,
    expected_aggregate_sha256: str,
    label: str,
) -> dict:
    """Re-hash an immutable directory against an exact file manifest."""
    require_sha256(manifest_path, expected_manifest_sha256, f"{label} manifest")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != 1 or manifest.get("status") != "pass":
        raise RuntimeError(f"invalid {label} manifest header: {manifest}")
    if manifest.get("aggregate_sha256") != expected_aggregate_sha256:
        raise RuntimeError(
            f"{label} manifest aggregate mismatch: "
            f"{manifest.get('aggregate_sha256')!r} != {expected_aggregate_sha256!r}"
        )
    expected_rows = manifest.get("files")
    if not isinstance(expected_rows, list) or not expected_rows:
        raise RuntimeError(f"{label} manifest contains no files")
    expected: dict[str, dict] = {}
    for row in expected_rows:
        relative = row.get("path") if isinstance(row, dict) else None
        if (
            not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or relative in expected
            or not isinstance(row.get("size"), int)
            or row["size"] < 0
            or re.fullmatch(r"[0-9a-f]{64}", str(row.get("sha256", ""))) is None
        ):
            raise RuntimeError(f"invalid {label} manifest row: {row!r}")
        expected[relative] = row
    actual_paths = {
        path.relative_to(root).as_posix(): path
        for path in root.rglob("*")
        if path.is_file()
    }
    if set(actual_paths) != set(expected):
        raise RuntimeError(
            f"{label} file-set mismatch: missing={sorted(set(expected) - set(actual_paths))}, "
            f"unexpected={sorted(set(actual_paths) - set(expected))}"
        )
    aggregate = hashlib.sha256()
    total_bytes = 0
    for relative in sorted(expected):
        path = actual_paths[relative]
        size = path.stat().st_size
        digest = sha256(path)
        row = expected[relative]
        if size != row["size"] or digest != row["sha256"]:
            raise RuntimeError(
                f"{label} file mismatch for {relative}: "
                f"size={size}/{row['size']} sha256={digest}/{row['sha256']}"
            )
        aggregate.update(
            relative.encode()
            + b"\0"
            + str(size).encode()
            + b"\0"
            + digest.encode()
            + b"\n"
        )
        total_bytes += size
    actual_aggregate = aggregate.hexdigest()
    if actual_aggregate != expected_aggregate_sha256:
        raise RuntimeError(
            f"{label} runtime aggregate mismatch: "
            f"{actual_aggregate} != {expected_aggregate_sha256}"
        )
    if manifest.get("file_count") != len(expected) or manifest.get("total_bytes") != total_bytes:
        raise RuntimeError(f"{label} manifest count/size metadata mismatch")
    return {
        "schema_version": 1,
        "status": "pass",
        "label": label,
        "runtime_root": str(root.resolve()),
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": expected_manifest_sha256,
        "aggregate_sha256": actual_aggregate,
        "file_count": len(expected),
        "total_bytes": total_bytes,
    }


def command_text(command: list[str]) -> str:
    return shlex.join(command)


def loopback_environment(base: dict[str, str]) -> dict[str, str]:
    result = dict(base)
    result.setdefault("OPENAI_API_KEY", "EMPTY")
    return result


def roles_for_mode(mode: str) -> tuple[str, str]:
    if mode in ("accuracy", "external-speed"):
        return ("external", "flashinfer")
    if mode == "triton-speed":
        return ("triton", "flashinfer")
    raise ValueError(f"unsupported mode: {mode}")


def expected_order(mode: str, repetition: int) -> list[str]:
    left, right = roles_for_mode(mode)
    return [left, right] if repetition % 2 else [right, left]


def validate_order(mode: str, repetition: int, observed: list[str]) -> None:
    expected = expected_order(mode, repetition)
    if observed != expected:
        raise RuntimeError(
            f"wrong {mode} rep{repetition:02d} order: {observed} != {expected}"
        )


def fixed_parity_required(mode: str) -> bool:
    return mode == "accuracy"


def within_noninferiority_margin(delta: float, margin: float) -> bool:
    return delta >= -margin or math.isclose(
        delta, -margin, rel_tol=0.0, abs_tol=1e-12
    )


def route_fragments(role: str) -> tuple[str, ...]:
    if role == "external":
        return (
            "main_attn=fmha_sm100",
            "msa_decode=True",
            "msa_owns_decode=True",
            "decode_cuda_graph=True",
        )
    if role == "flashinfer":
        return (
            "main_attn=flashinfer",
            "msa_decode=True",
            "msa_owns_decode=True",
            "decode_cuda_graph=True",
        )
    if role == "triton":
        return (
            "main_attn=triton",
            "msa_decode=False",
            "msa_owns_decode=False",
            "decode_cuda_graph=True",
        )
    raise ValueError(f"unsupported role: {role}")


def role_environment(base: dict[str, str], role: str) -> dict[str, str]:
    result = dict(base)
    for name in (
        "SGLANG_DISABLE_MSA",
        "SGLANG_MINIMAX_MSA_BACKEND",
        "SGLANG_OPT_USE_MSA_DECODE_UNDER_GRAPH",
    ):
        result.pop(name, None)
    if role == "external":
        result.update(
            {
                "SGLANG_MINIMAX_MSA_BACKEND": "fmha_sm100",
                "SGLANG_OPT_USE_MSA_DECODE_UNDER_GRAPH": "1",
            }
        )
    elif role == "flashinfer":
        result.update(
            {
                "SGLANG_MINIMAX_MSA_BACKEND": "flashinfer",
                "SGLANG_OPT_USE_MSA_DECODE_UNDER_GRAPH": "1",
            }
        )
    elif role == "triton":
        result["SGLANG_DISABLE_MSA"] = "1"
    else:
        raise ValueError(f"unsupported role: {role}")
    return result


def server_command(args: argparse.Namespace) -> list[str]:
    return [
        args.python,
        "-m",
        "sglang.launch_server",
        "--model-path",
        args.model,
        "--trust-remote-code",
        "--reasoning-parser",
        "auto",
        "--tool-call-parser",
        "auto",
        "--tp",
        "4",
        "--attention-backend",
        "fa4",
        "--page-size",
        "128",
        "--moe-runner-backend",
        "deep_gemm",
        "--chunked-prefill-size",
        "8192",
        "--mem-fraction-static",
        "0.75",
        "--disable-prefill-cuda-graph",
        "--random-seed",
        str(SEED),
        "--host",
        args.host,
        "--port",
        str(args.port),
    ]


def server_healthy(base_url: str, timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(
            base_url.rstrip("/") + "/health_generate", timeout=timeout
        ) as response:
            return response.status == 200
    except (OSError, urllib.error.URLError, http.client.HTTPException):
        return False


def tail(path: Path, lines: int = 120) -> str:
    if not path.exists():
        return "<server log does not exist>"
    return "\n".join(path.read_text(errors="replace").splitlines()[-lines:])


def wait_for_server(
    process: subprocess.Popen, base_url: str, log_path: Path, timeout: int
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError(
                f"server exited with status {return_code} before readiness:\n"
                + tail(log_path)
            )
        if server_healthy(base_url, timeout=5):
            return
        time.sleep(5)
    raise TimeoutError(f"server was not ready after {timeout}s:\n" + tail(log_path))


def stop_server(process: subprocess.Popen, base_url: str) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=120)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=30)
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        if not server_healthy(base_url):
            return
        time.sleep(2)
    raise RuntimeError(f"server still answers at {base_url} after shutdown")


def validate_route(log_path: Path, role: str) -> str:
    wanted = route_fragments(role)
    for line in log_path.read_text(errors="replace").splitlines():
        if all(fragment in line for fragment in wanted):
            return line
    raise RuntimeError(f"server did not confirm {role} route {wanted}:\n{tail(log_path)}")


def run_checked(command: list[str], *, cwd: Path, env: dict[str, str]) -> None:
    print(f"+ {command_text(command)}", flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def run_checked_log(
    command: list[str], *, cwd: Path, env: dict[str, str], output: Path
) -> None:
    print(f"+ {command_text(command)} > {output}", flush=True)
    with output.open("x") as destination:
        subprocess.run(
            command,
            cwd=cwd,
            env=env,
            check=True,
            stdout=destination,
            stderr=subprocess.STDOUT,
            text=True,
        )


class ThermalSampler:
    def __init__(self, output: Path, interval_seconds: float = 10.0):
        self.output = output
        self.interval_seconds = interval_seconds
        self.stop_event = threading.Event()
        self.errors: list[str] = []
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _sample(self, handle) -> None:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={THERMAL_QUERY}",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            capture_output=True,
        )
        if result.returncode:
            self.errors.append(result.stderr.strip() or f"nvidia-smi={result.returncode}")
        else:
            handle.write(result.stdout)
            handle.flush()

    def _run(self) -> None:
        with self.output.open("x") as handle:
            handle.write(
                "timestamp,index,temperature_gpu_c,power_draw_w,clocks_current_sm_mhz,"
                "clocks_max_sm_mhz,sw_thermal_slowdown,hw_thermal_slowdown\n"
            )
            self._sample(handle)
            while not self.stop_event.wait(self.interval_seconds):
                self._sample(handle)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=30)
        if self.thread.is_alive():
            self.errors.append("thermal sampler thread did not stop")
        with self.output.open("a") as handle:
            self._sample(handle)


def thermal_failures(path: Path, sampler_errors: list[str]) -> tuple[dict[str, int], list[str]]:
    with path.open() as source:
        rows = list(csv.DictReader(source))
    counts = {str(index): 0 for index in range(4)}
    failures = list(sampler_errors)
    for row in rows:
        index = row.get("index", "").strip()
        if index not in counts:
            failures.append(f"unexpected GPU index {index!r}")
            continue
        counts[index] += 1
        for key in ("sw_thermal_slowdown", "hw_thermal_slowdown"):
            if row.get(key, "").strip().lower() != "not active":
                failures.append(f"GPU {index} {key}={row.get(key)!r}")
    if any(count < 2 for count in counts.values()):
        failures.append(f"insufficient thermal samples: {counts}")
    return counts, failures


def log_segment(path: Path, start: int, end: int) -> tuple[bytes, dict[str, list[str]]]:
    if end < start:
        raise ValueError("server log shrank during measured window")
    with path.open("rb") as source:
        source.seek(start)
        data = source.read(end - start)
    text = data.decode(errors="replace")
    matches = {
        name: sorted(set(pattern.findall(text)))
        for name, pattern in FAILURE_PATTERNS.items()
    }
    return data, matches


def measured_audit(
    *,
    role_dir: Path,
    log_path: Path,
    start: int,
    end: int,
    sampler: ThermalSampler,
    expected_posts: int,
    expected_endpoint: str,
) -> dict:
    data, matches = log_segment(log_path, start, end)
    text = data.decode(errors="replace")
    posts = len(
        re.findall(
            rf'POST {re.escape(expected_endpoint)} HTTP/1\.1" 200 OK', text
        )
    )
    counts, thermals = thermal_failures(role_dir / "thermal.csv", sampler.errors)
    failures: list[str] = []
    for category, found in matches.items():
        if found:
            failures.append(f"{category}: {found}")
    failures.extend(thermals)
    if posts != expected_posts:
        failures.append(
            f"successful {expected_endpoint} posts {posts}/{expected_posts}"
        )
    payload = {
        "status": "pass" if not failures else "fail",
        "server_log_start_offset": start,
        "server_log_end_offset": end,
        "server_log_segment_bytes": len(data),
        "server_log_segment_sha256": hashlib.sha256(data).hexdigest(),
        "log_pattern_matches": matches,
        "successful_posts": posts,
        "expected_posts": expected_posts,
        "expected_endpoint": expected_endpoint,
        "thermal_sample_counts": counts,
        "thermal_failures": thermals,
        "failures": failures,
    }
    (role_dir / "measured_window_audit.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    if failures:
        raise RuntimeError("measured-window audit failed: " + "; ".join(failures))
    return payload


def startup_audit(log_path: Path, end: int, output: Path) -> dict:
    data, matches = log_segment(log_path, 0, end)
    failures = []
    for category in ("errors", "retries"):
        if matches[category]:
            failures.append(f"{category}: {matches[category]}")
    payload = {
        "status": "pass" if not failures else "fail",
        "server_log_end_offset": end,
        "server_log_segment_sha256": hashlib.sha256(data).hexdigest(),
        "jit_or_compilation_matches": matches["jit_or_compilation"],
        "errors": matches["errors"],
        "retries": matches["retries"],
        "failures": failures,
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if failures:
        raise RuntimeError("startup audit failed: " + "; ".join(failures))
    return payload


def client_log_audit(log_path: Path, output: Path) -> dict:
    """Reject any client-visible error, retry, exhaustion, or timeout."""
    data = log_path.read_bytes()
    text = data.decode(errors="replace")
    matches = {
        name: sorted(set(pattern.findall(text)))
        for name, pattern in CLIENT_FAILURE_PATTERNS.items()
    }
    failures = [f"{name}: {found}" for name, found in matches.items() if found]
    payload = {
        "schema_version": 1,
        "status": "pass" if not failures else "fail",
        "client_log_path": str(log_path),
        "client_log_bytes": len(data),
        "client_log_sha256": hashlib.sha256(data).hexdigest(),
        "pattern_matches": matches,
        "failures": failures,
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if failures:
        raise RuntimeError("client-log audit failed: " + "; ".join(failures))
    return payload


def last_jsonl(path: Path) -> dict:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    if len(rows) != 1:
        raise ValueError(
            f"expected exactly one benchmark record in {path}, observed {len(rows)}"
        )
    return rows[0]


def fixed_parity_rows(payload: dict | list) -> list:
    return payload.get("records", []) if isinstance(payload, dict) else []


def validate_fixed_parity_payload(payload: dict, role: str, model: str) -> None:
    rows = fixed_parity_rows(payload)
    names = ["short", "long_32768", "long_65536"]
    valid = (
        isinstance(payload, dict)
        and payload.get("label") == role
        and payload.get("model") == model
        and len(rows) == 3
        and [row.get("name") for row in rows] == names
        and all(row.get("exact_expected") is True for row in rows)
        and all(row.get("content") == row.get("expected") for row in rows)
        and all(isinstance(row.get("prompt_tokens"), int) and row["prompt_tokens"] > 0 for row in rows)
        and rows[1]["prompt_tokens"] >= 32768
        and rows[2]["prompt_tokens"] >= 65536
        and all(re.fullmatch(r"[0-9a-f]{64}", str(row.get("response_sha256", ""))) for row in rows)
        and all(isinstance(row.get("usage"), dict) for row in rows)
    )
    if not valid:
        raise RuntimeError(f"fixed parity failed for {role}: {payload}")


def validate_serving_record(row: dict, concurrency: int) -> None:
    if int(row.get("completed", -1)) != NUM_PROMPTS:
        raise RuntimeError(f"c{concurrency} completed != {NUM_PROMPTS}")
    for key in ("failed", "failed_requests", "num_failed_requests"):
        if key in row and int(row[key]) != 0:
            raise RuntimeError(f"c{concurrency} {key}={row[key]}")
    expected = {
        "dataset_name": "random-ids",
        "random_input_len": 8192,
        "random_output_len": 1024,
        "max_concurrency": concurrency,
        "total_input_tokens": NUM_PROMPTS * 8192,
        "total_output_tokens": NUM_PROMPTS * 1024,
    }
    for key, value in expected.items():
        if row.get(key) != value:
            raise RuntimeError(f"c{concurrency} {key}={row.get(key)!r}, expected {value!r}")
    ratio = float(row.get("random_range_ratio", float("nan")))
    if not math.isfinite(ratio) or ratio != 1.0:
        raise RuntimeError(f"c{concurrency} random_range_ratio={ratio!r}, expected 1.0")
    for key in ("output_throughput", "median_ttft_ms"):
        value = float(row.get(key, float("nan")))
        if not math.isfinite(value) or value <= 0:
            raise RuntimeError(f"c{concurrency} {key} must be finite and positive: {value!r}")


def ensure_absent_output(path: Path, label: str) -> None:
    if path.exists():
        raise RuntimeError(f"stale {label} output already exists: {path}")


def claim_fresh_json_output(
    source: Path, destination: Path, started_ns: int, label: str
) -> tuple[dict, dict]:
    if not source.is_file():
        raise RuntimeError(f"{label} output was not created: {source}")
    stat = source.stat()
    if stat.st_size <= 0:
        raise RuntimeError(f"{label} output is empty: {source}")
    if stat.st_mtime_ns < started_ns:
        raise RuntimeError(
            f"stale {label} mtime {stat.st_mtime_ns} predates start {started_ns}"
        )
    payload = json.loads(source.read_text())
    digest = sha256(source)
    receipt = {
        "source_path": str(source),
        "destination_path": str(destination),
        "started_ns": started_ns,
        "source_inode": stat.st_ino,
        "source_mtime_ns": stat.st_mtime_ns,
        "source_size": stat.st_size,
        "source_sha256": digest,
    }
    shutil.move(str(source), destination)
    if sha256(destination) != digest:
        raise RuntimeError(f"{label} output changed while moving to evidence")
    return payload, receipt


def initialize_fresh_cache(path: Path) -> None:
    if path.exists():
        raise RuntimeError(f"fresh arm cache already exists: {path}")
    path.mkdir(parents=True)


def arm_cache_environment(cache_dir: Path) -> dict[str, str]:
    return {
        "FLASHINFER_WORKSPACE_BASE": str(cache_dir / "flashinfer"),
        "TORCH_EXTENSIONS_DIR": str(cache_dir / "torch_extensions"),
        "XDG_CACHE_HOME": str(cache_dir / "xdg"),
        "MINFER_FMHA_CACHE_DIR": str(cache_dir / "minfer-fmha"),
        "SGLANG_CACHE_DIR": str(cache_dir / "sglang"),
        "SGLANG_JIT_CACHE_DIR": str(cache_dir / "sglang-jit"),
        "SGLANG_DG_CACHE_DIR": str(cache_dir / "deep-gemm"),
        "DG_JIT_CACHE_DIR": str(cache_dir / "deep-gemm"),
        "TRITON_CACHE_DIR": str(cache_dir / "triton"),
        "TORCHINDUCTOR_CACHE_DIR": str(cache_dir / "torchinductor"),
        "CUDA_CACHE_PATH": str(cache_dir / "cuda-driver"),
        "FLASH_ATTENTION_CUTE_DSL_CACHE_DIR": str(cache_dir / "cute-dsl"),
    }


def validate_checkout(
    source: Path, expected_head: str, expected_tree: str, label: str
) -> None:
    head = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    tree = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD^{tree}"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=all"],
        text=True,
    ).strip()
    failures = []
    if head != expected_head:
        failures.append(f"head={head!r}, expected={expected_head!r}")
    if tree != expected_tree:
        failures.append(f"tree={tree!r}, expected={expected_tree!r}")
    if dirty:
        failures.append(f"dirty={dirty!r}")
    if failures:
        raise RuntimeError(f"{label} checkout is not frozen: " + "; ".join(failures))


def lifecycle_failures(
    *, prelaunch_clear: bool, poststop_clear: bool, measured: dict | None
) -> list[str]:
    failures = []
    if not prelaunch_clear:
        failures.append("port occupied before launch")
    if not poststop_clear:
        failures.append("port still occupied after teardown")
    if not measured or measured.get("status") != "pass":
        failures.append("measured audit missing or failed")
    return failures


def require_lifecycle(
    *, prelaunch_clear: bool, poststop_clear: bool, measured: dict | None
) -> None:
    failures = lifecycle_failures(
        prelaunch_clear=prelaunch_clear,
        poststop_clear=poststop_clear,
        measured=measured,
    )
    if failures:
        raise RuntimeError("lifecycle receipt failed: " + "; ".join(failures))


def speed_command(
    args: argparse.Namespace, output: Path, concurrency: int
) -> list[str]:
    return [
        args.python,
        "-m",
        "sglang.benchmark.serving",
        "--backend",
        "sglang",
        "--base-url",
        args.base_url,
        "--model",
        args.model,
        "--dataset-name",
        "random-ids",
        "--tokenize-prompt",
        "--num-prompts",
        str(NUM_PROMPTS),
        "--random-input-len",
        "8192",
        "--random-output-len",
        "1024",
        "--random-range-ratio",
        "1",
        "--request-rate",
        "inf",
        "--warmup-requests",
        "0",
        "--max-concurrency",
        str(concurrency),
        "--seed",
        str(SEED),
        "--flush-cache",
        "--output-file",
        str(output),
    ]


def warmup_command(args: argparse.Namespace, output: Path) -> list[str]:
    return [
        args.python,
        "-m",
        "sglang.benchmark.serving",
        "--backend",
        "sglang",
        "--base-url",
        args.base_url,
        "--model",
        args.model,
        "--dataset-name",
        "random-ids",
        "--tokenize-prompt",
        "--num-prompts",
        "1",
        "--random-input-len",
        "8192",
        "--random-output-len",
        "1024",
        "--random-range-ratio",
        "1",
        "--request-rate",
        "inf",
        "--warmup-requests",
        "0",
        "--max-concurrency",
        "1",
        "--seed",
        str(SEED),
        "--flush-cache",
        "--output-file",
        str(output),
    ]


def gpqa_command(
    args: argparse.Namespace, public_output: Path, private_output: Path
) -> list[str]:
    return [
        args.python,
        "-m",
        "sglang.test.run_eval",
        "--base-url",
        args.base_url,
        "--model",
        args.model,
        "--eval-name",
        "gpqa",
        "--gpqa-data-path",
        str(args.gpqa_dataset),
        "--per-example-output",
        str(public_output),
        "--per-example-private-responses",
        str(private_output),
        "--num-examples",
        str(GPQA_TOTAL),
        "--num-threads",
        "1",
        "--max-tokens",
        "8192",
        "--temperature",
        "0",
        "--repeat",
        "1",
    ]


def longbench_command(args: argparse.Namespace) -> list[str]:
    return [
        args.python,
        "-m",
        "sglang.test.run_eval",
        "--base-url",
        args.base_url,
        "--model",
        args.model,
        "--eval-name",
        "longbench_v2",
        "--dataset-path",
        str(args.longbench_subset),
        "--num-examples",
        str(LONGBENCH_TOTAL),
        "--num-threads",
        "1",
        "--max-tokens",
        "4096",
        "--temperature",
        "0",
        "--repeat",
        "1",
    ]


def run_fixed_parity(
    args: argparse.Namespace, repo: Path, env: dict[str, str], role_dir: Path, role: str
) -> None:
    run_checked(
        [
            args.python,
            "benchmark/minimax_m3/record_fixed_parity.py",
            "--base-url",
            args.base_url,
            "--model",
            args.model,
            "--label",
            role,
            "--output",
            str(role_dir / "fixed_parity.json"),
            "--long-tokens",
            "32768",
            "65536",
        ],
        cwd=repo,
        env=env,
    )
    payload = json.loads((role_dir / "fixed_parity.json").read_text())
    validate_fixed_parity_payload(payload, role, args.model)


def run_warmup(
    args: argparse.Namespace, repo: Path, env: dict[str, str], role_dir: Path
) -> None:
    run_checked(
        warmup_command(args, role_dir / "warmup.jsonl"),
        cwd=repo,
        env=env,
    )
    if int(last_jsonl(role_dir / "warmup.jsonl").get("completed", -1)) != 1:
        raise RuntimeError("warmup did not complete exactly one request")


def run_accuracy(
    args: argparse.Namespace,
    repo: Path,
    env: dict[str, str],
    role_dir: Path,
    private_dir: Path,
) -> int:
    gpqa_public = role_dir / "gpqa_per_example.json"
    gpqa_private = private_dir / "gpqa_responses.private.jsonl"
    gpqa_metrics = Path(f"/tmp/gpqa_{args.model.replace('/', '_')}.json")
    ensure_absent_output(gpqa_metrics, "GPQA legacy metrics")
    gpqa_started_ns = time.time_ns()
    run_checked_log(
        gpqa_command(args, gpqa_public, gpqa_private),
        cwd=repo,
        env=env,
        output=role_dir / "gpqa.log",
    )
    client_log_audit(role_dir / "gpqa.log", role_dir / "gpqa_client_audit.json")
    gpqa_private.chmod(0o600)
    gpqa = json.loads(gpqa_public.read_text())
    if len(gpqa.get("examples", [])) != GPQA_TOTAL:
        raise RuntimeError("GPQA public evidence does not contain 198 examples")
    correct = sum(bool(row["correct"]) for row in gpqa["examples"])
    gpqa_score = float(gpqa["summary"]["score"])
    if not math.isfinite(gpqa_score) or not 0 <= gpqa_score <= 1:
        raise RuntimeError(f"GPQA score outside [0, 1]: {gpqa_score!r}")
    if not math.isclose(correct / GPQA_TOTAL, gpqa_score, abs_tol=1e-12):
        raise RuntimeError("GPQA summary disagrees with public per-example evidence")
    (role_dir / "gpqa.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source": "public_per_example_summary",
                "score": gpqa_score,
                "correct": correct,
                "total": GPQA_TOTAL,
                "per_example_sha256": sha256(gpqa_public),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    legacy_gpqa, gpqa_output_receipt = claim_fresh_json_output(
        gpqa_metrics,
        role_dir / "gpqa_legacy_metrics.json",
        gpqa_started_ns,
        "GPQA legacy metrics",
    )
    legacy_gpqa_score = float(legacy_gpqa["score"])
    if not math.isfinite(legacy_gpqa_score) or not math.isclose(
        legacy_gpqa_score, gpqa_score, abs_tol=1e-12
    ):
        raise RuntimeError(
            f"GPQA legacy score disagrees with per-example summary: "
            f"{legacy_gpqa_score!r} vs {gpqa_score!r}"
        )
    (role_dir / "gpqa_legacy_output_receipt.json").write_text(
        json.dumps(gpqa_output_receipt, indent=2, sort_keys=True) + "\n"
    )

    longbench_metrics = Path(
        f"/tmp/longbench_v2_{args.model.replace('/', '_')}.json"
    )
    ensure_absent_output(longbench_metrics, "LongBench metrics")
    longbench_started_ns = time.time_ns()
    run_checked_log(
        longbench_command(args),
        cwd=repo,
        env=env,
        output=role_dir / "longbench_v2.log",
    )
    client_log_audit(
        role_dir / "longbench_v2.log", role_dir / "longbench_v2_client_audit.json"
    )
    longbench, output_receipt = claim_fresh_json_output(
        longbench_metrics,
        role_dir / "longbench_v2.json",
        longbench_started_ns,
        "LongBench metrics",
    )
    (role_dir / "longbench_v2_output_receipt.json").write_text(
        json.dumps(output_receipt, indent=2, sort_keys=True) + "\n"
    )
    shutil.copy2(
        Path(str(args.longbench_subset) + ".manifest.json"),
        role_dir / "longbench_v2_subset_manifest.json",
    )
    longbench_score = float(longbench["score"])
    if not math.isfinite(longbench_score) or not 0 <= longbench_score <= 1:
        raise RuntimeError("LongBench score is outside [0, 1]")
    log_text = (role_dir / "longbench_v2.log").read_text(errors="replace")
    matches = re.findall(r"^Score: ([0-9]+(?:\.[0-9]+)?)$", log_text, re.MULTILINE)
    if len(matches) != 1 or not math.isclose(
        float(matches[0]), longbench_score, abs_tol=0.00051
    ):
        raise RuntimeError(
            f"LongBench stdout score does not match JSON: {matches} vs {longbench_score}"
        )
    return GPQA_TOTAL + LONGBENCH_TOTAL


def run_speed(
    args: argparse.Namespace, repo: Path, env: dict[str, str], role_dir: Path
) -> int:
    for concurrency in CONCURRENCIES:
        output = role_dir / f"serving_c{concurrency}.jsonl"
        run_checked_log(
            speed_command(args, output, concurrency),
            cwd=repo,
            env=env,
            output=role_dir / f"serving_c{concurrency}.log",
        )
        client_log_audit(
            role_dir / f"serving_c{concurrency}.log",
            role_dir / f"serving_c{concurrency}_client_audit.json",
        )
        row = last_jsonl(output)
        validate_serving_record(row, concurrency)
    return NUM_PROMPTS * len(CONCURRENCIES)


def precompile_source(
    args: argparse.Namespace,
    repo: Path,
    env: dict[str, str],
    role_dir: Path,
    cache_dir: Path,
) -> None:
    run_checked(
        [
            args.python,
            "benchmark/minimax_m3/precompile_fmha_sm100.py",
            "--cache-dir",
            str(cache_dir / "minfer-fmha"),
            "--output",
            str(role_dir / "source_precompile_receipt.json"),
        ],
        cwd=repo,
        env=env,
    )


def run_role(
    *,
    args: argparse.Namespace,
    repo: Path,
    base_environment: dict[str, str],
    repetition: int,
    role: str,
) -> None:
    rep_dir = args.output_root / f"rep{repetition:02d}"
    role_dir = rep_dir / role
    role_dir.mkdir()
    private_dir = args.output_root / "private" / f"rep{repetition:02d}" / role
    private_dir.mkdir(parents=True)
    private_dir.chmod(0o700)
    server_log = rep_dir / f"{role}_server.log"
    cache_dir = args.cache_root / args.mode / f"rep{repetition:02d}" / role
    initialize_fresh_cache(cache_dir)

    environment = role_environment(base_environment, role)
    environment["PYTHONUNBUFFERED"] = "1"
    environment.update(arm_cache_environment(cache_dir))
    if role == "external":
        precompile_source(args, repo, environment, role_dir, cache_dir)

    prelaunch_clear = not server_healthy(args.base_url)
    if not prelaunch_clear:
        raise RuntimeError(f"port is not clear before {role} launch")
    launch = server_command(args)
    print(f"+ {args.mode}/rep{repetition:02d}/{role}: {command_text(launch)}", flush=True)
    process: subprocess.Popen | None = None
    route_line = ""
    measured: dict | None = None
    try:
        with server_log.open("x") as log:
            process = subprocess.Popen(
                launch,
                cwd=repo,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            wait_for_server(process, args.base_url, server_log, args.server_timeout)
            log.flush()
            os.fsync(log.fileno())
            route_line = validate_route(server_log, role)
            run_warmup(args, repo, environment, role_dir)
            if fixed_parity_required(args.mode):
                run_fixed_parity(args, repo, environment, role_dir, role)
            log.flush()
            os.fsync(log.fileno())
            start = server_log.stat().st_size
            (role_dir / "measured_server_log_start_offset.txt").write_text(f"{start}\n")
            startup_audit(server_log, start, role_dir / "startup_audit.json")
            sampler = ThermalSampler(role_dir / "thermal.csv")
            sampler.start()
            try:
                if args.mode == "accuracy":
                    expected_posts = run_accuracy(
                        args, repo, environment, role_dir, private_dir
                    )
                else:
                    expected_posts = run_speed(args, repo, environment, role_dir)
                if process.poll() is not None:
                    raise RuntimeError(f"server exited during {role}:\n{tail(server_log)}")
            finally:
                sampler.stop()
            log.flush()
            os.fsync(log.fileno())
            end = server_log.stat().st_size
            (role_dir / "measured_server_log_end_offset.txt").write_text(f"{end}\n")
            measured = measured_audit(
                role_dir=role_dir,
                log_path=server_log,
                start=start,
                end=end,
                sampler=sampler,
                expected_posts=expected_posts,
                expected_endpoint=(
                    "/v1/chat/completions" if args.mode == "accuracy" else "/generate"
                ),
            )
    finally:
        if process is not None:
            stop_server(process, args.base_url)
    poststop_clear = not server_healthy(args.base_url)
    lifecycle_problems = lifecycle_failures(
        prelaunch_clear=prelaunch_clear,
        poststop_clear=poststop_clear,
        measured=measured,
    )
    lifecycle = {
        "schema_version": 1,
        "status": "pass" if not lifecycle_problems else "fail",
        "mode": args.mode,
        "repetition": repetition,
        "role": role,
        "route_fragments": list(route_fragments(role)),
        "matched_route_line": route_line,
        "prelaunch_port_clear": prelaunch_clear,
        "poststop_port_clear": poststop_clear,
        "fresh_server_process": True,
        "fresh_cache_policy": {
            "cache_root": str(cache_dir),
            "initially_absent": True,
            "source_serial_precompile": role == "external",
            "shared_across_arms": False,
            "redirected_environment": {
                name: environment[name]
                for name in (
                    "FLASHINFER_WORKSPACE_BASE",
                    "TORCH_EXTENSIONS_DIR",
                    "XDG_CACHE_HOME",
                    "MINFER_FMHA_CACHE_DIR",
                    "SGLANG_CACHE_DIR",
                    "SGLANG_JIT_CACHE_DIR",
                    "SGLANG_DG_CACHE_DIR",
                    "DG_JIT_CACHE_DIR",
                    "TRITON_CACHE_DIR",
                    "TORCHINDUCTOR_CACHE_DIR",
                    "CUDA_CACHE_PATH",
                    "FLASH_ATTENTION_CUTE_DSL_CACHE_DIR",
                )
            },
        },
        "fixed_parity_sha256": (
            sha256(role_dir / "fixed_parity.json")
            if fixed_parity_required(args.mode)
            else None
        ),
        "measured_window_audit_sha256": sha256(
            role_dir / "measured_window_audit.json"
        ),
        "failures": lifecycle_problems,
    }
    (role_dir / "lifecycle_route_cache_receipt.json").write_text(
        json.dumps(lifecycle, indent=2, sort_keys=True) + "\n"
    )
    require_lifecycle(
        prelaunch_clear=prelaunch_clear,
        poststop_clear=poststop_clear,
        measured=measured,
    )


def gpqa_pair_audit(source_path: Path, export_path: Path) -> dict:
    external = json.loads(source_path.read_text())
    flashinfer = json.loads(export_path.read_text())
    left = {row["question_id"]: row for row in external["examples"]}
    right = {row["question_id"]: row for row in flashinfer["examples"]}
    if list(left) != list(right) or len(left) != GPQA_TOTAL:
        raise ValueError("GPQA per-example IDs/order differ")
    rows = []
    for question_id in left:
        a, b = left[question_id], right[question_id]
        rows.append(
            {
                "question_id": question_id,
                "source_correct": bool(a["correct"]),
                "export_correct": bool(b["correct"]),
                "source_parsed_answer": a["parsed_answer"],
                "export_parsed_answer": b["parsed_answer"],
                "source_response_sha256": a["response_sha256"],
                "export_response_sha256": b["response_sha256"],
            }
        )
    source_correct = sum(row["source_correct"] for row in rows)
    export_correct = sum(row["export_correct"] for row in rows)
    return {
        "summary": {
            "total": GPQA_TOTAL,
            "source_correct": source_correct,
            "export_correct": export_correct,
            "delta_questions": export_correct - source_correct,
            "gain": sum(
                (not row["source_correct"]) and row["export_correct"] for row in rows
            ),
            "loss": sum(
                row["source_correct"] and (not row["export_correct"]) for row in rows
            ),
            "parsed_answer_changed": sum(
                row["source_parsed_answer"] != row["export_parsed_answer"]
                for row in rows
            ),
            "response_hash_changed": sum(
                row["source_response_sha256"] != row["export_response_sha256"]
                for row in rows
            ),
        },
        "examples": rows,
    }


def cross_rep_stability(root: Path, role: str) -> dict:
    per_rep = []
    for repetition in range(1, 4):
        payload = json.loads(
            (root / f"rep{repetition:02d}" / role / "gpqa_per_example.json").read_text()
        )
        per_rep.append({row["question_id"]: row for row in payload["examples"]})
    ids = list(per_rep[0])
    if any(list(rep) != ids for rep in per_rep[1:]):
        raise ValueError(f"{role} GPQA IDs/order differ across repetitions")
    pairwise = []
    for left_index, right_index in ((0, 1), (0, 2), (1, 2)):
        pairwise.append(
            {
                "left_rep": left_index + 1,
                "right_rep": right_index + 1,
                "correctness_changed": sum(
                    bool(per_rep[left_index][qid]["correct"])
                    != bool(per_rep[right_index][qid]["correct"])
                    for qid in ids
                ),
                "parsed_answer_changed": sum(
                    per_rep[left_index][qid]["parsed_answer"]
                    != per_rep[right_index][qid]["parsed_answer"]
                    for qid in ids
                ),
                "response_hash_changed": sum(
                    per_rep[left_index][qid]["response_sha256"]
                    != per_rep[right_index][qid]["response_sha256"]
                    for qid in ids
                ),
            }
        )
    return {"role": role, "total": GPQA_TOTAL, "pairwise": pairwise}


def summarize_accuracy(root: Path) -> dict:
    repetitions = []
    gpqa_deltas = []
    longbench_deltas = []
    failures = []
    for repetition in range(1, 4):
        rep_dir = root / f"rep{repetition:02d}"
        audit = gpqa_pair_audit(
            rep_dir / "external" / "gpqa_per_example.json",
            rep_dir / "flashinfer" / "gpqa_per_example.json",
        )
        (rep_dir / "gpqa_pair_audit.json").write_text(
            json.dumps(audit, indent=2, sort_keys=True) + "\n"
        )
        gpqa_delta = audit["summary"]["delta_questions"]
        source_lb = float(
            json.loads((rep_dir / "external" / "longbench_v2.json").read_text())["score"]
        )
        export_lb = float(
            json.loads((rep_dir / "flashinfer" / "longbench_v2.json").read_text())["score"]
        )
        if not math.isfinite(source_lb) or not 0 <= source_lb <= 1:
            raise RuntimeError(f"rep{repetition:02d} external LongBench invalid: {source_lb!r}")
        if not math.isfinite(export_lb) or not 0 <= export_lb <= 1:
            raise RuntimeError(f"rep{repetition:02d} FlashInfer LongBench invalid: {export_lb!r}")
        lb_delta = export_lb - source_lb
        gpqa_pass = gpqa_delta >= -GPQA_MARGIN_QUESTIONS
        lb_pass = within_noninferiority_margin(lb_delta, LONGBENCH_MARGIN)
        if not gpqa_pass:
            failures.append(f"rep{repetition:02d} GPQA delta {gpqa_delta} < -1")
        if not lb_pass:
            failures.append(f"rep{repetition:02d} LongBench delta {lb_delta} < -0.02")
        gpqa_deltas.append(gpqa_delta)
        longbench_deltas.append(lb_delta)
        repetitions.append(
            {
                "repetition": repetition,
                "order": expected_order("accuracy", repetition),
                "gpqa": {**audit["summary"], "pass": gpqa_pass},
                "longbench_v2": {
                    "source_score": source_lb,
                    "export_score": export_lb,
                    "delta": lb_delta,
                    "pass": lb_pass,
                },
            }
        )
    gpqa_mean = statistics.mean(gpqa_deltas)
    gpqa_median = statistics.median(gpqa_deltas)
    lb_mean = statistics.mean(longbench_deltas)
    lb_median = statistics.median(longbench_deltas)
    if gpqa_mean < -1 or gpqa_median < -1:
        failures.append("aggregate GPQA mean/median violates one-question margin")
    if not within_noninferiority_margin(
        lb_mean, LONGBENCH_MARGIN
    ) or not within_noninferiority_margin(lb_median, LONGBENCH_MARGIN):
        failures.append("aggregate LongBench mean/median violates 0.02 margin")
    payload = {
        "schema_version": 1,
        "status": "pass" if not failures else "fail",
        "repetitions": repetitions,
        "aggregate": {
            "gpqa_delta_questions_mean": gpqa_mean,
            "gpqa_delta_questions_median": gpqa_median,
            "longbench_delta_mean": lb_mean,
            "longbench_delta_median": lb_median,
        },
        "cross_rep_stability": {
            role: cross_rep_stability(root, role) for role in ("external", "flashinfer")
        },
        "failures": failures,
    }
    if failures:
        raise RuntimeError("accuracy summary failed: " + "; ".join(failures))
    return payload


def summarize_speed(root: Path, mode: str, min_median_gain: float) -> dict:
    if not math.isfinite(min_median_gain) or min_median_gain < 0:
        raise RuntimeError(
            f"minimum median throughput gain must be finite and nonnegative: {min_median_gain!r}"
        )
    baseline, candidate = roles_for_mode(mode)
    repetitions = []
    gains = {str(value): [] for value in CONCURRENCIES}
    for repetition in range(1, 4):
        rep = {"repetition": repetition, "order": expected_order(mode, repetition), "metrics": {}}
        for concurrency in CONCURRENCIES:
            left = last_jsonl(
                root / f"rep{repetition:02d}" / baseline / f"serving_c{concurrency}.jsonl"
            )
            right = last_jsonl(
                root / f"rep{repetition:02d}" / candidate / f"serving_c{concurrency}.jsonl"
            )
            validate_serving_record(left, concurrency)
            validate_serving_record(right, concurrency)
            left_throughput = float(left["output_throughput"])
            right_throughput = float(right["output_throughput"])
            left_ttft = float(left["median_ttft_ms"])
            right_ttft = float(right["median_ttft_ms"])
            throughput_gain = (
                right_throughput / left_throughput - 1
            )
            ttft_reduction = 1 - right_ttft / left_ttft
            if not math.isfinite(throughput_gain) or not math.isfinite(ttft_reduction):
                raise RuntimeError(
                    f"c{concurrency} derived speed metric is non-finite: "
                    f"throughput_gain={throughput_gain!r}, ttft_reduction={ttft_reduction!r}"
                )
            gains[str(concurrency)].append(throughput_gain)
            rep["metrics"][str(concurrency)] = {
                "baseline_role": baseline,
                "baseline_output_throughput": left["output_throughput"],
                "export_output_throughput": right["output_throughput"],
                "output_throughput_gain": throughput_gain,
                "baseline_median_ttft_ms": left["median_ttft_ms"],
                "export_median_ttft_ms": right["median_ttft_ms"],
                "ttft_latency_reduction": ttft_reduction,
            }
        repetitions.append(rep)
    aggregate = {}
    failures = []
    for concurrency in CONCURRENCIES:
        values = gains[str(concurrency)]
        aggregate[str(concurrency)] = {
            "output_throughput_gain_mean": statistics.mean(values),
            "output_throughput_gain_median": statistics.median(values),
        }
        if statistics.median(values) < min_median_gain:
            failures.append(
                f"c{concurrency} median throughput gain {statistics.median(values)} "
                f"< {min_median_gain}"
            )
    payload = {
        "schema_version": 1,
        "status": "pass" if not failures else "fail",
        "mode": mode,
        "baseline_role": baseline,
        "candidate_role": candidate,
        "repetitions": repetitions,
        "aggregate": aggregate,
        "minimum_median_output_throughput_gain": min_median_gain,
        "failures": failures,
    }
    if failures:
        raise RuntimeError("speed summary failed: " + "; ".join(failures))
    return payload


def run_test_only(output: Path | None) -> None:
    results: list[dict[str, str]] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            raise AssertionError(message)

    def passed(test_id: str, detail: str = "") -> None:
        results.append({"id": test_id, "status": "pass", "detail": detail})

    def expect_failure(
        test_id: str,
        function,
        exception_type: type[BaseException] = RuntimeError,
        contains: str | None = None,
    ) -> str:
        try:
            function()
        except exception_type as error:
            message = str(error)
            if contains is not None and contains not in message:
                raise AssertionError(
                    f"{test_id}: expected {contains!r} in {message!r}"
                ) from error
            passed(test_id, f"observed {type(error).__name__}: {message}")
            return message
        raise AssertionError(f"{test_id}: synthetic failure was accepted")

    orders = {
        mode: [expected_order(mode, rep) for rep in range(1, 4)]
        for mode in ("accuracy", "external-speed", "triton-speed")
    }
    require(
        loopback_environment({})["OPENAI_API_KEY"] == "EMPTY",
        "loopback evaluator dummy key was not injected",
    )
    require(
        loopback_environment({"OPENAI_API_KEY": "explicit"})["OPENAI_API_KEY"]
        == "explicit",
        "explicit evaluator API key was overwritten",
    )
    passed("loopback_api_key_default_and_preservation_contract")
    require(
        orders["accuracy"] == [
            ["external", "flashinfer"], ["flashinfer", "external"], ["external", "flashinfer"]
        ],
        "accuracy order contract mismatch",
    )
    require(
        orders["triton-speed"] == [
            ["triton", "flashinfer"], ["flashinfer", "triton"], ["triton", "flashinfer"]
        ],
        "triton speed order contract mismatch",
    )
    passed("alternating_order_contract")
    require(fixed_parity_required("accuracy"), "accuracy fixed parity was disabled")
    require(
        not fixed_parity_required("external-speed")
        and not fixed_parity_required("triton-speed"),
        "a speed-only mode would send accuracy requests",
    )
    passed("speed_modes_have_no_accuracy_requests")
    expect_failure(
        "wrong_order_rejected",
        lambda: validate_order("accuracy", 1, ["flashinfer", "external"]),
        contains="wrong accuracy",
    )

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "server.log"
        for role in ("external", "flashinfer", "triton"):
            line = "prefix " + ", ".join(route_fragments(role)) + " suffix"
            path.write_text(line + "\n")
            require(validate_route(path, role) == line, f"{role} route mismatch")
            path.write_text("wrong route\n")
            expect_failure(
                f"wrong_{role}_route_rejected",
                lambda role=role: validate_route(path, role),
                contains="did not confirm",
            )
        passed("all_routes_positive_contract")

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        server_log = root / "server.log"
        role_dir = root / "role"
        role_dir.mkdir()
        thermal_path = role_dir / "thermal.csv"

        def write_thermal(active: bool = False) -> None:
            rows = [
                "timestamp,index,temperature_gpu_c,power_draw_w,clocks_current_sm_mhz,"
                "clocks_max_sm_mhz,sw_thermal_slowdown,hw_thermal_slowdown"
            ]
            for sample in range(2):
                for index in range(4):
                    value = "Active" if active and sample == 0 and index == 0 else "Not Active"
                    rows.append(f"t{sample},{index},40,100,1000,2000,{value},Not Active")
            thermal_path.write_text("\n".join(rows) + "\n")

        sampler = argparse.Namespace(errors=[])
        startup = "startup JIT compile\n"
        post = 'POST /v1/chat/completions HTTP/1.1" 200 OK\n'
        server_log.write_text(startup + post)
        write_thermal()
        require(measured_audit(
            role_dir=role_dir, log_path=server_log, start=len(startup.encode()),
            end=server_log.stat().st_size, sampler=sampler, expected_posts=1,
            expected_endpoint="/v1/chat/completions",
        )["status"] == "pass", "measured offset contract failed")
        passed("measured_offset_excludes_startup_jit")
        expect_failure(
            "measured_log_shrink_rejected",
            lambda: log_segment(server_log, 10, 9),
            ValueError,
            "shrank",
        )
        expect_failure(
            "measured_post_count_rejected",
            lambda: measured_audit(
                role_dir=role_dir, log_path=server_log, start=len(startup.encode()),
                end=server_log.stat().st_size, sampler=sampler, expected_posts=2,
                expected_endpoint="/v1/chat/completions",
            ),
            contains="/v1/chat/completions posts 1/2",
        )
        for test_id, bad_line, wanted in (
            ("measured_jit_rejected", "NVRTC compilation\n", "jit_or_compilation"),
            ("measured_error_rejected", "RuntimeError: boom\n", "errors"),
            ("measured_generic_error_rejected", "[worker ERROR] request aborted\n", "errors"),
            ("measured_retry_rejected", "retrying request\n", "retries"),
        ):
            server_log.write_text(post + bad_line)
            write_thermal()
            expect_failure(
                test_id,
                lambda: measured_audit(
                    role_dir=role_dir, log_path=server_log, start=0,
                    end=server_log.stat().st_size, sampler=sampler, expected_posts=1,
                    expected_endpoint="/v1/chat/completions",
                ),
                contains=wanted,
            )
        server_log.write_text(post)
        write_thermal(active=True)
        expect_failure(
            "thermal_throttle_rejected",
            lambda: measured_audit(
                role_dir=role_dir, log_path=server_log, start=0,
                end=server_log.stat().st_size, sampler=sampler, expected_posts=1,
                expected_endpoint="/v1/chat/completions",
            ),
            contains="thermal_slowdown",
        )

        server_log.write_text(
            'POST /generate HTTP/1.1" 200 OK\n'
            'POST /generate HTTP/1.1" 200 OK\n'
        )
        write_thermal()
        require(
            measured_audit(
                role_dir=role_dir,
                log_path=server_log,
                start=0,
                end=server_log.stat().st_size,
                sampler=sampler,
                expected_posts=2,
                expected_endpoint="/generate",
            )["successful_posts"]
            == 2,
            "native serving endpoint count mismatch",
        )
        passed("native_generate_endpoint_count_contract")
        expect_failure(
            "wrong_endpoint_count_rejected",
            lambda: measured_audit(
                role_dir=role_dir,
                log_path=server_log,
                start=0,
                end=server_log.stat().st_size,
                sampler=sampler,
                expected_posts=2,
                expected_endpoint="/v1/chat/completions",
            ),
            contains="posts 0/2",
        )

        client_log = root / "client.log"
        client_receipt = root / "client-audit.json"
        client_log.write_text("Score: 0.5\nCompleted: 198\n")
        require(
            client_log_audit(client_log, client_receipt)["status"] == "pass",
            "positive client log was rejected",
        )
        passed("client_log_positive_contract")
        for test_id, bad_line, wanted in (
            ("client_log_retry_rejected", "retrying request 7\n", "retries"),
            ("client_log_error_rejected", "RuntimeError: request failed\n", "errors"),
            ("client_log_exhausted_rejected", "attempts exhausted\n", "errors"),
            ("client_log_timeout_rejected", "request timed out\n", "timeouts"),
        ):
            client_log.write_text(bad_line)
            expect_failure(
                test_id,
                lambda: client_log_audit(client_log, client_receipt),
                contains=wanted,
            )

    def valid_serving_record(
        concurrency: int, throughput: float = 100.0, ttft: float = 10.0
    ) -> dict:
        return {
            "completed": NUM_PROMPTS,
            "failed": 0,
            "dataset_name": "random-ids",
            "random_input_len": 8192,
            "random_output_len": 1024,
            "random_range_ratio": 1.0,
            "max_concurrency": concurrency,
            "total_input_tokens": NUM_PROMPTS * 8192,
            "total_output_tokens": NUM_PROMPTS * 1024,
            "output_throughput": throughput,
            "median_ttft_ms": ttft,
        }

    good_record = valid_serving_record(8)
    validate_serving_record(good_record, 8)
    passed("serving_request_count_positive")
    expect_failure(
        "serving_completed_count_rejected",
        lambda: validate_serving_record(
            {**valid_serving_record(8), "completed": NUM_PROMPTS - 1}, 8
        ),
        contains="completed",
    )
    for key in ("failed", "failed_requests", "num_failed_requests"):
        expect_failure(
            f"serving_{key}_rejected",
            lambda key=key: validate_serving_record(
                {**valid_serving_record(8), key: 1}, 8
            ),
            contains=key,
        )
    for key, bad_value in (
        ("dataset_name", "sharegpt"),
        ("random_input_len", 8191),
        ("random_output_len", 1023),
        ("random_range_ratio", 0.9),
        ("max_concurrency", 7),
        ("total_input_tokens", NUM_PROMPTS * 8192 - 1),
        ("total_output_tokens", NUM_PROMPTS * 1024 - 1),
    ):
        expect_failure(
            f"serving_{key}_tamper_rejected",
            lambda key=key, bad_value=bad_value: validate_serving_record(
                {**valid_serving_record(8), key: bad_value}, 8
            ),
            contains=key,
        )
    for key in ("output_throughput", "median_ttft_ms"):
        for suffix, bad_value in (
            ("nan", float("nan")),
            ("inf", float("inf")),
            ("zero", 0.0),
            ("negative", -1.0),
        ):
            expect_failure(
                f"serving_{key}_{suffix}_rejected",
                lambda key=key, bad_value=bad_value: validate_serving_record(
                    {**valid_serving_record(8), key: bad_value}, 8
                ),
                contains="finite and positive",
            )
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "serving.jsonl"
        path.write_text(json.dumps(valid_serving_record(1)) + "\n")
        require(last_jsonl(path)["completed"] == NUM_PROMPTS, "single JSONL rejected")
        passed("single_serving_record_positive")
        path.write_text(
            json.dumps(valid_serving_record(1))
            + "\n"
            + json.dumps(valid_serving_record(1))
            + "\n"
        )
        expect_failure(
            "duplicate_serving_records_rejected",
            lambda: last_jsonl(path),
            ValueError,
            "exactly one benchmark record",
        )

    args = argparse.Namespace(
        python="python", base_url="http://127.0.0.1:30000", model="/model",
        gpqa_dataset=Path("/gpqa.csv"), longbench_subset=Path("/longbench.json"),
    )
    command = speed_command(args, Path("/out.jsonl"), 32)
    option = lambda name: command[command.index(name) + 1]
    require(option("--backend") == "sglang", "speed backend is not native sglang")
    require(option("--num-prompts") == "256", "speed prompt count drifted")
    require(option("--random-input-len") == "8192", "speed input length drifted")
    require(option("--random-output-len") == "1024", "speed output length drifted")
    require(option("--warmup-requests") == "0", "speed implicit warmup was not disabled")
    require(option("--max-concurrency") == "32", "speed concurrency drifted")
    require(
        option("--seed") == str(SEED) and "--flush-cache" in command,
        "speed seed or cache-flush contract drifted",
    )
    passed("speed_fixed_workload_contract")
    warmup = warmup_command(args, Path("/warmup.jsonl"))
    warmup_option = lambda name: warmup[warmup.index(name) + 1]
    require(warmup_option("--num-prompts") == "1", "warmup prompt count drifted")
    require(
        warmup_option("--warmup-requests") == "0",
        "warmup command would send an implicit extra request",
    )
    passed("exactly_one_unmeasured_warmup_contract")
    gpqa = gpqa_command(args, Path("/public.json"), Path("/private.jsonl"))
    longbench = longbench_command(args)
    gpqa_option = lambda name: gpqa[gpqa.index(name) + 1]
    lb_option = lambda name: longbench[longbench.index(name) + 1]
    require(
        gpqa_option("--num-examples") == "198"
        and gpqa_option("--num-threads") == "1",
        "GPQA fixed workload drifted",
    )
    require(
        "--per-example-output" in gpqa
        and "--per-example-private-responses" in gpqa,
        "GPQA evidence flags are incomplete",
    )
    require(
        lb_option("--num-examples") == "100"
        and lb_option("--num-threads") == "1",
        "LongBench fixed workload drifted",
    )
    passed("accuracy_fixed_workload_and_private_evidence_contract")

    with tempfile.TemporaryDirectory() as directory:
        cache = Path(directory) / "cache"
        initialize_fresh_cache(cache)
        expect_failure(
            "stale_cache_rejected", lambda: initialize_fresh_cache(cache),
            contains="already exists",
        )
        first = arm_cache_environment(cache)
        second = arm_cache_environment(Path(directory) / "other-cache")
        require(set(first) == set(second), "cache variable sets differ across arms")
        require(
            all(Path(value).is_relative_to(cache) for value in first.values()),
            "a cache variable escapes its arm-local root",
        )
        require(
            set(first.values()).isdisjoint(second.values()),
            "arm-local cache paths overlap",
        )
        passed("all_jit_caches_are_arm_local_and_disjoint")

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "producer.json"
        destination = root / "evidence.json"
        ensure_absent_output(source, "synthetic")
        passed("fresh_output_absence_positive")
        source.write_text('{"score": 1}\n')
        expect_failure(
            "stale_output_preexisting_rejected",
            lambda: ensure_absent_output(source, "synthetic"),
            contains="already exists",
        )
        source.unlink()
        expect_failure(
            "missing_fresh_output_rejected",
            lambda: claim_fresh_json_output(source, destination, time.time_ns(), "synthetic"),
            contains="was not created",
        )
        started_ns = time.time_ns()
        source.write_bytes(b"")
        expect_failure(
            "empty_fresh_output_rejected",
            lambda: claim_fresh_json_output(source, destination, started_ns, "synthetic"),
            contains="empty",
        )
        source.write_text('{"score": 1}\n')
        os.utime(source, ns=(1, 1))
        expect_failure(
            "stale_output_mtime_rejected",
            lambda: claim_fresh_json_output(source, destination, started_ns, "synthetic"),
            contains="predates start",
        )
        source.unlink()
        started_ns = time.time_ns()
        source.write_text('{"score": 1}\n')
        payload, receipt = claim_fresh_json_output(
            source, destination, started_ns, "synthetic"
        )
        require(payload == {"score": 1}, "fresh output payload drifted")
        require(not source.exists() and destination.is_file(), "fresh output was not moved")
        require(
            receipt["source_sha256"] == sha256(destination)
            and receipt["source_mtime_ns"] >= started_ns
            and receipt["source_inode"] > 0,
            "fresh output receipt is incomplete",
        )
        passed("fresh_output_claim_receipt_positive")

    with tempfile.TemporaryDirectory() as directory:
        repo = Path(directory) / "repo"
        subprocess.run(["git", "init", "-q", str(repo)], check=True)
        (repo / "tracked").write_text("frozen\n")
        subprocess.run(["git", "-C", str(repo), "add", "tracked"], check=True)
        subprocess.run(
            ["git", "-C", str(repo), "-c", "user.name=Formal V2",
             "-c", "user.email=formal-v2@example.invalid", "commit", "-qm", "frozen"],
            check=True,
        )
        head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
        tree = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD^{tree}"], text=True).strip()
        validate_checkout(repo, head, tree, "synthetic")
        passed("frozen_checkout_positive")
        expect_failure(
            "stale_head_rejected", lambda: validate_checkout(repo, "0" * 40, tree, "synthetic"),
            contains="head=",
        )
        expect_failure(
            "stale_tree_rejected", lambda: validate_checkout(repo, head, "0" * 40, "synthetic"),
            contains="tree=",
        )
        (repo / "untracked").write_text("dirty\n")
        expect_failure(
            "dirty_checkout_rejected", lambda: validate_checkout(repo, head, tree, "synthetic"),
            contains="dirty=",
        )

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "input"
        path.write_text("frozen\n")
        expected = sha256(path)
        require_sha256(path, expected, "positive")
        passed("immutable_hash_positive")
        path.write_text("tampered\n")
        for test_id, label in (
            ("stale_dataset_hash_rejected", "dataset"),
            ("stale_container_hash_rejected", "container"),
            ("stale_bundle_hash_rejected", "bundle"),
        ):
            expect_failure(
                test_id, lambda label=label: require_sha256(path, expected, label),
                contains="SHA-256 mismatch",
            )

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory) / "checkpoint"
        root.mkdir()
        (root / "config.json").write_text("{}\n")
        (root / "weights.bin").write_bytes(b"weights")
        rows = []
        aggregate = hashlib.sha256()
        for path in sorted(root.iterdir()):
            relative = path.name
            size = path.stat().st_size
            digest = sha256(path)
            rows.append({"path": relative, "size": size, "sha256": digest})
            aggregate.update(
                relative.encode()
                + b"\0"
                + str(size).encode()
                + b"\0"
                + digest.encode()
                + b"\n"
            )
        manifest = Path(directory) / "manifest.json"
        payload = {
            "schema_version": 1,
            "status": "pass",
            "file_count": len(rows),
            "total_bytes": sum(row["size"] for row in rows),
            "aggregate_sha256": aggregate.hexdigest(),
            "files": rows,
        }
        manifest.write_text(json.dumps(payload, sort_keys=True) + "\n")
        receipt = verify_file_manifest(
            root, manifest, sha256(manifest), aggregate.hexdigest(), "synthetic model"
        )
        require(receipt["file_count"] == 2, "model manifest positive count drifted")
        passed("full_model_manifest_positive")
        expect_failure(
            "stale_model_manifest_hash_rejected",
            lambda: verify_file_manifest(
                root, manifest, "0" * 64, aggregate.hexdigest(), "synthetic model"
            ),
            contains="manifest SHA-256 mismatch",
        )
        expect_failure(
            "stale_model_aggregate_rejected",
            lambda: verify_file_manifest(
                root, manifest, sha256(manifest), "0" * 64, "synthetic model"
            ),
            contains="manifest aggregate mismatch",
        )
        (root / "weights.bin").write_bytes(b"tampered")
        expect_failure(
            "model_file_tamper_rejected",
            lambda: verify_file_manifest(
                root, manifest, sha256(manifest), aggregate.hexdigest(), "synthetic model"
            ),
            contains="file mismatch",
        )
        (root / "weights.bin").write_bytes(b"weights")
        (root / "unexpected").write_text("x")
        expect_failure(
            "model_file_set_drift_rejected",
            lambda: verify_file_manifest(
                root, manifest, sha256(manifest), aggregate.hexdigest(), "synthetic model"
            ),
            contains="file-set mismatch",
        )

    parity_records = [
        {
            "name": name,
            "expected": expected,
            "content": expected,
            "exact_expected": True,
            "prompt_tokens": prompt_tokens,
            "reasoning_content": "",
            "response_sha256": digest * 64,
            "usage": {},
        }
        for name, expected, prompt_tokens, digest in (
            ("short", "MSA-SHORT-4B19", 8, "a"),
            ("long_32768", "MSA-32768-C7F29A", 32768, "b"),
            ("long_65536", "MSA-65536-C7F29A", 65536, "c"),
        )
    ]
    valid_parity = {"label": "external", "model": "/model", "records": parity_records}
    validate_fixed_parity_payload(valid_parity, "external", "/model")
    passed("fixed_parity_real_producer_schema_positive")
    expect_failure(
        "fixed_parity_list_schema_rejected",
        lambda: validate_fixed_parity_payload(parity_records, "external", "/model"),
        contains="fixed parity failed",
    )
    expect_failure(
        "fixed_parity_results_schema_rejected",
        lambda: validate_fixed_parity_payload(
            {"label": "external", "model": "/model", "results": parity_records},
            "external",
            "/model",
        ),
        contains="fixed parity failed",
    )
    expect_failure(
        "fixed_parity_mismatch_rejected",
        lambda: validate_fixed_parity_payload(
            {
                "label": "flashinfer",
                "model": "/model",
                "records": [
                    parity_records[0],
                    {**parity_records[1], "exact_expected": False},
                    parity_records[2],
                ],
            },
            "flashinfer",
            "/model",
        ),
        contains="fixed parity failed",
    )

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for repetition in range(1, 4):
            rep = root / f"rep{repetition:02d}"
            for role in ("external", "flashinfer"):
                role_dir = rep / role
                role_dir.mkdir(parents=True)
                correct_limit = 100 if role == "external" else 98
                rows = [{"question_id": f"q{index:03d}", "correct": index < correct_limit,
                         "parsed_answer": "A", "response_sha256": f"{index:064x}"}
                        for index in range(GPQA_TOTAL)]
                (role_dir / "gpqa_per_example.json").write_text(json.dumps({"examples": rows}) + "\n")
                (role_dir / "longbench_v2.json").write_text(
                    json.dumps({"score": 0.64 if role == "external" else 0.61}) + "\n"
                )
        message = expect_failure(
            "accuracy_per_rep_and_aggregate_gates_rejected",
            lambda: summarize_accuracy(root),
            contains="rep01 GPQA",
        )
        require(
            "aggregate GPQA" in message and "aggregate LongBench" in message,
            "aggregate accuracy failures were not both reported",
        )
        for repetition in range(1, 4):
            role_dir = root / f"rep{repetition:02d}" / "flashinfer"
            payload = json.loads((role_dir / "gpqa_per_example.json").read_text())
            payload["examples"][98]["correct"] = True
            (role_dir / "gpqa_per_example.json").write_text(json.dumps(payload) + "\n")
            (role_dir / "longbench_v2.json").write_text(json.dumps({"score": 0.62}) + "\n")
        require(summarize_accuracy(root)["status"] == "pass", "accuracy boundary failed")
        passed("accuracy_boundary_positive")
        passed("longbench_exact_float_boundary_positive")
        require(
            not within_noninferiority_margin(0.61 - 0.64, LONGBENCH_MARGIN),
            "LongBench regression beyond the margin was accepted",
        )
        passed("longbench_beyond_margin_rejected")

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for repetition in range(1, 4):
            for role, throughput in (("external", 100.0), ("flashinfer", 99.0)):
                role_dir = root / f"rep{repetition:02d}" / role
                role_dir.mkdir(parents=True)
                for concurrency in CONCURRENCIES:
                    (role_dir / f"serving_c{concurrency}.jsonl").write_text(
                        json.dumps(valid_serving_record(concurrency, throughput, 10.0))
                        + "\n"
                    )
        expect_failure(
            "speed_median_gate_rejected", lambda: summarize_speed(root, "external-speed", 0.0),
            contains="median throughput gain",
        )
        for path in root.glob("rep*/flashinfer/serving_c*.jsonl"):
            concurrency = int(path.stem.removeprefix("serving_c"))
            path.write_text(
                json.dumps(valid_serving_record(concurrency, 101.0, 9.0)) + "\n"
            )
        require(
            summarize_speed(root, "external-speed", 0.0)["status"] == "pass",
            "speed positive boundary failed",
        )
        passed("speed_gate_positive")
        for test_id, threshold in (
            ("speed_nan_threshold_rejected", float("nan")),
            ("speed_inf_threshold_rejected", float("inf")),
            ("speed_negative_threshold_rejected", -0.001),
        ):
            expect_failure(
                test_id,
                lambda threshold=threshold: summarize_speed(
                    root, "external-speed", threshold
                ),
                contains="finite and nonnegative",
            )

    require(
        not lifecycle_failures(
            prelaunch_clear=True, poststop_clear=True, measured={"status": "pass"}
        ),
        "positive lifecycle contract failed",
    )
    passed("server_lifecycle_positive")
    for test_id, kwargs, wanted in (
        ("prelaunch_port_occupied_rejected",
         {"prelaunch_clear": False, "poststop_clear": True, "measured": {"status": "pass"}},
         "before launch"),
        ("teardown_port_cleanup_rejected",
         {"prelaunch_clear": True, "poststop_clear": False, "measured": {"status": "pass"}},
         "after teardown"),
        ("missing_measured_audit_rejected",
         {"prelaunch_clear": True, "poststop_clear": True, "measured": None},
         "measured audit"),
    ):
        expect_failure(
            test_id,
            lambda kwargs=kwargs: require_lifecycle(**kwargs),
            contains=wanted,
        )

    require(
        GPQA_MARGIN_QUESTIONS == 1 and math.isclose(LONGBENCH_MARGIN, 0.02),
        "accuracy margins drifted",
    )
    passed("accuracy_margin_constants")
    receipt = {
        "schema_version": 1,
        "status": "pass",
        "test_count": len(results),
        "test_results": results,
        "orders": orders,
        "roles": {role: list(route_fragments(role)) for role in ("external", "flashinfer", "triton")},
        "gpqa_margin_questions": GPQA_MARGIN_QUESTIONS,
        "longbench_margin": LONGBENCH_MARGIN,
        "private_evidence_mode": "0600",
        "fresh_cache_per_arm": True,
    }
    text = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    if output:
        output.write_text(text)
    print(text, end="")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-only", action="store_true")
    parser.add_argument("--test-receipt", type=Path)
    parser.add_argument(
        "--mode", choices=("accuracy", "external-speed", "triton-speed")
    )
    parser.add_argument("--model")
    parser.add_argument("--gpqa-dataset", type=Path)
    parser.add_argument("--longbench-subset", type=Path)
    parser.add_argument("--flashinfer-source-dir", type=Path)
    parser.add_argument("--expected-flashinfer-head")
    parser.add_argument("--expected-flashinfer-tree")
    parser.add_argument("--sglang-source-dir", type=Path)
    parser.add_argument("--expected-sglang-head")
    parser.add_argument("--expected-sglang-tree")
    parser.add_argument("--expected-model-config-sha256")
    parser.add_argument("--model-manifest", type=Path)
    parser.add_argument("--expected-model-manifest-sha256")
    parser.add_argument("--expected-model-aggregate-sha256")
    parser.add_argument("--expected-gpqa-sha256")
    parser.add_argument("--expected-longbench-sha256")
    parser.add_argument("--expected-longbench-manifest-sha256")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--server-timeout", type=int, default=7200)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--min-median-throughput-gain", type=float, default=0.0)
    args = parser.parse_args()
    if args.test_only:
        run_test_only(args.test_receipt)
        return
    required = (
        "mode",
        "model",
        "flashinfer_source_dir",
        "expected_flashinfer_head",
        "expected_flashinfer_tree",
        "sglang_source_dir",
        "expected_sglang_head",
        "expected_sglang_tree",
        "gpqa_dataset",
        "longbench_subset",
        "expected_model_config_sha256",
        "model_manifest",
        "expected_model_manifest_sha256",
        "expected_model_aggregate_sha256",
        "expected_gpqa_sha256",
        "expected_longbench_sha256",
        "expected_longbench_manifest_sha256",
        "output_root",
        "cache_root",
    )
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        parser.error("missing required arguments: " + ", ".join(missing))
    if args.mode == "accuracy":
        if args.min_median_throughput_gain != 0.0:
            parser.error("accuracy mode requires --min-median-throughput-gain=0.0")
    elif (
        not math.isfinite(args.min_median_throughput_gain)
        or args.min_median_throughput_gain < 0
    ):
        parser.error("minimum median throughput gain must be finite and nonnegative")
    args.base_url = f"http://{args.host}:{args.port}"
    args.output_root = args.output_root.resolve()
    args.cache_root = args.cache_root.resolve()
    repo = args.sglang_source_dir.resolve()
    flashinfer = args.flashinfer_source_dir.resolve()
    if args.output_root.exists():
        raise SystemExit(f"refusing to reuse output root: {args.output_root}")
    validate_checkout(
        repo, args.expected_sglang_head, args.expected_sglang_tree, "SGLang"
    )
    validate_checkout(
        flashinfer,
        args.expected_flashinfer_head,
        args.expected_flashinfer_tree,
        "FlashInfer",
    )
    require_sha256(
        Path(args.model) / "config.json",
        args.expected_model_config_sha256,
        "model config",
    )
    require_sha256(args.gpqa_dataset, args.expected_gpqa_sha256, "GPQA")
    require_sha256(
        args.longbench_subset, args.expected_longbench_sha256, "LongBench"
    )
    require_sha256(
        Path(str(args.longbench_subset) + ".manifest.json"),
        args.expected_longbench_manifest_sha256,
        "LongBench manifest",
    )
    args.output_root.mkdir(parents=True)
    checkpoint_receipt = verify_file_manifest(
        Path(args.model),
        args.model_manifest,
        args.expected_model_manifest_sha256,
        args.expected_model_aggregate_sha256,
        "model checkpoint",
    )
    (args.output_root / "model_checkpoint_verification.json").write_text(
        json.dumps(checkpoint_receipt, indent=2, sort_keys=True) + "\n"
    )
    (args.output_root / "private").mkdir(mode=0o700)
    environment = loopback_environment(dict(os.environ))
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(repo / "python"), str(flashinfer)]
        + ([environment["PYTHONPATH"]] if environment.get("PYTHONPATH") else [])
    )
    environment.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    manifest = {
        "schema_version": 1,
        "mode": args.mode,
        "model": args.model,
        "seed": SEED,
        "repetitions": 3,
        "orders": [expected_order(args.mode, rep) for rep in range(1, 4)],
        "roles": list(roles_for_mode(args.mode)),
        "server_command": server_command(args),
        "fresh_server_and_cache_per_arm": True,
        "runner_sha256": sha256(Path(__file__)),
        "flashinfer_head": args.expected_flashinfer_head,
        "flashinfer_tree": args.expected_flashinfer_tree,
        "sglang_head": args.expected_sglang_head,
        "sglang_tree": args.expected_sglang_tree,
        "immutable_input_sha256": {
            "model_config": args.expected_model_config_sha256,
            "model_manifest": args.expected_model_manifest_sha256,
            "model_aggregate": args.expected_model_aggregate_sha256,
            "gpqa": args.expected_gpqa_sha256,
            "longbench": args.expected_longbench_sha256,
            "longbench_manifest": args.expected_longbench_manifest_sha256,
        },
    }
    if args.mode == "accuracy":
        manifest["accuracy"] = {
            "fixed_parity_visible_tokens": ["short", 32768, 65536],
            "gpqa_examples": GPQA_TOTAL,
            "gpqa_threads": 1,
            "gpqa_margin_questions": GPQA_MARGIN_QUESTIONS,
            "longbench_examples": LONGBENCH_TOTAL,
            "longbench_threads": 1,
            "longbench_margin": LONGBENCH_MARGIN,
            "private_responses_mode": "0600",
        }
    else:
        manifest["serving"] = {
            "concurrencies": list(CONCURRENCIES),
            "num_prompts": NUM_PROMPTS,
            "input_tokens": 8192,
            "output_tokens": 1024,
            "random_range_ratio": 1,
            "backend": "sglang",
            "endpoint": "/generate",
            "warmup_requests_per_invocation": 0,
            "min_median_throughput_gain": args.min_median_throughput_gain,
        }
    (args.output_root / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    for repetition in range(1, 4):
        rep_dir = args.output_root / f"rep{repetition:02d}"
        rep_dir.mkdir()
        order = expected_order(args.mode, repetition)
        validate_order(args.mode, repetition, order)
        (rep_dir / "order.json").write_text(
            json.dumps({"order": order}, indent=2, sort_keys=True) + "\n"
        )
        for role in order:
            run_role(
                args=args,
                repo=repo,
                base_environment=environment,
                repetition=repetition,
                role=role,
            )
    if args.mode == "accuracy":
        summary = summarize_accuracy(args.output_root)
    else:
        summary = summarize_speed(
            args.output_root, args.mode, args.min_median_throughput_gain
        )
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
