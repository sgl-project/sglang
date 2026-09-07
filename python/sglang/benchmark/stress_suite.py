"""Run continuous, dynamically changing online serving workloads.

The suite builds one timestamped workload and delegates request generation,
transport, token accounting, and latency metrics to ``sglang.benchmark.serving``.
"""

from __future__ import annotations

import argparse
import json
import math
import shlex
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

DEFAULT_SOAK_DURATION_SEC = 20 * 60 * 60
DEFAULT_PROMETHEUS_METRICS = (
    "sglang:cache_hit_rate",
    "sglang:cached_tokens_total",
    "sglang:prompt_tokens_total",
    "sglang:generation_tokens_total",
    "sglang:num_running_reqs",
    "sglang:num_queue_reqs",
    "sglang:num_used_tokens",
)
SENSITIVE_NAMES = ("authorization", "api-key", "api_key", "cookie", "token")
TOOL_REQUEST_BODY = {
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "lookup_weather",
                "description": "Return synthetic weather for a city.",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ],
    "tool_choice": "auto",
}


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    input_len: int
    output_len: int
    duration: float
    max_concurrency: int
    qps_source: str = "baseline"
    qps_scale: float = 1.0
    fixed_qps: float | None = None
    scale_duration: bool = True
    source: str | None = None
    extra_request_body: dict | None = None


SCENARIOS = {
    scenario.name: scenario
    for scenario in (
        Scenario(
            "smoke",
            "Low-rate correctness check before pressure.",
            input_len=32,
            output_len=8,
            duration=3,
            max_concurrency=8,
            fixed_qps=1,
            scale_duration=False,
        ),
        Scenario(
            "steady",
            "Baseline traffic.",
            input_len=512,
            output_len=64,
            duration=20,
            max_concurrency=256,
        ),
        Scenario(
            "ramp_low",
            "First ramp step at 25% of peak QPS.",
            input_len=512,
            output_len=64,
            duration=10,
            max_concurrency=256,
            qps_source="peak",
            qps_scale=0.25,
        ),
        Scenario(
            "ramp_mid",
            "Second ramp step at 50% of peak QPS.",
            input_len=512,
            output_len=64,
            duration=10,
            max_concurrency=256,
            qps_source="peak",
            qps_scale=0.5,
        ),
        Scenario(
            "burst",
            "Peak-rate burst.",
            input_len=512,
            output_len=64,
            duration=10,
            max_concurrency=512,
            qps_source="peak",
        ),
        Scenario(
            "recovery",
            "Low-rate recovery after pressure.",
            input_len=256,
            output_len=32,
            duration=10,
            max_concurrency=128,
            qps_scale=0.5,
        ),
        Scenario(
            "zigzag_low",
            "Low half of an alternating load pair.",
            input_len=256,
            output_len=64,
            duration=5,
            max_concurrency=256,
            qps_scale=0.5,
        ),
        Scenario(
            "zigzag_high",
            "High half of an alternating load pair.",
            input_len=256,
            output_len=64,
            duration=5,
            max_concurrency=512,
            qps_source="peak",
        ),
        Scenario(
            "microburst",
            "Brief overload at twice the peak QPS.",
            input_len=256,
            output_len=32,
            duration=3,
            max_concurrency=1024,
            qps_source="peak",
            qps_scale=2,
        ),
        Scenario(
            "connection_churn",
            "Peak traffic through serving's per-request client sessions.",
            input_len=64,
            output_len=16,
            duration=5,
            max_concurrency=512,
            qps_source="peak",
        ),
        Scenario(
            "tool_rich",
            "Chat requests carrying a synthetic function schema.",
            input_len=256,
            output_len=128,
            duration=5,
            max_concurrency=128,
            extra_request_body=TOOL_REQUEST_BODY,
        ),
        Scenario(
            "tiny_input",
            "Minimal input and single-token output boundary.",
            input_len=4,
            output_len=1,
            duration=5,
            max_concurrency=128,
        ),
        Scenario(
            "long_prefill",
            "Long-context prefill pressure.",
            input_len=8192,
            output_len=32,
            duration=10,
            max_concurrency=64,
            qps_scale=0.25,
        ),
        Scenario(
            "long_decode",
            "Long decode pressure.",
            input_len=128,
            output_len=512,
            duration=10,
            max_concurrency=128,
            qps_scale=0.5,
        ),
        Scenario(
            "long_context_32k",
            "32K input context boundary.",
            input_len=32768,
            output_len=16,
            duration=5,
            max_concurrency=16,
            qps_scale=0.1,
        ),
        Scenario(
            "long_context_80k",
            "80K input context boundary.",
            input_len=81920,
            output_len=16,
            duration=5,
            max_concurrency=8,
            qps_scale=0.05,
        ),
        Scenario(
            "shared_prefix",
            "Generated shared-prefix traffic for cache behavior.",
            input_len=2048,
            output_len=64,
            duration=10,
            max_concurrency=256,
            source="generated-shared-prefix",
        ),
        Scenario(
            "high_concurrency",
            "Peak traffic with a large concurrency window.",
            input_len=256,
            output_len=64,
            duration=10,
            max_concurrency=1024,
            qps_source="peak",
        ),
    )
}

PROFILES = {
    "quick": ("smoke", "steady", "burst", "recovery"),
    "standard": (
        "smoke",
        "steady",
        "ramp_low",
        "ramp_mid",
        "burst",
        "recovery",
        "tiny_input",
        "long_prefill",
        "long_decode",
        "shared_prefix",
    ),
    "edge": (
        "smoke",
        "zigzag_low",
        "zigzag_high",
        "microburst",
        "recovery",
        "connection_churn",
        "tool_rich",
        "tiny_input",
        "long_prefill",
        "long_decode",
        "long_context_32k",
        "long_context_80k",
        "high_concurrency",
    ),
    "all": tuple(SCENARIOS),
    "soak": tuple(SCENARIOS),
}

RESULT_KEYS = (
    "duration",
    "completed",
    "request_throughput",
    "input_throughput",
    "output_throughput",
    "mean_ttft_ms",
    "p99_ttft_ms",
    "mean_e2e_latency_ms",
    "p99_e2e_latency_ms",
    "concurrency",
    "max_concurrent_requests",
    "cache_report",
)


def resolve_scenarios(profile: str, names: Sequence[str]) -> list[Scenario]:
    selected = names or PROFILES[profile]
    return [SCENARIOS[name] for name in dict.fromkeys(selected)]


def target_qps(args: argparse.Namespace, scenario: Scenario) -> float:
    if scenario.fixed_qps is not None:
        return scenario.fixed_qps
    base = args.peak_qps if scenario.qps_source == "peak" else args.baseline_qps
    return base * scenario.qps_scale


def scenario_duration(args: argparse.Namespace, scenario: Scenario) -> float:
    return scenario.duration * (args.duration_scale if scenario.scale_duration else 1)


def _target_duration(args: argparse.Namespace) -> float | None:
    if args.total_duration_sec is not None:
        return args.total_duration_sec
    if args.profile == "soak" and not args.scenario:
        return DEFAULT_SOAK_DURATION_SEC
    return None


def build_workload(args: argparse.Namespace) -> dict:
    scenarios = resolve_scenarios(args.profile, args.scenario)
    target_duration = _target_duration(args)
    phases = []
    elapsed = 0.0
    cycle = 1

    while True:
        repeatable_scenarios = [
            scenario for scenario in scenarios if scenario.name != "smoke"
        ]
        cycle_scenarios = (
            scenarios if cycle == 1 else (repeatable_scenarios or scenarios)
        )
        if not cycle_scenarios:
            break
        for scenario in cycle_scenarios:
            duration = scenario_duration(args, scenario)
            if target_duration is not None:
                remaining = target_duration - elapsed
                if remaining <= 0:
                    break
                duration = min(duration, remaining)
            name = scenario.name if cycle == 1 else f"{scenario.name}-c{cycle:03d}"
            phases.append(
                {
                    "name": name,
                    "scenario": scenario.name,
                    "description": scenario.description,
                    "start_time": elapsed,
                    "duration": duration,
                    "request_rate": target_qps(args, scenario),
                    "input_len": scenario.input_len,
                    "output_len": scenario.output_len,
                    "max_concurrency": scenario.max_concurrency,
                    "source": scenario.source or args.dataset_source,
                    "arrival_pattern": args.arrival_pattern,
                    "range_ratio": args.random_range_ratio,
                    **(
                        {"extra_request_body": scenario.extra_request_body}
                        if scenario.extra_request_body
                        else {}
                    ),
                }
            )
            elapsed += duration
            if target_duration is not None and elapsed >= target_duration:
                break
        if target_duration is None or elapsed >= target_duration:
            break
        cycle += 1

    return {
        "version": 1,
        "seed": args.seed,
        "source": args.dataset_source,
        "arrival_pattern": args.arrival_pattern,
        "prompt_pool_size": args.prompt_pool_size,
        "duration": elapsed,
        "phases": phases,
    }


def _get_json(url: str, headers: dict[str, str]) -> dict | None:
    try:
        request = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(request, timeout=5) as response:
            if response.status == 200:
                value = json.load(response)
                return value if isinstance(value, dict) else None
    except (OSError, ValueError, urllib.error.URLError):
        pass
    return None


def detect_capabilities(args: argparse.Namespace) -> dict:
    """Best-effort discovery used only to skip unsupported scenarios."""
    headers = _request_headers(args.header)
    server_info = _get_json(args.base_url.rstrip("/") + "/server_info", headers) or {}
    model_info = _get_json(args.base_url.rstrip("/") + "/model_info", headers) or {}
    context_length = args.context_length
    if context_length is None:
        for key in ("context_length", "max_model_len", "max_seq_len"):
            value = server_info.get(key)
            if isinstance(value, int) and value > 0:
                context_length = value
                break
    parser_advertised = (
        "tool_call_parser" in model_info or "tool_call_parser" in server_info
    )
    tool_parser = model_info.get(
        "tool_call_parser", server_info.get("tool_call_parser")
    )
    return {
        "context_length": context_length,
        "tool_call_parser": tool_parser,
        "tool_support": bool(tool_parser) if parser_advertised else None,
        "source": (
            "explicit"
            if args.context_length is not None
            else "server"
            if server_info or model_info
            else "unavailable"
        ),
    }


def apply_capabilities(workload: dict, capabilities: dict) -> list[dict]:
    """Remove unsupported phases and return explicit SKIP records."""
    runnable = []
    skipped = []
    context_length = capabilities.get("context_length")
    tool_support = capabilities.get("tool_support")
    for phase in workload["phases"]:
        reason = None
        if context_length and phase["input_len"] + phase["output_len"] > context_length:
            reason = (
                f"requires {phase['input_len'] + phase['output_len']} tokens, "
                f"server context length is {context_length}"
            )
        elif phase.get("scenario") == "tool_rich" and tool_support is False:
            reason = "server does not advertise a tool-call parser"
        if reason:
            skipped.append({**phase, "verdict": "SKIP", "reason": reason})
            continue
        runnable.append(phase)
    workload["phases"] = runnable
    return skipped


def estimated_requests(workload: dict) -> int:
    return sum(
        max(1, math.ceil(phase["duration"] * phase["request_rate"]))
        for phase in workload["phases"]
    )


def build_command(
    args: argparse.Namespace,
    workload_file: Path,
    result_file: Path,
    workload: dict,
) -> list[str]:
    max_concurrency = args.max_concurrency or max(
        phase["max_concurrency"] for phase in workload["phases"]
    )
    command = [
        sys.executable,
        "-m",
        "sglang.benchmark.serving",
        "--backend",
        args.backend,
        "--base-url",
        args.base_url.rstrip("/"),
        "--dataset-name",
        "dynamic",
        "--dynamic-workload-path",
        str(workload_file),
        "--use-trace-timestamps",
        "--num-prompts",
        str(estimated_requests(workload)),
        "--request-rate",
        "1",
        "--max-concurrency",
        str(max_concurrency),
        "--seed",
        str(args.seed),
        "--warmup-requests",
        str(args.warmup_requests),
        "--ready-check-timeout-sec",
        str(args.ready_check_timeout_sec),
        "--output-file",
        str(result_file),
        "--tag",
        args.profile,
        "--disable-tqdm",
        "--max-pending-requests",
        str(args.max_pending_requests),
        "--phase-output-file",
        str(result_file.with_name("phases.jsonl")),
    ]
    if args.dataset_path:
        command.extend(("--dataset-path", args.dataset_path))
    if args.model:
        command.extend(("--model", args.model))
    if args.tokenizer:
        command.extend(("--tokenizer", args.tokenizer))
    if args.extra_request_body:
        command.extend(("--extra-request-body", args.extra_request_body))
    if args.header:
        command.extend(("--header", *args.header))
    if args.flush_cache:
        command.append("--flush-cache")
    if args.cache_report:
        command.append("--cache-report")
    if args.prometheus:
        for metric in args.prometheus_metric:
            command.extend(("--prometheus-metric", metric))
    return command


def redact_value(value):
    if isinstance(value, dict):
        return {
            key: (
                "<redacted>"
                if any(name in key.lower() for name in SENSITIVE_NAMES)
                else redact_value(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact_value(item) for item in value]
    return value


def redact_url(value: str) -> str:
    scheme, separator, remainder = value.partition("://")
    if separator and "@" in remainder:
        return f"{scheme}{separator}<redacted>@{remainder.split('@', 1)[1]}"
    return value


def redact_command(command: Sequence[str]) -> list[str]:
    redacted = list(command)
    index = 0
    while index < len(redacted):
        option = redacted[index]
        if option == "--base-url" and index + 1 < len(redacted):
            redacted[index + 1] = redact_url(redacted[index + 1])
            index += 2
            continue
        if option == "--header" and index + 1 < len(redacted):
            index += 1
            while index < len(redacted) and not redacted[index].startswith("--"):
                key, separator, _ = redacted[index].partition("=")
                redacted[index] = f"{key}=<redacted>" if separator else "<redacted>"
                index += 1
            continue
        if option == "--extra-request-body" and index + 1 < len(redacted):
            try:
                redacted[index + 1] = json.dumps(
                    redact_value(json.loads(redacted[index + 1])), separators=(",", ":")
                )
            except json.JSONDecodeError:
                redacted[index + 1] = "<redacted>"
            index += 2
            continue
        index += 1
    return redacted


def load_result(path: Path) -> dict:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"benchmark produced no result: {path}")
    return rows[-1]


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open() as file:
        return sum(bool(line.strip()) for line in file)


def write_json_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def append_jsonl(path: Path, payload: dict) -> None:
    with path.open("a") as file:
        file.write(json.dumps(payload, sort_keys=True) + "\n")
        file.flush()


def health_summary(checks: Sequence[dict]) -> dict:
    return {
        "total": len(checks),
        "failed": sum(not check["ok"] for check in checks),
        "latest": checks[-1] if checks else None,
    }


def check_health(base_url: str, headers: dict[str, str] | None = None) -> dict:
    started = time.perf_counter()
    status = None
    error = ""
    try:
        request = urllib.request.Request(
            base_url.rstrip("/") + "/health", headers=headers or {}
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            status = response.status
    except urllib.error.HTTPError as exc:
        status = exc.code
        error = str(exc)
    except (OSError, urllib.error.URLError) as exc:
        error = str(exc)
    error = error.replace(base_url, redact_url(base_url))
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "ok": status == 200,
        "latency_ms": round((time.perf_counter() - started) * 1000, 3),
        "error": error,
    }


def monitor_health(
    stop: threading.Event,
    *,
    base_url: str,
    interval: float,
    progress_file: Path,
    health_file: Path,
    phase_file: Path,
    workload: dict,
    checks: list[dict],
    headers: dict[str, str],
    started_at: str,
) -> None:
    started = time.monotonic()
    while True:
        check = check_health(base_url, headers)
        checks.append(check)
        append_jsonl(health_file, check)
        elapsed = time.monotonic() - started
        phase = next(
            (
                item
                for item in reversed(workload["phases"])
                if item["start_time"] <= elapsed < item["start_time"] + item["duration"]
            ),
            None,
        )
        write_json_atomic(
            progress_file,
            {
                "status": "running",
                "started_at": started_at,
                "elapsed_sec": round(elapsed, 3),
                "planned_duration_sec": workload["duration"],
                "current_phase": phase["name"] if phase else None,
                "completed_phases": count_jsonl(phase_file),
                "health": health_summary(checks),
            },
        )
        if stop.wait(interval):
            return


def _request_headers(values: Sequence[str]) -> dict[str, str]:
    headers = {}
    for value in values:
        key, separator, header_value = value.partition("=")
        if separator and key and header_value:
            headers[key] = header_value
    return headers


def judge_phase(args: argparse.Namespace, phase: dict) -> tuple[str, str]:
    reasons = []
    completed = int(phase.get("completed", 0))
    planned = int(phase.get("planned", 0))
    if completed != planned:
        reasons.append(f"completed {completed}/{planned} requests")
    if incomplete := int(phase.get("incomplete_streams", 0)):
        reasons.append(f"{incomplete} streams ended without a completion marker")
    for option, key, label in (
        (args.max_ttft_p99_ms, "p99_ttft_ms", "TTFT"),
        (args.max_e2e_p99_ms, "p99_e2e_latency_ms", "E2E"),
    ):
        value = phase.get(key)
        if option is not None and value is not None and value > option:
            reasons.append(f"{label} p99 {value:.2f} ms exceeds {option:.2f} ms")
    return ("FAIL", "; ".join(reasons)) if reasons else ("PASS", "")


def judge_health(checks: Sequence[dict], allowed_failures: int) -> str:
    failures = sum(not check["ok"] for check in checks)
    if failures > allowed_failures:
        return f"health failed {failures} times (allowed {allowed_failures})"
    return ""


def write_summary(summary: dict, output_dir: Path) -> None:
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    rows = [
        "# SGLang continuous stress suite",
        "",
        f"Overall: **{summary['verdict']}**",
        "",
        f"Planned duration: {summary['planned_duration_sec']:.1f} seconds",
        "",
        "| Phase | Target QPS | Dispatch QPS | Lag p99 (ms) | Cache hit | Verdict | Completed | TTFT p99 (ms) | E2E p99 (ms) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for run in summary["runs"]:
        rows.append(
            f"| {run['name']} | {run.get('request_rate', '-')} | "
            f"{run.get('actual_dispatch_qps', '-')} | "
            f"{run.get('p99_schedule_lag_ms', '-')} | "
            f"{run.get('cache_hit_rate_pct', '-')} | {run['verdict']} | "
            f"{run.get('completed', '-')} / "
            f"{run.get('planned', '-')} | {run.get('p99_ttft_ms', '-')} | "
            f"{run.get('p99_e2e_latency_ms', '-')} |"
        )
    (output_dir / "summary.md").write_text("\n".join(rows) + "\n")


def run_suite(args: argparse.Namespace) -> int:
    workload = build_workload(args)
    phase_order = {
        phase["name"]: index for index, phase in enumerate(workload["phases"])
    }
    capabilities = {}
    skipped = []
    if not args.disable_capability_check and not args.dry_run:
        capabilities = detect_capabilities(args)
        skipped = apply_capabilities(workload, capabilities)
    if args.dry_run:
        workload_file = Path("workload.json")
        result_file = Path("result.jsonl")
        print(json.dumps(workload, indent=2))
        print(
            shlex.join(
                redact_command(
                    build_command(args, workload_file, result_file, workload)
                )
            )
        )
        return 0

    output_dir = args.output_dir or Path(
        "stress_results", time.strftime("%Y%m%d-%H%M%S")
    )
    output_dir = output_dir.resolve()
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"output directory exists: {output_dir}")
        if output_dir in (Path("/"), Path.home()):
            raise ValueError(f"refusing to overwrite broad path: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    workload_file = output_dir / "workload.json"
    result_file = output_dir / "result.jsonl"
    phase_file = output_dir / "phases.jsonl"
    progress_file = output_dir / "progress.json"
    health_file = output_dir / "health.jsonl"
    workload_file.write_text(json.dumps(workload, indent=2) + "\n")
    if not workload["phases"]:
        summary = {
            "verdict": "SKIP",
            "reason": "all selected phases are unsupported by the target server",
            "profile": args.profile,
            "base_url": redact_url(args.base_url),
            "baseline_qps": args.baseline_qps,
            "peak_qps": args.peak_qps,
            "planned_duration_sec": workload["duration"],
            "estimated_requests": 0,
            "health": health_summary([]),
            "capabilities": capabilities,
            "command": None,
            "metrics": None,
            "runs": skipped,
        }
        write_json_atomic(progress_file, {"status": "skipped", "completed_phases": 0})
        write_summary(summary, output_dir)
        print(f"SKIP: {summary['reason']}")
        return 0
    command = build_command(args, workload_file, result_file, workload)
    safe_command = redact_command(command)
    print(shlex.join(safe_command), flush=True)
    health_checks = []
    monitor_stop = threading.Event()
    started_at = datetime.now(timezone.utc).isoformat()
    monitor = None
    if args.health_check_interval_sec > 0:
        monitor = threading.Thread(
            target=monitor_health,
            kwargs={
                "stop": monitor_stop,
                "base_url": args.base_url,
                "interval": args.health_check_interval_sec,
                "progress_file": progress_file,
                "health_file": health_file,
                "phase_file": phase_file,
                "workload": workload,
                "checks": health_checks,
                "headers": _request_headers(args.header),
                "started_at": started_at,
            },
            daemon=True,
        )
        monitor.start()
    try:
        with (output_dir / "benchmark.log").open("w") as log:
            completed = subprocess.run(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
    finally:
        monitor_stop.set()
        if monitor is not None:
            monitor.join()
    final_health = check_health(args.base_url, _request_headers(args.header))
    health_checks.append(final_health)
    append_jsonl(health_file, final_health)

    try:
        result = load_result(result_file)
    except (FileNotFoundError, ValueError, json.JSONDecodeError):
        result = None

    phase_config = {phase["name"]: phase for phase in workload["phases"]}
    incremental_metrics = load_jsonl(phase_file)
    incremental_by_name = {item["name"]: item for item in incremental_metrics}
    phase_metrics = (result or {}).get("phase_metrics") or incremental_metrics
    phase_metrics = [
        {
            **metrics,
            **(
                {
                    "prometheus_snapshot": incremental_by_name[metrics["name"]][
                        "prometheus_snapshot"
                    ]
                }
                if incremental_by_name.get(metrics["name"], {}).get(
                    "prometheus_snapshot"
                )
                else {}
            ),
        }
        for metrics in phase_metrics
    ]
    runs = list(skipped)
    for metrics in phase_metrics:
        verdict, reason = judge_phase(args, metrics)
        runs.append(
            {
                **phase_config.get(metrics["name"], {}),
                **metrics,
                "verdict": verdict,
                "reason": reason,
            }
        )
    runs.sort(key=lambda run: phase_order.get(run["name"], len(phase_order)))

    reasons = []
    if completed.returncode != 0:
        reasons.append(f"serving benchmark exited {completed.returncode}")
    if result is None:
        reasons.append("serving benchmark produced no result")
    elif len(runs) - len(skipped) != len(workload["phases"]):
        reasons.append(
            f"reported {len(runs) - len(skipped)}/{len(workload['phases'])} phases"
        )
    if any(run["verdict"] == "FAIL" for run in runs):
        reasons.append("one or more phases failed")
    if health_reason := judge_health(health_checks, args.max_health_failures):
        reasons.append(health_reason)

    summary = {
        "verdict": "FAIL" if reasons else "PASS",
        "reason": "; ".join(reasons),
        "profile": args.profile,
        "base_url": redact_url(args.base_url),
        "baseline_qps": args.baseline_qps,
        "peak_qps": args.peak_qps,
        "planned_duration_sec": workload["duration"],
        "estimated_requests": estimated_requests(workload),
        "health": health_summary(health_checks),
        "capabilities": capabilities,
        "command": shlex.join(safe_command),
        "metrics": (
            {key: result[key] for key in RESULT_KEYS if key in result}
            if result
            else None
        ),
        "runs": runs,
    }
    write_summary(summary, output_dir)
    write_json_atomic(
        progress_file,
        {
            "status": "completed" if summary["verdict"] == "PASS" else "failed",
            "verdict": summary["verdict"],
            "started_at": started_at,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "return_code": completed.returncode,
            "planned_duration_sec": workload["duration"],
            "completed_phases": count_jsonl(phase_file),
            "health": health_summary(health_checks),
        },
    )
    print(f"{summary['verdict']}: {summary['reason']}" if reasons else "PASS")
    print(output_dir / "summary.md")
    return 1 if reasons else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url")
    parser.add_argument(
        "--backend",
        choices=("sglang", "sglang-oai", "sglang-oai-chat"),
        default="sglang-oai-chat",
    )
    parser.add_argument("--model")
    parser.add_argument("--tokenizer")
    parser.add_argument("--profile", choices=PROFILES, default="standard")
    parser.add_argument("--scenario", action="append", choices=SCENARIOS, default=[])
    parser.add_argument("--list-scenarios", action="store_true")
    parser.add_argument("--baseline-qps", type=float, default=1)
    parser.add_argument("--peak-qps", type=float, default=4)
    parser.add_argument("--duration-scale", type=float, default=1)
    parser.add_argument("--total-duration-sec", type=float)
    parser.add_argument(
        "--dataset-source",
        choices=("random-ids", "sharegpt", "generated-shared-prefix"),
        default="random-ids",
    )
    parser.add_argument("--dataset-path")
    parser.add_argument(
        "--arrival-pattern", choices=("constant", "poisson"), default="constant"
    )
    parser.add_argument("--random-range-ratio", type=float, default=1)
    parser.add_argument("--prompt-pool-size", type=int, default=8)
    parser.add_argument("--max-concurrency", type=int)
    parser.add_argument("--max-pending-requests", type=int, default=4096)
    parser.add_argument("--warmup-requests", type=int, default=1)
    parser.add_argument("--ready-check-timeout-sec", type=int, default=60)
    parser.add_argument("--health-check-interval-sec", type=float, default=60)
    parser.add_argument("--max-health-failures", type=int, default=0)
    parser.add_argument("--context-length", type=int)
    parser.add_argument("--disable-capability-check", action="store_true")
    parser.add_argument("--prometheus", action="store_true")
    parser.add_argument(
        "--prometheus-metric",
        action="append",
        default=list(DEFAULT_PROMETHEUS_METRICS),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--extra-request-body")
    parser.add_argument("--header", action="append", default=[])
    parser.add_argument("--flush-cache", action="store_true")
    parser.add_argument("--cache-report", action="store_true")
    parser.add_argument("--max-ttft-p99-ms", type=float)
    parser.add_argument("--max-e2e-p99-ms", type=float)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.list_scenarios:
        return
    if not args.base_url:
        parser.error("--base-url is required")
    if args.baseline_qps <= 0 or args.peak_qps <= 0:
        parser.error("--baseline-qps and --peak-qps must be positive")
    if args.peak_qps < args.baseline_qps:
        parser.error("--peak-qps must be greater than or equal to --baseline-qps")
    if args.duration_scale <= 0:
        parser.error("--duration-scale must be positive")
    if args.total_duration_sec is not None and args.total_duration_sec <= 0:
        parser.error("--total-duration-sec must be positive")
    if not 0 < args.random_range_ratio <= 1:
        parser.error("--random-range-ratio must be in (0, 1]")
    if args.max_concurrency is not None and args.max_concurrency <= 0:
        parser.error("--max-concurrency must be positive")
    if args.max_pending_requests <= 0:
        parser.error("--max-pending-requests must be positive")
    if args.context_length is not None and args.context_length <= 0:
        parser.error("--context-length must be positive")
    if args.prompt_pool_size <= 0:
        parser.error("--prompt-pool-size must be positive")
    if args.warmup_requests < 0:
        parser.error("--warmup-requests cannot be negative")
    if args.health_check_interval_sec < 0:
        parser.error("--health-check-interval-sec cannot be negative")
    if args.max_health_failures < 0:
        parser.error("--max-health-failures cannot be negative")
    if args.extra_request_body:
        try:
            value = json.loads(args.extra_request_body)
        except json.JSONDecodeError as exc:
            parser.error(f"--extra-request-body is not valid JSON: {exc}")
        if not isinstance(value, dict):
            parser.error("--extra-request-body must be a JSON object")


def cli_main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(parser, args)
    if args.list_scenarios:
        for scenario in SCENARIOS.values():
            print(f"{scenario.name:18} {scenario.description}")
        return
    raise SystemExit(run_suite(args))


if __name__ == "__main__":
    cli_main()
