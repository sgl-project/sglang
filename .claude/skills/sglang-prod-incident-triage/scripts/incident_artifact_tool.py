#!/usr/bin/env python3
"""Collect or inspect serving bundles and dumps for SGLang debug."""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import pickle
import re
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Sequence
from urllib import error, parse, request

METRIC_RE = re.compile(
    r"^(?P<name>[^{\s]+)(?:\{(?P<labels>[^}]*)\})?\s+(?P<value>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)$"
)
LABEL_RE = re.compile(r'([a-zA-Z_:][a-zA-Z0-9_:]*)="((?:[^"\\]|\\.)*)"')
ENDPOINT_SPECS = (
    ("text", "health.txt", "/health"),
    ("text", "health_generate.txt", "/health_generate"),
    ("text", "metrics.txt", "/metrics"),
    ("json", "model_info.json", "/model_info"),
    ("json", "server_info.json", "/server_info"),
    ("json", "loads_all.json", "/v1/loads?include=all"),
    (
        "json",
        "loads_core_queues_disagg.json",
        "/v1/loads?include=core,queues,disagg,spec",
    ),
    ("json", "hicache_storage_backend.json", "/hicache/storage-backend"),
)
BUNDLE_NOTES = [
    "This bundle is read-only. It does not start profiling or change trace level.",
    "HiCache status may fail if admin_api_key is not configured or the wrong bearer token was used.",
    "loads_all.json is the best point-in-time load snapshot in this bundle.",
    "metrics.txt is raw Prometheus text intended for follow-up parsing.",
]


def request_text(
    base_url: str,
    path: str,
    token: Optional[str],
    timeout: float = 10.0,
) -> tuple[bool, int, str]:
    url = parse.urljoin(base_url.rstrip("/") + "/", path.lstrip("/"))
    req = request.Request(url)
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            return True, resp.status, body
    except error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        return False, e.code, body
    except Exception as e:  # noqa: BLE001
        return False, -1, f"{type(e).__name__}: {e}"


def request_endpoint(
    base_url: str,
    path: str,
    token: Optional[str],
    parse_json: bool,
    timeout: float = 10.0,
) -> Dict[str, Any]:
    ok, status, body = request_text(base_url, path, token, timeout=timeout)
    result: Dict[str, Any] = {"ok": ok, "status": status, "path": path}
    if not ok:
        result["error"] = body
        return result
    if not parse_json:
        result["text"] = body
        return result
    try:
        result["json"] = json.loads(body)
    except json.JSONDecodeError:
        result["text"] = body
        result["decode_error"] = "response was not valid JSON"
    return result


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def format_summary_line(filename: str, result: Dict[str, Any]) -> str:
    if result.get("ok"):
        return f"{filename}: ok"
    return (
        f"{filename}: failed status={result.get('status')} "
        f"error={result.get('error')}"
    )


def collect_bundle(
    base_url: str,
    token: Optional[str],
    outdir: Optional[str],
    timeout: float,
) -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    bundle_dir = Path(outdir or f"./incident_bundle_{timestamp}").resolve()
    bundle_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "artifact_type": "incident_bundle",
        "base_url": base_url,
        "collected_at": timestamp,
        "token_provided": bool(token),
        "timeout_seconds": timeout,
    }
    write_json(bundle_dir / "metadata.json", metadata)

    summary_lines = []
    for kind, filename, path in ENDPOINT_SPECS:
        result = request_endpoint(
            base_url, path, token, parse_json=(kind == "json"), timeout=timeout
        )
        output_path = bundle_dir / filename
        if kind == "text" and result.get("ok"):
            write_text(output_path, str(result.get("text", "")))
        else:
            write_json(
                (
                    output_path
                    if kind == "json"
                    else bundle_dir / f"{filename}.error.json"
                ),
                result,
            )
        summary_lines.append(format_summary_line(filename, result))

    write_text(
        bundle_dir / "SUMMARY.txt",
        "\n".join(summary_lines + [""] + BUNDLE_NOTES) + "\n",
    )
    return bundle_dir


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def unwrap_result(path: Path) -> Optional[Dict[str, Any]]:
    obj = load_json(path)
    if obj is None:
        return None
    if isinstance(obj, dict) and "json" in obj:
        return obj.get("json")
    return obj


def read_text(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def endpoint_ok(bundle_dir: Path, stem: str) -> bool:
    return (bundle_dir / f"{stem}.txt").exists() and not (
        bundle_dir / f"{stem}.txt.error.json"
    ).exists()


def parse_labels(raw: Optional[str]) -> Dict[str, str]:
    if not raw:
        return {}
    labels = {}
    for key, value in LABEL_RE.findall(raw):
        labels[key] = bytes(value, "utf-8").decode("unicode_escape")
    return labels


def parse_metrics(metrics_text: str) -> Dict[str, list[dict[str, Any]]]:
    series: Dict[str, list[dict[str, Any]]] = defaultdict(list)
    for line in metrics_text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = METRIC_RE.match(line)
        if not match:
            continue
        series[match.group("name")].append(
            {
                "labels": parse_labels(match.group("labels")),
                "value": float(match.group("value")),
            }
        )
    return series


def metric_sum(metrics: Dict[str, list[dict[str, Any]]], name: str) -> float:
    return sum(item["value"] for item in metrics.get(name, []))


def safe_div(
    numerator: Optional[float], denominator: Optional[float]
) -> Optional[float]:
    if numerator is None or denominator in (None, 0):
        return None
    return numerator / denominator


def coalesce(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def fmt_float(value: Optional[float], digits: int = 3) -> str:
    if value is None or (
        isinstance(value, float) and (math.isnan(value) or math.isinf(value))
    ):
        return "n/a"
    return f"{value:.{digits}f}"


def is_positive_number(value: Any, threshold: float = 0.0) -> bool:
    return (
        isinstance(value, (int, float))
        and not math.isnan(value)
        and not math.isinf(value)
        and value > threshold
    )


def compute_stage_averages(
    metrics: Dict[str, list[dict[str, Any]]], sum_name: str, count_name: str
) -> Dict[str, float]:
    grouped_sum: Dict[str, float] = defaultdict(float)
    grouped_count: Dict[str, float] = defaultdict(float)
    for item in metrics.get(sum_name, []):
        stage = item["labels"].get("stage", "")
        rank = item["labels"].get("tp_rank", "")
        grouped_sum[f"{stage}|{rank}"] += item["value"]
    for item in metrics.get(count_name, []):
        stage = item["labels"].get("stage", "")
        rank = item["labels"].get("tp_rank", "")
        grouped_count[f"{stage}|{rank}"] += item["value"]

    result: Dict[str, float] = {}
    for key, total_sum in grouped_sum.items():
        stage, _rank = key.split("|", 1)
        avg = safe_div(total_sum, grouped_count.get(key))
        if avg is None:
            continue
        result[stage] = max(result.get(stage, 0.0), avg)
    return result


def add_signal(signals: list[str], text: str) -> None:
    if text not in signals:
        signals.append(text)


def build_bundle_summary(bundle_dir: Path) -> Dict[str, Any]:
    metadata = load_json(bundle_dir / "metadata.json") or {}
    model_info = unwrap_result(bundle_dir / "model_info.json") or {}
    server_info = unwrap_result(bundle_dir / "server_info.json") or {}
    loads_info = unwrap_result(bundle_dir / "loads_all.json") or {}
    metrics_text = read_text(bundle_dir / "metrics.txt") or ""
    metrics = parse_metrics(metrics_text)

    aggregate = loads_info.get("aggregate") or {}
    loads = loads_info.get("loads") or []
    load0 = loads[0] if loads else {}
    internal_states = server_info.get("internal_states") or []
    runtime_state = internal_states[0] if internal_states else {}
    memory_usage = runtime_state.get("memory_usage") or load0.get("memory") or {}

    ttft_avg = safe_div(
        metric_sum(metrics, "sglang:time_to_first_token_seconds_sum"),
        metric_sum(metrics, "sglang:time_to_first_token_seconds_count"),
    )
    e2e_avg = safe_div(
        metric_sum(metrics, "sglang:e2e_request_latency_seconds_sum"),
        metric_sum(metrics, "sglang:e2e_request_latency_seconds_count"),
    )
    queue_avg = safe_div(
        metric_sum(metrics, "sglang:queue_time_seconds_sum"),
        metric_sum(metrics, "sglang:queue_time_seconds_count"),
    )
    per_stage_avg = compute_stage_averages(
        metrics,
        "sglang:per_stage_req_latency_seconds_sum",
        "sglang:per_stage_req_latency_seconds_count",
    )

    summary: Dict[str, Any] = {
        "artifact_type": "incident_bundle",
        "bundle_dir": str(bundle_dir),
        "base_url": metadata.get("base_url"),
        "collected_at": metadata.get("collected_at"),
        "health": {
            "health_ok": endpoint_ok(bundle_dir, "health"),
            "health_generate_ok": endpoint_ok(bundle_dir, "health_generate"),
        },
        "model": {
            "model_path": model_info.get("model_path") or server_info.get("model_path"),
            "served_model_name": server_info.get("served_model_name"),
            "weight_version": model_info.get("weight_version")
            or server_info.get("weight_version"),
            "model_type": model_info.get("model_type"),
            "is_generation": model_info.get("is_generation"),
        },
        "topology": {
            "tp_size": server_info.get("tp_size"),
            "dp_size": server_info.get("dp_size"),
            "pp_size": server_info.get("pp_size"),
            "ep_size": server_info.get("ep_size"),
            "disaggregation_mode": server_info.get("disaggregation_mode"),
            "attention_backend": server_info.get("attention_backend"),
            "sampling_backend": server_info.get("sampling_backend"),
            "schedule_policy": server_info.get("schedule_policy"),
            "enable_trace": server_info.get("enable_trace"),
            "enable_metrics": server_info.get("enable_metrics"),
        },
        "capacity": {
            "max_total_num_tokens": server_info.get("max_total_num_tokens"),
            "max_req_input_len": server_info.get("max_req_input_len"),
            "effective_max_running_requests_per_dp": coalesce(
                runtime_state.get("effective_max_running_requests_per_dp"),
                load0.get("max_running_requests"),
            ),
            "weight_gb": coalesce(
                memory_usage.get("weight"), memory_usage.get("weight_gb")
            ),
            "kv_cache_gb": coalesce(
                memory_usage.get("kvcache"), memory_usage.get("kv_cache_gb")
            ),
            "graph_gb": coalesce(
                memory_usage.get("graph"), memory_usage.get("graph_gb")
            ),
            "token_capacity": memory_usage.get("token_capacity"),
        },
        "point_in_time_load": {
            "running_reqs": coalesce(
                aggregate.get("total_running_reqs"), load0.get("num_running_reqs")
            ),
            "waiting_reqs": coalesce(
                aggregate.get("total_waiting_reqs"), load0.get("num_waiting_reqs")
            ),
            "total_reqs": coalesce(
                aggregate.get("total_reqs"), load0.get("num_total_reqs")
            ),
            "token_usage": coalesce(
                aggregate.get("avg_token_usage"), load0.get("token_usage")
            ),
            "avg_throughput": coalesce(
                aggregate.get("avg_throughput"), load0.get("gen_throughput")
            ),
            "avg_utilization": coalesce(
                aggregate.get("avg_utilization"), load0.get("utilization")
            ),
            "cache_hit_rate": load0.get("cache_hit_rate"),
            "queues": load0.get("queues"),
            "disaggregation": load0.get("disaggregation"),
        },
        "metrics": {
            "request_count": metric_sum(metrics, "sglang:num_requests_total"),
            "prompt_tokens_total": metric_sum(metrics, "sglang:prompt_tokens_total"),
            "generation_tokens_total": metric_sum(
                metrics, "sglang:generation_tokens_total"
            ),
            "avg_ttft_seconds": ttft_avg,
            "avg_e2e_seconds": e2e_avg,
            "avg_queue_time_seconds": queue_avg,
            "stage_avg_seconds_max_tp_rank": per_stage_avg,
        },
        "signals": [],
    }

    signals = summary["signals"]
    health = summary["health"]
    point_in_time_load = summary["point_in_time_load"]
    running_reqs = point_in_time_load.get("running_reqs")
    waiting_reqs = point_in_time_load.get("waiting_reqs")

    if health["health_ok"] and not health["health_generate_ok"]:
        add_signal(
            signals,
            "/health is green but /health_generate failed. Suspect runtime or scheduler path, not just HTTP liveness.",
        )
    if not health["health_ok"]:
        add_signal(
            signals,
            "/health failed. Start with startup, crash, or global unhealthy paths.",
        )
    if is_positive_number(waiting_reqs):
        add_signal(
            signals,
            f"Point-in-time load shows queue buildup: waiting_reqs={waiting_reqs}.",
        )
    if (
        point_in_time_load.get("token_usage") is not None
        and point_in_time_load["token_usage"] >= 0.9
    ):
        add_signal(
            signals,
            "Token usage is near saturation. KV or token-capacity pressure may explain latency.",
        )
    if (
        ttft_avg is not None
        and queue_avg is not None
        and ttft_avg > 2.0
        and queue_avg < 0.2
    ):
        add_signal(
            signals,
            f"Average TTFT is high ({fmt_float(ttft_avg)}s) while average queue time is low ({fmt_float(queue_avg)}s). This looks more like prefill or request-path work than queue pressure.",
        )
    prefill_forward = per_stage_avg.get("prefill_forward")
    request_process = per_stage_avg.get("request_process")
    if (
        prefill_forward is not None
        and request_process is not None
        and prefill_forward > max(0.5, request_process * 10)
    ):
        add_signal(
            signals,
            f"Prefill forward dominates quick stage timing: prefill_forward~{fmt_float(prefill_forward)}s vs request_process~{fmt_float(request_process)}s.",
        )
    if running_reqs == 0 and waiting_reqs == 0:
        add_signal(
            signals,
            "Bundle snapshot was captured while the server was effectively idle. Reproduce under live traffic or replayed workload if the problem is intermittent.",
        )

    return summary


def render_bundle_text(summary: Dict[str, Any]) -> str:
    health = summary["health"]
    model = summary["model"]
    topology = summary["topology"]
    capacity = summary["capacity"]
    load = summary["point_in_time_load"]
    metrics = summary["metrics"]
    stage_avgs = metrics["stage_avg_seconds_max_tp_rank"]

    lines = [
        f"Bundle: {summary['bundle_dir']}",
        f"Base URL: {summary.get('base_url') or 'n/a'}",
        f"Collected At: {summary.get('collected_at') or 'n/a'}",
        "",
        f"Health: /health={'ok' if health['health_ok'] else 'failed'} /health_generate={'ok' if health['health_generate_ok'] else 'failed'}",
        f"Model: {model.get('model_path') or 'n/a'} weight_version={model.get('weight_version') or 'n/a'} type={model.get('model_type') or 'n/a'}",
        "Topology: "
        f"tp={topology.get('tp_size')} dp={topology.get('dp_size')} pp={topology.get('pp_size')} ep={topology.get('ep_size')} "
        f"disagg={topology.get('disaggregation_mode')} trace={topology.get('enable_trace')} metrics={topology.get('enable_metrics')}",
        "Capacity: "
        f"max_total_tokens={capacity.get('max_total_num_tokens')} "
        f"max_running_reqs={capacity.get('effective_max_running_requests_per_dp')} "
        f"weight_gb={fmt_float(capacity.get('weight_gb'))} "
        f"kv_cache_gb={fmt_float(capacity.get('kv_cache_gb'))} "
        f"graph_gb={fmt_float(capacity.get('graph_gb'))}",
        "Point-in-time load: "
        f"running={load.get('running_reqs')} waiting={load.get('waiting_reqs')} total={load.get('total_reqs')} "
        f"token_usage={fmt_float(load.get('token_usage'))} throughput={fmt_float(load.get('avg_throughput'))} "
        f"cache_hit_rate={fmt_float(load.get('cache_hit_rate'))}",
        "Metrics: "
        f"requests={fmt_float(metrics.get('request_count'), 0)} "
        f"prompt_tokens={fmt_float(metrics.get('prompt_tokens_total'), 0)} "
        f"generation_tokens={fmt_float(metrics.get('generation_tokens_total'), 0)} "
        f"avg_ttft_s={fmt_float(metrics.get('avg_ttft_seconds'))} "
        f"avg_e2e_s={fmt_float(metrics.get('avg_e2e_seconds'))} "
        f"avg_queue_s={fmt_float(metrics.get('avg_queue_time_seconds'))}",
    ]

    if stage_avgs:
        stage_parts = [
            f"{name}={fmt_float(value)}s" for name, value in sorted(stage_avgs.items())
        ]
        lines.append("Stage Averages (max across TP ranks): " + ", ".join(stage_parts))

    queues = load.get("queues") or {}
    if queues:
        lines.append(
            "Queues: "
            + ", ".join(f"{key}={value}" for key, value in sorted(queues.items()))
        )

    disagg = load.get("disaggregation") or {}
    if disagg:
        lines.append(
            "Disaggregation: "
            + ", ".join(f"{key}={value}" for key, value in sorted(disagg.items()))
        )

    lines.append("")
    lines.append("What stands out:")
    if summary["signals"]:
        lines.extend(f"- {signal}" for signal in summary["signals"])
    else:
        lines.append("- No strong signal from this bundle.")

    return "\n".join(lines) + "\n"


def get_field(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def iter_dump_files(
    input_file: Optional[str], input_folder: Optional[str]
) -> Sequence[Path]:
    if input_file:
        return [Path(input_file)]
    if input_folder:
        return [Path(p) for p in sorted(glob.glob(f"{input_folder}/*.pkl"))]
    raise SystemExit("Either --input-file or --input-folder must be provided.")


def load_dump_payload(path: Path) -> dict[str, Any]:
    with path.open("rb") as fh:
        payload = pickle.load(fh)
    if isinstance(payload, dict):
        return payload
    return {"requests": payload}


def pick_text_preview(req: Any) -> str:
    candidates = [
        get_field(req, "origin_input_text"),
        get_field(req, "text"),
        get_field(req, "prompt"),
    ]
    for value in candidates:
        if isinstance(value, str) and value:
            return value
        if isinstance(value, list) and value:
            first = value[0]
            if isinstance(first, str) and first:
                return first
    return ""


def format_timestamp(ts: Any) -> str:
    if not isinstance(ts, (int, float)):
        return "n/a"
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def summarize_request(
    record: tuple[Any, dict[str, Any], Any, Any], idx: int, preview_chars: int
) -> list[str]:
    req, output, start_time, end_time = record
    preview = pick_text_preview(req).replace("\n", " ").strip()
    if len(preview) > preview_chars:
        preview = preview[: preview_chars - 3] + "..."

    output_dict = output if isinstance(output, dict) else {}
    meta_info = get_field(output_dict, "meta_info", {}) or {}
    rid = get_field(req, "rid") or get_field(meta_info, "id")
    stream = bool(get_field(req, "stream", False))
    prompt_tokens = get_field(meta_info, "prompt_tokens")
    completion_tokens = get_field(meta_info, "completion_tokens")
    duration = (
        end_time - start_time
        if isinstance(start_time, (int, float)) and isinstance(end_time, (int, float))
        else None
    )

    elapsed_str = f"{duration:.3f}" if duration is not None else "n/a"
    lines = [
        f"[{idx}] rid={rid or 'n/a'} stream={stream} "
        f"prompt_tokens={prompt_tokens if prompt_tokens is not None else 'n/a'} "
        f"completion_tokens={completion_tokens if completion_tokens is not None else 'n/a'} "
        f"start={format_timestamp(start_time)} elapsed_s={elapsed_str}"
    ]
    if preview:
        lines.append(f"      text={preview}")
    return lines


def summarize_dump_file(path: Path, max_requests: int, preview_chars: int) -> str:
    payload = load_dump_payload(path)
    requests = payload.get("requests") or []
    server_args = payload.get("server_args")
    launch_command = payload.get("launch_command")

    model_path = get_field(server_args, "model_path")
    tp_size = get_field(server_args, "tp_size")
    dp_size = get_field(server_args, "dp_size")
    pp_size = get_field(server_args, "pp_size")
    host = get_field(server_args, "host")
    port = get_field(server_args, "port")

    timestamps = [
        record[2]
        for record in requests
        if isinstance(record, tuple)
        and len(record) >= 4
        and isinstance(record[2], (int, float))
    ]
    time_span = (
        max(timestamps) - min(timestamps)
        if len(timestamps) >= 2
        else 0.0 if len(timestamps) == 1 else None
    )

    lines = [
        f"File: {path}",
        "Dump Type: request_or_crash_dump",
        f"Requests: {len(requests)}",
        f"Model: {model_path or 'n/a'}",
        f"Topology: tp={tp_size if tp_size is not None else 'n/a'} "
        f"dp={dp_size if dp_size is not None else 'n/a'} "
        f"pp={pp_size if pp_size is not None else 'n/a'}",
        f"Endpoint: {host or 'n/a'}:{port if port is not None else 'n/a'}",
        (
            f"Time span seconds: {time_span:.3f}"
            if time_span is not None
            else "Time span seconds: n/a"
        ),
    ]
    if launch_command:
        lines.append(f"Launch command: {launch_command}")

    for idx, record in enumerate(requests[:max_requests]):
        if not isinstance(record, tuple) or len(record) < 4:
            lines.append(f"[{idx}] Unsupported record shape: {type(record)!r}")
            continue
        lines.extend(summarize_request(record, idx, preview_chars))

    if len(requests) > max_requests:
        lines.append(f"... truncated {len(requests) - max_requests} more requests")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Collect or inspect serving bundles and dumps for SGLang debug."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect_parser = subparsers.add_parser(
        "collect-bundle", help="Collect a read-only live bundle from a running server"
    )
    collect_parser.add_argument("--base-url", required=True)
    collect_parser.add_argument(
        "--token",
        default=os.environ.get("SGLANG_BEARER_TOKEN"),
        help="Bearer token for protected endpoints. Defaults to $SGLANG_BEARER_TOKEN.",
    )
    collect_parser.add_argument("--outdir", default=None)
    collect_parser.add_argument("--timeout", type=float, default=10.0)

    bundle_parser = subparsers.add_parser(
        "summarize-bundle", help="Summarize a bundle directory"
    )
    bundle_parser.add_argument("bundle_dir")
    bundle_parser.add_argument("--out", default=None)
    bundle_parser.add_argument("--json-out", default=None)
    bundle_parser.add_argument("--stdout-json", action="store_true")

    dump_parser = subparsers.add_parser(
        "summarize-dump", help="Summarize a trusted request dump or crash dump"
    )
    dump_parser.add_argument("--input-file", default=None)
    dump_parser.add_argument("--input-folder", default=None)
    dump_parser.add_argument("--max-requests", type=int, default=20)
    dump_parser.add_argument("--preview-chars", type=int, default=160)

    args = parser.parse_args()

    if args.command == "collect-bundle":
        bundle_dir = collect_bundle(
            args.base_url, args.token, args.outdir, args.timeout
        )
        print(bundle_dir)
        return 0

    if args.command == "summarize-bundle":
        bundle_dir = Path(args.bundle_dir).resolve()
        if not bundle_dir.is_dir():
            raise SystemExit(
                f"bundle_dir does not exist or is not a directory: {bundle_dir}"
            )
        summary = build_bundle_summary(bundle_dir)
        out_text = render_bundle_text(summary)
        text_path = Path(args.out) if args.out else bundle_dir / "SUMMARY_REPORT.txt"
        json_path = (
            Path(args.json_out) if args.json_out else bundle_dir / "SUMMARY_REPORT.json"
        )
        text_path.write_text(out_text, encoding="utf-8")
        json_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        if args.stdout_json:
            print(json.dumps(summary, indent=2, ensure_ascii=False))
        else:
            print(out_text, end="")
        return 0

    files = iter_dump_files(args.input_file, args.input_folder)
    if not files:
        raise SystemExit("No .pkl files matched the provided input.")
    for idx, path in enumerate(files):
        if idx:
            print()
        print(
            summarize_dump_file(
                path=path,
                max_requests=args.max_requests,
                preview_chars=args.preview_chars,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
