"""Run paper-relevant DSV4 IndexCache quality evals against one or more endpoints."""

from __future__ import annotations

import argparse
import ast
import json
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from indexcache_base_path import (
    fetch_server_info,
)
from indexcache_base_path import (
    validate_server_info_for_base_path as validate_base_server_info,
)

FORBIDDEN_TASKS = {"mmlu", "gsm8k"}
PRIMARY_METRIC_KEYS = (
    "score",
    "accuracy",
    "overall",
    "average",
    "pass_at_1",
    "exact_match",
    "f1",
)


@dataclass(frozen=True)
class EvalTask:
    name: str
    runner: str
    eval_name: str


TASKS = {
    "ruler": EvalTask("ruler", "sgl-eval", "ruler"),
    "mrcr_v2": EvalTask("mrcr_v2", "sgl-eval", "mrcr_v2"),
    "graphwalks": EvalTask("graphwalks", "sgl-eval", "graphwalks"),
    "longbench_v2": EvalTask("longbench_v2", "sglang.test.run_eval", "longbench_v2"),
    "aa_lcr": EvalTask("aa_lcr", "sgl-eval", "aa_lcr"),
    "aime25": EvalTask("aime25", "sglang.test.run_eval", "aime25"),
    "gpqa": EvalTask("gpqa", "sglang.test.run_eval", "gpqa"),
    "livecodebench": EvalTask("livecodebench", "sgl-eval", "livecodebench"),
    "ifbench": EvalTask("ifbench", "sgl-eval", "ifbench"),
}

SUITES = {
    "long-context": ["ruler", "mrcr_v2", "graphwalks", "longbench_v2", "aa_lcr"],
    "reasoning": ["aime25", "gpqa", "livecodebench", "ifbench"],
}


def parse_endpoint(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("endpoint must be label=base_url")
    label, base_url = value.split("=", maxsplit=1)
    if not label or not base_url:
        raise argparse.ArgumentTypeError("endpoint must be label=base_url")
    return label, base_url.rstrip("/")


def selected_tasks(task_names: list[str], suites: list[str]) -> list[EvalTask]:
    names = []
    for suite in suites:
        if suite not in SUITES:
            raise ValueError(f"unknown suite {suite}; choices: {sorted(SUITES)}")
        names.extend(SUITES[suite])
    names.extend(task_names)

    unknown = sorted(set(names) - set(TASKS) - FORBIDDEN_TASKS)
    if unknown:
        raise ValueError(f"unknown eval tasks {unknown}; choices: {sorted(TASKS)}")
    forbidden = sorted(set(names) & FORBIDDEN_TASKS)
    if forbidden:
        raise ValueError(
            f"{forbidden} are intentionally blocked for DSV4 IndexCache validation; "
            "use long-context/reasoning evals instead"
        )

    deduped = []
    seen = set()
    for name in names:
        if name not in seen:
            seen.add(name)
            deduped.append(TASKS[name])
    return deduped


def build_sglang_eval_cmd(task: EvalTask, base_url: str, args) -> list[str]:
    cmd = [
        "python",
        "-m",
        "sglang.test.run_eval",
        "--eval-name",
        task.eval_name,
        "--base-url",
        base_url,
        "--num-threads",
        str(args.num_threads),
        "--max-tokens",
        str(args.max_tokens),
        "--temperature",
        str(args.temperature),
        "--top-p",
        str(args.top_p),
    ]
    if args.num_examples is not None:
        cmd += ["--num-examples", str(args.num_examples)]
    if args.dataset_path and task.name == "longbench_v2":
        cmd += ["--dataset-path", args.dataset_path]
    if args.min_context_length and task.name == "longbench_v2":
        cmd += ["--min-context-length", str(args.min_context_length)]
    if args.max_context_length and task.name == "longbench_v2":
        cmd += ["--max-context-length", str(args.max_context_length)]
    return cmd


def build_sgl_eval_cmd(task: EvalTask, base_url: str, args) -> list[str]:
    out_dir = sgl_eval_out_dir(task, args)
    cmd = [
        args.sgl_eval_bin,
        "run",
        task.eval_name,
        "--base-url",
        f"{base_url}/v1",
        "--temperature",
        str(args.temperature),
        "--top-p",
        str(args.top_p),
        "--max-tokens",
        str(args.max_tokens),
        "--num-threads",
        str(args.num_threads),
        "--out-dir",
        str(out_dir),
    ]
    if args.num_examples is not None:
        cmd += ["--num-examples", str(args.num_examples)]
    return cmd


def sgl_eval_out_dir(task: EvalTask, args) -> Path:
    out_dir = args.out_dir
    if getattr(args, "endpoint_label", None) and getattr(args, "repeat_index", None):
        out_dir = (
            out_dir / args.endpoint_label / task.name / f"repeat_{args.repeat_index}"
        )
    return out_dir


def build_eval_cmd(task: EvalTask, base_url: str, args) -> list[str]:
    if task.runner == "sglang.test.run_eval":
        return build_sglang_eval_cmd(task, base_url, args)
    if task.runner == "sgl-eval":
        return build_sgl_eval_cmd(task, base_url, args)
    raise ValueError(f"unknown runner {task.runner}")


def validate_server_info_for_base_path(server_info: dict, label: str) -> list[str]:
    return validate_base_server_info(server_info, f"quality eval endpoint {label!r}")


def validate_endpoint_for_base_path(label: str, base_url: str, timeout: int) -> dict:
    server_info = fetch_server_info(base_url, timeout)
    validate_server_info_for_base_path(server_info, label)
    return server_info


def run_eval_task(task: EvalTask, base_url: str, args) -> dict:
    cmd = build_eval_cmd(task, base_url, args)
    if args.dry_run:
        return {
            "cmd": cmd,
            "returncode": 0,
            "elapsed_sec": 0,
            "output": "",
            "metrics": {},
            "primary_metric": None,
        }
    if task.runner == "sgl-eval" and shutil.which(args.sgl_eval_bin) is None:
        raise RuntimeError(f"{args.sgl_eval_bin!r} not found on PATH")
    result = run_command(cmd, args.timeout)
    out_dir = sgl_eval_out_dir(task, args) if task.runner == "sgl-eval" else None
    metrics = extract_metrics(result["output"], out_dir)
    result["metrics"] = metrics
    result["primary_metric"] = choose_primary_metric(metrics)
    if args.require_metrics and result["returncode"] == 0 and not result["metrics"]:
        raise RuntimeError(
            f"{task.name} completed but no parseable metrics were found in stdout"
            + (f" or {out_dir}" if out_dir else "")
        )
    return result


def run_command(cmd: list[str], timeout: int) -> dict:
    start = time.perf_counter()
    proc = subprocess.run(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
    )
    elapsed = time.perf_counter() - start
    return {
        "cmd": cmd,
        "returncode": proc.returncode,
        "elapsed_sec": elapsed,
        "output": proc.stdout,
    }


def metric_candidates_from_text(output: str) -> list[dict]:
    candidates = []
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        candidates.extend(metric_candidates_from_object(parse_object(stripped)))
        match = re.search(r"metrics=(\{.*\})(?:\s+score=|\s*$)", stripped)
        if match:
            candidates.extend(
                metric_candidates_from_object(parse_object(match.group(1), python=True))
            )
    return candidates


def metric_candidates_from_files(out_dir: Path | None) -> list[dict]:
    if out_dir is None or not out_dir.exists():
        return []
    candidates = []
    for path in sorted(out_dir.rglob("*.json")):
        try:
            candidates.extend(
                metric_candidates_from_object(json.loads(path.read_text()))
            )
        except (OSError, json.JSONDecodeError):
            continue
    return candidates


def parse_object(value: str, python: bool = False):
    try:
        return ast.literal_eval(value) if python else json.loads(value)
    except (SyntaxError, ValueError, json.JSONDecodeError):
        return None


def metric_candidates_from_object(value) -> list[dict]:
    if isinstance(value, dict):
        candidates = []
        metrics = value.get("metrics")
        if isinstance(metrics, dict):
            candidates.append(flatten_numeric_metrics(metrics))
        own_metrics = flatten_numeric_metrics(value)
        if own_metrics:
            candidates.append(own_metrics)
        for item in value.values():
            candidates.extend(metric_candidates_from_object(item))
        return candidates
    if isinstance(value, list):
        candidates = []
        for item in value:
            candidates.extend(metric_candidates_from_object(item))
        return candidates
    return []


def flatten_numeric_metrics(value: dict, prefix: str = "") -> dict:
    metrics = {}
    for key, item in value.items():
        if isinstance(item, bool):
            continue
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(item, (int, float)):
            metrics[name] = float(item)
        elif isinstance(item, dict):
            metrics.update(flatten_numeric_metrics(item, name))
    return metrics


def extract_metrics(output: str, out_dir: Path | None = None) -> dict:
    candidates = metric_candidates_from_text(output)
    candidates.extend(metric_candidates_from_files(out_dir))
    merged = {}
    for candidate in candidates:
        merged.update(candidate)
    return merged


def choose_primary_metric(metrics: dict) -> dict | None:
    for key in PRIMARY_METRIC_KEYS:
        if key in metrics:
            return {"name": key, "value": metrics[key]}
    for key in sorted(metrics):
        suffix = key.rsplit(".", maxsplit=1)[-1]
        if suffix in PRIMARY_METRIC_KEYS:
            return {"name": key, "value": metrics[key]}
    if metrics:
        key = sorted(metrics)[0]
        return {"name": key, "value": metrics[key]}
    return None


def summarize_primary_metrics(rows: list[dict]) -> dict:
    grouped: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        primary_metric = row.get("primary_metric")
        if not primary_metric:
            continue
        key = f"{row['endpoint']}::{row['task']}::{primary_metric['name']}"
        grouped.setdefault(key, {"values": []})["values"].append(
            primary_metric["value"]
        )

    summary = {}
    for key, data in grouped.items():
        endpoint, task, metric = key.split("::", maxsplit=2)
        values = data["values"]
        summary.setdefault(endpoint, {})[task] = {
            "metric": metric,
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
            "n": len(values),
        }
    return summary


def compare_primary_metrics(summary: dict, baseline_label: str) -> dict:
    baseline = summary.get(baseline_label, {})
    comparisons = {}
    for endpoint, task_metrics in summary.items():
        if endpoint == baseline_label:
            continue
        for task, item in task_metrics.items():
            baseline_item = baseline.get(task)
            comparison = {
                "metric": item["metric"],
                "mean": item["mean"],
                "baseline_mean": None,
                "delta": None,
                "status": "missing_baseline",
            }
            if baseline_item is not None:
                comparison["baseline_mean"] = baseline_item["mean"]
                if baseline_item["metric"] == item["metric"]:
                    comparison["delta"] = item["mean"] - baseline_item["mean"]
                    comparison["status"] = "compared"
                else:
                    comparison["status"] = "metric_mismatch"
                    comparison["baseline_metric"] = baseline_item["metric"]
            comparisons.setdefault(endpoint, {})[task] = comparison
    return comparisons


def enforce_quality_drop(comparisons: dict, max_drop: float | None) -> None:
    failures = quality_drop_failures(comparisons, max_drop)
    if failures:
        raise RuntimeError(
            "quality metric drop exceeded threshold: " + "; ".join(failures)
        )


def quality_drop_failures(comparisons: dict, max_drop: float | None) -> list[str]:
    if max_drop is None:
        return []
    failures = []
    for endpoint, task_metrics in comparisons.items():
        for task, comparison in task_metrics.items():
            if comparison["status"] != "compared":
                failures.append(f"{endpoint}/{task}: {comparison['status']}")
                continue
            if comparison["delta"] < -max_drop:
                failures.append(
                    f"{endpoint}/{task}: delta {comparison['delta']:.6g} "
                    f"< allowed {-max_drop:.6g}"
                )
    return failures


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--endpoint",
        action="append",
        type=parse_endpoint,
        required=True,
        help="Endpoint as label=base_url. Pass twice for baseline/indexcache.",
    )
    parser.add_argument("--suite", action="append", default=[])
    parser.add_argument("--task", action="append", default=[])
    parser.add_argument("--num-examples", type=int)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--num-threads", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--min-context-length", type=int)
    parser.add_argument("--max-context-length", type=int)
    parser.add_argument("--dataset-path")
    parser.add_argument("--sgl-eval-bin", default="sgl-eval")
    parser.add_argument("--out-dir", type=Path, default=Path("/tmp/sgl-eval-out"))
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-label", default="baseline")
    parser.add_argument("--max-primary-metric-drop", type=float)
    parser.add_argument("--require-metrics", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.repeats < 1:
        raise SystemExit("--repeats must be at least 1")
    endpoint_labels = [label for label, _ in args.endpoint]
    duplicate_labels = sorted(
        label for label in set(endpoint_labels) if endpoint_labels.count(label) > 1
    )
    if duplicate_labels:
        raise SystemExit(
            f"duplicate endpoint labels are not allowed: {duplicate_labels}"
        )
    if args.baseline_label not in endpoint_labels:
        raise SystemExit(
            f"--baseline-label {args.baseline_label!r} must match one endpoint label"
        )
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    suites = args.suite or (["long-context", "reasoning"] if not args.task else [])
    tasks = selected_tasks(args.task, suites)
    results = {
        "tasks": [task.name for task in tasks],
        "endpoints": [label for label, _ in args.endpoint],
        "server_checks": {},
        "results": [],
        "primary_metric_summary": {},
        "primary_metric_comparison": {},
        "quality_gate": {},
    }

    for label, base_url in args.endpoint:
        if args.dry_run:
            results["server_checks"][label] = {
                "server_info_checked": False,
                "speculative_decode": "dry run; /server_info not queried",
            }
        else:
            results["server_checks"][label] = {
                "server_info_checked": True,
                "speculative_decode": "confirmed off via /server_info",
                "server_info": validate_endpoint_for_base_path(
                    label, base_url, args.timeout
                ),
            }
        for repeat_index in range(args.repeats):
            args.endpoint_label = label
            args.repeat_index = repeat_index + 1
            for task in tasks:
                result = run_eval_task(task, base_url, args)
                results["results"].append(
                    {
                        "endpoint": label,
                        "repeat": repeat_index + 1,
                        "task": task.name,
                        "runner": task.runner,
                        **result,
                    }
                )
                print(json.dumps(results["results"][-1], default=str), flush=True)

    results["primary_metric_summary"] = summarize_primary_metrics(results["results"])
    results["primary_metric_comparison"] = compare_primary_metrics(
        results["primary_metric_summary"],
        args.baseline_label,
    )
    failures = quality_drop_failures(
        results["primary_metric_comparison"],
        args.max_primary_metric_drop,
    )
    results["quality_gate"] = {
        "max_primary_metric_drop": args.max_primary_metric_drop,
        "passed": not failures,
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, default=str) + "\n")
    if failures:
        raise RuntimeError(
            "quality metric drop exceeded threshold: " + "; ".join(failures)
        )


if __name__ == "__main__":
    main()
