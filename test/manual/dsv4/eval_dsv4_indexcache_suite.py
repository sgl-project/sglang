"""Run paper-relevant DSV4 IndexCache quality evals against one or more endpoints."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import requests

FORBIDDEN_TASKS = {"mmlu", "gsm8k"}


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
    out_dir = args.out_dir
    if getattr(args, "endpoint_label", None) and getattr(args, "repeat_index", None):
        out_dir = (
            out_dir / args.endpoint_label / task.name / f"repeat_{args.repeat_index}"
        )
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


def build_eval_cmd(task: EvalTask, base_url: str, args) -> list[str]:
    if task.runner == "sglang.test.run_eval":
        return build_sglang_eval_cmd(task, base_url, args)
    if task.runner == "sgl-eval":
        return build_sgl_eval_cmd(task, base_url, args)
    raise ValueError(f"unknown runner {task.runner}")


def _as_int(value) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def speculative_config_paths(value, path: str = "") -> list[str]:
    if isinstance(value, dict):
        paths = []
        for key, item in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            if key == "speculative_algorithm" and item:
                paths.append(f"{child_path}={item}")
            elif key in {
                "speculative_num_steps",
                "speculative_eagle_topk",
                "speculative_num_draft_tokens",
            }:
                int_value = _as_int(item)
                if int_value is not None and int_value > 0:
                    paths.append(f"{child_path}={item}")
            elif key == "enable_multi_layer_eagle" and item:
                paths.append(f"{child_path}={item}")
            paths.extend(speculative_config_paths(item, child_path))
        return paths
    if isinstance(value, list):
        paths = []
        for i, item in enumerate(value):
            child_path = f"{path}[{i}]" if path else f"[{i}]"
            paths.extend(speculative_config_paths(item, child_path))
        return paths
    return []


def validate_server_info_for_base_path(server_info: dict, label: str) -> None:
    speculative_paths = speculative_config_paths(server_info)
    if speculative_paths:
        raise RuntimeError(
            f"quality eval endpoint {label!r} reports speculative decoding enabled; "
            "disable EAGLE/spec decode for base-path IndexCache validation: "
            + ", ".join(speculative_paths)
        )


def validate_endpoint_for_base_path(label: str, base_url: str, timeout: int) -> dict:
    response = requests.get(base_url.rstrip("/") + "/server_info", timeout=timeout)
    response.raise_for_status()
    server_info = response.json()
    validate_server_info_for_base_path(server_info, label)
    return server_info


def run_eval_task(task: EvalTask, base_url: str, args) -> dict:
    cmd = build_eval_cmd(task, base_url, args)
    if args.dry_run:
        return {"cmd": cmd, "returncode": 0, "elapsed_sec": 0, "output": ""}
    if task.runner == "sgl-eval" and shutil.which(args.sgl_eval_bin) is None:
        raise RuntimeError(f"{args.sgl_eval_bin!r} not found on PATH")
    return run_command(cmd, args.timeout)


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
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.repeats < 1:
        raise SystemExit("--repeats must be at least 1")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    tasks = selected_tasks(args.task, args.suite or ["long-context", "reasoning"])
    results = {
        "tasks": [task.name for task in tasks],
        "endpoints": [label for label, _ in args.endpoint],
        "server_checks": {},
        "results": [],
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

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, default=str) + "\n")


if __name__ == "__main__":
    main()
