"""Run paper-relevant DSV4 IndexCache quality evals against one or more endpoints."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


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
        str(args.out_dir),
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


def main() -> None:
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
    args = parser.parse_args()

    tasks = selected_tasks(args.task, args.suite or ["long-context", "reasoning"])
    results = {
        "tasks": [task.name for task in tasks],
        "endpoints": [label for label, _ in args.endpoint],
        "results": [],
    }

    for label, base_url in args.endpoint:
        for task in tasks:
            result = run_eval_task(task, base_url, args)
            results["results"].append(
                {"endpoint": label, "task": task.name, "runner": task.runner, **result}
            )
            print(json.dumps(results["results"][-1], default=str), flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, default=str) + "\n")


if __name__ == "__main__":
    main()
