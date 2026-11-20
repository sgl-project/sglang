"""Compare AR and linear UNO using the PR #37667 offline math workload."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

from benchmark.uno.math_data import BENCHMARKS
from benchmark.uno.run_math_eval import _model_ref

REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="Qwen/Qwen3-8B")
    parser.add_argument("--tokenizer-path")
    parser.add_argument("--revision")
    parser.add_argument("--uno-lora-path", required=True)
    parser.add_argument("--device", choices=("npu", "cuda"), default="npu")
    parser.add_argument("--base-gpu-id", type=int, default=0)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--benchmarks", nargs="+", choices=tuple(BENCHMARKS), default=["gsm8k"]
    )
    parser.add_argument("--concurrencies", nargs="+", type=int, default=[1, 64])
    parser.add_argument("--widths", nargs="+", type=int, default=[8])
    parser.add_argument(
        "--graph-modes", nargs="+", choices=("graph", "eager"), default=["graph"]
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Default to 64 random problems/dataset, one sample and 100 output tokens. Explicit options override these defaults.",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--selection", choices=("first", "random"))
    parser.add_argument("--dataset-seed", type=int, default=42)
    parser.add_argument(
        "--num-samples", type=int, help="Default: 10 for AIME, 1 otherwise."
    )
    parser.add_argument("--context-length", type=int, default=40960)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--page-size", type=int)
    parser.add_argument("--mem-fraction-static", type=float)
    parser.add_argument("--warmup-requests", type=int, default=0)
    parser.add_argument("--warmup-max-tokens", type=int, default=32)
    parser.add_argument(
        "--run-timeout", type=float, help="Seconds per engine run; default: no timeout."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without loading models or datasets.",
    )
    args = parser.parse_args()
    if args.quick:
        if args.limit is None:
            args.limit = 64
        if args.num_samples is None:
            args.num_samples = 1
    if args.max_tokens is None:
        args.max_tokens = 100 if args.quick else 32768
    if args.selection is None:
        args.selection = "random" if args.quick else "first"
    for name in (
        "repeats",
        "context_length",
        "max_tokens",
        "top_k",
        "warmup_max_tokens",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    for name in ("limit", "num_samples", "page_size", "run_timeout"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    for name in ("widths", "concurrencies"):
        values = getattr(args, name)
        if any(value <= 0 for value in values):
            parser.error(f"--{name} must contain positive integers")
    for name in ("widths", "concurrencies", "benchmarks", "graph_modes"):
        values = getattr(args, name)
        if len(set(values)) != len(values):
            parser.error(f"--{name.replace('_', '-')} must not contain duplicates")
    if args.base_gpu_id < 0 or args.warmup_requests < 0:
        parser.error("--base-gpu-id and --warmup-requests must be non-negative")
    if args.temperature < 0 or not 0 < args.top_p <= 1:
        parser.error("Require temperature >= 0 and 0 < top-p <= 1")
    if args.mem_fraction_static is not None and not 0 < args.mem_fraction_static < 1:
        parser.error("--mem-fraction-static must be in (0, 1)")
    if not args.dry_run and args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error(
            "--output-dir must be new or empty; previous results are not overwritten"
        )
    return args


def build_jobs(args: argparse.Namespace) -> list[dict]:
    jobs = []
    for benchmark in args.benchmarks:
        samples = args.num_samples or (10 if benchmark.startswith("aime") else 1)
        available = (
            min(BENCHMARKS[benchmark].expected_rows, args.limit or sys.maxsize)
            * samples
        )
        for concurrency in args.concurrencies:
            if available < concurrency:
                print(
                    f"Note: {benchmark} has {available} completions, below concurrency {concurrency}.",
                    file=sys.stderr,
                )
            for graph in args.graph_modes:
                for repeat in range(1, args.repeats + 1):
                    variants = [("ar", None)] + [
                        ("uno-linear", width) for width in args.widths
                    ]
                    if repeat % 2 == 0:
                        variants.reverse()
                    for mode, width in variants:
                        label = "ar" if width is None else f"uno-f{width}"
                        directory = (
                            args.output_dir.resolve()
                            / f"{benchmark}-c{concurrency}-{graph}-r{repeat}-{label}"
                        )
                        command = [sys.executable, "-m", "benchmark.uno.run_math_eval"]
                        options = {
                            "model-path": _model_ref(args.model_path),
                            "tokenizer-path": _model_ref(args.tokenizer_path)
                            if args.tokenizer_path
                            else None,
                            "revision": args.revision,
                            "device": args.device,
                            "base-gpu-id": args.base_gpu_id,
                            "dtype": args.dtype,
                            "attention-backend": "ascend"
                            if args.device == "npu"
                            else "fa3",
                            "data-root": args.data_root.resolve(),
                            "output-dir": directory,
                            "benchmark": benchmark,
                            "num-samples": samples,
                            "max-running-requests": concurrency,
                            "context-length": args.context_length,
                            "max-tokens": args.max_tokens,
                            "temperature": args.temperature,
                            "top-k": args.top_k,
                            "top-p": args.top_p,
                            "random-seed": args.random_seed,
                            "page-size": args.page_size,
                            "mem-fraction-static": args.mem_fraction_static,
                            "limit": args.limit,
                            "selection": args.selection,
                            "dataset-seed": args.dataset_seed,
                            "warmup-requests": args.warmup_requests,
                            "warmup-max-tokens": args.warmup_max_tokens,
                            "cuda-graph-max-bs-decode": concurrency,
                            "cuda-graph-backend-decode": "full"
                            if graph == "graph"
                            else "disabled",
                        }
                        for name, value in options.items():
                            if value is not None:
                                command.extend([f"--{name}", str(value)])
                        # Isolate decode graphs; both modes use eager prefill.
                        command.append("--disable-prefill-cuda-graph")
                        # UNO on NPU requires synchronous scheduling. Match AR.
                        if args.device == "npu":
                            command.append("--disable-overlap-schedule")
                        if width is not None:
                            command.extend(
                                [
                                    "--speculative-algorithm",
                                    "UNO",
                                    "--uno-lora-path",
                                    _model_ref(args.uno_lora_path),
                                    "--speculative-num-steps",
                                    "1",
                                    "--speculative-eagle-topk",
                                    "1",
                                    "--speculative-num-draft-tokens",
                                    str(width),
                                ]
                            )
                        jobs.append(
                            dict(
                                benchmark=benchmark,
                                concurrency=concurrency,
                                graph=graph,
                                repeat=repeat,
                                mode=mode,
                                width=width,
                                directory=str(directory),
                                command=command,
                            )
                        )
    return jobs


def _live_process_group_members(pgid: int) -> list[int]:
    # killpg(pgid, 0) also reports orphan zombies, which cannot hold a device
    # context and may remain until PID 1 reaps them. Inspect state explicitly.
    result = subprocess.run(
        ["ps", "-axo", "pid=,pgid=,stat="],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    )
    return [
        int(pid)
        for pid, group, state in (line.split() for line in result.stdout.splitlines())
        if int(group) == pgid and not state.startswith(("Z", "X"))
    ]


def _wait_process_group(process: subprocess.Popen, timeout: float) -> list[int]:
    deadline = time.monotonic() + timeout
    while True:
        process.poll()  # Reap the group leader, including on timeout/interrupt.
        members = _live_process_group_members(process.pid)
        if not members:
            process.wait(timeout=5)
            return []
        if time.monotonic() >= deadline:
            return members
        time.sleep(0.1)


def _stop_process_group(process: subprocess.Popen) -> None:
    # Waiting for the launcher alone does not wait for orphaned schedulers.
    # start_new_session=True confines these signals to this benchmark run.
    process.poll()
    if not _live_process_group_members(process.pid):
        process.wait(timeout=5)
        return
    print(f"Cleaning up engine process group {process.pid}", flush=True)
    for sig, timeout in ((signal.SIGTERM, 5), (signal.SIGKILL, 15)):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            process.wait(timeout=5)
            return
        members = _wait_process_group(process, timeout)
        if not members:
            return
    raise RuntimeError(
        f"Engine process group {process.pid} still has live processes {members}; "
        "refusing to start the next engine on the same device."
    )


def _log_tail(path: Path) -> str:
    with path.open(errors="replace") as handle:
        return "".join(deque(handle, maxlen=40))


def run_job(job: dict, timeout: float | None) -> dict:
    directory = Path(job["directory"])
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "command.json").write_text(json.dumps(job, indent=2) + "\n")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT / "python"), str(REPO_ROOT), environment.get("PYTHONPATH", "")]
    )
    log = directory / "engine.log"
    print(f"Running {directory.name}; log: {log}", flush=True)
    with log.open("w") as handle:
        process = subprocess.Popen(
            job["command"],
            cwd=REPO_ROOT,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"Engine timed out after {timeout}s; see {log}") from exc
        finally:
            # Also clean up on SUCCESS: the launcher can exit before its
            # scheduler children, which otherwise retain the NPU context.
            _stop_process_group(process)
    if code:
        raise RuntimeError(
            f"Engine exited with code {code}; see {log}\n"
            f"Last 40 log lines:\n{_log_tail(log)}"
        )
    summary = json.loads((directory / "summary.json").read_text())
    tokens, forwards = summary["num_tokens"], summary["num_forwards"]
    if (
        summary["num_completions"] <= 0
        or tokens <= 0
        or summary["generation_seconds"] <= 0
    ):
        raise RuntimeError(f"Empty or invalid benchmark result in {directory}")
    if job["mode"] == "uno-linear" and forwards <= 0:
        raise RuntimeError(
            f"No UNO verification occurred in {directory}; increase output length"
        )
    return {
        **{
            key: job[key]
            for key in ("benchmark", "concurrency", "graph", "repeat", "mode", "width")
        },
        "accuracy": summary["accuracy"],
        "num_prompts": summary["num_prompts"],
        "num_completions": summary["num_completions"],
        "dataset_selection": summary["dataset_selection"],
        "max_tokens": summary["max_tokens"],
        "generation_seconds": summary["generation_seconds"],
        "output_tokens": tokens,
        "output_tok_s": summary["tokens_per_second"],
        "output_tok_s_per_configured_request": summary["tokens_per_second_per_request"],
        "tpf": summary["tokens_per_forward"],
        "length_capped_fraction": summary["num_length_capped"]
        / summary["num_completions"],
        "directory": str(directory),
    }


def write_comparison(root: Path, results: list[dict]) -> None:
    def key(row):
        return tuple(
            row[name] for name in ("benchmark", "concurrency", "graph", "repeat")
        )

    baselines = {key(row): row for row in results if row["mode"] == "ar"}
    rows = []
    for result in results:
        baseline = baselines.get(key(result))
        rows.append(
            {
                **result,
                "speedup_vs_ar": result["output_tok_s"] / baseline["output_tok_s"]
                if baseline
                else None,
                "accuracy_delta_vs_ar": result["accuracy"] - baseline["accuracy"]
                if baseline
                else None,
            }
        )
    (root / "comparison.json").write_text(json.dumps(rows, indent=2) + "\n")
    with (root / "comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "| Dataset | C | Graph | Repeat | Mode | F | Accuracy | tok/s | TPF | Speedup | Capped |",
        "| --- | ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        tpf = f"{row['tpf']:.3f}" if row["tpf"] is not None else "—"
        speedup = (
            f"{row['speedup_vs_ar']:.3f}x" if row["speedup_vs_ar"] is not None else "—"
        )
        lines.append(
            f"| {row['benchmark']} | {row['concurrency']} | {row['graph']} | {row['repeat']} | "
            f"{row['mode']} | {row['width'] or '—'} | {row['accuracy']:.2%} | "
            f"{row['output_tok_s']:.2f} | {tpf} | {speedup} | {row['length_capped_fraction']:.2%} |"
        )
    (root / "comparison.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    jobs = build_jobs(args)
    print(
        f"Plan: {len(jobs)} engine runs; limit={args.limit or 'all'}, "
        f"selection={args.selection}, dataset_seed={args.dataset_seed}, "
        f"samples={args.num_samples or 'dataset default'}, max_tokens={args.max_tokens}.",
        file=sys.stderr,
    )
    if args.quick:
        print(
            "Quick run: subset accuracy and length-capped throughput are exploratory; "
            "model loading and graph capture still occur for every engine run.",
            file=sys.stderr,
        )
    if args.dry_run:
        for job in jobs:
            print(shlex.join(job["command"]))
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "plan.json").write_text(json.dumps(jobs, indent=2) + "\n")
    results = []
    for job in jobs:
        results.append(run_job(job, args.run_timeout))
        write_comparison(args.output_dir, results)
    print((args.output_dir / "comparison.md").read_text())


if __name__ == "__main__":
    main()
