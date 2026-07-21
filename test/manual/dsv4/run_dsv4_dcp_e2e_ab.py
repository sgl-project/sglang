#!/usr/bin/env python3
"""Run the fixed-route DeepSeek-V4 DCP decode-only E2E matrix."""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

DEFAULT_MODEL = "/data/models/DeepSeek-V4-Flash-FP8"
SUMMARY_FIELDS = [
    "variant",
    "context",
    "concurrency",
    "repeat",
    "num_prompts",
    "completed",
    "cache_hit_rate",
    "median_tpot_ms",
    "output_throughput",
    "total_throughput",
    "median_ttft_ms",
    "duration_s",
    "valid",
    "failure_reason",
    "result_file",
    "log_file",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--served-model-name", default="deepseek-v4-flash")
    parser.add_argument("--tokenizer", default=DEFAULT_MODEL)
    parser.add_argument("--contexts", type=int, nargs="+", default=[3500, 16000, 64000])
    parser.add_argument(
        "--concurrencies", type=int, nargs="+", default=[8, 32, 64, 128]
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output-len", type=int, default=512)
    parser.add_argument("--warmup-requests", type=int, default=10)
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def build_command(
    args: argparse.Namespace,
    *,
    context: int,
    concurrency: int,
    result_file: Path,
) -> list[str]:
    num_prompts = 2 * concurrency
    return [
        sys.executable,
        "-m",
        "sglang.bench_serving",
        "--backend",
        "sglang",
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--dataset-name",
        "generated-shared-prefix",
        "--model",
        args.model,
        "--served-model-name",
        args.served_model_name,
        "--tokenizer",
        args.tokenizer,
        "--num-prompts",
        str(num_prompts),
        "--request-rate",
        "inf",
        "--max-concurrency",
        str(concurrency),
        "--output-file",
        str(result_file),
        "--output-details",
        "--disable-tqdm",
        "--cache-report",
        "--seed",
        str(args.seed),
        "--temperature",
        "0",
        "--warmup-requests",
        str(args.warmup_requests),
        "--gsp-num-groups",
        "1",
        "--gsp-prompts-per-group",
        str(num_prompts),
        "--gsp-system-prompt-len",
        str(context),
        "--gsp-question-len",
        "0",
        "--gsp-output-len",
        str(args.output_len),
    ]


def read_last_json(path: Path) -> dict[str, Any]:
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"empty benchmark result: {path}")
    return json.loads(lines[-1])


def validate_result(
    result: dict[str, Any], *, num_prompts: int, output_len: int
) -> tuple[bool, str, float]:
    reasons = []
    completed = int(result.get("completed", -1))
    if completed != num_prompts:
        reasons.append(f"completed={completed}, expected={num_prompts}")

    errors = result.get("errors", [])
    nonempty_errors = [error for error in errors if error]
    if nonempty_errors:
        reasons.append(f"request_errors={len(nonempty_errors)}")

    output_lens = result.get("output_lens", [])
    if len(output_lens) != num_prompts or any(
        int(length) != output_len for length in output_lens
    ):
        reasons.append("output lengths do not all match the requested length")

    if output_len > 0:
        generated_texts = result.get("generated_texts", [])
        if len(generated_texts) != num_prompts or any(
            not str(text).strip() for text in generated_texts
        ):
            reasons.append("generated texts are missing or empty")

        ttfts = result.get("ttfts", [])
        if len(ttfts) != num_prompts or any(float(ttft) <= 0 for ttft in ttfts):
            reasons.append("TTFT values are missing or non-positive")

    input_lens = result.get("input_lens", [])
    cached_tokens = result.get("cached_tokens", [])
    total_input = sum(int(length) for length in input_lens)
    cache_hit_rate = (
        sum(int(length) for length in cached_tokens) / total_input
        if total_input
        else 0.0
    )
    return not reasons, "; ".join(reasons), cache_hit_rate


def make_summary_row(
    args: argparse.Namespace,
    *,
    context: int,
    concurrency: int,
    repeat: int,
    result_file: Path,
    log_file: Path,
    result: dict[str, Any] | None,
    process_error: str = "",
) -> dict[str, Any]:
    num_prompts = 2 * concurrency
    if result is None:
        valid, failure_reason, cache_hit_rate = False, process_error, 0.0
        result = {}
    else:
        valid, failure_reason, cache_hit_rate = validate_result(
            result, num_prompts=num_prompts, output_len=args.output_len
        )
        if process_error:
            valid = False
            failure_reason = "; ".join(
                reason for reason in (process_error, failure_reason) if reason
            )

    return {
        "variant": args.variant,
        "context": context,
        "concurrency": concurrency,
        "repeat": repeat,
        "num_prompts": num_prompts,
        "completed": int(result.get("completed", 0)),
        "cache_hit_rate": cache_hit_rate,
        "median_tpot_ms": result.get("median_tpot_ms"),
        "output_throughput": result.get("output_throughput"),
        "total_throughput": result.get("total_throughput"),
        "median_ttft_ms": result.get("median_ttft_ms"),
        "duration_s": result.get("duration"),
        "valid": valid,
        "failure_reason": failure_reason,
        "result_file": str(result_file),
        "log_file": str(log_file),
    }


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    with path.open("a") as file:
        file.write(json.dumps(value, sort_keys=True) + "\n")


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    if args.repeats < 1:
        raise ValueError("--repeats must be positive")
    if any(value <= 0 for value in args.contexts + args.concurrencies):
        raise ValueError("contexts and concurrencies must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    commands_file = args.output_dir / f"{args.variant}_commands.jsonl"
    summary_jsonl = args.output_dir / f"{args.variant}_summary.jsonl"
    summary_tsv = args.output_dir / f"{args.variant}_summary.tsv"
    rows = []

    for context in args.contexts:
        for concurrency in args.concurrencies:
            for repeat in range(1, args.repeats + 1):
                stem = f"{args.variant}_ctx{context}_c{concurrency}_r{repeat}"
                result_file = args.output_dir / f"{stem}.jsonl"
                log_file = args.output_dir / f"{stem}.log"
                command = build_command(
                    args,
                    context=context,
                    concurrency=concurrency,
                    result_file=result_file,
                )
                append_jsonl(
                    commands_file,
                    {
                        "variant": args.variant,
                        "context": context,
                        "concurrency": concurrency,
                        "repeat": repeat,
                        "argv": command,
                        "shell": shlex.join(command),
                    },
                )

                if args.dry_run:
                    print(shlex.join(command))
                    continue

                if args.resume and result_file.exists():
                    result = read_last_json(result_file)
                    row = make_summary_row(
                        args,
                        context=context,
                        concurrency=concurrency,
                        repeat=repeat,
                        result_file=result_file,
                        log_file=log_file,
                        result=result,
                    )
                else:
                    if result_file.exists():
                        raise FileExistsError(
                            f"result already exists (use --resume): {result_file}"
                        )
                    print(
                        f"[{args.variant}] context={context} "
                        f"concurrency={concurrency} repeat={repeat}",
                        flush=True,
                    )
                    with log_file.open("w") as log:
                        log.write(f"Command: {shlex.join(command)}\n")
                        log.write("=" * 80 + "\n")
                        log.flush()
                        completed = subprocess.run(
                            command,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            check=False,
                        )
                    process_error = (
                        ""
                        if completed.returncode == 0
                        else f"benchmark exited with {completed.returncode}"
                    )
                    try:
                        result = read_last_json(result_file)
                    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
                        result = None
                        process_error = "; ".join(
                            reason for reason in (process_error, str(exc)) if reason
                        )
                    row = make_summary_row(
                        args,
                        context=context,
                        concurrency=concurrency,
                        repeat=repeat,
                        result_file=result_file,
                        log_file=log_file,
                        result=result,
                        process_error=process_error,
                    )

                rows.append(row)
                append_jsonl(summary_jsonl, row)
                write_tsv(summary_tsv, rows)
                print(
                    f"  valid={row['valid']} tpot={row['median_tpot_ms']} "
                    f"output_tok_s={row['output_throughput']} "
                    f"cache_hit={row['cache_hit_rate']:.4f}",
                    flush=True,
                )
                if not row["valid"] and not args.keep_going:
                    return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
