"""Run a resumable matrix of prefix-cache serving benchmarks."""

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

SGLANG_REPO_ROOT = Path(__file__).resolve().parents[2]
SGLANG_PYTHON_ROOT = SGLANG_REPO_ROOT / "python"


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"expected an integer > 0, got {value!r}")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError(f"expected an integer >= 0, got {value!r}")
    return parsed


def _cache_hit_percent(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0 or parsed > 100:
        raise argparse.ArgumentTypeError(
            f"expected a finite percentage in [0, 100], got {value!r}"
        )
    return parsed


def _format_number(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def make_tag(
    tag_prefix: str,
    input_len: int,
    output_len: int,
    cache_hit_percent: float,
    concurrency: int,
    repetition: int | None = None,
) -> str:
    tag = (
        f"{tag_prefix}-in{input_len}-out{output_len}-"
        f"hit{_format_number(cache_hit_percent)}-c{concurrency}"
    )
    return f"{tag}-r{repetition}" if repetition is not None else tag


def result_validation_error(
    row: dict,
    expected_requests: int,
    target_hit_rate_pct: float,
    cache_hit_tolerance: float,
) -> str | None:
    if row.get("completed") != expected_requests:
        return (
            f"completed {row.get('completed')!r} requests; "
            f"expected {expected_requests}"
        )

    cache_report = row.get("cache_report") or {}
    actual_hit_rate_pct = cache_report.get("cache_hit_rate_pct")
    prefix_config = row.get("prefix_cache_config") or {}
    expected_hit_rate_pct = prefix_config.get(
        "expected_hit_rate_pct", target_hit_rate_pct
    )
    if not isinstance(actual_hit_rate_pct, (int, float)) or not math.isfinite(
        actual_hit_rate_pct
    ):
        return "cache report is missing a finite achieved hit rate"
    if not isinstance(expected_hit_rate_pct, (int, float)) or not math.isfinite(
        expected_hit_rate_pct
    ):
        return "prefix-cache configuration is missing a finite expected hit rate"

    allowed_error = cache_hit_tolerance_for_row(row, cache_hit_tolerance)
    error = abs(actual_hit_rate_pct - expected_hit_rate_pct)
    if error > allowed_error:
        return (
            f"cache-hit error is {error:.2f} percentage points "
            f"({expected_hit_rate_pct:.2f}% expected, "
            f"{actual_hit_rate_pct:.2f}% actual), exceeding "
            f"the {allowed_error:.2f}-point tolerance"
        )
    return None


def cache_hit_tolerance_for_row(row: dict, tolerance_floor: float) -> float:
    server_info = row.get("server_info") or {}
    page_size = server_info.get("page_size")
    total_input_tokens = row.get("total_input_tokens")
    completed = row.get("completed")
    if (
        isinstance(page_size, (int, float))
        and page_size > 0
        and isinstance(total_input_tokens, (int, float))
        and total_input_tokens > 0
        and isinstance(completed, int)
        and completed > 0
    ):
        average_prompt_tokens = total_input_tokens / completed
        page_rounding_error = 100 * page_size / average_prompt_tokens
        return max(tolerance_floor, page_rounding_error)
    return tolerance_floor


def load_completed_results(
    result_path: Path,
    expected_requests: int,
    target_hit_rates: Mapping[str, float],
    cache_hit_tolerance: float,
) -> dict[str, dict]:
    if not result_path.exists():
        return {}
    completed = {}
    for line in result_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        tag = row.get("tag")
        if tag not in target_hit_rates:
            continue
        if (
            result_validation_error(
                row,
                expected_requests,
                target_hit_rates[tag],
                cache_hit_tolerance,
            )
            is None
        ):
            completed[tag] = row
        else:
            completed.pop(tag, None)
    return completed


def build_point_command(
    args: argparse.Namespace,
    result_path: Path,
    input_len: int,
    output_len: int,
    cache_hit_percent: float,
    concurrency: int,
    repetition: int,
) -> tuple[str, list[str]]:
    prefix_len = int(input_len * cache_hit_percent / 100)
    question_len = input_len - prefix_len
    if cache_hit_percent == 0:
        num_groups = args.num_prompts
        prompts_per_group = 1
    else:
        num_groups = args.num_groups
        prompts_per_group = args.num_prompts // args.num_groups

    tag = make_tag(
        args.tag_prefix,
        input_len,
        output_len,
        cache_hit_percent,
        concurrency,
        repetition if args.repetitions > 1 else None,
    )
    command = [
        sys.executable,
        "-m",
        "sglang.benchmark.serving",
        "--backend",
        args.backend,
        "--base-url",
        args.base_url,
        "--model",
        args.model,
        "--tokenizer",
        args.tokenizer,
        "--dataset-name",
        "generated-shared-prefix",
        "--num-prompts",
        str(args.num_prompts),
        "--gsp-num-groups",
        str(num_groups),
        "--gsp-prompts-per-group",
        str(prompts_per_group),
        "--gsp-system-prompt-len",
        str(prefix_len),
        "--gsp-question-len",
        str(question_len),
        "--gsp-output-len",
        str(output_len),
        "--gsp-range-ratio",
        "1.0",
        "--gsp-group-distribution",
        "uniform" if cache_hit_percent == 0 else args.group_distribution,
        "--max-concurrency",
        str(concurrency),
        "--request-rate",
        str(args.request_rate),
        "--warmup-requests",
        str(args.warmup_requests),
        "--flush-cache",
        "--seed",
        str(args.seed + repetition - 1),
        "--cache-report",
        "--disable-tqdm",
        "--output-file",
        str(result_path),
        "--tag",
        tag,
    ]
    if cache_hit_percent > 0:
        command.extend(
            [
                "--gsp-prewarm-prefixes",
                "--gsp-prewarm-concurrency",
                str(args.prewarm_concurrency),
            ]
        )
    if cache_hit_percent > 0 and args.group_distribution == "zipf":
        command.extend(["--gsp-zipf-alpha", str(args.zipf_alpha)])
    if args.output_details:
        command.append("--output-details")
    if args.extra_request_body:
        command.extend(["--extra-request-body", args.extra_request_body])
    return tag, command


def write_summary(
    result_path: Path, summary_path: Path, cache_hit_tolerance: float
) -> None:
    if not result_path.exists():
        return
    rows_by_tag = {}
    for line in result_path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get("tag"):
                rows_by_tag[row["tag"]] = row

    fields = [
        "tag",
        "completed",
        "max_concurrency",
        "expected_hit_rate_pct",
        "actual_hit_rate_pct",
        "cache_hit_error_percentage_points",
        "allowed_cache_hit_error_percentage_points",
        "cache_hit_within_tolerance",
        "mean_ttft_ms",
        "median_ttft_ms",
        "p90_ttft_ms",
        "p99_ttft_ms",
        "mean_tpot_ms",
        "p99_tpot_ms",
        "request_throughput",
        "input_throughput",
        "output_throughput",
    ]
    with summary_path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for tag, row in sorted(rows_by_tag.items()):
            prefix = row.get("prefix_cache_config") or {}
            cache = row.get("cache_report") or {}
            expected_hit_rate = prefix.get("expected_hit_rate_pct", 0)
            actual_hit_rate = cache.get("cache_hit_rate_pct")
            cache_hit_error = (
                abs(actual_hit_rate - expected_hit_rate)
                if isinstance(actual_hit_rate, (int, float))
                and isinstance(expected_hit_rate, (int, float))
                else None
            )
            allowed_cache_hit_error = cache_hit_tolerance_for_row(
                row, cache_hit_tolerance
            )
            writer.writerow(
                {
                    "tag": tag,
                    "completed": row.get("completed"),
                    "max_concurrency": row.get("max_concurrency"),
                    "expected_hit_rate_pct": expected_hit_rate,
                    "actual_hit_rate_pct": actual_hit_rate,
                    "cache_hit_error_percentage_points": cache_hit_error,
                    "allowed_cache_hit_error_percentage_points": (
                        allowed_cache_hit_error
                    ),
                    "cache_hit_within_tolerance": (
                        cache_hit_error is not None
                        and cache_hit_error <= allowed_cache_hit_error
                    ),
                    "mean_ttft_ms": row.get("mean_ttft_ms"),
                    "median_ttft_ms": row.get("median_ttft_ms"),
                    "p90_ttft_ms": row.get("p90_ttft_ms"),
                    "p99_ttft_ms": row.get("p99_ttft_ms"),
                    "mean_tpot_ms": row.get("mean_tpot_ms"),
                    "p99_tpot_ms": row.get("p99_tpot_ms"),
                    "request_throughput": row.get("request_throughput"),
                    "input_throughput": row.get("input_throughput"),
                    "output_throughput": row.get("output_throughput"),
                }
            )


def write_manifest(args: argparse.Namespace, result_dir: Path) -> None:
    manifest: dict[str, Any] = {
        "model": args.model,
        "tokenizer": args.tokenizer,
        "backend": args.backend,
        "base_url": args.base_url,
        "input_lengths": args.input_lens,
        "output_lengths": args.output_lens,
        "cache_hit_percentages": args.cache_hit_percentages,
        "concurrencies": args.concurrencies,
        "num_prompts": args.num_prompts,
        "num_groups": args.num_groups,
        "warmup_requests": args.warmup_requests,
        "prewarm_concurrency": args.prewarm_concurrency,
        "request_rate": args.request_rate,
        "seed": args.seed,
        "repetitions": args.repetitions,
        "group_distribution": args.group_distribution,
        "zipf_alpha": args.zipf_alpha,
        "cache_hit_tolerance_floor_percentage_points": args.cache_hit_tolerance,
        "command": sys.argv,
    }
    try:
        manifest["sglang_revision"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=SGLANG_REPO_ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        manifest["sglang_revision"] = None
    (result_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def run_point(
    command: Sequence[str], tag: str, log_path: Path, quiet: bool = False
) -> None:
    print(f"\n===== START {tag} =====", flush=True)
    print(" ".join(command), flush=True)
    started = time.monotonic()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SGLANG_PYTHON_ROOT) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    with log_path.open("w") as log_file:
        process = subprocess.Popen(
            command,
            cwd=SGLANG_REPO_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log_file.write(line)
            log_file.flush()
            if not quiet:
                print(line, end="", flush=True)
        return_code = process.wait()

    if return_code != 0:
        raise RuntimeError(f"{tag} failed with exit code {return_code}; see {log_path}")
    print(f"===== PASS {tag} ({time.monotonic() - started:.1f}s) =====", flush=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python3 benchmark/prefix_cache/bench_prefix_cache.py",
        description="Run a resumable prefix-cache benchmark matrix using bench_serving.",
    )
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument(
        "--backend",
        choices=("sglang", "sglang-native", "sglang-oai", "sglang-oai-chat"),
        default="sglang",
    )
    parser.add_argument("--input-lens", type=_positive_int, nargs="+", required=True)
    parser.add_argument("--output-lens", type=_positive_int, nargs="+", required=True)
    parser.add_argument(
        "--cache-hit-percentages",
        type=_cache_hit_percent,
        nargs="+",
        required=True,
    )
    parser.add_argument("--concurrencies", type=_positive_int, nargs="+", required=True)
    parser.add_argument("--num-prompts", type=_positive_int, default=50)
    parser.add_argument("--num-groups", type=_positive_int, default=2)
    parser.add_argument("--warmup-requests", type=_nonnegative_int, default=5)
    parser.add_argument("--prewarm-concurrency", type=_positive_int, default=1)
    parser.add_argument("--request-rate", type=float, default=float("inf"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repetitions", type=_positive_int, default=1)
    parser.add_argument(
        "--group-distribution", choices=("uniform", "zipf"), default="uniform"
    )
    parser.add_argument("--zipf-alpha", type=float)
    parser.add_argument(
        "--cache-hit-tolerance",
        type=_cache_hit_percent,
        default=0.5,
        help=(
            "Minimum allowed absolute expected-versus-actual cache-hit error "
            "in percentage points. The effective tolerance is the larger of "
            "this value and one server-reported cache page divided by the "
            "average prompt length (default: 0.5)."
        ),
    )
    parser.add_argument("--extra-request-body")
    parser.add_argument("--output-details", action="store_true")
    parser.add_argument("--tag-prefix", default="prefix-cache")
    parser.add_argument("--result-dir", type=Path, default=Path("prefix_cache_results"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Write per-point output to log files without echoing every line.",
    )
    args = parser.parse_args(argv)

    if any(hit > 0 for hit in args.cache_hit_percentages) and (
        args.num_prompts % args.num_groups != 0
    ):
        parser.error("--num-prompts must be divisible by --num-groups")
    if args.group_distribution == "zipf":
        if args.zipf_alpha is None or not math.isfinite(args.zipf_alpha):
            parser.error("--group-distribution=zipf requires finite --zipf-alpha > 0")
        if args.zipf_alpha <= 0:
            parser.error("--zipf-alpha must be > 0")
    elif args.zipf_alpha is not None:
        parser.error("--zipf-alpha requires --group-distribution=zipf")
    if not math.isfinite(args.request_rate) and args.request_rate != float("inf"):
        parser.error("--request-rate must be finite or positive infinity")
    if args.request_rate <= 0:
        parser.error("--request-rate must be > 0")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result_dir = args.result_dir.resolve()
    result_path = result_dir / "results.jsonl"
    summary_path = result_dir / "summary.csv"
    log_dir = result_dir / "logs"
    result_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(args, result_dir)

    commands = []
    for input_len in args.input_lens:
        for output_len in args.output_lens:
            for cache_hit_percent in args.cache_hit_percentages:
                for concurrency in args.concurrencies:
                    for repetition in range(1, args.repetitions + 1):
                        commands.append(
                            build_point_command(
                                args,
                                result_path,
                                input_len,
                                output_len,
                                cache_hit_percent,
                                concurrency,
                                repetition,
                            )
                        )

    target_hit_rates = {
        tag: cache_hit_percent
        for input_len in args.input_lens
        for output_len in args.output_lens
        for cache_hit_percent in args.cache_hit_percentages
        for concurrency in args.concurrencies
        for repetition in range(1, args.repetitions + 1)
        for tag in (
            make_tag(
                args.tag_prefix,
                input_len,
                output_len,
                cache_hit_percent,
                concurrency,
                repetition if args.repetitions > 1 else None,
            ),
        )
    }
    completed = load_completed_results(
        result_path,
        args.num_prompts,
        target_hit_rates,
        args.cache_hit_tolerance,
    )

    print(
        f"Matrix contains {len(commands)} points; {len(completed)} are already complete."
    )
    for tag, command in commands:
        if tag in completed:
            print(f"===== SKIP {tag} =====", flush=True)
            continue
        if args.dry_run:
            print(" ".join(command))
            continue
        run_point(command, tag, log_dir / f"{tag}.log", quiet=args.quiet)
        completed = load_completed_results(
            result_path,
            args.num_prompts,
            target_hit_rates,
            args.cache_hit_tolerance,
        )
        if tag not in completed:
            matching_rows = [
                row
                for line in result_path.read_text().splitlines()
                if line.strip()
                for row in (json.loads(line),)
                if row.get("tag") == tag
            ]
            reason = (
                result_validation_error(
                    matching_rows[-1],
                    args.num_prompts,
                    target_hit_rates[tag],
                    args.cache_hit_tolerance,
                )
                if matching_rows
                else "no result row was written"
            )
            raise RuntimeError(f"{tag} failed result validation: {reason}")
        write_summary(result_path, summary_path, args.cache_hit_tolerance)

    if not args.dry_run:
        write_summary(result_path, summary_path, args.cache_hit_tolerance)
        print(f"Completed all {len(commands)} matrix points. Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
