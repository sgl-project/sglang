"""Orchestrate the DSV4 IndexCache endpoint validation workflow."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def script_path(name: str) -> str:
    return str(Path(__file__).with_name(name))


def run_command(cmd: list[str], timeout: int, dry_run: bool) -> dict:
    if dry_run:
        return {"cmd": cmd, "returncode": 0, "elapsed_sec": 0.0, "output": ""}
    start = time.perf_counter()
    proc = subprocess.run(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
    )
    return {
        "cmd": cmd,
        "returncode": proc.returncode,
        "elapsed_sec": time.perf_counter() - start,
        "output": proc.stdout,
    }


def profile_cmd(args) -> list[str]:
    cmd = [
        sys.executable,
        script_path("profile_dsv4_indexcache_endpoint.py"),
        "--endpoint",
        args.indexcache_endpoint,
        "--profile-dir",
        str(args.profile_dir),
        "--profile-prefix",
        args.profile_prefix,
        "--profile-steps",
        str(args.profile_steps),
        "--prompt-tokens",
        str(args.profile_prompt_tokens),
        "--min-indexcache-prompt-tokens",
        str(args.min_indexcache_prompt_tokens),
        "--max-tokens",
        str(args.profile_max_tokens),
        "--output",
        str(args.output_dir / "profile.json"),
    ]
    if args.eagle_off_confirmed:
        cmd.append("--eagle-off-confirmed")
    if args.indexcache_profile_env_confirmed:
        cmd.append("--indexcache-profile-env-confirmed")
    return cmd


def search_cmd(args) -> list[str]:
    cmd = [
        sys.executable,
        script_path("search_dsv4_indexcache_pattern.py"),
        "--calibration-jsonl",
        str(args.calibration_jsonl),
        "--endpoint",
        args.search_endpoint,
        "--num-c4-layers",
        str(args.num_c4_layers),
        "--pp-block-c4-layers",
        str(args.pp_block_c4_layers),
        "--retention",
        "1/2",
        "--retention",
        "1/4",
        "--output",
        str(args.output_dir / "searched_patterns.json"),
        "--command-template",
        args.pattern_command_template,
        "--min-indexcache-prompt-tokens",
        str(args.min_indexcache_prompt_tokens),
    ]
    if args.calibration_limit > 0:
        cmd += ["--limit", str(args.calibration_limit)]
    return cmd


def quality_eval_endpoints(args) -> list[tuple[str, str]]:
    searched_half_endpoint = args.searched_half_endpoint or args.indexcache_endpoint
    searched_quarter_endpoint = (
        args.searched_quarter_endpoint or args.indexcache_endpoint
    )
    return [
        ("baseline", args.baseline_endpoint),
        ("searched_1_2", searched_half_endpoint),
        ("searched_1_4", searched_quarter_endpoint),
    ]


def eval_cmd(args) -> list[str]:
    cmd = [
        sys.executable,
        script_path("eval_dsv4_indexcache_suite.py"),
        "--suite",
        "long-context",
        "--suite",
        "reasoning",
        "--num-threads",
        str(args.eval_num_threads),
        "--repeats",
        str(args.eval_repeats),
        "--max-tokens",
        str(args.eval_max_tokens),
        "--min-context-length",
        str(args.eval_min_context_length),
        "--output",
        str(args.output_dir / "quality_eval.json"),
    ]
    if args.require_eval_metrics:
        cmd.append("--require-metrics")
    for label, endpoint in quality_eval_endpoints(args):
        cmd += ["--endpoint", f"{label}={endpoint}"]
    return cmd


def validate_args(args) -> None:
    if not args.dry_run and not args.eagle_off_confirmed:
        raise SystemExit(
            "--eagle-off-confirmed is required for real validation runs; "
            "speculative decoding must be disabled until base path validation passes"
        )
    if not args.dry_run and not args.indexcache_profile_env_confirmed:
        raise SystemExit(
            "--indexcache-profile-env-confirmed is required for real validation runs; "
            "the IndexCache endpoint must set SGLANG_DSV4_INDEXCACHE_PROFILE=true"
        )
    if not args.dry_run and not args.searched_half_endpoint:
        raise SystemExit(
            "--searched-half-endpoint is required for real validation runs; "
            "quality eval must target the searched 1/2 pattern explicitly"
        )
    if not args.dry_run and not args.searched_quarter_endpoint:
        raise SystemExit(
            "--searched-quarter-endpoint is required for real validation runs; "
            "quality eval must target the searched 1/4 pattern explicitly"
        )
    if "{pattern}" not in args.pattern_command_template:
        raise SystemExit("--pattern-command-template must contain {pattern}")
    if (
        not args.dry_run
        and args.searched_half_endpoint
        and args.searched_quarter_endpoint
        and args.searched_half_endpoint.rstrip("/")
        == args.searched_quarter_endpoint.rstrip("/")
    ):
        raise SystemExit(
            "--searched-half-endpoint and --searched-quarter-endpoint must be "
            "distinct real endpoints"
        )
    if args.profile_prompt_tokens < args.min_indexcache_prompt_tokens:
        raise SystemExit(
            "--profile-prompt-tokens must be at least "
            "--min-indexcache-prompt-tokens for DSV4 IndexCache validation"
        )
    if args.eval_min_context_length < args.min_indexcache_prompt_tokens:
        raise SystemExit(
            "--eval-min-context-length must be at least "
            "--min-indexcache-prompt-tokens for DSV4 IndexCache validation"
        )


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-endpoint", required=True)
    parser.add_argument("--indexcache-endpoint", required=True)
    parser.add_argument("--searched-half-endpoint")
    parser.add_argument("--searched-quarter-endpoint")
    parser.add_argument("--search-endpoint", required=True)
    parser.add_argument("--calibration-jsonl", type=Path, required=True)
    parser.add_argument("--num-c4-layers", type=int, required=True)
    parser.add_argument("--pp-block-c4-layers", type=int, default=0)
    parser.add_argument("--pattern-command-template", required=True)
    parser.add_argument("--calibration-limit", type=int, default=0)
    parser.add_argument("--profile-dir", type=Path, default=Path("/tmp"))
    parser.add_argument("--profile-prefix", default="dsv4-indexcache")
    parser.add_argument("--profile-steps", type=int, default=4)
    parser.add_argument("--profile-prompt-tokens", type=int, default=128000)
    parser.add_argument("--profile-max-tokens", type=int, default=256)
    parser.add_argument("--min-indexcache-prompt-tokens", type=int, default=75000)
    parser.add_argument("--eval-num-threads", type=int, default=64)
    parser.add_argument("--eval-repeats", type=int, default=3)
    parser.add_argument("--eval-max-tokens", type=int, default=32768)
    parser.add_argument("--eval-min-context-length", type=int, default=75000)
    parser.add_argument(
        "--require-eval-metrics",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--timeout", type=int, default=86400)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--eagle-off-confirmed", action="store_true")
    parser.add_argument("--indexcache-profile-env-confirmed", action="store_true")
    args = parser.parse_args(argv)
    validate_args(args)
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    commands = [
        ("profile", profile_cmd(args)),
        ("search", search_cmd(args)),
        ("eval", eval_cmd(args)),
    ]
    results = []
    for phase, cmd in commands:
        result = run_command(cmd, args.timeout, args.dry_run)
        result = {"phase": phase, **result}
        results.append(result)
        print(json.dumps(result), flush=True)
        if result["returncode"] != 0:
            break

    summary = {
        "eagle": (
            "confirmed off"
            if args.eagle_off_confirmed
            else "dry run; speculative decoding not exercised"
        ),
        "uniform_1_4": "not run; only searched 1/4 is generated",
        "quality_eval_endpoints": {
            label: endpoint for label, endpoint in quality_eval_endpoints(args)
        },
        "indexcache_profile_env": (
            "confirmed SGLANG_DSV4_INDEXCACHE_PROFILE=true"
            if args.indexcache_profile_env_confirmed
            else "dry run; profiler marker env not exercised"
        ),
        "context_gate": {
            "min_indexcache_prompt_tokens": args.min_indexcache_prompt_tokens,
            "profile_prompt_tokens": args.profile_prompt_tokens,
            "eval_min_context_length": args.eval_min_context_length,
        },
        "quality_eval_repeats": args.eval_repeats,
        "phases": results,
    }
    (args.output_dir / "validation_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
