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
    return [
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
        "--max-tokens",
        str(args.profile_max_tokens),
        "--output",
        str(args.output_dir / "profile.json"),
    ]


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
    ]
    if args.pattern_command_template:
        cmd += ["--command-template", args.pattern_command_template]
    if args.calibration_limit > 0:
        cmd += ["--limit", str(args.calibration_limit)]
    return cmd


def eval_cmd(args) -> list[str]:
    return [
        sys.executable,
        script_path("eval_dsv4_indexcache_suite.py"),
        "--endpoint",
        f"baseline={args.baseline_endpoint}",
        "--endpoint",
        f"indexcache={args.indexcache_endpoint}",
        "--suite",
        "long-context",
        "--suite",
        "reasoning",
        "--num-threads",
        str(args.eval_num_threads),
        "--max-tokens",
        str(args.eval_max_tokens),
        "--output",
        str(args.output_dir / "quality_eval.json"),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-endpoint", required=True)
    parser.add_argument("--indexcache-endpoint", required=True)
    parser.add_argument("--search-endpoint", required=True)
    parser.add_argument("--calibration-jsonl", type=Path, required=True)
    parser.add_argument("--num-c4-layers", type=int, required=True)
    parser.add_argument("--pp-block-c4-layers", type=int, default=0)
    parser.add_argument("--pattern-command-template")
    parser.add_argument("--calibration-limit", type=int, default=0)
    parser.add_argument("--profile-dir", type=Path, default=Path("/tmp"))
    parser.add_argument("--profile-prefix", default="dsv4-indexcache")
    parser.add_argument("--profile-steps", type=int, default=4)
    parser.add_argument("--profile-prompt-tokens", type=int, default=128000)
    parser.add_argument("--profile-max-tokens", type=int, default=256)
    parser.add_argument("--eval-num-threads", type=int, default=64)
    parser.add_argument("--eval-max-tokens", type=int, default=32768)
    parser.add_argument("--timeout", type=int, default=86400)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

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
        "eagle": "off only; do not use this workflow with speculative decoding enabled",
        "uniform_1_4": "not run; only searched 1/4 is generated",
        "phases": results,
    }
    (args.output_dir / "validation_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
