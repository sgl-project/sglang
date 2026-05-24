"""Collect and summarize DSV4 IndexCache profile traces from an existing endpoint."""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import time
from pathlib import Path

import requests


def _load_profile_analyzer():
    path = Path(__file__).with_name("analyze_dsv4_indexcache_profile.py")
    spec = importlib.util.spec_from_file_location(
        "analyze_dsv4_indexcache_profile", path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


summarize_trace = _load_profile_analyzer().summarize_trace

REQUIRED_PROFILE_CATEGORIES = (
    "csa_indexer",
    "raw_to_page_translation",
    "core_attention",
    "ffn_moe",
    "cuda_graph",
)


def make_prompt(token_target: int, seed: str) -> str:
    rng = random.Random(seed)
    words = [
        "attention",
        "kernel",
        "profile",
        "context",
        "index",
        "cache",
        "layer",
        "token",
    ]
    chunks = []
    for i in range(max(token_target // 7, 1)):
        chunks.append(f"{rng.choice(words)}-{i}-{rng.randint(0, 999999)}")
    return " ".join(chunks) + "\n\nSummarize the recurring technical theme."


def post_json(
    base_url: str, path: str, payload: dict, timeout: int
) -> requests.Response:
    return requests.post(base_url.rstrip("/") + path, json=payload, timeout=timeout)


def send_generate(base_url: str, prompt: str, max_tokens: int, timeout: int) -> dict:
    response = post_json(
        base_url,
        "/generate",
        {
            "text": prompt,
            "sampling_params": {"temperature": 0, "max_new_tokens": max_tokens},
        },
        timeout,
    )
    response.raise_for_status()
    return response.json().get("meta_info", {})


def start_profile(args) -> None:
    payload = {
        "output_dir": str(args.profile_dir),
        "num_steps": args.profile_steps,
        "activities": args.activities,
        "profile_by_stage": True,
        "profile_prefix": args.profile_prefix,
        "profile_stages": args.profile_stages,
    }
    response = post_json(args.endpoint, "/start_profile", payload, args.timeout)
    response.raise_for_status()


def collect_profile(args) -> list[dict]:
    prompt = make_prompt(args.prompt_tokens, args.seed)
    for i in range(args.warmup_requests):
        send_generate(args.endpoint, prompt, args.max_tokens, args.timeout)
        print(json.dumps({"phase": "warmup", "request": i}), flush=True)

    start_profile(args)
    results = []
    for i in range(args.profile_requests):
        meta_info = send_generate(args.endpoint, prompt, args.max_tokens, args.timeout)
        result = {"phase": "profile", "request": i, "meta_info": meta_info}
        results.append(result)
        print(json.dumps(result), flush=True)
    time.sleep(args.trace_flush_sleep)
    return results


def find_trace_files(profile_dir: Path, profile_prefix: str) -> list[Path]:
    patterns = [
        f"{profile_prefix}*.trace.json",
        f"{profile_prefix}*.trace.json.gz",
        f"*{profile_prefix}*.trace.json",
        f"*{profile_prefix}*.trace.json.gz",
    ]
    traces = []
    for pattern in patterns:
        traces.extend(profile_dir.glob(pattern))
    return sorted(set(traces))


def validate_args(args) -> None:
    if not args.dry_run and not args.eagle_off_confirmed:
        raise SystemExit(
            "--eagle-off-confirmed is required for real profile runs; "
            "speculative decoding must be disabled for base-path validation"
        )
    if not args.dry_run and not args.indexcache_profile_env_confirmed:
        raise SystemExit(
            "--indexcache-profile-env-confirmed is required for real profile runs; "
            "the server must run with SGLANG_DSV4_INDEXCACHE_PROFILE=true"
        )
    if args.prompt_tokens < args.min_indexcache_prompt_tokens:
        raise SystemExit(
            "--prompt-tokens must be at least --min-indexcache-prompt-tokens "
            "for DSV4 IndexCache validation"
        )


def profile_dir_note(profile_dir: Path) -> str:
    return (
        f"{profile_dir} must be readable by this driver after /start_profile returns. "
        "For remote endpoints, run the driver in the same container or mount/copy the "
        "server profile directory before analysis."
    )


def has_category(categories: dict, required: str) -> bool:
    if required == "core_attention":
        return any(category.startswith("core_attention") for category in categories)
    return required in categories


def validate_trace_summaries(args, trace_summaries: list[dict]) -> None:
    if args.dry_run:
        return
    if not trace_summaries:
        raise RuntimeError(
            "no profile traces found; check --profile-dir, --profile-prefix, "
            "server /start_profile support, and that the server-side profile "
            "directory is visible to this driver"
        )
    observed = {}
    for summary in trace_summaries:
        observed.update(summary.get("categories", {}))
    missing = [
        category
        for category in REQUIRED_PROFILE_CATEGORIES
        if not has_category(observed, category)
    ]
    if missing:
        raise RuntimeError(
            "profile traces are missing required DSV4 IndexCache regions "
            f"{missing}; check SGLANG_DSV4_INDEXCACHE_PROFILE=true and that "
            "IndexCache reuse was exercised"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument(
        "--profile-dir",
        type=Path,
        default=Path("/tmp"),
        help=(
            "Directory passed to server /start_profile and then read by this "
            "driver. For remote endpoints, this path must be shared or copied "
            "back before analysis."
        ),
    )
    parser.add_argument("--profile-prefix", default="dsv4-indexcache")
    parser.add_argument("--profile-stages", nargs="+", default=["prefill", "decode"])
    parser.add_argument("--activities", nargs="+", default=["CPU", "GPU"])
    parser.add_argument("--profile-steps", type=int, default=4)
    parser.add_argument("--warmup-requests", type=int, default=1)
    parser.add_argument("--profile-requests", type=int, default=4)
    parser.add_argument("--prompt-tokens", type=int, default=128000)
    parser.add_argument("--min-indexcache-prompt-tokens", type=int, default=75000)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--trace-flush-sleep", type=float, default=5.0)
    parser.add_argument("--seed", default="dsv4-indexcache-profile")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--eagle-off-confirmed", action="store_true")
    parser.add_argument("--indexcache-profile-env-confirmed", action="store_true")
    args = parser.parse_args()
    validate_args(args)

    args.profile_dir.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        request_results = []
    else:
        request_results = collect_profile(args)

    trace_files = find_trace_files(args.profile_dir, args.profile_prefix)
    trace_summaries = [summarize_trace(path) for path in trace_files]
    validate_trace_summaries(args, trace_summaries)

    result = {
        "endpoint": args.endpoint,
        "profile_dir": str(args.profile_dir),
        "profile_dir_note": profile_dir_note(args.profile_dir),
        "profile_prefix": args.profile_prefix,
        "min_indexcache_prompt_tokens": args.min_indexcache_prompt_tokens,
        "request_results": request_results,
        "trace_files": [str(path) for path in trace_files],
        "trace_summaries": trace_summaries,
        "indexcache_profile_env": (
            "confirmed SGLANG_DSV4_INDEXCACHE_PROFILE=true"
            if args.indexcache_profile_env_confirmed
            else "dry run; profiler marker env not exercised"
        ),
        "eagle": (
            "confirmed off"
            if args.eagle_off_confirmed
            else "dry run; speculative decoding not exercised"
        ),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
