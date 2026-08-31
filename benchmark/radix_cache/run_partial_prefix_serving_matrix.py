"""Run a resumable official bench_serving matrix for partial-page KV reuse.

This driver starts an ordinary CUDA FULL-attention SGLang server, invokes
``bench_partial_prefix_serving.py`` for deterministic exact-R requests, and
validates every persisted result before moving to the next case.

The aligned reusable prefix is expressed as the *actual* page-aligned match
length.  For example, ``--aligned-match-tokens 4096 --page-size 32`` builds a
4064-token common prefix followed by a fully matching 32-token child page, so
the legacy aligned match is exactly 4096 rather than 4096 + page_size.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import requests

PARTIALS_BY_PAGE = {
    16: [1, 4, 8, 12, 15],
    32: [1, 8, 16, 24, 31],
    64: [1, 8, 16, 32, 48, 63],
}
SUPPORTED_CONCURRENCIES = (1, 8, 16)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--page-size", type=int, choices=(16, 32, 64), required=True)
    parser.add_argument(
        "--configs",
        nargs="+",
        choices=("baseline", "reuse"),
        default=["baseline", "reuse"],
    )
    parser.add_argument("--aligned-match-tokens", type=int, default=4096)
    parser.add_argument("--concurrencies", type=int, nargs="+", default=[1, 8, 16])
    parser.add_argument("--num-prompts-c1", type=int, default=50)
    parser.add_argument("--num-prompts-c8", type=int, default=80)
    parser.add_argument("--num-prompts-c16", type=int, default=80)
    parser.add_argument(
        "--sparse-c8",
        action="store_true",
        help="For C=8, measure only R=1, page_size/2, and page_size-1.",
    )
    parser.add_argument("--partial-lens", type=int, nargs="+")
    parser.add_argument("--output-len", type=int, default=1)
    parser.add_argument("--suffix-len", type=int)
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--cuda-visible-devices", default="0,1")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--mem-fraction-static", type=float, default=0.8)
    parser.add_argument("--server-timeout", type=float, default=1200)
    parser.add_argument("--request-timeout", type=float, default=900)
    parser.add_argument("--manual-warmups", type=int, default=3)
    parser.add_argument("--extra-server-args", nargs=argparse.REMAINDER, default=[])
    args = parser.parse_args()

    if args.aligned_match_tokens % args.page_size:
        parser.error("--aligned-match-tokens must be divisible by --page-size")
    if args.aligned_match_tokens <= args.page_size:
        parser.error("--aligned-match-tokens must exceed one page")
    if any(c not in SUPPORTED_CONCURRENCIES for c in args.concurrencies):
        parser.error("supported concurrencies are 1, 8, and 16")
    partials = args.partial_lens or PARTIALS_BY_PAGE[args.page_size]
    if any(r <= 0 or r >= args.page_size for r in partials):
        parser.error("every partial length must satisfy 0 < R < page_size")
    args.partial_lens = partials
    args.suffix_len = args.suffix_len or args.page_size + 1
    args.num_prompts_by_concurrency = {
        1: args.num_prompts_c1,
        8: args.num_prompts_c8,
        16: args.num_prompts_c16,
    }
    for concurrency in args.concurrencies:
        if args.num_prompts_by_concurrency[concurrency] < 5 * concurrency:
            parser.error(
                f"C={concurrency} requires at least {5 * concurrency} prompts "
                "to follow the bench_serving steady-state guidance"
            )
    return args


def runtime_env(cuda_visible_devices: str) -> dict[str, str]:
    env = os.environ.copy()
    venv_root = Path(sys.prefix)
    cuda_home = venv_root / "lib/python3.12/site-packages/nvidia/cu13"
    env.update(
        {
            "CUDA_HOME": str(cuda_home),
            "CUDA_VISIBLE_DEVICES": cuda_visible_devices,
            "LD_LIBRARY_PATH": str(cuda_home / "lib64"),
            "SGLANG_ENABLE_UNIFIED_RADIX_TREE": "0",
            "SGLANG_EXPERIMENTAL_CPP_RADIX_TREE": "0",
            "SGLANG_USE_HND_KVCACHE": "0",
            "PYTHONUNBUFFERED": "1",
        }
    )
    env["PATH"] = f"{cuda_home / 'bin'}:{venv_root / 'bin'}:{env.get('PATH', '')}"
    return env


def wait_for_server(base_url: str, process: subprocess.Popen, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited early with code {process.returncode}")
        try:
            response = requests.get(f"{base_url}/v1/models", timeout=2)
            if response.ok:
                return
        except requests.RequestException as exc:
            last_error = exc
        time.sleep(1)
    raise TimeoutError(f"server did not become ready: {last_error}")


def stop_server(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGINT)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def load_last_json(path: Path) -> dict[str, Any]:
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"empty result file: {path}")
    return json.loads(lines[-1])


def validate_result(
    path: Path,
    *,
    expected_cached: int,
    expected_requests: int,
    expected_prompt_len: int,
    expected_tag: str,
) -> dict[str, Any]:
    result = load_last_json(path)
    cached = result.get("cached_tokens") or []
    input_lens = result.get("input_lens") or []
    errors = result.get("errors") or []
    if result.get("tag") != expected_tag:
        raise RuntimeError(f"tag mismatch in {path}: {result.get('tag')!r}")
    if result.get("completed") != expected_requests:
        raise RuntimeError(
            f"completed mismatch in {path}: {result.get('completed')} != {expected_requests}"
        )
    if len(cached) != expected_requests or set(cached) != {expected_cached}:
        raise RuntimeError(
            f"cached-token mismatch in {path}: expected {expected_cached}, "
            f"observed {sorted(set(cached))}, count={len(cached)}"
        )
    if len(input_lens) != expected_requests or set(input_lens) != {expected_prompt_len}:
        raise RuntimeError(
            f"input-length mismatch in {path}: expected {expected_prompt_len}, "
            f"observed {sorted(set(input_lens))}, count={len(input_lens)}"
        )
    if any(errors):
        raise RuntimeError(f"request errors in {path}: {[e for e in errors if e][:3]}")
    return result


def result_is_valid(path: Path, **expected: Any) -> bool:
    if not path.exists():
        return False
    try:
        validate_result(path, **expected)
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError):
        return False
    return True


def case_offset(
    page_size: int, partial_len: int, concurrency: int, warmup: bool
) -> int:
    return (
        page_size * 1_000_000
        + partial_len * 10_000
        + concurrency * 100
        + (1 if warmup else 50)
    )


def run_case(
    args: argparse.Namespace,
    *,
    config: str,
    partial_len: int,
    concurrency: int,
    warmup: bool,
    config_dir: Path,
    env: dict[str, str],
) -> dict[str, Any] | None:
    expected_cached = args.aligned_match_tokens + (
        partial_len if config == "reuse" else 0
    )
    num_prompts = (
        max(16, 2 * concurrency)
        if warmup
        else args.num_prompts_by_concurrency[concurrency]
    )
    phase = "shape_warmup" if warmup else "measured"
    tag = (
        f"{args.model_name}-p{args.page_size}-r{partial_len}-c{concurrency}-"
        f"o{args.output_len}-{config}-{phase}"
    )
    result_dir = config_dir / ("warmups" if warmup else "cases")
    result_dir.mkdir(parents=True, exist_ok=True)
    result_path = (
        result_dir
        / f"p{args.page_size}_r{partial_len}_c{concurrency}_o{args.output_len}.jsonl"
    )
    log_path = result_path.with_suffix(".log")
    prompt_len = args.aligned_match_tokens + partial_len + args.suffix_len
    expected = {
        "expected_cached": expected_cached,
        "expected_requests": num_prompts,
        "expected_prompt_len": prompt_len,
        "expected_tag": tag,
    }
    if result_is_valid(result_path, **expected):
        print(f"SKIP valid {tag}", flush=True)
        return None if warmup else validate_result(result_path, **expected)

    result_path.unlink(missing_ok=True)
    cache_salt = f"pp-{tag}-{time.time_ns()}"
    adapter = Path(__file__).with_name("bench_partial_prefix_serving.py")
    command = [
        sys.executable,
        str(adapter),
        "--pp-page-size",
        str(args.page_size),
        "--pp-aligned-prefix-tokens",
        str(args.aligned_match_tokens - args.page_size),
        "--pp-partial-len",
        str(partial_len),
        "--pp-suffix-len",
        str(args.suffix_len),
        "--pp-output-len",
        str(args.output_len),
        "--pp-cache-salt",
        cache_salt,
        "--pp-manual-warmups",
        str(args.manual_warmups),
        "--pp-concurrent-warmups",
        str(concurrency if not warmup and concurrency > 1 else 0),
        "--pp-sample-offset",
        str(case_offset(args.page_size, partial_len, concurrency, warmup)),
        "--pp-expected-cached-tokens",
        str(expected_cached),
        "--pp-request-timeout",
        str(args.request_timeout),
        "--backend",
        "sglang",
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--model",
        args.model_path,
        "--tokenizer",
        args.model_path,
        "--num-prompts",
        str(num_prompts),
        "--max-concurrency",
        str(concurrency),
        "--output-file",
        str(result_path),
        "--output-details",
        "--cache-report",
        "--disable-tqdm",
        "--warmup-requests",
        "0",
        "--temperature",
        "0",
        "--tag",
        tag,
    ]
    print(f"RUN {tag}", flush=True)
    with log_path.open("w") as log_file:
        completed = subprocess.run(
            command,
            cwd=Path(__file__).resolve().parents[2],
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            timeout=args.request_timeout + 300,
            check=False,
            text=True,
        )
    if completed.returncode:
        tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-80:])
        raise RuntimeError(
            f"benchmark failed ({completed.returncode}) for {tag}:\n{tail}"
        )
    result = validate_result(result_path, **expected)
    print(
        f"DONE {tag}: throughput={result['request_throughput']:.3f} req/s "
        f"ttft={result['mean_ttft_ms']:.3f} ms",
        flush=True,
    )
    return None if warmup else result


def run_config(
    args: argparse.Namespace,
    *,
    config: str,
    page_dir: Path,
    env: dict[str, str],
) -> None:
    config_dir = page_dir / config
    config_dir.mkdir(parents=True, exist_ok=True)
    server_log_path = config_dir / "server.log"
    command = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        args.model_path,
        "--tp",
        str(args.tp),
        "--page-size",
        str(args.page_size),
        "--attention-backend",
        "triton",
        "--mem-fraction-static",
        str(args.mem_fraction_static),
        "--random-seed",
        "0",
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
    ]
    if config == "reuse":
        command.append("--enable-partial-prefix-reuse")
    command.extend(args.extra_server_args)

    print(f"START server {args.model_name} p{args.page_size} {config}", flush=True)
    with server_log_path.open("a") as server_log:
        process = subprocess.Popen(
            command,
            cwd=Path(__file__).resolve().parents[2],
            env=env,
            stdout=server_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            text=True,
        )
        try:
            wait_for_server(
                f"http://127.0.0.1:{args.port}", process, args.server_timeout
            )
            summary: dict[str, Any] = {}
            for partial_len in args.partial_lens:
                for concurrency in args.concurrencies:
                    if (
                        concurrency == 8
                        and args.sparse_c8
                        and partial_len
                        not in {1, args.page_size // 2, args.page_size - 1}
                    ):
                        continue
                    result = run_case(
                        args,
                        config=config,
                        partial_len=partial_len,
                        concurrency=concurrency,
                        warmup=False,
                        config_dir=config_dir,
                        env=env,
                    )
                    if result is not None:
                        summary[f"r{partial_len}_c{concurrency}"] = {
                            "request_throughput": result["request_throughput"],
                            "mean_ttft_ms": result["mean_ttft_ms"],
                            "median_ttft_ms": result["median_ttft_ms"],
                            "p90_ttft_ms": result["p90_ttft_ms"],
                            "p99_ttft_ms": result["p99_ttft_ms"],
                            "mean_e2e_latency_ms": result["mean_e2e_latency_ms"],
                            "median_e2e_latency_ms": result["median_e2e_latency_ms"],
                            "p90_e2e_latency_ms": result["p90_e2e_latency_ms"],
                            "p99_e2e_latency_ms": result["p99_e2e_latency_ms"],
                        }
            (config_dir / "summary.json").write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n"
            )
        finally:
            stop_server(process)
    print(f"STOP server {args.model_name} p{args.page_size} {config}", flush=True)


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    page_dir = output_root / args.model_name / f"page_{args.page_size}"
    page_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "model_name": args.model_name,
        "model_path": str(Path(args.model_path).resolve()),
        "page_size": args.page_size,
        "aligned_match_tokens": args.aligned_match_tokens,
        "partial_lens": args.partial_lens,
        "concurrencies": args.concurrencies,
        "num_prompts_by_concurrency": args.num_prompts_by_concurrency,
        "sparse_c8": args.sparse_c8,
        "suffix_len": args.suffix_len,
        "output_len": args.output_len,
        "tp": args.tp,
        "cuda_visible_devices": args.cuda_visible_devices,
        "mem_fraction_static": args.mem_fraction_static,
        "configs": args.configs,
        "timestamp": time.time(),
    }
    (page_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    env = runtime_env(args.cuda_visible_devices)
    for config in args.configs:
        run_config(args, config=config, page_dir=page_dir, env=env)


if __name__ == "__main__":
    main()
