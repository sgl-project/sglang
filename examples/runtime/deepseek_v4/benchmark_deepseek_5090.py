#!/usr/bin/env python3
"""Run a one-shot DeepSeek V4 Flash expert-pack benchmark on one RTX 5090."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PROMPT = "Please introduce Shenzhen"
CHAT_PREFIX = "<\uff5cbegin\u2581of\u2581sentence\uff5c>You are a helpful assistant.<\uff5cUser\uff5c>"
CHAT_SUFFIX = "<\uff5cAssistant\uff5c><think>"
DEFAULT_LOCK = Path("/tmp/sglang-deepseek-v4-5090-benchmark.lock")
METADATA_FORMAT_VERSION = 4
ACTIVE_MOE_LAYERS = tuple(range(43))


def cache_root() -> Path:
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")).expanduser()


def artifact_dir_for_source(path: Path) -> Path:
    stat = path.stat()
    fingerprint = hashlib.sha256(
        f"{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}:{METADATA_FORMAT_VERSION}".encode()
    ).hexdigest()[:20]
    return cache_root() / "sglang-expert-pack" / "deepseek-v4-flash" / fingerprint


def find_sglang_repo() -> Path:
    configured = os.environ.get("SGLANG_REPO")
    if configured:
        return Path(configured).expanduser().resolve()
    for candidate in (SCRIPT_DIR, *SCRIPT_DIR.parents):
        if (candidate / "python" / "sglang").is_dir() and (
            candidate / "python" / "sglang" / "srt" / "model_loader" / "expert_pack"
        ).is_dir():
            return candidate
    raise RuntimeError("could not locate the SGLang repository")


def format_prompt(prompt: str) -> str:
    return f"{CHAT_PREFIX}{prompt}{CHAT_SUFFIX}"


def server_url(args: argparse.Namespace) -> str:
    return f"http://{args.host}:{args.port}"


def port_in_use(host: str, port: int, timeout: float = 0.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def detect_rtx_5090() -> str:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    gpu_zero = next(
        (line.split(",", 1)[1].strip() for line in rows if line.startswith("0,")),
        None,
    )
    if gpu_zero is None or "5090" not in gpu_zero:
        raise RuntimeError(
            f"CUDA device 0 must be an RTX 5090; detected: {', '.join(rows) or 'none'}"
        )
    return gpu_zero


def build_server_command(args: argparse.Namespace) -> list[str]:
    extra_config = {
        "cache_vram_mib": args.expert_cache_mib,
        "cache_vram_reserve_mib": args.expert_cache_reserve_mib,
        "stage_slots": args.stage_slots,
        "read_splits": args.read_splits,
        "direct_io": args.direct_io,
        "stats_flush_interval": len(ACTIVE_MOE_LAYERS),
        "stats_path": str(args.stats_path),
    }
    return [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        str(args.gguf),
        "--trust-remote-code",
        "--load-format",
        "expert_pack",
        "--model-loader-extra-config",
        json.dumps(extra_config, separators=(",", ":")),
        "--attention-backend",
        "dsv4",
        "--tp-size",
        "1",
        "--ep-size",
        "1",
        "--disable-flashinfer-autotune",
        "--skip-server-warmup",
        "--context-length",
        str(args.context_length),
        "--max-total-tokens",
        str(args.max_total_tokens),
        "--max-running-requests",
        "1",
        "--mem-fraction-static",
        str(args.mem_fraction_static),
        "--watchdog-timeout",
        str(args.watchdog_timeout),
        "--host",
        args.host,
        "--port",
        str(args.port),
    ]


def start_server(args: argparse.Namespace) -> subprocess.Popen:
    if port_in_use(args.host, args.port):
        raise RuntimeError(f"server address is already in use: {server_url(args)}")
    args.server_log.parent.mkdir(parents=True, exist_ok=True)
    log = args.server_log.open("wb", buffering=0)
    env = os.environ.copy()
    python_path = [str(args.sglang_repo), str(args.sglang_repo / "python")]
    if env.get("PYTHONPATH"):
        python_path.append(env["PYTHONPATH"])
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["PYTHONPATH"] = os.pathsep.join(python_path)
    env.setdefault("SGLANG_OPT_USE_TILELANG_INDEXER", "1")
    conda_lib = str(Path(sys.prefix) / "lib")
    cuda_root = Path("/usr/local/cuda")
    if (cuda_root / "bin" / "nvcc").is_file():
        env["CUDA_HOME"] = str(cuda_root)
        env["CUDA_PATH"] = str(cuda_root)
        env["PATH"] = os.pathsep.join((str(cuda_root / "bin"), env.get("PATH", "")))
    env["LD_LIBRARY_PATH"] = os.pathsep.join(
        value
        for value in (
            conda_lib,
            str(cuda_root / "lib64") if (cuda_root / "lib64").is_dir() else None,
            env.get("LD_LIBRARY_PATH"),
        )
        if value
    )
    command = build_server_command(args)
    print(
        f"SERVICE_STARTING url={server_url(args)} timeout={args.startup_timeout:.0f}s "
        f"log={args.server_log}",
        flush=True,
    )
    process = subprocess.Popen(
        command,
        cwd=args.sglang_repo,
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=subprocess.STDOUT,
        env=env,
        start_new_session=True,
    )
    process._benchmark_log = log  # type: ignore[attr-defined]
    try:
        deadline = time.monotonic() + args.startup_timeout
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(
                    f"SGLang exited during startup with code {process.returncode}; "
                    f"see {args.server_log}"
                )
            if port_in_use(args.host, args.port):
                print(
                    f"SERVICE_READY pid={process.pid} url={server_url(args)}",
                    flush=True,
                )
                return process
            time.sleep(2)
        raise TimeoutError(
            f"SGLang did not become ready within {args.startup_timeout:.0f}s; "
            f"see {args.server_log}"
        )
    except BaseException:
        stop_server(process, args)
        raise


def stop_server(process: subprocess.Popen | None, args: argparse.Namespace) -> None:
    if process is None:
        return
    log = getattr(process, "_benchmark_log", None)
    try:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=45)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=15)
        deadline = time.monotonic() + 10
        while (
            port_in_use(args.host, args.port, timeout=0.2)
            and time.monotonic() < deadline
        ):
            time.sleep(0.2)
        print(f"SERVICE_STOPPED pid={process.pid} url={server_url(args)}", flush=True)
    finally:
        if log is not None:
            log.close()


def generate(
    args: argparse.Namespace,
    prompt: str,
    max_new_tokens: int,
    *,
    stream_output: bool,
) -> dict[str, Any]:
    payload = {
        "text": format_prompt(prompt),
        "sampling_params": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "sampling_seed": args.seed,
            "max_new_tokens": max_new_tokens,
            "ignore_eos": True,
        },
        "stream": True,
    }
    request = urllib.request.Request(
        server_url(args) + "/generate",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter_ns()
    first_token = last_token = None
    completion_tokens = 0
    prompt_tokens = None
    output = ""
    finish_reason = None
    if stream_output:
        print(f"prompt: {prompt}", flush=True)
        print("output: ", end="", flush=True)

    with urllib.request.urlopen(request, timeout=args.request_timeout) as response:
        for raw_line in response:
            now = time.perf_counter_ns()
            line = raw_line.decode("utf-8").strip()
            if not line:
                continue
            if line.startswith("data: "):
                line = line[6:]
            if line == "[DONE]":
                continue
            event = json.loads(line)
            meta = event.get("meta_info") or {}
            current_tokens = int(meta.get("completion_tokens", 0))
            if current_tokens > completion_tokens:
                first_token = first_token or now
                last_token = now
                completion_tokens = current_tokens
            if meta.get("prompt_tokens") is not None:
                prompt_tokens = int(meta["prompt_tokens"])
            event_output = event.get("text")
            if event_output is not None:
                if stream_output and event_output != output:
                    if event_output.startswith(output):
                        print(event_output[len(output) :], end="", flush=True)
                    else:
                        print(f"\n[output revised]\n{event_output}", end="", flush=True)
                output = event_output
            finish_reason = meta.get("finish_reason", finish_reason)
    if stream_output:
        print(flush=True)
    if first_token is None or last_token is None or prompt_tokens is None:
        raise RuntimeError(
            "SGLang response did not contain complete token timing metadata"
        )
    ttft_s = (first_token - started) / 1e9
    decode_span_s = (last_token - first_token) / 1e9
    total_s = (time.perf_counter_ns() - started) / 1e9
    decode_intervals = max(0, completion_tokens - 1)
    return {
        "prompt": prompt,
        "output": output,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "finish_reason": finish_reason,
        "ttft_ms": ttft_s * 1000,
        "prefill_token_rate": prompt_tokens / ttft_s if ttft_s > 0 else None,
        "decode_token_rate": (
            decode_intervals / decode_span_s if decode_span_s > 0 else None
        ),
        "tpot_ms": (
            decode_span_s * 1000 / decode_intervals if decode_intervals else None
        ),
        "total_elapsed_s": total_s,
        "end_to_end_token_rate": completion_tokens / total_s if total_s > 0 else None,
    }


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    process = None
    try:
        if args.stats_path.exists():
            args.stats_path.unlink()
        process = start_server(args)
        return generate(args, args.prompt, args.max_new_tokens, stream_output=True)
    finally:
        stop_server(process, args)


def read_stats(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"expert-pack stats were not written: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def audit_routes(stats: dict[str, Any], expected_tokens: int) -> None:
    token_counts = stats.get("route_tokens_by_layer") or []
    call_counts = stats.get("route_calls_by_layer") or []
    if len(token_counts) != len(ACTIVE_MOE_LAYERS) or len(call_counts) != len(
        ACTIVE_MOE_LAYERS
    ):
        raise RuntimeError("DeepSeek Expert Pack stats have an unexpected layer count")
    for layer in ACTIVE_MOE_LAYERS:
        if call_counts[layer] <= 0 or token_counts[layer] != expected_tokens:
            raise RuntimeError(
                f"layer {layer} routed {token_counts[layer]} tokens in "
                f"{call_counts[layer]} calls; expected {expected_tokens} tokens"
            )
    if int(stats.get("fallback_count", 0)) != 0:
        raise RuntimeError("the request used an expert fallback")
    if int(stats.get("io_errors", 0)) != 0:
        raise RuntimeError("the request encountered Expert Pack I/O errors")


def _git_sha(repo: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def write_report(args: argparse.Namespace, gpu: str, result: dict[str, Any]) -> None:
    report = {
        "format": "SGLANG-DEEPSEEK-V4-FLASH-EXPERT-PACK-BENCHMARK-v1",
        "git_sha": _git_sha(args.sglang_repo),
        "gpu": gpu,
        "source_path": str(args.gguf),
        "result": result,
        "expert_pack_stats": read_stats(args.stats_path),
        "server_log": str(args.server_log),
    }
    temporary = args.report_path.with_suffix(args.report_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.report_path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gguf",
        type=Path,
        required=True,
        help="DeepSeek-V4-Flash source GGUF; server startup derives all artifacts",
    )
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.set_defaults(
        prompt=DEFAULT_PROMPT,
        temperature=0.0,
        top_p=0.95,
        seed=20260810,
        host="127.0.0.1",
        port=30000,
        startup_timeout=1200,
        request_timeout=3600,
        watchdog_timeout=1800,
        context_length=32768,
        max_total_tokens=32768,
        mem_fraction_static=0.96,
        expert_cache_mib=21 * 1024,
        expert_cache_reserve_mib=2 * 1024,
        stage_slots=12,
        read_splits=4,
        direct_io=True,
    )
    args = parser.parse_args(argv)
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    for name in (
        "expert_cache_mib",
        "expert_cache_reserve_mib",
        "stage_slots",
        "read_splits",
    ):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    args.gguf = args.gguf.expanduser().resolve(strict=True)
    args.sglang_repo = find_sglang_repo()
    args.artifact_dir = artifact_dir_for_source(args.gguf).resolve()
    args.server_log = args.artifact_dir / "deepseek-v4-5090-server.log"
    args.stats_path = args.artifact_dir / "deepseek-v4-expert-pack.stats.json"
    args.report_path = args.artifact_dir / "deepseek-v4-5090-benchmark.json"
    return args


def handle_termination(signum: int, _frame: object) -> None:
    raise KeyboardInterrupt(f"received signal {signum}")


def print_result(result: dict[str, Any], gpu: str) -> None:
    print(f"gpu: {gpu}")
    print(f"prompt_tokens: {result['prompt_tokens']}")
    print(f"completion_tokens: {result['completion_tokens']}")
    for name, suffix in (
        ("ttft_ms", ""),
        ("prefill_token_rate", " tok/s"),
        ("decode_token_rate", " tok/s"),
        ("tpot_ms", " ms/token"),
        ("end_to_end_token_rate", " tok/s"),
    ):
        value = result[name]
        print(f"{name}: {'n/a' if value is None else f'{value:.3f}'}{suffix}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    signal.signal(signal.SIGTERM, handle_termination)
    signal.signal(signal.SIGHUP, handle_termination)
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    lock = DEFAULT_LOCK.open("w")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(
            f"error: another benchmark is running (lock: {DEFAULT_LOCK})",
            file=sys.stderr,
        )
        return 2
    lock.write(f"{os.getpid()}\n")
    lock.flush()
    try:
        gpu = detect_rtx_5090()
        print(f"MODEL_INPUT_READY gpu={gpu} gguf={args.gguf}", flush=True)
        result = run_benchmark(args)
        stats = read_stats(args.stats_path)
        audit_routes(stats, result["prompt_tokens"] + result["completion_tokens"])
        write_report(args, gpu, result)
        print_result(result, gpu)
        print(f"report: {args.report_path}")
        print(f"server_log: {args.server_log}")
        return 0
    except KeyboardInterrupt:
        print("error: benchmark interrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        lock.close()


if __name__ == "__main__":
    raise SystemExit(main())
