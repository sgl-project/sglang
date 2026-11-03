"""SGLang auto-tune CLI (v1: attention-backend selection).

Create/edit this file on the machine where you run experiments (e.g. SeaWulf).

Example::

  python -m sglang.auto_tune \\
    --tune attention-backend \\
    --model-path Qwen/Qwen3.5-9B \\
    --backends triton,flashinfer \\
    --output-dir /tmp/sglang-attn-tune
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import requests

HIGHER_IS_BETTER = frozenset(
    {
        "output_throughput",
        "input_throughput",
        "total_throughput",
        "request_throughput",
    }
)
LOWER_IS_BETTER = frozenset(
    {
        "median_ttft_ms",
        "mean_ttft_ms",
        "median_tpot_ms",
        "mean_tpot_ms",
        "median_itl_ms",
        "mean_itl_ms",
        "median_e2e_latency_ms",
        "mean_e2e_latency_ms",
    }
)
DEFAULT_BACKENDS = ("triton", "flashinfer")
DEFAULT_PRIMARY_METRIC = "output_throughput"


@dataclass
class WorkloadConfig:
    input_len: int = 256
    output_len: int = 32
    max_concurrency: int = 16
    num_prompts: int = 80


@dataclass
class CandidateResult:
    backend: str
    ok: bool
    error: Optional[str] = None
    traceback: Optional[str] = None
    metrics: dict[str, Any] = field(default_factory=dict)

    def metric_value(self, name: str) -> Optional[float]:
        if not self.ok:
            return None
        value = self.metrics.get(name)
        if value is None:
            return None
        return float(value)


@dataclass
class TuneResult:
    model_path: str
    device_name: str
    tp: int
    workload: WorkloadConfig
    primary_metric: str
    candidates: list[CandidateResult]
    best_attention_backend: Optional[str]
    recommended_args: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_path": self.model_path,
            "device_name": self.device_name,
            "tp": self.tp,
            "workload": asdict(self.workload),
            "primary_metric": self.primary_metric,
            "candidates": [asdict(c) for c in self.candidates],
            "best_attention_backend": self.best_attention_backend,
            "recommended_args": self.recommended_args,
        }


def parse_backends(backends: str | list[str]) -> list[str]:
    if isinstance(backends, list):
        items = [str(part).strip() for part in backends]
    else:
        items = [part.strip() for part in backends.split(",")]
    out = [b for b in items if b]
    if not out:
        raise ValueError("At least one attention backend is required")
    return out


def pick_best_backend(
    candidates: list[CandidateResult],
    primary_metric: str = DEFAULT_PRIMARY_METRIC,
) -> Optional[str]:
    if primary_metric in HIGHER_IS_BETTER:
        higher_is_better = True
    elif primary_metric in LOWER_IS_BETTER:
        higher_is_better = False
    else:
        raise ValueError(
            f"Unsupported primary metric {primary_metric!r}. "
            f"Supported: {sorted(HIGHER_IS_BETTER | LOWER_IS_BETTER)}"
        )

    best_name: Optional[str] = None
    best_value: Optional[float] = None
    for cand in candidates:
        value = cand.metric_value(primary_metric)
        if value is None:
            continue
        if best_value is None:
            best_name, best_value = cand.backend, value
            continue
        if higher_is_better and value > best_value:
            best_name, best_value = cand.backend, value
        elif not higher_is_better and value < best_value:
            best_name, best_value = cand.backend, value
    return best_name


def detect_device_name() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return "unknown"


def build_server_command(
    *,
    model_path: str,
    backend: str,
    tp: int,
    host: str,
    port: int,
    extra_args: Optional[list[str]] = None,
) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        model_path,
        "--attention-backend",
        backend,
        "--tp",
        str(tp),
        "--host",
        host,
        "--port",
        str(port),
    ]
    if extra_args:
        cmd.extend(extra_args)
    return cmd


def get_auth_headers() -> dict[str, str]:
    """Mirror sglang.benchmark.serving.get_auth_headers() without importing that module."""
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if openai_api_key:
        return {"Authorization": f"Bearer {openai_api_key}"}
    api_key = os.environ.get("API_KEY")
    if api_key:
        return {"Authorization": api_key}
    return {}


def wait_until_ready(
    base_url: str,
    process: subprocess.Popen,
    timeout_s: float,
) -> None:
    url = f"{base_url.rstrip('/')}/v1/models"
    deadline = time.perf_counter() + timeout_s
    headers = get_auth_headers()
    while time.perf_counter() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"Server process exited early with code {process.returncode}"
            )
        try:
            resp = requests.get(url, headers=headers, timeout=5)
            if resp.status_code == 200:
                return
        except requests.exceptions.RequestException:
            pass
        time.sleep(1)
    raise TimeoutError(f"Server at {base_url} did not become ready within {timeout_s}s")


def build_bench_command(
    *,
    model_path: str,
    host: str,
    port: int,
    workload: WorkloadConfig,
    output_file: Path,
) -> list[str]:
    """Build the same CLI invocation a human would run by hand for bench_serving."""
    return [
        sys.executable,
        "-m",
        "sglang.benchmark.serving",
        "--backend",
        "sglang",
        "--model",
        model_path,
        "--host",
        host,
        "--port",
        str(port),
        "--dataset-name",
        "random",
        "--random-input-len",
        str(workload.input_len),
        "--random-output-len",
        str(workload.output_len),
        "--max-concurrency",
        str(workload.max_concurrency),
        "--num-prompts",
        str(workload.num_prompts),
        "--output-file",
        str(output_file),
    ]


def run_bench_subprocess(
    *,
    model_path: str,
    host: str,
    port: int,
    workload: WorkloadConfig,
    result_file: Path,
    timeout_s: float,
) -> dict[str, Any]:
    """Run the public bench_serving CLI and read its JSONL result file.

    Using the real CLI (instead of importing sglang.benchmark.serving.run_benchmark
    directly) avoids depending on that function's private argparse.Namespace surface
    and its module-global ``args`` state, which can change without notice.
    """
    if result_file.exists():
        result_file.unlink()

    bench_cmd = build_bench_command(
        model_path=model_path,
        host=host,
        port=port,
        workload=workload,
        output_file=result_file,
    )
    proc = subprocess.run(
        bench_cmd,
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"bench_serving exited with {proc.returncode}: "
            f"{proc.stderr[-2000:] if proc.stderr else proc.stdout[-2000:]}"
        )
    if not result_file.exists():
        raise RuntimeError(
            f"bench_serving did not write a result file at {result_file}"
        )

    lines = [
        line for line in result_file.read_text(encoding="utf-8").splitlines() if line
    ]
    if not lines:
        raise RuntimeError(f"Result file {result_file} is empty")
    return json.loads(lines[-1])


def _extract_metrics(bench_result: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "output_throughput",
        "input_throughput",
        "total_throughput",
        "request_throughput",
        "median_ttft_ms",
        "mean_ttft_ms",
        "median_tpot_ms",
        "mean_tpot_ms",
        "median_itl_ms",
        "mean_itl_ms",
        "median_e2e_latency_ms",
        "mean_e2e_latency_ms",
        "completed",
        "duration",
    ]
    return {k: bench_result.get(k) for k in keys if k in bench_result}


def run_one_backend(
    *,
    model_path: str,
    backend: str,
    tp: int,
    host: str,
    port: int,
    workload: WorkloadConfig,
    server_timeout_s: float,
    output_dir: Path,
    extra_server_args: Optional[list[str]] = None,
    dry_run: bool = False,
    bench_timeout_s: float = 900.0,
    bench_runner_fn: Optional[Callable[..., dict[str, Any]]] = None,
) -> CandidateResult:
    base_url = f"http://{host}:{port}"
    server_cmd = build_server_command(
        model_path=model_path,
        backend=backend,
        tp=tp,
        host=host,
        port=port,
        extra_args=extra_server_args,
    )

    if dry_run:
        print(f"[dry-run] server: {subprocess.list2cmdline(server_cmd)}")
        print(
            f"[dry-run] bench: random in={workload.input_len} "
            f"out={workload.output_len} concurrency={workload.max_concurrency} "
            f"prompts={workload.num_prompts} @ {base_url}"
        )
        return CandidateResult(backend=backend, ok=True, metrics={"dry_run": True})

    process: Optional[subprocess.Popen] = None
    log_path = output_dir / f"{backend}-{port}.server.log"
    result_file = output_dir / f"{backend}-{port}.bench.jsonl"
    try:
        with log_path.open("w", encoding="utf-8") as log_file:
            process = subprocess.Popen(
                server_cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                env=os.environ.copy(),
            )
            print(f"  server log: {log_path}")
            wait_until_ready(base_url, process, timeout_s=server_timeout_s)

            bench_runner = bench_runner_fn or run_bench_subprocess
            bench_result = bench_runner(
                model_path=model_path,
                host=host,
                port=port,
                workload=workload,
                result_file=result_file,
                timeout_s=bench_timeout_s,
            )
        if not isinstance(bench_result, dict):
            raise RuntimeError(f"Unexpected bench result type: {type(bench_result)}")
        return CandidateResult(
            backend=backend, ok=True, metrics=_extract_metrics(bench_result)
        )
    except Exception as exc:  # noqa: BLE001
        tb = traceback.format_exc()
        print(f"  error running {backend}:\n{tb}")
        return CandidateResult(backend=backend, ok=False, error=str(exc), traceback=tb)
    finally:
        if process is not None and process.poll() is None:
            try:
                from sglang.srt.utils import kill_process_tree

                kill_process_tree(process.pid)
            except Exception:
                process.kill()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)


def tune_attention_backends(
    *,
    model_path: str,
    backends: list[str],
    output_dir: str | Path,
    tp: int = 1,
    host: str = "127.0.0.1",
    port: int = 30000,
    workload: Optional[WorkloadConfig] = None,
    primary_metric: str = DEFAULT_PRIMARY_METRIC,
    server_timeout_s: float = 1800.0,
    bench_timeout_s: float = 900.0,
    extra_server_args: Optional[list[str]] = None,
    dry_run: bool = False,
    device_name: Optional[str] = None,
    run_one_fn: Optional[Callable[..., CandidateResult]] = None,
) -> TuneResult:
    workload = workload or WorkloadConfig()
    run_one = run_one_fn or run_one_backend
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates: list[CandidateResult] = []

    for backend in backends:
        print(f"=== Tuning attention backend: {backend} ===")
        cand = run_one(
            model_path=model_path,
            backend=backend,
            tp=tp,
            host=host,
            port=port,
            workload=workload,
            server_timeout_s=server_timeout_s,
            bench_timeout_s=bench_timeout_s,
            output_dir=output_dir,
            extra_server_args=extra_server_args,
            dry_run=dry_run,
        )
        candidates.append(cand)
        if cand.ok:
            print(f"  ok: {cand.metrics}")
        else:
            print(f"  failed filter: {cand.error}")

    best = (
        backends[0]
        if dry_run and backends
        else pick_best_backend(candidates, primary_metric=primary_metric)
    )
    recommended = ["--attention-backend", best] if best is not None else []
    return TuneResult(
        model_path=model_path,
        device_name=device_name or detect_device_name(),
        tp=tp,
        workload=workload,
        primary_metric=primary_metric,
        candidates=candidates,
        best_attention_backend=best,
        recommended_args=recommended,
    )


def write_result(result: TuneResult, output_dir: str | Path) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "attention_backend_tune.json"
    path.write_text(json.dumps(result.to_dict(), indent=2) + "\n", encoding="utf-8")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Auto-tune SGLang attention backends for a model."
    )
    parser.add_argument(
        "--tune",
        type=str,
        default="attention-backend",
        choices=["attention-backend"],
    )
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument(
        "--backends",
        type=str,
        default=",".join(DEFAULT_BACKENDS),
        help=f"Comma-separated backends (default: {','.join(DEFAULT_BACKENDS)}).",
    )
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--input-len", type=int, default=256)
    parser.add_argument("--output-len", type=int, default=32)
    parser.add_argument("--max-concurrency", type=int, default=16)
    parser.add_argument("--num-prompts", type=int, default=80)
    parser.add_argument(
        "--primary-metric",
        type=str,
        default=DEFAULT_PRIMARY_METRIC,
        choices=sorted(HIGHER_IS_BETTER | LOWER_IS_BETTER),
    )
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--server-timeout", type=float, default=1800.0)
    parser.add_argument("--bench-timeout", type=float, default=900.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--extra-server-arg", action="append", default=[])
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    backends = parse_backends(args.backends)
    workload = WorkloadConfig(
        input_len=args.input_len,
        output_len=args.output_len,
        max_concurrency=args.max_concurrency,
        num_prompts=args.num_prompts,
    )
    result = tune_attention_backends(
        model_path=args.model_path,
        backends=backends,
        output_dir=args.output_dir,
        tp=args.tp,
        host=args.host,
        port=args.port,
        workload=workload,
        primary_metric=args.primary_metric,
        server_timeout_s=args.server_timeout,
        bench_timeout_s=args.bench_timeout,
        extra_server_args=args.extra_server_arg or None,
        dry_run=args.dry_run,
    )
    path = write_result(result, args.output_dir)
    print()
    print(f"Wrote {path}")
    if result.best_attention_backend is None:
        print("No successful backend; see candidates in the JSON output.")
        return 1
    print(f"Best attention backend: {result.best_attention_backend}")
    print("Recommended args:", " ".join(result.recommended_args))
    return 0


if __name__ == "__main__":
    sys.exit(main())
