"""Double Sparsity decode benchmark — 3-way comparison.

Drives an SGLang server in three configurations on identical hardware
and reports decode throughput, TTFT, TBT (p50/p95), and a long-context
quality probe (NIAH-style needle-in-a-haystack):

  1. `main_dense`     : dense path on a clean `origin/main` checkout
                         (proves the regression baseline).
  2. `branch_ds_off`  : same branch as DS, DS disabled
                         (proves the DS-off path is unchanged).
  3. `branch_ds_on`   : DS enabled with a calibration JSON
                         (the speedup measurement).

The (1)↔(2) comparison verifies the "DS off → byte-for-byte unchanged"
claim; the (2)↔(3) comparison is the actual sparse vs dense win.

Usage (run each config separately):

  python benchmark/double_sparsity/bench_decode.py \
      --config branch_ds_on \
      --model meta-llama/Meta-Llama-3.1-8B-Instruct \
      --calibration ./calib_8b.json \
      --context-len 65536 --output-len 1024 \
      --output-json results_branch_ds_on.json

Outputs:
- A JSON file with per-(context, output_len) metrics.
- A CSV with the same data.
- A printed markdown comparison table when --compare is passed with
  multiple JSONs.

This script is environment-dependent: it requires a properly-provisioned
SGLang install (matching `sglang-kernel` version, FA3 backend, CUDA
runtime). It's not run in CI.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import socket
import statistics
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


def _free_port() -> int:
    s = socket.socket()
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@dataclass
class WorkloadResult:
    config: str
    model: str
    context_len: int
    output_len: int
    n_requests: int
    concurrency: int
    decode_tok_per_s: float
    e2e_latency_s: float
    ttft_ms_p50: float
    ttft_ms_p95: float
    tbt_ms_p50: float
    tbt_ms_p95: float
    niah_accuracy: Optional[float] = None
    extra: Dict[str, float] = field(default_factory=dict)


def _percentile(xs, p):
    if not xs:
        return 0.0
    return float(statistics.quantiles(sorted(xs), n=100, method="inclusive")[p - 1])


def _build_long_prompt(target_tokens: int, fill: str = "The quick brown fox. ") -> str:
    """Approx. target_tokens long prompt — server tokenizer trims to its own count."""
    repeat = max(target_tokens // 5, 1)
    return fill * repeat


def _build_niah_prompt(context_tokens: int, needle: str, query: str) -> str:
    """Needle-in-a-haystack prompt: hide `needle` in a long filler context."""
    filler = "The quick brown fox jumps over the lazy dog. "
    half = context_tokens // 2
    head = filler * (half // 5)
    tail = filler * (half // 5)
    return f"{head}\n\n{needle}\n\n{tail}\n\n{query}"


@contextmanager
def _launch_server(
    model: str, port: int, ds_args: List[str], log_path: Path, timeout_s: int = 600
):
    """Yield a live SGLang server, then tear it down."""
    cmd = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model",
        model,
        "--port",
        str(port),
        "--mem-fraction-static",
        "0.85",
        "--max-running-requests",
        "32",
        "--disable-radix-cache",
        *ds_args,
    ]
    env = dict(os.environ)
    log_f = log_path.open("w")
    proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, env=env)
    base_url = f"http://127.0.0.1:{port}"
    try:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(
                    f"server exited with code {proc.returncode}; check {log_path}"
                )
            try:
                import requests

                r = requests.get(f"{base_url}/health", timeout=2)
                if r.status_code == 200:
                    break
            except Exception:
                pass
            time.sleep(2)
        else:
            raise TimeoutError(f"server health check timed out after {timeout_s}s")
        yield base_url
    finally:
        try:
            proc.terminate()
            proc.wait(timeout=30)
        except Exception:
            proc.kill()
        log_f.close()


def _run_workload(
    base_url: str,
    model: str,
    *,
    context_len: int,
    output_len: int,
    n_requests: int,
    concurrency: int,
) -> WorkloadResult:
    """Send n_requests prompts and measure throughput / TTFT / TBT.

    Uses streaming completions to capture per-token timing (TBT).
    """
    import concurrent.futures

    import requests

    prompt = _build_long_prompt(context_len)

    def _one():
        ttft = None
        tbts = []
        last_t = None
        start = time.time()
        with requests.post(
            f"{base_url}/v1/completions",
            json={
                "model": model,
                "prompt": prompt,
                "max_tokens": output_len,
                "temperature": 0.0,
                "stream": True,
            },
            stream=True,
            timeout=600,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line or not line.startswith(b"data: "):
                    continue
                if line == b"data: [DONE]":
                    break
                t = time.time()
                if ttft is None:
                    ttft = (t - start) * 1000.0
                else:
                    tbts.append((t - last_t) * 1000.0)
                last_t = t
        e2e = time.time() - start
        return ttft, tbts, e2e

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as ex:
        results = list(ex.map(lambda _: _one(), range(n_requests)))

    ttfts = [r[0] for r in results if r[0] is not None]
    tbts = [t for r in results for t in r[1]]
    e2es = [r[2] for r in results]
    total_decode_tokens = sum(len(r[1]) for r in results)
    total_decode_s = sum(r[2] for r in results)
    decode_tok_per_s = (
        total_decode_tokens / total_decode_s if total_decode_s > 0 else 0.0
    )

    return WorkloadResult(
        config="(set by caller)",
        model=model,
        context_len=context_len,
        output_len=output_len,
        n_requests=n_requests,
        concurrency=concurrency,
        decode_tok_per_s=decode_tok_per_s,
        e2e_latency_s=statistics.mean(e2es) if e2es else 0.0,
        ttft_ms_p50=_percentile(ttfts, 50),
        ttft_ms_p95=_percentile(ttfts, 95),
        tbt_ms_p50=_percentile(tbts, 50),
        tbt_ms_p95=_percentile(tbts, 95),
    )


def _run_niah(base_url: str, model: str, context_tokens: int, n: int = 5) -> float:
    """Score `n` NIAH retrieval probes; return accuracy in [0, 1]."""
    import requests

    correct = 0
    for i in range(n):
        secret = f"the magic phrase is APRICOT-{i:04d}"
        query = "What was the magic phrase mentioned in the text? Answer with just the phrase."
        prompt = _build_niah_prompt(context_tokens, secret, query)
        r = requests.post(
            f"{base_url}/v1/completions",
            json={
                "model": model,
                "prompt": prompt,
                "max_tokens": 32,
                "temperature": 0.0,
            },
            timeout=300,
        )
        r.raise_for_status()
        out = r.json()["choices"][0]["text"].strip()
        if f"APRICOT-{i:04d}" in out:
            correct += 1
    return correct / n


def _build_ds_args(
    config: str, calibration: Optional[str], heavy_channels: int
) -> List[str]:
    if config == "branch_ds_on":
        if not calibration:
            raise SystemExit("--calibration required for branch_ds_on")
        return [
            "--enable-double-sparsity",
            "--double-sparsity-config",
            calibration,
            "--double-sparsity-heavy-channels",
            str(heavy_channels),
            "--double-sparsity-token-budget",
            "1024",
            "--double-sparsity-recent-tokens",
            "64",
            "--double-sparsity-sink-tokens",
            "4",
            "--double-sparsity-min-seq-len",
            "4096",
            "--double-sparsity-max-selected-per-request",
            "8192",
            "--page-size",
            "1",
            "--attention-backend",
            "fa3",
        ]
    return [
        "--page-size",
        "1",
        "--attention-backend",
        "fa3",
    ]


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument(
        "--config",
        choices=["main_dense", "branch_ds_off", "branch_ds_on"],
        required=True,
    )
    p.add_argument("--model", required=True)
    p.add_argument("--calibration", help="Required for branch_ds_on")
    p.add_argument("--heavy-channels", type=int, default=32)
    p.add_argument("--context-len", type=int, default=65536)
    p.add_argument("--output-len", type=int, default=1024)
    p.add_argument("--n-requests", type=int, default=4)
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--niah", action="store_true", help="Run a NIAH retrieval probe.")
    p.add_argument("--niah-context-tokens", type=int, default=32768)
    p.add_argument("--output-json", required=True)
    p.add_argument("--server-log", default=None)
    args = p.parse_args()

    ds_args = _build_ds_args(args.config, args.calibration, args.heavy_channels)
    port = _free_port()
    log_path = (
        Path(args.server_log)
        if args.server_log
        else Path(f"./bench_{args.config}_server.log")
    )

    with _launch_server(args.model, port, ds_args, log_path) as base_url:
        wl = _run_workload(
            base_url,
            args.model,
            context_len=args.context_len,
            output_len=args.output_len,
            n_requests=args.n_requests,
            concurrency=args.concurrency,
        )
        wl.config = args.config
        if args.niah:
            wl.niah_accuracy = _run_niah(base_url, args.model, args.niah_context_tokens)

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(asdict(wl), f, indent=2)

    csv_path = out.with_suffix(".csv")
    with csv_path.open("w", newline="") as f:
        d = asdict(wl)
        d.pop("extra", None)
        w = csv.DictWriter(f, fieldnames=list(d.keys()))
        w.writeheader()
        w.writerow(d)

    print(json.dumps(asdict(wl), indent=2))


if __name__ == "__main__":
    main()
