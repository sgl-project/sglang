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
    """Find an unused TCP port in 30000-49999 — must be <= 55535 because
    SGLang adds +10000 for the gRPC port and validates the sum <= 65535."""
    import random

    rng = random.Random()
    for _ in range(50):
        port = rng.randint(30000, 49999)
        s = socket.socket()
        try:
            s.bind(("", port))
        except OSError:
            continue
        finally:
            s.close()
        return port
    raise RuntimeError("could not find a free TCP port in 30000-49999")


@dataclass
class WorkloadResult:
    config: str
    model: str
    context_len: int
    output_len: int
    n_requests: int
    concurrency: int
    # Steady-state decode throughput per request. Numerator: tokens emitted
    # AFTER the first (i.e. one per inter-token interval). Denominator: per-
    # request decode-only time (e2e - ttft). For concurrency=1 this is the
    # most direct measurement of the selection+attention kernel speed.
    decode_tok_per_s: float
    # Aggregate system throughput: total tokens emitted across all requests
    # divided by wall-clock from first-request-start to last-request-end.
    # Captures concurrency benefits the per-request metric misses.
    aggregate_tok_per_s: float
    e2e_latency_s: float
    ttft_ms_p50: float
    ttft_ms_p95: float
    tbt_ms_p50: float
    tbt_ms_p95: float
    niah_accuracy: Optional[float] = None
    block_t: Optional[int] = None
    k_block: Optional[int] = None
    # Self-describing calibration tag, derived from the calibration JSON's
    # `calibration.dataset` field. README-cited result JSONs must satisfy
    # `calibration_mode != "synthetic"` (per plan ship-gate).
    calibration_mode: Optional[str] = None
    extra: Dict[str, float] = field(default_factory=dict)


def _percentile(xs, p):
    if not xs:
        return 0.0
    if len(xs) == 1:
        return float(xs[0])
    return float(statistics.quantiles(sorted(xs), n=100, method="inclusive")[p - 1])


def _build_long_prompt(target_tokens: int, fill: str = "The quick brown fox. ") -> str:
    """Approx. target_tokens long prompt — server tokenizer trims to its own count."""
    repeat = max(target_tokens // 5, 1)
    return fill * repeat


def _build_niah_prompt(context_tokens: int, needle: str, query: str) -> str:
    """Needle-in-a-haystack prompt: hide `needle` in a long filler context.

    Each filler repetition is ~10 Llama BPE tokens; divide by 10 (not 5) so
    head+tail tokens approximate `context_tokens` rather than overshoot by ~2x.
    """
    filler = "The quick brown fox jumps over the lazy dog. "
    half = context_tokens // 2
    head = filler * (half // 10)
    tail = filler * (half // 10)
    return f"{head}\n\n{needle}\n\n{tail}\n\n{query}"


@contextmanager
def _launch_server(
    model: str,
    port: int,
    ds_args: List[str],
    log_path: Path,
    context_length: int,
    *,
    tp_size: int = 1,
    mem_fraction_static: float = 0.85,
    max_running_requests: int = 32,
    timeout_s: int = 600,
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
        "--tp-size",
        str(tp_size),
        "--mem-fraction-static",
        str(mem_fraction_static),
        "--max-running-requests",
        str(max_running_requests),
        "--disable-radix-cache",
        "--context-length",
        str(context_length),
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
        tokens_emitted = 0
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
                tokens_emitted += 1
        end = time.time()
        e2e = end - start
        # Decode-only time = total wall - prefill (TTFT). Tokens emitted
        # during this window: tokens_emitted - 1 (the first one was the
        # TTFT-defining token; the rest came at TBT cadence).
        decode_only_s = (e2e - (ttft or 0.0) / 1000.0) if ttft is not None else 0.0
        return {
            "ttft": ttft,
            "tbts": tbts,
            "e2e": e2e,
            "start": start,
            "end": end,
            "tokens": tokens_emitted,
            "decode_only_s": decode_only_s,
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as ex:
        results = list(ex.map(lambda _: _one(), range(n_requests)))

    ttfts = [r["ttft"] for r in results if r["ttft"] is not None]
    tbts = [t for r in results for t in r["tbts"]]
    e2es = [r["e2e"] for r in results]

    # Per-request steady-state decode throughput: numerator counts tokens
    # emitted after the first (one per TBT interval), denominator is decode-
    # only time (e2e - ttft). Avoids the v1 mistake of dividing by total
    # request latency including prefill.
    total_decode_only_tokens = sum(max(r["tokens"] - 1, 0) for r in results)
    total_decode_only_s = sum(r["decode_only_s"] for r in results)
    decode_tok_per_s = (
        total_decode_only_tokens / total_decode_only_s
        if total_decode_only_s > 0
        else 0.0
    )

    # Aggregate system throughput: total tokens emitted across all requests
    # divided by wall-clock span. For concurrency>1 this captures overlap
    # benefits the per-request metric misses; for concurrency=1 it's a
    # full-stack number that includes prefill (lower than decode_tok_per_s).
    total_tokens = sum(r["tokens"] for r in results)
    wall_start = min(r["start"] for r in results) if results else 0.0
    wall_end = max(r["end"] for r in results) if results else 0.0
    wall_s = max(wall_end - wall_start, 1e-9)
    aggregate_tok_per_s = total_tokens / wall_s

    return WorkloadResult(
        config="(set by caller)",
        model=model,
        context_len=context_len,
        output_len=output_len,
        n_requests=n_requests,
        concurrency=concurrency,
        decode_tok_per_s=decode_tok_per_s,
        aggregate_tok_per_s=aggregate_tok_per_s,
        e2e_latency_s=statistics.mean(e2es) if e2es else 0.0,
        ttft_ms_p50=_percentile(ttfts, 50),
        ttft_ms_p95=_percentile(ttfts, 95),
        tbt_ms_p50=_percentile(tbts, 50),
        tbt_ms_p95=_percentile(tbts, 95),
    )


def _run_niah(
    base_url: str, model: str, context_tokens: int, n: int = 5
) -> float:
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
    config: str,
    calibration: Optional[str],
    heavy_channels: int,
    block_t: int,
    k_block: int,
    *,
    token_budget: int = 1024,
    recent_tokens: int = 64,
    sink_tokens: int = 4,
    min_seq_len: int = 4096,
    max_selected_per_request: int = 8192,
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
            str(token_budget),
            "--double-sparsity-recent-tokens",
            str(recent_tokens),
            "--double-sparsity-sink-tokens",
            str(sink_tokens),
            "--double-sparsity-min-seq-len",
            str(min_seq_len),
            "--double-sparsity-max-selected-per-request",
            str(max_selected_per_request),
            "--double-sparsity-block-t",
            str(block_t),
            "--double-sparsity-k-block",
            str(k_block),
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


def _calibration_mode(calibration_path: Optional[str]) -> Optional[str]:
    """Derive a self-describing calibration mode tag from the calibration JSON.

    Returns `"synthetic"` iff the file's `calibration.dataset` field is
    literally `"synthetic"`. Otherwise returns the dataset name or
    prompts-file path as written into the JSON by scripts/double_sparsity/
    calibrate.py:225-228. This is the field a README-cited result must
    check via `extra.calibration_mode != "synthetic"` (per plan).
    """
    if not calibration_path:
        return None
    try:
        with open(calibration_path, "r", encoding="utf-8") as f:
            blob = json.load(f)
        return blob.get("calibration", {}).get("dataset")
    except Exception:
        return "unknown"


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
    p.add_argument(
        "--block-t",
        type=int,
        default=1024,
        choices=[256, 512, 1024, 2048],
        help="DS stage-1 BLOCK_T (only used by branch_ds_on).",
    )
    p.add_argument(
        "--k-block",
        type=int,
        default=64,
        choices=[16, 32, 64, 128, 256],
        help="DS stage-1 K_BLOCK (only used by branch_ds_on).",
    )
    # Server-launch knobs (apply to all configs).
    p.add_argument(
        "--tp-size",
        type=int,
        default=1,
        help="Tensor-parallel size. 70B target uses 8.",
    )
    p.add_argument(
        "--mem-fraction-static",
        type=float,
        default=0.85,
        help="Server --mem-fraction-static. Tune for K_label overhead at long context.",
    )
    p.add_argument(
        "--max-running-requests",
        type=int,
        default=32,
        help="Server --max-running-requests. Decode batch admission cap.",
    )
    # DS tuning sweep knobs (only used by branch_ds_on).
    p.add_argument(
        "--token-budget",
        type=int,
        default=1024,
        help="--double-sparsity-token-budget (branch_ds_on only).",
    )
    p.add_argument(
        "--recent-tokens",
        type=int,
        default=64,
        help="--double-sparsity-recent-tokens (branch_ds_on only).",
    )
    p.add_argument(
        "--sink-tokens",
        type=int,
        default=4,
        help="--double-sparsity-sink-tokens (branch_ds_on only).",
    )
    p.add_argument(
        "--min-seq-len",
        type=int,
        default=4096,
        help="--double-sparsity-min-seq-len (branch_ds_on only).",
    )
    p.add_argument(
        "--max-selected-per-request",
        type=int,
        default=8192,
        help="--double-sparsity-max-selected-per-request (branch_ds_on only).",
    )
    p.add_argument(
        "--niah-n-samples",
        type=int,
        default=5,
        help="Number of NIAH probes. Discovery=5; publishable=20+.",
    )
    args = p.parse_args()

    ds_args = _build_ds_args(
        args.config,
        args.calibration,
        args.heavy_channels,
        args.block_t,
        args.k_block,
        token_budget=args.token_budget,
        recent_tokens=args.recent_tokens,
        sink_tokens=args.sink_tokens,
        min_seq_len=args.min_seq_len,
        max_selected_per_request=args.max_selected_per_request,
    )
    port = _free_port()
    log_path = (
        Path(args.server_log)
        if args.server_log
        else Path(f"./bench_{args.config}_server.log")
    )

    # Server's max-context window. Add headroom for generation.
    server_ctx_len = args.context_len + args.output_len + 256
    with _launch_server(
        args.model,
        port,
        ds_args,
        log_path,
        context_length=server_ctx_len,
        tp_size=args.tp_size,
        mem_fraction_static=args.mem_fraction_static,
        max_running_requests=args.max_running_requests,
    ) as base_url:
        wl = _run_workload(
            base_url,
            args.model,
            context_len=args.context_len,
            output_len=args.output_len,
            n_requests=args.n_requests,
            concurrency=args.concurrency,
        )
        wl.config = args.config
        if args.config == "branch_ds_on":
            wl.block_t = args.block_t
            wl.k_block = args.k_block
        # Derive and stamp calibration_mode from the calibration JSON. For
        # branch_ds_off / main_dense it stays None.
        if args.config == "branch_ds_on":
            wl.calibration_mode = _calibration_mode(args.calibration)
        if args.niah:
            wl.niah_accuracy = _run_niah(
                base_url,
                args.model,
                args.niah_context_tokens,
                n=args.niah_n_samples,
            )

        # Numeric server/DS config goes into `extra` (typed Dict[str, float])
        # for the CSV writer. calibration_mode is a top-level string field
        # so a README cite-check can `extra.calibration_mode != "synthetic"`.
        wl.extra.update(
            {
                "tp_size": float(args.tp_size),
                "mem_fraction_static": float(args.mem_fraction_static),
                "max_running_requests": float(args.max_running_requests),
                "niah_n_samples": float(args.niah_n_samples),
            }
        )
        if args.config == "branch_ds_on":
            wl.extra["token_budget"] = float(args.token_budget)
            wl.extra["recent_tokens"] = float(args.recent_tokens)
            wl.extra["sink_tokens"] = float(args.sink_tokens)
            wl.extra["min_seq_len"] = float(args.min_seq_len)
            wl.extra["max_selected_per_request"] = float(args.max_selected_per_request)

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
