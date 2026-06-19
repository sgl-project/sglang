#!/usr/bin/env python3
"""Double Sparsity performance eval — a thin wrapper over the stock
``sglang.bench_serving`` client.

Runs the fixed concurrency-64 generated-shared-prefix workload (4096 ISL /
512 OSL, ~55% shared prefix), one trial, and reports the two client-visible
numbers Double Sparsity is gated on:

* **p50 decode TPS** — the median over requests of
  ``(output_tokens - 1) / decode_duration`` where
  ``decode_duration = last_token_time - first_token_time`` (the sum of a
  request's inter-token latencies). This is the steady-state decode rate a
  single request sees, independent of the prefill TTFT.
* **P99 TTFT** — the 99th-percentile time-to-first-token.

The benchmark itself is stock ``bench_serving`` (no window flags ported); the
wrapper only pins the workload arguments and derives p50 decode TPS from the
per-request detail that ``--output-details`` already emits.

Usage (against an already-running DS server):

    python benchmarks/bench_double_sparsity.py \
        --model <served-model-path> --host 127.0.0.1 --port 30000 \
        --num-prompts 256 --seed 42

Exit code is 0 iff p50 decode TPS and P99 TTFT are both within the parity band.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

# Validated conc-64 reference (regression gate, NOT a 30-TPS SLO).
REF_DECODE_TPS = 26.9
REF_P99_TTFT_S = 25.1
# Parity band: decode TPS >= -10% of reference AND P99 TTFT <= +20% of reference.
DECODE_TPS_FLOOR = 0.90 * REF_DECODE_TPS  # ~24.2
P99_TTFT_CEIL_S = 1.20 * REF_P99_TTFT_S  # ~30.1

# Fixed conc-64 workload. The validated candidate ran a SINGLE shared-prefix
# group whose prompts-per-group equals the prompt count, so every request shares
# the one system prompt (the prefix-reuse the workload is built to exercise). The
# stock generated-shared-prefix dataset otherwise defaults to 64 groups x 16
# prompts and ignores --num-prompts, which is a different request shape — so the
# group count and prompts-per-group are pinned explicitly below.
GSP_SYSTEM_PROMPT_LEN = 2253
GSP_QUESTION_LEN = 1843  # ISL ~4096, system prompt is ~55% of each input
GSP_OUTPUT_LEN = 512
GSP_RANGE_RATIO = 1.0
GSP_NUM_GROUPS = 1
MAX_CONCURRENCY = 64


def build_bench_cmd(args, output_file: str) -> list[str]:
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
        "--model",
        args.model,
        "--dataset-name",
        "generated-shared-prefix",
        "--gsp-system-prompt-len",
        str(GSP_SYSTEM_PROMPT_LEN),
        "--gsp-question-len",
        str(GSP_QUESTION_LEN),
        "--gsp-output-len",
        str(GSP_OUTPUT_LEN),
        "--gsp-range-ratio",
        str(GSP_RANGE_RATIO),
        "--gsp-num-groups",
        str(GSP_NUM_GROUPS),
        "--gsp-prompts-per-group",
        str(args.num_prompts),
        "--max-concurrency",
        str(MAX_CONCURRENCY),
        "--num-prompts",
        str(args.num_prompts),
        "--seed",
        str(args.seed),
        "--output-details",
        "--output-file",
        output_file,
    ]


def p50_decode_tps(output_lens, itls) -> float:
    """Median over requests of (output_tokens - 1) / sum(inter-token-latencies).

    Skips requests that produced <= 1 token or recorded no decode latency (a
    one-token reply has no decode window to measure).
    """
    per_req = []
    for n, itl in zip(output_lens, itls):
        decode_duration = float(sum(itl)) if itl else 0.0
        if n is None or n <= 1 or decode_duration <= 0.0:
            continue
        per_req.append((n - 1) / decode_duration)
    if not per_req:
        raise ValueError("no request had a measurable decode window")
    return float(np.median(per_req))


def parse_result(output_file: str) -> dict:
    # bench_serving appends one JSON object per run; take the last line.
    with open(output_file) as f:
        lines = [ln for ln in f if ln.strip()]
    if not lines:
        raise ValueError(f"bench_serving wrote no result to {output_file}")
    return json.loads(lines[-1])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="Served model path/id.")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=30000)
    ap.add_argument("--num-prompts", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--evidence-dir",
        default=None,
        help="Directory to copy the bench JSON + a verdict file into.",
    )
    args = ap.parse_args()

    output_file = tempfile.NamedTemporaryFile(
        prefix="ds_bench_", suffix=".jsonl", delete=False
    ).name
    cmd = build_bench_cmd(args, output_file)
    print(">>> " + " ".join(cmd), flush=True)
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        print(f"bench_serving exited {proc.returncode}", file=sys.stderr)
        return proc.returncode

    result = parse_result(output_file)
    decode_tps = p50_decode_tps(result["output_lens"], result["itls"])
    # Prefer bench_serving's own p99 ttft (ms -> s); fall back to the array.
    if "p99_ttft_ms" in result:
        p99_ttft_s = float(result["p99_ttft_ms"]) / 1000.0
    else:
        p99_ttft_s = float(np.percentile(result["ttfts"], 99))

    # Guard the request shape: the single shared-prefix group must have produced
    # exactly num_prompts requests. A mismatch means the dataset grouping drifted
    # (e.g. the stock 64x16 default), so the numbers would not reflect the conc-64
    # workload — fail closed rather than report a result on the wrong shape.
    actual_completed = result.get("completed")
    if actual_completed is None:
        actual_completed = len(result["output_lens"])
    shape_ok = actual_completed == args.num_prompts

    tps_ok = decode_tps >= DECODE_TPS_FLOOR
    ttft_ok = p99_ttft_s <= P99_TTFT_CEIL_S
    verdict = {
        "p50_decode_tps": round(decode_tps, 3),
        "p99_ttft_s": round(p99_ttft_s, 3),
        "decode_tps_floor": round(DECODE_TPS_FLOOR, 3),
        "p99_ttft_ceil_s": round(P99_TTFT_CEIL_S, 3),
        "reference_decode_tps": REF_DECODE_TPS,
        "reference_p99_ttft_s": REF_P99_TTFT_S,
        "tps_within_band": tps_ok,
        "ttft_within_band": ttft_ok,
        "gsp_num_groups": GSP_NUM_GROUPS,
        "gsp_prompts_per_group": args.num_prompts,
        "expected_prompts": args.num_prompts,
        "actual_completed": actual_completed,
        "request_shape_ok": shape_ok,
        "parity": tps_ok and ttft_ok and shape_ok,
        "num_prompts": args.num_prompts,
        "seed": args.seed,
        "bench_json": output_file,
    }
    print(json.dumps(verdict, indent=2), flush=True)

    if args.evidence_dir:
        os.makedirs(args.evidence_dir, exist_ok=True)
        with open(os.path.join(args.evidence_dir, "verdict.json"), "w") as f:
            json.dump(verdict, f, indent=2)
        with open(os.path.join(args.evidence_dir, "bench_result.json"), "w") as f:
            json.dump(result, f)

    if not shape_ok:
        print(
            f"FAIL (wrong workload shape): completed {actual_completed} requests, "
            f"expected {args.num_prompts} (1 shared-prefix group). The result does "
            f"not reflect the conc-64 workload.",
            file=sys.stderr,
            flush=True,
        )
        return 1
    if verdict["parity"]:
        print(
            f"PASS: {actual_completed} reqs (1 group); p50 decode TPS "
            f"{decode_tps:.2f} >= {DECODE_TPS_FLOOR:.1f} and P99 TTFT "
            f"{p99_ttft_s:.2f}s <= {P99_TTFT_CEIL_S:.1f}s",
            flush=True,
        )
        return 0
    print(
        f"FAIL (regression): p50 decode TPS {decode_tps:.2f} "
        f"(floor {DECODE_TPS_FLOOR:.1f}); P99 TTFT {p99_ttft_s:.2f}s "
        f"(ceil {P99_TTFT_CEIL_S:.1f}s)",
        file=sys.stderr,
        flush=True,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
