"""Mixed prefill/decode and repeated-prefix serving-boundary benchmark.

Sweeps a small set of workload shapes against a running SGLang server across an
increasing client-concurrency ladder and reports where the latency tail breaks
down. It is a thin driver on top of ``sglang.bench_serving``: each (workload,
phase, concurrency, rep) cell is one ``python -m sglang.bench_serving`` run, and
this script aggregates the per-cell results into a boundary table.

Workloads (prompt / output tokens, approximate):
  balanced_2k      ~2048 / 128   continuity baseline
  long_decode      ~1024 / 512   decode / memory-bandwidth stress
  long_prefill_8k  ~8192 / 64    prefill / KV-allocation boundary
  repeated_prefix  ~2048 / 128   shared-prefix reuse (RadixAttention)
  agentic_session  multi-turn    interleaved agent sessions with per-session
                                 ~3072-token prefixes, ~512-token turn suffixes,
                                 and tool-call gaps between turns (cache
                                 retention / eviction-pressure boundary)

Phases:
  scaling   requests issued as fast as accepted (``--request-rate inf``)
  ttft      paced arrivals so time-to-first-token attributes to prefill, not
            queueing (``--request-rate`` finite, default = concurrency N)

Reported per cell, aggregated as the mean over reps:
  - aggregate decode throughput (tok/s) across the N concurrent streams
  - p50 / p95 end-to-end wall latency
  - p50 / p95 time-to-first-token
  - p95 wall-latency multiplier vs N=1 (the tail-breakdown signal)
  - completed / failed request counts

Launch the server first (see README.md), then for example:

    python3 bench_boundary.py --model Qwen/Qwen2.5-7B-Instruct --port 30000

To compare radix-cache on vs off, run twice against two servers with distinct
``--server-config-label`` values pointing at the same ``--output-dir``.
"""

import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------------------
# Workload presets: name -> list of bench_serving CLI flags for `num_prompts`.
# The three length-controlled shapes use the `random` dataset; repeated_prefix
# uses `generated-shared-prefix` so a long identical prefix is actually reused
# across requests (RadixAttention can then hit it).
# ---------------------------------------------------------------------------
def _random_workload(input_len, output_len, num_prompts, range_ratio):
    return [
        "--dataset-name",
        "random",
        "--random-input-len",
        str(input_len),
        "--random-output-len",
        str(output_len),
        "--random-range-ratio",
        str(range_ratio),
        "--num-prompts",
        str(num_prompts),
    ]


def _repeated_prefix_workload(num_prompts, range_ratio):
    # total prompts = num_groups * prompts_per_group; keep groups small so the
    # shared prefix is reused many times per group. ~1920 + ~128 ~= 2048 prompt.
    num_groups = max(1, min(8, num_prompts))
    prompts_per_group = max(1, num_prompts // num_groups)
    return [
        "--dataset-name",
        "generated-shared-prefix",
        "--gsp-num-groups",
        str(num_groups),
        "--gsp-prompts-per-group",
        str(prompts_per_group),
        "--gsp-system-prompt-len",
        "1920",
        "--gsp-question-len",
        "128",
        "--gsp-output-len",
        "128",
        "--gsp-range-ratio",
        str(range_ratio),
    ]


def _agentic_session_workload(args, n):
    # Each gsp group is one agent session: a unique ~3k-token session prefix
    # (system prompt + scaffold), played as a multi-turn conversation where
    # every turn appends a ~512-token suffix (tool output) and pauses for a
    # tool-call gap before the next turn. Sessions interleave under the
    # concurrency ladder, so turn k+1 either hits the session prefix in cache
    # or re-prefills it after eviction -- the p95 TTFT spread across turns is
    # the retention signal. Session count auto-scales with N so cells stay
    # comparable in wall-clock; lengths can be overridden via
    # --extra-bench-args (later duplicate flags win).
    sessions = args.agentic_sessions if args.agentic_sessions > 0 else max(2 * n, 8)
    return [
        "--dataset-name",
        "generated-shared-prefix",
        "--gsp-num-groups",
        str(sessions),
        "--gsp-prompts-per-group",
        "1",
        "--gsp-num-turns",
        str(args.agentic_turns),
        "--gsp-system-prompt-len",
        "3072",
        "--gsp-question-len",
        "512",
        "--gsp-output-len",
        "256",
        "--gsp-range-ratio",
        str(args.random_range_ratio if args.random_range_ratio > 0 else 1.0),
        "--gsp-turn-gap-short-s",
        str(args.agentic_gap_short_s),
        "--gsp-turn-gap-long-s",
        str(args.agentic_gap_long_s),
        "--gsp-turn-gap-long-prob",
        str(args.agentic_gap_long_prob),
    ]


def workload_flags(name, args, n):
    if name == "balanced_2k":
        return _random_workload(2048, 128, args.num_prompts, args.random_range_ratio)
    if name == "long_decode":
        return _random_workload(1024, 512, args.num_prompts, args.random_range_ratio)
    if name == "long_prefill_8k":
        return _random_workload(8192, 64, args.num_prompts, args.random_range_ratio)
    if name == "repeated_prefix":
        return _repeated_prefix_workload(args.num_prompts, args.random_range_ratio)
    if name == "agentic_session":
        return _agentic_session_workload(args, n)
    raise ValueError(f"unknown workload: {name}")


def workload_backend(name):
    # agentic_session plays rounds sequentially with real assistant responses,
    # which bench_serving only supports on chat backends.
    return "sglang-oai-chat" if name == "agentic_session" else "sglang"


ALL_WORKLOADS = [
    "balanced_2k",
    "long_decode",
    "long_prefill_8k",
    "repeated_prefix",
    "agentic_session",
]
ALL_PHASES = ["scaling", "ttft"]
TAG_SEP = ":"


# ---------------------------------------------------------------------------
# Per-cell record parsed back from the bench_serving JSONL row.
# ---------------------------------------------------------------------------
@dataclass
class Cell:
    label: str
    phase: str
    workload: str
    n: int
    rep: int
    decode_tps: float
    p50_wall_ms: float
    p95_wall_ms: float
    p50_ttft_ms: float
    p95_ttft_ms: float
    completed: int
    failed: int


def _percentile_ms(values_s, q):
    vals = [v for v in values_s if v is not None and v > 0]
    if not vals:
        return float("nan")
    return float(np.percentile(vals, q)) * 1000.0


def parse_row(row):
    """Turn one bench_serving --output-details JSONL row into a Cell."""
    tag = row.get("tag") or ""
    parts = tag.split(TAG_SEP)
    if len(parts) != 5:
        return None  # not one of ours
    label, phase, workload, n_field, rep_field = parts
    n = int(n_field.lstrip("N"))
    rep = int(rep_field.lstrip("r"))

    ttfts = row.get("ttfts") or []
    itls = row.get("itls") or []
    errors = row.get("errors") or [None] * len(ttfts)

    # per-request e2e (s) = ttft + sum(inter-token latencies); skip failed
    e2e_s = []
    ttft_ok_s = []
    for i, ttft in enumerate(ttfts):
        if i < len(errors) and errors[i]:
            continue
        itl = itls[i] if i < len(itls) else []
        e2e_s.append(ttft + sum(itl))
        ttft_ok_s.append(ttft)

    num_requests = len(ttfts)
    completed = int(row.get("completed", len(e2e_s)))
    failed = max(0, num_requests - completed)

    return Cell(
        label=label,
        phase=phase,
        workload=workload,
        n=n,
        rep=rep,
        decode_tps=float(row.get("output_throughput", float("nan"))),
        p50_wall_ms=float(row.get("median_e2e_latency_ms", float("nan"))),
        p95_wall_ms=_percentile_ms(e2e_s, 95),
        p50_ttft_ms=float(row.get("median_ttft_ms", float("nan"))),
        p95_ttft_ms=_percentile_ms(ttft_ok_s, 95),
        completed=completed,
        failed=failed,
    )


# ---------------------------------------------------------------------------
# Cell aggregation (mean over reps) + p95 multiplier vs N=1.
# ---------------------------------------------------------------------------
@dataclass
class AggCell:
    label: str
    phase: str
    workload: str
    n: int
    reps: int
    decode_tps: float
    p50_wall_ms: float
    p95_wall_ms: float
    p50_ttft_ms: float
    p95_ttft_ms: float
    failed: int
    p95_wall_mult_vs_n1: float = field(default=float("nan"))


def _nanmean(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and np.isnan(x))]
    return float(np.mean(xs)) if xs else float("nan")


def aggregate(cells):
    by_cell = defaultdict(list)
    for c in cells:
        by_cell[(c.label, c.phase, c.workload, c.n)].append(c)

    aggs = {}
    for key, group in by_cell.items():
        label, phase, workload, n = key
        aggs[key] = AggCell(
            label=label,
            phase=phase,
            workload=workload,
            n=n,
            reps=len(group),
            decode_tps=_nanmean([g.decode_tps for g in group]),
            p50_wall_ms=_nanmean([g.p50_wall_ms for g in group]),
            p95_wall_ms=_nanmean([g.p95_wall_ms for g in group]),
            p50_ttft_ms=_nanmean([g.p50_ttft_ms for g in group]),
            p95_ttft_ms=_nanmean([g.p95_ttft_ms for g in group]),
            failed=sum(g.failed for g in group),
        )

    # p95 multiplier vs the N=1 baseline within the same (label, phase, workload)
    for (label, phase, workload, n), agg in aggs.items():
        base = aggs.get((label, phase, workload, 1))
        if base and base.p95_wall_ms and not np.isnan(base.p95_wall_ms):
            agg.p95_wall_mult_vs_n1 = agg.p95_wall_ms / base.p95_wall_ms
    return aggs


# ---------------------------------------------------------------------------
# Running cells.
# ---------------------------------------------------------------------------
def build_cmd(args, base_url, phase, workload, n, rep, raw_path):
    tag = TAG_SEP.join([args.server_config_label, phase, workload, f"N{n}", f"r{rep}"])
    if phase == "scaling":
        request_rate = "inf"
    else:  # ttft: paced; default rate = concurrency unless overridden
        request_rate = str(args.ttft_request_rate if args.ttft_request_rate > 0 else n)

    cmd = [
        sys.executable,
        "-m",
        "sglang.bench_serving",
        "--backend",
        workload_backend(workload),
        "--base-url",
        base_url,
        "--max-concurrency",
        str(n),
        "--request-rate",
        request_rate,
        "--output-details",
        "--output-file",
        raw_path,
        "--tag",
        tag,
        "--seed",
        str(args.seed + rep),
        "--disable-tqdm",
    ]
    if args.model:
        cmd += ["--model", args.model]
    if args.flush_cache:
        cmd += ["--flush-cache"]
    cmd += workload_flags(workload, args, n)
    if args.extra_bench_args:
        cmd += args.extra_bench_args.split()
    return tag, cmd


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model path/id (defaults to whatever the server is serving).",
    )
    p.add_argument("--host", type=str, default="127.0.0.1", help="Server host.")
    p.add_argument("--port", type=int, default=30000, help="Server port.")
    p.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Full server base URL; overrides --host/--port.",
    )
    p.add_argument(
        "--workloads",
        type=str,
        default=",".join(ALL_WORKLOADS),
        help="Comma-separated workloads.",
    )
    p.add_argument(
        "--phases",
        type=str,
        default=",".join(ALL_PHASES),
        help="Comma-separated phases (scaling,ttft).",
    )
    p.add_argument(
        "--concurrency",
        type=str,
        default="1,2,4,8,16,32",
        help="Comma-separated concurrency ladder. Include 1 for the p95-multiplier baseline.",
    )
    p.add_argument(
        "--reps",
        type=int,
        default=3,
        help="Repetitions per cell (aggregated as the mean).",
    )
    p.add_argument(
        "--num-prompts", type=int, default=200, help="Requests issued per cell."
    )
    p.add_argument(
        "--random-range-ratio",
        type=float,
        default=0.0,
        help="Length jitter for random/gsp datasets (0 = fixed lengths).",
    )
    p.add_argument(
        "--ttft-request-rate",
        type=float,
        default=0.0,
        help="Paced arrival rate (req/s) for the ttft phase; 0 = use concurrency N.",
    )
    p.add_argument(
        "--flush-cache",
        action="store_true",
        default=True,
        help="Flush the server cache before each cell (default on).",
    )
    p.add_argument(
        "--no-flush-cache",
        dest="flush_cache",
        action="store_false",
        help="Do not flush between cells.",
    )
    p.add_argument(
        "--agentic-sessions",
        type=int,
        default=0,
        help="agentic_session: number of concurrent agent sessions per cell; 0 = auto (2x concurrency, min 8).",
    )
    p.add_argument(
        "--agentic-turns",
        type=int,
        default=6,
        help="agentic_session: turns played per session.",
    )
    p.add_argument(
        "--agentic-gap-short-s",
        type=float,
        default=0.5,
        help="agentic_session: pause before a turn for a fast tool call (seconds).",
    )
    p.add_argument(
        "--agentic-gap-long-s",
        type=float,
        default=15.0,
        help="agentic_session: pause before a turn for a slow external tool call (seconds).",
    )
    p.add_argument(
        "--agentic-gap-long-prob",
        type=float,
        default=0.15,
        help="agentic_session: probability a turn's pause is the long gap.",
    )
    p.add_argument(
        "--server-config-label",
        type=str,
        default="default",
        help="Label for the server config (e.g. radix_on/radix_off); embedded in results so multiple configs can share one --output-dir.",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="./boundary_results",
        help="Directory for raw + summary outputs.",
    )
    p.add_argument(
        "--seed", type=int, default=42, help="Base RNG seed (offset by rep index)."
    )
    p.add_argument(
        "--extra-bench-args",
        type=str,
        default=None,
        help="Extra args appended verbatim to every sglang.bench_serving call.",
    )
    p.add_argument(
        "--dry-run", action="store_true", help="Print the planned commands and exit."
    )
    args = p.parse_args()

    workloads = [w.strip() for w in args.workloads.split(",") if w.strip()]
    phases = [ph.strip() for ph in args.phases.split(",") if ph.strip()]
    concurrency = [int(n) for n in args.concurrency.split(",") if n.strip()]
    for w in workloads:
        workload_flags(w, args, n=1)  # validate early
    for ph in phases:
        if ph not in ALL_PHASES:
            p.error(f"unknown phase: {ph}")

    os.makedirs(args.output_dir, exist_ok=True)
    base_url = args.base_url or f"http://{args.host}:{args.port}"
    raw_path = os.path.join(args.output_dir, f"raw_{args.server_config_label}.jsonl")
    # fresh raw file for this label so re-runs of the same config don't pile up
    if os.path.exists(raw_path) and not args.dry_run:
        os.remove(raw_path)

    plan = [
        (ph, w, n, rep)
        for ph in phases
        for w in workloads
        for n in concurrency
        for rep in range(args.reps)
    ]
    print(
        f"Planned cells: {len(plan)}  "
        f"(phases={phases} workloads={workloads} N={concurrency} reps={args.reps})"
    )
    print(f"Server: {base_url}   config-label: {args.server_config_label}\n")

    for i, (ph, w, n, rep) in enumerate(plan, 1):
        tag, cmd = build_cmd(args, base_url, ph, w, n, rep, raw_path)
        print(f"[{i}/{len(plan)}] {tag}")
        if args.dry_run:
            print("    " + " ".join(cmd))
            continue
        proc = subprocess.run(cmd)
        if proc.returncode != 0:
            print(f"    !! cell failed (exit {proc.returncode}); continuing")

    if args.dry_run:
        return

    report(args, raw_path)


# ---------------------------------------------------------------------------
# Reporting.
# ---------------------------------------------------------------------------
def report(args, raw_path):
    cells = []
    if os.path.exists(raw_path):
        with open(raw_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                cell = parse_row(json.loads(line))
                if cell is not None:
                    cells.append(cell)
    if not cells:
        print("\nNo parseable cells found; nothing to report.")
        return

    aggs = aggregate(cells)
    cells_path = os.path.join(
        args.output_dir, f"boundary_cells_{args.server_config_label}.jsonl"
    )
    summary_path = os.path.join(
        args.output_dir, f"boundary_summary_{args.server_config_label}.jsonl"
    )
    with open(cells_path, "w") as f:
        for c in cells:
            f.write(json.dumps(c.__dict__) + "\n")
    with open(summary_path, "w") as f:
        for agg in aggs.values():
            f.write(json.dumps(agg.__dict__) + "\n")

    header = (
        f"{'phase':8} {'workload':16} {'N':>4} "
        f"{'decode_tps':>11} {'p50_wall':>9} {'p95_wall':>9} {'p95xN1':>7} "
        f"{'p50_ttft':>9} {'p95_ttft':>9} {'fail':>5}"
    )
    bar = "=" * len(header)
    print("\n" + bar)
    print(f" Serving boundary results  (config: {args.server_config_label})")
    print(bar)
    print(header)
    print("-" * len(header))
    for key in sorted(aggs, key=lambda k: (k[1], k[2], k[3])):
        a = aggs[key]
        mult = (
            "" if np.isnan(a.p95_wall_mult_vs_n1) else f"{a.p95_wall_mult_vs_n1:.2f}x"
        )
        print(
            f"{a.phase:8} {a.workload:16} {a.n:>4} "
            f"{a.decode_tps:>11.1f} {a.p50_wall_ms:>9.1f} {a.p95_wall_ms:>9.1f} {mult:>7} "
            f"{a.p50_ttft_ms:>9.1f} {a.p95_ttft_ms:>9.1f} {a.failed:>5}"
        )
    print(bar)
    print(f"raw cells:   {cells_path}")
    print(f"summary:     {summary_path}")


if __name__ == "__main__":
    main()
