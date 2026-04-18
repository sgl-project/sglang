"""SLRU three-way benchmark compare runner.

Launches ``python -m sglang.launch_server`` under each of {LRU, legacy SLRU,
optimized SLRU}, waits for readiness, runs ``python -m sglang.bench_serving``
with a dataset-specific preset, collects metrics, shuts the server down, and
emits a markdown comparison table alongside the raw JSONL outputs.

Designed to run on a single GPU box (PAI-DSW A10, AutoDL 4090, RunPod A10, etc.)
without any manual orchestration — one command, three runs, one report.

Usage:

    python benchmark/slru/run_bench_compare.py \\
        --model Qwen/Qwen2.5-7B-Instruct \\
        --dataset sharegpt \\
        --output-dir results/

    # Or pick a different dataset preset:
    python benchmark/slru/run_bench_compare.py \\
        --model Qwen/Qwen2.5-7B-Instruct \\
        --dataset gsp \\
        --output-dir results/

Outputs (in ``<output-dir>/<timestamp>/``):
  * ``compare.md``    — side-by-side table with deltas vs LRU baseline
  * ``<policy>.jsonl``— raw bench_serving output per policy
  * ``<policy>.log``  — stdout/stderr of the server + bench for each run
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import shlex
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

try:
    from sglang.srt.utils.common import kill_process_tree
except ImportError:  # sglang not installed (e.g. running --help in a fresh env)
    kill_process_tree = None  # type: ignore[assignment]

# -----------------------------------------------------------------------------
# Dataset presets. Tune via CLI overrides when adapting the stress profile to a
# new model.
# -----------------------------------------------------------------------------

DATASET_PRESETS: Dict[str, Dict[str, List[str]]] = {
    "sharegpt": {
        "bench_args": [
            "--dataset-name",
            "sharegpt",
            "--num-prompts",
            "1000",
            "--request-rate",
            "10",
        ],
        # Required dataset file — bench_serving expects a ShareGPT dump.
        # Users pass it via --sharegpt-path (we forward through).
    },
    "gsp": {
        "bench_args": [
            "--dataset-name",
            "generated-shared-prefix",
            "--gsp-num-groups",
            "4",
            "--gsp-prompts-per-group",
            "256",
            "--gsp-system-prompt-len",
            "2048",
            "--num-prompts",
            "1024",
            "--request-rate",
            "8",
        ],
    },
    "longbench": {
        "bench_args": [
            "--dataset-name",
            "longbench_v2",
            "--num-prompts",
            "500",
            "--request-rate",
            "5",
        ],
    },
    "random": {
        # Smoke-test preset: no dataset file needed, synthetic prompts. Useful
        # for verifying the compare runner end-to-end on a fresh GPU box
        # before committing to a full ShareGPT run.
        "bench_args": [
            "--dataset-name",
            "random",
            "--random-input",
            "1024",
            "--random-output",
            "256",
            "--random-range-ratio",
            "0.5",
            "--num-prompts",
            "300",
            "--request-rate",
            "8",
        ],
    },
    "mooncake": {
        # Production trace from Moonshot AI's Kimi.ai service, released with
        # the Mooncake paper (FAST'25, "Mooncake: A KVCache-centric
        # Disaggregated Architecture for LLM Serving"). Drives the three
        # policies with a real prefix-reuse distribution: ``hash_id=0``
        # appears in 100% of requests (universal system prompt), ~24% of
        # remaining hash_ids are repeated across sessions (legitimate
        # hotness), and ~76% are one-shot (the "burst-false-hot" trap
        # optimized-slru is designed to filter out).
        #
        # ``--mooncake-num-rounds 2`` turns each trace record into a 2-turn
        # burst (chat_history grows across rounds). Two rounds is enough to
        # exercise the within-session prefix-reuse burst — the second round
        # arriving back-to-back on the same shared prefix is exactly the
        # debounce/cap trigger. Higher values saturate the single-4090
        # prefill budget; see load math below.
        #
        # ``--mooncake-slowdown-factor 3.0`` paces the replay at one third
        # real-trace speed. Native arrival rate on the conversation trace
        # is ~3 sessions/s; with num_rounds=2 and mean effective prefill
        # ~3,250 tokens per request, sustainable QPS on a 4090 is about
        # 2–2.5 req/s → slowdown=3 puts us at ~2 req/s, ~65% of the low
        # estimate of prefill capacity. Bump to 4.0 on older GPUs if
        # queue-req grows past 20 in the first 2 min of benching.
        #
        # ``--num-prompts 2000`` subsamples 2,000 of the 12,031 records,
        # giving ~4,000 generated requests once the 2-round expansion is
        # applied — ~40 samples at p99 TTFT, enough for a bootstrap CI.
        #
        # NOTE: the mooncake loader IGNORES --request-rate; pacing is
        # controlled entirely by the replay slowdown factor.
        "bench_args": [
            "--dataset-name",
            "mooncake",
            "--mooncake-workload",
            "conversation",
            "--mooncake-num-rounds",
            "2",
            "--mooncake-slowdown-factor",
            "3.0",
            "--num-prompts",
            "2000",
        ],
    },
}


@dataclasses.dataclass(frozen=True)
class Policy:
    """One eviction-policy variant to benchmark."""

    label: str  # shown in output table
    eviction_policy: str  # --radix-eviction-policy value
    optimization_enabled: bool  # toggles SGLANG_ENABLE_SLRU_OPTIMIZATION


POLICIES: List[Policy] = [
    Policy(label="lru", eviction_policy="lru", optimization_enabled=False),
    Policy(
        label="naive-slru",
        eviction_policy="slru",
        optimization_enabled=False,
    ),
    Policy(
        label="optimized-slru",
        eviction_policy="slru",
        optimization_enabled=True,
    ),
]


# -----------------------------------------------------------------------------
# Server lifecycle
# -----------------------------------------------------------------------------


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _wait_for_server_ready(
    host: str, port: int, timeout_sec: int, log_path: Path
) -> bool:
    """Poll ``/health`` until ready or timeout."""
    deadline = time.time() + timeout_sec
    url = f"http://{host}:{port}/health"
    last_err: Optional[str] = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, ConnectionError, socket.timeout) as e:
            last_err = repr(e)
        time.sleep(2)
    print(
        f"  [readiness] timed out after {timeout_sec}s; last error: {last_err}. "
        f"See {log_path} for server logs.",
        file=sys.stderr,
    )
    return False


def _launch_server(
    policy: Policy,
    model: str,
    port: int,
    log_path: Path,
    extra_server_args: List[str],
    slru_knobs: Dict[str, float],
) -> subprocess.Popen:
    """Spawn ``sglang.launch_server`` for the given policy.

    Returns the Popen handle so the caller can terminate it when the
    bench run finishes.
    """
    cmd = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        model,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--radix-eviction-policy",
        policy.eviction_policy,
    ]

    # Forward SLRU tuning knobs — only meaningful when policy is slru
    # AND optimization_enabled, but passing them for lru/legacy SLRU is a no-op.
    # at the server_args level and keeps the launch command uniform.
    if policy.eviction_policy == "slru":
        cmd += [
            "--slru-protected-threshold",
            str(int(slru_knobs["protected_threshold"])),
            "--slru-debounce-sec",
            str(slru_knobs["debounce_sec"]),
            "--slru-decay-sec",
            str(slru_knobs["decay_sec"]),
        ]

    cmd += extra_server_args

    env = os.environ.copy()
    env["SGLANG_ENABLE_SLRU_OPTIMIZATION"] = "1" if policy.optimization_enabled else "0"

    log_f = open(log_path, "w")
    log_f.write(f"# cmd: {shlex.join(cmd)}\n")
    log_f.write(
        f"# SGLANG_ENABLE_SLRU_OPTIMIZATION="
        f"{env['SGLANG_ENABLE_SLRU_OPTIMIZATION']}\n"
    )
    log_f.flush()

    # Start in its own process group so we can clean-kill children if the
    # server spawns workers.
    return subprocess.Popen(
        cmd,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        env=env,
        start_new_session=True,
    )


# The currently-running server, so the SIGINT/SIGTERM handler can hard-kill
# it before the interpreter exits. Without this, ^C during a bench leaves
# the server orphaned (PPID=1) holding all its GPU memory.
_current_server: Optional[subprocess.Popen] = None


def _install_signal_handlers() -> None:
    """Hard-kill the current server + descendants on SIGINT/SIGTERM before
    re-raising. The ``finally`` block in ``main`` may not run at all on a
    double-^C; this guarantees the GPU memory is always released."""

    def _handler(signum: int, frame) -> None:
        server = _current_server
        if server is not None and server.poll() is None:
            print(
                f"\n[compare] signal {signum} received; killing server tree "
                f"pid={server.pid}",
                file=sys.stderr,
            )
            try:
                if kill_process_tree is not None:
                    kill_process_tree(server.pid, include_parent=True)
                else:
                    os.killpg(os.getpgid(server.pid), signal.SIGKILL)
            except Exception as e:  # best-effort — never mask the signal
                print(f"[compare] kill failed: {e}", file=sys.stderr)
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)


def _shutdown_server(proc: subprocess.Popen, grace_sec: float = 20.0) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=grace_sec)
    except subprocess.TimeoutExpired:
        # psutil-based recursive kill catches scheduler/detokenizer children
        # that may have escaped the process group (SGLang spawns them via
        # multiprocessing with their own session).
        try:
            if kill_process_tree is not None:
                kill_process_tree(proc.pid, include_parent=True)
            else:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            pass
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass


# -----------------------------------------------------------------------------
# Benchmark execution
# -----------------------------------------------------------------------------


def _run_bench(
    dataset: str,
    host: str,
    port: int,
    output_file: Path,
    log_path: Path,
    model: str,
    extra_bench_args: List[str],
) -> Optional[dict]:
    """Run sglang.bench_serving and return the parsed result dict."""
    preset = DATASET_PRESETS[dataset]
    cmd = [
        sys.executable,
        "-m",
        "sglang.bench_serving",
        "--backend",
        "sglang",
        "--host",
        host,
        "--port",
        str(port),
        "--model",
        model,
        "--output-file",
        str(output_file),
        *preset["bench_args"],
        *extra_bench_args,
    ]

    with open(log_path, "a") as log_f:
        log_f.write(f"\n# bench cmd: {shlex.join(cmd)}\n")
        log_f.flush()
        rc = subprocess.call(cmd, stdout=log_f, stderr=subprocess.STDOUT)

    if rc != 0:
        print(
            f"  [bench] non-zero exit code {rc}; see {log_path}",
            file=sys.stderr,
        )
        return None

    # bench_serving appends one JSONL line per run — there will be exactly one.
    if not output_file.exists():
        print(f"  [bench] no output file at {output_file}", file=sys.stderr)
        return None
    with open(output_file) as f:
        lines = [json.loads(line) for line in f if line.strip()]
    if not lines:
        return None
    return lines[-1]


# -----------------------------------------------------------------------------
# Reporting
# -----------------------------------------------------------------------------

HEADLINE_KEYS = [
    ("median_ttft_ms", "median TTFT (ms)", "lower"),
    ("p99_ttft_ms", "p99 TTFT (ms)", "lower"),
    ("mean_e2e_latency_ms", "mean e2e (ms)", "lower"),
    ("output_throughput", "output tok/s", "higher"),
    ("request_throughput", "req/s", "higher"),
    ("median_itl_ms", "median ITL (ms)", "lower"),
    ("p99_itl_ms", "p99 ITL (ms)", "lower"),
]


def _format_delta(value: float, baseline: float, direction: str) -> str:
    if baseline == 0 or value is None or baseline is None:
        return "—"
    pct = (value - baseline) / baseline * 100.0
    # Color/sign convention: down-arrow is "good" for "lower", up-arrow for "higher".
    good = (direction == "lower" and pct < 0) or (direction == "higher" and pct > 0)
    marker = "↓" if pct < 0 else "↑"
    qualifier = " ✓" if good and abs(pct) >= 1 else ""
    return f"{pct:+.1f}% {marker}{qualifier}"


def _render_markdown(
    dataset: str,
    model: str,
    results: Dict[str, dict],
    extra_server_args: List[str],
    extra_bench_args: List[str],
    slru_knobs: Dict[str, float],
    timestamp: str,
) -> str:
    lines: List[str] = []
    lines.append(f"# SLRU benchmark compare — {dataset} — {timestamp}")
    lines.append("")
    lines.append(f"- **Model**: `{model}`")
    lines.append(
        f"- **Dataset**: `{dataset}` (preset: `{' '.join(DATASET_PRESETS[dataset]['bench_args'])}`)"
    )
    if extra_bench_args:
        lines.append(f"- **Extra bench args**: `{' '.join(extra_bench_args)}`")
    if extra_server_args:
        lines.append(f"- **Extra server args**: `{' '.join(extra_server_args)}`")
    lines.append(
        f"- **SLRU knobs**: threshold={int(slru_knobs['protected_threshold'])}, "
        f"debounce={slru_knobs['debounce_sec']}s, decay={slru_knobs['decay_sec']}s"
    )
    lines.append("")
    lines.append("## Headline metrics")
    lines.append("")

    header = (
        ["metric"]
        + [p.label for p in POLICIES]
        + [f"{p.label} Δ vs lru" for p in POLICIES if p.label != "lru"]
    )
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")

    lru_result = results.get("lru") or {}
    for key, display, direction in HEADLINE_KEYS:
        row = [display]
        for p in POLICIES:
            r = results.get(p.label) or {}
            v = r.get(key)
            row.append(f"{v:.2f}" if isinstance(v, (int, float)) else "—")
        # Delta columns (only for non-lru rows)
        baseline = lru_result.get(key)
        for p in POLICIES:
            if p.label == "lru":
                continue
            r = results.get(p.label) or {}
            v = r.get(key)
            row.append(
                _format_delta(v, baseline, direction)
                if isinstance(v, (int, float)) and isinstance(baseline, (int, float))
                else "—"
            )
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    lines.append("## Raw outputs")
    lines.append("")
    for p in POLICIES:
        lines.append(f"- `{p.label}.jsonl` — bench_serving record for {p.label}")
        lines.append(f"- `{p.label}.log` — combined server + bench stdout/stderr")
    lines.append("")
    lines.append(
        "> Interpretation: compare `optimized-slru` vs `naive-slru` on the same row — "
        "the optimization should improve (lower) TTFT percentiles and leave "
        "throughput within a few percent on ShareGPT (baseline regression guard), "
        "and produce a larger improvement on GSP (which has heavy prefix reuse)."
    )

    return "\n".join(lines) + "\n"


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main(argv: List[str]) -> int:
    _install_signal_handlers()
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model path/name for both the server and the bench client.",
    )
    parser.add_argument(
        "--dataset",
        default="sharegpt",
        choices=list(DATASET_PRESETS.keys()),
        help="Dataset preset to run.",
    )
    parser.add_argument(
        "--output-dir",
        default="benchmark/slru/results",
        help="Directory to write compare.md + per-policy raw outputs into.",
    )
    parser.add_argument(
        "--policies",
        default="lru,naive-slru,optimized-slru",
        help="Comma-separated subset of policies to benchmark. Useful for "
        "re-running just one variant. Default: all three.",
    )
    parser.add_argument(
        "--readiness-timeout-sec",
        type=int,
        default=600,
        help="Seconds to wait for the server to become /health-ready after "
        "launch. Default: 600 (first load pulls the model weights).",
    )
    parser.add_argument(
        "--server-arg",
        action="append",
        default=[],
        help="Extra argument forwarded to sglang.launch_server. Repeat as "
        "needed, e.g. --server-arg=--mem-fraction-static=0.8 "
        "--server-arg=--tp=1 .",
    )
    parser.add_argument(
        "--bench-arg",
        action="append",
        default=[],
        help="Extra argument forwarded to sglang.bench_serving. Repeat as "
        "needed, e.g. --bench-arg=--sharegpt-path=/data/sharegpt.json .",
    )
    parser.add_argument(
        "--slru-protected-threshold",
        type=int,
        default=2,
    )
    parser.add_argument("--slru-debounce-sec", type=float, default=0.1)
    parser.add_argument("--slru-decay-sec", type=float, default=60.0)
    args = parser.parse_args(argv)

    selected = [p for p in POLICIES if p.label in args.policies.split(",")]
    if not selected:
        print(f"No known policies in {args.policies!r}.", file=sys.stderr)
        return 2

    slru_knobs = {
        "protected_threshold": args.slru_protected_threshold,
        "debounce_sec": args.slru_debounce_sec,
        "decay_sec": args.slru_decay_sec,
    }

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.output_dir) / f"{args.dataset}-{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[compare] output dir: {out_dir}")

    results: Dict[str, dict] = {}
    for policy in selected:
        print(
            f"[compare] === {policy.label}  "
            f"(eviction={policy.eviction_policy}, "
            f"optimization={policy.optimization_enabled}) ==="
        )

        port = _find_free_port()
        server_log = out_dir / f"{policy.label}.log"
        bench_out = out_dir / f"{policy.label}.jsonl"

        server = _launch_server(
            policy=policy,
            model=args.model,
            port=port,
            log_path=server_log,
            extra_server_args=args.server_arg,
            slru_knobs=slru_knobs,
        )
        global _current_server
        _current_server = server
        print(f"  [server] pid={server.pid} port={port} log={server_log}")

        try:
            if not _wait_for_server_ready(
                "127.0.0.1", port, args.readiness_timeout_sec, server_log
            ):
                print(f"  [server] readiness check FAILED; skipping bench run")
                continue

            print(f"  [bench] writing output to {bench_out}")
            result = _run_bench(
                dataset=args.dataset,
                host="127.0.0.1",
                port=port,
                output_file=bench_out,
                log_path=server_log,
                model=args.model,
                extra_bench_args=args.bench_arg,
            )
            if result is None:
                print(f"  [bench] FAILED; see {server_log}")
                continue

            results[policy.label] = result
            print(
                f"  [bench] OK — "
                f"median_ttft_ms={result.get('median_ttft_ms'):.1f}, "
                f"p99_ttft_ms={result.get('p99_ttft_ms'):.1f}, "
                f"output_tok/s={result.get('output_throughput'):.1f}"
            )
        finally:
            print(f"  [server] shutting down pid={server.pid}")
            _shutdown_server(server)
            _current_server = None

    report = _render_markdown(
        dataset=args.dataset,
        model=args.model,
        results=results,
        extra_server_args=args.server_arg,
        extra_bench_args=args.bench_arg,
        slru_knobs=slru_knobs,
        timestamp=timestamp,
    )
    report_path = out_dir / "compare.md"
    report_path.write_text(report)
    print(f"[compare] wrote report: {report_path}")
    if len(results) < len(selected):
        print(
            f"[compare] WARNING — only {len(results)}/{len(selected)} runs "
            f"produced metrics; check per-policy .log files.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
