"""TTFT/ITL A/B/C: Python TM vs Rust TM (native MM) vs Rust TM (Python MM)
on an image workload.

Arms (aliases in parentheses):
  python                    — Python tokenizer manager
  rust (native)             — Rust TM + native Qwen MM pipeline
  rust_py_mm (py_mm)        — Rust TM, MM forced through Python mm_processor
                              (SGLANG_DISABLE_NATIVE_MM=1)
  rust_py_mm_proc (py_mm_proc) — like rust_py_mm, but the Python mm_processor
                              runs in a standalone process
                              (SGLANG_ENABLE_STANDALONE_MM=1)

Sweeps either concurrency (default) or image-count (`--image-counts`),
launching each selected arm once with identical server args. Outputs a
table, raw.json, and sweep.png (TTFT and ITL panels; mean solid, p99
dashed).

Partial re-runs keep the same raw.json shape: pass `--arms native` (or
`--arms rust`) and either reuse `--output-dir` (merges existing
raw.json) or `--merge-from <prior raw.json|dir>` so baselines stay on
the plot without re-running.

    # concurrency sweep (fixed image-count)
    python benchmark/rust_tokenizer_manager/bench_mm_ab.py --gpu 1 \
        --concurrencies 16 64 128 256 512 --num-prompts 1024 --image-count 4 --output-len 256

    # image-count sweep (fixed concurrency)
    python benchmark/rust_tokenizer_manager/bench_mm_ab.py --gpu 1 \
        --concurrencies 64 --num-prompts 1024 --image-counts 1 2 3 4 --output-len 256

    # re-run native rust only; keep python / rust_py_mm from a prior dir
    python benchmark/rust_tokenizer_manager/bench_mm_ab.py --gpu 1 \
        --arms native --merge-from results/0723_205224_nolog_conc \
        --output-dir results/0723_rerun_native

    # TTFT decomposition: concurrency 1, image-count sweep with text-only
    # floor (0 = random-ids); mm stage times land in the server logs
    python benchmark/rust_tokenizer_manager/bench_mm_ab.py --gpu 0 \
        --arms python rust py_mm py_mm_proc --concurrencies 1 \
        --image-counts 0 1 2 4 8 --num-prompts 32 --output-len 32

    # ITL vs throughput + scheduler GIL-steal per level (needs py-spy)
    python benchmark/rust_tokenizer_manager/bench_mm_ab.py --gpu 0 \
        --arms python rust py_mm py_mm_proc --concurrencies 1 4 16 64 256 \
        --image-count 1 --num-prompts 1024 --output-len 256 --gil-profile
"""

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

# Canonical arm name -> accepted CLI aliases (first alias is the canonical name).
ARM_ALIASES = {
    "python": "python",
    "rust": "rust",
    "native": "rust",
    "rust_native": "rust",
    "rust_py_mm": "rust_py_mm",
    "py_mm": "rust_py_mm",
    "rust_py_mm_proc": "rust_py_mm_proc",
    "py_mm_proc": "rust_py_mm_proc",
}
ARM_ORDER = ("python", "rust", "rust_py_mm", "rust_py_mm_proc")

# Prefer the workspace checkout over any installed sglang, here and in the
# launched server subprocesses (via PYTHONPATH).
_WORKSPACE_PY = Path(__file__).resolve().parents[2] / "python"
if (_WORKSPACE_PY / "sglang").is_dir():
    sys.path.insert(0, str(_WORKSPACE_PY))
    os.environ["PYTHONPATH"] = (
        f"{_WORKSPACE_PY}{os.pathsep}{os.environ.get('PYTHONPATH', '')}"
    )

import sglang.benchmark.serving as serving
from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import get_benchmark_args, popen_launch_server

METRICS = ("mean_ttft_ms", "p99_ttft_ms", "mean_itl_ms", "p99_itl_ms")

# Seeded dataset is identical across levels that share image_count, so build
# the largest size once per image_count and slice — avoids repeating minutes of
# client-side jpeg encoding + HF-processor token counting.
_dataset = {}
_orig_get_dataset = serving.get_dataset


def _cached_get_dataset(a, tokenizer, model_id=None):
    key = a.image_count if a.dataset_name == "image" else 0
    cache = _dataset.setdefault(key, {})
    n = a.num_prompts
    if "rows" not in cache:
        a.num_prompts = cache["build_n"] = _dataset["build_n"]
        cache["rows"] = _orig_get_dataset(a, tokenizer, model_id)
        a.num_prompts = n
    return cache["rows"][:n]


serving.get_dataset = _cached_get_dataset


def level_prompts(args, concurrency):
    """Enough requests for steady state at each level without serializing
    thousands of ~185ms requests at low concurrency. ITL needs a long steady
    window (default factor 16); TTFT-focused sweeps can pass
    --level-prompts-factor 1-2 (never below the concurrency itself, or the
    level would run at a lower effective concurrency than its label)."""
    return min(
        args.num_prompts,
        max(128, args.level_prompts_factor * concurrency, concurrency),
    )


def sweep_levels(args):
    """[(x_label_value, concurrency, image_count), ...]"""
    if args.image_counts:
        if len(args.concurrencies) != 1:
            raise SystemExit("--image-counts requires a single --concurrencies value")
        c = args.concurrencies[0]
        return [(n, c, n) for n in args.image_counts]
    return [(c, c, args.image_count) for c in args.concurrencies]


def bench_args(args, concurrency, image_count, out_dir, arm):
    ns = get_benchmark_args(
        base_url=f"http://127.0.0.1:{args.port}",
        backend="sglang",
        dataset_name=args.dataset_name,
        tokenizer=args.model,
        num_prompts=level_prompts(args, concurrency),
        random_input_len=args.input_len,
        random_output_len=args.output_len,
        max_concurrency=concurrency,
        seed=args.seed,
    )
    ns.model = args.model
    ns.image_count, ns.image_resolution = image_count, args.image_resolution
    ns.image_format, ns.image_content, ns.random_image_count = "jpeg", "random", False
    if image_count == 0:
        # Text-only floor level: same token counts, no mm at all — pins down
        # each arm's plumbing + prefill baseline so mm cost is separable.
        ns.dataset_name = "random-ids"
    ns.output_file = str(out_dir / f"{arm}.jsonl")
    ns.warmup_requests, ns.disable_tqdm = 3, True
    return ns


def arm_env(arm, args):
    """Server env for one A/B/C arm.

    - python:          Python tokenizer manager
    - rust:            Rust TM + native Qwen MM pipeline
    - rust_py_mm:      Rust TM, but MM forced onto Python mm_processor
                       (SGLANG_DISABLE_NATIVE_MM=1)
    - rust_py_mm_proc: rust_py_mm with the Python mm_processor hosted in a
                       standalone process (SGLANG_ENABLE_STANDALONE_MM=1)
    """
    rust = arm in ("rust", "rust_py_mm", "rust_py_mm_proc")
    py_mm = arm in ("rust_py_mm", "rust_py_mm_proc")
    env = {
        "SGLANG_RUST_SERVER": str(int(rust)),
        "SGLANG_VLM_CACHE_SIZE_MB": "0",
        "SGLANG_DISABLE_NATIVE_MM": str(int(py_mm)),
        "SGLANG_ENABLE_STANDALONE_MM": str(int(arm == "rust_py_mm_proc")),
        # Aggregated mm stage times in the server logs (TTFT decomposition).
        "SGLANG_LOG_MM_STAGE_INTERVAL": str(args.mm_stage_log_interval),
    }
    if args.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.gpu
    return env


def _scheduler_pid():
    """Find the scheduler by process title. NOT pgrep -f: that pattern-matches
    full cmdlines, including this script's own shell."""
    out = subprocess.run(
        ["ps", "-eo", "pid,comm"], capture_output=True, text=True
    ).stdout
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1].startswith("sglang::schedul"):
            return int(parts[0])
    return None


def _parse_gil_speedscope(path, duration, rate):
    """mm GIL steal = share of wall time a non-MainThread (mm work) holds the
    GIL; py-spy --gil records a sample only while someone holds it."""
    d = json.loads(Path(path).read_text())
    total = main = 0
    for prof in d["profiles"]:
        n = len(prof.get("samples", []))
        total += n
        if "MainThread" in prof["name"]:
            main += n
    possible = duration * rate
    return {
        "gil_wall_pct": round(total / possible * 100, 2),
        "main_wall_pct": round(main / possible * 100, 2),
        "mm_wall_pct": round((total - main) / possible * 100, 2),
    }


def profile_gil_during_level(server_log, out_path, holder, duration=30, rate=200):
    """Run on a side thread while the level's benchmark runs: wait until the
    server is actually decoding (the client-side dataset build leaves it idle
    for minutes — profiling that window measures nothing), then sample the
    scheduler's GIL holders with py-spy."""

    def count_decode():
        try:
            return sum(
                1 for l in open(server_log, errors="replace") if "Decode batch" in l
            )
        except FileNotFoundError:
            return 0

    base = count_decode()
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        if count_decode() >= base + 3:
            break
        time.sleep(2)
    else:
        print(f"gil profile: no decode traffic seen; skipping ({out_path.name})")
        return
    pid = _scheduler_pid()
    if pid is None:
        print("gil profile: scheduler process not found; skipping")
        return
    res = subprocess.run(
        [
            "py-spy",
            "record",
            "--pid",
            str(pid),
            "--gil",
            "--duration",
            str(duration),
            "--rate",
            str(rate),
            "--format",
            "speedscope",
            "--output",
            str(out_path),
        ],
        capture_output=True,
        text=True,
    )
    if res.returncode != 0 or not out_path.is_file():
        print(f"gil profile failed: {res.stderr.strip()[:200]}")
        return
    holder.update(_parse_gil_speedscope(out_path, duration, rate))


def run_arm(arm, args, out_dir, levels):
    # Wait for the previous arm's port to be released.
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", args.port)) != 0:
                break
        time.sleep(1)

    # The same seeded images are replayed at every level, so turn off the
    # vision-embedding LRU (--disable-radix-cache does not cover it); a hit
    # would skip the ViT forward and understate TTFT at later levels.
    env = arm_env(arm, args)
    log = open(out_dir / f"server_{arm}.log", "w")
    print(
        f"\n=== launching {arm} arm "
        f"(SGLANG_RUST_SERVER={env['SGLANG_RUST_SERVER']}, "
        f"SGLANG_DISABLE_NATIVE_MM={env['SGLANG_DISABLE_NATIVE_MM']}) ==="
    )
    proc = popen_launch_server(
        args.model,
        f"http://127.0.0.1:{args.port}",
        timeout=args.launch_timeout,
        other_args=["--disable-radix-cache"] + args.server_args,
        env=env,
        return_stdout_stderr=(log, log),
    )
    results = {}
    try:
        for x, concurrency, image_count in levels:
            print(
                f"\n--- [{arm}] x={x} concurrency={concurrency} image_count={image_count} ---"
            )
            gil, gil_thread = {}, None
            if args.gil_profile:
                gil_thread = threading.Thread(
                    target=profile_gil_during_level,
                    kwargs=dict(
                        server_log=out_dir / f"server_{arm}.log",
                        out_path=out_dir / f"gil_{arm}_x{x}.speedscope",
                        holder=gil,
                    ),
                    daemon=True,
                )
                gil_thread.start()
            try:
                res = serving.run_benchmark(
                    bench_args(args, concurrency, image_count, out_dir, arm)
                )
                results[x] = {
                    k: res[k] for k in METRICS + ("request_throughput", "completed")
                }
            except Exception as e:
                print(f"[{arm}] x={x} FAILED: {e}")
                results[x] = None
            if gil_thread is not None:
                gil_thread.join(60)
                if results[x] is not None and gil:
                    results[x]["gil"] = gil
                    print(f"[{arm}] x={x} gil: {gil}")
    finally:
        kill_process_tree(proc.pid)
        log.close()
    return results


def normalize_arm(name):
    key = name.lower().replace("-", "_")
    if key not in ARM_ALIASES:
        raise SystemExit(
            f"unknown arm {name!r}; choose from {', '.join(sorted(ARM_ALIASES))}"
        )
    return ARM_ALIASES[key]


def normalize_results(results):
    """JSON object keys are strings; coerce level keys back to int."""
    out = {}
    for arm, arm_results in results.items():
        if not isinstance(arm_results, dict):
            continue
        out[arm] = {int(x): (None if v is None else v) for x, v in arm_results.items()}
    return out


def load_raw_json(path):
    """Load raw.json from a file or a results directory."""
    p = Path(path)
    if p.is_dir():
        p = p / "raw.json"
    if not p.is_file():
        raise SystemExit(f"merge source not found: {p}")
    return normalize_results(json.loads(p.read_text()))


def merge_results(base, overlay):
    """Keep prior arms; overlay arms overwrite on conflict."""
    merged = {arm: dict(levels) for arm, levels in base.items()}
    for arm, levels in overlay.items():
        merged[arm] = dict(levels)
    return {arm: merged[arm] for arm in sorted(merged, key=_arm_sort_key)}


def _arm_sort_key(arm):
    try:
        return (0, ARM_ORDER.index(arm))
    except ValueError:
        return (1, arm)


ARM_COLORS = {
    "python": "tab:blue",
    "rust": "tab:orange",
    "rust_py_mm": "tab:green",
    "rust_py_mm_proc": "tab:red",
}


def plot_itl_vs_throughput(results, args, out_dir):
    """Latency-throughput curves: at equal delivered req/s, which arm has the
    lower ITL. Removes the operating-point confound of comparing arms at equal
    concurrency (a slow-admission arm runs emptier batches and flatters its
    ITL). Points are annotated with the concurrency that produced them."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, stat in zip(axes, ("mean", "p99")):
        for arm in sorted(results, key=_arm_sort_key):
            pts = [
                (r["request_throughput"], r[f"{stat}_itl_ms"], c)
                for c, r in sorted(results[arm].items())
                if r
            ]
            if not pts:
                continue
            color = ARM_COLORS.get(arm, "tab:gray")
            xs, ys, _ = zip(*pts)
            ax.plot(xs, ys, "-o", color=color, label=arm)
            for x, y, c in pts:
                ax.annotate(
                    str(c),
                    (x, y),
                    xytext=(0, 5),
                    fontsize=7,
                    textcoords="offset points",
                    ha="center",
                    color=color,
                )
        ax.set_yscale("log")
        ax.set(
            xlabel="request throughput (req/s)",
            ylabel="ms",
            title=f"{stat} ITL vs throughput",
        )
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(
        f"{args.model} — {args.image_count} image/req, "
        f"in={args.input_len} out={args.output_len} (labels = concurrency)"
    )
    fig.tight_layout()
    fig.savefig(out_dir / "itl_vs_throughput.png", dpi=120)


def plot_gil(results, args, out_dir):
    """Scheduler-process GIL steal by mm threads vs delivered throughput —
    the direct mechanism plot for 'in-process python mm hurts the scheduler'."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    any_pts = False
    for arm in sorted(results, key=_arm_sort_key):
        pts = [
            (r["request_throughput"], r["gil"]["mm_wall_pct"], c)
            for c, r in sorted(results[arm].items())
            if r and r.get("gil")
        ]
        if not pts:
            continue
        any_pts = True
        color = ARM_COLORS.get(arm, "tab:gray")
        xs, ys, _ = zip(*pts)
        ax.plot(xs, ys, "-o", color=color, label=arm)
        for x, y, c in pts:
            ax.annotate(
                str(c),
                (x, y),
                xytext=(0, 5),
                fontsize=7,
                textcoords="offset points",
                ha="center",
                color=color,
            )
    if not any_pts:
        plt.close(fig)
        return False
    ax.set(
        xlabel="request throughput (req/s)",
        ylabel="mm-thread GIL hold (% of wall time)",
        title="scheduler-process GIL steal by mm threads",
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.suptitle(f"{args.model} — {args.image_count} image/req (labels = concurrency)")
    fig.tight_layout()
    fig.savefig(out_dir / "gil_steal.png", dpi=120)
    return True


def plot(results, args, out_dir, levels):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = [lv[0] for lv in levels]
    xlabel = "image count" if args.image_counts else "max concurrency"
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    colors = ARM_COLORS
    xpos = {x: i for i, x in enumerate(xs)}
    for ax, metric in zip(axes, ("ttft", "itl")):
        for arm in sorted(results, key=_arm_sort_key):
            arm_results = results[arm]
            color = colors.get(arm, "tab:gray")
            for stat, style in (("mean", "-o"), ("p99", "--s")):
                pts = [
                    (xpos[x], r[f"{stat}_{metric}_ms"])
                    for x, r in sorted(arm_results.items())
                    if r and x in xpos
                ]
                if pts:
                    ax.plot(*zip(*pts), style, color=color, label=f"{arm} {stat}")
                    for px, y in pts:
                        ax.annotate(
                            f"{y:.1f}",
                            (px, y),
                            xytext=(0, 5),
                            fontsize=7,
                            textcoords="offset points",
                            ha="center",
                            color=color,
                        )
        ax.set_xticks(list(xpos.values()), [str(x) for x in xs])
        ax.set_yscale("log")
        ax.set(xlabel=xlabel, ylabel="ms", title=metric.upper())
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
    if args.image_counts:
        workload = f"concurrency={args.concurrencies[0]}, {args.image_resolution}"
    else:
        workload = (
            f"{args.image_count}x{args.image_resolution} image"
            if args.dataset_name == "image"
            else "text-only"
        )
    fig.suptitle(
        f"{args.model} — {workload}, "
        f"in={args.input_len} out={args.output_len}, n={args.num_prompts}"
    )
    fig.tight_layout()
    fig.savefig(out_dir / "sweep.png", dpi=120)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument(
        "--arms",
        nargs="+",
        default=["python", "rust", "rust_py_mm"],
        metavar="ARM",
        help="which arms to launch (default: all). Aliases: native/rust_native→rust, "
        "py_mm→rust_py_mm, py_mm_proc→rust_py_mm_proc. Prior arms can be kept via "
        "--merge-from / existing output-dir raw.json so the plot still shows the "
        "full A/B/C set.",
    )
    parser.add_argument(
        "--merge-from",
        default=None,
        help="prior raw.json (or results dir containing it) to merge before plot; "
        "also auto-merges raw.json already in --output-dir",
    )
    parser.add_argument(
        "--concurrencies",
        type=int,
        nargs="+",
        default=[1, 4, 16, 64, 128, 256, 512, 1024],
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=2048,
        help="max requests per level; low-concurrency levels use "
        "max(128, 16*concurrency) so they don't serialize for minutes",
    )
    parser.add_argument("--input-len", type=int, default=128)
    parser.add_argument("--output-len", type=int, default=64)
    parser.add_argument(
        "--level-prompts-factor",
        type=int,
        default=16,
        help="requests per level = max(128, factor*concurrency); "
        "use 1-2 for TTFT-only sweeps",
    )
    parser.add_argument(
        "--dataset-name",
        default="image",
        choices=["image", "random-ids"],
        help="'random-ids' = text-only baseline with the same token counts",
    )
    parser.add_argument(
        "--image-count",
        type=int,
        default=1,
        help="images per request (concurrency sweep)",
    )
    parser.add_argument(
        "--image-counts",
        type=int,
        nargs="+",
        default=None,
        help="sweep these image counts (x-axis); requires a "
        "single --concurrencies value",
    )
    parser.add_argument("--image-resolution", default="720p")
    parser.add_argument(
        "--gil-profile",
        action="store_true",
        help="py-spy --gil the scheduler during each level; adds per-level "
        "mm-thread GIL-steal%% to raw.json and a gil_steal.png plot",
    )
    parser.add_argument(
        "--mm-stage-log-interval",
        type=int,
        default=20,
        help="SGLANG_LOG_MM_STAGE_INTERVAL for the servers (0 disables): "
        "aggregated mm stage times in server logs every N mm requests",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--port", type=int, default=31300)
    parser.add_argument("--gpu", default=None, help="CUDA_VISIBLE_DEVICES value")
    parser.add_argument("--launch-timeout", type=float, default=1800)
    parser.add_argument(
        "--server-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="extra args passed through to sglang serve",
    )
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    args.arms = list(dict.fromkeys(normalize_arm(a) for a in args.arms))
    if args.gil_profile and not shutil.which("py-spy"):
        raise SystemExit("--gil-profile requires py-spy (pip install py-spy)")

    levels = sweep_levels(args)
    out_dir = Path(
        args.output_dir
        or Path(__file__).parent / "results" / datetime.now().strftime("%m%d_%H%M%S")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"output dir: {out_dir}")
    print(f"arms to run: {', '.join(args.arms)}")
    _dataset["build_n"] = max(level_prompts(args, c) for _, c, _ in levels)

    base = {}
    if args.merge_from:
        base = merge_results(base, load_raw_json(args.merge_from))
        print(f"merged from: {args.merge_from} (arms: {', '.join(base) or 'none'})")
    existing = out_dir / "raw.json"
    if existing.is_file():
        base = merge_results(base, load_raw_json(existing))
        print(f"merged existing: {existing} (arms: {', '.join(base) or 'none'})")

    ran = {arm: run_arm(arm, args, out_dir, levels) for arm in args.arms}
    results = merge_results(base, ran)
    (out_dir / "raw.json").write_text(json.dumps(results, indent=2))

    xname = "imgs" if args.image_counts else "conc"
    for arm in sorted(results, key=_arm_sort_key):
        arm_results = results[arm]
        tag = " (ran)" if arm in ran else " (merged)"
        print(
            f"\n[{arm}]{tag}  {xname:>6}  mean_ttft   p99_ttft   mean_itl    p99_itl  req/s"
        )
        for x, r in sorted(arm_results.items()):
            if r:
                vals = "".join(f"{r[k]:>11.1f}" for k in METRICS)
                print(f"  {x:>6}{vals}{r['request_throughput']:>7.2f}")
            else:
                print(f"  {x:>6}  FAILED")
    plot(results, args, out_dir, levels)
    saved = [f"{out_dir}/raw.json", f"{out_dir}/sweep.png"]
    if not args.image_counts:
        plot_itl_vs_throughput(results, args, out_dir)
        saved.append(f"{out_dir}/itl_vs_throughput.png")
        if plot_gil(results, args, out_dir):
            saved.append(f"{out_dir}/gil_steal.png")
    print(f"\nsaved: {', '.join(saved)}")
    print(f"arms in plot: {', '.join(sorted(results, key=_arm_sort_key))}")


if __name__ == "__main__":
    main()
