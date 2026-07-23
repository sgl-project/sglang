"""TTFT/ITL A/B: Python vs Rust tokenizer manager (SGLANG_RUST_SERVER)
on an image workload.

Sweeps either concurrency (default) or image-count (`--image-counts`),
launching each arm once with identical server args. Outputs a table,
raw.json, and sweep.png (TTFT and ITL panels; mean solid, p99 dashed).

    # concurrency sweep (fixed image-count)
    python benchmark/rust_tokenizer_manager/bench_mm_ab.py --gpu 1 \
        --concurrencies 16 64 128 256 512 --num-prompts 1024 --image-count 4 --output-len 256

    # image-count sweep (fixed concurrency)
    python benchmark/rust_tokenizer_manager/bench_mm_ab.py --gpu 1 \
        --concurrencies 64 --num-prompts 1024 --image-counts 1 2 3 4 --output-len 256
"""

import argparse
import json
import os
import socket
import sys
import time
from datetime import datetime
from pathlib import Path

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
    thousands of ~185ms requests at low concurrency."""
    return min(args.num_prompts, max(128, 16 * concurrency))


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
    ns.output_file = str(out_dir / f"{arm}.jsonl")
    ns.warmup_requests, ns.disable_tqdm = 3, True
    return ns


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
    env = {
        "SGLANG_RUST_SERVER": str(int(arm == "rust")),
        "SGLANG_VLM_CACHE_SIZE_MB": "0",
    }
    if args.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.gpu
    log = open(out_dir / f"server_{arm}.log", "w")
    print(f"\n=== launching {arm} arm (SGLANG_RUST_SERVER={env['SGLANG_RUST_SERVER']}) ===")
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
            print(f"\n--- [{arm}] x={x} concurrency={concurrency} image_count={image_count} ---")
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
    finally:
        kill_process_tree(proc.pid)
        log.close()
    return results


def plot(results, args, out_dir, levels):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = [lv[0] for lv in levels]
    xlabel = "image count" if args.image_counts else "max concurrency"
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    colors = {"python": "tab:blue", "rust": "tab:orange"}
    xpos = {x: i for i, x in enumerate(xs)}
    for ax, metric in zip(axes, ("ttft", "itl")):
        for arm, arm_results in results.items():
            for stat, style in (("mean", "-o"), ("p99", "--s")):
                pts = [
                    (xpos[x], r[f"{stat}_{metric}_ms"])
                    for x, r in sorted(arm_results.items())
                    if r
                ]
                if pts:
                    ax.plot(*zip(*pts), style, color=colors[arm], label=f"{arm} {stat}")
                    for px, y in pts:
                        ax.annotate(f"{y:.1f}", (px, y), xytext=(0, 5), fontsize=7,
                                    textcoords="offset points", ha="center",
                                    color=colors[arm])
        ax.set_xticks(list(xpos.values()), [str(x) for x in xs])
        ax.set_yscale("log")
        ax.set(xlabel=xlabel, ylabel="ms", title=metric.upper())
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
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
    parser.add_argument("--arms", nargs="+", default=["python", "rust"],
                        choices=["python", "rust"])
    parser.add_argument("--concurrencies", type=int, nargs="+",
                        default=[1, 4, 16, 64, 128, 256, 512, 1024])
    parser.add_argument(
        "--num-prompts", type=int, default=2048,
        help="max requests per level; low-concurrency levels use "
        "max(128, 16*concurrency) so they don't serialize for minutes",
    )
    parser.add_argument("--input-len", type=int, default=128)
    parser.add_argument("--output-len", type=int, default=64)
    parser.add_argument(
        "--dataset-name", default="image", choices=["image", "random-ids"],
        help="'random-ids' = text-only baseline with the same token counts",
    )
    parser.add_argument("--image-count", type=int, default=1,
                        help="images per request (concurrency sweep)")
    parser.add_argument("--image-counts", type=int, nargs="+", default=None,
                        help="sweep these image counts (x-axis); requires a "
                        "single --concurrencies value")
    parser.add_argument("--image-resolution", default="720p")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--port", type=int, default=31300)
    parser.add_argument("--gpu", default=None, help="CUDA_VISIBLE_DEVICES value")
    parser.add_argument("--launch-timeout", type=float, default=1800)
    parser.add_argument("--server-args", nargs=argparse.REMAINDER, default=[],
                        help="extra args passed through to sglang serve")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    levels = sweep_levels(args)
    out_dir = Path(
        args.output_dir
        or Path(__file__).parent / "results" / datetime.now().strftime("%m%d_%H%M%S")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"output dir: {out_dir}")
    _dataset["build_n"] = max(level_prompts(args, c) for _, c, _ in levels)

    results = {arm: run_arm(arm, args, out_dir, levels) for arm in args.arms}
    (out_dir / "raw.json").write_text(json.dumps(results, indent=2))

    xname = "imgs" if args.image_counts else "conc"
    for arm, arm_results in results.items():
        print(f"\n[{arm}]  {xname:>6}  mean_ttft   p99_ttft   mean_itl    p99_itl  req/s")
        for x, r in sorted(arm_results.items()):
            if r:
                vals = "".join(f"{r[k]:>11.1f}" for k in METRICS)
                print(f"  {x:>6}{vals}{r['request_throughput']:>7.2f}")
            else:
                print(f"  {x:>6}  FAILED")
    plot(results, args, out_dir, levels)
    print(f"\nsaved: {out_dir}/raw.json, {out_dir}/sweep.png")


if __name__ == "__main__":
    main()
