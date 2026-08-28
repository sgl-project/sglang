#!/usr/bin/env python3
"""Capture a GPU-operator profile of the complete DSV4 q16/q8 prefill paths."""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(os.environ.get("SGLANG_REPO_ROOT", THIS_DIR.parents[3]))
if not (REPO_ROOT / "python/sglang").is_dir():
    REPO_ROOT = Path("/upfs/abing/sglang/pr/cpp_radix_tree/sglang")
os.environ.setdefault("SGLANG_REPO_ROOT", str(REPO_ROOT))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


base_path = THIS_DIR / "benchmark_dsv4_prefill_paths.py"
if not base_path.is_file():
    base_path = REPO_ROOT / "benchmark/kernels/deepseek/benchmark_dsv4_prefill_paths.py"
base = _load("dsv4_profile_base", base_path)
bench_path = THIS_DIR / "benchmark_dsv4_q8kv8.py"
if not bench_path.is_file():
    bench_path = (
        REPO_ROOT / "benchmark/kernels/deepseek/sm100_q8kv8/benchmark_dsv4_q8kv8.py"
    )
bench = _load("dsv4_q8_bench", bench_path)
torch = base.torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--compress-ratio", type=int, default=4)
    parser.add_argument("--active-heads", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--results-dir", type=Path, default=THIS_DIR / "results")
    args = parser.parse_args()

    if os.environ.get("CUDA_VISIBLE_DEVICES") != "7":
        raise SystemExit("Safety check failed: use CUDA_VISIBLE_DEVICES=7")
    device = torch.device("cuda:0")
    case = base.build_case(
        seq_len=args.seq_len,
        batch_size=1,
        compress_ratio=args.compress_ratio,
        num_heads=max(64, args.active_heads),
        device=device,
        seed=42,
    )
    bench._reset_backend(case)
    q_active = case.q[:, :, : args.active_heads].contiguous()
    sink_active = case.attn_sink[: args.active_heads].contiguous()
    q16_call = bench._make_q16_call(case)
    q8_call = bench._make_q8_call(case, q_active, sink_active)

    for _ in range(3):
        q16_call()
        q8_call()
    torch.cuda.synchronize()

    activities = [
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ]
    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        with_stack=False,
    ) as profiler:
        for _ in range(args.iterations):
            with torch.profiler.record_function("q16_complete"):
                q16_call()
            with torch.profiler.record_function("q8_complete"):
                q8_call()
        torch.cuda.synchronize()

    args.results_dir.mkdir(parents=True, exist_ok=True)
    stem = (
        f"{time.strftime('%Y%m%d-%H%M%S')}-profile-s{args.seq_len}"
        f"-c{args.compress_ratio}-h{args.active_heads}"
    )
    trace_path = args.results_dir / f"{stem}.json"
    table_path = args.results_dir / f"{stem}.txt"
    profiler.export_chrome_trace(str(trace_path))
    table = profiler.key_averages().table(sort_by="self_cuda_time_total", row_limit=80)
    table_path.write_text(table + "\n")
    print(table)
    print(f"Trace: {trace_path}")
    print(f"Table: {table_path}")


if __name__ == "__main__":
    with base.get_parallel().override(
        world_size=1,
        world_rank=0,
        tp_size=1,
        tp_rank=0,
        attn_tp_size=1,
        attn_tp_rank=0,
        attn_cp_size=1,
        attn_cp_rank=0,
    ):
        main()
