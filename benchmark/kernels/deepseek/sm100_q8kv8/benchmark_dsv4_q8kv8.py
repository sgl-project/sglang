#!/usr/bin/env python3
"""Benchmark DeepSeek-V4-Flash BF16 sparse prefill against SM100 Q8KV8.

The measurement covers the complete backend functions, including Q casting,
KV workspace gathering/conversion, sparse-index adaptation, attention, and
output slicing.  It intentionally excludes the rest of the model and NCCL.

For safety on the shared B200 host this script refuses to run unless physical
GPU 7 is the only CUDA-visible device.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(os.environ.get("SGLANG_REPO_ROOT", THIS_DIR.parents[3]))
if not (REPO_ROOT / "python/sglang").is_dir():
    REPO_ROOT = Path("/upfs/abing/sglang/pr/cpp_radix_tree/sglang")
os.environ.setdefault("SGLANG_REPO_ROOT", str(REPO_ROOT))
BASE_BENCHMARK = THIS_DIR / "benchmark_dsv4_prefill_paths.py"
if not BASE_BENCHMARK.is_file():
    BASE_BENCHMARK = (
        REPO_ROOT / "benchmark/kernels/deepseek/benchmark_dsv4_prefill_paths.py"
    )
RESULTS_DIR = THIS_DIR / "results"


def _load_base_benchmark():
    spec = importlib.util.spec_from_file_location("dsv4_prefill_base", BASE_BENCHMARK)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load benchmark helpers from {BASE_BENCHMARK}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


base = _load_base_benchmark()
torch = base.torch


def _git_output(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _make_q16_call(case) -> Callable[[], torch.Tensor]:
    def call() -> torch.Tensor:
        return case.backend._forward_prefill_sparse(
            q=case.q,
            layer_id=0,
            compress_ratio=case.compress_ratio,
            forward_batch=case.forward_batch,
            token_to_kv_pool=case.token_pool,
            core_attn_metadata=case.core_metadata,
            attn_sink=case.attn_sink,
        )

    return call


def _make_q8_call(
    case, q_active: torch.Tensor, sink_active: torch.Tensor
) -> Callable[[], torch.Tensor]:
    def call() -> torch.Tensor:
        return case.backend._forward_prefill_sparse_q8kv8(
            q=q_active,
            layer_id=0,
            compress_ratio=case.compress_ratio,
            forward_batch=case.forward_batch,
            token_to_kv_pool=case.token_pool,
            core_attn_metadata=case.core_metadata,
            attn_sink=sink_active,
        )

    return call


def _reset_backend(case) -> None:
    case.backend.forward_metadata.sparse_prefill_cache = None
    case.backend._q8kv8_qpad_buf = None
    case.backend._q8kv8_attn_sink_pad = None
    case.backend._q8kv8_attn_sink_pad_cache = None
    case.backend._q8kv8_identity_scale = None
    case.backend._q8kv8_output_buffers = None
    case.backend._q8kv8_sparse_prefill_log_emitted = True


def _accuracy(q8: torch.Tensor, q16: torch.Tensor) -> dict[str, float]:
    lhs = q8.float()
    rhs = q16[:, : q8.shape[1]].float()
    diff = (lhs - rhs).abs()
    lhs_flat = lhs.flatten()
    rhs_flat = rhs.flatten()
    cosine = torch.nn.functional.cosine_similarity(lhs_flat, rhs_flat, dim=0)
    return {
        "mean_abs_error": diff.mean().item(),
        "max_abs_error": diff.max().item(),
        "cosine_similarity": cosine.item(),
    }


def _weighted_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[int, int], dict[int, dict[str, object]]] = {}
    for row in rows:
        key = (int(row["seq_len"]), int(row["batch_size"]))
        grouped.setdefault(key, {})[int(row["compress_ratio"])] = row

    summary: list[dict[str, object]] = []
    for (seq_len, batch_size), ratio_rows in sorted(grouped.items()):
        if not all(ratio in ratio_rows for ratio in base.DEFAULT_LAYER_WEIGHTS):
            continue
        q16_ms = sum(
            float(ratio_rows[ratio]["q16_median_ms"]) * weight
            for ratio, weight in base.DEFAULT_LAYER_WEIGHTS.items()
        )
        q8_ms = sum(
            float(ratio_rows[ratio]["q8_median_ms"]) * weight
            for ratio, weight in base.DEFAULT_LAYER_WEIGHTS.items()
        )
        summary.append(
            {
                "seq_len": seq_len,
                "batch_size": batch_size,
                "q16_weighted_43_layers_ms": q16_ms,
                "q8_weighted_43_layers_ms": q8_ms,
                "speedup": q16_ms / q8_ms,
                "gain_percent": (q16_ms / q8_ms - 1.0) * 100.0,
            }
        )
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seq-lens",
        type=int,
        nargs="+",
        default=[512, 1024, 2048, 4096, 8192, 16384, 32768],
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--compress-ratios", type=int, nargs="+", default=[0, 4, 128])
    parser.add_argument(
        "--num-heads",
        type=int,
        default=8,
        help="TP-local Q heads; DeepSeek-V4-Flash TP8 uses 8.",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tag", default="baseline")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    return parser.parse_args()


def _main() -> None:
    args = _parse_args()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != "7":
        raise SystemExit(
            "Safety check failed: run with CUDA_VISIBLE_DEVICES=7; "
            f"current value is {visible!r}"
        )
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if torch.cuda.device_count() != 1:
        raise SystemExit(
            "Safety check failed: exactly one CUDA-visible GPU is required, got "
            f"{torch.cuda.device_count()}"
        )
    capability = torch.cuda.get_device_capability(0)
    if capability[0] != 10:
        raise SystemExit(f"SM100 is required, got compute capability {capability}")

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_name = f"{timestamp}-{args.tag}"
    args.results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.results_dir / f"{run_name}.csv"
    json_path = args.results_dir / f"{run_name}.json"

    environment = {
        "timestamp": timestamp,
        "hostname": socket.gethostname(),
        "physical_gpu": 7,
        "cuda_visible_devices": visible,
        "gpu_name": torch.cuda.get_device_name(device),
        "compute_capability": list(capability),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "git_branch": _git_output("branch", "--show-current"),
        "git_commit": _git_output("rev-parse", "HEAD"),
        "git_diff_stat": _git_output("diff", "--stat"),
        "sm100_q8kv8_kernel": {
            "implementation": "cuda_tcgen05",
            "block_h": 32,
            "block_topk": 128,
            "producer_dispatch": {
                "short": {"max_s_q": 4095, "threads": 384, "warps": 7},
                "long": {"min_s_q": 4096, "threads": 512, "warps": 11},
            },
            "kv_buffers_d512": 3,
            "kv_buffers_d576": 2,
            "double_block_tcgen05": True,
            "d512_tma_width_bytes": 128,
            "kv_tma_cache_hint": "evict_last",
            "compact_q_storage": True,
            "output_epilogue": "direct_global_active_heads",
            "batched_output_tma": False,
            "store_meta": False,
        },
        "arguments": vars(args) | {"results_dir": str(args.results_dir)},
    }
    print(json.dumps(environment, indent=2), flush=True)

    rows: list[dict[str, object]] = []
    for seq_index, seq_len in enumerate(sorted(set(args.seq_lens))):
        for ratio_index, compress_ratio in enumerate(args.compress_ratios):
            case = base.build_case(
                prefix_len=0,
                extend_len=seq_len,
                batch_size=args.batch_size,
                cp_size=1,
                cp_rank=0,
                compress_ratio=compress_ratio,
                # The BF16 FlashMLA baseline has the production 64-head
                # padding requirement. Q8 below receives only the active TP
                # prefix, matching the optimized SM100 model path.
                num_heads=max(64, args.num_heads),
                device=device,
                seed=args.seed,
            )
            _reset_backend(case)
            q_active = case.q[:, :, : args.num_heads].contiguous()
            sink_active = case.attn_sink[: args.num_heads].contiguous()
            q16_call = _make_q16_call(case)
            q8_call = _make_q8_call(case, q_active, sink_active)

            # Alternate order across shapes to reduce persistent clock bias.
            if (seq_index + ratio_index) % 2:
                q8_timing, q8_out = base.benchmark_cuda(
                    q8_call, warmup=args.warmup, repeats=args.repeats
                )
                q16_timing, q16_out = base.benchmark_cuda(
                    q16_call, warmup=args.warmup, repeats=args.repeats
                )
            else:
                q16_timing, q16_out = base.benchmark_cuda(
                    q16_call, warmup=args.warmup, repeats=args.repeats
                )
                q8_timing, q8_out = base.benchmark_cuda(
                    q8_call, warmup=args.warmup, repeats=args.repeats
                )

            accuracy = _accuracy(q8_out, q16_out)
            speedup = q16_timing.median_ms / q8_timing.median_ms
            row: dict[str, object] = {
                "seq_len": seq_len,
                "batch_size": args.batch_size,
                "total_q_tokens": seq_len * args.batch_size,
                "compress_ratio": compress_ratio,
                "num_heads": args.num_heads,
                "q16_mean_ms": q16_timing.mean_ms,
                "q16_median_ms": q16_timing.median_ms,
                "q16_p10_ms": q16_timing.p10_ms,
                "q16_p90_ms": q16_timing.p90_ms,
                "q8_mean_ms": q8_timing.mean_ms,
                "q8_median_ms": q8_timing.median_ms,
                "q8_p10_ms": q8_timing.p10_ms,
                "q8_p90_ms": q8_timing.p90_ms,
                "speedup": speedup,
                "gain_percent": (speedup - 1.0) * 100.0,
                **accuracy,
            }
            rows.append(row)
            print(
                f"seq={seq_len:>6} C{compress_ratio:<3} "
                f"q16={q16_timing.median_ms:>8.3f} ms "
                f"q8={q8_timing.median_ms:>8.3f} ms "
                f"speedup={speedup:>6.3f}x "
                f"gain={(speedup - 1.0) * 100.0:>+7.2f}% "
                f"cos={accuracy['cosine_similarity']:.6f}",
                flush=True,
            )
            del case, q16_out, q8_out, q16_call, q8_call, q_active, sink_active
            torch.cuda.empty_cache()

    weighted = _weighted_summary(rows)
    with csv_path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with json_path.open("w") as output:
        json.dump(
            {"environment": environment, "rows": rows, "weighted": weighted},
            output,
            indent=2,
        )

    print("Weighted C0x3 + C4x20 + C128x20:")
    for row in weighted:
        print(
            f"seq={row['seq_len']:>6} speedup={row['speedup']:.3f}x "
            f"gain={row['gain_percent']:+.2f}%"
        )
    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")


def main() -> None:
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
        _main()


if __name__ == "__main__":
    main()
