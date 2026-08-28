#!/usr/bin/env python3
"""Validate SM100 Q8KV8 sparse prefill against the BF16 backend golden.

The default matrix is the DeepSeek-V4-Flash TP8 production shape across
4K/8K/16K/32K query lengths and C0/C4/C128 layers.  The comparison covers the
complete backend functions rather than calling only the attention kernels.

For safety on the shared B200 host this script refuses to run unless physical
GPU 7 is the only CUDA-visible device.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import time
from pathlib import Path

import benchmark_dsv4_q8kv8 as benchmark

torch = benchmark.torch
base = benchmark.base
THIS_DIR = Path(__file__).resolve().parent
RESULTS_DIR = THIS_DIR / "results"


def _git_output(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=benchmark.REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seq-lens", type=int, nargs="+", default=[4096, 8192, 16384, 32768]
    )
    parser.add_argument("--compress-ratios", type=int, nargs="+", default=[0, 4, 128])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--num-heads",
        type=int,
        default=8,
        help="TP-local active heads; DeepSeek-V4-Flash TP8 uses 8.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-cosine", type=float, default=0.999)
    parser.add_argument("--max-mean-abs-error", type=float, default=1e-3)
    parser.add_argument("--max-abs-error", type=float, default=2e-2)
    parser.add_argument("--tag", default="gpu7-long-accuracy")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
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
    if args.batch_size <= 0 or args.num_heads <= 0:
        raise SystemExit("--batch-size and --num-heads must be positive")
    if any(seq_len <= 0 for seq_len in args.seq_lens):
        raise SystemExit("--seq-lens must be positive")
    invalid_ratios = set(args.compress_ratios) - {0, 4, 128}
    if invalid_ratios:
        raise SystemExit(f"Unsupported compress ratios: {sorted(invalid_ratios)}")


@torch.no_grad()
def _main() -> None:
    args = _parse_args()
    _validate_args(args)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    args.results_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.results_dir / f"{timestamp}-{args.tag}.json"
    environment = {
        "timestamp": timestamp,
        "hostname": socket.gethostname(),
        "physical_gpu": 7,
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "gpu_name": torch.cuda.get_device_name(device),
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "git_branch": _git_output("branch", "--show-current"),
        "git_commit": _git_output("rev-parse", "HEAD"),
        "arguments": vars(args) | {"results_dir": str(args.results_dir)},
        "golden": "DeepseekV4AttnBackend._forward_prefill_sparse",
        "actual": "DeepseekV4AttnBackend._forward_prefill_sparse_q8kv8",
    }

    rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    for seq_len in sorted(set(args.seq_lens)):
        for compress_ratio in args.compress_ratios:
            case = base.build_case(
                prefix_len=0,
                extend_len=seq_len,
                batch_size=args.batch_size,
                cp_size=1,
                cp_rank=0,
                compress_ratio=compress_ratio,
                num_heads=max(64, args.num_heads),
                device=device,
                seed=args.seed,
            )
            benchmark._reset_backend(case)
            q_active = case.q[:, :, : args.num_heads].contiguous()
            sink_active = case.attn_sink[: args.num_heads].contiguous()

            bf16_out = benchmark._make_q16_call(case)()
            q8_out = benchmark._make_q8_call(case, q_active, sink_active)()
            torch.cuda.synchronize()
            accuracy = benchmark._accuracy(q8_out, bf16_out)
            passed = (
                accuracy["cosine_similarity"] >= args.min_cosine
                and accuracy["mean_abs_error"] <= args.max_mean_abs_error
                and accuracy["max_abs_error"] <= args.max_abs_error
            )
            row: dict[str, object] = {
                "seq_len": seq_len,
                "batch_size": args.batch_size,
                "compress_ratio": compress_ratio,
                "num_heads": args.num_heads,
                **accuracy,
                "passed": passed,
            }
            rows.append(row)
            if not passed:
                failures.append(row)
            print(
                f"{'PASS' if passed else 'FAIL'} seq={seq_len:>6} "
                f"C{compress_ratio:<3} cos={accuracy['cosine_similarity']:.8f} "
                f"mean_abs={accuracy['mean_abs_error']:.8g} "
                f"max_abs={accuracy['max_abs_error']:.8g}",
                flush=True,
            )

            del case, bf16_out, q8_out, q_active, sink_active
            torch.cuda.empty_cache()

    result = {
        "environment": environment,
        "thresholds": {
            "min_cosine": args.min_cosine,
            "max_mean_abs_error": args.max_mean_abs_error,
            "max_abs_error": args.max_abs_error,
        },
        "passed": not failures,
        "num_cases": len(rows),
        "num_failures": len(failures),
        "rows": rows,
    }
    with json_path.open("w") as output:
        json.dump(result, output, indent=2)
    print(f"JSON: {json_path}")
    if failures:
        raise SystemExit(f"Accuracy validation failed for {len(failures)} case(s)")
    print(f"All {len(rows)} accuracy cases passed")


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
