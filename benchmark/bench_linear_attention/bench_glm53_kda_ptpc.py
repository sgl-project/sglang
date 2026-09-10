"""Benchmark quantization-inclusive GLM-5.3 KDA PTPC projections on gfx950."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

import torch

from sglang.srt.layers.quantization.fp8_utils import apply_fp8_ptpc_linear

TP_SHAPES = {
    4: {"qkv_proj": (6144, 4096), "o_proj": (4096, 2048)},
    8: {"qkv_proj": (3072, 4096), "o_proj": (4096, 1024)},
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp", type=int, choices=TP_SHAPES, required=True)
    parser.add_argument(
        "--modules",
        nargs="+",
        choices=("qkv_proj", "o_proj"),
        default=("qkv_proj",),
    )
    parser.add_argument(
        "--m",
        nargs="+",
        type=int,
        default=(1, 8, 17, 512, 513, 2048, 2049, 8192, 16384, 131072),
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--large-iters", type=int, default=20)
    parser.add_argument(
        "--inner-iters",
        type=int,
        default=1,
        help="Operations per timed sample; increase for sub-millisecond cases",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Optional GLM-5.3 checkpoint snapshot for real layer weights",
    )
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--tp-rank", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def measure_samples(fn, warmup: int, iters: int, inner_iters: int) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(inner_iters):
            fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / inner_iters)
    return samples


def summarize(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)

    def percentile(fraction: float) -> float:
        index = round((len(ordered) - 1) * fraction)
        return ordered[index]

    mean = statistics.fmean(samples)
    stddev = statistics.stdev(samples) if len(samples) > 1 else 0.0
    return {
        "median_ms": statistics.median(samples),
        "p5_ms": percentile(0.05),
        "p95_ms": percentile(0.95),
        "mean_ms": mean,
        "stddev_ms": stddev,
        "cv": stddev / mean if mean else 0.0,
    }


def load_checkpoint_weight(
    checkpoint: Path,
    layer: int,
    module_name: str,
    tp: int,
    tp_rank: int,
) -> torch.Tensor:
    import json

    from safetensors import safe_open

    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    prefix = f"model.language_model.layers.{layer}.self_attn"

    def load(key: str) -> torch.Tensor:
        shard = checkpoint / index[key]
        with safe_open(shard, framework="pt", device="cpu") as file:
            return file.get_tensor(key)

    if module_name == "qkv_proj":
        shards = []
        for name in ("q_proj", "k_proj", "v_proj"):
            weight = load(f"{prefix}.{name}.weight")
            shard_size = weight.shape[0] // tp
            shards.append(weight.narrow(0, tp_rank * shard_size, shard_size))
        return torch.cat(shards).contiguous()

    weight = load(f"{prefix}.o_proj.weight")
    shard_size = weight.shape[1] // tp
    return weight.narrow(1, tp_rank * shard_size, shard_size).contiguous()


@torch.inference_mode()
def run_case(tp: int, module_name: str, m: int, args) -> dict:
    import aiter
    from aiter.ops.shuffle import shuffle_weight
    from aiter.tuned_gemm import tgemm

    n, k = TP_SHAPES[tp][module_name]
    generator = torch.Generator(device="cuda")
    generator.manual_seed(args.seed + tp + m + n + k)
    x = torch.randn(m, k, generator=generator, device="cuda", dtype=torch.bfloat16)
    if args.checkpoint is None:
        weight = torch.randn(
            n, k, generator=generator, device="cuda", dtype=torch.bfloat16
        )
        weight_source = "synthetic"
    else:
        weight = load_checkpoint_weight(
            args.checkpoint, args.layer, module_name, tp, args.tp_rank
        ).to(device="cuda")
        if weight.dtype != torch.bfloat16 or tuple(weight.shape) != (n, k):
            raise ValueError(
                f"Expected BF16 {(n, k)} {module_name} weight, got "
                f"{weight.dtype} {tuple(weight.shape)}"
            )
        weight_source = str(args.checkpoint)
    fp8_weight, weight_scale = aiter.pertoken_quant(
        weight, quant_dtype=aiter.dtypes.fp8
    )
    fp8_weight = shuffle_weight(fp8_weight, (16, 16)).contiguous()

    def bf16():
        return tgemm.mm(x, weight, otype=torch.bfloat16)

    def ptpc():
        q_input = aiter.per_token_quant_hip(x, quant_dtype=aiter.dtypes.fp8)
        return apply_fp8_ptpc_linear(q_input, fp8_weight, weight_scale)

    iters = args.large_iters if m >= 131072 else args.iters
    bf16_samples = measure_samples(bf16, args.warmup, iters, args.inner_iters)
    ptpc_samples = measure_samples(ptpc, args.warmup, iters, args.inner_iters)
    bf16_summary = summarize(bf16_samples)
    ptpc_summary = summarize(ptpc_samples)
    delta = ptpc_summary["median_ms"] / bf16_summary["median_ms"] - 1
    result = {
        "tp": tp,
        "module": module_name,
        "m": m,
        "n": n,
        "k": k,
        "warmup": args.warmup,
        "iters": iters,
        "inner_iters": args.inner_iters,
        "seed": args.seed,
        "weight_source": weight_source,
        "bf16": {**bf16_summary, "samples_ms": bf16_samples},
        "ptpc": {**ptpc_summary, "samples_ms": ptpc_samples},
        "median_delta": delta,
    }
    torch.cuda.empty_cache()
    return result


def write_outputs(results: list[dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".json").write_text(json.dumps(results, indent=2) + "\n")
    with output.with_suffix(".csv").open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=(
                "tp",
                "module",
                "m",
                "n",
                "k",
                "inner_iters",
                "weight_source",
                "bf16_median_ms",
                "bf16_p5_ms",
                "bf16_p95_ms",
                "bf16_cv",
                "ptpc_median_ms",
                "ptpc_p5_ms",
                "ptpc_p95_ms",
                "ptpc_cv",
                "median_delta",
            ),
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "tp": result["tp"],
                    "module": result["module"],
                    "m": result["m"],
                    "n": result["n"],
                    "k": result["k"],
                    "inner_iters": result["inner_iters"],
                    "weight_source": result["weight_source"],
                    "bf16_median_ms": result["bf16"]["median_ms"],
                    "bf16_p5_ms": result["bf16"]["p5_ms"],
                    "bf16_p95_ms": result["bf16"]["p95_ms"],
                    "bf16_cv": result["bf16"]["cv"],
                    "ptpc_median_ms": result["ptpc"]["median_ms"],
                    "ptpc_p5_ms": result["ptpc"]["p5_ms"],
                    "ptpc_p95_ms": result["ptpc"]["p95_ms"],
                    "ptpc_cv": result["ptpc"]["cv"],
                    "median_delta": result["median_delta"],
                }
            )


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A gfx950 GPU is required")
    results = [
        run_case(args.tp, module_name, m, args)
        for module_name in args.modules
        for m in args.m
    ]
    write_outputs(results, args.output)
    for result in results:
        print(
            f"TP{result['tp']} {result['module']} M={result['m']}: "
            f"BF16={result['bf16']['median_ms']:.4f} ms "
            f"PTPC={result['ptpc']['median_ms']:.4f} ms "
            f"delta={result['median_delta']:+.2%}"
        )


if __name__ == "__main__":
    main()
