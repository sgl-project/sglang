"""
Benchmark fused allreduce+rmsnorm on AMD with correctness checks.

This script targets the same fused op used by SGLang:
`tensor_model_parallel_fused_allreduce_rmsnorm`.

It reports:
- eager mode latency (prefill-like)
- graph mode latency (decode-like)
- fused availability (whether fused path returns non-None)
- correctness (fused output matches split allreduce + rmsnorm reference)

Usage example:
  torchrun --nproc_per_node=8 \
    benchmark/kernels/all_reduce/benchmark_fused_ar_rms_amd.py \
    --dtype bfloat16 \
    --prefill-shapes 2048x8192,8192x8192 \
    --decode-shapes 1x8192,4x8192,16x8192 \
    --warmup 10 --iters 30 --repeats 5
"""

import argparse
import csv
import os
import statistics
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F

from sglang.srt.distributed.communication_op import (
    tensor_model_parallel_all_reduce,
    tensor_model_parallel_fused_allreduce_rmsnorm,
)
from sglang.srt.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    graph_capture,
    init_distributed_environment,
    initialize_model_parallel,
    set_custom_all_reduce,
)

Shape = Tuple[int, int]


def parse_shapes(raw: str) -> List[Shape]:
    shapes: List[Shape] = []
    for item in [x.strip() for x in raw.split(",") if x.strip()]:
        if "x" not in item:
            raise ValueError(f"Invalid shape '{item}', expected MxN format.")
        m_str, n_str = item.split("x", 1)
        m = int(m_str)
        n = int(n_str)
        if m <= 0 or n <= 0:
            raise ValueError(f"Invalid shape '{item}', both dims must be positive.")
        shapes.append((m, n))
    if not shapes:
        raise ValueError("Empty shape list is not allowed.")
    return shapes


def dtype_from_name(name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype: {name}")
    return mapping[name]


def check_close(
    a: torch.Tensor, b: torch.Tensor, dtype: torch.dtype
) -> Tuple[bool, str]:
    if dtype == torch.bfloat16:
        rtol, atol = 2e-2, 1.25e-1
    else:
        rtol, atol = 1e-2, 2e-2
    try:
        torch.testing.assert_close(a, b, rtol=rtol, atol=atol)
        return True, "PASS"
    except AssertionError:
        max_diff = torch.max(torch.abs(a - b)).item()
        mean_diff = torch.mean(torch.abs(a - b)).item()
        return False, f"FAIL(max={max_diff:.6f},mean={mean_diff:.6f})"


def _measure_us(
    fn,
    warmup: int,
    iters: int,
    repeats: int,
    device: torch.device,
) -> Tuple[float, Dict[str, float]]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    samples_us: List[float] = []

    for _ in range(max(1, repeats)):
        _barrier(device)
        torch.cuda.synchronize()
        start_event.record()
        for _ in range(iters):
            fn()
        end_event.record()
        end_event.synchronize()
        samples_us.append(start_event.elapsed_time(end_event) * 1000.0 / iters)

    sorted_samples = sorted(samples_us)
    p50 = float(statistics.median(sorted_samples))
    p95 = float(sorted_samples[int((len(sorted_samples) - 1) * 0.95)])
    return p50, {
        "p50_us": p50,
        "p95_us": p95,
        "min_us": float(sorted_samples[0]),
        "max_us": float(sorted_samples[-1]),
    }


def _barrier(device: torch.device):
    try:
        dist.barrier(device_ids=[device.index])
    except TypeError:
        dist.barrier()


def _mean_across_ranks(value: float, device: torch.device) -> float:
    t = torch.tensor([value], dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    t /= dist.get_world_size()
    return float(t.item())


def _all_true_across_ranks(value: bool, device: torch.device) -> bool:
    t = torch.tensor([1 if value else 0], dtype=torch.int32, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.MIN)
    return bool(int(t.item()))


def _make_inputs(
    shape: Shape,
    dtype: torch.dtype,
    seed: int,
    residual_mode: str,
    rank: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    m, n = shape
    torch.manual_seed(seed + rank * 17)
    x = torch.randn((m, n), dtype=torch.float32, device=device).to(dtype)
    if residual_mode == "self":
        residual = x.clone()
    elif residual_mode == "random":
        residual = torch.randn((m, n), dtype=torch.float32, device=device).to(dtype)
    elif residual_mode == "zero":
        residual = torch.zeros((m, n), dtype=dtype, device=device)
    else:
        raise ValueError(f"Unknown residual_mode: {residual_mode}")
    weight = torch.randn((n,), dtype=torch.float32, device=device).to(dtype)
    return x, residual, weight


def _split_reference(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    ar_out = tensor_model_parallel_all_reduce(x.clone())
    residual_out = ar_out + residual
    out = F.rms_norm(
        input=residual_out,
        normalized_shape=(residual_out.shape[-1],),
        weight=weight,
        eps=eps,
    )
    return out, residual_out


def bench_eager(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    warmup: int,
    iters: int,
    repeats: int,
) -> Dict[str, object]:
    split_fn = lambda: _split_reference(x, residual, weight, eps)
    split_us, split_stats = _measure_us(split_fn, warmup, iters, repeats, x.device)

    fused_probe = tensor_model_parallel_fused_allreduce_rmsnorm(
        x.clone(), residual.clone(), weight, eps
    )
    fused_available = fused_probe is not None

    fused_us: Optional[float] = None
    fused_stats: Optional[Dict[str, float]] = None
    if fused_available:
        fused_fn = lambda: tensor_model_parallel_fused_allreduce_rmsnorm(
            x, residual, weight, eps
        )
        fused_us, fused_stats = _measure_us(fused_fn, warmup, iters, repeats, x.device)

    ref_out, ref_residual = _split_reference(x, residual, weight, eps)
    if fused_available:
        fused_out, fused_residual = tensor_model_parallel_fused_allreduce_rmsnorm(
            x.clone(), residual.clone(), weight, eps
        )
        out_ok, out_detail = check_close(fused_out, ref_out, x.dtype)
        res_ok, res_detail = check_close(fused_residual, ref_residual, x.dtype)
        correctness_ok = out_ok and res_ok
        correctness_detail = f"out={out_detail}, residual={res_detail}"
    else:
        correctness_ok = True
        correctness_detail = "SKIP(fused_unavailable)"

    return {
        "split_us": split_us,
        "split_stats": split_stats,
        "fused_available": fused_available,
        "fused_us": fused_us,
        "fused_stats": fused_stats,
        "correctness_ok": correctness_ok,
        "correctness_detail": correctness_detail,
    }


def bench_graph(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    warmup: int,
    iters: int,
    repeats: int,
) -> Dict[str, object]:
    split_x = x.clone()
    split_res = residual.clone()
    split_graph_out: Optional[torch.Tensor] = None

    with graph_capture() as gc:
        split_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(split_graph, stream=gc.stream):
            split_graph_out, _ = _split_reference(split_x, split_res, weight, eps)

    def split_replay():
        split_graph.replay()

    split_us, split_stats = _measure_us(split_replay, warmup, iters, repeats, x.device)

    fused_probe = tensor_model_parallel_fused_allreduce_rmsnorm(
        x.clone(), residual.clone(), weight, eps
    )
    fused_available = fused_probe is not None

    fused_us: Optional[float] = None
    fused_stats: Optional[Dict[str, float]] = None
    fused_graph_out: Optional[torch.Tensor] = None
    fused_graph_residual: Optional[torch.Tensor] = None

    if fused_available:
        fused_x = x.clone()
        fused_res = residual.clone()
        with graph_capture() as gc:
            fused_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(fused_graph, stream=gc.stream):
                fused_graph_out, fused_graph_residual = (
                    tensor_model_parallel_fused_allreduce_rmsnorm(
                        fused_x, fused_res, weight, eps
                    )
                )

        def fused_replay():
            fused_graph.replay()

        fused_us, fused_stats = _measure_us(
            fused_replay, warmup, iters, repeats, x.device
        )

    ref_out, ref_residual = _split_reference(x, residual, weight, eps)
    if (
        fused_available
        and fused_graph_out is not None
        and fused_graph_residual is not None
    ):
        fused_graph.replay()
        torch.cuda.synchronize()
        out_ok, out_detail = check_close(fused_graph_out, ref_out, x.dtype)
        res_ok, res_detail = check_close(fused_graph_residual, ref_residual, x.dtype)
        correctness_ok = out_ok and res_ok
        correctness_detail = f"out={out_detail}, residual={res_detail}"
    else:
        correctness_ok = True
        correctness_detail = "SKIP(fused_unavailable)"

    return {
        "split_us": split_us,
        "split_stats": split_stats,
        "fused_available": fused_available,
        "fused_us": fused_us,
        "fused_stats": fused_stats,
        "correctness_ok": correctness_ok,
        "correctness_detail": correctness_detail,
    }


def _shape_bytes(shape: Shape, dtype: torch.dtype) -> int:
    m, n = shape
    return m * n * torch.tensor([], dtype=dtype).element_size()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark fused allreduce+rmsnorm (prefill eager + decode graph)."
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=["fp16", "bf16", "float16", "bfloat16"],
    )
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--residual-mode",
        type=str,
        default="self",
        choices=["self", "random", "zero"],
        help="Use residual=x (self) to match aiter test behavior by default.",
    )
    parser.add_argument(
        "--prefill-shapes",
        type=str,
        default="2048x8192,8192x8192,16384x8192",
        help="Comma-separated MxN shapes for eager mode.",
    )
    parser.add_argument(
        "--decode-shapes",
        type=str,
        default="1x8192,2x8192,4x8192,8x8192,16x8192",
        help="Comma-separated MxN shapes for graph mode.",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--mode",
        type=str,
        default="both",
        choices=["eager", "graph", "both"],
    )
    parser.add_argument(
        "--csv-out",
        type=str,
        default=None,
        help="Optional output CSV path (written on rank 0 only).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dtype = dtype_from_name(args.dtype)
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    torch.cuda.set_device(local_rank % torch.cuda.device_count())
    device = torch.device(f"cuda:{local_rank % torch.cuda.device_count()}")

    set_custom_all_reduce(True)
    init_distributed_environment(
        world_size=world_size,
        rank=rank,
        local_rank=local_rank,
        distributed_init_method="env://",
        backend="nccl",
    )
    initialize_model_parallel(tensor_model_parallel_size=world_size)

    prefill_shapes = parse_shapes(args.prefill_shapes)
    decode_shapes = parse_shapes(args.decode_shapes)

    if rank == 0:
        print(
            "Config: "
            f"world_size={world_size}, dtype={dtype}, residual_mode={args.residual_mode}, "
            f"warmup={args.warmup}, iters={args.iters}, repeats={args.repeats}"
        )

    run_modes: Sequence[str]
    if args.mode == "both":
        run_modes = ("eager", "graph")
    else:
        run_modes = (args.mode,)
    csv_rows: List[Dict[str, object]] = []

    for mode in run_modes:
        shapes = prefill_shapes if mode == "eager" else decode_shapes
        if rank == 0:
            phase_name = "prefill(eager)" if mode == "eager" else "decode(graph)"
            print("\n" + "=" * 120)
            print(f"Mode: {phase_name}")
            print(
                "| Shape | Input bytes/rank | Split p50 (us) | Fused p50 (us) | Speedup | Fused available | Correctness |"
            )
            print(
                "|:------|-----------------:|---------------:|---------------:|--------:|:----------------|:------------|"
            )

        for shape in shapes:
            x, residual, weight = _make_inputs(
                shape=shape,
                dtype=dtype,
                seed=args.seed,
                residual_mode=args.residual_mode,
                rank=rank,
                device=device,
            )

            if mode == "eager":
                metrics = bench_eager(
                    x=x,
                    residual=residual,
                    weight=weight,
                    eps=args.eps,
                    warmup=args.warmup,
                    iters=args.iters,
                    repeats=args.repeats,
                )
            else:
                metrics = bench_graph(
                    x=x,
                    residual=residual,
                    weight=weight,
                    eps=args.eps,
                    warmup=args.warmup,
                    iters=args.iters,
                    repeats=args.repeats,
                )

            split_us = _mean_across_ranks(float(metrics["split_us"]), device)
            fused_available = _all_true_across_ranks(
                bool(metrics["fused_available"]), device
            )
            correctness_ok = _all_true_across_ranks(
                bool(metrics["correctness_ok"]), device
            )

            fused_us: Optional[float] = None
            if fused_available and metrics["fused_us"] is not None:
                fused_us = _mean_across_ranks(float(metrics["fused_us"]), device)

            if rank == 0:
                m, n = shape
                shape_str = f"{m}x{n}"
                bytes_per_rank = _shape_bytes(shape, dtype)
                if fused_us is not None and fused_us > 0:
                    speedup = split_us / fused_us
                    speedup_str = f"{speedup:.3f}x"
                    fused_str = f"{fused_us:.1f}"
                else:
                    speedup_str = "N/A"
                    fused_str = "N/A"
                correctness_text = (
                    "PASS" if correctness_ok else str(metrics["correctness_detail"])
                )
                print(
                    f"| {shape_str} | {bytes_per_rank} | {split_us:.1f} | {fused_str} | "
                    f"{speedup_str} | {str(fused_available)} | {correctness_text} |"
                )
                csv_rows.append(
                    {
                        "mode": mode,
                        "shape": shape_str,
                        "m": m,
                        "n": n,
                        "bytes_per_rank": bytes_per_rank,
                        "split_p50_us": split_us,
                        "fused_p50_us": fused_us if fused_us is not None else "",
                        "speedup_split_over_fused": (
                            split_us / fused_us
                            if fused_us is not None and fused_us > 0
                            else ""
                        ),
                        "fused_available": fused_available,
                        "correctness_ok": correctness_ok,
                        "correctness_detail": correctness_text,
                        "dtype": str(dtype),
                        "world_size": world_size,
                        "residual_mode": args.residual_mode,
                        "warmup": args.warmup,
                        "iters": args.iters,
                        "repeats": args.repeats,
                    }
                )

    if rank == 0 and args.csv_out:
        os.makedirs(os.path.dirname(args.csv_out) or ".", exist_ok=True)
        fieldnames = [
            "mode",
            "shape",
            "m",
            "n",
            "bytes_per_rank",
            "split_p50_us",
            "fused_p50_us",
            "speedup_split_over_fused",
            "fused_available",
            "correctness_ok",
            "correctness_detail",
            "dtype",
            "world_size",
            "residual_mode",
            "warmup",
            "iters",
            "repeats",
        ]
        with open(args.csv_out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"\nSaved CSV to: {args.csv_out}")

    _barrier(device)
    destroy_model_parallel()
    destroy_distributed_environment()


if __name__ == "__main__":
    main()
