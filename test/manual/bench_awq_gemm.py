"""Compare packed AWQ GEMM with dequantize + matmul on the same inputs.

Example:
    python test/manual/bench_awq_gemm.py --k 5120 --n 17408 --m 16 --ternary
"""

import argparse
import json
import statistics

import torch

from sglang.kernels.ops.quantization.awq_triton import (
    awq_dequantize_triton,
    awq_gemm_triton,
)


def make_inputs(args):
    torch.manual_seed(0)
    dtype = getattr(torch, args.dtype)
    x = torch.randn((args.m, args.k), dtype=dtype, device="cuda")
    shape = (args.k, args.n // 8)
    if args.ternary:
        qweight = torch.zeros(shape, dtype=torch.int32, device="cuda")
        for shift in range(0, 32, 4):
            qweight |= (
                torch.randint(7, 10, shape, dtype=torch.int32, device="cuda") << shift
            )
        qzeros = torch.full(
            (args.k // args.group_size, args.n // 8),
            -2004318072,  # 0x88888888 interpreted as int32
            dtype=torch.int32,
            device="cuda",
        )
    else:
        qweight = torch.randint(
            -(2**31), 2**31 - 1, shape, dtype=torch.int32, device="cuda"
        )
        qzeros = torch.randint(
            -(2**31),
            2**31 - 1,
            (args.k // args.group_size, args.n // 8),
            dtype=torch.int32,
            device="cuda",
        )
    scales = (
        torch.rand((args.k // args.group_size, args.n), dtype=dtype, device="cuda")
        * 0.05
    )
    return x, qweight, scales, qzeros


def measure_paired(functions, rounds=9, repeats=40):
    graphs = {}
    for name, fn in functions.items():
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(repeats):
                fn()
        graphs[name] = graph
    for _ in range(8):
        for graph in graphs.values():
            graph.replay()
    torch.cuda.synchronize()

    samples = []
    names = list(functions)
    for i in range(rounds):
        row = {}
        for name in names[:: 1 if i % 2 == 0 else -1]:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            graphs[name].replay()
            end.record()
            end.synchronize()
            row[name] = start.elapsed_time(end) * 1000 / repeats
        samples.append(row)
    return samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, default=1)
    parser.add_argument("--k", type=int, default=5120)
    parser.add_argument("--n", type=int, default=10240)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--ternary", action="store_true")
    args = parser.parse_args()
    x, qweight, scales, qzeros = make_inputs(args)
    functions = {
        "dequantize_matmul": lambda: x @ awq_dequantize_triton(qweight, scales, qzeros),
        "packed_gemm": lambda: awq_gemm_triton(
            x,
            qweight,
            scales,
            qzeros,
            split_k_iters=8,
            block_size_m=16,
            block_size_n=64,
            block_size_k=32,
        ),
    }
    reference = x.float() @ awq_dequantize_triton(qweight, scales, qzeros).float()
    errors = {
        name: (
            (fn().float() - reference).square().sum() / reference.square().sum()
        ).item()
        for name, fn in functions.items()
    }
    samples = measure_paired(functions)
    print(
        json.dumps(
            {
                "device": torch.cuda.get_device_name(),
                "torch": torch.__version__,
                "hip": torch.version.hip,
                "config": vars(args),
                "relative_squared_error": errors,
                "samples_us": samples,
                "median_us": {
                    name: statistics.median(r[name] for r in samples)
                    for name in functions
                },
                "paired_speedup": statistics.median(
                    r["dequantize_matmul"] / r["packed_gemm"] for r in samples
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
