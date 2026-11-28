"""Benchmark fused SiTU against two existing BF16 GEMMs plus SiTU."""

import argparse

import torch

from sglang.kernels.ops.gemm.cutedsl_bf16_gemm import cutedsl_bf16_gemm
from sglang.kernels.ops.gemm.cutedsl_bf16_gemm_fused_situ import (
    _TGV_SITU_TACTICS,
    cutedsl_bf16_gemm_fused_situ,
)


@torch.compile(fullgraph=True)
def separate_situ(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    gate = gate.float()
    up = up.float()
    return (
        4.0
        * torch.tanh(gate / 4.0)
        * torch.sigmoid(gate)
        * 25.0
        * torch.tanh(up / 25.0)
    ).bfloat16()


def fp32_reference(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    gate_weight, up_weight = weight.chunk(2, dim=0)
    gate = x.float() @ gate_weight.float().T
    up = x.float() @ up_weight.float().T
    return separate_situ(gate, up)


def graph_time_us(fn, warmup: int, iterations: int) -> float:
    output = [None]
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        output[0] = fn()

    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()

    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    for _ in range(iterations):
        graph.replay()
    end.record()
    end.synchronize()
    return begin.elapsed_time(end) * 1000.0 / iterations


def run_case(
    tokens: int,
    intermediate: int,
    k: int,
    tactics: list[int],
    warmup: int,
    iterations: int,
):
    torch.manual_seed(1234 + tokens)
    x = torch.randn(tokens, k, device="cuda", dtype=torch.bfloat16) * 0.02
    weight = (
        torch.randn(2 * intermediate, k, device="cuda", dtype=torch.bfloat16) * 0.02
    )
    gate_weight, up_weight = weight.chunk(2, dim=0)

    def previous():
        gate = cutedsl_bf16_gemm(x, gate_weight)
        up = cutedsl_bf16_gemm(x, up_weight)
        return separate_situ(gate, up)

    previous_out = previous()
    reference = fp32_reference(x, weight)
    previous_us = graph_time_us(previous, warmup, iterations)

    for tactic in tactics:
        config = _TGV_SITU_TACTICS[tactic] if tactic >= 0 else "auto"

        def fused():
            return cutedsl_bf16_gemm_fused_situ(x, weight, tactic=tactic)

        fused_out = fused()
        torch.testing.assert_close(fused_out, reference, rtol=2e-2, atol=0.25)
        torch.testing.assert_close(fused_out, previous_out, rtol=2e-2, atol=0.25)
        fused_us = graph_time_us(fused, warmup, iterations)
        print(
            f"T={tokens:4d} I={intermediate:5d} K={k:5d} "
            f"tactic={tactic:2d} config={str(config):24s} "
            f"fused={fused_us:8.3f} us previous={previous_us:8.3f} us "
            f"speedup={previous_us / fused_us:6.3f}x"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 8, 16, 64])
    parser.add_argument("--intermediate", type=int, default=1024)
    parser.add_argument("--k", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--tactics", type=int, nargs="+")
    parser.add_argument("--sweep", action="store_true")
    args = parser.parse_args()

    if args.sweep and args.tactics:
        parser.error("use either --sweep or --tactics")
    tactics = (
        list(range(len(_TGV_SITU_TACTICS)))
        if args.sweep
        else (args.tactics if args.tactics else [-1])
    )

    print(torch.cuda.get_device_name())
    for tokens in args.tokens:
        run_case(
            tokens,
            args.intermediate,
            args.k,
            tactics,
            args.warmup,
            args.iterations,
        )


if __name__ == "__main__":
    main()
