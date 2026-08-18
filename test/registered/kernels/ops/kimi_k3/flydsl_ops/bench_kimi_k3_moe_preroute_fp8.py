#!/usr/bin/env python3
"""Graph microbenchmark for Kimi-K3 preroute projections."""

import argparse
import statistics

import torch

from sglang.kernels.ops.kimi_k3.flydsl.kimi_k3_moe_preroute_fp8 import (
    kimi_k3_moe_tri_projection_cooperative_preactivated_fp8,
    kimi_k3_moe_tri_projection_fp8,
)


def quantize_rows(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    value = weight.float()
    scale = value.abs().amax(dim=1).clamp_min(1e-12) / 448.0
    return (
        (value / scale[:, None])
        .clamp(-448, 448)
        .to(torch.float8_e4m3fn)
        .contiguous(),
        scale.contiguous(),
    )


def bench(fn, warmup: int, iters: int, trials: int) -> tuple[float, float, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()
    samples = []
    for _ in range(trials):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / iters)
    ordered = sorted(samples)
    return (
        statistics.median(samples),
        ordered[max(0, trials // 10 - 1)],
        ordered[min(trials - 1, 9 * trials // 10)],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, choices=(1, 2, 4), required=True)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=500)
    parser.add_argument("--trials", type=int, default=11)
    args = parser.parse_args()
    torch.manual_seed(20260817)
    device = torch.device("cuda")
    hidden = torch.randn(
        (args.tokens, 7168), dtype=torch.bfloat16, device=device
    )
    routed_bf16 = torch.randn((3584, 7168), dtype=torch.bfloat16, device=device)
    shared_bf16 = torch.randn((1536, 7168), dtype=torch.bfloat16, device=device)
    router = torch.randn((896, 7168), dtype=torch.bfloat16, device=device)
    routed, routed_scale = quantize_rows(routed_bf16)
    shared, shared_scale = quantize_rows(shared_bf16)
    merged = torch.cat((shared_bf16, router, routed_bf16), dim=0).contiguous()
    if args.tokens in (2, 4):
        shared = (
            shared.view(2, 768, 7168)
            .permute(1, 0, 2)
            .contiguous()
            .view(1536, 7168)
        )
        shared_scale = (
            shared_scale.view(2, 768).t().contiguous().view(1536)
        )

        def rowdot_fn():
            return kimi_k3_moe_tri_projection_cooperative_preactivated_fp8(
                hidden,
                routed,
                routed_scale,
                shared,
                shared_scale,
                router,
                situ_beta=4.0,
                situ_linear_beta=25.0,
                fast_situ=True,
            )

    else:

        def rowdot_fn():
            return kimi_k3_moe_tri_projection_fp8(
                hidden, routed, routed_scale, shared, shared_scale, router
            )

    rowdot = bench(
        rowdot_fn,
        args.warmup,
        args.iters,
        args.trials,
    )
    bf16 = bench(
        lambda: torch.mm(hidden, merged.t()),
        args.warmup,
        args.iters,
        args.trials,
    )
    print(
        f"M={args.tokens} rowdot_p50_us={rowdot[0]:.3f} "
        f"rowdot_p10_us={rowdot[1]:.3f} rowdot_p90_us={rowdot[2]:.3f} "
        f"bf16_p50_us={bf16[0]:.3f} "
        f"rowdot_speedup={bf16[0] / rowdot[0]:.3f}x"
    )


if __name__ == "__main__":
    main()
