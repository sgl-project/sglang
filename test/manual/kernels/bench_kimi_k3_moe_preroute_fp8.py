"""Graph microbenchmark for the Kimi-K3 FP8 pre-route projections on gfx950.

Run: python test/manual/kernels/bench_kimi_k3_moe_preroute_fp8.py --tokens 1
"""

import argparse
import statistics

import torch

from sglang.kernels.ops.moe.kimi_k3_flydsl.kimi_k3_moe_preroute_fp8 import (
    kimi_k3_moe_tri_projection_cooperative_preactivated_fp8,
    kimi_k3_moe_tri_projection_fp8,
    quantize_preroute_weight,
)

# Each FP8 weight set is ~50 MB; eight of them exceed MI355X's 256 MB MALL, so
# every replay reads its weights from HBM as decode does.
_NUM_WEIGHT_SETS = 8


def bench(fns, warmup: int, iters: int, trials: int) -> tuple[float, float, float]:
    """p50/p10/p90 microseconds per call; one graph replays every fn once."""
    for fn in fns:
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for fn in fns:
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
        samples.append(start.elapsed_time(end) * 1000 / (iters * len(fns)))
    ordered = sorted(samples)
    return (
        statistics.median(samples),
        ordered[max(0, trials // 10 - 1)],
        ordered[min(trials - 1, 9 * trials // 10)],
    )


def _weight_set(tokens: int, device: torch.device):
    routed_bf16 = torch.randn((3584, 7168), dtype=torch.bfloat16, device=device)
    shared_bf16 = torch.randn((1536, 7168), dtype=torch.bfloat16, device=device)
    router = torch.randn((896, 7168), dtype=torch.bfloat16, device=device)
    routed, routed_scale = quantize_preroute_weight(routed_bf16)
    shared, shared_scale = quantize_preroute_weight(shared_bf16)
    if tokens in (2, 4):
        # The cooperative kernel reads gate and up rows interleaved.
        shared = (
            shared.view(2, 768, 7168).permute(1, 0, 2).contiguous().view(1536, 7168)
        )
        shared_scale = shared_scale.view(2, 768).t().contiguous().view(1536)
    merged = torch.cat((shared_bf16, router, routed_bf16), dim=0).contiguous()
    return (routed, routed_scale, shared, shared_scale, router), merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, choices=(1, 2, 4), required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--trials", type=int, default=11)
    args = parser.parse_args()
    torch.manual_seed(20260817)
    device = torch.device("cuda")
    hidden = torch.randn((args.tokens, 7168), dtype=torch.bfloat16, device=device)
    sets = [_weight_set(args.tokens, device) for _ in range(_NUM_WEIGHT_SETS)]

    def fp8_call(weights):
        if args.tokens == 1:
            return lambda: kimi_k3_moe_tri_projection_fp8(hidden, *weights)
        return lambda: kimi_k3_moe_tri_projection_cooperative_preactivated_fp8(
            hidden, *weights, situ_beta=4.0, situ_linear_beta=25.0, fast_situ=True
        )

    fp8 = bench(
        [fp8_call(weights) for weights, _ in sets],
        args.warmup,
        args.iters,
        args.trials,
    )
    bf16 = bench(
        [(lambda m=merged: torch.mm(hidden, m.t())) for _, merged in sets],
        args.warmup,
        args.iters,
        args.trials,
    )
    print(
        f"M={args.tokens} fp8_p50_us={fp8[0]:.3f} fp8_p10_us={fp8[1]:.3f} "
        f"fp8_p90_us={fp8[2]:.3f} bf16_p50_us={bf16[0]:.3f} "
        f"speedup={bf16[0] / fp8[0]:.3f}x"
    )


if __name__ == "__main__":
    main()
