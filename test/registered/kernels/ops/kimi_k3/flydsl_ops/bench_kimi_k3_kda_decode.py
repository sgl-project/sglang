"""Allocation-free microbenchmark for the Kimi-K3 fused f_b + KDA decode."""

import argparse
import runpy
import statistics
from pathlib import Path

import torch

from sglang.test.ci.ci_register import register_amd_ci
from sglang.utils import is_in_ci

register_amd_ci(est_time=120, stage="jit-kernel-benchmark", runner_config="amd")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--trials", type=int, default=21)
    parser.add_argument("--mode", choices=("eager", "graph"), default="graph")
    args = parser.parse_args()

    test = runpy.run_path(str(Path(__file__).with_name("test_kimi_k3_kda_decode.py")))
    f_a, f_b_weight, inputs = test["_make_fb_inputs"](args.batch)
    out = torch.empty((1, args.batch, 12, 128), dtype=torch.bfloat16, device="cuda")
    kwargs = dict(
        f_a=f_a,
        f_b_weight=f_b_weight,
        x=inputs.x,
        conv_weight=inputs.conv_weight,
        conv_bias=None,
        conv_state=inputs.conv_state,
        raw_beta=inputs.raw_beta,
        A_log=inputs.A_log,
        dt_bias=inputs.dt_bias,
        lower_bound=test["_LOWER_BOUND"],
        state=inputs.state,
        state_indices=inputs.state_indices,
        output_gate=inputs.output_gate,
        norm_weight=inputs.norm_weight,
        norm_eps=test["_NORM_EPS"],
        out=out,
    )
    fn = test["flydsl_kimi_k3_kda_decode_with_f_b"]
    for _ in range(args.warmup):
        fn(**kwargs)
    torch.cuda.synchronize()

    if args.mode == "graph":
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fn(**kwargs)
        replay = graph.replay
    else:
        replay = lambda: fn(**kwargs)

    for _ in range(args.warmup):
        replay()
    torch.cuda.synchronize()
    samples = []
    for _ in range(args.trials):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(args.iters):
            replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / args.iters)

    print(
        f"batch={args.batch} mode={args.mode} "
        f"p50_us={statistics.median(samples):.4f} "
        f"mean_us={statistics.mean(samples):.4f} "
        f"p10_us={sorted(samples)[max(0, args.trials // 10 - 1)]:.4f} "
        f"p90_us={sorted(samples)[min(args.trials - 1, 9 * args.trials // 10)]:.4f}"
    )


if __name__ == "__main__":
    if is_in_ci():
        print("Skipping bench_kimi_k3_kda_decode.py in CI")
        raise SystemExit(0)
    main()
