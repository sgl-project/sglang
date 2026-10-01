"""Correctness-gated AITER vs Gluon FAv3 benchmark for Wan2.2 shapes."""

from __future__ import annotations

import argparse
import statistics

import aiter
import torch
import triton

from sglang.kernels.ops.attention.gluon_fav3_gfx1250 import (
    FAv3LaunchConfig,
    gluon_fav3_attention,
    select_fav3_launch_config,
)

PRESETS = {
    "reference": (1, 8192, 8192, 16),
    # Wan2.2-T2V-A14B at 720x1280x193: latent tokens are 49*45*80.
    "wan-cross": (2, 176_400, 512, 40),
    "wan-self": (2, 176_400, 176_400, 40),
}


def aiter_attention(query, key, value):
    return aiter.flash_attn_func(
        query,
        key,
        value,
        dropout_p=0.0,
        causal=False,
        return_attn_probs=False,
        return_lse=False,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=PRESETS, default="reference")
    parser.add_argument("--batch", type=int)
    parser.add_argument("--sq", type=int)
    parser.add_argument("--sk", type=int)
    parser.add_argument("--heads", type=int)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument(
        "--provider", choices=("both", "aiter", "gluon"), default="both"
    )
    parser.add_argument("--skip-check", action="store_true")
    parser.add_argument("--trace-provider", choices=("aiter", "gluon"))
    parser.add_argument(
        "--schedule",
        choices=("auto", "pipeline", "pingpong"),
    )
    parser.add_argument("--scheduler")
    return parser.parse_args()


def main():
    args = parse_args()
    batch, seqlen_q, seqlen_k, num_heads = PRESETS[args.preset]
    batch = args.batch or batch
    seqlen_q = args.sq or seqlen_q
    seqlen_k = args.sk or seqlen_k
    num_heads = args.heads or num_heads

    torch.manual_seed(0)
    query = torch.randn(
        batch,
        seqlen_q,
        num_heads,
        128,
        device="cuda",
        dtype=torch.bfloat16,
    )
    key = torch.randn(
        batch,
        seqlen_k,
        num_heads,
        128,
        device="cuda",
        dtype=torch.bfloat16,
    )
    value = torch.randn_like(key)

    config = select_fav3_launch_config(batch, num_heads, seqlen_q, seqlen_k=seqlen_k)
    if args.schedule not in (None, "auto"):
        schedule_configs = {
            "pipeline": FAv3LaunchConfig("pipeline", 128, 64, 4, False, ""),
            "pingpong": FAv3LaunchConfig("pingpong", 256, 64, 8, True, ""),
        }
        config = schedule_configs[args.schedule]
    if args.scheduler is not None:
        scheduler = (
            ""
            if args.scheduler in ("", "default")
            else f"amdgpu-sched-strategy={args.scheduler}"
        )
        config = config._replace(llvm_fn_attrs=scheduler)

    def gluon_attention():
        return gluon_fav3_attention(
            query,
            key,
            value,
            launch_config=config,
        )

    if args.trace_provider is not None:
        function = (
            (lambda: aiter_attention(query, key, value))
            if args.trace_provider == "aiter"
            else gluon_attention
        )
        function()
        torch.cuda.synchronize()
        function()
        torch.cuda.synchronize()
        print(f"traced {args.trace_provider}")
        return

    if not args.skip_check:
        expected = aiter_attention(query, key, value)
        actual = gluon_attention()
        torch.cuda.synchronize()
        difference = (actual.float() - expected.float()).abs()
        torch.testing.assert_close(
            actual.float(),
            expected.float(),
            rtol=0.04,
            atol=0.04,
        )
        print(
            "correctness "
            f"max={difference.max().item():.6f} "
            f"mean={difference.mean().item():.6f}"
        )

    print(f"shape=B{batch},Sq{seqlen_q},Sk{seqlen_k},H{num_heads},D128 config={config}")

    all_functions = {
        "aiter": lambda: aiter_attention(query, key, value),
        "gluon": gluon_attention,
    }
    names = ("aiter", "gluon") if args.provider == "both" else (args.provider,)
    functions = {name: all_functions[name] for name in names}
    timings = {name: [] for name in functions}
    for round_index in range(args.rounds):
        order = (
            names if round_index % 2 == 0 or len(names) == 1 else tuple(reversed(names))
        )
        for name in order:
            milliseconds = triton.testing.do_bench(
                functions[name],
                warmup=args.warmup,
                rep=args.rep,
            )
            timings[name].append(float(milliseconds))

    flops = 4.0 * batch * seqlen_q * seqlen_k * num_heads * 128
    means = {}
    for name in names:
        mean_ms = statistics.fmean(timings[name])
        means[name] = mean_ms
        tflops = flops / (mean_ms * 1e-3) / 1e12
        print(
            f"{name:6s} mean={mean_ms:9.4f} ms "
            f"tflops={tflops:9.1f} runs={timings[name]}"
        )
    if args.provider == "both":
        print(f"speedup={means['aiter'] / means['gluon']:.4f}x")


if __name__ == "__main__":
    main()
