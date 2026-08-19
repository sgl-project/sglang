from __future__ import annotations

import argparse

import torch

from sglang.kernels.ops.attention.triton_gdn_fused_proj import (
    fused_qkv_split_gdn_prefill,
)
from sglang.kernels.ops.mamba.causal_conv1d_triton import causal_conv1d_fn

DTYPES = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark KDA prefill three-way QKV preprocessing against fusion."
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--total-tokens", "--seq-len", type=int, default=8192)
    parser.add_argument(
        "--seq-lens",
        type=str,
        help="Comma-separated variable sequence lengths; overrides batch/total tokens.",
    )
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--conv-width", type=int, default=4)
    parser.add_argument("--dtype", choices=DTYPES, default="bf16")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=200)
    return parser.parse_args()


def make_inputs(args):
    if args.seq_lens:
        seq_lens_cpu = [int(seq_len) for seq_len in args.seq_lens.split(",")]
        if not seq_lens_cpu or any(seq_len <= 0 for seq_len in seq_lens_cpu):
            raise ValueError("--seq-lens must contain positive integers")
        batch_size = len(seq_lens_cpu)
        total_tokens = sum(seq_lens_cpu)
    else:
        if args.total_tokens % args.batch_size != 0:
            raise ValueError("--total-tokens must be divisible by --batch-size")
        batch_size = args.batch_size
        total_tokens = args.total_tokens
        seq_lens_cpu = [total_tokens // batch_size] * batch_size

    torch.manual_seed(42)
    dtype = DTYPES[args.dtype]
    channel_dim = args.num_heads * args.head_dim
    qkv_dim = 3 * channel_dim
    query_start_loc = torch.tensor(
        [0, *torch.tensor(seq_lens_cpu).cumsum(0).tolist()],
        device="cuda",
        dtype=torch.int32,
    )

    return {
        "mixed_qkv": torch.randn(total_tokens, qkv_dim, device="cuda", dtype=dtype),
        "conv_weights": torch.randn(
            qkv_dim, args.conv_width, device="cuda", dtype=dtype
        ),
        "conv_states": torch.randn(
            batch_size,
            qkv_dim,
            args.conv_width - 1,
            device="cuda",
            dtype=dtype,
        ),
        "cache_indices": torch.arange(batch_size, device="cuda", dtype=torch.int32),
        "has_initial_state": torch.ones(batch_size, device="cuda", dtype=torch.bool),
        "query_start_loc": query_start_loc,
        "seq_lens_cpu": seq_lens_cpu,
        "batch_size": batch_size,
        "total_tokens": total_tokens,
        "channel_dim": channel_dim,
    }


def run_baseline_conv(inputs, conv_states):
    outputs = []
    for qkv, weight, state in zip(
        inputs["mixed_qkv"].transpose(0, 1).split(inputs["channel_dim"], dim=0),
        inputs["conv_weights"].split(inputs["channel_dim"], dim=0),
        conv_states.split(inputs["channel_dim"], dim=1),
    ):
        output = causal_conv1d_fn(
            qkv,
            weight,
            None,
            activation="silu",
            conv_states=state,
            has_initial_state=inputs["has_initial_state"],
            cache_indices=inputs["cache_indices"],
            query_start_loc=inputs["query_start_loc"],
            seq_lens_cpu=inputs["seq_lens_cpu"],
        ).transpose(0, 1)
        outputs.append(output)
    return outputs


def run_baseline(inputs, args, conv_states):
    return [
        output.view(
            1, inputs["total_tokens"], args.num_heads, args.head_dim
        ).contiguous()
        for output in run_baseline_conv(inputs, conv_states)
    ]


def run_packed_conv(inputs, conv_states):
    mixed_qkv = causal_conv1d_fn(
        inputs["mixed_qkv"].transpose(0, 1),
        inputs["conv_weights"],
        None,
        activation="silu",
        conv_states=conv_states,
        has_initial_state=inputs["has_initial_state"],
        cache_indices=inputs["cache_indices"],
        query_start_loc=inputs["query_start_loc"],
        seq_lens_cpu=inputs["seq_lens_cpu"],
    ).transpose(0, 1)
    return mixed_qkv


def run_packed_aten(inputs, args, conv_states):
    mixed_qkv = run_packed_conv(inputs, conv_states)
    return tuple(
        tensor.view(
            1, inputs["total_tokens"], args.num_heads, args.head_dim
        ).contiguous()
        for tensor in mixed_qkv.split(inputs["channel_dim"], dim=-1)
    )


def run_fused(inputs, args, conv_states):
    mixed_qkv = run_packed_conv(inputs, conv_states)
    return fused_qkv_split_gdn_prefill(
        mixed_qkv,
        args.num_heads,
        args.num_heads,
        args.num_heads,
        args.head_dim,
        args.head_dim,
        args.head_dim,
    )


def benchmark(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000 / iters


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA or ROCm is required")

    inputs = make_inputs(args)
    baseline_states = inputs["conv_states"].clone()
    fused_states = inputs["conv_states"].clone()
    expected = run_baseline(inputs, args, baseline_states)
    packed_aten_states = inputs["conv_states"].clone()
    packed_aten = run_packed_aten(inputs, args, packed_aten_states)
    actual = run_fused(inputs, args, fused_states)
    for actual_tensor, expected_tensor in zip(packed_aten, expected):
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0, atol=0)
    for actual_tensor, expected_tensor in zip(actual, expected):
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0, atol=0)
    torch.testing.assert_close(packed_aten_states, baseline_states, rtol=0, atol=0)
    torch.testing.assert_close(fused_states, baseline_states, rtol=0, atol=0)

    baseline_us = benchmark(
        lambda: run_baseline(inputs, args, inputs["conv_states"]),
        args.warmup,
        args.iters,
    )
    baseline_conv_us = benchmark(
        lambda: run_baseline_conv(inputs, inputs["conv_states"]),
        args.warmup,
        args.iters,
    )
    packed_conv_us = benchmark(
        lambda: run_packed_conv(inputs, inputs["conv_states"]),
        args.warmup,
        args.iters,
    )
    packed_aten_us = benchmark(
        lambda: run_packed_aten(inputs, args, inputs["conv_states"]),
        args.warmup,
        args.iters,
    )
    fused_us = benchmark(
        lambda: run_fused(inputs, args, inputs["conv_states"]),
        args.warmup,
        args.iters,
    )

    print(
        f"device={torch.cuda.get_device_name()} dtype={args.dtype} "
        f"batch={inputs['batch_size']} tokens={inputs['total_tokens']} "
        f"seq_lens={inputs['seq_lens_cpu']} "
        f"heads={args.num_heads} head_dim={args.head_dim} width={args.conv_width}"
    )
    print(f"baseline_us={baseline_us:.2f}")
    print(f"baseline_conv_us={baseline_conv_us:.2f}")
    print(f"packed_conv_us={packed_conv_us:.2f}")
    print(f"packed_aten_us={packed_aten_us:.2f}")
    print(f"packed_triton_us={fused_us:.2f}")
    print(f"packed_aten_speedup={baseline_us / packed_aten_us:.2f}x")
    print(f"packed_triton_speedup={baseline_us / fused_us:.2f}x")


if __name__ == "__main__":
    main()
