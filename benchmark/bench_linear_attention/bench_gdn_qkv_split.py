from __future__ import annotations

import argparse

import torch

from sglang.kernels.ops.attention.triton_gdn_fused_proj import (
    fused_qkv_split_gdn_prefill,
)

DTYPES = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark GDN prefill QKV split fallback vs fused Triton path."
    )
    parser.add_argument("--seq-len", type=int, default=8192)
    parser.add_argument("--num-q-heads", type=int, default=16)
    parser.add_argument("--num-k-heads", type=int, default=16)
    parser.add_argument("--num-v-heads", type=int, default=16)
    parser.add_argument("--head-q", type=int, default=128)
    parser.add_argument("--head-k", type=int, default=128)
    parser.add_argument("--head-v", type=int, default=128)
    parser.add_argument("--dtype", choices=DTYPES.keys(), default="bf16")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    return parser.parse_args()


def make_non_contiguous_view(src: torch.Tensor) -> torch.Tensor:
    backing = torch.empty(
        src.shape[1],
        src.shape[0],
        dtype=src.dtype,
        device=src.device,
    )
    view = backing.transpose(0, 1)
    view.copy_(src)
    return view


def split_reference(
    mixed_qkv: torch.Tensor,
    num_q_heads: int,
    num_k_heads: int,
    num_v_heads: int,
    head_q: int,
    head_k: int,
    head_v: int,
):
    q_dim = num_q_heads * head_q
    k_dim = num_k_heads * head_k
    v_dim = num_v_heads * head_v
    actual_seq_len = mixed_qkv.shape[0]
    query, key, value = torch.split(mixed_qkv, [q_dim, k_dim, v_dim], dim=-1)
    query = query.reshape(1, actual_seq_len, num_q_heads, head_q).contiguous()
    key = key.reshape(1, actual_seq_len, num_k_heads, head_k).contiguous()
    value = value.reshape(1, actual_seq_len, num_v_heads, head_v).contiguous()
    return query, key, value


@torch.inference_mode()
def benchmark(fn, warmup: int, iters: int) -> float:
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
    return start.elapsed_time(end) * 1000.0 / iters


def check_close(actual, expected):
    for actual_tensor, expected_tensor in zip(actual, expected):
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0, atol=0)


def run_case(name: str, mixed_qkv: torch.Tensor, args):
    shape_args = (
        args.num_q_heads,
        args.num_k_heads,
        args.num_v_heads,
        args.head_q,
        args.head_k,
        args.head_v,
    )
    expected = split_reference(mixed_qkv, *shape_args)
    actual = fused_qkv_split_gdn_prefill(mixed_qkv, *shape_args)
    check_close(actual, expected)

    baseline_us = benchmark(
        lambda: split_reference(mixed_qkv, *shape_args),
        args.warmup,
        args.iters,
    )
    fused_us = benchmark(
        lambda: fused_qkv_split_gdn_prefill(mixed_qkv, *shape_args),
        args.warmup,
        args.iters,
    )
    speedup = baseline_us / fused_us
    print(f"{name:>12} {baseline_us:12.2f} {fused_us:12.2f} {speedup:10.2f}x")


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = DTYPES[args.dtype]
    qkv_dim = (
        args.num_q_heads * args.head_q
        + args.num_k_heads * args.head_k
        + args.num_v_heads * args.head_v
    )
    mixed_qkv = torch.randn(args.seq_len, qkv_dim, dtype=dtype, device=device)
    mixed_qkv_strided = make_non_contiguous_view(mixed_qkv)

    print(
        f"seq_len={args.seq_len} qkv_dim={qkv_dim} dtype={args.dtype} "
        f"warmup={args.warmup} iters={args.iters}"
    )
    print(f"{'layout':>12} {'baseline_us':>12} {'fused_us':>12} {'speedup':>11}")
    run_case("contiguous", mixed_qkv, args)
    run_case("strided", mixed_qkv_strided, args)


if __name__ == "__main__":
    main()
