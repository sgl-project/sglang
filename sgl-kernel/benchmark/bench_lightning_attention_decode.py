import itertools
import math
import os

import torch
import triton
import triton.language as tl
from sgl_kernel import lightning_attention_decode

# CI environment detection
IS_CI = (
    os.getenv("CI", "false").lower() == "true"
    or os.getenv("GITHUB_ACTIONS", "false").lower() == "true"
)


def next_power_of_2(n):
    return 2 ** (int(math.ceil(math.log(n, 2))))


@triton.jit
def _decode_kernel(
    Q,
    K,
    V,
    KV,
    Out,
    S,
    b: tl.constexpr,
    h: tl.constexpr,
    n: tl.constexpr,
    d: tl.constexpr,
    d_original: tl.constexpr,
    e: tl.constexpr,
    e_original: tl.constexpr,
):
    off_bh = tl.program_id(0)
    off_h = off_bh % h

    qk_offset = off_bh * n * d
    v_offset = off_bh * n * e
    o_offset = off_bh * n * e
    kv_offset = off_bh * d * e

    s = tl.load(S + off_h)
    ratio = tl.exp(-s)

    d_idx = tl.arange(0, d)
    e_idx = tl.arange(0, e)

    # Create masks for original dimensions
    d_mask = d_idx < d_original
    e_mask = e_idx < e_original

    # Load with masking
    q = tl.load(Q + qk_offset + d_idx, mask=d_mask, other=0.0)
    k = tl.load(K + qk_offset + d_idx, mask=d_mask, other=0.0)
    v = tl.load(V + v_offset + e_idx, mask=e_mask, other=0.0)

    # Load KV with 2D masking
    kv = tl.load(
        KV + kv_offset + d_idx[:, None] * e + e_idx[None, :],
        mask=(d_mask[:, None] & e_mask[None, :]),
        other=0.0,
    )

    # Compute outer product using element-wise operations
    k_v_prod = k[:, None] * v[None, :]
    kv = ratio * kv + k_v_prod

    # Store KV with 2D masking
    tl.store(
        KV + kv_offset + d_idx[:, None] * e + e_idx[None, :],
        kv.to(KV.dtype.element_ty),
        mask=(d_mask[:, None] & e_mask[None, :]),
    )

    # Compute matrix-vector multiplication using element-wise operations and reduction
    o = tl.sum(q[:, None] * kv, axis=0)

    # Store output with masking
    tl.store(Out + o_offset + e_idx, o.to(Out.dtype.element_ty), mask=e_mask)


def triton_lightning_attn_decode(q, k, v, kv, s):
    """Triton implementation of Lightning Attention decode operation"""
    b, h, n, d = q.shape
    e = v.shape[-1]
    assert n == 1, "Sequence length must be 1 in decode mode"

    # Get padded dimensions (power of 2)
    d_padded = next_power_of_2(d)
    e_padded = next_power_of_2(e)

    # Create output tensor (padded)
    o_padded = torch.empty(b, h, n, e_padded, dtype=v.dtype, device=v.device)

    # Create padded tensors without actually padding the data
    q_padded = torch.empty(b, h, n, d_padded, dtype=q.dtype, device=q.device)
    k_padded = torch.empty(b, h, n, d_padded, dtype=k.dtype, device=k.device)
    v_padded = torch.empty(b, h, n, e_padded, dtype=v.dtype, device=v.device)
    kv_padded = torch.empty(
        b, h, d_padded, e_padded, dtype=torch.float32, device=kv.device
    )

    # Copy data to padded tensors
    q_padded[..., :d] = q
    k_padded[..., :d] = k
    v_padded[..., :e] = v
    kv_padded[..., :d, :e] = kv

    # Launch kernel
    grid = (b * h, 1)
    _decode_kernel[grid](
        q_padded,
        k_padded,
        v_padded,
        kv_padded,
        o_padded,
        s,
        b=b,
        h=h,
        n=n,
        d=d_padded,
        d_original=d,
        e=e_padded,
        e_original=e,
    )

    # Get unpadded outputs
    o = o_padded[..., :e]
    kv_out = kv_padded[..., :d, :e]

    return o, kv_out


def lightning_attention_decode_naive(q, k, v, past_kv, slope):
    """Naive implementation of lightning attention decode"""
    original_dtype = q.dtype
    ratio = torch.exp(-slope)  # [h, 1, 1]

    kv = past_kv
    b, h, n, d = q.shape

    output = []
    for i in range(n):
        kv = ratio * kv.to(torch.float32) + torch.einsum(
            "... n d, ... n e -> ... d e",
            k[:, :, i : i + 1],
            v[:, :, i : i + 1],
        )
        qkv = torch.einsum(
            "... n e, ... e d -> ... n d",
            q[:, :, i : i + 1].to(torch.float32),
            kv.to(torch.float32),
        )
        output.append(qkv)
    output = torch.cat(output, dim=-2)

    return output.to(original_dtype), kv


def lightning_attention_decode_kernel(q, k, v, past_kv, slope, output, new_kv):
    return lightning_attention_decode(q, k, v, past_kv, slope, output, new_kv)


def calculate_diff(batch_size):
    dtype = torch.bfloat16
    device = torch.device("cuda")
    num_heads = 64
    head_dim = 96
    seq_len = 1

    q = torch.randn(
        batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype
    )
    k = torch.randn(
        batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype
    )
    v = torch.randn(
        batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype
    )
    past_kv = torch.randn(batch_size, num_heads, head_dim, head_dim, device=device)
    slope = torch.randn(num_heads, 1, 1, device=device)

    output_naive, new_kv_naive = lightning_attention_decode_naive(
        q.clone(), k.clone(), v.clone(), past_kv.clone(), slope.clone()
    )

    output_kernel = torch.empty_like(output_naive)
    new_kv_kernel = torch.empty_like(new_kv_naive)
    lightning_attention_decode_kernel(
        q.clone(),
        k.clone(),
        v.clone(),
        past_kv.clone(),
        slope.clone(),
        output_kernel,
        new_kv_kernel,
    )

    output_triton, new_kv_triton = triton_lightning_attn_decode(
        q.clone(), k.clone(), v.clone(), past_kv.clone(), slope.clone()
    )

    if (
        torch.allclose(output_naive, output_kernel, atol=1e-2, rtol=1e-2)
        and torch.allclose(output_naive, output_triton, atol=1e-2, rtol=1e-2)
        and torch.allclose(new_kv_naive, new_kv_kernel, atol=1e-2, rtol=1e-2)
        and torch.allclose(new_kv_naive, new_kv_triton, atol=1e-2, rtol=1e-2)
    ):
        print("✅ All implementations match")
    else:
        print("❌ Implementations differ")


# Simplified for CI environment
if IS_CI:
    batch_size_range = [1]  # Single batch size for CI
else:
    batch_size_range = [i for i in range(1, 65)]  # 1 to 64

configs = [(bs,) for bs in batch_size_range]


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch_size"],
        x_vals=[list(_) for _ in configs],
        line_arg="provider",
        line_vals=["naive", "kernel", "triton"],
        line_names=["PyTorch Naive", "SGL Kernel", "Triton"],
        styles=[("blue", "-"), ("red", "-"), ("green", "-")],
        ylabel="us",
        plot_name="lightning-attention-decode-performance",
        args={},
    )
)
def benchmark(batch_size, provider):
    dtype = torch.bfloat16
    device = torch.device("cuda")
    num_heads = 64
    head_dim = 96
    seq_len = 1

    q = torch.randn(
        batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype
    )
    k = torch.randn(
        batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype
    )
    v = torch.randn(
        batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype
    )
    past_kv = torch.randn(batch_size, num_heads, head_dim, head_dim, device=device)
    slope = torch.randn(num_heads, 1, 1, device=device)

    quantiles = [0.5, 0.2, 0.8]

    if provider == "naive":
        ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(
            lambda: lightning_attention_decode_naive(
                q.clone(), k.clone(), v.clone(), past_kv.clone(), slope.clone()
            ),
            quantiles=quantiles,
        )
    elif provider == "kernel":
        output = torch.empty(
            batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype
        )
        new_kv = torch.empty(batch_size, num_heads, head_dim, head_dim, device=device)
        ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(
            lambda: lightning_attention_decode_kernel(
                q.clone(),
                k.clone(),
                v.clone(),
                past_kv.clone(),
                slope.clone(),
                output,
                new_kv,
            ),
            quantiles=quantiles,
        )
    elif provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(
            lambda: triton_lightning_attn_decode(
                q.clone(), k.clone(), v.clone(), past_kv.clone(), slope.clone()
            ),
            quantiles=quantiles,
        )

    return 1000 * ms, 1000 * max_ms, 1000 * min_ms


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_path",
        type=str,
        default="./configs/benchmark_ops/lightning_attention_decode_sgl/",
        help="Path to save lightning attention decode benchmark results",
    )
    args = parser.parse_args()

    # Run correctness test - simplified for CI
    test_batch_size = 1 if IS_CI else 4
    calculate_diff(batch_size=test_batch_size)

    # Run performance benchmark
    benchmark.run(print_data=True)
