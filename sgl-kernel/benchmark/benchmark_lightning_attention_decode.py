import itertools

import torch
import triton
from sgl_kernel import lightning_attention_decode


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
                "... n e, ... e d -> ... n d", q[:, :, i : i + 1].to(torch.float32), kv.to(torch.float32)
        )
        output.append(qkv)
    output = torch.concat(output, dim=-2)
    
    return output.to(original_dtype), kv


def lightning_attention_decode_kernel(q, k, v, past_kv, slope, output, new_kv):
    return lightning_attention_decode(q, k, v, past_kv, slope, output, new_kv)

def calculate_diff(batch_size):
    dtype = torch.bfloat16
    device = torch.device("cuda")
    num_heads = 64
    head_dim = 96
    seq_len = 1

    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    past_kv = torch.randn(batch_size, num_heads, head_dim, head_dim, device=device)
    slope = torch.randn(num_heads, 1, 1, device=device)

    output_naive, new_kv_naive = lightning_attention_decode_naive(
        q.clone(), k.clone(), v.clone(), past_kv.clone(), slope.clone()
    )
    
    output_kernel = torch.empty_like(output_naive)
    new_kv_kernel = torch.empty_like(new_kv_naive)
    lightning_attention_decode_kernel(
        q.clone(), k.clone(), v.clone(), past_kv.clone(), slope.clone(),
        output_kernel, new_kv_kernel
    )

    print(f"Naive output={output_naive}")
    print(f"Kernel output={output_kernel}")

    if torch.allclose(output_naive, output_kernel, atol=1e-2, rtol=1e-2):
        print("✅ Both implementations match")
    else:
        print("❌ Implementations differ")


batch_size_range = [i for i in range(1, 65)]  # 1 to 128
configs = [(bs,) for bs in batch_size_range]


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch_size"],
        x_vals=[list(_) for _ in configs],
        line_arg="provider",
        line_vals=["naive", "kernel"],
        line_names=["PyTorch Naive", "SGL Kernel"],
        styles=[("blue", "-"), ("red", "-")],
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

    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    past_kv = torch.randn(batch_size, num_heads, head_dim, head_dim, device=device)
    slope = torch.randn(num_heads, 1, 1, device=device)

    quantiles = [0.5, 0.2, 0.8]

    if provider == "naive":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: lightning_attention_decode_naive(
                q.clone(), k.clone(), v.clone(), past_kv.clone(), slope.clone()
            ),
            quantiles=quantiles,
        )
    else:
        output = torch.empty(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
        new_kv = torch.empty(batch_size, num_heads, head_dim, head_dim, device=device)
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: lightning_attention_decode_kernel(
                q.clone(), k.clone(), v.clone(), past_kv.clone(), slope.clone(),
                output, new_kv
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

    # Run correctness test
    calculate_diff(batch_size=4)

    # Run performance benchmark
    benchmark.run(print_data=True, save_path=args.save_path)

