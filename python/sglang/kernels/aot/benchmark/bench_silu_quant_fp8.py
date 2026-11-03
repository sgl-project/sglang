"""Benchmark: fused_silu_mul_quant_fp8 vs separate silu_and_mul + per_token_group_quant_fp8 (H200).

Compares two paths for the activation + quantization step after MoE gate-up GEMM:
  - baseline: silu_and_mul -> per_token_group_quant_fp8 (two kernel launches)
  - fused:    fused_silu_mul_quant_fp8 (one kernel launch)
"""

import sys
import time

import torch
import triton
import triton.language as tl

sys.path.insert(0, "/tmp")


def bench():
    from sglang.kernels.ops.moe.fused_moe_triton_kernels import fused_silu_mul_quant_fp8
    from sglang.kernels.ops.quantization.fp8_kernel import per_token_group_quant_fp8

    SWIGLU_LIMIT = 0.0  # DSV4 uses swiglu_limit=10; test without clamp first

    configs = [
        # (num_tokens, hidden_dim, group_size)
        (128, 2048, 128),
        (256, 2048, 128),
        (512, 2048, 128),
        (1024, 2048, 128),
        (2048, 2048, 128),
        (4096, 2048, 128),
        (8192, 2048, 128),
        (128, 4096, 128),
        (1024, 4096, 128),
        (8192, 4096, 128),
        (128, 8192, 128),
        (1024, 8192, 128),
        (8192, 8192, 128),
    ]

    # Correctness check
    torch.manual_seed(42)
    x = torch.randn(256, 2 * 2048, device="cuda", dtype=torch.bfloat16)
    d = 2048
    result_ref = torch.nn.functional.silu(x[:, :d]) * x[:, d:]
    ref_fp8, ref_scale = per_token_group_quant_fp8(result_ref, 128)
    fused_fp8, fused_scale = fused_silu_mul_quant_fp8(x, 128)
    p_diff = (ref_fp8.float() - fused_fp8.float()).abs().max().item()
    s_diff = (ref_scale - fused_scale).abs().max().item()
    print(f"Correctness: fp8_diff={p_diff}, scale_diff={s_diff:.6f}")
    print()

    print("=" * 85)
    print(f"fused_silu_mul_quant_fp8 vs separate silu_and_mul + quant (H200)")
    print(f"swiglu_limit={SWIGLU_LIMIT}, group_size=128")
    print("=" * 85)
    print(
        f"{'tokens':>7} {'hidden':>8} |{'baseline(ms)':>13}{'fused(ms)':>11}{'speedup':>8} | {'scale_diff':>11}"
    )
    print("-" * 65)

    for num_tokens, hidden_dim, group_size in configs:
        torch.manual_seed(42)
        x = torch.randn(num_tokens, 2 * hidden_dim, device="cuda", dtype=torch.bfloat16)
        n_iter = 200 if num_tokens <= 1024 else 50

        # baseline: silu_and_mul + per_token_group_quant_fp8
        def run_baseline():
            d = hidden_dim
            result = torch.nn.functional.silu(x[:, :d]) * x[:, d:]
            result_fp8, result_scale = per_token_group_quant_fp8(result, group_size)
            return result_fp8, result_scale

        for _ in range(10):
            run_baseline()
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(n_iter):
            run_baseline()
        torch.cuda.synchronize()
        lat_base = (time.time() - t0) / n_iter * 1000

        # fused
        def run_fused():
            return fused_silu_mul_quant_fp8(x, group_size)

        for _ in range(10):
            run_fused()
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(n_iter):
            run_fused()
        torch.cuda.synchronize()
        lat_fused = (time.time() - t0) / n_iter * 1000

        # Per-config correctness
        b_fp8, b_scale = run_baseline()
        f_fp8, f_scale = run_fused()
        s_diff = (b_scale.float() - f_scale.float()).abs().max().item()

        speedup = lat_base / lat_fused if lat_fused > 0 else 0
        print(
            f"{num_tokens:>7} {hidden_dim:>8} |{lat_base:>12.4f}ms{lat_fused:>10.4f}ms{speedup:>7.2f}x | {s_diff:>10.6f}"
        )

    print("-" * 65)
    print("baseline = F.silu(gate)*up + per_token_group_quant_fp8 (2 launches)")
    print("fused    = fused_silu_mul_quant_fp8 (1 launch)")


if __name__ == "__main__":
    bench()
