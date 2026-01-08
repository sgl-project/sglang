#!/usr/bin/env python3
"""
Warmup script to pre-build AITER JIT kernels.

This script triggers compilation of commonly used AITER kernels by importing
the relevant modules and calling functions with sample data. This avoids
timeouts during actual tests when kernels need to be compiled on first use.

Run this after clearing pre-built AITER kernels from the Docker image.
"""

import os
import sys
import time

# Ensure AITER is enabled
os.environ["SGLANG_USE_AITER"] = "1"


def warmup_aiter_kernels():
    """Trigger AITER JIT kernel compilation."""
    import torch

    if not torch.cuda.is_available():
        print("CUDA/ROCm not available, skipping AITER warmup")
        return

    print("=" * 60)
    print("AITER JIT Kernel Warmup")
    print("=" * 60)

    device = torch.device("cuda:0")
    start_time = time.time()

    # Warmup flash attention kernel (module_quant) - CRITICAL for multi-GPU DeepSeek tests
    # This prevents deadlocks when multiple TP workers try to JIT compile simultaneously
    try:
        print("\n[1/6] Warming up flash attention kernel (flash_attn_varlen_func)...")
        from aiter.ops.mha import flash_attn_varlen_func

        # Minimal test case to trigger JIT compilation
        batch_size = 2
        seq_len = 32
        num_heads = 8
        head_dim = 64

        q = torch.randn(
            batch_size * seq_len,
            num_heads,
            head_dim,
            dtype=torch.bfloat16,
            device=device,
        )
        k = torch.randn(
            batch_size * seq_len,
            num_heads,
            head_dim,
            dtype=torch.bfloat16,
            device=device,
        )
        v = torch.randn(
            batch_size * seq_len,
            num_heads,
            head_dim,
            dtype=torch.bfloat16,
            device=device,
        )
        cu_seqlens_q = torch.tensor(
            [0, seq_len, 2 * seq_len], dtype=torch.int32, device=device
        )
        cu_seqlens_k = torch.tensor(
            [0, seq_len, 2 * seq_len], dtype=torch.int32, device=device
        )

        _ = flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q=seq_len,
            max_seqlen_k=seq_len,
            softmax_scale=1.0 / (head_dim**0.5),
            causal=True,
        )
        torch.cuda.synchronize()
        print(
            "   Flash attention kernel (flash_attn_varlen_func) compiled successfully"
        )
    except Exception as e:
        print(f"   Flash attention warmup skipped (may not be available): {e}")

    # Warmup RMSNorm kernel (module_rmsnorm) - most commonly used
    # SGLang uses rmsnorm2d_fwd and rmsnorm2d_fwd_with_add from aiter
    try:
        print("\n[2/6] Warming up RMSNorm kernel (rmsnorm2d_fwd)...")
        from aiter import rmsnorm2d_fwd

        hidden_size = 4096
        batch_size = 512  # Use larger batch to match CUDA graph capture
        x = torch.randn(batch_size, hidden_size, dtype=torch.bfloat16, device=device)
        weight = torch.ones(hidden_size, dtype=torch.bfloat16, device=device)
        eps = 1e-6

        # This triggers JIT compilation
        _ = rmsnorm2d_fwd(x, weight, eps)
        torch.cuda.synchronize()
        print("   RMSNorm kernel (rmsnorm2d_fwd) compiled successfully")
    except Exception as e:
        print(f"   RMSNorm warmup failed (may not be available): {e}")

    # Warmup fused add RMSNorm kernel
    try:
        print("\n[3/6] Warming up fused add RMSNorm kernel (rmsnorm2d_fwd_with_add)...")
        from aiter import rmsnorm2d_fwd_with_add

        hidden_size = 4096
        batch_size = 512
        x = torch.randn(batch_size, hidden_size, dtype=torch.bfloat16, device=device)
        residual = torch.randn(
            batch_size, hidden_size, dtype=torch.bfloat16, device=device
        )
        weight = torch.ones(hidden_size, dtype=torch.bfloat16, device=device)
        eps = 1e-6

        # This triggers JIT compilation
        _ = rmsnorm2d_fwd_with_add(x, residual, weight, eps)
        torch.cuda.synchronize()
        print("   Fused add RMSNorm kernel compiled successfully")
    except Exception as e:
        print(f"   Fused add RMSNorm warmup failed (may not be available): {e}")

    # Warmup rotary embedding kernel if available
    try:
        print("\n[4/6] Warming up rotary embedding kernel...")
        from aiter import rotary_embedding

        head_size = 128
        seq_len = 32
        num_heads = 32
        positions = torch.arange(seq_len, device=device)
        query = torch.randn(
            seq_len, num_heads, head_size, dtype=torch.bfloat16, device=device
        )
        key = torch.randn(
            seq_len, num_heads, head_size, dtype=torch.bfloat16, device=device
        )
        cos = torch.ones(seq_len, head_size // 2, dtype=torch.bfloat16, device=device)
        sin = torch.zeros(seq_len, head_size // 2, dtype=torch.bfloat16, device=device)

        _ = rotary_embedding(positions, query, key, head_size, cos, sin, True)
        torch.cuda.synchronize()
        print("   Rotary embedding kernel compiled successfully")
    except Exception as e:
        print(f"   Rotary embedding warmup skipped (may not be available): {e}")

    # Warmup activation kernels if available
    try:
        print("\n[5/6] Warming up activation kernels...")
        from aiter import silu_and_mul

        hidden_size = 4096
        batch_size = 512
        x = torch.randn(
            batch_size, hidden_size * 2, dtype=torch.bfloat16, device=device
        )
        out = torch.empty(batch_size, hidden_size, dtype=torch.bfloat16, device=device)

        silu_and_mul(out, x)
        torch.cuda.synchronize()
        print("   Activation kernel compiled successfully")
    except Exception as e:
        print(f"   Activation warmup skipped (may not be available): {e}")

    # Warmup quantization kernel (module_quant) - used by MXFP4/FP8 models
    try:
        print("\n[6/6] Warming up quantization kernel (module_quant)...")
        from aiter.ops.quant import dynamic_per_token_scaled_quant_fp8

        batch_size = 512
        hidden_size = 4096
        x = torch.randn(batch_size, hidden_size, dtype=torch.bfloat16, device=device)
        out = torch.empty(
            batch_size, hidden_size, dtype=torch.float8_e4m3fnuz, device=device
        )
        scale = torch.empty(batch_size, 1, dtype=torch.float32, device=device)

        dynamic_per_token_scaled_quant_fp8(out, x, scale)
        torch.cuda.synchronize()
        print("   Quantization kernel (module_quant) compiled successfully")
    except Exception as e:
        print(f"   Quantization warmup skipped (may not be available): {e}")

    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"AITER warmup completed in {elapsed:.1f}s")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    warmup_aiter_kernels()
