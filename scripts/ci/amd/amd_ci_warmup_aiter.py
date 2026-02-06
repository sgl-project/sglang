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

    # Warmup RMSNorm kernel (module_rmsnorm) - most commonly used
    # SGLang uses rmsnorm2d_fwd and rmsnorm2d_fwd_with_add from aiter
    try:
        print("\n[1/4] Warming up RMSNorm kernel (rmsnorm2d_fwd)...")
        from aiter import rmsnorm2d_fwd

        hidden_size = 4096
        batch_size = 512  # Use larger batch to match CUDA graph capture
        x = torch.randn(batch_size, hidden_size, dtype=torch.bfloat16, device=device)
        weight = torch.ones(hidden_size, dtype=torch.bfloat16, device=device)
        eps = 1e-6

        # This triggers JIT compilation
        _ = rmsnorm2d_fwd(x, weight, eps)
        torch.cuda.synchronize()
        print(f"   RMSNorm kernel (rmsnorm2d_fwd) compiled successfully")
    except Exception as e:
        print(f"   RMSNorm warmup failed (may not be available): {e}")

    # Warmup fused add RMSNorm kernel
    try:
        print("\n[2/4] Warming up fused add RMSNorm kernel (rmsnorm2d_fwd_with_add)...")
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
        print(f"   Fused add RMSNorm kernel compiled successfully")
    except Exception as e:
        print(f"   Fused add RMSNorm warmup failed (may not be available): {e}")

    # Warmup rotary embedding kernel if available
    try:
        print("\n[3/4] Warming up rotary embedding kernel...")
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
        print(f"   Rotary embedding kernel compiled successfully")
    except Exception as e:
        print(f"   Rotary embedding warmup skipped (may not be available): {e}")

    # Warmup activation kernels if available
    try:
        print("\n[4/4] Warming up activation kernels...")
        from aiter import silu_and_mul

        hidden_size = 4096
        batch_size = 512
        x = torch.randn(
            batch_size, hidden_size * 2, dtype=torch.bfloat16, device=device
        )
        out = torch.empty(batch_size, hidden_size, dtype=torch.bfloat16, device=device)

        silu_and_mul(out, x)
        torch.cuda.synchronize()
        print(f"   Activation kernel compiled successfully")
    except Exception as e:
        print(f"   Activation warmup skipped (may not be available): {e}")

    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"AITER warmup completed in {elapsed:.1f}s")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    warmup_aiter_kernels()
