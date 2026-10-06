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

    # Warmup module_rmsnorm_quant (small module, ~2MB)
    # Triggered by rmsnorm2d_fwd when hidden_size <= 8192
    try:
        print(
            "\n[1/5] Warming up module_rmsnorm_quant (rmsnorm2d_fwd, hidden<=8192)..."
        )
        from aiter import rmsnorm2d_fwd

        hidden_size = 4096
        batch_size = 512  # Use larger batch to match CUDA graph capture
        x = torch.randn(batch_size, hidden_size, dtype=torch.bfloat16, device=device)
        weight = torch.ones(hidden_size, dtype=torch.bfloat16, device=device)
        eps = 1e-6

        # hidden_size=4096 <= 8192 -> takes rmsnorm() path -> compiles module_rmsnorm_quant
        _ = rmsnorm2d_fwd(x, weight, eps)
        torch.cuda.synchronize()
        print("   module_rmsnorm_quant compiled successfully")
    except Exception as e:
        print(f"   module_rmsnorm_quant warmup failed: {e}")

    # Warmup module_rmsnorm (large CK module, ~159MB)
    # Triggered by rmsnorm2d_fwd_with_add (always uses CK path)
    # NOTE: rmsnorm2d_fwd_with_add signature is:
    #   rmsnorm2d_fwd_with_add(out, input, residual_in, residual_out, weight, epsilon)
    try:
        print("\n[2/5] Warming up module_rmsnorm (rmsnorm2d_fwd_with_add, CK path)...")
        from aiter import rmsnorm2d_fwd_with_add

        hidden_size = 4096
        batch_size = 512
        x = torch.randn(batch_size, hidden_size, dtype=torch.bfloat16, device=device)
        residual_in = torch.randn(
            batch_size, hidden_size, dtype=torch.bfloat16, device=device
        )
        output = torch.empty_like(x)
        residual_out = torch.empty_like(x)
        weight = torch.ones(hidden_size, dtype=torch.bfloat16, device=device)
        eps = 1e-6

        # This triggers JIT compilation of module_rmsnorm (CK kernels)
        rmsnorm2d_fwd_with_add(output, x, residual_in, residual_out, weight, eps)
        torch.cuda.synchronize()
        print("   module_rmsnorm compiled successfully")
    except Exception as e:
        print(f"   module_rmsnorm warmup failed: {e}")

    # Warmup module_rmsnorm via rmsnorm2d_fwd with large hidden_size (CK path)
    # When hidden_size > 8192, rmsnorm2d_fwd takes the rmsnorm2d_fwd_ck path
    # which also uses module_rmsnorm (already compiled in step 2, but this
    # ensures the CK rmsnorm2d_fwd path is exercised as well)
    try:
        print("\n[3/5] Warming up rmsnorm2d_fwd CK path (hidden>8192)...")
        from aiter import rmsnorm2d_fwd

        hidden_size = 16384  # > 8192 to trigger rmsnorm2d_fwd_ck (module_rmsnorm)
        batch_size = 32
        x = torch.randn(batch_size, hidden_size, dtype=torch.bfloat16, device=device)
        weight = torch.ones(hidden_size, dtype=torch.bfloat16, device=device)
        eps = 1e-6

        _ = rmsnorm2d_fwd(x, weight, eps)
        torch.cuda.synchronize()
        print("   rmsnorm2d_fwd CK path compiled successfully")
    except Exception as e:
        print(f"   rmsnorm2d_fwd CK path warmup skipped: {e}")

    # Warmup rotary embedding kernel if available
    try:
        print("\n[4/5] Warming up rotary embedding kernel...")
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
        print("\n[5/5] Warming up activation kernels...")
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

    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"AITER warmup completed in {elapsed:.1f}s")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    warmup_aiter_kernels()
