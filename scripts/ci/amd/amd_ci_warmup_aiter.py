#!/usr/bin/env python3
"""
Warmup script to pre-build AITER JIT kernels.

This script triggers compilation of commonly used AITER kernels by importing
the relevant modules and calling functions with sample data. This avoids
timeouts during actual tests when kernels need to be compiled on first use.

Run this after clearing pre-built AITER kernels from the Docker image.
"""

import os
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

    # Warmup RMSNorm kernels - AITER has two separate modules:
    # - module_rmsnorm_quant: used when hidden_size <= 8192 (via rmsnorm function)
    # - module_rmsnorm: used when hidden_size > 8192 (via rmsnorm2d_fwd_ck function)
    # We need to trigger both modules for complete warmup.
    eps = 1e-6
    batch_size = 512  # Use larger batch to match CUDA graph capture

    # First, warmup module_rmsnorm_quant (for hidden_size <= 8192)
    try:
        print("\n[1/5] Warming up RMSNorm kernel (module_rmsnorm_quant)...")
        from aiter.ops.rmsnorm import rmsnorm

        for hidden_size in [4096, 5120, 7168, 8192]:
            x = torch.randn(
                batch_size, hidden_size, dtype=torch.bfloat16, device=device
            )
            out = torch.empty_like(x)
            weight = torch.ones(hidden_size, dtype=torch.bfloat16, device=device)

            # This triggers module_rmsnorm_quant JIT compilation
            rmsnorm(out, x, weight, eps)
            torch.cuda.synchronize()
            print(
                f"   rmsnorm (module_rmsnorm_quant, hidden_size={hidden_size}) compiled"
            )
    except Exception as e:
        print(f"   module_rmsnorm_quant warmup failed: {e}")

    # Second, warmup module_rmsnorm (CK version, for hidden_size > 8192 or explicit call)
    # Note: rmsnorm2d_fwd_ck signature: (input, weight, epsilon, use_model_sensitive_rmsnorm=0) -> Tensor
    try:
        print("\n[2/5] Warming up RMSNorm kernel (module_rmsnorm - CK version)...")
        from aiter.ops.rmsnorm import rmsnorm2d_fwd_ck

        # Use hidden_size > 8192 to trigger module_rmsnorm CK kernel
        for hidden_size in [14336, 16384]:
            x = torch.randn(
                batch_size, hidden_size, dtype=torch.bfloat16, device=device
            )
            weight = torch.ones(hidden_size, dtype=torch.bfloat16, device=device)

            # This triggers module_rmsnorm JIT compilation (CK version)
            # Signature: rmsnorm2d_fwd_ck(input, weight, epsilon, use_model_sensitive_rmsnorm=0) -> Tensor
            _ = rmsnorm2d_fwd_ck(x, weight, eps)
            torch.cuda.synchronize()
            print(
                f"   rmsnorm2d_fwd_ck (module_rmsnorm, hidden_size={hidden_size}) compiled"
            )
    except Exception as e:
        print(f"   module_rmsnorm CK warmup failed: {e}")

    # Third, warmup fused add RMSNorm CK kernel (module_rmsnorm)
    try:
        print(
            "\n[3/5] Warming up fused add RMSNorm kernel (module_rmsnorm - CK version)..."
        )
        from aiter.ops.rmsnorm import rmsnorm2d_fwd_with_add_ck

        for hidden_size in [4096, 5120, 7168, 8192, 14336]:
            x = torch.randn(
                batch_size, hidden_size, dtype=torch.bfloat16, device=device
            )
            residual = torch.randn(
                batch_size, hidden_size, dtype=torch.bfloat16, device=device
            )
            output = torch.empty_like(x)
            residual_out = torch.empty_like(x)
            weight = torch.ones(hidden_size, dtype=torch.bfloat16, device=device)

            # rmsnorm2d_fwd_with_add_ck(out, input, residual_in, residual_out, weight, epsilon, use_model_sensitive_rmsnorm)
            rmsnorm2d_fwd_with_add_ck(output, x, residual, residual_out, weight, eps, 0)
            torch.cuda.synchronize()
            print(
                f"   rmsnorm2d_fwd_with_add_ck (module_rmsnorm, hidden_size={hidden_size}) compiled"
            )
    except Exception as e:
        print(f"   Fused add RMSNorm CK warmup failed: {e}")

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
