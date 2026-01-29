#!/usr/bin/env python3
"""
Warmup script to pre-compile DeepGEMM JIT kernels for CI.

This script triggers compilation of commonly used DeepGEMM kernels during
the CI install phase, so tests don't timeout waiting for JIT compilation.

DeepGEMM kernel compilation can take ~60 seconds for 128 kernels, which
causes test timeouts. By warming up during install, the compiled kernels
are cached and reused during test execution.

NOTE: This script ONLY runs in CI environments (SGLANG_IS_IN_CI=true).
"""

import os
import sys
import time


def is_in_ci() -> bool:
    """Check if we're running in CI environment."""
    return os.environ.get("SGLANG_IS_IN_CI", "").lower() == "true"


# Exit immediately if not in CI
if not is_in_ci():
    print("Not in CI environment (SGLANG_IS_IN_CI != true), skipping DeepGEMM warmup")
    sys.exit(0)


def warmup_deep_gemm():
    """Pre-compile DeepGEMM kernels for common configurations."""
    print("=" * 60)
    print("DeepGEMM JIT Kernel Warmup")
    print("=" * 60)

    start_time = time.time()

    # Check if CUDA is available
    try:
        import torch

        if not torch.cuda.is_available():
            print("CUDA not available, skipping DeepGEMM warmup")
            return
    except ImportError:
        print("PyTorch not installed, skipping DeepGEMM warmup")
        return

    # Check if DeepGEMM is enabled and available
    try:
        from sglang.srt.layers.deep_gemm_wrapper.configurer import ENABLE_JIT_DEEPGEMM

        if not ENABLE_JIT_DEEPGEMM:
            print("JIT DeepGEMM is disabled, skipping warmup")
            return

        import deep_gemm
    except ImportError as e:
        print(f"DeepGEMM not available: {e}")
        print("Skipping warmup")
        return

    device = torch.device("cuda:0")
    print(f"Using device: {device}")
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    # Common kernel configurations for DeepSeek models
    # These are typical N/K values from DeepSeek-V3 architecture
    CONFIGS = [
        # (N, K, num_groups) - typical MoE configurations
        (2048, 7168, 1),  # FFN intermediate
        (7168, 2048, 1),  # FFN output
        (1536, 7168, 1),  # Another common config
        (7168, 1536, 1),
    ]

    # M values to pre-compile (subset to avoid timeout)
    # Focus on common batch sizes used in tests
    M_VALUES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

    print(f"\nPre-compiling {len(CONFIGS)} kernel configurations")
    print(f"M values: {M_VALUES}")
    print()

    total_compiled = 0

    for config_idx, (n, k, num_groups) in enumerate(CONFIGS):
        print(
            f"[{config_idx + 1}/{len(CONFIGS)}] Config: N={n}, K={k}, num_groups={num_groups}"
        )

        try:
            # Create input tensors
            from sglang.srt.utils import ceil_div

            block_size = 128
            max_m = max(M_VALUES)

            # LHS: (max_m, k) with row-wise scaling
            lhs_q = torch.empty((max_m, k), device=device, dtype=torch.float8_e4m3fn)
            lhs_s = torch.empty(
                (max_m, ceil_div(k, block_size)), device=device, dtype=torch.float32
            )

            # RHS: (n, k) with block-wise scaling
            rhs_q = torch.empty((n, k), device=device, dtype=torch.float8_e4m3fn)
            rhs_s = torch.empty(
                (ceil_div(n, block_size), ceil_div(k, block_size)),
                device=device,
                dtype=torch.float32,
            )

            # Output
            out = torch.empty((max_m, n), device=device, dtype=torch.bfloat16)

            # Set compile mode to force compilation
            old_mode = deep_gemm.get_compile_mode()
            deep_gemm.set_compile_mode(1)

            compiled_count = 0
            for m in M_VALUES:
                try:
                    deep_gemm.fp8_gemm_nt(
                        (lhs_q[:m], lhs_s[:m]),
                        (rhs_q, rhs_s),
                        out[:m],
                    )
                    compiled_count += 1
                except Exception as e:
                    print(f"  Warning: Failed to compile M={m}: {e}")

            deep_gemm.set_compile_mode(old_mode)
            torch.cuda.synchronize()

            print(f"  Compiled {compiled_count}/{len(M_VALUES)} kernels")
            total_compiled += compiled_count

            # Clean up
            del lhs_q, lhs_s, rhs_q, rhs_s, out
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"  Error warming up config: {e}")
            continue

    elapsed = time.time() - start_time
    print()
    print("=" * 60)
    print(f"DeepGEMM warmup completed in {elapsed:.1f}s")
    print(f"Total kernels compiled: {total_compiled}")
    print("=" * 60)


if __name__ == "__main__":
    warmup_deep_gemm()
