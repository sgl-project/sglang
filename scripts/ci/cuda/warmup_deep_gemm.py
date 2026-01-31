#!/usr/bin/env python3
"""
Warmup script to pre-compile DeepGEMM JIT kernels for CI.

This script uses sglang.compile_deep_gemm to pre-compile DeepGEMM kernels
for the test model used in CI. This ensures that tests don't timeout waiting
for JIT compilation on fresh runners.

DeepGEMM kernel compilation can take 2+ minutes per N/K configuration on
a fresh runner (no kernel cache). By warming up during install, the compiled
kernels are cached and reused during test execution (~1 sec per config).

NOTE: This script ONLY runs in CI environments (SGLANG_IS_IN_CI=true).
"""

import os
import subprocess
import sys
import time


def is_in_ci() -> bool:
    """Check if we're running in CI environment."""
    return os.environ.get("SGLANG_IS_IN_CI", "").lower() == "true"


def get_gpu_count() -> int:
    """Get the number of available CUDA GPUs."""
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.device_count()
        return 0
    except ImportError:
        return 0


def check_deep_gemm_enabled() -> bool:
    """Check if JIT DeepGEMM is enabled."""
    try:
        from sglang.srt.layers.deep_gemm_wrapper.configurer import ENABLE_JIT_DEEPGEMM

        return ENABLE_JIT_DEEPGEMM
    except ImportError:
        return False


def warmup_deep_gemm():
    """Pre-compile DeepGEMM kernels using sglang.compile_deep_gemm."""
    print("=" * 60)
    print("DeepGEMM JIT Kernel Warmup (using sglang.compile_deep_gemm)")
    print("=" * 60)

    # Check prerequisites
    gpu_count = get_gpu_count()
    if gpu_count == 0:
        print("CUDA not available, skipping DeepGEMM warmup")
        return

    # DeepEP tests require 4+ GPUs, so only warmup on 4+ GPU runners
    # This avoids running the warmup on 1-GPU runners that don't need it
    if gpu_count < 4:
        print(f"Only {gpu_count} GPU(s) available, need 4+ for DeepEP tests")
        print("Skipping DeepGEMM warmup (not needed for this runner)")
        return

    if not check_deep_gemm_enabled():
        print("JIT DeepGEMM is disabled, skipping warmup")
        return

    start_time = time.time()

    # Test model used by DeepEP tests (lmsys/sglang-ci-dsv3-test)
    # This model has different N/K dimensions than full DeepSeek-V3
    test_model = "lmsys/sglang-ci-dsv3-test"

    print(f"\nDetected {gpu_count} GPUs")
    print(f"Pre-compiling DeepGEMM kernels for: {test_model}")
    print("This may take a few minutes on a fresh runner...")
    print()

    # Run sglang.compile_deep_gemm with the test model
    # --tp 4 matches the DeepEP test configuration
    # --timeout 600 allows enough time for compilation
    cmd = [
        sys.executable,
        "-m",
        "sglang.compile_deep_gemm",
        "--model",
        test_model,
        "--tp",
        "4",
        "--trust-remote-code",
        "--timeout",
        "600",
    ]

    print(f"Running: {' '.join(cmd)}")
    print()

    try:
        result = subprocess.run(
            cmd,
            capture_output=False,
            text=True,
            timeout=700,  # Allow extra time beyond the internal timeout
        )

        elapsed = time.time() - start_time

        if result.returncode == 0:
            print()
            print("=" * 60)
            print(f"DeepGEMM warmup completed successfully in {elapsed:.1f}s")
            print("=" * 60)
        else:
            print()
            print("=" * 60)
            print(f"DeepGEMM warmup finished with return code {result.returncode}")
            print(f"Elapsed time: {elapsed:.1f}s")
            print("Kernels may be partially compiled - tests should still benefit")
            print("=" * 60)

    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        print()
        print("=" * 60)
        print(f"DeepGEMM warmup timed out after {elapsed:.1f}s")
        print("Kernels may be partially compiled - tests should still benefit")
        print("=" * 60)

    except Exception as e:
        elapsed = time.time() - start_time
        print()
        print("=" * 60)
        print(f"DeepGEMM warmup failed after {elapsed:.1f}s: {e}")
        print("Tests will compile kernels on-demand (may be slower)")
        print("=" * 60)


if __name__ == "__main__":
    # Exit immediately if not in CI
    if not is_in_ci():
        print(
            "Not in CI environment (SGLANG_IS_IN_CI != true), skipping DeepGEMM warmup"
        )
        sys.exit(0)

    warmup_deep_gemm()
