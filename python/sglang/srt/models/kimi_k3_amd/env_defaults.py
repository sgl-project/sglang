"""AITER settings Kimi-K3.1 is served with on ROCm."""

import os


def apply_kimi_k31_rocm_env_defaults() -> None:
    """AITER settings Kimi-K3.1 is served with on ROCm, unless the user set them."""
    # AITER's opus MoE stage 2 otherwise stores per-slot down outputs in fp8 before the
    # top-k sum: 11x the error of the bf16 partials the reference keeps.
    os.environ.setdefault("AITER_OPUS_STAGE2_FP8", "0")
    if "AITER_CONFIG_GEMM_BF16" not in os.environ:
        from sglang.srt.models.kimi_k3_amd.bf16_gemm_table import (
            write_k31_bf16_gemm_table,
        )

        os.environ["AITER_CONFIG_GEMM_BF16"] = write_k31_bf16_gemm_table()
