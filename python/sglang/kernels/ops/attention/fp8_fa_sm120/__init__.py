# SPDX-License-Identifier: Apache-2.0
"""SM120 FP8 attention: CuTe-DSL kernel, fused Triton prep and the per-call entry point."""

from sglang.kernels.ops.attention.fp8_fa_sm120.plan import HEAD_DIM, fp8_attention

__all__ = ["HEAD_DIM", "fp8_attention"]
