# SPDX-License-Identifier: Apache-2.0
"""Step5 zero-centered norms with checkpoint-preserving weight precision."""

import torch

from sglang.kernels.ops.layernorm.minimax_m3_rmsnorm import (
    gemma_rmsnorm as mixed_dtype_gemma_rmsnorm,
)
from sglang.srt.layers.layernorm import GemmaRMSNorm


class Step5RMSNorm(GemmaRMSNorm):
    def __init__(self, hidden_size, eps=1e-6, *, weight_dtype=None):
        super().__init__(hidden_size, eps=eps)
        if weight_dtype is not None:
            self.weight.data = self.weight.data.to(weight_dtype)
            self.gemma_weight = self.gemma_weight.to(weight_dtype)

    def _forward_impl(self, x, residual=None, post_residual_addition=None):
        # Idle DP ranks must not launch a zero-grid CUDA norm.
        if x.numel() == 0:
            if residual is not None:
                if post_residual_addition is not None:
                    residual = residual + post_residual_addition
                return x, residual
            return x
        if (
            residual is None
            and x.is_cuda
            and x.dtype == torch.bfloat16
            and self.weight.dtype == torch.float32
        ):
            # Add one before rounding the checkpoint's FP32 weights.
            original_shape = x.shape
            if x.dim() != 2:
                x = x.contiguous().reshape(-1, original_shape[-1])
            out = mixed_dtype_gemma_rmsnorm(x, self.weight, self.variance_epsilon)
            return out.reshape(original_shape) if len(original_shape) != 2 else out
        return super()._forward_impl(x, residual, post_residual_addition)
