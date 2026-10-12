# SPDX-License-Identifier: Apache-2.0
"""Scoped precision policy for strict FP32 numerical oracles."""

from contextlib import contextmanager

import torch


@contextmanager
def strict_fp32_oracle():
    """Disable reduced-precision FP32 math, then restore the caller's policy.

    Call inside each distributed worker. This is a test-only, process-global
    policy; production BF16 checks should run outside this context.
    """
    precision = torch.get_float32_matmul_precision()
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    benchmark = torch.backends.cudnn.benchmark
    try:
        torch.set_float32_matmul_precision("highest")
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        yield
    finally:
        torch.backends.cudnn.benchmark = benchmark
        torch.backends.cudnn.allow_tf32 = cudnn_tf32
        torch.set_float32_matmul_precision(precision)
