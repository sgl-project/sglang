# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""BF16 GEMMs for 1-16 rows on SM100-family GPUs: FlashInfer's CuTe DSL ``mm_bf16``
low-M kernels (BF16 out) and the split-K ``ll_bf16`` kernels (FP32 out)."""

from __future__ import annotations

from functools import lru_cache

import torch

_CUTEDSL_BACKEND = "cute-dsl"


@lru_cache(maxsize=1)
def cutedsl_low_m_supported() -> bool:
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    if major != 10:
        return False
    try:
        from flashinfer import mm_bf16

        return bool(mm_bf16.is_backend_supported(_CUTEDSL_BACKEND, major * 10 + minor))
    except ImportError:
        return False


@lru_cache(maxsize=1)
def _use_pdl() -> bool:
    from sglang.kernels.jit.utils.arch import is_arch_support_pdl

    return is_arch_support_pdl()


def cutedsl_low_m_mm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """BF16 ``x @ weight.T``; the caller checks rows, dtype and contiguity."""
    from flashinfer import mm_bf16

    # CuTe DSL takes operands over DLPack, which refuses tensors that require grad.
    return mm_bf16(
        x.detach(), weight.detach().t(), pdl=_use_pdl(), backend=_CUTEDSL_BACKEND
    )


def fits(x: torch.Tensor, max_rows: int) -> bool:
    return (
        0 < x.shape[0] <= max_rows
        and x.dim() == 2
        and x.dtype == torch.bfloat16
        and x.is_contiguous()
    )


def ll_bf16_supported() -> bool:
    from sglang.srt.batch_invariant_ops import is_batch_invariant_mode_enabled
    from sglang.srt.models.dsv41_low_latency import ll_bf16

    return ll_bf16.is_available() and not is_batch_invariant_mode_enabled()
