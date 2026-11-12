# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
"""Per-row softmax statistics for DFLASH sampled verify at any row count.

One read of the fp32 verify logits gives each row's (row_max, 1 / denominator, inv_temp, 0), the layout
``selector_logit_accept`` reads (as ``verify_softmax_wide_stats`` writes it, which covers <= 128 rows only).
row_max and inv_temp carry the served softmax's bits; the denominator is the same fp32 sum in another order, so p
evaluated from these stats differs from the served p by about the denominator's relative difference (~1e-7).
See csrc/speculative/verify_row_stats.cuh.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, is_arch_support_pdl, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module

_SLICE = 4096  # kRsSlice: columns per partial CTA


@cache_once
def _module(use_pdl: bool) -> Module:
    args = make_cpp_args(use_pdl)
    return load_jit(
        "verify_row_stats",
        *args,
        cuda_files=["speculative/verify_row_stats.cuh"],
        cuda_wrappers=[("run", f"VerifyRowStatsKernel<{args}>::run")],
        # The same flags as verify_softmax_wide: inv_temp and the max must carry the served bits.
        extra_cuda_cflags=["-use_fast_math"],
    )


def verify_row_stats_supported(logits: torch.Tensor, rows_per_request: int) -> bool:
    if not logits.is_cuda or logits.dim() != 2:
        return False
    rows, vocab = logits.shape
    return (
        rows >= 1
        and vocab >= 4
        and vocab % 4 == 0
        and rows_per_request >= 1
        and rows % rows_per_request == 0
    )


def verify_row_stats(
    *,
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    rows_per_request: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns the fp32 contiguous logits and stats fp32 [rows, 4] = (row_max, inv_denominator, inv_temp, 0) for
    softmax(logits[r] / temperatures[r // rows_per_request]); ``temperatures`` holds one value per request."""
    num_rows, vocab = logits.shape
    bs = num_rows // rows_per_request
    assert verify_row_stats_supported(logits, rows_per_request), (
        tuple(logits.shape),
        rows_per_request,
    )
    logits_2d = logits.to(torch.float32).contiguous()
    temps = temperatures.reshape(bs).to(torch.float32).contiguous()
    assert temps.device == logits_2d.device
    num_slices = (vocab + _SLICE - 1) // _SLICE
    partials = torch.empty(
        (num_rows, num_slices, 2), dtype=torch.float32, device=logits_2d.device
    )
    stats = torch.empty((num_rows, 4), dtype=torch.float32, device=logits_2d.device)
    _module(is_arch_support_pdl()).run(
        logits_2d, stats, partials, temps, rows_per_request
    )
    return logits_2d, stats
