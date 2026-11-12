# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
"""Temperature-scaled target softmax for DFLASH sampled verify, bitwise equal to
``flashinfer.sampling.softmax`` (0.7.2, vocabulary-split path) on the same rows.

flashinfer's split path normalizes each row in one 1024-thread CTA (8 CTAs write
the whole [8, 163840] fp32 result at bs 1) and needs a per-row temperature copy
(``torch.repeat_interleave``). This module keeps flashinfer's arithmetic, reads
the per-request temperature directly, and spreads the normalization over
(row, 4096-column tile) CTAs. See csrc/speculative/verify_softmax_wide.cuh.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, is_arch_support_pdl, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module

# flashinfer 0.7.2 OnlineSoftmax: the vocabulary-split path runs for batch <= 128
# and d >= 24576 (SMALL_BATCH_THRESHOLD / LARGE_VOCAB_THRESHOLD); VEC_SIZE is
# gcd(4, d), and this kernel is written for VEC_SIZE 4.
_MAX_ROWS = 128
_MIN_VOCAB = 24576
_SLICE = 8192


@cache_once
def _module(use_pdl: bool) -> Module:
    args = make_cpp_args(use_pdl)
    return load_jit(
        "verify_softmax_wide",
        *args,
        cuda_files=["speculative/verify_softmax_wide.cuh"],
        cuda_wrappers=[("run", f"VerifySoftmaxWideKernel<{args}>::run")],
        # flashinfer builds its sampling module with -use_fast_math; the division
        # and exp instructions must be the same ones for the bits to match.
        extra_cuda_cflags=["-use_fast_math"],
    )


@cache_once
def _stats_module(use_pdl: bool) -> Module:
    args = make_cpp_args(use_pdl)
    return load_jit(
        "verify_softmax_stats",
        *args,
        cuda_files=["speculative/verify_softmax_stats.cuh"],
        cuda_wrappers=[("run", f"VerifySoftmaxStatsKernel<{args}>::run")],
        # The same flags as _module: the stats must carry the norm kernel's bits.
        extra_cuda_cflags=["-use_fast_math"],
    )


def verify_softmax_wide_supported(logits: torch.Tensor, rows_per_request: int) -> bool:
    """True when flashinfer would take its vocabulary-split path with VEC_SIZE 4."""
    if not logits.is_cuda or logits.dim() != 2:
        return False
    rows, vocab = logits.shape
    return (
        1 <= rows <= _MAX_ROWS
        and vocab >= _MIN_VOCAB
        and vocab % 4 == 0
        and rows_per_request >= 1
        and rows % rows_per_request == 0
    )


def verify_softmax_wide(
    *,
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    rows_per_request: int,
) -> torch.Tensor:
    """softmax(logits[r] / temperatures[r // rows_per_request]) per row, fp32,
    with flashinfer's bits. ``temperatures`` holds one value per request."""
    num_rows, vocab = logits.shape
    bs = num_rows // rows_per_request
    assert verify_softmax_wide_supported(logits, rows_per_request), (
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
    probs = torch.empty_like(logits_2d)
    _module(is_arch_support_pdl()).run(
        logits_2d, probs, partials, temps, rows_per_request
    )
    return probs


def verify_softmax_wide_stats(
    *,
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    rows_per_request: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """verify_softmax_wide without the [rows, vocab] probabilities: returns the fp32
    contiguous logits and per-row stats fp32 [rows, 4] = (row_max, inv_denominator,
    inv_temp, 0). __expf(__fmul_rn(logit, inv_temp) - row_max) * inv_denominator,
    evaluated with the fast-math instructions, is verify_softmax_wide's p bit for bit
    (see csrc/speculative/verify_softmax_stats.cuh)."""
    num_rows, vocab = logits.shape
    bs = num_rows // rows_per_request
    assert verify_softmax_wide_supported(logits, rows_per_request), (
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
    _stats_module(is_arch_support_pdl()).run(
        logits_2d, stats, partials, temps, rows_per_request
    )
    return logits_2d, stats
