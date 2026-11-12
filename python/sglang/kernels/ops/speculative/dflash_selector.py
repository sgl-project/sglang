# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
"""Kernels of the fused DFlash2 selector head (SGLANG_OPT_DFLASH_FUSED_SELECTOR).

``dflash_selector_local_topk`` replaces flashinfer's local top-k, its int64 copy, the shard offset and the FP32 cast;
``dflash_selector_select`` replaces torch.topk + torch.gather of the gathered candidates and the inductor gathers of
the lattice operands (one launch each, jit/csrc/speculative/dflash_selector.cuh); ``selector_walk_fused`` is
dflash.py's ``_selector_walk_kernel`` reading the lattice GEMM output plus the unary logits in place and storing
straight into the draft sampler's static buffers.

SGLANG_OPT_DFLASH_SELECTOR_LOCAL_WIDE=1 (default off = unchanged): the local top-k kernel also takes rows above flashinfer's
FilteredTopK bound (local_topk_matches_flashinfer), in FilteredTopK's own tie order (_filtered_order), so it returns what
flashinfer.top_k returns there: the same set (deterministic FilteredTopK collects the pivot ties thread-strided over 1024
threads, as the radix path does) and FinalizeTopKIndices' index order among equal values.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import torch
import triton
import triton.language as tl

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module

logger = logging.getLogger(__name__)


@cache_once
def _select_module(width: int, top_k: int, rank: int, id_dtype: torch.dtype) -> Module:
    args = make_cpp_args(width, top_k, rank, id_dtype)
    return load_jit(
        "dflash_selector_select",
        *args,
        cuda_files=["speculative/dflash_selector.cuh"],
        cuda_wrappers=[("run", f"DFlashSelectorSelect<{args}>::run")],
    )


def dflash_selector_select(
    *,
    gathered_vals: torch.Tensor,
    gathered_ids: torch.Tensor,
    hidden: torch.Tensor,
    anchors: torch.Tensor,
    pred_table: torch.Tensor,
    succ_table: torch.Tensor,
    candidates: torch.Tensor,
    unary: torch.Tensor,
    pred_h: torch.Tensor,
    keys: torch.Tensor,
) -> None:
    """Per draft row n = (b, e) of gathered [bs * slots, width]: the top-k exactly as torch.topk returns it, then
    candidates[n] = gathered_ids[n, sel], unary[n] = values, keys[n, c] = succ_table[candidates[n, c]],
    pred_h[n, p] = bf16(pred_table[pred_id] * hidden[b, e]) with pred_id the anchor at e = 0, else candidates[n - 1, p].
    """
    _, top_k, rank = pred_h.shape
    module = _select_module(
        gathered_vals.shape[1], top_k, rank, gathered_ids.dtype
    )
    module.run(
        gathered_vals,
        gathered_ids,
        hidden,
        anchors,
        pred_table,
        succ_table,
        candidates,
        unary,
        pred_h,
        keys,
    )


# flashinfer's radix top-k block size: its deterministic tie order is thread-strided over 1024 threads.
_LOCAL_THREADS = 1024


@cache_once
def _local_topk_module(per_thread: int, top_k: int, index_order: bool) -> Module:
    args = make_cpp_args(per_thread, top_k, index_order)
    return load_jit(
        "dflash_selector_local_topk",
        *args,
        cuda_files=["speculative/dflash_selector.cuh"],
        cuda_wrappers=[("run", f"DFlashSelectorLocalTopk<{args}>::run")],
    )


def local_topk_supported(length: int) -> bool:
    """Whether dflash_selector_local_topk takes rows of this length (its position code needs <= 64 per thread)."""
    per_thread = -(-length // _LOCAL_THREADS)
    return 2 <= per_thread <= 64


def local_topk_matches_flashinfer(rows: int, length: int) -> bool:
    """Whether flashinfer.top_k(BF16 logits [rows, length], deterministic=True) takes its multi-CTA radix path, the one
    whose tie order dflash_selector_local_topk reproduces. flashinfer's ShouldUseFilteredTopK (include/flashinfer/topk.cuh)
    switches 16-bit deterministic rows to FilteredTopK when rows > length / 256 (a TP16 shard of 10240: above 40 rows),
    which orders equal values by index (_filtered_order); FLASHINFER_TOPK_ALGO overrides that choice. False means: call
    flashinfer itself (always identical; also when the GPU lacks FilteredTopK's shared memory and stays on radix).
    With SGLANG_OPT_DFLASH_SELECTOR_LOCAL_WIDE=1 (and FLASHINFER_TOPK_ALGO unset) it is True above the bound as well: the
    local kernel then takes FilteredTopK's index order among equal values (module doc). It assumes FilteredTopK runs
    there, i.e. a GPU with FilteredTopK's 128 KB of dynamic shared memory (Blackwell / Rubin)."""
    algo = os.environ.get("FLASHINFER_TOPK_ALGO")
    if algo == "multi_cta":
        return True
    if algo == "filtered":
        return False
    if rows <= length // 256:
        return True
    if _local_wide():
        # SGLANG_OPT_DFLASH_SELECTOR_LOCAL_WIDE: take the local kernel above the bound too (AL class, see the module doc).
        if not _WIDE_LOGGED:
            _WIDE_LOGGED.add(True)
            logger.info(
                "DFlash selector LOCAL_WIDE ENGAGED: %d rows of %d logits take the local top-k above flashinfer's "
                "FilteredTopK bound %d (FilteredTopK tie order: index order among equal values)", rows, length, length // 256,
            )
        return True
    return False


_WIDE_KNOB = "SGLANG_OPT_DFLASH_SELECTOR_LOCAL_WIDE"
_WIDE = []
_WIDE_LOGGED = set()


def _local_wide() -> bool:
    """SGLANG_OPT_DFLASH_SELECTOR_LOCAL_WIDE, parsed once: unset / '' / '0' = off, '1' = on, else ValueError."""
    if not _WIDE:
        v = os.environ.get(_WIDE_KNOB, "")
        if v not in ("", "0", "1"):
            raise ValueError(f"{_WIDE_KNOB} must be unset, '', '0' or '1', got {v!r}")
        _WIDE.append(v == "1")
    return _WIDE[0]


def _filtered_order(rows: int, length: int) -> bool:
    """Whether flashinfer.top_k(BF16 [rows, length], k, sorted=True, deterministic=True) runs FilteredTopK here
    (topk.cuh ShouldUseFilteredTopK: rows > length / 256 for 16-bit deterministic rows, or FLASHINFER_TOPK_ALGO=filtered).
    Its set is the radix path's (keys above the pivot, then the pivot ties thread-strided over 1024 threads), but
    FinalizeTopKIndices sorts the set by index before StableSortTopKByValue, so equal values come out in index order;
    the radix path keeps the thread-strided collect order."""
    algo = os.environ.get("FLASHINFER_TOPK_ALGO")
    if algo == "filtered":
        return True
    return algo != "multi_cta" and rows > length // 256


def dflash_selector_local_topk(*, logits: torch.Tensor, k: int, offset: int) -> tuple:
    """flashinfer.top_k(logits, k, sorted=True, deterministic=True) per row, returned as (int64 global ids = index +
    offset, FP32 values) in one launch; logits BF16 [rows, length]."""
    rows, length = logits.shape
    global_ids = torch.empty((rows, k), dtype=torch.int64, device=logits.device)
    vals = torch.empty((rows, k), dtype=torch.float32, device=logits.device)
    _local_topk_module(-(-length // _LOCAL_THREADS), k, _filtered_order(rows, length)).run(
        logits, vals, global_ids, offset
    )
    return global_ids, vals


@triton.jit
def _selector_walk_fused_kernel(
    lattice_ptr,
    unary_ptr,
    candidate_ptr,
    uniforms_ptr,
    temperatures_ptr,
    greedy_ptr,
    tokens_ptr,
    q_ptr,
    slots: tl.constexpr,
    top_k: tl.constexpr,
):
    # Keep the arithmetic below identical to dflash.py _selector_walk_kernel: the walk must stay bitwise equal.
    row = tl.program_id(0)
    offsets = tl.arange(0, top_k)
    temperature = tl.load(temperatures_ptr + row)
    greedy = tl.load(greedy_ptr + row) != 0
    previous = 0
    for slot in range(slots):
        base = (row * slots + slot) * top_k
        # The selector's fp32 score: unary logit + the bf16 lattice GEMM output, as _score_edges adds them.
        scores = tl.load(unary_ptr + base + offsets) + tl.load(
            lattice_ptr + (base + previous) * top_k + offsets
        ).to(tl.float32)
        if greedy:
            # Same pick as torch.argmax: the first NaN if any, else the first max.
            is_nan = scores != scores
            best = tl.max(tl.where(is_nan, -float("inf"), scores), axis=0)
            has_nan = tl.max(is_nan.to(tl.int32), axis=0) > 0
            hit = tl.where(has_nan, is_nan, scores == best)
            index = tl.min(tl.where(hit, offsets, top_k), axis=0)
            probabilities = tl.where(offsets == index, 1.0, 0.0)
        else:
            scaled = scores / temperature
            exponentials = tl.exp(scaled - tl.max(scaled, axis=0))
            probabilities = exponentials / tl.sum(exponentials, axis=0)
            uniform = tl.load(uniforms_ptr + row * slots + slot)
            index = tl.sum(
                tl.where(uniform >= tl.cumsum(probabilities, axis=0), 1, 0), axis=0
            )
            index = tl.minimum(index, top_k - 1)
        tl.store(q_ptr + base + offsets, probabilities)
        tl.store(tokens_ptr + row * slots + slot, tl.load(candidate_ptr + base + index))
        previous = index


def selector_walk_fused(
    *,
    lattice: torch.Tensor,
    unary: torch.Tensor,
    candidates: torch.Tensor,
    uniforms: torch.Tensor,
    temperatures: torch.Tensor,
    greedy_mask: torch.Tensor,
    tokens: torch.Tensor,
    q_rows: torch.Tensor,
) -> None:
    """selector_walk_triton on scores = unary[:, :, None] + lattice, writing tokens [bs * slots] int64 and
    q_rows [bs, slots, top_k] fp32 in place. lattice is bf16 [bs, slots, top_k, top_k]; every tensor contiguous."""
    batch, slots, top_k = candidates.shape
    assert lattice.shape == (batch, slots, top_k, top_k) and lattice.is_contiguous()
    assert lattice.dtype == torch.bfloat16 and unary.dtype == torch.float32
    assert unary.shape == (batch * slots, top_k) and unary.is_contiguous()
    assert candidates.dtype == torch.int64 and candidates.is_contiguous()
    assert tokens.dtype == torch.int64 and tokens.shape == (batch * slots,)
    assert q_rows.shape == (batch, slots, top_k) and q_rows.is_contiguous()
    assert uniforms.shape == (batch, slots) and uniforms.is_contiguous()
    _selector_walk_fused_kernel[(batch,)](
        lattice,
        unary,
        candidates,
        uniforms,
        temperatures,
        greedy_mask,
        tokens,
        q_rows,
        slots=slots,
        top_k=top_k,
        num_warps=1,
    )
