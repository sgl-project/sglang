# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
"""Kernels of the DFlash target-context stage: a cuBLASLt GEMM run with an explicitly pinned algorithm, and the
fused RoPE + FP8 store of every MLA draft layer's context K rows."""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Sequence

import torch
import triton
import triton.language as tl

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module

# The workspace limit torch's cuBLAS handle passes to cublasLt (the 32 MiB it logs on SM100+).
TORCH_CUBLAS_WORKSPACE_BYTES = 32 << 20
_NUM_HEURISTICS = 16


@cache_once
def _pinned_lt_module() -> Module:
    return load_jit(
        "pinned_lt_matmul",
        cuda_files=["gemm/pinned_lt_matmul.cuh"],
        cuda_wrappers=[
            ("heuristics", "PinnedLtMatmul::heuristics"),
            ("matmul", "PinnedLtMatmul::matmul"),
        ],
        extra_ldflags=["-lcublasLt"],
    )


def lt_heuristics(
    rows: int, weight: torch.Tensor, ws_bytes: int = TORCH_CUBLAS_WORKSPACE_BYTES
) -> List[torch.Tensor]:
    """cuBLASLt's ranked algorithms for F.linear(x[rows, K], weight): each an int64 [9] CPU tensor (8 algo
    words, then the workspace bytes it needs)."""
    out = torch.zeros(_NUM_HEURISTICS, 9, dtype=torch.int64)
    n_out, k = weight.shape
    got = _pinned_lt_module().heuristics(rows, n_out, k, weight.stride(0), ws_bytes, out)
    return [out[i].clone() for i in range(got)]


def lt_matmul(
    x: torch.Tensor,
    weight: torch.Tensor,
    algo: torch.Tensor,
    workspace: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """x[M, K] @ weight[N, K]^T with the given algorithm (BF16 in/out, FP32 compute)."""
    if out is None:
        out = torch.empty(x.shape[0], weight.shape[0], dtype=x.dtype, device=x.device)
    _pinned_lt_module().matmul(x, weight, out, algo[:8], workspace)
    return out


@cache_once
def _ctx_kv_store_module(use_pdl: bool) -> Module:
    args = make_cpp_args(use_pdl)
    return load_jit(
        "dflash_ctx_kv_store",
        *args,
        cuda_files=["speculative/dflash_ctx_kv_store.cuh"],
        cuda_wrappers=[("run", f"DFlashCtxKVStoreKernel<{args}>::run")],
    )


def dflash_ctx_kv_store(
    k_nope: Sequence[torch.Tensor],
    latent: Sequence[torch.Tensor],
    kv_buffers: Sequence[torch.Tensor],
    cos_sin_caches: Sequence[torch.Tensor],
    positions: torch.Tensor,
    loc: torch.Tensor,
    commit_lens: Optional[torch.Tensor] = None,
) -> None:
    """Per layer l and row i: RoPE latent[l][i, 512:576] in place, then write
    fp8(k_nope[l][i]) | fp8(roped k_pe) to kv_buffers[l][loc_i], skipping loc_i == 0.

    loc is [rows] int64, or [bs, width] int64 with commit_lens [bs] int32: row (b, j) then
    writes to loc[b, j] when j < commit_lens[b], else to the skipped slot 0.
    """
    rows = positions.shape[0]
    layers = len(k_nope)
    assert layers == len(latent) == len(kv_buffers) == len(cos_sin_caches)
    for kn, lt, kv, cs in zip(k_nope, latent, kv_buffers, cos_sin_caches):
        assert kn.dtype == lt.dtype == torch.bfloat16 and kv.dtype == torch.uint8
        assert kn.shape == (rows, 512) and kn.stride(1) == 1 and kn.stride(0) == k_nope[0].stride(0)
        assert lt.shape[0] == rows and lt.shape[1] >= 576 and lt.stride(1) == 1
        assert lt.stride(0) == latent[0].stride(0)
        assert kv.dim() == 2 and kv.shape[1] == 576 and kv.stride(0) == kv_buffers[0].stride(0)
        assert cs.dtype == torch.float32 and cs.dim() == 2 and cs.shape[1] == 64
        assert cs.is_contiguous()
    assert positions.dtype == torch.int64 and loc.dtype == torch.int64 and loc.numel() == rows
    assert loc.is_contiguous() and positions.is_contiguous()
    if commit_lens is not None:
        assert loc.dim() == 2 and commit_lens.dtype == torch.int32
        assert commit_lens.shape == (loc.shape[0],) and commit_lens.is_contiguous()
        width = loc.shape[1]
    else:
        width = 1
    ptrs = torch.tensor(
        [
            [t.data_ptr() for t in k_nope],
            [t.data_ptr() for t in latent],
            [t.data_ptr() for t in kv_buffers],
            [t.data_ptr() for t in cos_sin_caches],
        ],
        dtype=torch.int64,
    )
    _ctx_kv_store_module(is_arch_support_pdl()).run(
        ptrs,
        positions,
        loc,
        commit_lens if commit_lens is not None else positions,
        rows,
        width,
        k_nope[0].stride(0) * 2,
        latent[0].stride(0) * 2,
        kv_buffers[0].stride(0),
        int(commit_lens is not None),
    )


@cache_once
def _ctx_kv_store_norm_module(use_pdl: bool) -> Module:
    args = make_cpp_args(use_pdl)
    return load_jit(
        "dflash_ctx_kv_store_norm",
        *args,
        cuda_files=["speculative/dflash_ctx_kv_store.cuh"],
        cuda_wrappers=[("run", f"DFlashCtxKVStoreNormKernel<{args}>::run")],
    )


def dflash_ctx_kv_store_norm(
    latent: Sequence[torch.Tensor],
    norm_weights: Sequence[torch.Tensor],
    norm_eps: Sequence[float],
    kv_buffers: Sequence[torch.Tensor],
    cos_sin_caches: Sequence[torch.Tensor],
    positions: torch.Tensor,
    loc: torch.Tensor,
    commit_lens: Optional[torch.Tensor] = None,
    dbg_out: Optional[torch.Tensor] = None,
) -> None:
    """dflash_ctx_kv_store with the kv_a RMSNorm folded in: per layer l and row i, k_nope = RMSNorm(latent[l][i, :512],
    norm_weights[l], norm_eps[l]) in flashinfer's H = 512 BF16 arithmetic, then the same RoPE + FP8 store. dbg_out (test
    only): bf16 [layers, rows, 512] receives the normalized rows."""
    rows = positions.shape[0]
    layers = len(latent)
    assert layers == len(norm_weights) == len(norm_eps) == len(kv_buffers) == len(cos_sin_caches)
    for lt, w, kv, cs in zip(latent, norm_weights, kv_buffers, cos_sin_caches):
        assert lt.dtype == torch.bfloat16 and kv.dtype == torch.uint8
        assert lt.shape[0] == rows and lt.shape[1] >= 576 and lt.stride(1) == 1
        assert lt.stride(0) == latent[0].stride(0) and (lt.stride(0) * 2) % 16 == 0 and lt.data_ptr() % 16 == 0
        assert w.dtype == torch.bfloat16 and w.shape == (512,) and w.is_contiguous() and w.data_ptr() % 16 == 0
        assert kv.dim() == 2 and kv.shape[1] == 576 and kv.stride(0) == kv_buffers[0].stride(0)
        assert cs.dtype == torch.float32 and cs.dim() == 2 and cs.shape[1] == 64
        assert cs.is_contiguous()
    assert positions.dtype == torch.int64 and loc.dtype == torch.int64 and loc.numel() == rows
    assert loc.is_contiguous() and positions.is_contiguous()
    if commit_lens is not None:
        assert loc.dim() == 2 and commit_lens.dtype == torch.int32
        assert commit_lens.shape == (loc.shape[0],) and commit_lens.is_contiguous()
        width = loc.shape[1]
    else:
        width = 1
    if dbg_out is not None:
        assert dbg_out.dtype == torch.bfloat16 and dbg_out.shape == (layers, rows, 512) and dbg_out.is_contiguous()
    ptrs = torch.tensor(
        [
            [t.data_ptr() for t in latent],
            [t.data_ptr() for t in kv_buffers],
            [t.data_ptr() for t in cos_sin_caches],
            [t.data_ptr() for t in norm_weights],
        ],
        dtype=torch.int64,
    )
    eps = torch.tensor([float(e) for e in norm_eps], dtype=torch.float32)
    _ctx_kv_store_norm_module(is_arch_support_pdl()).run(
        ptrs,
        eps,
        positions,
        loc,
        commit_lens if commit_lens is not None else positions,
        dbg_out if dbg_out is not None else positions,
        rows,
        width,
        latent[0].stride(0) * 2,
        kv_buffers[0].stride(0),
        int(commit_lens is not None),
        int(dbg_out is not None),
    )


@triton.jit
def _compact_draft_window_kernel(
    seq_lens_ptr, draft_lens_ptr, suffix_start_ptr, n, window, page_size, PAGED: tl.constexpr, BLOCK: tl.constexpr
):
    offs = tl.arange(0, BLOCK)
    mask = offs < n
    seq = tl.load(seq_lens_ptr + offs, mask=mask, other=0).to(tl.int64)
    visible = tl.minimum(seq.to(tl.int32), window)
    if PAGED:
        start = seq - visible.to(tl.int64)
        aligned = start - start % page_size
        draft = (seq - aligned).to(tl.int32)
    else:
        draft = visible
    tl.store(draft_lens_ptr + offs, draft, mask=mask)
    tl.store(suffix_start_ptr + offs, seq - draft.to(tl.int64), mask=mask)


def compact_draft_window(seq_lens: torch.Tensor, window: int, page_size: int):
    """(draft_prefix_lens int32, suffix_start int64) of the compact draft window in one launch, as
    DFlashWorkerV2._compute_compact_draft_seq_lens and _rebuild_compact_draft_cache compute them."""
    n = seq_lens.numel()
    draft = torch.empty(n, dtype=torch.int32, device=seq_lens.device)
    start = torch.empty(n, dtype=torch.int64, device=seq_lens.device)
    _compact_draft_window_kernel[(1,)](
        seq_lens,
        draft,
        start,
        n,
        int(window),
        int(page_size),
        PAGED=page_size > 1,
        BLOCK=max(16, triton.next_power_of_2(n)),
    )
    return draft, start
