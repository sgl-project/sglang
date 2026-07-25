"""Fused MLA decode prepare tail: paged-KV scatter + absorbed-q concat.

One launch replacing the back-to-back ``set_mla_kv_buffer`` +
``concat_mla_absorb_q`` pair on the trtllm-mla decode graph path. Both
workloads are launch-bound data movement at decode batch sizes; the fusion
saves a kernel launch per MLA layer and keeps the PDL chain to the decode
fmha kernel intact. SM90+ only (TMA bulk store) — gate via ``covered()``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def set_mla_kv_concat_q_module(
    nope_bytes: int, rope_bytes: int, use_pdl: bool
) -> Module:
    args = make_cpp_args(nope_bytes, rope_bytes, use_pdl)
    return load_jit(
        f"set_mla_kv_concat_q_{nope_bytes}_{rope_bytes}",
        *args,
        cuda_files=["elementwise/set_mla_kv_concat_q.cuh"],
        cuda_wrappers=[
            ("set_mla_kv_concat_q", f"SetMlaKVConcatQKernel<{args}>::run"),
        ],
    )


@cache_once
def can_use_set_mla_kv_concat_q(nope_bytes: int, rope_bytes: int) -> bool:
    """Whether the fused kernel supports these row byte widths on this arch.

    Static gate (per process): SM90+ for the TMA bulk store, plus the
    compile-time width constraints (total row 16B-aligned for TMA; nope dim
    filling whole int4 warp rounds and rope dim exactly one int warp round
    for the concat side — 1024/128 bytes i.e. 512/64 bf16 satisfies both).
    """
    if torch.cuda.get_device_capability()[0] < 9:
        return False
    if nope_bytes % 4 != 0 or rope_bytes % 4 != 0:
        return False
    if (nope_bytes + rope_bytes) % 16 != 0:
        return False
    # Concat-side vector layout: int4 (8 bf16) x 32 lanes per round for nope,
    # one int (2 bf16) x 32 lanes round for rope.
    if (nope_bytes // 2) % (8 * 32) != 0 or rope_bytes // 2 != 2 * 32:
        return False
    try:
        set_mla_kv_concat_q_module(nope_bytes, rope_bytes, is_arch_support_pdl())
        return True
    except Exception:  # pragma: no cover - compile-time only
        return False


def _row_aligned(t: torch.Tensor, align: int) -> bool:
    """Base pointer and all non-unit strides aligned to ``align`` bytes."""
    if t.data_ptr() % align != 0:
        return False
    esize = t.element_size()
    return all(s * esize % align == 0 for s in t.stride()[:-1])


def covered(
    kv_buffer: torch.Tensor,
    loc: torch.Tensor,
    k_nope: torch.Tensor,
    k_rope: torch.Tensor,
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
) -> bool:
    """Per-call gate mirroring the launcher's alignment/layout tripwires, so
    uncovered layouts fall back to the two-kernel path instead of faulting.

    Expects the already-flattened views the wrapper launches with:
    kv_buffer [pages, D], k_nope/k_rope [B, d], q_nope/q_rope [B, H, d],
    loc [B].
    """
    if not (
        kv_buffer.dtype == torch.bfloat16
        and k_nope.dtype == torch.bfloat16
        and k_rope.dtype == torch.bfloat16
        and q_nope.dtype == torch.bfloat16
        and q_rope.dtype == torch.bfloat16
    ):
        return False
    if loc.dtype not in (torch.int32, torch.int64):
        return False
    if loc.dim() != 1 or not loc.is_contiguous():
        return False
    if not (
        k_nope.shape[0]
        == k_rope.shape[0]
        == loc.shape[0]
        == q_nope.shape[0]
        == q_rope.shape[0]
    ):
        return False
    if q_nope.shape[1] != q_rope.shape[1]:
        return False
    nope_bytes = k_nope.shape[-1] * 2
    rope_bytes = k_rope.shape[-1] * 2
    if q_nope.shape[-1] * 2 != nope_bytes or q_rope.shape[-1] * 2 != rope_bytes:
        return False
    if not can_use_set_mla_kv_concat_q(nope_bytes, rope_bytes):
        return False
    # Last dims must be dense for the vectorised row accesses.
    if any(t.stride(-1) != 1 for t in (kv_buffer, k_nope, k_rope, q_nope, q_rope)):
        return False
    return (
        _row_aligned(kv_buffer, 16)
        and _row_aligned(k_nope, 16)
        and _row_aligned(k_rope, 4)
        and _row_aligned(q_nope, 16)
        and _row_aligned(q_rope, 4)
    )


def _pick_num_warps(total_items: int) -> int:
    # Same GB300 heuristic as set_mla_kv_buffer: small grids favour more CTAs.
    return 4 if total_items <= 768 else 8


def set_mla_kv_concat_q(
    kv_buffer: torch.Tensor,
    loc: torch.Tensor,
    cache_k_nope: torch.Tensor,
    cache_k_rope: torch.Tensor,
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    num_warps: int = 0,
) -> torch.Tensor:
    """Scatter [k_nope | k_rope] rows into ``kv_buffer`` at ``loc`` and return
    the concatenated query [q_nope | q_rope], all in one kernel launch.

    Shapes (leading singleton dims on the k sources are flattened away):
        kv_buffer:    [num_pages, total_dim] or [num_pages, 1, total_dim]
        loc:          [n_loc]
        cache_k_nope: [n_loc, nope_dim] or [n_loc, 1, nope_dim]
        cache_k_rope: [n_loc, rope_dim] or [n_loc, 1, rope_dim]
        q_nope:       [n_loc, num_heads, nope_dim]
        q_rope:       [n_loc, num_heads, rope_dim]
    Returns:
        query: [n_loc, num_heads, nope_dim + rope_dim] (new contiguous tensor)
    """
    n_loc = loc.shape[0]
    src_nope = cache_k_nope.view(n_loc, -1) if cache_k_nope.dim() != 2 else cache_k_nope
    src_rope = cache_k_rope.view(n_loc, -1) if cache_k_rope.dim() != 2 else cache_k_rope
    buf = kv_buffer.view(kv_buffer.shape[0], -1) if kv_buffer.dim() != 2 else kv_buffer

    q_out = torch.empty(
        (*q_nope.shape[:-1], q_nope.shape[-1] + q_rope.shape[-1]),
        dtype=q_nope.dtype,
        device=q_nope.device,
    )

    nope_bytes = src_nope.shape[-1] * src_nope.element_size()
    rope_bytes = src_rope.shape[-1] * src_rope.element_size()
    if num_warps <= 0:
        num_warps = _pick_num_warps(n_loc + q_nope.shape[0] * q_nope.shape[1])

    module = set_mla_kv_concat_q_module(nope_bytes, rope_bytes, is_arch_support_pdl())
    module.set_mla_kv_concat_q(
        buf, loc, src_nope, src_rope, q_nope, q_rope, q_out, num_warps
    )
    return q_out
