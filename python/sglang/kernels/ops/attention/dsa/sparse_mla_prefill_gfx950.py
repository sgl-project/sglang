"""DSA sparse MLA prefill for fp8_ds_mla KV on gfx950 (HIP).

csrc/attention/sparse_mla_prefill_gfx950.cuh: one CTA per query token, 4 waves, one wave
per fp8 scale group, BF16 MFMA with the group scales applied after the dots. Same contract
as ``triton_sparse_mla_fwd`` for H <= 16 heads and 656-byte KV rows.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module

_ROW_BYTES = 656


@cache_once
def _jit_module(stages: int) -> Module:
    args = make_cpp_args(stages)
    return load_jit(
        "sparse_mla_prefill_gfx950",
        *args,
        cuda_files=["attention/sparse_mla_prefill_gfx950.cuh"],
        cuda_wrappers=[
            ("run", f"SparseMlaPrefillGfx950Kernel<{args}>::run"),
            ("run_split", f"SparseMlaPrefillGfx950Kernel<{args}>::run_split"),
        ],
    )


def supports_sparse_mla_prefill_gfx950(
    q_nope: torch.Tensor, q_rope: torch.Tensor, kv: torch.Tensor, d_v: int
) -> bool:
    return (
        q_nope.dtype == torch.bfloat16
        and q_rope.dtype == torch.bfloat16
        and d_v == 512
        and q_rope.shape[-1] == 64
        and q_nope.shape[1] <= 16
        and q_nope.stride(-1) == 1
        and q_rope.stride(-1) == 1
        and all(s % 8 == 0 for s in (*q_nope.stride()[:2], *q_rope.stride()[:2]))
        and kv.shape[-1] == _ROW_BYTES
        and kv.element_size() == 1
        and kv.stride(0) == _ROW_BYTES
        # rows are addressed by a 31-bit buffer offset
        and kv.shape[0] * _ROW_BYTES <= 0x7FFFFFFF
    )


def sparse_mla_prefill_gfx950(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int = 512,
    return_lse: bool = False,
    stages: int = 2,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """q_nope [T, H, 512] / q_rope [T, H, 64] bf16, kv [N, 1, 656] fp8_ds_mla rows,
    indices [T, 1, topk] or [T, topk] int32 (-1 = no key).
    Returns [1, T, H, 512] bf16 and, with return_lse, the [T, H] fp32 natural-log LSE."""
    seq, H, _ = q_nope.shape
    idx = indices.squeeze(1) if indices.dim() == 3 else indices
    if idx.dtype != torch.int32:
        idx = idx.to(torch.int32)
    idx = idx.contiguous()
    kv_rows = kv.view(torch.uint8).reshape(kv.shape[0], _ROW_BYTES)
    out = torch.empty(seq, H, d_v, device=q_nope.device, dtype=torch.bfloat16)
    lse = torch.empty(seq, H, device=q_nope.device, dtype=torch.float32)
    _jit_module(stages).run(
        q_nope, q_rope, kv_rows, idx, out, lse, float(sm_scale), int(return_lse)
    )
    return (out.unsqueeze(0), lse) if return_lse else out.unsqueeze(0)


def sparse_mla_decode_splits(rows: int) -> int:
    """CTAs per row for sparse_mla_split_gfx950 on a decode / verify batch of ``rows`` query
    rows (top-2048, 12 heads, MI355X, 2 stages; us at rows 6 / 24 / 48 / 96 / 192 / 288:
    S16 17.9 / 24.0 / 30.3, S8 18.5 / 20.6 / 26.2 / 32.0 / 53.4, S4 36.6 / 47.5 / 71.1,
    vs the Triton split-K 22.8 / 41.0 / 66.1 / 115 / 231 / 453)."""
    if rows <= 6:
        return 16
    if rows <= 128:
        return 8
    if rows <= 512:
        return 4
    return 2


def sparse_mla_split_gfx950(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    splits: int,
    d_v: int = 512,
    return_lse: bool = False,
    stages: int = 2,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """sparse_mla_prefill_gfx950 with each row's top-k split over ``splits`` CTAs and an
    fp32 log-sum-exp combine: for decode / verify batches too small to fill the GPU at
    one CTA per row. Returns [1, T, H, 512] bf16 and, with return_lse, the [T, H] fp32
    natural-log LSE."""
    seq, H, _ = q_nope.shape
    idx = indices.squeeze(1) if indices.dim() == 3 else indices
    if idx.dtype != torch.int32:
        idx = idx.to(torch.int32)
    idx = idx.contiguous()
    kv_rows = kv.view(torch.uint8).reshape(kv.shape[0], _ROW_BYTES)
    out = torch.empty(seq, H, d_v, device=q_nope.device, dtype=torch.bfloat16)
    lse = torch.empty(seq, H, device=q_nope.device, dtype=torch.float32)
    part_acc = torch.empty(
        seq, splits, H, d_v, device=q_nope.device, dtype=torch.float32
    )
    part_ml = torch.empty(2, seq, splits, H, device=q_nope.device, dtype=torch.float32)
    _jit_module(stages).run_split(
        q_nope,
        q_rope,
        kv_rows,
        idx,
        out,
        lse,
        part_acc,
        part_ml,
        float(sm_scale),
        int(return_lse),
        splits,
    )
    return (out.unsqueeze(0), lse) if return_lse else out.unsqueeze(0)
