"""Sparse MLA decode on gfx950 over fp8_ds_mla KV, one query per row (a DCP rank's partial).

csrc/attention/sparse_mla_decode_gfx950.cuh, ported from aiter's opus MXFP8 MLA decode. Q's
NoPE is quantized to MXFP8 (one E8M0 scale per 32 channels) for the scaled-MFMA QK dot; KV is
read in place. Each block takes one (row, split) and reads the row's first topk_length keys, so
the launch is graph-safe and splits follow each row's actual top-k share. Same contract as
triton_sparse_mla_decode_splitk with return_lse=True and topk_length.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch
import triton
import triton.language as tl

from sglang.kernels.jit.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module

D_V, D_ROPE, ROW_BYTES, TILE = 512, 64, 656, 32


@cache_once
def _jit_module() -> Module:
    return load_jit(
        "sparse_mla_decode_gfx950",
        cuda_files=["attention/sparse_mla_decode_gfx950.cuh"],
        cuda_wrappers=[("run", "SparseMlaDecodeGfx950Kernel::run")],
        # aiter's opus headers
        extra_dependencies=["aiter"],
    )


def supports_sparse_mla_decode_gfx950(
    q_nope: torch.Tensor, kv: torch.Tensor, d_v: int
) -> bool:
    heads = q_nope.shape[1]
    return (
        d_v == D_V
        and heads % 16 == 0
        and heads <= 128
        and q_nope.dtype == torch.bfloat16
        and kv.shape[-1] * kv.element_size() == ROW_BYTES
        # the kernel addresses the pool with 32-bit byte offsets
        and kv.numel() * kv.element_size() <= 0x7FFFFFFF
    )


@triton.jit
def _mxfp8_quant_kernel(
    q_ptr,
    out_ptr,
    scale_ptr,
    rope_ptr,
    rope_out_ptr,
    stride_qr,
    stride_qh,
    stride_rr,
    stride_rh,
    H,
    HB: tl.constexpr,
):
    # One program per (row, 16 heads): MXFP8 NoPE plus RoPE repacked into the dense head
    # stride the kernel's Q layout hard-codes.
    r = tl.program_id(0)
    h = tl.program_id(1) * HB + tl.arange(0, HB)
    hm = h < H
    d = tl.arange(0, 64)
    rope = tl.load(
        rope_ptr + r * stride_rr + h[:, None] * stride_rh + d[None, :], mask=hm[:, None]
    )
    tl.store(
        rope_out_ptr + (r * H + h)[:, None] * 64 + d[None, :], rope, mask=hm[:, None]
    )
    c = tl.arange(0, 512)
    x = tl.load(
        q_ptr + r * stride_qr + h[:, None] * stride_qh + c[None, :],
        mask=hm[:, None],
        other=0.0,
    )
    xb = tl.reshape(x.to(tl.float32), (HB, 16, 32))
    amax = tl.max(tl.abs(xb), axis=2)
    # smallest power of two that keeps the block inside the e4m3 range, as E8M0
    e = tl.ceil(tl.log2(tl.maximum(amax, 1e-30) / 448.0))
    e = tl.minimum(tl.maximum(e, -127.0), 127.0)
    y = tl.reshape(xb * tl.exp2(-e)[:, :, None], (HB, 512))
    tl.store(
        out_ptr + (r * H + h)[:, None] * 512 + c[None, :],
        y.to(tl.float8e4nv),
        mask=hm[:, None],
    )
    s = tl.arange(0, 16)
    tl.store(
        scale_ptr + (r * H + h)[:, None] * 16 + s[None, :],
        (e + 127.0).to(tl.uint8),
        mask=hm[:, None],
    )


@triton.jit
def _merge_kernel(o_acc_ptr, lse_acc_ptr, out_ptr, lse_ptr, H, S: tl.constexpr):
    pid = tl.program_id(0)
    r, h = pid // H, pid % H
    s = tl.arange(0, S)
    ls = tl.load(lse_acc_ptr + (r * S + s) * H + h)
    ls = tl.where(ls < -1e30, -float("inf"), ls)
    lmax = tl.max(ls, axis=0)
    w = tl.where(ls == -float("inf"), 0.0, tl.exp(ls - lmax))
    denom = tl.sum(w, axis=0)
    d = tl.arange(0, 512)
    o = tl.load(o_acc_ptr + ((r * S + s) * H + h)[:, None] * 512 + d[None, :])
    acc = tl.sum(w[:, None] * o, axis=0)
    tl.store(
        out_ptr + pid * 512 + d, tl.where(denom > 0, acc / denom, 0.0).to(tl.bfloat16)
    )
    tl.store(lse_ptr + pid, tl.where(denom > 0, lmax + tl.log(denom), -float("inf")))


def num_splits_for(rows: int, expected_topk: int) -> int:
    # Measured on MI350X (96 heads, 256 keys per row): unsplit wins at 64 rows, the 32-row
    # cut is untuned; small batches split up to 8 ways.
    if rows >= 32:
        return 1
    tiles = max(1, -(-expected_topk // TILE))
    per_row = max(1, 64 // max(rows, 1))
    return min(triton.next_power_of_2(tiles), 1 << (per_row.bit_length() - 1), 8)


def sparse_mla_decode_gfx950(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    topk_length: torch.Tensor,
    expected_topk: Optional[int] = None,
    num_splits: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sparse MLA decode of one query per row over fp8_ds_mla KV rows.

    q_nope [R, H, 512] / q_rope [R, H, 64] bf16 (any row / head strides, unit last dim),
    kv [N, 1, 656] fp8_ds_mla rows, indices [R, 1, topk] int32 slots of which row r reads its
    first topk_length[r]. Returns ([1, R, H, 512] bf16, [R, H] fp32 natural-log LSE, -inf for
    rows with no key)."""
    R, H, _ = q_nope.shape
    idx = indices.reshape(R, -1)
    assert idx.is_contiguous() and idx.dtype == torch.int32
    assert topk_length.dtype == torch.int32 and topk_length.numel() >= R
    topk = idx.shape[1]
    device = q_nope.device
    if num_splits is None:
        num_splits = num_splits_for(R, min(expected_topk or topk, topk))
    S = num_splits

    q8 = torch.empty(R, H, D_V, device=device, dtype=torch.float8_e4m3fn)
    qs = torch.empty(R, H, D_V // 32, device=device, dtype=torch.uint8)
    qr = torch.empty(R, H, D_ROPE, device=device, dtype=torch.bfloat16)
    if R > 0:
        _mxfp8_quant_kernel[(R, triton.cdiv(H, 16))](
            q_nope,
            q8,
            qs,
            q_rope,
            qr,
            q_nope.stride(0),
            q_nope.stride(1),
            q_rope.stride(0),
            q_rope.stride(1),
            H,
            HB=16,
        )

    out = torch.empty(R, H, D_V, device=device, dtype=torch.bfloat16)
    lse = torch.empty(R, H, device=device, dtype=torch.float32)
    slots = R * S if S > 1 else 1
    o_acc = torch.empty(slots, H, D_V, device=device, dtype=torch.float32)
    lse_acc = torch.empty(slots, H, device=device, dtype=torch.float32)
    _jit_module().run(
        q8,
        qs,
        qr,
        kv.view(torch.uint8).reshape(-1, ROW_BYTES),
        idx,
        topk_length,
        o_acc,
        lse_acc,
        out,
        lse,
        float(sm_scale),
        S,
    )
    if S > 1 and R > 0:
        _merge_kernel[(R * H,)](o_acc, lse_acc, out, lse, H, S=S)
    return out.unsqueeze(0), lse
