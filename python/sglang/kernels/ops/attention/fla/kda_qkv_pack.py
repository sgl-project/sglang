# SPDX-License-Identifier: Apache-2.0
"""Normalize strided KDA Q/K and pack V without temporary Q/K copies."""

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["ROWS"])
def _kda_qkv_pack(
    q,
    k,
    v,
    qo,
    ko,
    vo,
    eps,
    ROWS,
    qs_t: tl.constexpr,
    qs_h: tl.constexpr,
    ks_t: tl.constexpr,
    ks_h: tl.constexpr,
    vs_t: tl.constexpr,
    vs_h: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    BT: tl.constexpr,
):
    rows = tl.program_id(0) * BT + tl.arange(0, BT)
    cols = tl.arange(0, D)
    mask = rows[:, None] < ROWS
    token = rows // H
    head = rows % H
    qval = tl.load(
        q + token[:, None] * qs_t + head[:, None] * qs_h + cols[None, :],
        mask=mask,
        other=0,
    ).to(tl.float32)
    kval = tl.load(
        k + token[:, None] * ks_t + head[:, None] * ks_h + cols[None, :],
        mask=mask,
        other=0,
    ).to(tl.float32)
    vval = tl.load(
        v + token[:, None] * vs_t + head[:, None] * vs_h + cols[None, :],
        mask=mask,
        other=0,
    )
    # Match l2norm_fwd_kernel's FP32 reduction/division and BF16 store.
    qnorm = qval / tl.sqrt(tl.sum(qval * qval, axis=1) + eps)[:, None]
    knorm = kval / tl.sqrt(tl.sum(kval * kval, axis=1) + eps)[:, None]
    offsets = rows[:, None] * D + cols[None, :]
    tl.store(qo + offsets, qnorm.to(qo.dtype.element_ty), mask=mask)
    tl.store(ko + offsets, knorm.to(ko.dtype.element_ty), mask=mask)
    tl.store(vo + offsets, vval, mask=mask)


def supports_kda_qkv_pack(q, k, v):
    return (
        q.is_cuda
        and q.dtype == torch.bfloat16
        and q.ndim == 4
        and q.shape[0] == 1
        and q.shape[1] > 0
        and q.shape == k.shape == v.shape
        and q.shape[-1] == 128
        and q.device == k.device == v.device
        and q.dtype == k.dtype == v.dtype
        and q.stride(-1) == k.stride(-1) == v.stride(-1) == 1
    )


def kda_qkv_pack(q, k, v, eps=1e-6):
    assert supports_kda_qkv_pack(q, k, v)
    result = torch.empty((3, *q.shape), device=q.device, dtype=q.dtype)
    qo, ko, vo = result.unbind(0)
    rows = q.shape[1] * q.shape[2]
    _kda_qkv_pack[(triton.cdiv(rows, 16),)](
        q,
        k,
        v,
        qo,
        ko,
        vo,
        eps,
        ROWS=rows,
        qs_t=q.stride(1),
        qs_h=q.stride(2),
        ks_t=k.stride(1),
        ks_h=k.stride(2),
        vs_t=v.stride(1),
        vs_h=v.stride(2),
        H=q.shape[2],
        D=128,
        BT=16,
        num_warps=8,
        num_stages=3,
    )
    return qo, ko, vo
