# SPDX-License-Identifier: Apache-2.0
"""Ordered local routed finalize shared with the draft prototype.

Kernel body is the draft prototype's finalize, unchanged.
No PDL: the caller must retain the producer's stream/event completion. Output
is local routed BF16, before native shared gate/add, TP sum and HC.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _q38_mk_mtp_finalize(ExpertRows, Weights, Out,
                       H: tl.constexpr, TOPK: tl.constexpr, BLOCK: tl.constexpr):
    col = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row = tl.program_id(1)
    total = tl.full((BLOCK,), 0, tl.float32)
    # Same top-10 order as the serial prototype, with the same BF16 expert rows.
    for slot in tl.static_range(TOPK):
        value = tl.load(ExpertRows + (row * TOPK + slot) * H + col,
                        col < H, 0).to(tl.float32)
        weight = tl.load(Weights + row * TOPK + slot).to(tl.float32)
        total = total + value * weight
    tl.store(Out + row * H + col, total.to(tl.bfloat16), col < H)



def finalize(expert_rows, weights, out=None):
    m, topk, h = expert_rows.shape
    assert (topk, h) == (10, 2560)
    assert expert_rows.dtype == weights.dtype == torch.bfloat16
    if out is None:
        out = torch.empty((m, h), device=expert_rows.device, dtype=torch.bfloat16)
    _q38_mk_mtp_finalize[(triton.cdiv(h, 256), m)](
        expert_rows, weights, out, H=h, TOPK=topk, BLOCK=256,
        num_warps=4, enable_fp_fusion=False,
    )
    return out
