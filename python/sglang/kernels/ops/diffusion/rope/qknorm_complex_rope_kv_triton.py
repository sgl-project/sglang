# SPDX-License-Identifier: Apache-2.0
"""Write normalized/rotated K and unmodified V into their final prefix buffers."""

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.diffusion.rope.complex_rope_triton import _fuse_real_sin
from sglang.kernels.ops.diffusion.rope.qknorm_complex_rope_triton import (
    _qknorm_complex_rope_rows,
    can_use_qknorm_complex_rope,
)
from sglang.srt.utils.custom_op import register_custom_op


@triton.jit
def _qknorm_complex_rope_kv_kernel(
    k_ptr,
    weight_ptr,
    rope_ptr,
    v_ptr,
    kp_ptr,
    vp_ptr,
    kout_ptr,
    vout_ptr,
    ROWS: tl.constexpr,
    SEQ: tl.constexpr,
    HEADS: tl.constexpr,
    PREFIX: tl.constexpr,
    BATCH: tl.constexpr,
    EPS: tl.constexpr,
    FUSE_REAL_SIN: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid < tl.cdiv(ROWS, 4):
        row = pid * 4 + tl.arange(0, 4)
        column = tl.arange(0, 128)
        key = _qknorm_complex_rope_rows(
            k_ptr, weight_ptr, rope_ptr, row, ROWS, SEQ, HEADS, EPS, FUSE_REAL_SIN
        )
        out_row = row + (row // (SEQ * HEADS) + 1) * PREFIX * HEADS
        output_index = out_row[:, None] * 128 + column[None, :]
        mask = row[:, None] < ROWS
        tl.store(kout_ptr + output_index, key, mask)
        value = tl.load(v_ptr + row[:, None] * 128 + column[None, :], mask, 0)
        tl.store(vout_ptr + output_index, value, mask)
    else:
        index = (pid - tl.cdiv(ROWS, 4)) * 1024 + tl.arange(0, 1024)
        prefix_mask = index < BATCH * PREFIX * HEADS * 128
        prefix_index = index + (index // (PREFIX * HEADS * 128)) * SEQ * HEADS * 128
        prefix_key = tl.load(kp_ptr + index, prefix_mask, 0)
        prefix_value = tl.load(vp_ptr + index, prefix_mask, 0)
        tl.store(kout_ptr + prefix_index, prefix_key, prefix_mask)
        tl.store(vout_ptr + prefix_index, prefix_value, prefix_mask)


def can_use_qknorm_complex_rope_kv(k, weight, rope, v, k_prefix, v_prefix):
    return (
        can_use_qknorm_complex_rope(k, weight, rope)
        and v.shape == k.shape
        and k_prefix.ndim == 4
        and k_prefix.shape[0] == k.shape[0]
        and k_prefix.shape[1] > 0
        and k_prefix.shape[2:] == k.shape[2:]
        and v_prefix.shape == k_prefix.shape
        and all(
            x.device == k.device and x.dtype == k.dtype and x.is_contiguous()
            for x in (v, k_prefix, v_prefix)
        )
    )


def _fake_qknorm_complex_rope_kv(k, weight, rope, v, k_prefix, v_prefix, eps):
    shape = (k.shape[0], k_prefix.shape[1] + k.shape[1], *k.shape[2:])
    return k.new_empty(shape), v.new_empty(shape)


@register_custom_op(
    op_name="qknorm_complex_rope_kv",
    mutates_args=[],
    fake_impl=_fake_qknorm_complex_rope_kv,
)
def qknorm_complex_rope_kv(
    k: torch.Tensor,
    weight: torch.Tensor,
    rope: torch.Tensor,
    v: torch.Tensor,
    k_prefix: torch.Tensor,
    v_prefix: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert can_use_qknorm_complex_rope_kv(k, weight, rope, v, k_prefix, v_prefix)
    kout, vout = _fake_qknorm_complex_rope_kv(
        k, weight, rope, v, k_prefix, v_prefix, eps
    )
    batch, seq, heads, dim = k.shape
    prefix = k_prefix.shape[1]
    with torch.cuda.device(k.device):
        _qknorm_complex_rope_kv_kernel[
            (
                triton.cdiv(batch * seq * heads, 4)
                + triton.cdiv(batch * prefix * heads * dim, 1024),
            )
        ](
            k,
            weight,
            torch.view_as_real(rope),
            v,
            k_prefix,
            v_prefix,
            kout,
            vout,
            batch * seq * heads,
            seq,
            heads,
            prefix,
            batch,
            eps,
            _fuse_real_sin(k.device),
            num_warps=4,
            enable_fp_fusion=False,
        )
    return kout, vout
