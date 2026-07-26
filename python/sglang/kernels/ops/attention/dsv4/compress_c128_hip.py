"""HIP c128 compression kernels for DSV4 (RFC #29630, Phase 2.5).

Migrated from ``sglang.srt.layers.attention.dsv4.compressor_v2``; the
kernels are defined under an ``is_hip`` guard exactly as before.
"""

from __future__ import annotations

from typing import Union

import torch

from sglang.kernels.jit.utils import is_hip_runtime
from sglang.kernels.ops.attention.dsv4 import (
    CompressorDecodePlan,
    CompressorPrefillPlan,
)

_is_hip = is_hip_runtime()

if _is_hip:
    import triton
    import triton.language as tl

    @triton.jit
    def _c128_compress_decode_kernel(
        buf_ptr,
        input_ptr,
        ape_ptr,
        out_ptr,
        plan_ptr,
        buf_stride_slot,
        input_stride_b,
        ape_stride_r,
        out_stride_b,
        bs,
        HEAD_DIM: tl.constexpr,
        BLOCK_D: tl.constexpr,
        COMPRESS_RATIO: tl.constexpr,
    ):
        """Fused C128 decode: write to state buffer + online softmax-pool.

        plan_ptr points to int32 view: [bs, 4] where each row is
        {seq_len, write_loc, read_page_0, read_page_1}.
        """
        bid = tl.program_id(0)
        if bid >= bs:
            return

        # Parse plan
        plan_base = plan_ptr + bid * 4
        seq_len = tl.load(plan_base).to(tl.int32)
        write_loc = tl.load(plan_base + 1).to(tl.int32)
        read_page_0 = tl.load(plan_base + 2).to(tl.int32)

        d = tl.arange(0, BLOCK_D)
        last_dim: tl.constexpr = HEAD_DIM * 2

        # Step 1: Write kv_score_input to state buffer at write_loc
        d_mask_full = d < last_dim
        input_val = tl.load(
            input_ptr + bid * input_stride_b + d, mask=d_mask_full, other=0.0
        )
        tl.store(buf_ptr + write_loc * buf_stride_slot + d, input_val, mask=d_mask_full)

        # Step 2: Check boundary condition
        d_mask_hd = d < HEAD_DIM
        if seq_len % COMPRESS_RATIO != 0:
            tl.store(
                out_ptr + bid * out_stride_b + d,
                tl.zeros([BLOCK_D], tl.float32),
                mask=d_mask_hd,
            )
            return

        # Step 3: Online softmax-pool over 128 slots in the page
        page_base = read_page_0 * COMPRESS_RATIO * buf_stride_slot
        m_prev = tl.full([BLOCK_D], float("-inf"), tl.float32)
        kv_acc = tl.zeros([BLOCK_D], tl.float32)
        w_acc = tl.zeros([BLOCK_D], tl.float32)

        for k in tl.static_range(COMPRESS_RATIO):
            slot_addr = page_base + k * buf_stride_slot
            kv_val = tl.load(buf_ptr + slot_addr + d, mask=d_mask_hd, other=0.0).to(
                tl.float32
            )
            sc_val = tl.load(
                buf_ptr + slot_addr + HEAD_DIM + d, mask=d_mask_hd, other=0.0
            ).to(tl.float32)
            ape_val = tl.load(
                ape_ptr + k * ape_stride_r + d, mask=d_mask_hd, other=0.0
            ).to(tl.float32)
            score_k = sc_val + ape_val

            m_new = tl.maximum(m_prev, score_k)
            exp_old = tl.where(m_prev == float("-inf"), 0.0, tl.exp(m_prev - m_new))
            exp_cur = tl.where(score_k == float("-inf"), 0.0, tl.exp(score_k - m_new))
            kv_acc = kv_acc * exp_old + exp_cur * kv_val
            w_acc = w_acc * exp_old + exp_cur
            m_prev = m_new

        compressed = kv_acc / w_acc
        tl.store(out_ptr + bid * out_stride_b + d, compressed, mask=d_mask_hd)

    @triton.jit
    def _c128_compress_prefill_write_kernel(
        buf_ptr,
        input_ptr,
        plan_w_ptr,
        buf_stride_slot,
        input_stride_b,
        num_w,
        BLOCK_D: tl.constexpr,
        LAST_DIM: tl.constexpr,
    ):
        """Prefill write phase: scatter kv_score_input tokens into state buffer."""
        wid = tl.program_id(0)
        if wid >= num_w:
            return

        # WritePlan: {ragged_id(u32), write_loc(i32)} = 8 bytes = 2 int32s
        plan_base = plan_w_ptr + wid * 2
        ragged_id = (tl.load(plan_base).to(tl.int32)) & 0xFFFF
        write_loc = tl.load(plan_base + 1).to(tl.int32)

        d = tl.arange(0, BLOCK_D)
        d_mask = d < LAST_DIM

        if write_loc >= 0:
            input_val = tl.load(
                input_ptr + ragged_id * input_stride_b + d, mask=d_mask, other=0.0
            )
            tl.store(buf_ptr + write_loc * buf_stride_slot + d, input_val, mask=d_mask)

    @triton.jit
    def _c128_compress_prefill_compress_kernel(
        buf_ptr,
        ape_ptr,
        out_ptr,
        plan_c_ptr,
        buf_stride_slot,
        ape_stride_r,
        out_stride_b,
        num_c,
        HEAD_DIM: tl.constexpr,
        BLOCK_D: tl.constexpr,
        COMPRESS_RATIO: tl.constexpr,
    ):
        """Prefill compress phase: online softmax-pool for each compress plan entry."""
        cid = tl.program_id(0)
        if cid >= num_c:
            return

        # CompressPlan: {seq_len(u32), ragged_id(u16)|buffer_len(u16), read_page_0(i32), read_page_1(i32)}
        plan_base = plan_c_ptr + cid * 4
        read_page_0 = tl.load(plan_base + 2).to(tl.int32)

        d = tl.arange(0, BLOCK_D)
        d_mask_hd = d < HEAD_DIM

        if read_page_0 < 0:
            tl.store(
                out_ptr + cid * out_stride_b + d,
                tl.zeros([BLOCK_D], tl.float32),
                mask=d_mask_hd,
            )
            return

        page_base = read_page_0 * COMPRESS_RATIO * buf_stride_slot
        m_prev = tl.full([BLOCK_D], float("-inf"), tl.float32)
        kv_acc = tl.zeros([BLOCK_D], tl.float32)
        w_acc = tl.zeros([BLOCK_D], tl.float32)

        for k in tl.static_range(COMPRESS_RATIO):
            slot_addr = page_base + k * buf_stride_slot
            kv_val = tl.load(buf_ptr + slot_addr + d, mask=d_mask_hd, other=0.0).to(
                tl.float32
            )
            sc_val = tl.load(
                buf_ptr + slot_addr + HEAD_DIM + d, mask=d_mask_hd, other=0.0
            ).to(tl.float32)
            ape_val = tl.load(
                ape_ptr + k * ape_stride_r + d, mask=d_mask_hd, other=0.0
            ).to(tl.float32)
            score_k = sc_val + ape_val

            m_new = tl.maximum(m_prev, score_k)
            exp_old = tl.where(m_prev == float("-inf"), 0.0, tl.exp(m_prev - m_new))
            exp_cur = tl.where(score_k == float("-inf"), 0.0, tl.exp(score_k - m_new))
            kv_acc = kv_acc * exp_old + exp_cur * kv_val
            w_acc = w_acc * exp_old + exp_cur
            m_prev = m_new

        compressed = kv_acc / w_acc
        tl.store(out_ptr + cid * out_stride_b + d, compressed, mask=d_mask_hd)


def _compress_forward_c128_triton(
    kv_score_buffer: torch.Tensor,
    kv_score_input: torch.Tensor,
    ape: torch.Tensor,
    plan: Union[CompressorDecodePlan, CompressorPrefillPlan],
    head_dim: int,
) -> torch.Tensor:
    """Triton C128 compress_forward for HIP (wave64).

    Fuses write + online-softmax-pool into Triton kernels.
    CUDA graph compatible.
    """
    num_total_slots = kv_score_buffer.shape[0] * kv_score_buffer.shape[1]
    num_pages = kv_score_buffer.shape[0]
    last_dim = kv_score_buffer.shape[-1]
    compress_ratio = 128

    buf_flat = kv_score_buffer.view(-1, last_dim)
    buf_stride_slot = last_dim  # elements per slot

    BLOCK_D = triton.next_power_of_2(last_dim)

    if plan.is_decode:
        # Decode path: single kernel does write + compress
        plan_raw = plan[1].view(torch.int32)  # [bs, 4]
        bs = plan_raw.shape[0]
        out = torch.empty(
            bs, head_dim, dtype=torch.float32, device=kv_score_input.device
        )

        if bs > 0 and num_total_slots > 0:
            grid = (bs,)
            _c128_compress_decode_kernel[grid](
                buf_flat,
                kv_score_input,
                ape,
                out,
                plan_raw,
                buf_stride_slot,
                kv_score_input.stride(0),
                ape.stride(0),
                out.stride(0),
                bs,
                HEAD_DIM=head_dim,
                BLOCK_D=triton.next_power_of_2(head_dim),
                COMPRESS_RATIO=compress_ratio,
                num_warps=8,
            )
        return out
    else:
        # Prefill path: separate write kernel + compress kernel
        plan_c_raw = plan[1].view(torch.int32)  # [num_c, 4]
        plan_w = plan[2]  # [num_w, 8] uint8
        plan_w_raw = plan_w.view(torch.int32)  # [num_w, 2]
        num_c = plan_c_raw.shape[0]
        num_w = plan_w_raw.shape[0]

        out = torch.empty(
            num_c, head_dim, dtype=torch.float32, device=kv_score_input.device
        )

        # Phase 1: Write
        if num_w > 0 and num_total_slots > 0:
            grid_w = (num_w,)
            _c128_compress_prefill_write_kernel[grid_w](
                buf_flat,
                kv_score_input,
                plan_w_raw,
                buf_stride_slot,
                kv_score_input.stride(0),
                num_w,
                BLOCK_D=BLOCK_D,
                LAST_DIM=last_dim,
                num_warps=4,
            )

        # Phase 2: Compress
        if num_c > 0 and num_pages > 0:
            grid_c = (num_c,)
            _c128_compress_prefill_compress_kernel[grid_c](
                buf_flat,
                ape,
                out,
                plan_c_raw,
                buf_stride_slot,
                ape.stride(0),
                out.stride(0),
                num_c,
                HEAD_DIM=head_dim,
                BLOCK_D=triton.next_power_of_2(head_dim),
                COMPRESS_RATIO=compress_ratio,
                num_warps=8,
            )

        return out
