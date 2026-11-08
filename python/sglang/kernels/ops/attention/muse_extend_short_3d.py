"""Muse Spark full-layer short extend on aiter's split-KV 3D unified-attention kernel with 32-row q-blocks.

For extends ``unified_attention`` sends to its 3D kernel (full attention, few query rows, long context), aiter's
q-block is BLOCK_M 16 rows = 2 query tokens at GQA 8, so every KV tile is loaded once per 2 tokens. This runs the
same kernel with BLOCK_M 32 (4 tokens per program), keeping aiter's own 2D / 3D choice and ``select_3d_config``
(tile, segment count, warps) computed from the stock 2-token q-blocks. Each row still accumulates the same KV tiles
of the same segments in the same order (the extra causally masked tiles add exact zeros), so the output matches
``unified_attention`` bit for bit.
"""

from __future__ import annotations

import torch
import triton

try:
    from aiter.ops.triton.attention.unified_attention import (
        IS_DEVICE_ARCH_GFX12,
        e4m3_dtype,
        get_num_sms,
        kernel_unified_attention_3d,
        reduce_segments,
        select_3d_config,
        use_2d_kernel,
    )
except ImportError:
    kernel_unified_attention_3d = None

_GQA = 8
_STOCK_BLOCK_Q = 16 // _GQA
_BLOCK_M = 32
_BLOCK_Q = _BLOCK_M // _GQA


def muse_extend_short_3d(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seqused_k: torch.Tensor,
    block_table: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: float,
    q_descale: torch.Tensor | None,
    k_descale: torch.Tensor,
    v_descale: torch.Tensor,
) -> bool:
    """Causal full-attention extend over the paged FP8 cache ``k`` / ``v`` [blocks, page, kv_heads, 64] into
    ``out``; False (nothing launched) where ``unified_attention`` would not take its 3D kernel or the shape is
    not Muse's."""
    if kernel_unified_attention_3d is None or IS_DEVICE_ARCH_GFX12:
        return False
    num_tokens, num_query_heads, head_size = q.shape
    num_blocks, block_size, num_kv_heads, _ = k.shape
    if not (
        head_size == 64
        and num_query_heads == _GQA * num_kv_heads
        and k.dtype == v.dtype == e4m3_dtype
        and q.dtype in (torch.bfloat16, e4m3_dtype)
        and int(max_seqlen_q) > 1
    ):
        return False
    num_seqs = len(seqused_k)
    target_num_prgms = get_num_sms() * 4
    num_2d_prgms = (num_tokens // _STOCK_BLOCK_Q + num_seqs) * num_kv_heads
    if use_2d_kernel(
        head_size,
        0,
        False,
        max_seqlen_q,
        max_seqlen_k,
        target_num_prgms,
        num_2d_prgms,
    ):
        return False
    attn_config, reduce_config = select_3d_config(
        head_size,
        block_size,
        max_seqlen_k,
        target_num_prgms,
        num_2d_prgms,
        q.dtype,
        k.dtype,
        False,
        1,
        0,
    )
    num_segments = attn_config["NUM_SEGMENTS_PER_SEQ"]
    if num_segments > 1:
        segm_output = torch.empty(
            num_tokens,
            num_query_heads,
            num_segments,
            triton.next_power_of_2(head_size),
            dtype=torch.float32,
            device=q.device,
        )
        segm_max = torch.empty(
            num_tokens,
            num_query_heads,
            num_segments,
            dtype=torch.float32,
            device=q.device,
        )
        segm_expsum = torch.empty_like(segm_max)
    else:
        segm_output = segm_max = segm_expsum = out
    kernel_unified_attention_3d[
        (num_tokens // _BLOCK_Q + num_seqs, num_kv_heads, num_segments)
    ](
        segm_output_ptr=segm_output,
        segm_max_ptr=segm_max,
        segm_expsum_ptr=segm_expsum,
        query_ptr=q,
        key_cache_ptr=k,
        value_cache_ptr=v,
        sink_ptr=None,
        block_tables_ptr=block_table,
        seq_lens_ptr=seqused_k,
        alibi_slopes_ptr=None,
        qq_bias_ptr=None,
        scale=softmax_scale,
        q_descale_ptr=q_descale,
        k_descale_ptr=k_descale,
        v_descale_ptr=v_descale,
        out_scale_ptr=None,
        softcap=0,
        num_query_heads=num_query_heads,
        num_queries_per_kv=_GQA,
        block_table_stride=block_table.stride(0),
        query_stride_0=q.stride(0),
        query_stride_1=q.stride(1),
        qq_bias_stride_0=0,
        BLOCK_SIZE=block_size,
        HEAD_SIZE=head_size,
        HEAD_SIZE_PADDED=triton.next_power_of_2(head_size),
        USE_ALIBI_SLOPES=False,
        USE_QQ_BIAS=False,
        USE_SOFTCAP=False,
        USE_SINKS=False,
        SLIDING_WINDOW=0,
        stride_k_cache_0=k.stride(0),
        stride_k_cache_1=k.stride(1),
        stride_k_cache_2=k.stride(2),
        stride_k_cache_3=k.stride(3),
        stride_v_cache_0=v.stride(0),
        stride_v_cache_1=v.stride(1),
        stride_v_cache_2=v.stride(2),
        stride_v_cache_3=v.stride(3),
        query_start_len_ptr=cu_seqlens_q,
        BLOCK_Q=_BLOCK_Q,
        num_seqs=num_seqs,
        BLOCK_M=_BLOCK_M,
        ALL_DECODE=False,
        SHUFFLED_KV_CACHE=False,
        K_WIDTH=16,
        IS_Q_FP8=q.dtype == e4m3_dtype,
        IS_KV_FP8=True,
        # the stock 16-row tile's MFMA shape; the compiler's pick for 32 rows changes the dot's summation order
        matrix_instr_nonkdim=16,
        **attn_config,
    )
    if num_segments > 1:
        reduce_segments[(num_tokens, num_query_heads)](
            output_ptr=out,
            segm_output_ptr=segm_output,
            segm_max_ptr=segm_max,
            segm_expsum_ptr=segm_expsum,
            seq_lens_ptr=seqused_k,
            num_seqs=num_seqs,
            num_query_heads=num_query_heads,
            out_scale_ptr=None,
            output_stride_0=out.stride(0),
            output_stride_1=out.stride(1),
            block_table_stride=block_table.stride(0),
            HEAD_SIZE=head_size,
            HEAD_SIZE_PADDED=triton.next_power_of_2(head_size),
            query_start_len_ptr=cu_seqlens_q,
            BLOCK_Q=_BLOCK_Q,
            **reduce_config,
        )
    return True
