"""Public adapter for Ascend split HBM Copy-SFA operators."""

from __future__ import annotations

import torch


def load_copy_sfa_operator():
    """Resolve the registered A3 implementation before graph capture."""
    name = "fused_scatter_copy_sparse_flash_attention_split"
    try:
        import sgl_kernel_npu  # noqa: F401

        return getattr(torch.ops.npu, name).default
    except (ImportError, AttributeError) as error:
        raise RuntimeError(
            f"Copy-SFA requires an sgl-kernel-npu A3 wheel containing npu::{name}."
        ) from error


def fused_scatter_copy_sparse_flash_attention(
    query,
    query_rope,
    hbm_ckv,
    hbm_rope,
    actual_seq_lengths_query,
    actual_seq_lengths_kv,
    num_cache_tokens,
    topk_src_ids,
    topk_dst_slots,
    topk_miss_counts,
    miss_src_ids,
    miss_dst_slots,
    miss_counts,
    hbm_block_table,
    dram_block_table,
    dram_kv_dev_ptr,
    attention_out,
    scale_value,
    *,
    operator,
    current,
    positions,
    valid,
    request_slots,
    req_to_token,
    resident_begin,
    resident_end,
    host_token_capacity,
    dram_rope_dev_ptr,
    full_visible=False,
    resident_refresh_tail=False,
) -> None:
    """Copy resident KV and attend to selected cache tokens.

    Query tensors use contiguous TND ``[T, H, 512|64]`` and request-level
    cumulative query lengths. ``topk_src_ids`` and ``topk_dst_slots`` use
    contiguous ``[T, 2048]`` tables. The default mode retains the native sparse-plus-
    tail contract. ``full_visible=True`` selects exactly K=2048 resident slots
    without appending a tail, with a request union miss list of up to 32768
    entries. The query width comes from cumulative query lengths.
    Native host validation checks the remaining
    metadata, layout and device contract without reading device scalar values.
    ``resident_refresh_tail=True`` enables source-aware overlap for native LIM
    plans whose rewritten-resident entries form a suffix of at most Q items.
    It requires full-visible mode; other plans retain the complete-copy path.
    Buffer ownership stays with the runtime manager. An injected operator must
    honor the same validation contract as the native host implementation.
    """
    operator(
        query,
        query_rope,
        hbm_ckv,
        hbm_rope,
        actual_seq_lengths_query,
        actual_seq_lengths_kv,
        num_cache_tokens,
        topk_src_ids,
        topk_dst_slots,
        topk_miss_counts,
        miss_src_ids,
        miss_dst_slots,
        miss_counts,
        hbm_block_table,
        dram_block_table,
        dram_kv_dev_ptr,
        attention_out,
        scale_value,
        full_visible=full_visible,
        resident_refresh_tail=resident_refresh_tail,
        current=current,
        positions=positions,
        valid=valid,
        request_slots=request_slots,
        req_to_token=req_to_token,
        resident_begin=resident_begin,
        resident_end=resident_end,
        host_token_capacity=host_token_capacity,
        dram_rope_dev_ptr=dram_rope_dev_ptr,
    )
