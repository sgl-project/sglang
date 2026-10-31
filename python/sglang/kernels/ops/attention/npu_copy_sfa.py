"""Public adapter for the normal-wheel Ascend combined-cache Copy-SFA operator."""

from __future__ import annotations

import torch


def load_copy_sfa_operator():
    """Resolve the registered A3 implementation before graph capture."""
    try:
        import sgl_kernel_npu  # noqa: F401

        return torch.ops.npu.fused_scatter_copy_sparse_flash_attention.default
    except (ImportError, AttributeError) as error:
        raise RuntimeError(
            "Copy-SFA requires an sgl-kernel-npu A3 wheel containing "
            "npu::fused_scatter_copy_sparse_flash_attention."
        ) from error


def fused_scatter_copy_sparse_flash_attention(
    query,
    query_rope,
    combined_hbm_cache,
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
    operator=None,
    full_visible=False,
    resident_refresh_tail=False,
) -> None:
    """Copy resident KV and attend to selected combined-cache tokens.

    Query tensors use contiguous TND ``[T, H, 512|64]`` and request-level
    cumulative query lengths. The default mode retains the native sparse-plus-
    tail contract. ``full_visible=True`` selects exactly K=2048 resident slots
    without appending a tail; Q4 supports C=12288 and a request union miss list
    with up to 32768 entries. Native host validation checks the remaining
    metadata, layout and device contract without reading device scalar values.
    ``resident_refresh_tail=True`` enables source-aware overlap for native LIM
    plans whose rewritten-resident entries form a suffix of at most Q items.
    It requires full-visible mode; other plans retain the complete-copy path.
    Buffer ownership stays with the runtime manager.
    """
    if resident_refresh_tail and not full_visible:
        raise ValueError("Resident-refresh overlap requires full_visible=True.")
    if query.ndim != 3 or query.shape[2] != 512:
        raise ValueError("Copy-SFA query must be [T,H,512].")
    if query_rope.shape != (*query.shape[:2], 64):
        raise ValueError("Copy-SFA query rope must be [T,H,64].")
    if query.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("Copy-SFA requires FP16/BF16 query and KV storage.")
    if operator is None:
        operator = load_copy_sfa_operator()
    inputs = (
        query,
        query_rope,
        combined_hbm_cache,
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
    )
    if resident_refresh_tail:
        operator(*inputs, full_visible=True, resident_refresh_tail=True)
    elif full_visible:
        operator(*inputs, full_visible=True)
    else:
        operator(*inputs)
