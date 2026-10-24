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
) -> None:
    """Fill Q1 resident KV and attend to K=2048 selected combined-cache tokens.

    Resident capacity C is 2048, 4096 or 8192; request miss rows have width C.

    Native host validation checks the remaining metadata/layout/device contract
    without device scalar reads. Buffer ownership stays with the runtime manager.
    """
    if query.ndim != 3 or query.shape[2] != 512:
        raise ValueError("Copy-SFA query must be [B,H,512].")
    if query_rope.shape != (*query.shape[:2], 64):
        raise ValueError("Copy-SFA query rope must be [B,H,64].")
    if query.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("Copy-SFA requires FP16/BF16 query and KV storage.")
    if operator is None:
        operator = load_copy_sfa_operator()
    operator(
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
