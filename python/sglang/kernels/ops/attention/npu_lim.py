"""Adapter for the optional Ascend FusedLightningIndexerManage native extension.

The in-place ABI is implemented by sgl-kernel-npu. Its optional full-visible
mode extends the default 19-tensor call with a scalar flag. Native source and
direct AscendC launch live in that repository; importing this
module neither imports torch_npu nor loads/compiles a native library.
"""

from __future__ import annotations

import torch


def load_fused_lim_operator():
    """Resolve the operator from the installed sgl-kernel-npu wheel."""
    try:
        from sgl_kernel_npu.sparsity_driven_kv_offload import (
            fused_lightning_indexer_manage as operator,
        )

        # Resolve eagerly before graph capture, including the native registration.
        torch.ops.npu.fused_lightning_indexer_manage.default
        return operator
    except (ImportError, AttributeError) as error:
        raise RuntimeError(
            "LIM requires an sgl-kernel-npu A2/A3 build containing "
            "npu::fused_lightning_indexer_manage. Build/install the matching "
            "kernel wheel; no external LIM library or custom vendor is loaded."
        ) from error


def fused_lightning_indexer_manage(
    *,
    weights: torch.Tensor,
    query_scale: torch.Tensor,
    query: torch.Tensor,
    key_scale: torch.Tensor,
    key: torch.Tensor,
    block_table: torch.Tensor,
    query_ends: torch.Tensor,
    seq_lens: torch.Tensor,
    prefix_lens: torch.Tensor,
    cache_sizes: torch.Tensor,
    request_state: torch.Tensor,
    request_slots: torch.Tensor,
    slot_pool: torch.Tensor,
    source_ids: torch.Tensor,
    device_slots: torch.Tensor,
    topk_misses: torch.Tensor,
    miss_sources: torch.Tensor,
    miss_slots: torch.Tensor,
    miss_counts: torch.Tensor,
    operator=None,
    full_visible=False,
) -> None:
    """Launch BF16/FP16, page-128, TopK-2048 native LIM.

    The experimental full-visible mode requires a matching kernel wheel.
    """
    if query.ndim != 3 or query.shape[1] not in (32, 64) or query.shape[2] != 128:
        raise ValueError("LIM requires query [T, 32|64, 128].")
    if query.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("LIM requires FP16/BF16 query, index keys and weights.")
    if key.ndim != 4 or key.shape[1:] != (128, 1, 128):
        raise ValueError("LIM requires paged index keys [pages, 128, 1, 128].")
    if key.dtype != query.dtype or weights.dtype != query.dtype:
        raise ValueError("LIM query/key/weights dtypes must match.")
    if slot_pool.shape[1] != block_table.shape[1] * 128:
        raise ValueError(
            "LIM source-map width must equal index block-table width * 128."
        )
    inputs = (
        weights,
        query_scale,
        query,
        key_scale,
        key,
        block_table,
        query_ends,
        seq_lens,
        prefix_lens,
        cache_sizes,
        request_state,
        request_slots,
        slot_pool,
        source_ids,
        device_slots,
        topk_misses,
        miss_sources,
        miss_slots,
        miss_counts,
    )
    if any(not tensor.is_contiguous() for tensor in inputs):
        raise ValueError("LIM inputs and preallocated outputs must be contiguous.")
    if any(tensor.device != query.device for tensor in inputs):
        raise ValueError("LIM inputs and outputs must be on one device.")
    if operator is None:
        operator = load_fused_lim_operator()
    if full_visible:
        operator(*inputs, full_visible=True)
    else:
        operator(*inputs)
