"""ROCm index-K rows of `DeepSeekV4IndexerPool`: the FlyDSL kernels read the fp4 payload
and the packed ue8m0 scales from two buffers (`uses_aiter_fp4_layout`), where the
CUDA pool keeps one `[.., page_size * 68]`-byte page."""

from __future__ import annotations

import torch


def store_fp4_index_k(pool, layer_idx: int, loc: torch.Tensor, cache_k: torch.Tensor):
    from sglang.kernels.ops.attention.dsv4.fp4_indexer_hip import (
        store_fp4_index_k_cache_split,
    )

    return store_fp4_index_k_cache_split(
        cache_k,
        pool.index_k_payload_buffer[layer_idx],
        pool.index_k_scale_buffer[layer_idx],
        loc,
        page_size=pool.page_size,
        rne=pool.index_k_rne,
    )


def read_fp4_index_k(pool, layer_idx: int, slots: torch.Tensor):
    """(payload int8 [n, 64], scales int32 [n]) at `slots`."""
    from sglang.kernels.ops.attention.dsv4.fp4_indexer_hip import (
        read_fp4_index_k_split,
    )

    return read_fp4_index_k_split(
        pool.index_k_payload_buffer[layer_idx],
        pool.index_k_scale_buffer[layer_idx],
        slots,
        page_size=pool.page_size,
    )


def dequant_fp4_index_k(pool, layer_idx: int, slots) -> torch.Tensor:
    """Dequantized bf16 [n, index_head_dim] index K at `slots` (every slot when None)."""
    from sglang.srt.layers.quantization.fp8 import DSV4_DEQUANT_FP4_TABLE

    if slots is None:
        slots = torch.arange(pool.size, device=pool.device)
    payload, packed = read_fp4_index_k(pool, layer_idx, slots.to(torch.int64))
    payload_u8 = payload.view(torch.uint8)  # [n, 64]
    scale_exps = torch.stack(
        [(packed >> (8 * c)) & 0xFF for c in range(4)], dim=-1
    )  # [n, 4]
    fp4_codes = torch.stack(
        [payload_u8 & 0x0F, (payload_u8 >> 4) & 0x0F], dim=-1
    )  # [n, 64, 2]
    dequant = DSV4_DEQUANT_FP4_TABLE.to(payload_u8.device)[fp4_codes.long()].flatten(
        1
    )  # [n, 128]
    scales = torch.exp2(scale_exps.float() - 127).repeat_interleave(32, dim=-1)
    return (dequant * scales).to(torch.bfloat16)
