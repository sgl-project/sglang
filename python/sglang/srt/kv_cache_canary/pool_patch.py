from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Tuple

import torch

if TYPE_CHECKING:
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

logger = logging.getLogger(__name__)

_CANARY_POOL_ATTR = "_kv_cache_canary_attached"


def attach_shadow_buffers(pool: "MHATokenToKVPool") -> None:
    if getattr(pool, _CANARY_POOL_ATTR, False):
        return

    layer_num = pool.layer_num
    if layer_num <= 0:
        raise RuntimeError(f"kv-canary: pool has invalid layer_num={layer_num}")

    k_template = pool.k_buffer[0]
    v_template = pool.v_buffer[0]

    k_shape = tuple(k_template.shape)
    v_shape = tuple(v_template.shape)
    dtype = k_template.dtype
    device = k_template.device

    pool.canary_k_head = torch.zeros(k_shape, dtype=dtype, device=device)
    pool.canary_k_tail = torch.zeros(k_shape, dtype=dtype, device=device)
    pool.canary_v_head = torch.zeros(v_shape, dtype=dtype, device=device)
    pool.canary_v_tail = torch.zeros(v_shape, dtype=dtype, device=device)

    pool.canary_slot_stride_bytes = int(k_template[0].nbytes)
    pool.canary_num_slots = int(k_template.shape[0])

    _patch_get_contiguous_buf_infos(pool)

    setattr(pool, _CANARY_POOL_ATTR, True)
    logger.info(
        "kv-canary: attached shadow tensors to MHATokenToKVPool "
        "(k_shape=%s, v_shape=%s, dtype=%s, slot_stride_bytes=%d)",
        k_shape,
        v_shape,
        dtype,
        pool.canary_slot_stride_bytes,
    )


def _patch_get_contiguous_buf_infos(pool: "MHATokenToKVPool") -> None:
    original = pool.get_contiguous_buf_infos

    def patched_get_contiguous_buf_infos() -> Tuple[List[int], List[int], List[int]]:
        kv_data_ptrs, kv_data_lens, kv_item_lens = original()
        num = len(kv_data_ptrs) // 2
        k_ptrs = kv_data_ptrs[:num]
        v_ptrs = kv_data_ptrs[num:]
        k_lens = kv_data_lens[:num]
        v_lens = kv_data_lens[num:]
        k_item_lens = kv_item_lens[:num]
        v_item_lens = kv_item_lens[num:]

        k_head = pool.canary_k_head
        k_tail = pool.canary_k_tail
        v_head = pool.canary_v_head
        v_tail = pool.canary_v_tail

        new_ptrs = (
            k_ptrs
            + [k_head.data_ptr(), k_tail.data_ptr()]
            + v_ptrs
            + [v_head.data_ptr(), v_tail.data_ptr()]
        )
        new_lens = (
            k_lens
            + [k_head.nbytes, k_tail.nbytes]
            + v_lens
            + [v_head.nbytes, v_tail.nbytes]
        )
        new_item_lens = (
            k_item_lens
            + [k_head[0].nbytes * pool.page_size, k_tail[0].nbytes * pool.page_size]
            + v_item_lens
            + [v_head[0].nbytes * pool.page_size, v_tail[0].nbytes * pool.page_size]
        )
        return new_ptrs, new_lens, new_item_lens

    pool.get_contiguous_buf_infos = patched_get_contiguous_buf_infos


def get_shadow_buffers(
    pool: "MHATokenToKVPool",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        pool.canary_k_head,
        pool.canary_k_tail,
        pool.canary_v_head,
        pool.canary_v_tail,
    )
