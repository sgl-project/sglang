from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt
import torch

from sglang.kernels.ops.kvcache.pd_dcp_gather import copy_mla_rows_into_pack
from sglang.srt.disaggregation.common.staging_buffer import StagingBuffer
from sglang.srt.runtime_context import get_schedule, max_prefill_buffer_tokens

logger = logging.getLogger(__name__)


def dcp_pack_buffer_bytes(
    kv_item_lens: Sequence[int], page_size: int, max_tokens: int, dcp_size: int = 1
) -> int:
    if page_size <= 0:
        raise ValueError(f"page_size must be positive, got {page_size}")
    if dcp_size <= 0:
        raise ValueError(f"dcp_size must be positive, got {dcp_size}")
    if any(item_len < page_size for item_len in kv_item_lens):
        raise ValueError(
            "PD DCP pack requires each kv_item_len to span at least one page, "
            f"got {list(kv_item_lens)} with page_size={page_size}"
        )
    if any(item_len % page_size != 0 for item_len in kv_item_lens):
        raise ValueError(
            "PD DCP pack requires page-aligned kv_item_lens, "
            f"got {list(kv_item_lens)} with page_size={page_size}"
        )
    token_item_lens = [item_len // page_size for item_len in kv_item_lens]
    rank_tokens = (max_tokens + dcp_size - 1) // dcp_size
    return dcp_size * rank_tokens * sum(token_item_lens)


def try_pack_dcp_src(
    *,
    pack_buffer: StagingBuffer,
    kv_data_ptrs: Sequence[int],
    src_token_indices: npt.NDArray[np.integer],
    token_item_lens: Sequence[int],
    pack_offset_bytes: int = 0,
) -> Optional[Tuple[List[int], npt.NDArray[np.int64]]]:
    if pack_offset_bytes < 0:
        raise ValueError(
            f"pack_offset_bytes must be non-negative, got {pack_offset_bytes}"
        )
    n = int(src_token_indices.size)
    if n == 0:
        empty = np.empty((0,), dtype=np.int64)
        return [], empty
    required = n * sum(int(item_len) for item_len in token_item_lens)
    required_end = pack_offset_bytes + required
    if not pack_buffer.fits(required_end):
        logger.warning(
            "PD DCP pack buffer too small for byte range [%s, %s) (have %s); "
            "falling back to per-token RDMA",
            pack_offset_bytes,
            required_end,
            pack_buffer.get_size(),
        )
        return None

    pack = pack_buffer.buffer.narrow(0, pack_offset_bytes, required)
    row_indices = torch.as_tensor(
        src_token_indices, device=pack.device, dtype=torch.int64
    )
    gather_stream = pack_buffer.get_gather_stream()
    gather_stream.wait_stream(torch.cuda.default_stream(pack.device))
    with torch.cuda.stream(gather_stream):
        copy_mla_rows_into_pack(kv_data_ptrs, row_indices, pack, token_item_lens)
    gather_stream.synchronize()

    packed_ptrs: List[int] = []
    offset = 0
    base = pack_buffer.get_ptr() + pack_offset_bytes
    for item_len in token_item_lens:
        packed_ptrs.append(base + offset)
        offset += n * int(item_len)
    return packed_ptrs, np.arange(n, dtype=np.int64)


def init_dcp_pack_buffers(
    register_fn,
    kv_args,
    count: int,
    dcp_size: int,
) -> List[StagingBuffer]:
    from sglang.srt.disaggregation.common.staging_handler import (
        _get_custom_mem_pool,
    )

    max_tokens = max_prefill_buffer_tokens()
    if max_tokens <= 0:
        max_tokens = get_schedule().max_prefill_tokens
    # Note(kpham-sgl): size = dcp_size x ceil(max_tokens / dcp_size)
    # x sum(per-layer token bytes). At 32,768 tokens and 61 MLA layers
    # x 576 bf16 dims x 2 B: 2.14 GiB/buffer, 8.58 GiB for 4 queues.
    size_bytes = dcp_pack_buffer_bytes(
        kv_args.kv_item_lens, kv_args.page_size, max_tokens, dcp_size
    )
    gpu_id = kv_args.gpu_id
    device = f"cuda:{gpu_id}"
    custom_mem_pool, _ = _get_custom_mem_pool(device)

    buffers = []
    for _ in range(count):
        buf = StagingBuffer(size_bytes, device, gpu_id, custom_mem_pool=custom_mem_pool)
        register_fn(buf.get_ptr(), buf.get_size())
        buffers.append(buf)
    logger.info(
        "PD DCP pack buffers allocated: %d x %.1f MB (max_tokens=%d)",
        count,
        size_bytes / (1024 * 1024),
        max_tokens,
    )
    return buffers
