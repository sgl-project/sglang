from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt
import torch

from sglang.kernels.ops.kvcache.pd_dcp_gather import copy_mla_rows_into_pack
from sglang.srt.disaggregation.common.staging_buffer import StagingBuffer
from sglang.srt.runtime_context import get_schedule

logger = logging.getLogger(__name__)


def dcp_pack_buffer_bytes(
    kv_item_lens: Sequence[int], page_size: int, max_tokens: int
) -> int:
    if page_size <= 0:
        raise ValueError(f"page_size must be positive, got {page_size}")
    token_item_lens = [item_len // page_size for item_len in kv_item_lens]
    if any(item_len <= 0 for item_len in token_item_lens):
        raise ValueError(
            "PD DCP pack requires page-aligned kv_item_lens, "
            f"got {list(kv_item_lens)} with page_size={page_size}"
        )
    return max_tokens * sum(token_item_lens)


def plan_packed_dcp_blocks(
    dst_token_indices: npt.NDArray[np.integer],
) -> List[Tuple[int, int, int]]:
    dst = np.asarray(dst_token_indices, dtype=np.int64)
    n = int(dst.size)
    if n == 0:
        return []
    brk = np.where(np.diff(dst) != 1)[0] + 1
    starts = np.concatenate((np.array([0], dtype=np.int64), brk))
    ends = np.concatenate((brk, np.array([n], dtype=np.int64)))
    return [
        (int(start), int(dst[start]), int(end - start))
        for start, end in zip(starts, ends)
    ]


def try_pack_dcp_src(
    *,
    pack_buffer: StagingBuffer,
    kv_data_ptrs: Sequence[int],
    src_token_indices: npt.NDArray[np.integer],
    token_item_lens: Sequence[int],
) -> Optional[Tuple[List[int], npt.NDArray[np.int64]]]:
    n = int(src_token_indices.size)
    if n == 0:
        empty = np.empty((0,), dtype=np.int64)
        return [], empty
    required = n * sum(int(item_len) for item_len in token_item_lens)
    if not pack_buffer.fits(required):
        logger.warning(
            "PD DCP pack buffer too small for %s bytes (have %s); "
            "falling back to per-token RDMA",
            required,
            pack_buffer.get_size(),
        )
        return None

    pack = pack_buffer.buffer
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
    base = pack_buffer.get_ptr()
    for item_len in token_item_lens:
        packed_ptrs.append(base + offset)
        offset += n * int(item_len)
    return packed_ptrs, np.arange(n, dtype=np.int64)


def init_dcp_pack_buffers(
    register_fn,
    kv_args,
    count: int,
) -> List[StagingBuffer]:
    from sglang.srt.disaggregation.common.staging_handler import (
        _get_custom_mem_pool,
    )

    chunk = get_schedule().chunked_prefill_size
    max_tokens = int(chunk) if chunk is not None and chunk > 0 else 32768
    size_bytes = dcp_pack_buffer_bytes(
        kv_args.kv_item_lens, kv_args.page_size, max_tokens
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
