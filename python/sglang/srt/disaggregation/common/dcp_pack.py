from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt
import torch

from sglang.kernels.ops.kvcache.pd_dcp_gather import copy_mla_rows_into_pack
from sglang.srt.disaggregation.common.staging_buffer import StagingBuffer
from sglang.srt.utils import device_stream_context

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
    src_token_item_lens: Optional[Sequence[int]] = None,
    pack_offset_bytes: int = 0,
    pack_capacity_bytes: Optional[int] = None,
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
    if (
        pack_capacity_bytes is not None and required > pack_capacity_bytes
    ) or not pack_buffer.fits(required_end):
        logger.warning(
            "PD DCP pack buffer too small for byte range [%s, %s) "
            "(have %s, region capacity %s); "
            "falling back to per-token RDMA",
            pack_offset_bytes,
            required_end,
            pack_buffer.get_size(),
            pack_capacity_bytes,
        )
        return None

    pack = pack_buffer.buffer.narrow(0, pack_offset_bytes, required)
    row_indices = torch.as_tensor(
        src_token_indices, device=pack.device, dtype=torch.int64
    )
    gather_stream = pack_buffer.get_gather_stream()
    gather_stream.wait_stream(pack_buffer.producer_stream())
    with device_stream_context(gather_stream):
        copy_mla_rows_into_pack(
            kv_data_ptrs, row_indices, pack, token_item_lens, src_token_item_lens
        )
    gather_stream.synchronize()

    packed_ptrs: List[int] = []
    offset = 0
    base = pack_buffer.get_ptr() + pack_offset_bytes
    for item_len in token_item_lens:
        packed_ptrs.append(base + offset)
        offset += n * int(item_len)
    return packed_ptrs, np.arange(n, dtype=np.int64)


# copy_mla_rows_into_pack puts the row-dependent block count on grid.y.
_MAX_GATHER_GRID_Y = 65535
_GATHER_BLOCK_BYTES = 1024


def pack_dcp_row_segments(
    pack_buffer: StagingBuffer,
    segments: Sequence[Tuple[Sequence[int], npt.NDArray[np.integer], Sequence[int]]],
) -> Optional[List[List[int]]]:
    """Gather several row sets into the pack buffer, then wait once.

    Each segment is (source buffers, row indices, row bytes): row ``i`` of
    buffer ``j`` is the ``row_bytes[j]`` bytes at ``rows[i] * row_bytes[j]``.
    The segments are packed one after another, each buffer by buffer (a
    buffer's rows are contiguous, in ``rows`` order). Returns each segment's
    packed base per buffer, or None when the rows do not fit the buffer or one
    gather launch, so the caller falls back to smaller packs.
    """
    plan = []
    offset = 0
    for ptrs, rows, widths in segments:
        n = int(np.asarray(rows).size)
        widths = [int(width) for width in widths]
        if len(ptrs) != len(widths):
            raise ValueError(
                f"PD DCP pack segment has {len(ptrs)} buffers but {len(widths)} widths"
            )
        if n == 0 or not ptrs:
            plan.append((ptrs, rows, widths, offset, n))
            continue
        if -(-n * max(widths) // _GATHER_BLOCK_BYTES) > _MAX_GATHER_GRID_Y:
            return None
        plan.append((ptrs, rows, widths, offset, n))
        offset += n * sum(widths)
    if not pack_buffer.fits(offset):
        return None

    pack = pack_buffer.buffer
    # Transfer worker threads start on device 0 (see try_pack_dcp_src).
    torch.cuda.set_device(pack.device)
    gather_stream = pack_buffer.get_gather_stream()
    gather_stream.wait_stream(pack_buffer.producer_stream())
    bases: List[List[int]] = []
    base_ptr = pack_buffer.get_ptr()
    with device_stream_context(gather_stream):
        all_rows = [np.asarray(rows, dtype=np.int64) for _, rows, _, _, n in plan if n]
        if all_rows:
            # One host-to-device copy, ordered on the gather stream.
            row_indices = torch.from_numpy(np.concatenate(all_rows)).to(pack.device)
        row_offset = 0
        for ptrs, _, widths, seg_offset, n in plan:
            seg_bases = []
            entry_offset = seg_offset
            for width in widths:
                seg_bases.append(base_ptr + entry_offset)
                entry_offset += n * width
            bases.append(seg_bases)
            if n == 0 or not ptrs:
                continue
            copy_mla_rows_into_pack(
                list(ptrs),
                row_indices.narrow(0, row_offset, n),
                pack.narrow(0, seg_offset, n * sum(widths)),
                widths,
            )
            row_offset += n
    gather_stream.synchronize()
    return bases


def dsa_index_k_item_lens(kv_args) -> List[int]:
    if kv_args.dsa_index_k_token_bytes is None:
        return []
    from sglang.srt.disaggregation.base.conn import StateType

    return [
        item_len
        for state_type, item_lens in zip(kv_args.state_types, kv_args.state_item_lens)
        if state_type == StateType.DSA
        for item_len in item_lens
    ]


def init_dcp_pack_buffers(
    register_fn,
    kv_args,
    count: int,
    dcp_size: int,
    max_tokens: int,
    *,
    device_type: str,
    include_draft: bool = False,
) -> List[StagingBuffer]:
    from sglang.srt.disaggregation.common.staging_handler import (
        _bare_device_type,
        _get_custom_mem_pool,
    )

    device_type = _bare_device_type(device_type)
    # copy_mla_rows_into_pack packs kv_data_ptrs as int64, which rejects XPU
    # pointers with bit 63 set; no buffers routes callers to per-token RDMA.
    if device_type != "cuda":
        logger.info("PD DCP pack disabled on %s; using per-token transfer", device_type)
        return []

    kv_item_lens = kv_args.kv_item_lens
    if not include_draft and kv_args.num_draft_entries:
        kv_item_lens = kv_item_lens[: -kv_args.num_draft_entries]
    # Note(kpham-sgl): size = dcp_size x ceil(max_tokens / dcp_size)
    # x sum(per-layer token bytes). At 32,768 tokens and 61 MLA layers
    # x 576 bf16 dims x 2 B: 2.14 GiB/buffer, 8.58 GiB for 4 queues.
    size_bytes = dcp_pack_buffer_bytes(
        kv_item_lens, kv_args.page_size, max_tokens, dcp_size
    )
    index_k_token_bytes = sum(
        item_len // kv_args.page_size for item_len in dsa_index_k_item_lens(kv_args)
    )
    size_bytes += dcp_size * -(-max_tokens // dcp_size) * index_k_token_bytes
    gpu_id = kv_args.gpu_id
    device = f"{device_type}:{gpu_id}"
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
