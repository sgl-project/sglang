"""Prefill-side packing for DCP1 → DCP-N PD KV transfers.

Cyclic DCP ownership (``pos % c == r``) makes source rows strided in the dense
prefill pool. ``group_concurrent_contiguous`` then emits one RDMA per token.
This module gathers owned rows into a registered contiguous buffer so the
existing dest-contiguous grouping can emit page-sized (or larger) blocks.

Decode is already packed (``pos // c``), so dest scatter is not required.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt
import torch

from sglang.srt.disaggregation.common.staging_buffer import StagingBuffer
from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_schedule

logger = logging.getLogger(__name__)


def dcp_pack_max_tokens() -> int:
    override = envs.SGLANG_DISAGG_DCP_PACK_MAX_TOKENS.get()
    if override is not None and override > 0:
        return int(override)
    chunk = get_schedule().chunked_prefill_size
    if chunk is not None and chunk > 0:
        return int(chunk)
    return 32768


def dcp_pack_buffer_bytes(kv_item_lens: Sequence[int], page_size: int) -> int:
    if page_size <= 0:
        raise ValueError(f"page_size must be positive, got {page_size}")
    token_item_lens = [item_len // page_size for item_len in kv_item_lens]
    if any(item_len <= 0 for item_len in token_item_lens):
        raise ValueError(
            "PD DCP pack requires page-aligned kv_item_lens, "
            f"got {list(kv_item_lens)} with page_size={page_size}"
        )
    return dcp_pack_max_tokens() * sum(token_item_lens)


def plan_packed_dcp_blocks(
    dst_token_indices: npt.NDArray[np.integer],
) -> List[Tuple[int, int, int]]:
    """Split packed src ``0..N-1`` on dest-contiguous runs.

    Returns ``(src_start, dst_start, n_tokens)`` triples. Src is dense after
    gather, so a consecutive dest page becomes one block.
    """
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


def gather_mla_owned_tokens(
    kv_buffers: Sequence[torch.Tensor],
    src_token_indices: npt.NDArray[np.integer],
    pack: torch.Tensor,
    token_item_lens: Sequence[int],
    *,
    gpu_id: int,
) -> None:
    """Copy owned MLA rows into ``pack`` in dest-token order, layer-major."""
    n = int(src_token_indices.size)
    if n == 0:
        return
    if len(kv_buffers) != len(token_item_lens):
        raise ValueError(
            "kv_buffers and token_item_lens length mismatch: "
            f"{len(kv_buffers)} vs {len(token_item_lens)}"
        )

    if pack.device.type == "cuda":
        torch.cuda.set_device(gpu_id)
    idx = torch.as_tensor(src_token_indices, device=pack.device, dtype=torch.int64)
    offset = 0
    for buf, item_len in zip(kv_buffers, token_item_lens):
        row_nbytes = int(buf[0].nbytes)
        if row_nbytes != item_len:
            raise ValueError(
                "MLA token geometry mismatch during DCP pack: "
                f"buffer row={row_nbytes} bytes, item_len={item_len}"
            )
        gathered = buf.index_select(0, idx).contiguous()
        nbytes = n * item_len
        pack[offset : offset + nbytes].view(gathered.dtype).copy_(gathered.view(-1))
        offset += nbytes
    if pack.device.type == "cuda":
        torch.cuda.current_stream(pack.device).synchronize()


def try_pack_dcp_src(
    *,
    pack_buffer: StagingBuffer,
    kv_buffers: Sequence[torch.Tensor],
    src_token_indices: npt.NDArray[np.integer],
    token_item_lens: Sequence[int],
    gpu_id: int,
) -> Optional[Tuple[List[int], npt.NDArray[np.int64]]]:
    """Gather owned rows. Returns packed layer ptrs and dense src indices, or None."""
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

    gather_mla_owned_tokens(
        kv_buffers,
        src_token_indices,
        pack_buffer.buffer,
        token_item_lens,
        gpu_id=gpu_id,
    )
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
    """Allocate one registered pack buffer per transfer-queue shard."""
    from sglang.srt.disaggregation.common.staging_handler import (
        _get_custom_mem_pool,
    )

    size_bytes = dcp_pack_buffer_bytes(kv_args.kv_item_lens, kv_args.page_size)
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
        dcp_pack_max_tokens(),
    )
    return buffers
