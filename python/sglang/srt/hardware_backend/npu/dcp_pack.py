"""NPU buffers for the shared DCP gather and bulk-transfer path."""

from functools import partial

import torch
import triton
import triton.language as tl

from sglang.srt.disaggregation.common.dcp_pack import (
    allocate_registered_pack_buffers,
    dcp_pack_buffer_bytes,
)
from sglang.srt.disaggregation.common.staging_buffer import StagingBuffer


@triton.jit
def _pack_rows_kernel(
    metadata,
    indices,
    pack,
    num_rows,
    ROW_BYTES: tl.constexpr,
    SRC_STRIDE: tl.constexpr,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
    COL_TILES: tl.constexpr,
):
    entry = tl.program_id(0)
    tile = tl.program_id(1)
    source = tl.load(metadata + 2 * entry).to(pack.dtype)
    prefix_bytes = tl.load(metadata + 2 * entry + 1)
    rows = tile // COL_TILES * ROWS + tl.arange(0, ROWS)
    cols = tile % COL_TILES * COLS + tl.arange(0, COLS)
    source_rows = tl.load(indices + rows, rows < num_rows, other=0)
    mask = (rows[:, None] < num_rows) & (cols[None, :] < ROW_BYTES)
    values = tl.load(
        source + source_rows[:, None] * SRC_STRIDE + cols[None, :], mask, other=0
    )
    tl.store(
        pack
        + num_rows * prefix_bytes
        + rows[:, None].to(tl.int64) * ROW_BYTES
        + cols[None, :],
        values,
        mask,
    )


class NPUDCPPackBuffer(StagingBuffer):
    """Use the NPU allocator with the shared staging stream/size interface."""

    def __init__(self, size_bytes: int, gpu_id: int):
        self.size_bytes = size_bytes
        self.device = torch.device("npu", gpu_id)
        self.gpu_id = gpu_id
        self.device_module = torch.get_device_module(self.device)
        self.device_module.set_device(gpu_id)
        self.buffer = torch.empty(size_bytes, dtype=torch.uint8, device=self.device)
        self.data_ptr = self.buffer.data_ptr()
        self._gather_stream = None
        self._pack_key = None
        self._pack_groups = []

    def pack_rows(self, pointers, indices, pack, widths, strides=None):
        """Specialize row widths so Ascend can copy contiguous row tiles."""
        strides = widths if strides is None else strides
        key = (tuple(pointers), tuple(widths), tuple(strides))
        if key != self._pack_key:
            groups, prefix_bytes = {}, 0
            for pointer, width, stride in zip(*key):
                # A per-row prefix lets the cached table also serve tail chunks.
                groups.setdefault((width, stride), []).extend((pointer, prefix_bytes))
                prefix_bytes += width
            self._pack_groups = [
                (
                    width,
                    stride,
                    torch.tensor(table, dtype=torch.int64, device=self.device),
                )
                for (width, stride), table in groups.items()
            ]
            self._pack_key = key
        num_rows = indices.numel()
        for width, stride, table in self._pack_groups:
            cols = min(1024, triton.next_power_of_2(width))
            col_tiles = triton.cdiv(width, cols)
            grid = (table.numel() // 2, triton.cdiv(num_rows, 8) * col_tiles)
            _pack_rows_kernel[grid](
                table,
                indices,
                pack,
                num_rows,
                ROW_BYTES=width,
                SRC_STRIDE=stride,
                ROWS=8,
                COLS=cols,
                COL_TILES=col_tiles,
            )


def init_npu_dcp_pack_buffers(
    register_fn,
    kv_args,
    count,
    dcp_size,
    max_tokens,
    *,
    device_type,
    include_draft=False,
):
    # Index-K and replicated draft rows use their own source buffers.
    item_lens = kv_args.kv_item_lens
    if kv_args.num_draft_entries:
        item_lens = item_lens[: -kv_args.num_draft_entries]
    size_bytes = dcp_pack_buffer_bytes(
        item_lens, kv_args.page_size, max_tokens, dcp_size
    )
    make_buffer = partial(NPUDCPPackBuffer, size_bytes, kv_args.gpu_id)
    return allocate_registered_pack_buffers(
        register_fn, make_buffer, count, size_bytes, max_tokens
    )
