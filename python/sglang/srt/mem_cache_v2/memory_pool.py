from abc import ABC, abstractmethod
from typing import NamedTuple

import torch
import triton
import triton.language as tl

from sglang.srt.utils import next_power_of_2


class BufferInfo(NamedTuple):
    data_ptr: int  # base address
    data_len: int  # buffer size in bytes
    # the stride used as address = base_address + index * item_len
    item_len: int  # stride for physical indices


class MemoryPool(ABC):
    """
    Memory pool is the wrapper of the underlying tensors.

    It maps the logical indices to physical indices, and to memory addresses.
    """

    def __init__(
        self,
        size: int,
        page_size: int,
        layer_num: int,
        dtype: torch.dtype,
        device: str,
        layer_id: list[int] | None = None,
    ):
        self.size = size
        self.page_size = page_size
        self.layer_num = layer_num
        self.dtype = dtype
        self.device = device
        if layer_id is None:
            layer_id = list(range(layer_num))
        self.layer_id = layer_id

    @abstractmethod
    def clear(self) -> None:
        pass

    @abstractmethod
    def transform_indices(self, indices: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def get_buf_info(self) -> list[BufferInfo]:
        pass

    def get_kv_buffer(self, layer_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        pass

    def set_kv_buffer(
        self, layer_id: int, tgt_loc: torch.Tensor, src_loc: torch.Tensor
    ) -> None:
        pass

    def move_kv_cache(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor) -> None:
        N = tgt_loc.numel()
        if N == 0:
            return
        assert (
            tgt_loc.device == src_loc.device == self.device
        ), "All tensors must be on the same device"

        buf_infos = self.get_buf_info()
        stride_bytes = buf_infos[0].item_len
        assert all(
            info.item_len == stride_bytes for info in buf_infos
        ), "All buffers must have identical stride"

        # Prepare data pointers tensor
        data_ptrs = torch.tensor(
            [info.data_ptr for info in buf_infos],
            dtype=torch.uint64,
            device=self.device,
        )

        N_upper = next_power_of_2(N)

        max_byte_tiles = (stride_bytes + 127) // 128
        grid = (len(buf_infos), max_byte_tiles)

        copy_all_layer_kv_cache_tiled[grid](
            data_ptrs,
            tgt_loc,
            src_loc,
            N,
            N_upper,
            STRIDE=stride_bytes,
        )


class MHAMemoryPool(MemoryPool):
    def __init__(
        self,
        size: int,
        page_size: int,
        layer_num: int,
        head_num: int,
        head_dim: int,
        dtype: torch.dtype,
        device: str,
        layer_id: list[int] | None = None,
    ):
        super().__init__(size, page_size, layer_num, dtype, device, layer_id)
        self.head_num = head_num
        self.head_dim = head_dim

    def _create_buffers(self):
        # create buffers
        self.k_buffers = [
            torch.zeros(
                (self.size + self.page_size, self.head_num, self.head_dim),
                dtype=self.dtype,
                device=self.device,
            )
            for layer_id in self.layer_id
        ]
        self.v_buffers = [
            torch.zeros(
                (self.size + self.page_size, self.head_num, self.head_dim),
                dtype=self.dtype,
                device=self.device,
            )
            for layer_id in self.layer_id
        ]

    def clear(self) -> None:
        self._create_buffers()

    def get_kv_buffer(self, layer_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.k_buffers[layer_id], self.v_buffers[layer_id]

    def set_kv_buffer(
        self, layer_id: int, tgt_loc: torch.Tensor, src_loc: torch.Tensor
    ) -> None:
        k_buffer, v_buffer = self.get_kv_buffer(layer_id)
        k_buffer[tgt_loc] = k_buffer[src_loc]
        v_buffer[tgt_loc] = v_buffer[src_loc]

    def transform_indices(self, indices: torch.Tensor) -> torch.Tensor:
        return indices

    def get_buf_info(self) -> list[BufferInfo]:
        return [
            BufferInfo(
                data_ptr=buffer.data_ptr(),
                data_len=buffer.nbytes,
                item_len=buffer.stride(0) * buffer.dtype.itemsize * self.page_size,
            )
            for buffer in self.k_buffers + self.v_buffers
        ]


@triton.autotune(
    configs=[
        triton.Config({"BYTES_PER_TILE": tile}, num_warps=warps, num_stages=stages)
        for tile in [128, 256, 512]
        for warps in [4, 8]
        for stages in [2, 3]
    ],
    key=[],
)
@triton.jit
def copy_all_layer_kv_cache_tiled(
    data_ptrs,
    tgt_loc_ptr,
    src_loc_ptr,
    num_locs,
    num_locs_upper: tl.constexpr,
    STRIDE: tl.constexpr,
    BYTES_PER_TILE: tl.constexpr,
):
    """2D tiled kernel for bulk KV cache copy. Safe for in-place copy."""
    bid = tl.program_id(0)
    tid = tl.program_id(1)

    base_ptr = tl.load(data_ptrs + bid)
    base_ptr = tl.cast(base_ptr, tl.pointer_type(tl.uint8))

    byte_off = tid * BYTES_PER_TILE + tl.arange(0, BYTES_PER_TILE)
    mask_byte = byte_off < STRIDE
    tl.multiple_of(byte_off, 16)

    loc_idx = tl.arange(0, num_locs_upper)
    mask_loc = loc_idx < num_locs

    src = tl.load(src_loc_ptr + loc_idx, mask=mask_loc, other=0)
    tgt = tl.load(tgt_loc_ptr + loc_idx, mask=mask_loc, other=0)

    src_ptr = base_ptr + src[:, None] * STRIDE + byte_off[None, :]
    tgt_ptr = base_ptr + tgt[:, None] * STRIDE + byte_off[None, :]

    mask = mask_loc[:, None] & mask_byte[None, :]
    vals = tl.load(src_ptr, mask=mask)
    tl.store(tgt_ptr, vals, mask=mask)
