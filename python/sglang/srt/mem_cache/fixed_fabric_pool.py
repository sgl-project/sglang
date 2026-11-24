# SPDX-License-Identifier: Apache-2.0
"""Fixed startup GLM DSA backing. No resize, staging pool, or allocation fallback.

The allocation stays mapped until the process exits; nothing releases it earlier.
"""

from math import prod
from typing import NamedTuple

import torch


class BufferSlice(NamedTuple):
    offset: int
    nbytes: int


def _plan_buffers(shapes, itemsize):
    """Place each buffer at the next 256-byte aligned offset."""
    slices, end = [], 0
    for shape in shapes:
        offset = (end + 255) // 256 * 256
        slices.append(BufferSlice(offset, prod(shape) * itemsize))
        end = offset + slices[-1].nbytes
    return slices, end


def validate_glm_geometry(
    *,
    size,
    page_size,
    layer_num,
    kv_lora_rank,
    qk_rope_head_dim,
    kv_cache_dim,
    index_head_dim,
    dtype,
    index_kpool,
    index_kpool_compress,
    tail_extra_slots,
    skip_topk_layers,
    index_buf_size,
    dcp_replicated,
    enable_memory_saver,
    use_fp4_indexer,
):
    """Fail before allocating any tensor for unsupported physical layouts."""
    if (
        size <= 0
        or size % 64
        or page_size != 64
        or layer_num != 78
        or kv_lora_rank != 512
        or qk_rope_head_dim != 64
        or kv_cache_dim != 576
        or index_head_dim != 128
        or dtype != torch.float8_e4m3fn
        or index_kpool != 1
        or index_kpool_compress
        or tail_extra_slots
        or (
            skip_topk_layers is not None
            and (len(skip_topk_layers) != layer_num or any(skip_topk_layers))
        )
        or index_buf_size not in (None, size)
        or dcp_replicated
        or enable_memory_saver
        or use_fp4_indexer
    ):
        raise ValueError(
            "fixed FABRIC pool requires the 78-layer GLM layout: raw FP8 MLA rows "
            "of 576 bytes, FP8 index keys of 128 with FP32 scales, page size 64; "
            "no index compression, skip-topk, FP4 indexer, DCP or memory saver"
        )


class _FabricAllocation:
    def __init__(self, nbytes, device_id):
        from sglang.srt.utils import cuda_vmm_utils as vmm

        self.device_id, self.nbytes = device_id, nbytes
        with torch.cuda.device(device_id):
            drv = vmm._get_cuda_driver()
            fabric = drv.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC
            # Explicit: unlike the ordinary VMM allocator, never probe POSIX/NONE
            # and never omit RDMA registration capability for the existing P->D path.
            prop = vmm.make_device_allocation_prop(
                device_id, handle_types=fabric, gpu_direct_rdma=True
            )
            self.granularity = vmm.get_allocation_granularity(prop)
            self.size = vmm.align_up(nbytes, self.granularity)
            self.reservation = vmm.VmmReservation(
                self.size, prop, device_id, alignment=self.granularity
            )
            self.base = self.reservation.base
            handle = self.reservation.map(0, self.size, retain_handle=True)
            # Export once here so a device without FABRIC support fails at startup.
            vmm.check_drv(
                drv.cuMemExportToShareableHandle(handle, fabric, 0),
                "cuMemExportToShareableHandle(fixed pool)",
            )
        self.__cuda_array_interface__ = {
            "shape": (nbytes,),
            "typestr": "|u1",
            "data": (self.base, False),
            "version": 3,
            "strides": None,
        }


class FixedFabricBufferOwner:
    """Carve 256-byte aligned buffers from one FABRIC allocation."""

    def __init__(self, *, device, dtype, shapes):
        device = torch.device(device)
        self.device_id = (
            torch.cuda.current_device() if device.index is None else device.index
        )
        self.dtype = dtype
        self.buffer_slices, total = _plan_buffers(shapes, dtype.itemsize)
        self.allocation = _FabricAllocation(total, self.device_id)
        root = torch.as_tensor(self.allocation, device=f"cuda:{self.device_id}")
        self.tensors = [
            root.narrow(0, span.offset, span.nbytes).view(dtype).reshape(shape).zero_()
            for shape, span in zip(shapes, self.buffer_slices)
        ]
