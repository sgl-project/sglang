# SPDX-License-Identifier: Apache-2.0
"""Fixed startup GLM DSA backing. No resize, staging pool, or allocation fallback.

The CUDA array interface owner is retained by PyTorch's storage deleter (and
therefore by tensor views). Never use non-owning pointer storage for this pool.
"""

import logging
import sys
from math import prod
from typing import NamedTuple

import torch

logger = logging.getLogger(__name__)
_QUARANTINE = []
_MAX_BYTES = 2**63 - 1


class BufferSlice(NamedTuple):
    offset: int
    nbytes: int


def _plan_buffers(shapes, itemsize):
    """Pure byte planning: reject malformed/overflowing layouts before CUDA."""
    if type(itemsize) is not int or itemsize <= 0:
        raise ValueError("dtype itemsize must be a positive integer")
    try:
        shapes = [tuple(shape) for shape in shapes]
    except TypeError as exc:
        raise ValueError("fixed buffers require integer shapes") from exc
    if not shapes or any(
        not shape or any(type(d) is not int or d <= 0 for d in shape)
        for shape in shapes
    ):
        raise ValueError("fixed buffers require nonempty positive integer shapes")
    slices, end = [], 0
    for shape in shapes:
        nbytes = prod(shape) * itemsize
        offset = (end + 255) // 256 * 256
        if nbytes > _MAX_BYTES or offset + nbytes > _MAX_BYTES:
            raise ValueError("fixed buffer byte extent exceeds signed 64-bit range")
        slices.append(BufferSlice(offset, nbytes))
        end = offset + nbytes
    return shapes, slices, end


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
    kv_scale_ue8m0,
):
    """Fail before allocating any tensor for unsupported physical layouts."""
    integers = (size, page_size, layer_num, kv_lora_rank, qk_rope_head_dim)
    integers += (kv_cache_dim, index_head_dim, index_kpool)
    if (
        any(type(value) is not int for value in integers)
        or (index_buf_size is not None and type(index_buf_size) is not int)
        or size <= 0
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
        or kv_scale_ue8m0
    ):
        raise ValueError(
            "fixed FABRIC pool requires the 78-layer GLM layout: raw FP8 MLA rows "
            "of 576 bytes, FP8 index keys of 128 with FP32 scales, page size 64; "
            "no index compression, skip-topk, FP4 indexer, UE8M0 scales, "
            "DCP or memory saver"
        )


class _FabricAllocation:
    def __init__(self, nbytes, device_id):
        from sglang.srt.utils import cuda_vmm_utils as vmm

        self.device_id, self.nbytes = device_id, nbytes
        self.reservation = None
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
            if self.size > _MAX_BYTES:
                raise ValueError(
                    "rounded FABRIC allocation exceeds signed 64-bit range"
                )
            self.reservation = vmm.VmmReservation(
                self.size, prop, device_id, alignment=self.granularity
            )
            self.base = self.reservation.base
            handle = self.reservation.map(0, self.size, retain_handle=True)
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

    def __del__(self):
        # At interpreter exit torch and logging may be gone; process exit frees it.
        if self.reservation is None or sys.is_finalizing():
            return
        try:
            # External memory is outside torch's caching allocator and stream
            # recording. Drain before releasing even when the last view dies.
            with torch.cuda.device(self.device_id):
                torch.cuda.synchronize(self.device_id)
                self.reservation.close()
            self.reservation = None
        except Exception:
            # Uncertain GPU completion must leak, never free in-flight memory.
            _QUARANTINE.append(self)
            logger.exception("fixed FABRIC allocation quarantined during teardown")


class FixedFabricBufferOwner:
    """Carve 256-byte aligned buffers from one FABRIC allocation.

    close drops references; live tensor views keep the allocation alive.
    """

    def __init__(self, *, device, dtype, shapes):
        self.tensors, self.buffer_slices = [], []
        self.allocation = None
        self.closed = False
        shapes, slices, total = _plan_buffers(shapes, dtype.itemsize)
        device = torch.device(device)
        if device.type != "cuda":
            raise ValueError("fixed FABRIC backing requires CUDA")
        self.device_id = (
            torch.cuda.current_device() if device.index is None else device.index
        )
        self.dtype = dtype
        try:
            self.allocation = _FabricAllocation(total, self.device_id)
            root = torch.as_tensor(self.allocation, device=f"cuda:{self.device_id}")
            for shape, span in zip(shapes, slices):
                self.tensors.append(
                    root.narrow(0, span.offset, span.nbytes)
                    .view(dtype)
                    .reshape(shape)
                    .zero_()
                )
            self.buffer_slices = slices
        except BaseException:
            self.close()
            raise

    def close(self):
        self.closed = True
        self.tensors.clear()
        self.buffer_slices.clear()
        self.allocation = None
