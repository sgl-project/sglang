"""Page-granular HiCache transfers for the ``page_first_direct`` host layout.

These back ``--hicache-io-backend direct_kernel``: page-row copies
(``kvcacheio/page_row.cuh``) that read page ids on the GPU and access the pinned
host pool in place. The ``direct`` backend instead syncs the indices to the CPU
and builds copy-engine descriptors there, submitted as one cudaMemcpyBatchAsync
when the runtime has it and as per-page copies otherwise.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch

from sglang.kernels.jit.utils import is_hip_runtime
from sglang.kernels.ops.kvcache.page_row import (
    prepare_page_ids,
    transfer_page_rows,
    warmup_page_row_transfer,
)
from sglang.srt.utils.device import check_cuda_result

# Caps of 304-1024 CTAs measured as fast as an uncapped grid on GB300 (152 SMs);
# 1024 leaves room for GPUs with more SMs.
_BLOCK_QUOTA = 1024


class HiCacheDirectKernel:
    """page_row transfers for one contiguous ``[host_page, layer, ...]`` host pool.

    ``device_layers[l]`` is the device buffer stored as host layer ``l``; its pages
    are one host row (``host_pool[0, 0]``) wide. Construction checks the pool and
    compiles the kernels, so it fails at startup rather than on a transfer;
    ``unsupported`` is the pool's own reason it cannot use them, if any.
    """

    def __init__(
        self,
        *,
        pool_name: str,
        unsupported: Optional[str],
        host_pool: torch.Tensor,
        device_layers: Sequence[torch.Tensor],
        page_size: int,
    ):
        reason = unsupported or _unavailable_reason(host_pool[0, 0].nbytes)
        if reason is not None:
            raise ValueError(
                f"{pool_name} cannot use HiCache IO backend 'direct_kernel': {reason}"
            )
        num_layers = host_pool.shape[1]
        assert host_pool.is_contiguous() and len(device_layers) == num_layers
        self.row_bytes = host_pool[0, 0].nbytes
        self.host_stride = num_layers * self.row_bytes
        self.page_size = page_size
        self.host_tokens = host_pool.shape[0] * page_size
        self.device_tokens = (
            min(layer.nbytes for layer in device_layers) // self.row_bytes * page_size
        )
        device = device_layers[0].device
        self.host_ptrs = torch.tensor(
            [host_pool.data_ptr() + i * self.row_bytes for i in range(num_layers)],
            dtype=torch.uint64,
            device=device,
        )
        self.device_ptrs = torch.tensor(
            [layer.data_ptr() for layer in device_layers],
            dtype=torch.uint64,
            device=device,
        )
        # Built once: slicing per call is a measurable share of a small transfer.
        self.layer_ptrs = [
            (self.host_ptrs[i : i + 1], self.device_ptrs[i : i + 1])
            for i in range(num_layers)
        ]
        one_page = torch.arange(page_size, device=device)
        self.page_ids(one_page, one_page)

    def page_ids(self, host_indices, device_indices):
        """Validated page ids of paired whole-page CUDA token slots."""
        return prepare_page_ids(
            host_indices,
            device_indices,
            self.host_tokens,
            self.device_tokens,
            page_size=self.page_size,
        )

    def copy(self, host_pages, device_pages, *, load, host_layer=None):
        """Copy ``page_ids`` pages of every layer, or of ``host_layer``, on the
        current stream."""
        host_ptrs, device_ptrs = (
            (self.host_ptrs, self.device_ptrs)
            if host_layer is None
            else self.layer_ptrs[host_layer]
        )
        transfer_page_rows(
            host_ptrs,
            device_ptrs,
            host_pages,
            device_pages,
            host_stride=self.host_stride,
            device_stride=self.row_bytes,
            row_bytes=self.row_bytes,
            load=load,
            block_quota=_BLOCK_QUOTA,
        )


def _unavailable_reason(page_bytes: int) -> Optional[str]:
    if is_hip_runtime() or not torch.cuda.is_available():
        return "it is CUDA-only"
    if page_bytes <= 0 or page_bytes % 128 != 0:
        return f"{page_bytes} B pages are not a positive multiple of 128 B"
    try:
        warmup_page_row_transfer(block_quota=_BLOCK_QUOTA)
    except Exception as e:
        return f"its kernels failed to load: {e}"
    import cuda.bindings.runtime as cuda_rt

    device_id = torch.cuda.current_device()
    (host_pointer_usable,) = check_cuda_result(
        cuda_rt.cudaDeviceGetAttribute(
            cuda_rt.cudaDeviceAttr.cudaDevAttrCanUseHostPointerForRegisteredMem,
            device_id,
        )
    )
    if not host_pointer_usable:
        # The tables hold CPU virtual addresses, not cudaHostGetDevicePointer aliases.
        return (
            f"GPU {device_id} cannot access registered host memory at its CPU "
            "virtual address (cudaDevAttrCanUseHostPointerForRegisteredMem is 0)"
        )
    return None
