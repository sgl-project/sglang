"""The host copy of every expert and the transfer of experts into GPU slots."""

from __future__ import annotations

import logging
import math
import os
import weakref
from typing import Dict, Sequence, Tuple

import torch

logger = logging.getLogger(__name__)


def _pin_until_freed(shape, dtype: torch.dtype) -> torch.Tensor:
    """Page-locked host memory of exactly the tensor's size (torch's pinned allocator rounds
    every allocation up to a power of two): shared memory of that size, registered with CUDA
    in place, as multimodal_gen's layerwise offload does.

    The registration lasts as long as the storage, so as long as any tensor or view on it.
    When the storage is freed, the device first finishes its queued reads (page-in copies,
    gather), then the pages are unregistered, before torch releases them. Stores live as long
    as their model, so that sync happens only at exit or in tests.
    """
    nbytes = math.prod(shape) * dtype.itemsize
    fd = os.memfd_create("sglang-paged-experts", os.MFD_CLOEXEC)
    try:
        os.ftruncate(fd, nbytes)
        storage = torch.UntypedStorage.from_file(
            f"/proc/self/fd/{fd}", shared=True, nbytes=nbytes
        )
    finally:
        os.close(fd)
    rc = int(torch.cuda.cudart().cudaHostRegister(storage.data_ptr(), nbytes, 0))
    if rc != 0:
        raise RuntimeError(
            f"Paged experts: cudaHostRegister of {nbytes} bytes failed (error {rc})"
        )
    weakref.finalize(storage, _unregister, storage.data_ptr())
    # On the CPU, as its storage, also under a default CUDA device (model construction).
    return torch.empty(0, dtype=dtype, device="cpu").set_(storage, 0, tuple(shape))


def _unregister(data_ptr: int) -> None:
    if torch.cuda.is_initialized():
        torch.cuda.synchronize()
    rc = int(torch.cuda.cudart().cudaHostUnregister(data_ptr))
    if rc != 0:
        logger.warning("Paged experts: cudaHostUnregister failed (error %d)", rc)


class HostExpertStore:
    """Holds all E experts of each paged tensor on the host, in the layer's GPU layout.
    Page-locked (``pin``) when it serves paging; a store only staged for a post-load repack
    stays pageable. ``page_in`` is planned on the host; ``gather`` reads a plan on the GPU.

    The GPU tensors are looked up on the layer at every transfer, because a quantization
    method may rebind them after loading.
    """

    def __init__(self, layer, names: Sequence[str], num_experts: int, pin: bool = True):
        self.pin = pin
        self.host: Dict[str, torch.Tensor] = {
            name: self._alloc(
                shape=(num_experts, *getattr(layer, name).shape[1:]),
                dtype=getattr(layer, name).dtype,
            )
            for name in names
        }

    def _alloc(self, shape, dtype: torch.dtype) -> torch.Tensor:
        if not self.pin or math.prod(shape) == 0:
            return torch.empty(shape, dtype=dtype, device="cpu")
        return _pin_until_freed(shape, dtype)

    def page_in(self, layer, loads: Sequence[Tuple[int, int]]) -> None:
        """Copy each ``(expert, slot)`` pair's expert, of every paged tensor, into its slot."""
        for name, host in self.host.items():
            gpu = getattr(layer, name).data
            for expert, slot in loads:
                gpu[slot].copy_(host[expert], non_blocking=True)

    def bind_device_gather(self, layer) -> None:
        """Prepare ``gather``; called once the layer's tensors are final. The gather kernel
        reads the pinned store directly, through its UVA address."""
        from sglang.kernels.ops.moe.paged_experts import (
            paged_experts_host_device_pointer,
        )

        hosts, gpus, sizes = [], [], []
        for name, host in self.host.items():
            gpu = getattr(layer, name).data
            row_bytes = host[0].numel() * host.element_size()
            if row_bytes == 0:
                continue  # e.g. GPTQ without act-order keeps empty g_idx tensors
            if not (self.pin and gpu.is_contiguous() and row_bytes % 2 == 0):
                raise RuntimeError(
                    f"Paged experts: CUDA graphs need pinned, contiguous expert rows of an "
                    f"even size; {name} has {row_bytes} bytes per expert, "
                    f"pinned={host.is_pinned()}, contiguous={gpu.is_contiguous()}"
                )
            hosts.append(paged_experts_host_device_pointer(host))
            gpus.append(gpu.data_ptr())
            sizes.append(row_bytes)
        self._gather_args = [
            torch.tensor(v, dtype=torch.int64, device=gpu.device)
            for v in (hosts, gpus, sizes)
        ]

    def gather(self, src: torch.Tensor, dst: torch.Tensor, count: torch.Tensor) -> None:
        """Copy expert ``src[i]`` into GPU slot ``dst[i]`` for ``i < count[0]``, with every
        index read on the GPU (a device-planned step): capturable in a CUDA graph."""
        from sglang.kernels.ops.moe.paged_experts import paged_experts_gather

        paged_experts_gather(*self._gather_args, src, dst, count)
