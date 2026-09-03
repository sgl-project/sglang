"""CUDA-IPC descriptors for the physical data plane of a shared KV pool.

The descriptor intentionally owns no lifetime or allocation policy.  The shared
KV pool control service owns those concerns; this module only lets a consumer
map an owner-held, contiguous CUDA tensor at the same storage offset.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CudaIpcTensorDescriptor:
    """Serializable description of a contiguous CUDA tensor's backing storage.

    The exporting process must keep the original tensor alive for every mapped
    consumer.  Consumers should use the control-plane page generations before
    reading a page; mapping a storage handle alone is not a cache lease.
    """

    storage_handle: tuple
    dtype: torch.dtype
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    byte_offset: int
    nbytes: int

    @classmethod
    def export(cls, tensor: torch.Tensor) -> "CudaIpcTensorDescriptor":
        if not tensor.is_cuda:
            raise ValueError("CUDA IPC requires a CUDA tensor")
        if not tensor.is_contiguous():
            raise ValueError("CUDA IPC shared KV tensors must be contiguous")
        storage = tensor.untyped_storage()
        return cls(
            storage_handle=tuple(storage._share_cuda_()),
            dtype=tensor.dtype,
            shape=tuple(tensor.shape),
            stride=tuple(tensor.stride()),
            byte_offset=tensor.storage_offset() * tensor.element_size(),
            nbytes=tensor.numel() * tensor.element_size(),
        )

    def open(self, device_index: int) -> torch.Tensor:
        """Open the exported storage on ``cuda:device_index``.

        A same-GPU pool uses the same index for owner and consumer.  Redirecting
        the handle's first element mirrors SGLang's existing CUDA-IPC transport
        and keeps PyTorch's CUDA device guard on the receiving device.
        """
        if device_index < 0:
            raise ValueError("device_index must be non-negative")
        handle = (device_index,) + self.storage_handle[1:]
        device = torch.device(f"cuda:{device_index}")
        with torch.cuda.device(device):
            storage = torch.UntypedStorage._new_shared_cuda(*handle)
            view_storage = storage[self.byte_offset : self.byte_offset + self.nbytes]
            return torch.empty(0, dtype=self.dtype, device=device).set_(
                view_storage,
                storage_offset=0,
                size=self.shape,
                stride=self.stride,
            )
