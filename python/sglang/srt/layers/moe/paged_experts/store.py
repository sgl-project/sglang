"""The host copy of every expert and the transfer of experts into GPU slots."""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import torch


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
        return torch.empty(shape, dtype=dtype, device="cpu", pin_memory=self.pin)

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

        hosts, gpus, words = [], [], []
        for name, host in self.host.items():
            gpu = getattr(layer, name).data
            row_bytes = host[0].numel() * host.element_size()
            if row_bytes == 0:
                continue  # e.g. GPTQ without act-order keeps empty g_idx tensors
            if not (self.pin and gpu.is_contiguous() and row_bytes % 4 == 0):
                raise RuntimeError(
                    f"Paged experts: CUDA graphs need pinned, contiguous expert rows of whole "
                    f"4-byte words; {name} has {row_bytes} bytes per expert, "
                    f"pinned={host.is_pinned()}, contiguous={gpu.is_contiguous()}"
                )
            hosts.append(paged_experts_host_device_pointer(host))
            gpus.append(gpu.data_ptr())
            words.append(row_bytes // 4)
        self._gather_args = [
            torch.tensor(v, dtype=torch.int64, device=gpu.device)
            for v in (hosts, gpus, words)
        ]

    def gather(self, src: torch.Tensor, dst: torch.Tensor, count: torch.Tensor) -> None:
        """Copy expert ``src[i]`` into GPU slot ``dst[i]`` for ``i < count[0]``, with every
        index read on the GPU (a device-planned step): capturable in a CUDA graph."""
        from sglang.kernels.ops.moe.paged_experts import paged_experts_gather

        paged_experts_gather(*self._gather_args, src, dst, count)
