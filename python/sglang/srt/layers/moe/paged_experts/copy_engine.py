"""Copy-engine staging for the device-planned CUDA gather."""

from __future__ import annotations

import weakref

import torch

from sglang.kernels.jit.utils import cache_once, load_jit
from sglang.kernels.jit.utils.compile.toolchain import cuda_stubs_dir
from sglang.kernels.ops.moe.paged_experts import paged_experts_host_device_pointer


@cache_once
def _module():
    exports = (
        "create_copy_engine",
        "destroy_copy_engine",
        "add_copy_engine_layer",
        "remove_copy_engine_layer",
        "gather_copy_engine",
    )
    return load_jit(
        "paged_experts_copy_engine",
        cuda_files=["moe/paged_experts_copy_engine.cuh"],
        cuda_wrappers=[(name, name) for name in exports],
        extra_ldflags=[f"-L{cuda_stubs_dir()}", "-lcuda", "-pthread"],
    )


class _Worker:
    def __init__(self, device: torch.device, delay_us: int):
        self.module = _module()
        self.handle = self.module.create_copy_engine(device.index, delay_us)
        self._close = weakref.finalize(
            self, self.module.destroy_copy_engine, self.handle
        )


_workers: weakref.WeakValueDictionary[torch.device, _Worker] = (
    weakref.WeakValueDictionary()
)


def _release(worker, handle, allocations):
    # The finalizer owns the tensors backing the native pointers until outstanding
    # replays and DMA have drained and the worker no longer holds this layer.
    worker.module.remove_copy_engine_layer(worker.handle, handle)


class CopyEngineTransfer:
    def __init__(
        self,
        hosts: list[torch.Tensor],
        slots: list[torch.Tensor],
        capacity: int,
        *,
        timeout_ns: int = 3_000_000,
        delay_us: int = 0,
        collect_stats: bool = False,
    ):
        device = slots[0].device
        self._timeout_ns = timeout_ns
        with torch.cuda.device(device):
            worker = _workers.get(device) if delay_us == 0 else None
            if worker is None:
                worker = _Worker(device, delay_us)
                if delay_us == 0:
                    _workers[device] = worker
            self._gather = worker.module.gather_copy_engine
            host_ptrs, uva, gpu_ptrs, sizes, staging = [], [], [], [], []
            for host, dest in zip(hosts, slots, strict=True):
                size = host[0].numel() * host.element_size()
                if size == 0:
                    continue
                if size % 16 or not dest.is_contiguous():
                    raise ValueError(
                        "Copy-engine staging requires contiguous, 16-byte-aligned rows"
                    )
                host_ptrs.append(host.data_ptr())
                uva.append(paged_experts_host_device_pointer(host))
                gpu_ptrs.append(dest.data_ptr())
                sizes.append(size)
                staging.append(
                    torch.empty((capacity, size), dtype=torch.uint8, device=device)
                )
            self.stats = (
                torch.zeros(2, dtype=torch.int64, device=device)
                if collect_stats
                else None
            )
            self._handle = worker.module.add_copy_engine_layer(
                worker.handle,
                host_ptrs,
                uva,
                gpu_ptrs,
                [tensor.data_ptr() for tensor in staging],
                sizes,
                capacity,
                self.stats.data_ptr() if self.stats is not None else 0,
            )
        self.close = weakref.finalize(
            self, _release, worker, self._handle, (hosts, slots, staging, self.stats)
        )

    def gather(self, src: torch.Tensor, dst: torch.Tensor, count: torch.Tensor) -> None:
        self._gather(self._handle, src, dst, count, self._timeout_ns)
