"""CPU device operations for the SRT platform layer."""

import gc
import platform as _platform
from functools import cached_property
from typing import Optional

import psutil
import torch

from sglang.srt.platforms.device_mixin import (
    CpuArchEnum,
    DeviceCapability,
    DeviceMixin,
    PlatformEnum,
)
from sglang.srt.platforms.interface import SRTPlatform


class CpuDeviceMixin(DeviceMixin):
    """CPU implementation of the shared device operations."""

    _enum: PlatformEnum = PlatformEnum.CPU
    device_name: str = "cpu"
    device_type: str = "cpu"

    @cached_property
    def cpu_arch(self) -> CpuArchEnum:
        """Host CPU architecture (X86 / ARM / UNSPECIFIED), resolved once.

        First-class identity attribute parallel to ``_enum`` — callers branch
        on CPU arch through this instead of recomputing ``platform.machine()``.
        ``get_cpu_architecture()`` is process-stable, so caching is safe.
        """
        return self.get_cpu_architecture()

    def get_device_total_memory(self, device_id: int = 0) -> int:
        return int(psutil.virtual_memory().total)

    def get_current_memory_usage(
        self, device: Optional["torch.device"] = None
    ) -> float:
        """Whole-machine used memory (``total - available``) in bytes.

        Chosen so the [Active] contract
        ``free = get_device_total_memory() - get_current_memory_usage()``
        yields ``psutil.available`` — the real free RAM on a machine shared
        with the OS and other processes. Per-process RSS would wrongly ignore
        their usage. There is no per-device allocator peak on CPU (unlike
        ``torch.cuda.max_memory_allocated``), so this is current usage, not a
        peak. Returns whole-machine bytes; per-rank NUMA division for CPU TP
        is the caller's concern (kept in ``get_available_gpu_memory``'s CPU
        branch), not here.
        """
        vm = psutil.virtual_memory()
        return float(vm.total - vm.available)

    def get_device(self, local_rank: int) -> "torch.device":
        # local_rank is ignored: all CPU ranks share the one CPU device, so
        # there is nothing rank-specific to return. PyTorch enforces this —
        # Device::validate() asserts a CPU index must be -1 or 0 (c10/core/
        # Device.h). Per-rank isolation is done via OpenMP/numactl binding
        # (ModelRunner.init_threads_binding), not the device object.
        # TODO(zijiexia): make per-rank placement NUMA-affinity aware
        # (rank -> NUMA node) when the platform layer takes this over.
        return torch.device("cpu")

    def set_device(self, device: "torch.device") -> None:
        # Documented no-op on CPU — torch.cpu.set_device is "in CPU we do
        # nothing". Called (rather than left as ``pass``) for symmetry with
        # CudaDeviceMixin.set_device. Note this is deliberately NOT
        # torch.set_default_device("cpu"), which would flip the process-wide
        # default tensor device; per-rank CPU isolation is via OpenMP/numactl
        # binding (see get_device), not here.
        torch.cpu.set_device(device)

    def get_device_name(self, device_id: int = 0) -> str:
        # Arch-only label. We deliberately avoid platform.processor(): it
        # spawns a subprocess (~ms) on some platforms (e.g. macOS) and on Linux
        # is usually empty or redundant with the arch (e.g. "x86_64: x86_64").
        if self.cpu_arch == CpuArchEnum.ARM:
            return "cpu (aarch64)"
        if self.cpu_arch == CpuArchEnum.X86:
            return "cpu (x86_64)"
        return "cpu"

    def get_device_uuid(self, device_id: int = 0) -> str:
        # CPU has no per-device UUID; return the arch string as a stable
        # host-level identifier (matches the multimodal CpuPlatform).
        return _platform.machine()

    def get_device_capability(self, device_id: int = 0) -> Optional[DeviceCapability]:
        return None

    def empty_cache(self) -> None:
        # No torch.cpu.empty_cache() exists; do a GC pass at the teardown
        # points where this is called (flush_cache, idle sleep, weight reload).
        #
        # gc.collect() caveats:
        # - the pause grows with heap size (full walk of tracked objects);
        # - it only reclaims reference cycles — refcounting already frees
        #   everything else, so it may do little;
        # - freed memory returns to the allocator, not the OS, so RSS may not
        #   drop. glibc malloc_trim would not help: it is a no-op under the
        #   tcmalloc / TBB malloc the CPU guide preloads via LD_PRELOAD. Real
        #   RSS reclaim belongs in a separate allocator-aware, benchmarked
        #   change.
        gc.collect()

    def synchronize(self) -> None:
        # Documented no-op on CPU (no async streams to drain). Called for
        # symmetry with CudaDeviceMixin's torch.cuda.synchronize().
        torch.cpu.synchronize()

    def get_available_memory(self, device_id: int = 0) -> tuple[int, int]:
        vm = psutil.virtual_memory()
        return (vm.available, vm.total)

    def get_torch_distributed_backend_str(self) -> str:
        return "gloo"


class CpuSRTPlatform(CpuDeviceMixin, SRTPlatform):
    """Default in-tree CPU SRT platform.

    supports_fp8 / support_cuda_graph / support_piecewise_cuda_graph keep the
    conservative SRTPlatform defaults (all False), so they are not repeated
    here. Only is_pin_memory_available is overridden: the base defaults to
    True, but CPU has no GPU to pin host memory to.
    """

    def is_pin_memory_available(self) -> bool:
        return False

    def get_graph_runner_cls(self) -> type:
        from sglang.srt.model_executor.cpu_graph_runner import CPUGraphRunner

        return CPUGraphRunner
