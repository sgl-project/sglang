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
        return float(psutil.Process().memory_info().rss)

    def get_device(self, local_rank: int) -> "torch.device":
        # ``local_rank`` is intentionally ignored. PyTorch's CPU device is
        # unindexed — ``torch.device("cpu", n)`` parses but the index is a
        # semantic no-op, unlike CUDA where it selects a physical GPU. CPU
        # tensor parallelism does have per-rank meaning (one rank per
        # sub-NUMA cluster), but that isolation is applied out-of-band via
        # OpenMP thread binding + numactl pinning (see
        # ModelRunner.init_threads_binding), not through the device object.
        return torch.device("cpu")

    def set_device(self, device: "torch.device") -> None:
        # No-op by design. On CUDA, set_device moves the current thread's
        # active-device cursor (so subsequent allocations land on that GPU).
        # CPU has a single unindexed logical device — there is no cursor to
        # move, and per-rank isolation is handled via OpenMP/numactl binding
        # (see get_device), not here.
        #
        # We also deliberately avoid ``torch.set_default_device("cpu")``: that
        # flips the *process-wide* default tensor device, which would break
        # code paths that explicitly build tensors on another device.
        pass

    def get_device_name(self, device_id: int = 0) -> str:
        proc = _platform.processor() or _platform.machine() or "unknown"
        if self.cpu_arch == CpuArchEnum.ARM:
            return f"cpu (aarch64: {proc})"
        if self.cpu_arch == CpuArchEnum.X86:
            return f"cpu (x86_64: {proc})"
        return f"cpu ({proc})"

    def get_device_uuid(self, device_id: int = 0) -> str:
        return _platform.machine()

    def get_device_capability(self, device_id: int = 0) -> Optional[DeviceCapability]:
        return None

    def empty_cache(self) -> None:
        # CPU has no device-side allocator cache to drop, and PyTorch exposes
        # no ``torch.cpu.empty_cache()`` (nor any arch-specific equivalent on
        # the x86/ARM CPUs we support). The honest portable action is a GC
        # pass to reclaim Python-level cycles at the teardown points where this
        # is called (Scheduler.flush_cache, periodic idle sleep, weight reload).
        #
        # This intentionally does NOT force freed arenas back to the OS. glibc
        # ``malloc_trim`` could, but it is (a) a no-op under the tcmalloc / TBB
        # malloc allocators the CPU deployment guide preloads via LD_PRELOAD,
        # and (b) on a periodic path. Real RSS reclaim, if needed, belongs in a
        # separate benchmarked change gated on the active allocator.
        gc.collect()

    def get_available_memory(self, device_id: int = 0) -> tuple[int, int]:
        vm = psutil.virtual_memory()
        return (vm.available, vm.total)

    def get_torch_distributed_backend_str(self) -> str:
        return "gloo"


class CpuSRTPlatform(CpuDeviceMixin, SRTPlatform):
    """Default in-tree CPU SRT platform."""

    def supports_fp8(self) -> bool:
        return False

    def support_cuda_graph(self) -> bool:
        return False

    def support_piecewise_cuda_graph(self) -> bool:
        return False

    def is_pin_memory_available(self) -> bool:
        return False
