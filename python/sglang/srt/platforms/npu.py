"""Ascend NPU device operations for the SRT platform layer.

``NpuDeviceMixin`` implements the shared :class:`DeviceMixin` device ops on
top of the ``torch.npu`` / ``torch_npu`` API surface, mirroring
:class:`~sglang.srt.platforms.cuda.CudaDeviceMixin`. ``torch.npu`` is touched
only inside methods (never at import time) so this module loads on non-NPU
hosts — the same lazy-import discipline used by the NPU graph runner and
backend.

``NpuSRTPlatform`` adds the SRT subsystem factory hooks (graph runner, KV
pools, paged allocator, piecewise backend) so the in-tree decode/prefill
paths can resolve NPU classes through ``current_platform`` instead of
``device == "npu"`` string checks.
"""

from typing import Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.platforms.device_mixin import (
    DeviceCapability,
    DeviceMixin,
    PlatformEnum,
)
from sglang.srt.platforms.interface import SRTPlatform


class NpuDeviceMixin(DeviceMixin):
    """Ascend NPU implementation of the shared device operations."""

    _enum: PlatformEnum = PlatformEnum.NPU
    device_name: str = "npu"
    device_type: str = "npu"

    def get_device_total_memory(self, device_id: int = 0) -> int:
        return int(torch.npu.mem_get_info(device_id)[1])

    def get_current_memory_usage(
        self, device: Optional["torch.device"] = None
    ) -> float:
        return float(torch.npu.max_memory_allocated(device))

    def get_device(self, local_rank: int) -> "torch.device":
        return torch.device("npu", local_rank)

    def set_device(self, device: "torch.device") -> None:
        torch.npu.set_device(device)

    def get_device_name(self, device_id: int = 0) -> str:
        return str(torch.npu.get_device_name(device_id))

    def get_device_uuid(self, device_id: int = 0) -> str:
        # Ascend NPUs expose no NVML/CUDA-style device UUID (NPUDeviceProperties
        # has no `uuid` field), and no in-tree path consumes one — the CUDA-IPC
        # uuid path in utils.patch_torch is CUDA-only. Return the device name as
        # a stable identifier, mirroring CpuDeviceMixin's use of a host-level id
        # in place of a real UUID.
        return str(torch.npu.get_device_name(device_id))

    def get_device_capability(self, device_id: int = 0) -> Optional[DeviceCapability]:
        # NPU has no CUDA-style (major, minor) compute capability; in-tree code
        # paths skip the capability query on NPU (e.g. model_loader.loader).
        return None

    def empty_cache(self) -> None:
        torch.npu.empty_cache()

    def synchronize(self) -> None:
        torch.npu.synchronize()

    def get_available_memory(self, device_id: int = 0) -> tuple[int, int]:
        return torch.npu.mem_get_info(device_id)

    def get_torch_distributed_backend_str(self) -> str:
        # zbal is the Ascend zero-bubble balanced backend; it replaces hccl when
        # a local memory budget is configured.
        return "hccl" if envs.SGLANG_ZBAL_LOCAL_MEM_SIZE.get() <= 0 else "zbal"

    @classmethod
    def seed_everything(cls, seed: int | None = None) -> None:
        if seed is not None:
            super().seed_everything(seed)
            torch.npu.manual_seed_all(seed)


class NpuSRTPlatform(NpuDeviceMixin, SRTPlatform):
    """Default in-tree Ascend NPU SRT platform."""

    def support_cuda_graph(self) -> bool:
        # NPUGraphRunner / NPUCudaGraphBackend
        return True

    def support_piecewise_cuda_graph(self) -> bool:
        # NPUPiecewiseBackend.
        return True

    def get_default_attention_backend(self) -> str:
        return "ascend"

    def get_compile_backend(self, mode: str | None = None) -> str:
        try:
            import torchair
            import torchair.ge_concrete_graph.ge_converter.experimental.patch_for_hcom_allreduce  # noqa: F401
            from torchair.configs.compiler_config import CompilerConfig
        except ImportError as e:
            raise ImportError(
                "NPU detected, but torchair package is not installed. "
                "Please install torchair for torch.compile support on NPU."
            ) from e
        compiler_config = CompilerConfig()
        compiler_config.mode = "max-autotune"
        if mode == "npugraph_ex":
            compiler_config.mode = "reduce-overhead"
            compiler_config.debug.run_eagerly = True
        return torchair.get_npu_backend(compiler_config=compiler_config)

    def get_dispatch_key_name(self) -> str:
        return "npu"

    def get_graph_runner_cls(self) -> type:
        from sglang.srt.hardware_backend.npu.graph_runner.npu_graph_runner import (
            NPUGraphRunner,
        )

        return NPUGraphRunner

    def get_mha_kv_pool_cls(self) -> type:
        from sglang.srt.hardware_backend.npu.memory_pool_npu import NPUMHATokenToKVPool

        return NPUMHATokenToKVPool

    def get_mla_kv_pool_cls(self) -> type:
        from sglang.srt.hardware_backend.npu.memory_pool_npu import NPUMLATokenToKVPool

        return NPUMLATokenToKVPool

    def get_paged_allocator_cls(self) -> type:
        from sglang.srt.hardware_backend.npu.allocator_npu import (
            NPUPagedTokenToKVPoolAllocator,
        )

        return NPUPagedTokenToKVPoolAllocator

    def get_piecewise_backend_cls(self) -> type:
        from sglang.srt.compilation.npu_piecewise_backend import NPUPiecewiseBackend

        return NPUPiecewiseBackend
