"""Built-in SRT platform implementations."""

from __future__ import annotations

import logging
import os

import torch

from sglang.srt.platforms.device_mixin import PlatformEnum
from sglang.srt.platforms.interface import SRTPlatform, TorchCompileStrategy

logger = logging.getLogger(__name__)


class BuiltinPlatform(SRTPlatform):
    """Common behavior for in-tree hardware platforms."""

    def get_compile_backend(self, mode: str | None = None):
        return "inductor"

    def get_dispatch_key_name(self) -> str:
        return self.device_name

    def get_device_total_memory(self, device_id: int = 0) -> int:
        device_module = torch.get_device_module(self.device_type)
        try:
            return device_module.get_device_properties(device_id).total_memory
        except AttributeError:
            try:
                _, total = device_module.mem_get_info(device_id)
                return total
            except AttributeError as exc:
                raise NotImplementedError from exc

    def get_current_memory_usage(self, device: torch.device | None = None) -> float:
        device_module = torch.get_device_module(self.device_type)
        try:
            return float(device_module.max_memory_allocated(device))
        except AttributeError:
            return 0.0


class CudaPlatform(BuiltinPlatform):
    _enum = PlatformEnum.CUDA
    device_name = "cuda"
    device_type = "cuda"

    def support_cuda_graph(self) -> bool:
        return True

    def support_piecewise_cuda_graph(self) -> bool:
        return True


class OnnxExportPlatformMixin:
    """Mixin for platforms that execute exported callsites through ONNX."""

    def torch_compile_strategy(self) -> TorchCompileStrategy:
        return "export"

    def torch_compile_defaults(self):
        from sglang.srt.compilation.torch_compile import TorchCompileConfig

        return TorchCompileConfig(
            export_format="onnx",
            run_exported=True,
            forced_fields={"export_format", "run_exported"},
        )

    def get_export_runtime(self, compile_config):
        from sglang.srt.compilation.export_backends import OnnxExportRuntime

        return OnnxExportRuntime()


class CudaOnnxPlatform(OnnxExportPlatformMixin, CudaPlatform):
    """CUDA-backed demo platform that runs decorated targets through ONNX."""

    device_name = "cuda_onnx"


class OnnxPlatform(OnnxExportPlatformMixin, BuiltinPlatform):
    """Graph-runtime-only ONNX demo platform without CUDA graph assumptions."""

    _enum = PlatformEnum.CPU
    device_name = "onnx"
    device_type = "cpu"


class RocmPlatform(BuiltinPlatform):
    _enum = PlatformEnum.ROCM
    device_name = "rocm"
    device_type = "cuda"

    def support_cuda_graph(self) -> bool:
        return True

    def get_dispatch_key_name(self) -> str:
        return "hip"


class CpuPlatform(BuiltinPlatform):
    _enum = PlatformEnum.CPU
    device_name = "cpu"
    device_type = "cpu"

    def get_device_total_memory(self, device_id: int = 0) -> int:
        import os

        if hasattr(os, "sysconf"):
            return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        raise NotImplementedError


class XpuPlatform(BuiltinPlatform):
    _enum = PlatformEnum.XPU
    device_name = "xpu"
    device_type = "xpu"


class MusaPlatform(BuiltinPlatform):
    _enum = PlatformEnum.MUSA
    device_name = "musa"
    device_type = "musa"


class NpuPlatform(BuiltinPlatform):
    _enum = PlatformEnum.NPU
    device_name = "npu"
    device_type = "npu"

    def torch_compile_strategy(self) -> TorchCompileStrategy:
        try:
            import torchair  # noqa: F401
        except ImportError:
            return "noop"
        return "compile"

    def get_compile_backend(self, mode: str | None = None):
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


_BUILTIN_PLATFORM_CLASSES = {
    "cuda": CudaPlatform,
    "cuda_onnx": CudaOnnxPlatform,
    "onnx": OnnxPlatform,
    "rocm": RocmPlatform,
    "hip": RocmPlatform,
    "cpu": CpuPlatform,
    "xpu": XpuPlatform,
    "musa": MusaPlatform,
    "npu": NpuPlatform,
}


def get_builtin_platform_names() -> tuple[str, ...]:
    return tuple(_BUILTIN_PLATFORM_CLASSES)


def resolve_builtin_platform_by_name(name: str) -> SRTPlatform | None:
    """Resolve a selected in-tree platform name, or None if not built in."""
    platform_cls = _BUILTIN_PLATFORM_CLASSES.get(name)
    if platform_cls is None:
        return None

    if not _is_builtin_platform_available(name):
        raise RuntimeError(
            f"SGLANG_PLATFORM={name!r} selects an in-tree platform, "
            "but that hardware is not available on this machine."
        )
    return platform_cls()


def _is_builtin_platform_available(name: str) -> bool:
    if name in ("cpu", "onnx"):
        return _is_cpu()
    if name in ("cuda", "cuda_onnx"):
        return _is_cuda()
    if name in ("rocm", "hip"):
        return _is_hip()
    if name == "xpu":
        return _is_xpu()
    if name == "musa":
        return _is_musa()
    if name == "npu":
        return _is_npu()
    return False


def resolve_builtin_platform() -> SRTPlatform:
    """Detect the active in-tree platform."""

    if _is_cpu():
        return CpuPlatform()
    if _is_npu():
        return NpuPlatform()
    if _is_xpu():
        return XpuPlatform()
    if _is_musa():
        return MusaPlatform()
    if _is_hip():
        return RocmPlatform()
    if _is_cuda():
        return CudaPlatform()

    logger.debug("No built-in platform detected. Using base SRTPlatform with defaults.")
    return SRTPlatform()


def _is_cuda() -> bool:
    return torch.cuda.is_available() and torch.version.cuda is not None


def _is_hip() -> bool:
    return torch.version.hip is not None


def _is_cpu() -> bool:
    machine = os.uname().machine.lower() if hasattr(os, "uname") else ""
    is_host_cpu_supported = machine in (
        "x86_64",
        "amd64",
        "i386",
        "i686",
        "arm64",
        "aarch64",
    )
    return os.getenv("SGLANG_USE_CPU_ENGINE", "0") == "1" and is_host_cpu_supported


def _is_xpu() -> bool:
    return hasattr(torch, "xpu") and torch.xpu.is_available()


def _is_musa() -> bool:
    try:
        import torchada  # noqa: F401
    except ImportError:
        return False
    return hasattr(torch.version, "musa") and torch.version.musa is not None


def _is_npu() -> bool:
    return hasattr(torch, "npu") and torch.npu.is_available()
