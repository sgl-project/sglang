"""Per-device compile target and launch policy for the GDN CuTe-DSL kernels.

The DSL resolves its default ``GPUArch`` from ``CUTE_DSL_ARCH`` or CUDA device 0, so
compile options have to name the operand's device or a kernel gets built for someone
else's architecture. The compile-cache key names it for a separate reason, see
:attr:`GdnDeviceTarget.compile_key`.
"""

import functools
import os
import re
from typing import NamedTuple, Union

import cutlass.cute as cute
import torch

_ARCH_RE = re.compile(r"^sm_(\d+)(\d)[a-z]?$")


class GdnDeviceTarget(NamedTuple):
    """Architecture and launch policy of a single CUDA device."""

    device_index: int
    arch: str
    major: int
    minor: int
    num_sms: int
    use_packed_fma: bool

    @property
    def compile_key(self) -> tuple:
        """Compile-cache identity.

        The device index is part of it because these cache entries also hold
        device-resident defaults that nothing else keys by device. The compiled
        artifact itself is device agnostic from DSL 4.6.2 on: a TVM-FFI function's
        ``to()`` returns itself, and ``CudaDialectJitCompiledFunction.to()`` ignores
        its device argument and loads the cubin on every device.
        """
        return (self.device_index, self.arch)


def target_arch(target_key: tuple) -> str:
    """The arch component of :attr:`GdnDeviceTarget.compile_key`.

    On-disk artifacts are per arch, so the device index -- which separates
    in-process entries only -- must not reach a kernel's specialization name.
    """
    _device_index, arch = target_key
    return arch


def _arch_string(major: int, minor: int) -> str:
    return f"sm_{major}{minor}{'a' if major >= 9 else ''}"


def _dsl_runtime_arch() -> Union[str, None]:
    """The arch the DSL will accept a JIT engine for, or ``None`` if unavailable."""
    try:
        from cutlass.cutlass_dsl import CuTeDSL

        return CuTeDSL._get_dsl().envar.arch
    except Exception:
        return None


def _check_dsl_can_run(target: GdnDeviceTarget) -> None:
    """Reject a target the DSL would silently turn into a cross-compile.

    The DSL builds a JIT engine only when its process-global arch (``CUTE_DSL_ARCH``,
    else CUDA device 0) can run the requested target, so one process serves one
    architecture. Without this the failure surfaces as an internal DSL error, or on
    an unpinned build as ``cudaErrorNoKernelImageForDevice`` at launch.
    """
    runtime_arch = _dsl_runtime_arch()
    if runtime_arch is None or runtime_arch == target.arch:
        return
    try:
        from cutlass.base_dsl import Arch

        if Arch.from_string(runtime_arch).can_run_binary_built_for(
            Arch.from_string(target.arch)
        ):
            return
    except Exception:
        return
    raise RuntimeError(
        f"GDN CuTe-DSL cannot target cuda:{target.device_index} ({target.arch}): the "
        f"CuTe DSL compiles this process for {runtime_arch}, taken from CUDA device 0 "
        "unless CUTE_DSL_ARCH is set, and one process can only serve one architecture. "
        f"Set CUTE_DSL_ARCH={target.arch} or restrict CUDA_VISIBLE_DEVICES to devices "
        "of a single architecture."
    )


@functools.lru_cache(maxsize=None)
def _resolve(device_index: int) -> GdnDeviceTarget:
    major, minor = torch.cuda.get_device_capability(device_index)
    # CUTE_DSL_ARCH is the DSL's own cross-compile override; an explicit GPUArch would
    # otherwise beat it, so honor it here and derive the policy from it too.
    env_arch = os.environ.get("CUTE_DSL_ARCH")
    if env_arch:
        match = _ARCH_RE.match(env_arch)
        if match is None:
            raise ValueError(f"CUTE_DSL_ARCH is not a recognized arch: {env_arch!r}")
        major, minor = int(match.group(1)), int(match.group(2))
    target = GdnDeviceTarget(
        device_index=device_index,
        arch=env_arch or _arch_string(major, minor),
        major=major,
        minor=minor,
        num_sms=torch.cuda.get_device_properties(device_index).multi_processor_count,
        use_packed_fma=major >= 10,
    )
    _check_dsl_can_run(target)
    return target


def gdn_device_target(device: Union[str, torch.device]) -> GdnDeviceTarget:
    """Resolve the compile target from an operand's device."""
    if not isinstance(device, torch.device):
        device = torch.device(device)
    if device.type != "cuda":
        raise ValueError(f"GDN CuTe-DSL kernels require CUDA tensors, got {device}")
    index = device.index
    return _resolve(torch.cuda.current_device() if index is None else index)


@functools.lru_cache(maxsize=None)
def _arch_options(arch: str) -> tuple:
    return (cute.GPUArch(arch),)


def gdn_compile_options(device: Union[str, torch.device], *extra) -> tuple:
    """``cute.compile`` options pinned to ``device``'s architecture.

    Apply via ``cute.compile[options](...)``: a string ``options=`` kwarg replaces
    these wholesale rather than merging, which would silently drop the arch.
    """
    return _arch_options(gdn_device_target(device).arch) + tuple(extra)


__all__ = [
    "GdnDeviceTarget",
    "gdn_compile_options",
    "gdn_device_target",
    "target_arch",
]
