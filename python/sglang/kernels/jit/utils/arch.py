"""CUDA/ROCm architecture detection and default compile target flags."""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass
from typing import List

import torch

from sglang.kernels.jit.utils.common import (
    cache_once,
    is_hip_runtime,
    is_musa_runtime,
)

logger = logging.getLogger(__name__)


@dataclass
class ArchInfo:
    major: int
    minor: int
    suffix: str

    @property
    def target_name(self) -> str:
        return f"{self.major}.{self.minor}{self.suffix}"

    @property
    def jit_flag(self) -> str:
        return f"-DSGL_CUDA_ARCH={self.major * 100 + self.minor * 10}"


@cache_once
def _jit_cuda_version() -> tuple[int, ...]:
    """CUDA version of the nvcc that JIT builds actually run.

    The target has to match the compiler, not the toolkit PyTorch was built
    against: a cu129 wheel on a CUDA 12.8 toolkit would otherwise select
    `sm_120f`, which nvcc 12.8 rejects. Resolve nvcc the way tvm-ffi does
    (`CUDA_HOME` / `CUDA_PATH`, then `$PATH`, then `/usr/local/cuda`) and fall
    back to `torch.version.cuda` when it cannot be probed.
    """
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home is None:
        nvcc_path = shutil.which("nvcc")
        cuda_home = (
            os.path.dirname(os.path.dirname(nvcc_path))
            if nvcc_path is not None
            else "/usr/local/cuda"
        )
    nvcc = os.path.join(cuda_home, "bin", "nvcc")
    try:
        output = subprocess.check_output([nvcc, "--version"], text=True)
        match = re.search(r"release (\d+)\.(\d+)", output)
        if match is not None:
            return int(match.group(1)), int(match.group(2))
        logger.warning("Cannot parse `%s --version` output: %s", nvcc, output)
    except (OSError, subprocess.SubprocessError) as error:
        logger.warning("Cannot run `%s --version`: %s", nvcc, error)
    from sglang.srt.utils.common import get_cuda_version

    return get_cuda_version()


def _cuda_arch_suffix(major: int, minor: int) -> str:
    """Mirror FlashInfer's `_normalize_cuda_arch`: 9.x/10.x+ -> "a"; 12.0 -> "f"
    and 12.x (x>0) -> "a" (SM120/SM121 need separate cubins to avoid
    cudaErrorIllegalInstruction); below 9.0 -> plain.

    The family-specific "f" target needs a CUDA >= 12.9 nvcc; older toolkits
    fall back to "a", which SM120 has had since 12.8. Plain sm_120 is never a
    valid fallback: without an a/f target CUTLASS's SM120 atoms lose LDSM/STSM
    and compile down to trap stubs (verified via SASS), which asserts at launch
    instead of failing the build.
    """
    if major == 9:
        return "a"
    if major == 12:
        if minor == 0 and _jit_cuda_version() >= (12, 9):
            return "f"
        return "a"
    if major >= 10:
        return "a"
    return ""


@cache_once
def _init_jit_cuda_arch_once():
    global _CUDA_ARCH
    try:
        device = torch.cuda.current_device()
        major, minor = torch.cuda.get_device_capability(device)
    except Exception:
        logger.warning("Cannot detect CUDA architecture.")
        major, minor = 0, 0  # invalid value to trigger compile error if used
    # JIT builds target the exact local GPU, so the arch-specific target is
    # always correct on Hopper+ and unlocks arch-only instructions (redux.f32).
    # HIP/MUSA capability numbers aren't CUDA SM versions and stay unsuffixed.
    suffix = (
        ""
        if (is_hip_runtime() or is_musa_runtime())
        else _cuda_arch_suffix(major, minor)
    )
    _CUDA_ARCH = ArchInfo(major, minor, suffix)


def get_default_target_flags(arch: ArchInfo | None = None) -> List[str]:
    """Default compile flags for `arch`, defaulting to the detected local GPU."""
    if is_hip_runtime():
        flags = ["-DUSE_ROCM", "-std=c++20", "-O3"]
        # Detect FP8 type based on GPU architecture
        try:
            device = torch.cuda.current_device()
            gcn_arch = torch.cuda.get_device_properties(device).gcnArchName
            if "gfx942" in gcn_arch:
                flags.append("-DHIP_FP8_TYPE_FNUZ=1")
            else:
                flags.append("-DHIP_FP8_TYPE_E4M3=1")
        except Exception:
            flags.append("-DHIP_FP8_TYPE_E4M3=1")
        return flags
    else:
        if arch is None:
            arch = get_jit_cuda_arch()
        return [
            arch.jit_flag,
            "-std=c++20",
            "-O3",
            "--expt-relaxed-constexpr",
        ]


def make_jit_cuda_arch(major: int, minor: int) -> ArchInfo:
    """Build the JIT target for an explicitly requested capability."""
    return ArchInfo(major, minor, _cuda_arch_suffix(major, minor))


@contextmanager
def override_jit_cuda_arch(major: int, minor: int, suffix: str | None = None):
    """A context manager to temporarily override CUDA architecture.

    `suffix` defaults to the arch-specific target detection would pick for that
    capability; pass it explicitly only to force a different one (an unsuffixed
    target loses the arch-only instructions CUTLASS needs, see
    `_cuda_arch_suffix`).

    Kernels do not need this to reach an arch-specific target: `get_jit_cuda_arch`
    already resolves the local GPU to its a/f target. Reach for it only to compile
    for an arch the local GPU is not.
    """
    global _CUDA_ARCH
    old_value = get_jit_cuda_arch()
    _CUDA_ARCH = (
        make_jit_cuda_arch(major, minor)
        if suffix is None
        else ArchInfo(major, minor, suffix)
    )
    try:
        yield
    finally:
        _CUDA_ARCH = old_value


def get_jit_cuda_arch() -> ArchInfo:
    """Get the current CUDA architecture info."""
    _init_jit_cuda_arch_once()
    return _CUDA_ARCH


@cache_once
def is_arch_support_pdl() -> bool:
    if is_hip_runtime() or is_musa_runtime():
        return False
    return get_jit_cuda_arch().major >= 9
