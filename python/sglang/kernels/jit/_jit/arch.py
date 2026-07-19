"""CUDA/ROCm architecture detection and default compile target flags."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass
from typing import List

import torch

from sglang.kernels.jit.common import (
    cache_once,
    is_hip_runtime,
    is_musa_runtime,
)
from sglang.srt.utils.common import get_cuda_version

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


def _cuda_arch_suffix(major: int, minor: int) -> str:
    """Mirror FlashInfer's `_normalize_cuda_arch`: 9.x/10.x+ -> "a"; 12.0 -> "f"
    and 12.x (x>0) -> "a" (SM120/SM121 need separate cubins to avoid
    cudaErrorIllegalInstruction, requires CUDA >= 12.9); below 9.0 -> plain.
    Unlike FlashInfer, pre-12.9 CUDA falls back to plain instead of raising.
    """
    if major == 9:
        return "a"
    if major == 12:
        if get_cuda_version() < (12, 9):
            return ""
        return "f" if minor == 0 else "a"
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


def get_default_target_flags() -> List[str]:
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
        return [
            get_jit_cuda_arch().jit_flag,
            "-std=c++20",
            "-O3",
            "--expt-relaxed-constexpr",
        ]


@contextmanager
def override_jit_cuda_arch(major: int, minor: int, suffix: str = ""):
    """A context manager to temporarily override CUDA architecture."""
    global _CUDA_ARCH
    old_value = get_jit_cuda_arch()
    _CUDA_ARCH = ArchInfo(major, minor, suffix)
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
