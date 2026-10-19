"""SM80 capability floor shared by the diffusion CUDA JIT builders."""

from __future__ import annotations

from sglang.kernels.jit.utils import get_jit_cuda_arch
from sglang.kernels.jit.utils.common import is_hip_runtime, is_musa_runtime

_MIN_SM_MAJOR = 8


def require_sm80() -> None:
    """Refuse pre-SM80 CUDA targets before any nvcc build is attempted.

    No diffusion JIT kernel supports pre-SM80 GPUs, so a build attempt there
    only burns tens of seconds of nvcc time per module before failing (measured
    ~49 s on the first FLUX denoise step on sm75). Raising here reaches the
    same per-op build-failure fallbacks instantly. HIP/MUSA capability numbers
    are not SM versions and pass through.
    """
    if is_hip_runtime() or is_musa_runtime():
        return
    arch = get_jit_cuda_arch()
    if arch.major < _MIN_SM_MAJOR:
        raise RuntimeError(
            "diffusion CUDA JIT kernels need SM80 or later; "
            f"detected sm_{arch.major}{arch.minor}"
        )
