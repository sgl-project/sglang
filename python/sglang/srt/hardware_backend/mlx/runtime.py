"""Runtime gate for the opt-in MLX backend on Apple silicon."""

from functools import lru_cache

import torch
from packaging.version import InvalidVersion, Version

from sglang.srt.environ import envs

_MIN_MLX_VERSION = Version("0.32.0")
_MIN_TORCH_VERSION = Version("2.13.0")


def _version_at_least(raw_version: object, minimum: Version) -> bool:
    try:
        return Version(str(raw_version)) >= minimum
    except InvalidVersion:
        return False


@lru_cache(maxsize=1)
def _validate_runtime() -> None:
    try:
        import mlx.core as mx
    except ImportError:
        raise RuntimeError(
            "SGLANG_USE_MLX requires MLX >= 0.32.0, but MLX is not installed"
        ) from None
    if not _version_at_least(mx.__version__, _MIN_MLX_VERSION):
        raise RuntimeError(
            f"SGLANG_USE_MLX requires MLX >= 0.32.0; found MLX {mx.__version__}"
        )
    if not _version_at_least(torch.__version__, _MIN_TORCH_VERSION):
        raise RuntimeError(
            f"SGLANG_USE_MLX requires Torch >= 2.13.0; found Torch {torch.__version__}"
        )
    if not torch.backends.mps.is_available():
        raise RuntimeError("SGLANG_USE_MLX requires an available PyTorch MPS device")

    metal = getattr(mx, "metal", None)
    is_available = getattr(metal, "is_available", None)
    if not callable(is_available) or not is_available():
        raise RuntimeError("SGLANG_USE_MLX requires an available MLX Metal device")


@lru_cache(maxsize=1)
def use_mlx() -> bool:
    """Return whether the validated MLX backend was explicitly enabled."""
    enabled = bool(envs.SGLANG_USE_MLX.get())
    if enabled:
        _validate_runtime()
    return enabled
