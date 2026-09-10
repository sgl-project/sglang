"""Runtime gate for the opt-in MLX backend on Apple silicon."""

from functools import lru_cache

import torch
from packaging.version import InvalidVersion, Version

from sglang.srt.environ import envs

_MIN_MLX_VERSION = Version("0.32.0")
_SUPPORTED_TORCH_SERIES = (2, 13)


def _is_stable_series(raw_version: object, series: tuple[int, int]) -> bool:
    try:
        version = Version(str(raw_version))
    except InvalidVersion:
        return False
    return not version.is_prerelease and (version.major, version.minor) == series


def _is_stable_at_least(raw_version: object, minimum: Version) -> bool:
    try:
        version = Version(str(raw_version))
    except InvalidVersion:
        return False
    return not version.is_prerelease and version >= minimum


@lru_cache(maxsize=1)
def _validate_runtime() -> None:
    try:
        import mlx.core as mx
    except ImportError:
        raise RuntimeError(
            "SGLANG_USE_MLX requires stable Torch 2.13.x and MLX >= 0.32.0, "
            "but MLX is not installed; reinstall with "
            "the srt_mps extra"
        ) from None
    mlx_version = getattr(mx, "__version__", None)
    torch_version = getattr(torch, "__version__", None)
    if not _is_stable_series(
        torch_version, _SUPPORTED_TORCH_SERIES
    ) or not _is_stable_at_least(mlx_version, _MIN_MLX_VERSION):
        raise RuntimeError(
            "SGLANG_USE_MLX requires stable Torch 2.13.x and MLX >= 0.32.0; "
            "found "
            f"Torch {torch_version or 'unknown'} + MLX {mlx_version or 'unknown'}; "
            "reinstall with the srt_mps extra"
        )

    mps_backend = getattr(torch.backends, "mps", None)
    is_mps_available = getattr(mps_backend, "is_available", None)
    if not callable(is_mps_available) or not is_mps_available():
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
