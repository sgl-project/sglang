"""Runtime gate for the opt-in MLX backend on Apple silicon."""

from functools import lru_cache

import torch
from packaging.version import InvalidVersion, Version

from sglang.srt.environ import envs

_MIN_MLX_VERSION = Version("0.32.0")
_MIN_TRANSFORMERS_MAJOR = 5
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


def _validate_transformers_version() -> None:
    """Reject transformers 4.x, which breaks config imports on Apple Silicon installs.

    A common failure mode on macOS is installing outside ``python[srt_mps]`` /
    ``python[all_mps]`` (or letting a transitive dep like a free ``xgrammar``
    resolve pull ``transformers<5``). That surfaces as
    ``ImportError: cannot import name 'PreTrainedConfig'``.
    """
    try:
        import transformers
    except ImportError as exc:
        raise RuntimeError(
            "SGLANG_USE_MLX requires transformers>=5 (pinned to 5.12.1 via the "
            "srt_mps extra), but transformers is not installed; reinstall with "
            "the srt_mps extra from python/pyproject_other.toml"
        ) from exc

    transformers_version = getattr(transformers, "__version__", None)
    try:
        version = Version(str(transformers_version))
    except InvalidVersion as exc:
        raise RuntimeError(
            "SGLANG_USE_MLX could not parse transformers version "
            f"{transformers_version!r}; reinstall with the srt_mps extra"
        ) from exc

    if version.major < _MIN_TRANSFORMERS_MAJOR:
        raise RuntimeError(
            "SGLANG_USE_MLX requires transformers>=5 "
            f"(found {transformers_version}). On Apple Silicon, install from "
            "python/pyproject_other.toml with the srt_mps (or all_mps) extra; "
            "do not use the default CUDA python/pyproject.toml. "
            "See docs/docs/hardware-platforms/apple_metal.mdx"
        )


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

    _validate_transformers_version()

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
