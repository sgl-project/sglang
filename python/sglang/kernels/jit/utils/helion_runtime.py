"""Runtime helpers for the optional Muse Spark v1.2 Helion kernels.

Backend selection uses the standard fused-op capabilities and priority. The
existing production backends remain preferred; Helion can be selected with
``SGLANG_FORCE_FUSED_OP_BACKEND=helion``. Dynamic token dimensions are
supported without runtime shape gates.
"""

from __future__ import annotations

import contextlib
import contextvars
import logging
import threading
from collections.abc import Iterator, Mapping
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import torch

logger = logging.getLogger(__name__)

_SHORT_NUM_TOKENS = 256
_config_bucket: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "muse_spark_helion_config_bucket", default=None
)
_compiled_lock = threading.Lock()
_compiled: dict[tuple[Any, str], Any] = {}


@lru_cache(maxsize=None)
def warn_helion_unsupported(reason: str) -> bool:
    logger.warning("Muse Spark Helion kernels are disabled: %s", reason)
    return False


@lru_cache(maxsize=None)
def base_package_version(package: str) -> str | None:
    try:
        return version(package).split("+", 1)[0]
    except PackageNotFoundError:
        return None


def helion_supported(tensor: torch.Tensor) -> bool:
    """Return whether a selected kernel is validated for this runtime/device."""
    if torch.compiler.is_compiling():
        return False
    # The unified fused-op override takes precedence over automatic dispatch.
    from sglang.kernels.fused_op import get_fused_op_backend
    from sglang.kernels.spec import KernelBackend

    forced_backend = get_fused_op_backend()
    if forced_backend is not None and forced_backend is not KernelBackend.HELION:
        return False
    helion_version = base_package_version("helion")
    if helion_version != "1.4.0":
        return warn_helion_unsupported(
            f"Helion 1.4.0 is required, found {helion_version or 'not installed'}"
        )
    return tensor.is_cuda


def token_count_specialization_key(first_tensor: torch.Tensor, *_: Any) -> int:
    """Keep empty, unit, and general dynamic token dimensions in distinct bounds."""
    num_tokens = int(first_tensor.shape[0])
    if num_tokens <= 0:
        return 0
    return 1 if num_tokens == 1 else 2


@contextlib.contextmanager
def helion_config_bucket(bucket: str) -> Iterator[None]:
    """Temporarily force a pinned config bucket for tests and benchmarks."""
    if bucket not in ("short", "long"):
        raise ValueError(f"Unsupported Helion config bucket: {bucket}")
    token = _config_bucket.set(bucket)
    try:
        yield
    finally:
        _config_bucket.reset(token)


def current_helion_config_bucket(num_tokens: int) -> str:
    bucket = _config_bucket.get()
    if bucket is not None:
        return bucket
    return "short" if num_tokens <= _SHORT_NUM_TOKENS else "long"


def _pdl_launcher(
    triton_kernel: Any,
    grid: tuple[int, ...],
    *args: Any,
    num_warps: int,
    num_stages: int,
    **kwargs: Any,
) -> Any:
    """Launch a Helion-generated Triton kernel with PDL enabled."""
    from helion.runtime import default_launcher

    return default_launcher(
        triton_kernel,
        grid,
        *args,
        num_warps=num_warps,
        num_stages=num_stages,
        launch_pdl=True,
        **kwargs,
    )


def run_pinned_helion(
    kernel: Any,
    configs: Mapping[str, Any],
    args: tuple[Any, ...],
    *,
    num_tokens: int,
    launch_pdl: bool = False,
) -> Any:
    """Run a dynamic-shape kernel with its token-length tuned config."""
    if num_tokens <= 0:
        raise ValueError("Helion kernels require at least one token")
    bucket = current_helion_config_bucket(num_tokens)
    config = configs.get(bucket)
    kernel_name = getattr(kernel, "__name__", type(kernel).__name__)
    if config is None:
        raise KeyError(f"No pinned {bucket} config for {kernel_name}")

    bound = kernel.bind(args)
    # Kernel.bind() returns Helion's canonical BoundKernel for a specialization.
    # Keep the object itself in the key: using id(bound) could alias a stale
    # compiled callable if a temporary BoundKernel were ever reclaimed.
    # Every kernel's decorator key distinguishes T=1 from the general dynamic
    # token dimension, while both bounds retain the same bucket-wide config.
    cache_key = (bound, bucket)
    compiled = _compiled.get(cache_key)
    if compiled is None:
        with _compiled_lock:
            compiled = _compiled.get(cache_key)
            if compiled is None:
                compiled = bound.compile_config(config)
                _compiled[cache_key] = compiled
                logger.info(
                    "Compiled pinned Helion kernel %s for %s", kernel_name, bucket
                )
    # PDL-enabled kernels bracket one flat tile with wait/trigger. A future
    # persistent config could process multiple tiles after its first trigger,
    # so leave PDL disabled for that config shape rather than exposing partial
    # output to a dependent launch.
    if launch_pdl and getattr(config, "pid_type", None) == "flat":
        return compiled(*args, _launcher=_pdl_launcher)
    return compiled(*args)


__all__ = [
    "base_package_version",
    "current_helion_config_bucket",
    "helion_config_bucket",
    "helion_supported",
    "run_pinned_helion",
    "token_count_specialization_key",
    "warn_helion_unsupported",
]
