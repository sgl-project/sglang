"""Shared helpers: caching decorator, CI test gating, and runtime detection."""

from __future__ import annotations

import functools
from typing import Any, Callable, Dict, List, TypeVar

import torch

from sglang.srt.environ import envs
from sglang.utils import is_in_ci

F = TypeVar("F", bound=Callable[..., Any])
T = TypeVar("T")


def should_run_full_tests() -> bool:
    return envs.SGLANG_JIT_KERNEL_RUN_FULL_TESTS.get()


def get_ci_test_range(full_range: List[Any], ci_range: List[Any]) -> List[Any]:
    if should_run_full_tests():
        return full_range
    return ci_range if is_in_ci() else full_range


def cache_once(fn: F) -> F:
    """
    NOTE: `functools.lru_cache` is not compatible with `torch.compile`
    So we manually implement a simple cache_once decorator to replace it.
    """
    result_map = {}

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        key = (args, tuple(sorted(kwargs.items())))
        if key not in result_map:
            result_map[key] = fn(*args, **kwargs)
        return result_map[key]

    return wrapper  # type: ignore


@functools.lru_cache(maxsize=None)
def empty_sentinel(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Cached 0-element tensor for optional-tensor FFI slots (the numel-0
    "not present" convention). Allocating a fresh empty per call costs
    ~1.2us CPU on eager paths; the sentinel is never read, so one cached
    instance per (device, dtype) is safe to share."""
    return torch.empty(0, dtype=dtype, device=device)


@cache_once
def is_hip_runtime() -> bool:
    return bool(torch.version.hip)


@cache_once
def is_musa_runtime() -> bool:
    return hasattr(torch.version, "musa") and torch.version.musa is not None


_REGISTERED_CLASSES: Dict[type, type] = {}


def lazy_register_class(name: str, init_fn: Callable[[], None]) -> Callable[[T], T]:
    """A decorator to lazily register a tvm-ffi object class on first use.

    `init_fn` runs once (typically JIT-compiling and registering the C++
    reflection) right before the class is registered under the FFI type key
    `name`; afterwards instantiation proceeds normally.
    """

    def decorator(cls: T) -> T:
        def __new__(cls, *args, **kwargs):
            import tvm_ffi

            if cls not in _REGISTERED_CLASSES:
                init_fn()  # lazy initialization before registration once
                _REGISTERED_CLASSES[cls] = tvm_ffi.register_object(name)(cls)
            cls = _REGISTERED_CLASSES[cls]
            return original_new(cls, *args, **kwargs)

        original_new = cls.__new__
        cls.__new__ = __new__
        return cls

    return decorator
