"""KV cache dtypes that out-of-tree platform plugins register."""

from typing import Optional

import torch

from sglang.srt.arg_groups.choices import (
    BUILT_IN_KV_CACHE_DTYPES,
    add_kv_cache_dtype_choices,
)

_REGISTERED_KV_CACHE_DTYPES: dict[str, torch.dtype] = {}


def register_kv_cache_dtype(
    name: str, *, storage_dtype: torch.dtype = torch.uint8
) -> None:
    """Register a ``--kv-cache-dtype`` value that only a platform KV pool stores.

    Adds ``name`` to the CLI choices and records ``storage_dtype``, the dtype
    the platform's KV pool is constructed with. Call it from the platform
    package's general plugin entry point, which runs in every process before
    arguments are parsed. Registering a name again with the same storage dtype
    is a no-op, and a platform may register several names for one format.
    """
    if name in BUILT_IN_KV_CACHE_DTYPES:
        raise ValueError(f"KV cache dtype {name!r} is already built in")
    registered = _REGISTERED_KV_CACHE_DTYPES.get(name)
    if registered is not None:
        if registered != storage_dtype:
            raise ValueError(
                f"KV cache dtype {name!r} is already registered with storage "
                f"dtype {registered}, not {storage_dtype}"
            )
        return
    _REGISTERED_KV_CACHE_DTYPES[name] = storage_dtype
    add_kv_cache_dtype_choices([name])


def registered_kv_cache_storage_dtype(name: str) -> Optional[torch.dtype]:
    """Storage dtype of a platform-registered KV cache dtype; None for any other."""
    return _REGISTERED_KV_CACHE_DTYPES.get(name)
