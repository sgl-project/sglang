# SPDX-License-Identifier: Apache-2.0
import threading
from collections import OrderedDict
from typing import TypeVar

V = TypeVar("V")

# Postprocess model caches are module-level OrderedDicts shared across threads.
_CACHE_LOCK = threading.Lock()


def cache_get(cache: "OrderedDict[str, V]", key: str) -> V | None:
    """Return the cached value and mark it most recently used."""
    with _CACHE_LOCK:
        if key not in cache:
            return None
        cache.move_to_end(key)
        return cache[key]


def cache_put(cache: "OrderedDict[str, V]", key: str, value: V, *, max_size: int):
    """Insert, evicting least recently used entries beyond max_size."""
    with _CACHE_LOCK:
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > max_size:
            cache.popitem(last=False)
