# Adapted from:
# https://github.com/outlines-dev/outlines/blob/6c6966cfa24e9c120494ebb317c6126aa2ae94af/outlines/caching.py
import asyncio
import hashlib
import os
from typing import Callable, Optional

import cloudpickle
from diskcache import Cache

home_dir = os.path.expanduser("~")
cache_dir = os.environ.get("SGLANG_CACHE_DIR", f"{home_dir}/.cache/sglang")
memory = Cache(cache_dir, eviction_policy="none", cull_limit=0)
_caching_enabled = True


def hash_arguments(*args, **kwargs) -> str:
    """Create a hash out of the args and kwargs provided"""
    result = hashlib.md5()
    for item in list(args) + sorted(kwargs.items()):
        result.update(cloudpickle.dumps(item))
    return result.hexdigest()


def disk_cache(key_function: Optional[Callable] = None):
    def decorator(cached_function: Callable):
        def wrapper(*args, **kwargs):
            if not _caching_enabled:
                return cached_function(*args, **kwargs)
            if key_function:
                key_args = key_function(*args, **kwargs)
                cache_key = hash_arguments(*key_args)
            else:
                cache_key = hash_arguments(*args, **kwargs)
            if cache_key in memory:
                return memory[cache_key]
            result = cached_function(*args, **kwargs)
            memory[cache_key] = result
            return result

        async def async_wrapper(*args, **kwargs):
            if not _caching_enabled:
                return await cached_function(*args, **kwargs)
            if key_function:
                key_args = key_function(*args, **kwargs)
                cache_key = hash_arguments(*key_args)
            else:
                cache_key = hash_arguments(*args, **kwargs)
            if cache_key in memory:
                return memory[cache_key]
            result = await cached_function(*args, **kwargs)
            memory[cache_key] = result
            return result

        if asyncio.iscoroutinefunction(cached_function):
            return async_wrapper
        else:
            return wrapper

    return decorator


def disable_cache():
    global _caching_enabled
    _caching_enabled = False


def clear_cache():
    global memory
    memory.clear()
