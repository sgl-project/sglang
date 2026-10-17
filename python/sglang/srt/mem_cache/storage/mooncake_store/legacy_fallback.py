"""Opt-in migration of an operator-verified, immutable legacy namespace."""

import ctypes
import logging

logger = logging.getLogger(__name__)


def validate_legacy_fallback(options, storage_config, host_pool):
    """Validate declarations, not the contents of an unversioned legacy store."""
    if options is None:
        return False
    if not isinstance(options, dict):
        raise ValueError("legacy_cache_fallback must be an object")
    expected = {
        "kv_cache_dtype": getattr(storage_config, "kv_cache_dtype", None),
        "layout": getattr(host_pool, "layout", None),
        "page_size": getattr(host_pool, "page_size", None),
    }
    if any(value is None for value in expected.values()):
        raise ValueError("legacy_cache_fallback requires a physical KV storage format")
    if set(options) != {*expected, "verified_compatible"}:
        raise ValueError(
            "legacy_cache_fallback requires exactly kv_cache_dtype, layout, "
            "page_size, and verified_compatible"
        )
    if options["verified_compatible"] is not True:
        raise ValueError("legacy_cache_fallback requires verified_compatible=true")
    for name, value in expected.items():
        if type(options[name]) is not type(value) or options[name] != value:
            raise ValueError(f"legacy_cache_fallback {name} must match {value!r}")
    return True


class LegacyCacheFallback:
    """Read legacy bytes without registering inference buffers on another engine.

    The legacy client is read-only. Only the primary client receives promotions.
    The extra CPU copy is deliberately limited to this temporary migration mode.
    """

    def __init__(self, primary, legacy):
        self.primary = primary
        self.legacy = legacy

    def exists(self, keys, results):
        missing = [i for i, result in enumerate(results) if result == 0]
        if not missing:
            return results
        try:
            old_results = self.legacy.batch_is_exist([keys[i] for i in missing])
            for i, result in zip(missing, old_results):
                if result == 1:
                    results[i] = 1
        except Exception:
            logger.warning("Legacy cache existence lookup failed", exc_info=True)
        return results

    def read(self, keys, pointers, sizes, results):
        failed = [i for i, result in enumerate(results) if result < 0]
        if not failed:
            return results
        # A transport error or an incompatible existing object is not a miss.
        try:
            present = self.primary.batch_is_exist([keys[i] for i in failed])
        except Exception:
            logger.warning("Cannot confirm primary cache misses", exc_info=True)
            return results
        for i, status in zip(failed, present):
            if status != 0:
                continue
            try:
                value = self.legacy.get(keys[i])
                lengths = (
                    sizes[i] if isinstance(sizes[i], (list, tuple)) else [sizes[i]]
                )
                destinations = (
                    pointers[i]
                    if isinstance(pointers[i], (list, tuple))
                    else [pointers[i]]
                )
                if (
                    len(destinations) != len(lengths)
                    or not isinstance(value, bytes)
                    or len(value) != sum(lengths)
                ):
                    continue
                offset = 0
                for pointer, length in zip(destinations, lengths):
                    ctypes.memmove(pointer, value[offset : offset + length], length)
                    offset += length
                results[i] = len(value)
            except Exception:
                logger.warning("Legacy cache read failed", exc_info=True)
                continue
            # Promotion failure must not invalidate a successfully restored page.
            try:
                if self.primary.put(keys[i], value) != 0:
                    logger.warning("Legacy cache promotion failed")
            except Exception:
                logger.warning("Legacy cache promotion failed", exc_info=True)
        return results
