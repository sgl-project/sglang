from __future__ import annotations

import importlib
import logging
import os
import sys

logger = logging.getLogger(__name__)

_PRELOADED = False


def preload_mooncake_engine_global() -> None:
    """Load Mooncake's shared engine before importing mooncake.ep.

    FIXME: Drop this workaround after the Mooncake wheel links the
    torch-2.12 EP extension against engine.so/libglog directly.

    Mooncake 0.3.11.post1's torch-2.12 EP extension references symbols provided
    by engine.so, but the wheel does not declare engine.so as a direct dynamic
    dependency for ep_2_12_0.so. Loading engine.so with RTLD_GLOBAL makes those
    symbols available before mooncake.ep imports the versioned EP extension.
    In the same wheel, the older torch-specific EP extensions such as
    ep_2_9_1.so, ep_2_10_0.so, and ep_2_11_0.so do declare engine.so as a
    dependency; ep_2_12_0.so is the outlier.

    This covers the observed failure:
      ImportError: ep_2_12_0...so: undefined symbol:
      _ZN6google10LogMessageC1EPKcii

    Local wheel inspection/repro confirmed direct ``from mooncake import ep``
    fails with the missing glog symbol, while preloading mooncake.engine first
    makes the same import succeed.
    """
    global _PRELOADED
    if _PRELOADED:
        return

    old_flags = sys.getdlopenflags()
    try:
        sys.setdlopenflags(old_flags | os.RTLD_GLOBAL)
        importlib.import_module("mooncake.engine")
        _PRELOADED = True
    except Exception:
        logger.debug("Failed to preload mooncake.engine", exc_info=True)
    finally:
        sys.setdlopenflags(old_flags)
