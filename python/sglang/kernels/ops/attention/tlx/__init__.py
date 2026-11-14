"""Attention kernels written in TLX.

TLX kernels live in a ``tlx/`` subpackage of their logical group. They import
``triton.language.extra.tlx``, which only a TLX-capable Triton provides
(fbtriton, or Triton built with ``TRITON_EXT_ENABLED`` plus the uTLX plugin),
so this package imports none of them; callers probe availability first.
"""

from __future__ import annotations

import functools


@functools.cache
def is_available() -> bool:
    """Whether this Triton provides ``triton.language.extra.tlx``."""
    try:
        from triton.language.extra import tlx  # noqa: F401
    except ImportError:
        return False
    return True
