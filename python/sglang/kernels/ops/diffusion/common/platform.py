"""Platform predicates and the import-time fallback selector.

Several diffusion Triton kernels have no Triton on the live device (Ascend
NPU, Apple MPS, MUSA, CPU) and must resolve to a pure-``torch`` — or
MLX-accelerated — implementation.  That choice is made once at import time,
which used to mean a hand-rolled four-branch ``if`` block repeated in every
such module, each importing ``current_platform`` directly.

This module owns both halves of that:

- :func:`platform_key` — the one place the diffusion kernels ask what device
  they are on;
- :func:`select_impl` — the one place the Triton-vs-fallback choice is made.

Layering note: the authority on "what platform is this" is the
``multimodal_gen`` platform plugin registry, because it is the only one that
consults out-of-tree vendor plugins (NPU/MUSA).  ``kernels.spec.PlatformInfo``
cannot replace it until it grows MPS/MUSA members and plugin support (see the
``DeviceType`` TODO in ``kernels/spec.py``).  Until then the dependency is
deliberately confined to this single file and resolved lazily, so no other
kernel module imports upward.
"""

from __future__ import annotations

from typing import Callable, TypeVar

F = TypeVar("F", bound=Callable)

_CUDA_LIKE = frozenset({"cuda", "hip"})


def platform_key() -> str:
    """Return the live device family: ``cuda``/``hip``/``npu``/``mps``/``musa``/``cpu``."""
    from sglang.multimodal_gen.runtime.platforms import current_platform

    for name in ("cuda", "hip", "npu", "mps", "musa"):
        if getattr(current_platform, f"is_{name}")():
            return name
    return "cpu"


def is_cuda() -> bool:
    return platform_key() == "cuda"


def is_hip() -> bool:
    return platform_key() == "hip"


def has_triton() -> bool:
    """True when the live device runs the Triton implementations."""
    return platform_key() in _CUDA_LIKE


def lazy_fallback(kind: str, name: str) -> Callable:
    """Name a fallback without importing its module.

    ``select_impl`` is handed every candidate at once, so a plain import here
    would pull in *all* fallback modules on every platform -- including MLX on
    CUDA hosts.  The returned shim imports ``common.fallback_<kind>`` on its
    first call instead, which for the unselected candidates never happens.
    """

    def _call(*args, **kwargs):
        from importlib import import_module

        impl = getattr(
            import_module(f"sglang.kernels.ops.diffusion.common.fallback_{kind}"), name
        )
        return impl(*args, **kwargs)

    _call.__name__ = name
    _call.__qualname__ = f"{kind}_fallback.{name}"
    return _call


def select_impl(triton_impl: F, **fallbacks: F) -> F:
    """Pick ``triton_impl`` on CUDA/HIP, else the fallback for this platform.

    Callers pass the fallbacks they actually have, keyed by platform
    (``npu=``, ``mps=``, ``musa=``, ``cpu=``); an unlisted platform keeps the
    Triton implementation, which is what the pre-existing per-module ``if``
    chains did.  Keeping the whole decision in one call means a module's
    exported name is bound exactly once, so the fallback wiring stays greppable
    and can later be replaced wholesale by a ``BaseFusedOp`` dispatch without
    touching call sites.
    """
    return fallbacks.get(platform_key(), triton_impl)
