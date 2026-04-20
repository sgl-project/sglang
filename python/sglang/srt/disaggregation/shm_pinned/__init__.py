"""
Shared-memory pinned disaggregation backend.
"""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "ShmPinnedKVManager",
    "ShmPinnedKVSender",
    "ShmPinnedKVReceiver",
    "ShmPinnedKVBootstrapServer",
]


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module("sglang.srt.disaggregation.shm_pinned.conn")
    return getattr(module, name)


def __dir__():
    return sorted(__all__)
