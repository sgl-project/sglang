"""A stand-in for the ``mori`` RDMA package, so its KV connector imports on runners without it."""

import importlib
import sys
from enum import IntEnum
from types import ModuleType
from unittest.mock import patch


class _StatusCode(IntEnum):
    SUCCESS = 0
    IN_PROGRESS = 1
    FAILED = 2


def mori_stub():
    """A ``sys.modules`` patch standing in for ``mori``, or None when the real
    package imports."""
    try:
        importlib.import_module("mori.cpp")
        importlib.import_module("mori.io")
        return None
    except (ImportError, OSError):
        pass
    package = ModuleType("mori")
    package.__path__ = []
    cpp = ModuleType("mori.cpp")
    cpp.TransferStatus = type("TransferStatus", (), {})
    io = ModuleType("mori.io")
    for name in (
        "BackendType",
        "EngineDesc",
        "IOEngine",
        "IOEngineConfig",
        "MemoryDesc",
        "MemoryLocationType",
        "PollCqMode",
        "RdmaBackendConfig",
    ):
        setattr(io, name, type(name, (), {}))
    io.StatusCode = _StatusCode
    package.cpp, package.io = cpp, io
    return patch.dict(sys.modules, {"mori": package, "mori.cpp": cpp, "mori.io": io})
