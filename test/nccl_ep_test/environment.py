"""Capability skip shared by the local CUDA Graph fixture."""


class Unavailable(RuntimeError):
    """A missing hardware capability is SKIP (exit 77), never PASS."""
