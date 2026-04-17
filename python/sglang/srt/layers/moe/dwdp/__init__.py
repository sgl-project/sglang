"""DWDP (Distributed Weight Data Parallelism) for MoE layers.

Tokens stay on-rank; expert weights are prefetched via NVLink.
"""

from typing import Optional

_global_dwdp_manager = None


def get_global_dwdp_manager():
    """Return the global DwdpManager instance, or None if disabled."""
    return _global_dwdp_manager


def set_global_dwdp_manager(manager) -> None:
    """Set or clear the global DwdpManager singleton."""
    global _global_dwdp_manager
    _global_dwdp_manager = manager


def enable_dwdp() -> bool:
    """Return True if DWDP is active."""
    return _global_dwdp_manager is not None
