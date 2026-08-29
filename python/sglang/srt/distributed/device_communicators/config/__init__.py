"""Dispatch tables for the device communicators.

The package is the public surface: ``custom_all_reduce_v2_dispatch`` holds the
band table and its vocabulary, ``custom_all_reduce_v2_tuning`` the per-arch
crossovers, and the two are wired together here so neither has to reach into
the other.
"""

from functools import cache

from .custom_all_reduce_v2_dispatch import Band, Dispatch, RawTuningSpec, Tuning
from .custom_all_reduce_v2_tuning import get_raw_tunings


@cache
def get_supported_world_sizes() -> set[int]:
    return set(get_raw_tunings())


def get_tuning(world_size: int, can_use_multicast: bool) -> Tuning:
    return get_raw_tunings()[world_size].compile(can_use_multicast)


__all__ = [
    "Band",
    "Dispatch",
    "RawTuningSpec",
    "Tuning",
    "get_supported_world_sizes",
    "get_tuning",
]
