"""Deterministic path routing for NIXL FILE-backed HiCache storage."""

import hashlib

_BUCKET_HEX_CHARS = 2
_BUCKET_MASK = (1 << (4 * _BUCKET_HEX_CHARS)) - 1


def stable_key_hash(key: str) -> int:
    """Return a process-stable 64-bit hash for a NIXL storage key."""
    return int.from_bytes(
        hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest(), "big"
    )


def route_key(key: str, num_disks: int) -> tuple[int, str]:
    """Return the storage disk index and bucket directory for a storage key."""
    if num_disks <= 0:
        raise ValueError("num_disks must be positive")
    key_hash = stable_key_hash(key)
    return (
        (key_hash >> 16) % num_disks,
        f"{key_hash & _BUCKET_MASK:0{_BUCKET_HEX_CHARS}x}",
    )


def route_disk(key: str, num_disks: int) -> int:
    """Return the storage disk index for a storage key."""
    return route_key(key, num_disks)[0]
