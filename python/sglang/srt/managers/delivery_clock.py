"""Identify the Linux monotonic clock domain of the current process."""

import hashlib
import os
import sys
import uuid
from pathlib import Path


def read_clock_domain() -> bytes | None:
    """Return an opaque boot/time-namespace identity, or None if unavailable.

    Read in the actual sender or receiver process: caching in a parent before a
    fork would incorrectly identify a child in a different time namespace.
    """
    if sys.platform != "linux":
        return None
    try:
        boot_id = uuid.UUID(
            Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        ).bytes
        namespace = os.readlink("/proc/self/ns/time")
        if not namespace.startswith("time:[") or not namespace.endswith("]"):
            return None
        namespace_inode = int(namespace[6:-1]).to_bytes(8, "big")
        return hashlib.sha256(boot_id + namespace_inode).digest()[:24]
    except (OSError, ValueError, OverflowError):
        return None
