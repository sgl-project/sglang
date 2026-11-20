"""Optional delivery timestamps for local, metrics-enabled IPC paths."""

import hashlib
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Optional, Tuple

import zmq

from sglang.srt.managers.io_struct import sock_recv, sock_send


def read_clock_domain() -> Optional[bytes]:
    """Identify this process's Linux boot and monotonic time namespace.

    Read once in each sender/receiver process, not in a parent before fork:
    a child can run in a different time namespace. Unknown domains are not
    eligible for delivery observations.
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


def sock_send_with_timestamp(
    socket: zmq.Socket, obj: Any, *, clock_domain: bytes
) -> None:
    """Send the existing payload codec plus an atomic metadata frame.

    The 33-byte metadata matches the scheduler-to-router delivery proposal:
    eight big-endian monotonic nanoseconds, version 1, then a 24-byte domain.
    Take a fresh timestamp for every outgoing message, before serialization.
    """
    if len(clock_domain) != 24:
        raise ValueError("Invalid monotonic clock-domain identity length")
    metadata = time.monotonic_ns().to_bytes(8, "big") + b"\x01" + clock_domain
    sock_send(socket, obj, flags=zmq.SNDMORE)
    socket.send(metadata)


def sock_recv_with_timestamp(
    socket: zmq.Socket, *, expected_clock_domain: Optional[bytes]
) -> Tuple[Optional[int], Any]:
    """Decode a payload and return only a clock-verified send timestamp.

    Legacy single-frame messages and unverified metadata remain routable.
    Drain all metadata frames even when invalid, so the next receive starts
    at a payload rather than interpreting metadata as another message.
    """
    obj = sock_recv(socket)
    frames = []
    while socket.getsockopt(zmq.RCVMORE):
        frames.append(socket.recv())
    if expected_clock_domain is None or len(frames) != 1:
        return None, obj
    metadata = frames[0]
    if len(metadata) != 33 or metadata[8] != 1 or metadata[9:] != expected_clock_domain:
        return None, obj
    return int.from_bytes(metadata[:8], "big"), obj
