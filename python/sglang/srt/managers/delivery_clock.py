"""Clock identity and timestamp framing for scheduler-to-router delivery metrics."""

import hashlib
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import zmq

from sglang.srt.managers.io_struct import encode_ipc_payload, sock_recv


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


_IPC_TIMESTAMP_FRAME_BYTES = 8
_IPC_CLOCK_DOMAIN_BYTES = 24
_IPC_CLOCK_DOMAIN_VERSION = 1


def sock_send_with_timestamp(
    socket: zmq.Socket,
    obj: Any,
    flags: int = 0,
    *,
    clock_domain: bytes | None = None,
) -> None:
    """Send an IPC payload with a pre-serialization monotonic timestamp.

    The timestamp follows the payload only for the metrics-enabled
    scheduler-to-detokenizer-router path. Encoding completes before the
    atomic multipart send starts.
    """
    if clock_domain is not None and len(clock_domain) != _IPC_CLOCK_DOMAIN_BYTES:
        raise ValueError("Invalid monotonic clock-domain identity length")
    send_time_ns = time.monotonic_ns()
    payload = encode_ipc_payload(obj)
    timestamp_frame = send_time_ns.to_bytes(_IPC_TIMESTAMP_FRAME_BYTES, byteorder="big")
    if clock_domain is not None:
        timestamp_frame += bytes([_IPC_CLOCK_DOMAIN_VERSION]) + clock_domain
    socket.send_multipart([payload, timestamp_frame], flags=flags)


def sock_recv_with_timestamp(
    socket: zmq.Socket,
    flags: int = 0,
    *,
    expected_clock_domain: bytes | None = None,
    require_clock_domain: bool = False,
) -> tuple[int | None, Any]:
    """Receive a timestamped payload, accepting legacy single-frame IPC.

    A legacy sender is decoded normally and returns ``None`` for the send
    timestamp, so the delivery histogram simply omits that observation.
    """
    # Keep payload decoding in the existing IPC codec; metadata is only bytes.
    obj = sock_recv(socket, flags=flags)
    if not socket.getsockopt(zmq.RCVMORE):
        return None, obj

    timestamp_frame = socket.recv(flags=flags)
    if socket.getsockopt(zmq.RCVMORE):
        raise ValueError("Expected at most two frames in a timestamped IPC message")
    if len(timestamp_frame) == _IPC_TIMESTAMP_FRAME_BYTES:
        # Older timestamp frames have no clock identity. A router requiring
        # verification must still route the payload, but cannot observe it.
        if require_clock_domain:
            return None, obj
    elif (
        len(timestamp_frame) == _IPC_TIMESTAMP_FRAME_BYTES + 1 + _IPC_CLOCK_DOMAIN_BYTES
    ):
        version = timestamp_frame[_IPC_TIMESTAMP_FRAME_BYTES]
        domain = timestamp_frame[_IPC_TIMESTAMP_FRAME_BYTES + 1 :]
        if version != _IPC_CLOCK_DOMAIN_VERSION or domain != expected_clock_domain:
            return None, obj
    else:
        raise ValueError(
            f"Invalid timestamp frame length: {len(timestamp_frame)} (expected 8 or 33)"
        )
    return int.from_bytes(
        timestamp_frame[:_IPC_TIMESTAMP_FRAME_BYTES], byteorder="big"
    ), obj
