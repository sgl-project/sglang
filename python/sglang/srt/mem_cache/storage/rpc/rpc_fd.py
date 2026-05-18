# SPDX-License-Identifier: Apache-2.0
"""FD registration helper: magic preamble + SCM_RIGHTS on a Unix-domain stream."""

from __future__ import annotations

import array
import socket
import struct
from typing import Tuple

# 64-bit sentinel; distinct from HTTP/2 client preface beginning with ASCII 'PRI '.
RPC_REG_MAGIC: bytes = struct.pack("<Q", 0x8DF6FBFD)


def parse_unix_socket_uri(uri_or_path: str) -> str:
    """Normalize ``unix:///path`` or bare absolute path → filesystem path."""
    s = uri_or_path.strip()
    if s.startswith("unix://"):
        s = s[7:]
    return s


def send_pool_registration(sock: socket.socket, pool_name: str, fd: int) -> None:
    """Send registration payload plus passed FD (single SCM_RIGHTS)."""
    nm = pool_name.encode("utf-8") if isinstance(pool_name, str) else pool_name
    if len(nm) > 2**31 - 1:
        raise ValueError("pool name too long")
    payload = RPC_REG_MAGIC + struct.pack("<I", len(nm)) + nm
    fd_arr = array.array("i", [int(fd)])
    ancillary = [(socket.SOL_SOCKET, socket.SCM_RIGHTS, fd_arr)]
    sent = sock.sendmsg([payload], ancillary)
    if sent < len(payload):
        raise IOError("truncated registration send")


def recv_pool_registration(conn: socket.socket) -> Tuple[str, int]:
    """Receive magic + ``<u32 pool_len><pool_utf8`` and one FD."""
    bufsize = min(8192 + 8 + 4 + 4096, 65536)
    anc_sz = socket.CMSG_SPACE(struct.calcsize("i"))
    data, ancdata, msg_flags, _addr = conn.recvmsg(bufsize, anc_sz)
    if len(data) < 8 + 4:
        raise ValueError(f"registration too short: len={len(data)}")
    if data[:8] != RPC_REG_MAGIC:
        raise ValueError("registration magic mismatch")
    plen = struct.unpack_from("<I", data, 8)[0]
    if 8 + 4 + plen > len(data):
        raise ValueError("truncated registration name")
    pname = data[12 : 12 + plen].decode("utf-8")
    fds = []
    for cmsg_level, cmsg_type, cmsg_data in ancdata:
        if (
            cmsg_level == socket.SOL_SOCKET
            and cmsg_type == socket.SCM_RIGHTS
        ):
            isz = struct.calcsize("i")
            n_fds = len(cmsg_data) // isz
            fds.extend(struct.unpack("<" + "i" * n_fds, cmsg_data[: n_fds * isz]))
    if len(fds) != 1:
        raise ValueError(f"expected 1 scm fd, got {fds!r}")
    return pname, fds[0]


def recv_pool_registration_reply_pool_id(conn: socket.socket) -> int:
    """Read 4-byte little-endian ``pool_id`` sent by StorageServer after RegisterPool."""
    buf = b""
    while len(buf) < 4:
        chunk = conn.recv(4 - len(buf))
        if not chunk:
            raise IOError("closed before pool_id")
        buf += chunk
    return struct.unpack("<I", buf)[0]


def unix_connect(path: str) -> socket.socket:
    p = parse_unix_socket_uri(path)
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.connect(p)
    return s

