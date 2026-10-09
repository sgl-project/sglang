"""Pass file descriptors between local ranks over a Unix domain socket.

``SCM_RIGHTS`` (``socket.send_fds``) installs a copy of the descriptor in the
receiving process, so unlike the ``/proc/<pid>/fd`` reopen used for memfds the
received fd stays valid after the sender closes its own; this is the only path
that works for fds from ``cuMemExportToShareableHandle``.
"""

from __future__ import annotations

import os
import socket
import tempfile
import uuid

_HANDSHAKE_TIMEOUT_S = 300.0


def bind_fd_server(path: str, num_peers: int) -> socket.socket:
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(path)
    server.listen(num_peers)
    server.settimeout(_HANDSHAKE_TIMEOUT_S)
    return server


def serve_fd(server: socket.socket, path: str, fd: int, num_peers: int) -> None:
    """Send ``fd`` to ``num_peers`` connections, then tear the socket down."""
    try:
        for _ in range(num_peers):
            conn, _ = server.accept()
            with conn:
                socket.send_fds(conn, [b"f"], [fd])
    finally:
        server.close()
        os.unlink(path)


def fetch_fd(path: str) -> int:
    """Receive one fd from the server at ``path``; the caller owns closing it."""
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.settimeout(_HANDSHAKE_TIMEOUT_S)
        sock.connect(path)
        msg, fds, _flags, _addr = socket.recv_fds(sock, 1, 1)
    if not msg or len(fds) != 1:
        raise RuntimeError(f"fd exchange at {path}: peer sent {len(fds)} fds")
    return fds[0]


def exchange_fd(group, fd: int | None, *, name: str, src: int = 0) -> int:
    """Give every rank of ``group`` a copy of rank 0's ``fd``.

    All ranks must be on one host. Rank 0 passes its ``fd`` (returned
    unchanged); the other ranks pass None and get their own copy, which they
    own and must close after use. Rank 0 may close its fd as soon as this
    returns: the copies are independent descriptors.
    """
    if group.rank_in_group == src:
        assert fd is not None
        short = f"sglang_fdx_{name}_{os.getpid()}_{uuid.uuid4().hex[:8]}.sock"
        path = os.path.join(tempfile.gettempdir(), short)
        # Bind before the peers learn the path, so no connect can race it.
        server = bind_fd_server(path, num_peers=group.world_size - 1)
        group.broadcast_object(path, src=src)
        serve_fd(server, path, fd, num_peers=group.world_size - 1)
        return fd
    assert fd is None
    path = group.broadcast_object(None, src=src)
    return fetch_fd(path)
