# SPDX-License-Identifier: Apache-2.0
"""Exchange HIP VMM allocation file descriptors between local ranks."""

import array
import fcntl
import hashlib
import os
import socket
import tempfile
import threading
import time
from datetime import timedelta
from typing import List, Tuple

_FD_EXCHANGE_TIMEOUT_S = 120.0


def _remaining_s(deadline: float) -> float:
    return max(0.0, deadline - time.monotonic())


def _send_fd(
    path: str,
    fd: int,
    *,
    deadline: float | None = None,
    cancel_event: threading.Event | None = None,
) -> None:
    deadline = deadline or time.monotonic() + _FD_EXCHANGE_TIMEOUT_S
    cancel_event = cancel_event or threading.Event()
    while True:
        if cancel_event.is_set():
            raise TimeoutError(f"cancelled VMM fd send to {path}")
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
                sock.settimeout(max(0.01, _remaining_s(deadline)))
                sock.connect(path)
                fds = array.array("i", [fd])
                sent = sock.sendmsg(
                    [b"\0"],
                    [(socket.SOL_SOCKET, socket.SCM_RIGHTS, fds.tobytes())],
                )
                if sent != 1:
                    raise RuntimeError(f"sent {sent} fd marker bytes, expected 1")
                return
        except (ConnectionRefusedError, FileNotFoundError):
            if time.monotonic() >= deadline:
                raise TimeoutError(f"timed out connecting to VMM fd socket {path}")
            time.sleep(0.01)


def _recv_fd(
    path: str,
    *,
    deadline: float | None = None,
    cancel_event: threading.Event | None = None,
) -> int:
    deadline = deadline or time.monotonic() + _FD_EXCHANGE_TIMEOUT_S
    cancel_event = cancel_event or threading.Event()
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass

    fd_item_size = array.array("i").itemsize
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
        server.bind(path)
        server.listen(1)
        while True:
            if cancel_event.is_set() or _remaining_s(deadline) <= 0:
                raise TimeoutError(f"timed out receiving VMM fd on {path}")
            server.settimeout(min(0.1, _remaining_s(deadline)))
            try:
                conn, _ = server.accept()
                break
            except socket.timeout:
                continue
        with conn:
            recv_flags = getattr(socket, "MSG_CMSG_CLOEXEC", 0)
            while True:
                if cancel_event.is_set() or _remaining_s(deadline) <= 0:
                    raise TimeoutError(f"timed out receiving VMM fd on {path}")
                conn.settimeout(min(0.1, _remaining_s(deadline)))
                try:
                    data, ancdata, message_flags, _ = conn.recvmsg(
                        1, socket.CMSG_SPACE(fd_item_size), recv_flags
                    )
                    break
                except socket.timeout:
                    continue

    fds = array.array("i")
    for level, message_type, message_data in ancdata:
        if level == socket.SOL_SOCKET and message_type == socket.SCM_RIGHTS:
            usable = len(message_data) - len(message_data) % fd_item_size
            fds.frombytes(message_data[:usable])
    if data != b"\0" or message_flags & socket.MSG_CTRUNC or len(fds) != 1:
        for fd in fds:
            os.close(fd)
        raise RuntimeError(
            "invalid VMM fd message: "
            f"marker={data!r}, truncated={bool(message_flags & socket.MSG_CTRUNC)}, "
            f"fd_count={len(fds)}"
        )
    received_fd = int(fds[0])
    # MSG_CMSG_CLOEXEC is not available on every Python/platform combination.
    # Enforce the same protection explicitly before the descriptor escapes.
    current_flags = fcntl.fcntl(received_fd, fcntl.F_GETFD)
    fcntl.fcntl(received_fd, fcntl.F_SETFD, current_flags | fcntl.FD_CLOEXEC)
    return received_fd


def exchange_vmm_fds(
    rank: int,
    world_size: int,
    key: str,
    local_fd: int,
    local_size: int,
    store,
    ranks_tag: str,
) -> Tuple[List[int], List[int]]:
    """Exchange one VMM allocation fd per rank through Unix sockets.

    The returned lists are rank ordered. The local rank has fd ``-1`` because
    its allocation is already mapped. The caller owns every non-negative fd.
    """
    key_hash = hashlib.blake2s(key.encode(), digest_size=6).hexdigest()
    socket_dir = tempfile.mkdtemp(prefix=f"sgl_qr_vmm_{os.getpid()}_{key_hash}_")
    store_prefix = f"sgl_qr_vmm/{ranks_tag}/{key}"
    deadline = time.monotonic() + _FD_EXCHANGE_TIMEOUT_S
    cancel_event = threading.Event()

    recv_paths = []
    for peer_rank in range(world_size):
        if peer_rank == rank:
            continue
        path = os.path.join(socket_dir, f"r{rank}_from_r{peer_rank}.sock")
        recv_paths.append((peer_rank, path))
        store.set(f"{store_prefix}/socket/r{rank}_from_r{peer_rank}", path.encode())
    store.set(f"{store_prefix}/size/r{rank}", str(local_size).encode())

    received_fds = {}
    receive_errors = {}

    def receive(peer_rank: int, path: str) -> None:
        try:
            received_fds[peer_rank] = _recv_fd(
                path, deadline=deadline, cancel_event=cancel_event
            )
        except BaseException as exc:
            receive_errors[peer_rank] = exc

    threads = []
    try:
        for peer_rank, path in recv_paths:
            thread = threading.Thread(
                target=receive,
                args=(peer_rank, path),
                name=f"sgl-qr-vmm-recv-r{rank}-from-r{peer_rank}",
            )
            thread.start()
            threads.append(thread)

        peer_keys = []
        for peer_rank in range(world_size):
            if peer_rank == rank:
                continue
            peer_keys.extend(
                [
                    f"{store_prefix}/socket/r{peer_rank}_from_r{rank}",
                    f"{store_prefix}/size/r{peer_rank}",
                ]
            )
        if peer_keys:
            store.wait(peer_keys, timedelta(seconds=_remaining_s(deadline)))

        for peer_rank in range(world_size):
            if peer_rank == rank:
                continue
            peer_path = store.get(
                f"{store_prefix}/socket/r{peer_rank}_from_r{rank}"
            ).decode()
            duplicated_fd = os.dup(local_fd)
            try:
                _send_fd(
                    peer_path,
                    duplicated_fd,
                    deadline=deadline,
                    cancel_event=cancel_event,
                )
            finally:
                os.close(duplicated_fd)

        for thread in threads:
            thread.join(_remaining_s(deadline))
        if any(thread.is_alive() for thread in threads):
            raise TimeoutError("timed out receiving QuickReduce VMM fds")
        if receive_errors:
            raise RuntimeError(f"QuickReduce VMM fd exchange failed: {receive_errors}")

        peer_fds = [-1] * world_size
        peer_sizes = [0] * world_size
        peer_sizes[rank] = local_size
        for peer_rank in range(world_size):
            if peer_rank == rank:
                continue
            peer_fds[peer_rank] = received_fds[peer_rank]
            peer_sizes[peer_rank] = int(
                store.get(f"{store_prefix}/size/r{peer_rank}").decode()
            )
        received_fds.clear()
        return peer_fds, peer_sizes
    except BaseException:
        cancel_event.set()
        for thread in threads:
            thread.join(1.0)
        for fd in received_fds.values():
            os.close(fd)
        raise
    finally:
        cancel_event.set()
        for thread in threads:
            if thread.is_alive():
                thread.join(1.0)
        for _, path in recv_paths:
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass
        try:
            os.rmdir(socket_dir)
        except OSError:
            pass
