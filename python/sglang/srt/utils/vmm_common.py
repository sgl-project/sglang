"""Device-neutral parts of the VMM backends: byte alignment, cross-rank agreement,
and POSIX fd exchange. Shared by cuda_vmm_utils, xpu_vmm_utils and the vmm_backend
facade, so nothing here may import a driver binding."""

from __future__ import annotations

import array
import os
import socket
import struct
import tempfile
import threading
from typing import Dict, List, Optional, Protocol, Tuple

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

# must stay in step with the "<QQQ" header send_fd packs
_FD_HEADER_BYTES = 24
FD_SEND_TIMEOUT_S = 120.0


class Reservation(Protocol):
    """One VA range plus its mappings, as the device-neutral callers use it;
    cuda_vmm_utils and xpu_vmm_utils each implement it. ``map`` returns a handle
    that outlives the reservation only with ``retain_handle=True``.
    """

    base: int
    size: int

    def map(self, offset: int, size: int, *, retain_handle: bool) -> int: ...

    def map_existing(self, offset: int, size: int, handle: int) -> None: ...

    def unmap_existing(self, offset: int, size: int) -> None: ...

    def close(self, *, release_handles: bool = True) -> None: ...


def align_up(value: int, alignment: int) -> int:
    """Round ``value`` up to a positive byte ``alignment``."""
    return (int(value) + alignment - 1) // alignment * alignment


def align_down(value: int, alignment: int) -> int:
    """Round ``value`` down to a positive byte ``alignment``."""
    return int(value) // alignment * alignment


def all_ranks_ok(group: ProcessGroup, ok: bool) -> bool:
    """True iff ``ok`` holds on every rank in ``group`` (BAND all-reduce)."""
    flag = torch.tensor([1 if ok else 0], dtype=torch.int32, device="cpu")
    dist.all_reduce(flag, op=dist.ReduceOp.BAND, group=group)
    return flag.item() == 1


def send_fd(sock: socket.socket, fd: int, src_rank: int, base_idx: int) -> None:
    fds = array.array("i", [int(fd)])
    header = struct.pack("<QQQ", int(src_rank), int(base_idx), 1)
    sent = sock.sendmsg(
        [header],
        [(socket.SOL_SOCKET, socket.SCM_RIGHTS, fds.tobytes())],
    )
    if sent != len(header):
        raise RuntimeError(f"sendmsg sent {sent} bytes, expected {len(header)}")


def recv_fd(sock: socket.socket) -> Optional[Tuple[int, int, int]]:
    fd_item_size = array.array("i").itemsize
    data, ancdata, _, _ = sock.recvmsg(
        _FD_HEADER_BYTES, socket.CMSG_SPACE(fd_item_size)
    )
    if not data:
        return None
    if len(data) != _FD_HEADER_BYTES:
        raise RuntimeError(
            f"received truncated fd header: {len(data)} < {_FD_HEADER_BYTES}"
        )
    src_rank, base_idx, fd_count = struct.unpack("<QQQ", data)
    fds = array.array("i")
    for level, cmsg_type, cmsg_data in ancdata:
        if level == socket.SOL_SOCKET and cmsg_type == socket.SCM_RIGHTS:
            fds.frombytes(cmsg_data[: len(cmsg_data) - (len(cmsg_data) % fd_item_size)])
    if fd_count != 1 or len(fds) != 1:
        for fd in fds:
            os.close(fd)
        raise RuntimeError(
            f"expected one fd, got header={fd_count}, ancillary={len(fds)}"
        )
    return int(src_rank), int(base_idx), int(fds[0])


def exchange_posix_fds(
    group: ProcessGroup,
    rank: int,
    world_size: int,
    local_fds: List[int],
    peer_base_counts: List[int],
) -> Dict[Tuple[int, int], int]:
    """Exchange POSIX file descriptors across ranks via SCM_RIGHTS over a UNIX
    socket. Returns ``{(src_rank, base_idx): fd}`` for every peer. The caller
    owns the received fds and must close them.

    ``local_fds`` are only sent, never consumed. Whether they may be closed
    afterwards is the driver's rule rather than this function's: CUDA hands out a
    fresh fd, Level Zero's belongs to the driver. See VmmBackend.owns_exported_fds.
    """
    sock_kind = socket.SOCK_SEQPACKET
    sock_dir = tempfile.mkdtemp(prefix="sgl_ar_fd_")
    sock_path = os.path.join(sock_dir, f"rank_{rank}.sock")
    server = socket.socket(socket.AF_UNIX, sock_kind)
    server.settimeout(FD_SEND_TIMEOUT_S)
    received_fds: Dict[Tuple[int, int], int] = {}
    errors: List[BaseException] = []

    def recv_loop() -> None:
        try:
            for _ in range(world_size - 1):
                conn, _ = server.accept()
                with conn:
                    conn.settimeout(FD_SEND_TIMEOUT_S)
                    while True:
                        packet = recv_fd(conn)
                        if packet is None:
                            break
                        src_rank, base_idx, fd = packet
                        key = (src_rank, base_idx)
                        if key in received_fds:
                            os.close(fd)
                            raise RuntimeError(f"duplicate fd for {key}")
                        received_fds[key] = fd
        except BaseException as error:
            errors.append(error)

    try:
        server.bind(sock_path)
        server.listen(world_size)
        paths = [None] * world_size
        dist.all_gather_object(paths, sock_path, group=group)

        thread = threading.Thread(target=recv_loop, daemon=True)
        thread.start()
        try:
            for peer_rank, peer_path in enumerate(paths):
                if peer_rank == rank:
                    continue
                with socket.socket(socket.AF_UNIX, sock_kind) as sock:
                    sock.settimeout(FD_SEND_TIMEOUT_S)
                    sock.connect(peer_path)
                    for base_idx, fd in enumerate(local_fds):
                        send_fd(sock, fd, rank, base_idx)
        finally:
            thread.join(FD_SEND_TIMEOUT_S)

        if thread.is_alive():
            raise RuntimeError("timed out waiting for POSIX fd exchange")
        if errors:
            raise RuntimeError("POSIX fd exchange receive failed") from errors[0]

        expected = {
            (src_rank, base_idx)
            for src_rank, count in enumerate(peer_base_counts)
            if src_rank != rank
            for base_idx in range(count)
        }
        missing = expected.difference(received_fds)
        extra = set(received_fds).difference(expected)
        if missing or extra:
            for fd in received_fds.values():
                os.close(fd)
            raise RuntimeError(
                "POSIX fd exchange mismatch: "
                f"missing={sorted(missing)[:8]}, extra={sorted(extra)[:8]}"
            )
        return received_fds
    finally:
        server.close()
        try:
            os.unlink(sock_path)
        except FileNotFoundError:
            pass
        try:
            os.rmdir(sock_dir)
        except OSError:
            pass
