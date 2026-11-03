"""VMM and fd-passing helpers shared by the VmmBackend implementations and their device-neutral callers; nothing here imports a driver binding."""

import os
import socket
import struct
import tempfile
import threading
from array import array
from typing import Dict, List, Optional, Protocol, Tuple

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

# (src_rank, base_idx, fd_count) sent in front of every SCM_RIGHTS fd
_FD_HEADER = struct.Struct("<QQQ")
FD_SEND_TIMEOUT_S = 120.0


class Reservation(Protocol):
    """One VA range plus its mappings, as the device-neutral callers use it;
    each VmmBackend's driver module implements it. map returns a handle
    that outlives the reservation only with retain_handle=True.
    """

    base: int
    size: int

    def map(self, offset: int, size: int, *, retain_handle: bool) -> int: ...

    def map_existing(self, offset: int, size: int, handle: int) -> None: ...

    def unmap_existing(self, offset: int, size: int) -> None: ...

    def close(self, *, release_handles: bool = True) -> None: ...


def align_up(value: int, alignment: int) -> int:
    """Round value up to a positive byte alignment."""
    return (int(value) + alignment - 1) // alignment * alignment


def align_down(value: int, alignment: int) -> int:
    """Round value down to a positive byte alignment."""
    return int(value) // alignment * alignment


def all_ranks_ok(group: ProcessGroup, ok: bool) -> bool:
    """True iff ok holds on every rank in group (BAND all-reduce)."""
    flag = torch.tensor([1 if ok else 0], dtype=torch.int32, device="cpu")
    dist.all_reduce(flag, op=dist.ReduceOp.BAND, group=group)
    return flag.item() == 1


def send_fd(sock: socket.socket, fd: int, src_rank: int, base_idx: int) -> None:
    """Send one fd as an SCM_RIGHTS message behind a (src_rank, base_idx) header."""
    fds = array("i", [int(fd)])
    header = _FD_HEADER.pack(int(src_rank), int(base_idx), 1)
    sent = sock.sendmsg(
        [header],
        [(socket.SOL_SOCKET, socket.SCM_RIGHTS, fds.tobytes())],
    )
    if sent != len(header):
        raise RuntimeError(f"sendmsg sent {sent} bytes, expected {len(header)}")


def recv_fd(sock: socket.socket) -> Optional[Tuple[int, int, int]]:
    """Receive one send_fd message as (src_rank, base_idx, fd); None at end of stream."""
    fd_item_size = array("i").itemsize
    data, ancdata, _, _ = sock.recvmsg(_FD_HEADER.size, socket.CMSG_SPACE(fd_item_size))
    if not data:
        return None
    if len(data) != _FD_HEADER.size:
        raise RuntimeError(
            f"received truncated fd header: {len(data)} < {_FD_HEADER.size}"
        )
    src_rank, base_idx, fd_count = _FD_HEADER.unpack(data)
    fds = array("i")
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


def close_fds_quietly(fds: List[int]) -> None:
    """Close every fd on a cleanup path, where a close error must not replace the error being raised."""
    for fd in fds:
        try:
            os.close(fd)
        except OSError:
            pass


def _check_received_keys(
    received_fds: Dict[Tuple[int, int], int], rank: int, peer_base_counts: List[int]
) -> None:
    """Raise unless exactly one fd arrived per (peer_rank, base_idx) the peers announced."""
    expected = {
        (src_rank, base_idx)
        for src_rank, count in enumerate(peer_base_counts)
        if src_rank != rank
        for base_idx in range(count)
    }
    missing = expected.difference(received_fds)
    extra = set(received_fds).difference(expected)
    if missing or extra:
        raise RuntimeError(
            "POSIX fd exchange mismatch: "
            f"missing={sorted(missing)[:8]}, extra={sorted(extra)[:8]}"
        )


def exchange_posix_fds(
    group: ProcessGroup,
    rank: int,
    world_size: int,
    local_fds: List[int],
    peer_base_counts: List[int],
) -> Dict[Tuple[int, int], int]:
    """Exchange POSIX file descriptors across ranks via SCM_RIGHTS over a UNIX
    socket. Returns {(src_rank, base_idx): fd} for every peer. The caller
    owns the received fds and must close them.

    local_fds are only sent, never consumed. Whether they may be closed
    afterwards is the driver's rule rather than this function's: CUDA hands out a
    fresh fd, other drivers may keep theirs. See VmmBackend.owns_exported_fds.
    """
    sock_kind = socket.SOCK_SEQPACKET
    sock_dir = tempfile.mkdtemp(prefix="sgl_ar_fd_")
    sock_path = os.path.join(sock_dir, f"rank_{rank}.sock")
    try:
        server = socket.socket(socket.AF_UNIX, sock_kind)
    except BaseException:
        os.rmdir(sock_dir)
        raise
    server.settimeout(FD_SEND_TIMEOUT_S)
    received_fds: Dict[Tuple[int, int], int] = {}
    errors: List[BaseException] = []
    # a timed-out exchange abandons the recv thread while it can still receive,
    # so it closes late fds itself rather than storing them for nobody
    received_lock = threading.Lock()
    abandoned = threading.Event()

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
                        with received_lock:
                            if abandoned.is_set():
                                os.close(fd)
                                return
                            if key in received_fds:
                                os.close(fd)
                                raise RuntimeError(f"duplicate fd for {key}")
                            received_fds[key] = fd
        except BaseException as error:
            errors.append(error)

    try:
        server.bind(sock_path)
        server.listen(world_size)
        paths: List[Optional[str]] = [None] * world_size
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

        _check_received_keys(received_fds, rank, peer_base_counts)
        return received_fds
    except BaseException:
        # on any failure the caller never sees these fds, so they are closed here;
        # a close error must not replace the failure being raised
        with received_lock:
            abandoned.set()
            leftover = list(received_fds.values())
            received_fds.clear()
        close_fds_quietly(leftover)
        raise
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


def visible_physical_device_ids(env_name: str, device_count: int) -> List[str]:
    """Physical ids of the devices this process sees, by local ordinal, from a device-visibility variable such as CUDA_VISIBLE_DEVICES."""
    value = os.environ.get(env_name)
    if not value:
        return [str(ordinal) for ordinal in range(device_count)]
    return [entry.strip() for entry in value.split(",")][:device_count]


def exchange_exported_handles(
    fabric_handles: List[bytes],
    local_fds: List[int],
    use_fabric: bool,
    group: ProcessGroup,
    rank: int,
    world_size: int,
    *,
    close_local_fds: bool,
) -> Tuple[bool, Dict[Tuple[int, int], bytes | int]]:
    """Collect every peer's exported handles as (use_fabric, {(peer_rank, idx): fabric handle or received fd}).

    POSIX fds only reach ranks on this host, so a POSIX group spanning hosts raises; the caller closes the received fds, and this rank's own fds are closed here when close_local_fds is set.
    """
    if use_fabric:
        all_fabric = [None] * world_size
        dist.all_gather_object(all_fabric, fabric_handles, group=group)
        return True, {
            (peer_rank, index): handle
            for peer_rank, handles in enumerate(all_fabric)
            if peer_rank != rank
            for index, handle in enumerate(handles)
        }
    try:
        host_and_counts = [None] * world_size
        dist.all_gather_object(
            host_and_counts, (socket.gethostname(), len(local_fds)), group=group
        )
        if len({host for host, _ in host_and_counts}) > 1:
            raise RuntimeError(
                "VMM handles cannot reach ranks on other hosts without FABRIC "
                "handles (one NVLink domain with IMEX)"
            )
        return False, exchange_posix_fds(
            group, rank, world_size, local_fds, [count for _, count in host_and_counts]
        )
    finally:
        if close_local_fds:
            close_fds_quietly(local_fds)
