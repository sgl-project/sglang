import logging
import os
import struct
import time
from typing import Any, List, Optional

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from sglang.srt.utils import log_info_on_rank0

logger = logging.getLogger(__name__)

_drv = None
_FD_HEADER_BYTES = 24
_FD_SEND_TIMEOUT_S = 120.0


def _get_cuda_driver():
    """Lazily import cuda.bindings.driver (cached after first call)."""
    global _drv
    if _drv is None:
        from cuda.bindings import driver

        _drv = driver
    return _drv


def _check_drv(result_tuple, label):
    """Check a cuda.bindings driver call result and return the value."""
    if not isinstance(result_tuple, tuple):
        result_tuple = (result_tuple,)
    err = result_tuple[0]
    drv = _get_cuda_driver()
    if err != drv.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"{label}: {err}")
    return result_tuple[1] if len(result_tuple) > 1 else None


def is_vmm_pointer(ptr: int) -> bool:
    """Check if a device pointer is VMM-backed (cuMemCreate/cuMemMap).

    cuMemRetainAllocationHandle succeeds only on pointers from cuMemCreate;
    it fails on cudaMalloc pointers.
    """
    drv = _get_cuda_driver()
    err, handle = drv.cuMemRetainAllocationHandle(ptr)
    if err == drv.CUresult.CUDA_SUCCESS:
        drv.cuMemRelease(handle)
        return True
    return False


def _send_fd(sock, fd: int, src_rank: int, base_idx: int) -> None:
    import array
    import socket

    fds = array.array("i", [int(fd)])
    header = struct.pack("<QQQ", int(src_rank), int(base_idx), 1)
    sent = sock.sendmsg(
        [header],
        [(socket.SOL_SOCKET, socket.SCM_RIGHTS, fds.tobytes())],
    )
    if sent != len(header):
        raise RuntimeError(f"sendmsg sent {sent} bytes, expected {len(header)}")


def _recv_fd(sock):
    import array
    import socket

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


class VmmGraphInputManager:
    def __init__(
        self,
        obj: Any,
        group: ProcessGroup,
        rank: int,
        world_size: int,
    ) -> None:
        self.obj = obj
        self.group = group
        self.rank = rank
        self.world_size = world_size
        self._peer_mappings = []

    def register_graph_inputs(self):
        """Register graph capture inputs via VMM handle exchange.

        VMM-compatible path for expandable_segments. The C++ side deduplicates
        graph capture pointers into unique base allocations via cuMemGetAddressRange.
        Python exports handles for each unique base, imports + cuMemMaps peer
        allocations, then registers the peer VAs. FABRIC handles are preferred;
        POSIX file descriptors are used when FABRIC is unavailable.
        """
        drv = _get_cuda_driver()
        FABRIC = drv.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC
        POSIX_FD = (
            drv.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
        )
        FABRIC_HANDLE_BYTES = 64
        MAX_VMM_BASES = 4096
        MAX_CHUNKS_PER_INPUT = 16

        t0 = time.perf_counter()

        bases_info, input_chunk_indices, input_offsets = (
            self.obj.get_graph_capture_bases()
        )
        if not bases_info:
            return
        new_count = len(input_chunk_indices)
        num_bases = len(bases_info)
        device_id = torch.cuda.current_device()

        if num_bases > MAX_VMM_BASES:
            raise RuntimeError(
                f"Too many VMM bases to share: {num_bases} > {MAX_VMM_BASES}"
            )

        local_fabric_handles: List[bytes] = []
        local_posix_fds: List[int] = []
        retained_handles = []
        try:
            for base_ptr, _ in bases_info:
                alloc_h = _check_drv(
                    drv.cuMemRetainAllocationHandle(base_ptr),
                    "cuMemRetainAllocationHandle",
                )
                retained_handles.append(alloc_h)

            local_fabric_error: Optional[Exception] = None
            try:
                for alloc_h in retained_handles:
                    fabric_h = _check_drv(
                        drv.cuMemExportToShareableHandle(alloc_h, FABRIC, 0),
                        "cuMemExportToShareableHandle(FABRIC)",
                    )
                    local_fabric_handles.append(bytes(fabric_h.data))
                local_fabric_ok = True
            except Exception as e:
                local_fabric_error = e
                local_fabric_ok = False
                local_fabric_handles = []
                logger.info(
                    "FABRIC handle export failed on rank %s; falling back to "
                    "POSIX fd transport: %s",
                    self.rank,
                    e,
                )

            use_fabric = self._all_ranks_ok(local_fabric_ok)
            if not use_fabric:
                local_posix_error: Optional[Exception] = None
                try:
                    for alloc_h in retained_handles:
                        fd = _check_drv(
                            drv.cuMemExportToShareableHandle(alloc_h, POSIX_FD, 0),
                            "cuMemExportToShareableHandle(POSIX_FD)",
                        )
                        local_posix_fds.append(int(fd))
                    local_posix_ok = True
                except Exception as e:
                    local_posix_error = e
                    local_posix_ok = False
                    for fd in local_posix_fds:
                        try:
                            os.close(fd)
                        except OSError:
                            pass
                    local_posix_fds = []

                if not self._all_ranks_ok(local_posix_ok):
                    local_cause = local_posix_error or local_fabric_error
                    message = (
                        "VMM graph input registration failed: FABRIC export "
                        "failed on at least one rank and POSIX fd export failed "
                        "on at least one rank"
                    )
                    if local_cause is not None:
                        message += f"; local rank {self.rank} error: {local_cause}"
                    raise RuntimeError(message) from local_posix_error

            local_input_chunks = [
                [int(idx) for idx in indices] for indices in input_chunk_indices
            ]
            for chunks in local_input_chunks:
                if len(chunks) > MAX_CHUNKS_PER_INPUT:
                    raise RuntimeError(
                        "Too many VMM chunks for graph input: "
                        f"{len(chunks)} > {MAX_CHUNKS_PER_INPUT}"
                    )

            # All-gather base metadata and per-input VMM spans. A captured tensor
            # can cross expandable-segment allocation boundaries, so peer mappings
            # must preserve each input's contiguous virtual-address span. FABRIC
            # handles are inline metadata; POSIX fds are exchanged separately via
            # SCM_RIGHTS because fd integers are process-local.
            header_struct = struct.Struct("<QQ")
            base_struct = struct.Struct(
                f"<QQ{FABRIC_HANDLE_BYTES}s" if use_fabric else "<QQ"
            )
            input_struct = struct.Struct(f"<QQ{MAX_CHUNKS_PER_INPUT}Q")
            base_offset = header_struct.size
            input_offset = base_offset + MAX_VMM_BASES * base_struct.size
            payload_size = input_offset + new_count * input_struct.size
            local_payload = bytearray(payload_size)

            header_struct.pack_into(local_payload, 0, num_bases, new_count)
            for i, (base_ptr, alloc_size) in enumerate(bases_info):
                if use_fabric:
                    base_struct.pack_into(
                        local_payload,
                        base_offset + i * base_struct.size,
                        int(base_ptr),
                        int(alloc_size),
                        local_fabric_handles[i],
                    )
                else:
                    base_struct.pack_into(
                        local_payload,
                        base_offset + i * base_struct.size,
                        int(base_ptr),
                        int(alloc_size),
                    )
            for i, (chunks, offset) in enumerate(
                zip(local_input_chunks, input_offsets)
            ):
                padded_chunks = chunks + [0] * (MAX_CHUNKS_PER_INPUT - len(chunks))
                input_struct.pack_into(
                    local_payload,
                    input_offset + i * input_struct.size,
                    int(offset),
                    len(chunks),
                    *padded_chunks,
                )

            in_buf = torch.frombuffer(local_payload, dtype=torch.uint8).clone()
            gather_list = [torch.empty_like(in_buf) for _ in range(self.world_size)]
            dist.all_gather(gather_list, in_buf, group=self.group)

            all_base_payload = []
            all_input_chunks = []
            all_input_offsets = []
            for rank, gathered in enumerate(gather_list):
                payload = gathered.numpy().tobytes()
                peer_num_bases, peer_new_count = header_struct.unpack_from(payload, 0)
                if peer_new_count != new_count:
                    raise RuntimeError(
                        "Mismatched graph input count across ranks: "
                        f"rank {rank} has {peer_new_count}, expected {new_count}"
                    )

                peer_bases = []
                for i in range(peer_num_bases):
                    if use_fabric:
                        base_ptr, alloc_size, fabric_handle = base_struct.unpack_from(
                            payload, base_offset + i * base_struct.size
                        )
                    else:
                        base_ptr, alloc_size = base_struct.unpack_from(
                            payload, base_offset + i * base_struct.size
                        )
                        fabric_handle = None
                    peer_bases.append((base_ptr, fabric_handle, alloc_size))

                peer_chunks = []
                peer_offsets = []
                for i in range(new_count):
                    unpacked = input_struct.unpack_from(
                        payload, input_offset + i * input_struct.size
                    )
                    offset, chunk_count, *chunks = unpacked
                    peer_offsets.append(offset)
                    peer_chunks.append(list(chunks[:chunk_count]))

                all_base_payload.append(peer_bases)
                all_input_chunks.append(peer_chunks)
                all_input_offsets.append(peer_offsets)

            posix_peer_fds = {}
            if not use_fabric:
                posix_peer_fds = self._exchange_posix_fds(
                    local_posix_fds,
                    [len(peer_bases) for peer_bases in all_base_payload],
                )

            # Import + map peer allocations. Individual base mappings are kept for
            # single-chunk inputs; span mappings reserve a contiguous VA range and
            # map each chunk at its original relative offset.
            peer_base_va = {}  # (rank, base_idx) -> local VA
            peer_span_va = {}  # (rank, chunk_indices...) -> (local VA, peer base)
            new_mappings = []

            def import_peer_handle(peer_rank: int, base_idx: int, fabric_handle):
                if use_fabric:
                    return _check_drv(
                        drv.cuMemImportFromShareableHandle(fabric_handle, FABRIC),
                        f"cuMemImportFromShareableHandle(rank={peer_rank})",
                    )
                fd = posix_peer_fds[(peer_rank, base_idx)]
                dup_fd = os.dup(fd)
                try:
                    return _check_drv(
                        drv.cuMemImportFromShareableHandle(dup_fd, POSIX_FD),
                        f"cuMemImportFromShareableHandle(rank={peer_rank}, POSIX_FD)",
                    )
                finally:
                    try:
                        os.close(dup_fd)
                    except OSError:
                        pass

            try:
                for peer_rank in range(self.world_size):
                    if peer_rank == self.rank:
                        for idx, (bp, _) in enumerate(bases_info):
                            peer_base_va[(peer_rank, idx)] = int(bp)
                        continue

                    peer_bases = all_base_payload[peer_rank]
                    for idx, (_, fb, alloc_size) in enumerate(peer_bases):
                        imp_h = import_peer_handle(peer_rank, idx, fb)
                        prop = _check_drv(
                            drv.cuMemGetAllocationPropertiesFromHandle(imp_h),
                            "cuMemGetAllocationPropertiesFromHandle",
                        )
                        gran = _check_drv(
                            drv.cuMemGetAllocationGranularity(
                                prop,
                                drv.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_RECOMMENDED,
                            ),
                            "cuMemGetAllocationGranularity",
                        )
                        va = _check_drv(
                            drv.cuMemAddressReserve(alloc_size, int(gran), 0, 0),
                            "cuMemAddressReserve",
                        )
                        _check_drv(
                            drv.cuMemMap(int(va), alloc_size, 0, imp_h, 0),
                            "cuMemMap",
                        )
                        access = drv.CUmemAccessDesc()
                        access.location.type = (
                            drv.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
                        )
                        access.location.id = device_id
                        access.flags = (
                            drv.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
                        )
                        _check_drv(
                            drv.cuMemSetAccess(int(va), alloc_size, [access], 1),
                            "cuMemSetAccess",
                        )
                        peer_base_va[(peer_rank, idx)] = int(va)
                        new_mappings.append((int(va), alloc_size, [(0, alloc_size)]))
                        _check_drv(drv.cuMemRelease(imp_h), "cuMemRelease(peer)")

                # Build per-input peer VA lists and register.
                peer_ptrs = []
                for j in range(new_count):
                    ptrs_j = []
                    for rank in range(self.world_size):
                        chunks = all_input_chunks[rank][j]
                        off = all_input_offsets[rank][j]
                        if len(chunks) == 1:
                            ptrs_j.append(peer_base_va[(rank, chunks[0])] + off)
                            continue

                        span_key = (rank, *chunks)
                        if span_key not in peer_span_va:
                            peer_bases = all_base_payload[rank]
                            first_base = peer_bases[chunks[0]][0]
                            last_base, _, last_size = peer_bases[chunks[-1]]
                            span_size = (
                                int(last_base) + int(last_size) - int(first_base)
                            )
                            if rank == self.rank:
                                span_va = int(first_base)
                            else:
                                span_va = _check_drv(
                                    drv.cuMemAddressReserve(span_size, 0, 0, 0),
                                    "cuMemAddressReserve(span)",
                                )
                                mapped_chunks = []
                                for chunk_idx in chunks:
                                    base_ptr, fb, alloc_size = peer_bases[chunk_idx]
                                    rel = int(base_ptr) - int(first_base)
                                    imp_h = import_peer_handle(rank, chunk_idx, fb)
                                    _check_drv(
                                        drv.cuMemMap(
                                            int(span_va) + rel,
                                            int(alloc_size),
                                            0,
                                            imp_h,
                                            0,
                                        ),
                                        "cuMemMap(span)",
                                    )
                                    access = drv.CUmemAccessDesc()
                                    access.location.type = (
                                        drv.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
                                    )
                                    access.location.id = device_id
                                    access.flags = (
                                        drv.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
                                    )
                                    _check_drv(
                                        drv.cuMemSetAccess(
                                            int(span_va) + rel,
                                            int(alloc_size),
                                            [access],
                                            1,
                                        ),
                                        "cuMemSetAccess(span)",
                                    )
                                    mapped_chunks.append((rel, int(alloc_size)))
                                    _check_drv(
                                        drv.cuMemRelease(imp_h), "cuMemRelease(span)"
                                    )
                                new_mappings.append(
                                    (int(span_va), span_size, mapped_chunks)
                                )
                            peer_span_va[span_key] = (int(span_va), int(first_base))

                        span_va, _ = peer_span_va[span_key]
                        ptrs_j.append(span_va + off)
                    peer_ptrs.append(ptrs_j)

                self.obj.register_peer_mapped_inputs(peer_ptrs)
                self._peer_mappings.extend(new_mappings)
            except Exception:
                self._release_peer_mappings(new_mappings)
                raise
            finally:
                for fd in posix_peer_fds.values():
                    os.close(fd)

            elapsed_ms = (time.perf_counter() - t0) * 1000
            transport = "FABRIC" if use_fabric else "POSIX fd"
            log_info_on_rank0(
                logger,
                f"Registered {new_count} cuda graph addresses via "
                f"{transport} handles ({num_bases} unique allocations) "
                f"in {elapsed_ms:.1f} ms",
            )
        finally:
            for fd in local_posix_fds:
                os.close(fd)
            for h in retained_handles:
                _check_drv(drv.cuMemRelease(h), "cuMemRelease(retained)")

    def close(self):
        if not self._peer_mappings:
            return
        self._release_peer_mappings(self._peer_mappings)

    def _all_ranks_ok(self, ok: bool) -> bool:
        flag = torch.tensor([1 if ok else 0], dtype=torch.int32)
        dist.all_reduce(flag, op=dist.ReduceOp.BAND, group=self.group)
        return flag.item() == 1

    def _exchange_posix_fds(self, local_fds: List[int], peer_base_counts: List[int]):
        import socket
        import tempfile
        import threading

        sock_kind = getattr(socket, "SOCK_SEQPACKET", socket.SOCK_STREAM)
        sock_dir = tempfile.mkdtemp(prefix="sgl_ar_fd_")
        sock_path = os.path.join(sock_dir, f"rank_{self.rank}.sock")
        server = socket.socket(socket.AF_UNIX, sock_kind)
        server.settimeout(_FD_SEND_TIMEOUT_S)
        received_fds = {}
        errors = []

        def recv_loop():
            try:
                for _ in range(self.world_size - 1):
                    conn, _ = server.accept()
                    with conn:
                        conn.settimeout(_FD_SEND_TIMEOUT_S)
                        while True:
                            packet = _recv_fd(conn)
                            if packet is None:
                                break
                            src_rank, base_idx, fd = packet
                            key = (src_rank, base_idx)
                            if key in received_fds:
                                os.close(fd)
                                raise RuntimeError(f"duplicate fd for {key}")
                            received_fds[key] = fd
            except BaseException as e:
                errors.append(e)

        try:
            server.bind(sock_path)
            server.listen(self.world_size)
            paths = [None] * self.world_size
            dist.all_gather_object(paths, sock_path, group=self.group)

            thread = threading.Thread(target=recv_loop, daemon=True)
            thread.start()
            try:
                for peer_rank, peer_path in enumerate(paths):
                    if peer_rank == self.rank:
                        continue
                    with socket.socket(socket.AF_UNIX, sock_kind) as sock:
                        sock.settimeout(_FD_SEND_TIMEOUT_S)
                        sock.connect(peer_path)
                        for base_idx, fd in enumerate(local_fds):
                            _send_fd(sock, fd, self.rank, base_idx)
            finally:
                thread.join(_FD_SEND_TIMEOUT_S)

            if thread.is_alive():
                raise RuntimeError("timed out waiting for POSIX fd exchange")
            if errors:
                raise RuntimeError("POSIX fd exchange receive failed") from errors[0]

            expected = {
                (rank, base_idx)
                for rank, count in enumerate(peer_base_counts)
                if rank != self.rank
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

    def _release_peer_mappings(self, mappings):
        drv = _get_cuda_driver()
        while mappings:
            va, span_size, mapped_chunks = mappings.pop()
            for rel, size in mapped_chunks:
                _check_drv(drv.cuMemUnmap(int(va) + int(rel), int(size)), "cuMemUnmap")
            _check_drv(
                drv.cuMemAddressFree(int(va), int(span_size)), "cuMemAddressFree"
            )
