# SPDX-License-Identifier: Apache-2.0
"""Demo StorageServer: file-backed KV L3, gRPC + separate mempool UDS for RegisterPool."""

from __future__ import annotations

import argparse
import hashlib
import logging
import mmap
import os
import socket
import struct
import tempfile
import threading
from concurrent import futures
from typing import Dict, Optional

import grpc

from sglang.srt.mem_cache.storage.rpc import hicache_pb2_grpc
from sglang.srt.mem_cache.storage.rpc.hicache_pb2 import (
    BatchGetRequest,
    BatchGetResponse,
    BatchPutRequest,
    BatchPutResponse,
    ExistRequest,
    ExistResponse,
)
from sglang.srt.mem_cache.storage.rpc.rpc_fd import (
    parse_unix_socket_uri,
    recv_pool_registration,
)

logger = logging.getLogger(__name__)


def _safe_path_key(id_bytes: bytes) -> str:
    return hashlib.sha256(id_bytes).hexdigest()


def _repr_request_id(raw: bytes) -> str:
    """Short debug string for proto bytes id."""
    if not raw:
        return "<empty>"
    if len(raw) <= 64:
        try:
            s = raw.decode("utf-8")
            if "\x00" not in s and all(ch.isprintable() or ch in "\t\n\r" for ch in s):
                return repr(s)
        except UnicodeDecodeError:
            pass
        return raw.hex()
    return f"{raw[:32].hex()}...<{len(raw)} bytes>"


class _FileServicer(hicache_pb2_grpc.StorageServerServicer):
    """File-backed L3 keyed by HASH(id); raw bytes come from pool mmap slices."""

    def __init__(self, data_dir: str, pool_mmap: Dict[int, mmap.mmap]):
        self.data_dir = data_dir
        self.pool_mmap = pool_mmap
        os.makedirs(data_dir, exist_ok=True)

    def Exist(self, request: ExistRequest, context):
        results = []
        for raw in request.id:
            fp = os.path.join(self.data_dir, _safe_path_key(raw))
            results.append(os.path.isfile(fp))
        return ExistResponse(results=results)

    @staticmethod
    def _status_i32(code: grpc.StatusCode) -> int:
        return int(code.value[0])

    def _put_one(self, mm: mmap.mmap, pool_id: int, id_b: bytes, iovec) -> int:
        """Return 0 on success, else a gRPC status code as int32."""
        if not iovec:
            return self._status_i32(grpc.StatusCode.INVALID_ARGUMENT)
        logger.debug(
            "BatchPut item pool_id=%s n_iovec=%d id=%s",
            pool_id,
            len(iovec),
            _repr_request_id(bytes(id_b)),
        )
        chunks = []
        for iv in iovec:
            off = int(iv.offset)
            ln = int(iv.length)
            if off < 0 or ln < 0 or off + ln > len(mm):
                return self._status_i32(grpc.StatusCode.OUT_OF_RANGE)
            chunks.append(bytes(mm[off : off + ln]))
        data = b"".join(chunks)
        fp = os.path.join(self.data_dir, _safe_path_key(bytes(id_b)))
        try:
            with open(fp, "wb") as fout:
                fout.write(data)
        except OSError:
            return self._status_i32(grpc.StatusCode.INTERNAL)
        return 0

    def BatchPut(self, request: BatchPutRequest, context):
        results = []
        for put in request.puts:
            pid = int(put.pool_id)
            if pid == 0:
                results.append(self._status_i32(grpc.StatusCode.INVALID_ARGUMENT))
                continue
            if pid not in self.pool_mmap:
                results.append(self._status_i32(grpc.StatusCode.FAILED_PRECONDITION))
                continue
            results.append(
                self._put_one(self.pool_mmap[pid], pid, bytes(put.id), put.iovec)
            )
        return BatchPutResponse(results=results)

    def _get_one(self, mm: mmap.mmap, pool_id: int, id_b: bytes, iovec) -> int:
        """Return 0 on success, else a gRPC status code as int32."""
        if not iovec:
            return self._status_i32(grpc.StatusCode.INVALID_ARGUMENT)
        logger.debug(
            "BatchGet item pool_id=%s n_iovec=%d id=%s",
            pool_id,
            len(iovec),
            _repr_request_id(bytes(id_b)),
        )
        fp = os.path.join(self.data_dir, _safe_path_key(bytes(id_b)))
        try:
            with open(fp, "rb") as fin:
                raw = fin.read()
        except FileNotFoundError:
            return self._status_i32(grpc.StatusCode.NOT_FOUND)
        expect = sum(int(iv.length) for iv in iovec)
        if len(raw) != expect:
            return self._status_i32(grpc.StatusCode.INVALID_ARGUMENT)
        pos = 0
        for iv in iovec:
            off = int(iv.offset)
            ln = int(iv.length)
            if off < 0 or ln < 0 or off + ln > len(mm):
                return self._status_i32(grpc.StatusCode.OUT_OF_RANGE)
            chunk = raw[pos : pos + ln]
            mm[off : off + ln] = chunk
            pos += ln
        return 0

    def BatchGet(self, request: BatchGetRequest, context):
        results = []
        for get in request.gets:
            pid = int(get.pool_id)
            if pid == 0:
                results.append(self._status_i32(grpc.StatusCode.INVALID_ARGUMENT))
                continue
            if pid not in self.pool_mmap:
                results.append(self._status_i32(grpc.StatusCode.FAILED_PRECONDITION))
                continue
            results.append(
                self._get_one(self.pool_mmap[pid], pid, bytes(get.id), get.iovec)
            )
        return BatchGetResponse(results=results)


def mmap_from_fd_recv(fd: int) -> mmap.mmap:
    sz = os.fstat(fd).st_size
    logger.info("Mmap memory %s bytes", sz)
    mm = mmap.mmap(fd, int(sz), access=mmap.ACCESS_WRITE)
    try:
        os.close(fd)
    except OSError:
        pass
    return mm


def _unlink_if_exists(path: str) -> None:
    try:
        if os.path.exists(path):
            os.unlink(path)
    except OSError:
        pass


def run_mempool_server(
    mempool_path: str,
    pool_mmap: Dict[int, mmap.mmap],
    pool_mmap_lock: threading.Lock,
    next_pool_id_state: list,
    shutdown: threading.Event,
) -> threading.Thread:
    """Listen on mempool UDS: RegisterPool (SCM_RIGHTS FD) → assign ``pool_id``, keep session alive."""
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    p = parse_unix_socket_uri(mempool_path)
    _unlink_if_exists(p)
    srv.bind(p)
    srv.listen(256)
    srv.setblocking(True)
    logger.info("Mempool RegisterPool listening on %s", p)

    def handle_conn(conn: socket.socket) -> None:
        pool_id_assigned: Optional[int] = None
        mm: Optional[mmap.mmap] = None
        try:
            conn.settimeout(600.0)
            _pool_name, rfd = recv_pool_registration(conn)
            mm = mmap_from_fd_recv(rfd)
            with pool_mmap_lock:
                pool_id_assigned = next_pool_id_state[0]
                next_pool_id_state[0] = pool_id_assigned + 1
                pool_mmap[pool_id_assigned] = mm
            conn.sendall(struct.pack("<I", pool_id_assigned))
            logger.info(
                "Registered pool mmap pool_id=%s len=%s",
                pool_id_assigned,
                len(mm),
            )
            conn.settimeout(0.5)
            while not shutdown.is_set():
                try:
                    chunk = conn.recv(65536)
                except TimeoutError:
                    continue
                if not chunk:
                    break
        except OSError as e:
            logger.debug("mempool connection: %s", e)
        except ValueError as e:
            logger.warning("mempool registration failed: %s", e)
        finally:
            if pool_id_assigned is not None:
                with pool_mmap_lock:
                    pool_mmap.pop(pool_id_assigned, None)
            if mm is not None:
                try:
                    mm.close()
                except OSError:
                    pass
            try:
                conn.close()
            except OSError:
                pass
            if pool_id_assigned is not None:
                logger.info("Unmapped pool_id=%s (mempool session end)", pool_id_assigned)

    def loop() -> None:
        while not shutdown.is_set():
            try:
                srv.settimeout(0.5)
                try:
                    conn, _addr = srv.accept()
                except TimeoutError:
                    continue
            except OSError:
                break
            threading.Thread(target=handle_conn, args=(conn,), daemon=True).start()
        try:
            srv.close()
        except OSError:
            pass
        _unlink_if_exists(p)

    t = threading.Thread(target=loop, daemon=True)
    t.start()
    return t


def start_demo_server(data_dir: str, mempool_socket: str, grpc_socket: str) -> dict:
    """Start file-backed StorageServer on ``grpc_socket`` + mempool listener on ``mempool_socket``."""
    shutdown = threading.Event()
    pool_mmap: Dict[int, mmap.mmap] = {}
    pool_mmap_lock = threading.Lock()
    next_pool_id_state = [1]

    servicer = _FileServicer(data_dir, pool_mmap)
    grpc_serv = grpc.server(
        futures.ThreadPoolExecutor(max_workers=8),
        options=(
            ("grpc.max_send_message_length", 512 * 1024 * 1024),
            ("grpc.max_receive_message_length", 512 * 1024 * 1024),
        ),
    )
    hicache_pb2_grpc.add_StorageServerServicer_to_server(servicer, grpc_serv)
    grpc_path = parse_unix_socket_uri(grpc_socket)
    _unlink_if_exists(grpc_path)
    uds_addr = f"unix://{grpc_path}"
    if grpc_serv.add_insecure_port(uds_addr) == 0:
        raise RuntimeError(f"failed to bind gRPC to {uds_addr}")
    grpc_serv.start()
    mempool_thread = run_mempool_server(
        mempool_socket,
        pool_mmap,
        pool_mmap_lock,
        next_pool_id_state,
        shutdown,
    )
    return {
        "grpc_server": grpc_serv,
        "pool_mmap": pool_mmap,
        "pool_mmap_lock": pool_mmap_lock,
        "mempool_thread": mempool_thread,
        "data_dir": data_dir,
        "shutdown": shutdown,
        "mempool_socket": mempool_socket,
        "grpc_socket": grpc_socket,
    }


def main(argv: Optional[list[str]] = None) -> None:
    logging.basicConfig(
        level=os.environ.get("LOGLEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    ap = argparse.ArgumentParser(description="HiCache demo StorageServer")
    ap.add_argument(
        "--data-dir",
        default=None,
        help="Directory storing cache blobs (default: tempfile)",
    )
    ap.add_argument(
        "--mempool-socket",
        required=True,
        help="Unix socket for RegisterPool (magic + name + SCM_RIGHTS memfd)",
    )
    ap.add_argument(
        "--grpc-socket",
        required=True,
        help="Unix socket for gRPC StorageServer",
    )
    ns = ap.parse_args(argv)

    data_dir = ns.data_dir or tempfile.mkdtemp(prefix="hicache_demo_")
    logger.info("Data directory: %s", data_dir)
    os.makedirs(data_dir, exist_ok=True)
    ctx = start_demo_server(data_dir, ns.mempool_socket, ns.grpc_socket)
    grpc_serv = ctx["grpc_server"]
    shutdown = ctx["shutdown"]
    logger.info(
        "Demo StorageServer running (data-dir=%s grpc=%s mempool=%s)",
        data_dir,
        parse_unix_socket_uri(ns.grpc_socket),
        parse_unix_socket_uri(ns.mempool_socket),
    )
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        shutdown.set()
        grpc_serv.stop(grace=2.0)
        _unlink_if_exists(parse_unix_socket_uri(ns.grpc_socket))


if __name__ == "__main__":
    main()
