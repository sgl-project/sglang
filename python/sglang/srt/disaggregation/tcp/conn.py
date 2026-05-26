"""Pure-Python TCP backend for PD KV transfer.

Replaces the RDMA path (mooncake) with a thread + socket implementation:
- prefill side: one worker per transfer queue, opens a persistent TCP
  connection to each decode rank's data port and streams (layer, indices,
  payload) chunks. KV bytes go GPU -> pinned host buffer -> socket.
- decode side: one TCP server per rank accepts connections from prefill
  ranks; an accept-then-recv loop dispatches each chunk to the destination
  KV pool slot via cudaMemcpyAsync (host -> GPU).
- bootstrap, rank routing, request status, heartbeat, and failure
  propagation reuse the common base implementation unchanged.

Failure model is per-request: a socket exception on one chunk marks that
bootstrap_room as Failed and tears down only its in-flight state, never
permanently blacklisting a peer.
"""

from __future__ import annotations

import ctypes
import dataclasses
import logging
import socket
import struct
import threading
import time
from typing import Dict, List, Optional

import numpy as np
import numpy.typing as npt
import torch
import zmq

from sglang.srt.disaggregation.base.conn import KVArgs, KVPoll
from sglang.srt.disaggregation.common.conn import (
    CommonKVBootstrapServer,
    CommonKVManager,
    CommonKVReceiver,
    CommonKVSender,
)
from sglang.srt.disaggregation.common.utils import (
    FastQueue,
    TransferKVChunk,
    pack_int_lists,
    unpack_int_lists,
)
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.environ import envs
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils.network import NetworkAddress

logger = logging.getLogger(__name__)


# ---- CUDA runtime bridge -----------------------------------------------------

# Use libcudart via the handle torch already loaded, so we don't add a dep.
_cudart = torch.cuda.cudart()
_cudart.cudaMemcpyAsync.argtypes = [
    ctypes.c_void_p,  # dst
    ctypes.c_void_p,  # src
    ctypes.c_size_t,  # count
    ctypes.c_int,  # kind
    ctypes.c_void_p,  # stream
]
_cudart.cudaMemcpyAsync.restype = ctypes.c_int

_CUDA_MEMCPY_HOST_TO_DEVICE = 1
_CUDA_MEMCPY_DEVICE_TO_HOST = 2


def _cuda_memcpy_async(
    dst_ptr: int, src_ptr: int, nbytes: int, kind: int, stream_ptr: int
) -> int:
    return _cudart.cudaMemcpyAsync(
        ctypes.c_void_p(dst_ptr),
        ctypes.c_void_p(src_ptr),
        nbytes,
        kind,
        ctypes.c_void_p(stream_ptr),
    )


# ---- Wire protocol -----------------------------------------------------------
#
# Each chunk message on the data socket is a length-prefixed frame:
#
#   | u32 total_body_len | body |
#
# where body =
#
#   | u64 room | u32 prefill_unique_rank | u8 is_last_chunk | u32 num_buffers |
#   for each buffer:
#     | u8 kind | u8 comp_idx | u32 layer_idx | u32 num_indices | u32 item_len |
#     | u32[num_indices] dst_indices | bytes payload (num_indices * item_len) |
#
# kinds:
#   0 = main KV (kv_data_ptrs[layer]); comp_idx unused
#   1 = state (state_data_ptrs[comp_idx][layer]); covers SWA / DSA / MAMBA
#   2 = aux (aux_data_ptrs[layer]); dst_indices has length 1 = the aux_index;
#       comp_idx unused.

_KIND_MAIN = 0
_KIND_STATE = 1
_KIND_AUX = 2

_HDR_FMT = "<QIBI"  # room (u64), prefill_unique_rank (u32), is_last_chunk (u8), num_buffers (u32)
_HDR_SIZE = struct.calcsize(_HDR_FMT)

_BUF_FMT = "<BBIII"  # kind, comp_idx, layer, num_indices, item_len
_BUF_SIZE = struct.calcsize(_BUF_FMT)


def _recv_exact(sock: socket.socket, nbytes: int) -> bytes:
    chunks = []
    remaining = nbytes
    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError(f"socket closed with {remaining} bytes remaining")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _send_all(sock: socket.socket, data: bytes) -> None:
    sock.sendall(data)


# ---- Decode-side TransferInfo & KVArgsRegisterInfo (same shape as mooncake) --


@dataclasses.dataclass
class TcpTransferInfo:
    """Metadata pushed from decode -> prefill via zmq, per request."""

    room: int
    endpoint: str  # decode ip
    dst_port: int  # decode rank_port (zmq for status sync)
    data_port: int  # decode data_port (TCP for KV payload)
    session_id: str  # "ip:data_port"
    dst_kv_indices: npt.NDArray[np.int32]
    dst_aux_index: Optional[int]
    dst_state_indices: List[List[int]]
    required_dst_info_num: int
    is_dummy: bool
    decode_prefix_len: Optional[int] = None

    @classmethod
    def from_zmq(cls, msg: List[bytes]) -> "TcpTransferInfo":
        is_dummy = msg[4] == b"" and msg[5] == b""
        if is_dummy:
            dst_kv_indices = np.array([], dtype=np.int32)
            dst_aux_index = None
            dst_state_indices = []
        else:
            dst_kv_indices = np.frombuffer(msg[4], dtype=np.int32)
            dst_aux_index = int(msg[5].decode("ascii"))
            dst_state_indices = unpack_int_lists(msg[6], "i")
        return cls(
            room=int(msg[0].decode("ascii")),
            endpoint=msg[1].decode("ascii"),
            dst_port=int(msg[2].decode("ascii")),
            data_port=int(msg[3].decode("ascii")),
            session_id=msg[9].decode("ascii"),
            dst_kv_indices=dst_kv_indices,
            dst_aux_index=dst_aux_index,
            dst_state_indices=dst_state_indices,
            required_dst_info_num=int(msg[7].decode("ascii")),
            is_dummy=is_dummy,
            decode_prefix_len=(
                int(msg[8].decode("ascii")) if len(msg) > 8 and msg[8] != b"" else None
            ),
        )


@dataclasses.dataclass
class TcpKVArgsRegisterInfo:
    """Decode-side KV pool descriptors registered to prefill once at bootstrap."""

    session_id: str
    endpoint: str
    dst_port: int
    data_port: int
    dst_kv_ptrs: List[int]
    dst_aux_ptrs: List[int]
    dst_state_data_ptrs: List[List[int]]
    dst_kv_item_len: int
    dst_aux_item_len: int
    dst_state_item_lens: List[List[int]]

    @classmethod
    def from_zmq(cls, msg: List[bytes]) -> "TcpKVArgsRegisterInfo":
        # msg layout for registration packet ("None" as room sentinel):
        # [0]=b"None" [1]=endpoint [2]=dst_port [3]=data_port
        # [4]=kv_ptrs(u64*) [5]=aux_ptrs(u64*) [6]=state_data_ptrs(packed list-of-u64)
        # [7]=kv_item_len [8]=aux_item_len [9]=state_item_lens(packed list-of-u32) [10]=session_id
        return cls(
            session_id=msg[10].decode("ascii"),
            endpoint=msg[1].decode("ascii"),
            dst_port=int(msg[2].decode("ascii")),
            data_port=int(msg[3].decode("ascii")),
            dst_kv_ptrs=list(struct.unpack(f"<{len(msg[4]) // 8}Q", msg[4])),
            dst_aux_ptrs=list(struct.unpack(f"<{len(msg[5]) // 8}Q", msg[5])),
            dst_state_data_ptrs=unpack_int_lists(msg[6], "Q"),
            dst_kv_item_len=int(msg[7].decode("ascii")),
            dst_aux_item_len=int(msg[8].decode("ascii")),
            dst_state_item_lens=unpack_int_lists(msg[9], "I"),
        )


# ---- KVManager ---------------------------------------------------------------


class TcpKVManager(CommonKVManager):
    """Per-process manager for the TCP backend.

    On prefill it owns the worker pool that pushes KV bytes to decode ranks.
    On decode it owns the TCP listening server that receives KV bytes and
    writes them into the local KV pool via cudaMemcpyAsync.
    """

    def __init__(
        self,
        args: KVArgs,
        disaggregation_mode: DisaggregationMode,
        server_args: ServerArgs,
        is_mla_backend: Optional[bool] = False,
    ):
        super().__init__(args, disaggregation_mode, server_args, is_mla_backend)

        # No engine to init — TCP backend uses plain sockets. The KV ptrs
        # are read directly from self.kv_args at transfer time.
        self.data_host = self.local_ip

        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            self.transfer_infos: Dict[int, Dict[str, TcpTransferInfo]] = {}
            self.decode_kv_args_table: Dict[str, TcpKVArgsRegisterInfo] = {}
            # Persistent client sockets, keyed by "ip:data_port".
            self._client_socks: Dict[str, socket.socket] = {}
            self._client_locks: Dict[str, threading.Lock] = {}
            self._client_pool_lock = threading.Lock()
            # Stream + pinned host bounce buffer used by D2H staging.
            self._d2h_stream = torch.cuda.Stream()
            # Worker pool — single queue is sufficient for spike; mooncake
            # uses N for shardable contention reduction. We can scale later.
            queue_count = max(1, envs.SGLANG_DISAGGREGATION_QUEUE_SIZE.get())
            self.transfer_queues: List[FastQueue] = [
                FastQueue() for _ in range(queue_count)
            ]
            for q in self.transfer_queues:
                threading.Thread(
                    target=self._prefill_transfer_worker,
                    args=(q,),
                    name="TcpKVPrefillWorker",
                    daemon=True,
                ).start()
            self.start_prefill_thread()
        else:  # DECODE
            self.session_id = ""  # set after data port is bound
            self._h2d_stream = torch.cuda.Stream()
            self._start_decode_data_server()
            # session_id format mirrors mooncake's "ip:rpc_port" convention.
            self.session_id = f"{self.local_ip}:{self.data_port}"
            self.start_decode_thread()

    # -- prefill bookkeeping ---------------------------------------------------

    def _prefill_unique_rank(self) -> int:
        return (
            self.attn_tp_rank * (self.pp_size * self.attn_cp_size)
            + self.pp_rank * self.attn_cp_size
            + self.attn_cp_rank
        )

    def start_prefill_thread(self) -> None:
        """Consume metadata + registration packets from decode ranks."""

        def bootstrap_thread():
            while True:
                msg = self.server_socket.recv_multipart()
                room = msg[0].decode("ascii")
                if room == "None":
                    info = TcpKVArgsRegisterInfo.from_zmq(msg)
                    self.decode_kv_args_table[info.session_id] = info
                    logger.debug("Registered KVArgs from %s", info.session_id)
                    continue
                room_i = int(room)
                info = TcpTransferInfo.from_zmq(msg)
                self.transfer_infos.setdefault(room_i, {})[info.session_id] = info
                if len(self.transfer_infos[room_i]) == info.required_dst_info_num:
                    self.req_to_decode_prefix_len[room_i] = next(
                        (
                            it.decode_prefix_len
                            for it in self.transfer_infos[room_i].values()
                            if it.decode_prefix_len is not None
                        ),
                        0,
                    )
                    self.update_status(room_i, KVPoll.WaitingForInput)

        threading.Thread(
            target=bootstrap_thread, name="TcpKVBootstrapRecv", daemon=True
        ).start()

    def add_transfer_request(
        self,
        bootstrap_room: int,
        kv_indices: npt.NDArray[np.int32],
        index_slice: slice,
        is_last_chunk: bool,
        aux_index: Optional[int] = None,
        state_indices: Optional[List] = None,
    ):
        assert self.disaggregation_mode == DisaggregationMode.PREFILL
        if (
            bootstrap_room not in self.request_status
            or self.check_status(bootstrap_room) == KVPoll.Failed
        ):
            return
        if bootstrap_room not in self.transfer_infos:
            # Dummy rank for this room.
            return
        # Shard across queues by room hash so traffic to the same decode set
        # serialises on one worker.
        shard = bootstrap_room % len(self.transfer_queues)
        self.transfer_queues[shard].put(
            TransferKVChunk(
                room=bootstrap_room,
                prefill_kv_indices=kv_indices,
                index_slice=index_slice,
                is_last_chunk=is_last_chunk,
                prefill_aux_index=aux_index,
                state_indices=state_indices,
            )
        )

    # -- prefill worker --------------------------------------------------------

    def _get_client_sock(self, endpoint: str, data_port: int) -> tuple:
        key = f"{endpoint}:{data_port}"
        with self._client_pool_lock:
            sock = self._client_socks.get(key)
            if sock is None:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                sock.connect((endpoint, data_port))
                self._client_socks[key] = sock
                self._client_locks[key] = threading.Lock()
            return sock, self._client_locks[key]

    def _drop_client_sock(self, endpoint: str, data_port: int) -> None:
        key = f"{endpoint}:{data_port}"
        with self._client_pool_lock:
            sock = self._client_socks.pop(key, None)
            self._client_locks.pop(key, None)
        if sock is not None:
            try:
                sock.close()
            except Exception:
                pass

    def _build_chunk_payload(
        self,
        kv_chunk: TransferKVChunk,
        req: TcpTransferInfo,
        reg: TcpKVArgsRegisterInfo,
        prefill_unique_rank: int,
    ) -> bytes:
        """Stage all buffers GPU->pinned host then build the wire frame."""
        # Slice dst KV indices to the same chunk window as prefill indices.
        dst_kv_chunk = req.dst_kv_indices[kv_chunk.index_slice]
        if len(dst_kv_chunk) > len(kv_chunk.prefill_kv_indices):
            dst_kv_chunk = dst_kv_chunk[: len(kv_chunk.prefill_kv_indices)]
        src_kv_chunk = kv_chunk.prefill_kv_indices[: len(dst_kv_chunk)]

        buffers: List[tuple] = (
            []
        )  # (kind, comp_idx, layer, dst_indices_np, payload_bytes)

        if len(src_kv_chunk) > 0:
            for layer_idx, (src_ptr, item_len) in enumerate(
                zip(self.kv_args.kv_data_ptrs, self.kv_args.kv_item_lens)
            ):
                payload = self._stage_d2h(src_ptr, item_len, src_kv_chunk)
                buffers.append(
                    (_KIND_MAIN, 0, layer_idx, dst_kv_chunk.astype(np.int32), payload)
                )

            # SWA / DSA / extra state, indexed by component then by layer.
            for comp_idx, comp_src_ptrs in enumerate(self.kv_args.state_data_ptrs):
                src_state_chunk_np = (
                    np.asarray(kv_chunk.state_indices[comp_idx], dtype=np.int32)
                    if kv_chunk.state_indices and comp_idx < len(kv_chunk.state_indices)
                    else None
                )
                if src_state_chunk_np is None or len(src_state_chunk_np) == 0:
                    continue
                dst_state_chunk_np = (
                    np.asarray(req.dst_state_indices[comp_idx], dtype=np.int32)
                    if comp_idx < len(req.dst_state_indices)
                    else None
                )
                if dst_state_chunk_np is None or len(dst_state_chunk_np) == 0:
                    continue
                n = min(len(src_state_chunk_np), len(dst_state_chunk_np))
                src_state_chunk_np = src_state_chunk_np[:n]
                dst_state_chunk_np = dst_state_chunk_np[:n]
                state_item_lens = self.kv_args.state_item_lens[comp_idx]
                for layer_idx, (src_ptr, item_len) in enumerate(
                    zip(comp_src_ptrs, state_item_lens)
                ):
                    payload = self._stage_d2h(src_ptr, item_len, src_state_chunk_np)
                    buffers.append(
                        (_KIND_STATE, comp_idx, layer_idx, dst_state_chunk_np, payload)
                    )

        # Aux data goes only with the last chunk.
        if kv_chunk.is_last_chunk and kv_chunk.prefill_aux_index is not None:
            src_aux_idx = np.array([kv_chunk.prefill_aux_index], dtype=np.int32)
            dst_aux_idx = np.array([req.dst_aux_index], dtype=np.int32)
            for layer_idx, (src_ptr, item_len) in enumerate(
                zip(self.kv_args.aux_data_ptrs, self.kv_args.aux_item_lens)
            ):
                payload = self._stage_d2h(src_ptr, item_len, src_aux_idx)
                buffers.append((_KIND_AUX, 0, layer_idx, dst_aux_idx, payload))

        # Wait for all D2H copies before serialising.
        self._d2h_stream.synchronize()

        return self._frame(
            kv_chunk.room, prefill_unique_rank, kv_chunk.is_last_chunk, buffers
        )

    def _stage_d2h(
        self, src_base_ptr: int, item_len: int, indices: npt.NDArray[np.int32]
    ) -> bytes:
        """GPU pages at src_base_ptr[indices*item_len] -> pinned host bytes."""
        n = len(indices)
        if n == 0:
            return b""
        nbytes = n * item_len
        host_buf = torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
        host_ptr = host_buf.data_ptr()
        stream_ptr = self._d2h_stream.cuda_stream
        # Each index is one item; copy sequentially. Sequential issue is OK
        # since they're on the same stream and overlap with the socket I/O
        # of the previous chunk.
        for i, idx in enumerate(indices.tolist()):
            ret = _cuda_memcpy_async(
                host_ptr + i * item_len,
                src_base_ptr + idx * item_len,
                item_len,
                _CUDA_MEMCPY_DEVICE_TO_HOST,
                stream_ptr,
            )
            if ret != 0:
                raise RuntimeError(f"cudaMemcpyAsync D2H failed: {ret}")
        return bytes(memoryview(host_buf.numpy()))

    def _frame(
        self,
        room: int,
        prefill_unique_rank: int,
        is_last_chunk: bool,
        buffers: List[tuple],
    ) -> bytes:
        parts = [
            struct.pack(
                _HDR_FMT, room, prefill_unique_rank, int(is_last_chunk), len(buffers)
            )
        ]
        for kind, comp_idx, layer, dst_indices_np, payload in buffers:
            n = max(len(dst_indices_np), 1)
            parts.append(
                struct.pack(
                    _BUF_FMT,
                    kind,
                    comp_idx,
                    layer,
                    len(dst_indices_np),
                    len(payload) // n,
                )
            )
            parts.append(dst_indices_np.tobytes())
            parts.append(payload)
        body = b"".join(parts)
        return struct.pack("<I", len(body)) + body

    def _prefill_transfer_worker(self, queue: FastQueue) -> None:
        prefill_unique_rank = self._prefill_unique_rank()
        while True:
            kv_chunk: TransferKVChunk = queue.get()
            reqs = self.transfer_infos.get(kv_chunk.room, {}).values()
            polls = []
            dst_ranks = []
            for req in list(reqs):
                if req.is_dummy:
                    polls.append(True)
                    dst_ranks.append((req.endpoint, req.dst_port, req.room))
                    continue
                reg = self.decode_kv_args_table.get(req.session_id)
                if reg is None:
                    self._fail_room(
                        kv_chunk.room,
                        req,
                        prefill_unique_rank,
                        f"decode session {req.session_id} not registered",
                    )
                    polls.append(False)
                    dst_ranks.append((req.endpoint, req.dst_port, req.room))
                    continue
                try:
                    frame = self._build_chunk_payload(
                        kv_chunk, req, reg, prefill_unique_rank
                    )
                    sock, lock = self._get_client_sock(req.endpoint, req.data_port)
                    with lock:
                        _send_all(sock, frame)
                    polls.append(True)
                except Exception as e:
                    self._drop_client_sock(req.endpoint, req.data_port)
                    self._fail_room(
                        kv_chunk.room,
                        req,
                        prefill_unique_rank,
                        f"chunk send to {req.endpoint}:{req.data_port} failed: {e!r}",
                    )
                    polls.append(False)
                dst_ranks.append((req.endpoint, req.dst_port, req.room))

            if kv_chunk.is_last_chunk:
                status = KVPoll.Success if all(polls) else KVPoll.Failed
                self.update_status(kv_chunk.room, status)
                for endpoint, dst_port, room in dst_ranks:
                    self._sync_status_to_decode_endpoint(
                        endpoint, dst_port, room, status, prefill_unique_rank
                    )

    def _fail_room(
        self,
        room: int,
        req: TcpTransferInfo,
        prefill_unique_rank: int,
        reason: str,
    ) -> None:
        self.record_failure(room, reason)
        self.update_status(room, KVPoll.Failed)
        self._sync_status_to_decode_endpoint(
            req.endpoint, req.dst_port, room, KVPoll.Failed, prefill_unique_rank
        )

    # -- status sync from prefill -> decode (zmq PUSH) -------------------------

    def _sync_status_to_decode_endpoint(
        self, remote: str, dst_port: int, room: int, status: int, prefill_rank: int
    ) -> None:
        na = NetworkAddress(remote, dst_port)
        sock = _zmq_push_socket(na.to_tcp(), is_ipv6=na.is_ipv6)
        sock.send_multipart(
            [
                str(room).encode("ascii"),
                str(status).encode("ascii"),
                str(prefill_rank).encode("ascii"),
            ]
        )

    # -- decode side -----------------------------------------------------------

    def _start_decode_data_server(self) -> None:
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((self.local_ip, 0))
        self.data_port = srv.getsockname()[1]
        srv.listen(64)

        def accept_loop():
            while True:
                conn, addr = srv.accept()
                conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                threading.Thread(
                    target=self._decode_recv_loop,
                    args=(conn, addr),
                    name=f"TcpKVDecodeRecv-{addr}",
                    daemon=True,
                ).start()

        threading.Thread(
            target=accept_loop, name="TcpKVDecodeAccept", daemon=True
        ).start()
        logger.info(
            "TcpKVManager listening for KV payload on %s:%s",
            self.local_ip,
            self.data_port,
        )

    def _decode_recv_loop(self, conn: socket.socket, addr) -> None:
        try:
            while True:
                hdr = _recv_exact(conn, 4)
                body_len = struct.unpack("<I", hdr)[0]
                body = _recv_exact(conn, body_len)
                self._apply_frame(body)
        except (ConnectionError, OSError) as e:
            logger.debug("Decode recv loop %s closed: %s", addr, e)
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def _apply_frame(self, body: bytes) -> None:
        room, prefill_rank, is_last, num_buffers = struct.unpack_from(_HDR_FMT, body, 0)
        off = _HDR_SIZE
        stream_ptr = self._h2d_stream.cuda_stream
        for _ in range(num_buffers):
            kind, comp_idx, layer, n_idx, item_len = struct.unpack_from(
                _BUF_FMT, body, off
            )
            off += _BUF_SIZE
            dst_indices = np.frombuffer(body, dtype=np.int32, count=n_idx, offset=off)
            off += n_idx * 4
            payload_len = n_idx * item_len
            payload_view = memoryview(body)[off : off + payload_len]
            off += payload_len

            dst_base_ptr = self._dst_base_ptr(kind, comp_idx, layer)
            if dst_base_ptr == 0:
                continue
            # Pin the slice and issue per-page H2D.
            # We allocate a pinned staging buffer per frame; copy_ ensures the
            # bytes are page-locked before the async cudaMemcpy is enqueued.
            host_buf = torch.empty(payload_len, dtype=torch.uint8, pin_memory=True)
            host_arr = host_buf.numpy()
            host_arr[:] = np.frombuffer(payload_view, dtype=np.uint8)
            host_ptr = host_buf.data_ptr()
            for i, idx in enumerate(dst_indices.tolist()):
                ret = _cuda_memcpy_async(
                    dst_base_ptr + idx * item_len,
                    host_ptr + i * item_len,
                    item_len,
                    _CUDA_MEMCPY_HOST_TO_DEVICE,
                    stream_ptr,
                )
                if ret != 0:
                    raise RuntimeError(f"cudaMemcpyAsync H2D failed: {ret}")
        self._h2d_stream.synchronize()

    def _dst_base_ptr(self, kind: int, comp_idx: int, layer: int) -> int:
        if kind == _KIND_MAIN:
            return self.kv_args.kv_data_ptrs[layer]
        if kind == _KIND_AUX:
            return self.kv_args.aux_data_ptrs[layer]
        if kind == _KIND_STATE:
            if comp_idx >= len(self.kv_args.state_data_ptrs):
                return 0
            return self.kv_args.state_data_ptrs[comp_idx][layer]
        return 0

    def start_decode_thread(self) -> None:
        """Consume status pushes from prefill ranks (Success/Failed per room)."""

        def decode_thread():
            while True:
                msg = self.server_socket.recv_multipart()
                if len(msg) != 3:
                    continue
                room = int(msg[0].decode("ascii"))
                status = int(msg[1].decode("ascii"))
                prefill_rank = int(msg[2].decode("ascii"))
                if room not in self.request_status:
                    continue
                if status == KVPoll.Success:
                    self.prefill_response_tracker[room].add(prefill_rank)
                    expected = self.required_prefill_response_num_table.get(room, 1)
                    if len(self.prefill_response_tracker[room]) == expected:
                        self.update_status(room, KVPoll.Success)
                elif status == KVPoll.Failed:
                    self.record_failure(room, "Prefill reported transfer failure")
                    self.update_status(room, KVPoll.Failed)

        threading.Thread(
            target=decode_thread, name="TcpKVDecodeStatus", daemon=True
        ).start()


# ---- zmq PUSH socket cache (mirrors common._connect) -------------------------

_zmq_ctx = zmq.Context.instance()
_zmq_socks: Dict[str, zmq.Socket] = {}
_zmq_lock = threading.Lock()


def _zmq_push_socket(endpoint: str, is_ipv6: bool = False) -> zmq.Socket:
    with _zmq_lock:
        sock = _zmq_socks.get(endpoint)
        if sock is None:
            sock = _zmq_ctx.socket(zmq.PUSH)
            if is_ipv6:
                sock.setsockopt(zmq.IPV6, 1)
            sock.connect(endpoint)
            _zmq_socks[endpoint] = sock
        return sock


# ---- KVSender ----------------------------------------------------------------


class TcpKVSender(CommonKVSender):
    def __init__(
        self,
        mgr: TcpKVManager,
        bootstrap_addr: str,
        bootstrap_room: int,
        dest_tp_ranks: List[int],
        pp_rank: int,
    ):
        super().__init__(mgr, bootstrap_addr, bootstrap_room, dest_tp_ranks, pp_rank)
        self.conclude_state: Optional[KVPoll] = None
        self.init_time = time.time()

    def send(
        self,
        kv_indices: npt.NDArray[np.int32],
        state_indices: Optional[List] = None,
    ):
        kv_indices, index_slice, is_last_chunk, should_skip = (
            self._prepare_send_indices(kv_indices, state_indices)
        )
        if should_skip:
            return
        self.kv_mgr.add_transfer_request(
            self.bootstrap_room,
            kv_indices,
            index_slice,
            is_last_chunk,
            aux_index=(self.aux_index if is_last_chunk else None),
            state_indices=state_indices if is_last_chunk else None,
        )
        self._record_transfer_indices(kv_indices, state_indices)

    def poll(self) -> KVPoll:
        if self.conclude_state is not None:
            return self.conclude_state
        status = self.kv_mgr.check_status(self.bootstrap_room)
        if status in (KVPoll.Success, KVPoll.Failed):
            self.conclude_state = status
        elif status == KVPoll.Bootstrapping:
            timeout_result = self._check_bootstrap_timeout()
            if timeout_result is not None:
                return timeout_result
        return status

    def failure_exception(self):
        from sglang.srt.disaggregation.mooncake.conn import KVTransferError

        if self.conclude_state is None:
            self.conclude_state = KVPoll.Failed
        self.clear()
        with self.kv_mgr.failure_lock:
            reason = self.kv_mgr.failure_records.pop(
                self.bootstrap_room, "Failed due to an unknown reason from another rank"
            )
        raise KVTransferError(self.bootstrap_room, reason)


# ---- KVReceiver --------------------------------------------------------------


class TcpKVReceiver(CommonKVReceiver):
    def __init__(
        self,
        mgr: TcpKVManager,
        bootstrap_addr: str,
        bootstrap_room: Optional[int] = None,
    ):
        # Compose the session id once per decode-side instance — it's the
        # decode rank's (ip, data_port).
        self.session_id = mgr.session_id
        self.init_time: Optional[float] = None
        super().__init__(mgr, bootstrap_addr, bootstrap_room)

    def _register_kv_args(self) -> None:
        """Push our KV pool descriptors to each prefill rank in our target set."""
        kv_ptrs_bytes = struct.pack(
            f"<{len(self.kv_mgr.kv_args.kv_data_ptrs)}Q",
            *self.kv_mgr.kv_args.kv_data_ptrs,
        )
        aux_ptrs_bytes = struct.pack(
            f"<{len(self.kv_mgr.kv_args.aux_data_ptrs)}Q",
            *self.kv_mgr.kv_args.aux_data_ptrs,
        )
        state_ptrs_packed = pack_int_lists(self.kv_mgr.kv_args.state_data_ptrs, "Q")
        state_item_lens_packed = pack_int_lists(
            self.kv_mgr.kv_args.state_item_lens, "I"
        )
        kv_item_len = (
            self.kv_mgr.kv_args.kv_item_lens[0]
            if self.kv_mgr.kv_args.kv_item_lens
            else 0
        )
        aux_item_len = (
            self.kv_mgr.kv_args.aux_item_lens[0]
            if self.kv_mgr.kv_args.aux_item_lens
            else 0
        )

        for bootstrap_info in self.bootstrap_infos:
            sock, lock = self._connect_to_bootstrap_server(bootstrap_info)
            with lock:
                sock.send_multipart(
                    [
                        b"None",
                        self.kv_mgr.local_ip.encode("ascii"),
                        str(self.kv_mgr.rank_port).encode("ascii"),
                        str(self.kv_mgr.data_port).encode("ascii"),
                        kv_ptrs_bytes,
                        aux_ptrs_bytes,
                        state_ptrs_packed,
                        str(kv_item_len).encode("ascii"),
                        str(aux_item_len).encode("ascii"),
                        state_item_lens_packed,
                        self.session_id.encode("ascii"),
                    ]
                )

    def send_metadata(
        self,
        kv_indices: npt.NDArray[np.int32],
        aux_index: Optional[int] = None,
        state_indices: Optional[List] = None,
        decode_prefix_len: Optional[int] = None,
    ):
        if self.bootstrap_infos is None:
            self.kv_mgr.record_failure(
                self.bootstrap_room,
                f"Could not fetch prefill parallel info from {self.bootstrap_addr}",
            )
            self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Failed)
            return
        for bootstrap_info in self.bootstrap_infos:
            sock, lock = self._connect_to_bootstrap_server(bootstrap_info)
            is_dummy = bootstrap_info.get("is_dummy", False)
            with lock:
                sock.send_multipart(
                    [
                        str(self.bootstrap_room).encode("ascii"),
                        self.kv_mgr.local_ip.encode("ascii"),
                        str(self.kv_mgr.rank_port).encode("ascii"),
                        str(self.kv_mgr.data_port).encode("ascii"),
                        kv_indices.tobytes() if not is_dummy else b"",
                        str(aux_index).encode("ascii") if not is_dummy else b"",
                        (
                            pack_int_lists(state_indices, "i")
                            if not is_dummy and state_indices
                            else b""
                        ),
                        str(self.required_dst_info_num).encode("ascii"),
                        str(decode_prefix_len or 0).encode("ascii"),
                        self.session_id.encode("ascii"),
                    ]
                )
        self.init_time = time.time()

    def poll(self) -> KVPoll:
        if self.conclude_state is not None:
            return self.conclude_state
        status = self.kv_mgr.check_status(self.bootstrap_room)
        if status in (KVPoll.Success, KVPoll.Failed):
            self.conclude_state = status
        elif status == KVPoll.WaitingForInput:
            timeout_result = self._check_waiting_timeout()
            if timeout_result is not None:
                return timeout_result
        return status

    def failure_exception(self):
        from sglang.srt.disaggregation.mooncake.conn import KVTransferError

        if self.conclude_state is None:
            self.conclude_state = KVPoll.Failed
        self.clear()
        with self.kv_mgr.failure_lock:
            reason = self.kv_mgr.failure_records.pop(
                self.bootstrap_room, "Failed due to an unknown reason from another rank"
            )
        raise KVTransferError(self.bootstrap_room, reason)


# ---- Bootstrap server (no backend-specific extension) ------------------------


class TcpKVBootstrapServer(CommonKVBootstrapServer):
    pass
