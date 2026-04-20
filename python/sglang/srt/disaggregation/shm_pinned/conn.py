"""
Connection classes for the shm_pinned backend.
"""

from __future__ import annotations

import ctypes
import json
import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import numpy.typing as npt
import zmq

from sglang.srt.disaggregation.base.conn import KVArgs, KVPoll
from sglang.srt.disaggregation.common.conn import (
    CommonKVBootstrapServer,
    CommonKVManager,
    CommonKVReceiver,
    CommonKVSender,
)
from sglang.srt.disaggregation.common.utils import group_concurrent_contiguous
from sglang.srt.disaggregation.shm_pinned.transfer_engine import (
    ShmPinnedTransferEngine,
)
from sglang.srt.disaggregation.shm_pinned.utils import (
    DEFAULT_CHUNK_TOKENS,
    DEFAULT_SLOT_COUNT,
    ShmPinnedInfo,
)
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils.network import NetworkAddress

logger = logging.getLogger(__name__)

RequestKey = tuple[str, int]


def make_request_key(session_id: str, room: int) -> RequestKey:
    return (session_id, int(room))


@dataclass
class TransferRequest:
    room: int
    session_id: str
    dst_kv_indices: npt.NDArray[np.int32]
    dst_aux_index: Optional[int] = None
    engine: Optional[ShmPinnedTransferEngine] = None
    total_pages: int = 0
    received_pages: int = 0
    last_chunk_seen: bool = False
    received_bitmap: Optional[bytearray] = None


class ShmPinnedKVManager(CommonKVManager):
    def __init__(
        self,
        args: KVArgs,
        disaggregation_mode: DisaggregationMode,
        server_args: ServerArgs,
        is_mla_backend: Optional[bool] = False,
    ):
        super().__init__(args, disaggregation_mode, server_args, is_mla_backend)
        self._validate_state_support()

        self.engine: Optional[ShmPinnedTransferEngine] = None
        self.session_shm_table: Dict[str, ShmPinnedInfo] = {}
        self.session_engines: Dict[str, ShmPinnedTransferEngine] = {}
        self.pending_requests: Dict[RequestKey, TransferRequest] = {}
        self.room_sessions: Dict[int, set[str]] = defaultdict(set)
        self.pending_lock = threading.Lock()
        self.decode_connections: Dict[str, tuple[object, threading.Lock]] = {}

        self.slot_count = (
            server_args.disaggregation_shm_slot_count or DEFAULT_SLOT_COUNT
        )
        self.chunk_tokens = (
            server_args.disaggregation_shm_chunk_tokens or DEFAULT_CHUNK_TOKENS
        )
        page_size = max(1, int(getattr(self.kv_args, "page_size", 1)))
        self.chunk_pages = max(1, (self.chunk_tokens + page_size - 1) // page_size)

        if disaggregation_mode == DisaggregationMode.DECODE:
            self._init_engine_decode()
        elif disaggregation_mode == DisaggregationMode.PREFILL:
            self._start_prefill_thread()

    def _validate_state_support(self) -> None:
        state_type = getattr(self.kv_args, "state_type", "none") or "none"
        if state_type != "none":
            raise ValueError(
                "shm_pinned PR1 supports KV + aux only; state transfer is not supported."
            )

    def _init_engine_decode(self) -> None:
        session_id = f"decode_{self.kv_args.engine_rank}_{int(time.time() * 1000)}"
        aux_bytes = int(sum(getattr(self.kv_args, "aux_item_lens", []) or []))
        self.engine = ShmPinnedTransferEngine(
            session_id=session_id,
            gpu_id=self.kv_args.gpu_id,
            slot_count=self.slot_count,
            chunk_pages=self.chunk_pages,
            kv_item_lens=self.kv_args.kv_item_lens,
            extra_slot_bytes=aux_bytes,
            create=True,
        )
        self._decode_thread_running = True
        self._decode_thread = threading.Thread(
            target=self._decode_receive_loop,
            daemon=True,
        )
        self._decode_thread.start()

    def _start_prefill_thread(self) -> None:
        self._prefill_thread_running = True
        self._prefill_thread = threading.Thread(
            target=self._prefill_transfer_loop,
            daemon=True,
        )
        self._prefill_thread.start()

    def _add_pending_request(self, req: TransferRequest) -> None:
        key = make_request_key(req.session_id, req.room)
        with self.pending_lock:
            self.pending_requests[key] = req
            self.room_sessions[req.room].add(req.session_id)

    def _pop_pending_request(self, key: RequestKey) -> Optional[TransferRequest]:
        with self.pending_lock:
            req = self.pending_requests.pop(key, None)
            if req is None:
                return None
            sessions = self.room_sessions.get(req.room)
            if sessions is not None:
                sessions.discard(req.session_id)
                if not sessions:
                    self.room_sessions.pop(req.room, None)
            return req

    def get_pending_request(
        self, session_id: str, room: int
    ) -> Optional[TransferRequest]:
        with self.pending_lock:
            return self.pending_requests.get(make_request_key(session_id, room))

    def get_pending_request_for_room(self, room: int) -> Optional[TransferRequest]:
        with self.pending_lock:
            sessions = self.room_sessions.get(room)
            if not sessions or len(sessions) != 1:
                return None
            session_id = next(iter(sessions))
            return self.pending_requests.get(make_request_key(session_id, room))

    def get_status(self, room: int) -> KVPoll:
        return self.request_status.get(room, KVPoll.Failed)

    def get_failure_message(self, room: int) -> Optional[str]:
        with self.failure_lock:
            return self.failure_records.get(room)

    def _connect_to_decode(self, session_id: str) -> Optional[tuple[object, threading.Lock]]:
        if session_id in self.decode_connections:
            return self.decode_connections[session_id]

        shm_info = self.session_shm_table.get(session_id)
        if shm_info is None or not shm_info.decode_host or not shm_info.decode_port:
            return None

        na = NetworkAddress(shm_info.decode_host, shm_info.decode_port)
        sock, lock = CommonKVReceiver._connect(na.to_tcp(), is_ipv6=na.is_ipv6)
        self.decode_connections[session_id] = (sock, lock)
        return sock, lock

    def _send_failure_to_decode(self, session_id: str, room: int, reason: str) -> None:
        connection = self._connect_to_decode(session_id)
        if connection is None:
            logger.warning(
                "No decode control channel for session=%s room=%s", session_id, room
            )
            return

        sock, lock = connection
        logger.info(
            "shm_pinned sending FAIL to decode: session=%s room=%s reason=%s",
            session_id,
            room,
            reason,
        )
        with lock:
            sock.send_multipart(
                [
                    b"FAIL",
                    str(room).encode("utf-8"),
                    session_id.encode("utf-8"),
                    reason.encode("utf-8"),
                ]
            )

    def _prefill_transfer_loop(self) -> None:
        while self._prefill_thread_running:
            try:
                msg = self.server_socket.recv_multipart(flags=zmq.NOBLOCK)
            except zmq.Again:
                time.sleep(0.05)
                continue
            except Exception as e:
                if self._prefill_thread_running:
                    logger.error("Prefill transfer loop error: %s", e)
                    time.sleep(0.1)
                continue

            try:
                msg_type = msg[0].decode("utf-8")
                if msg_type == "SHM_INFO":
                    session_id = msg[1].decode("utf-8")
                    shm_info = ShmPinnedInfo.from_dict(
                        json.loads(msg[2].decode("utf-8"))
                    )
                    self._handle_shm_info(session_id, shm_info)
                elif msg_type == "TRANSFER_REQ":
                    room = int(msg[1].decode("utf-8"))
                    session_id = msg[2].decode("utf-8")
                    dst_kv_indices = np.frombuffer(msg[3], dtype=np.int32).copy()
                    self._handle_transfer_request(room, session_id, dst_kv_indices)
                elif msg_type == "ABORT":
                    room = int(msg[1].decode("utf-8"))
                    session_id = msg[2].decode("utf-8") if len(msg) > 2 else ""
                    self._handle_abort(room, session_id)
            except Exception as e:
                logger.error("Failed to handle prefill control message: %s", e)

    def _drain_decode_control_messages(self) -> None:
        while True:
            try:
                msg = self.server_socket.recv_multipart(flags=zmq.NOBLOCK)
            except zmq.Again:
                return
            except Exception as e:
                if self._decode_thread_running:
                    logger.error("Decode control loop error: %s", e)
                return

            try:
                msg_type = msg[0].decode("utf-8")
                if msg_type not in {"FAIL", "ABORT"}:
                    continue
                room = int(msg[1].decode("utf-8"))
                session_id = msg[2].decode("utf-8")
                reason = (
                    msg[3].decode("utf-8")
                    if len(msg) > 3
                    else "Remote transfer failed"
                )
                self.record_failure(room, reason)
                self.update_status(room, KVPoll.Failed)
                self._pop_pending_request(make_request_key(session_id, room))
            except Exception as e:
                logger.error("Failed to handle decode control message: %s", e)

    def _decode_receive_loop(self) -> None:
        while self._decode_thread_running:
            self._drain_decode_control_messages()

            if self.engine is None:
                time.sleep(0.05)
                continue

            slot_idx = None
            slot_released = False
            try:
                slot_idx = self.engine.wait_ready(timeout=0.5)
                meta = self.engine.read_meta(slot_idx)
                key = make_request_key(self.engine.session_id, meta.room)
                req = self.get_pending_request(self.engine.session_id, meta.room)
                logger.debug(
                    "shm_pinned decode received slot: session=%s room=%s slot=%s "
                    "index_start=%s index_len=%s is_last=%s seqno=%s",
                    self.engine.session_id,
                    meta.room,
                    slot_idx,
                    meta.index_start,
                    meta.index_len,
                    meta.is_last,
                    meta.seqno,
                )
                if req is None:
                    logger.warning(
                        "shm_pinned decode missing pending request: session=%s room=%s",
                        self.engine.session_id,
                        meta.room,
                    )
                    self.engine.post_free(slot_idx)
                    slot_released = True
                    continue

                start_idx = meta.index_start
                end_idx = meta.index_start + meta.index_len
                total_pages = req.total_pages or len(req.dst_kv_indices)

                def fail(reason: str) -> None:
                    nonlocal slot_released
                    self.record_failure(meta.room, reason)
                    self.update_status(meta.room, KVPoll.Failed)
                    self._pop_pending_request(key)
                    if not slot_released:
                        self.engine.post_free(slot_idx)
                        slot_released = True

                if end_idx > total_pages:
                    fail("Decode received out-of-range dst indices")
                    continue
                if meta.is_last and end_idx != total_pages:
                    fail("Decode received last chunk not aligned to total pages")
                    continue

                if req.received_bitmap is None:
                    req.received_bitmap = bytearray(total_pages)
                if any(req.received_bitmap[i] for i in range(start_idx, end_idx)):
                    fail("Decode received duplicate pages")
                    continue

                dst_indices = req.dst_kv_indices[start_idx:end_idx]
                kv_bytes = self._calculate_chunk_bytes(meta.index_len)
                self._do_h2d_copy(slot_idx, dst_indices)

                if meta.is_last and self._calculate_aux_bytes() > 0:
                    self._copy_aux_from_slot(slot_idx, kv_bytes, req.dst_aux_index)

                self.engine.post_free(slot_idx)
                slot_released = True

                for i in range(start_idx, end_idx):
                    req.received_bitmap[i] = 1
                req.received_pages += end_idx - start_idx
                if meta.is_last:
                    req.last_chunk_seen = True

                if req.received_pages == total_pages and req.last_chunk_seen:
                    self._pop_pending_request(key)
                    self.update_status(meta.room, KVPoll.Success)
                    logger.debug(
                        "shm_pinned decode transfer complete: session=%s room=%s total_pages=%s",
                        self.engine.session_id,
                        meta.room,
                        total_pages,
                    )
            except Exception as e:
                if self._decode_thread_running and slot_idx is not None:
                    logger.error("Decode receive loop error: %s", e)
            finally:
                if slot_idx is not None and not slot_released and self.engine is not None:
                    try:
                        self.engine.post_free(slot_idx)
                    except Exception:
                        pass

    def _handle_shm_info(self, session_id: str, shm_info: ShmPinnedInfo) -> None:
        if session_id in self.session_engines:
            return
        logger.info(
            "shm_pinned prefill opening shared memory: session=%s slot_count=%s slot_bytes=%s",
            session_id,
            shm_info.slot_count,
            shm_info.slot_bytes,
        )
        engine = ShmPinnedTransferEngine(
            session_id=session_id,
            gpu_id=self.kv_args.gpu_id,
            create=False,
        )
        engine.open_from_info(shm_info)
        self.session_shm_table[session_id] = shm_info
        self.session_engines[session_id] = engine

    def _handle_transfer_request(
        self,
        room: int,
        session_id: str,
        dst_kv_indices: npt.NDArray[np.int32],
    ) -> None:
        engine = self.session_engines.get(session_id)
        if engine is None:
            reason = f"Missing SHM engine for session {session_id}"
            self.record_failure(room, reason)
            self.update_status(room, KVPoll.Failed)
            self._send_failure_to_decode(session_id, room, reason)
            return

        logger.debug(
            "shm_pinned prefill received TRANSFER_REQ: session=%s room=%s total_pages=%s",
            session_id,
            room,
            len(dst_kv_indices),
        )
        self._add_pending_request(
            TransferRequest(
                room=room,
                session_id=session_id,
                dst_kv_indices=dst_kv_indices,
                engine=engine,
                total_pages=len(dst_kv_indices),
            )
        )
        self.update_status(room, KVPoll.WaitingForInput)

    def _handle_abort(self, room: int, session_id: str) -> None:
        logger.info(
            "shm_pinned received ABORT: session=%s room=%s",
            session_id,
            room,
        )
        if session_id:
            self._pop_pending_request(make_request_key(session_id, room))
        self.update_status(room, KVPoll.Failed)

    def _do_h2d_copy(
        self,
        slot_idx: int,
        dst_indices: npt.NDArray[np.int32],
    ) -> None:
        if self.engine is None or dst_indices.size == 0:
            return

        slot_ptr = self.engine.get_slot_data_ptr(slot_idx)
        offset = 0
        contiguous = np.all(np.diff(dst_indices) == 1)

        for buf_ptr, item_len in zip(
            self.kv_args.kv_data_ptrs,
            self.kv_args.kv_item_lens,
        ):
            item_len = int(item_len)
            if contiguous:
                start_page = int(dst_indices[0])
                total_bytes = item_len * int(dst_indices.size)
                self.engine.cuda_memcpy(
                    dst_ptr=buf_ptr + start_page * item_len,
                    src_ptr=slot_ptr + offset,
                    num_bytes=total_bytes,
                )
                offset += total_bytes
            else:
                for j, page_idx in enumerate(dst_indices.tolist()):
                    self.engine.cuda_memcpy(
                        dst_ptr=buf_ptr + int(page_idx) * item_len,
                        src_ptr=slot_ptr + offset + j * item_len,
                        num_bytes=item_len,
                    )
                offset += item_len * int(dst_indices.size)

    def _calculate_chunk_bytes(self, num_indices: int) -> int:
        return int(num_indices) * int(sum(self.kv_args.kv_item_lens))

    def _calculate_aux_bytes(self) -> int:
        aux_item_lens = getattr(self.kv_args, "aux_item_lens", None) or []
        return int(sum(aux_item_lens))

    def _copy_aux_from_slot(
        self,
        slot_idx: int,
        kv_bytes: int,
        aux_index: Optional[int],
    ) -> None:
        if self.engine is None:
            raise RuntimeError("Transfer engine not initialized")
        if aux_index is None:
            raise RuntimeError("Aux index is required for aux copy")

        slot_ptr = self.engine.get_slot_data_ptr(slot_idx)
        offset = int(kv_bytes)
        for aux_ptr, item_len in zip(
            self.kv_args.aux_data_ptrs,
            self.kv_args.aux_item_lens,
        ):
            item_len = int(item_len)
            ctypes.memmove(
                aux_ptr + item_len * int(aux_index),
                slot_ptr + offset,
                item_len,
            )
            offset += item_len

    def close(self) -> None:
        if hasattr(self, "_prefill_thread_running"):
            self._prefill_thread_running = False
        if hasattr(self, "_decode_thread_running"):
            self._decode_thread_running = False
        if hasattr(self, "_prefill_thread"):
            self._prefill_thread.join(timeout=1.0)
        if hasattr(self, "_decode_thread"):
            self._decode_thread.join(timeout=1.0)
        if self.engine is not None:
            self.engine.close()
        for engine in self.session_engines.values():
            if engine is not self.engine:
                engine.close()


class ShmPinnedKVSender(CommonKVSender):
    def __init__(
        self,
        mgr: ShmPinnedKVManager,
        bootstrap_addr: str,
        bootstrap_room: int,
        dest_tp_ranks: List[int],
        pp_rank: int,
    ):
        super().__init__(mgr, bootstrap_addr, bootstrap_room, dest_tp_ranks, pp_rank)
        self.kv_mgr: ShmPinnedKVManager = mgr
        self.dest_tp_ranks = dest_tp_ranks
        self.pp_rank = pp_rank
        self.session_id: Optional[str] = None
        self.session_engine: Optional[ShmPinnedTransferEngine] = None
        self.dst_kv_indices: Optional[npt.NDArray[np.int32]] = None
        self.sent_chunks = 0

    def init(self, num_kv_indices: int, aux_index: Optional[int] = None) -> None:
        super().init(num_kv_indices, aux_index)

    def _mark_failed(self, reason: str) -> None:
        self.kv_mgr.record_failure(self.bootstrap_room, reason)
        self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Failed)
        if self.session_id is not None:
            self.kv_mgr._pop_pending_request(
                make_request_key(self.session_id, self.bootstrap_room)
            )
            self.kv_mgr._send_failure_to_decode(
                self.session_id,
                self.bootstrap_room,
                reason,
            )

    def _ensure_session_ready(self) -> bool:
        if self.session_engine is not None and self.dst_kv_indices is not None:
            return True

        req = (
            self.kv_mgr.get_pending_request(self.session_id, self.bootstrap_room)
            if self.session_id is not None
            else self.kv_mgr.get_pending_request_for_room(self.bootstrap_room)
        )
        if req is None:
            return False

        self.session_id = req.session_id
        if req.engine is None:
            req.engine = self.kv_mgr.session_engines.get(req.session_id)
        self.session_engine = req.engine
        self.dst_kv_indices = req.dst_kv_indices
        logger.debug(
            "shm_pinned sender attached session: session=%s room=%s total_pages=%s",
            self.session_id,
            self.bootstrap_room,
            len(self.dst_kv_indices),
        )
        return self.session_engine is not None and self.dst_kv_indices is not None

    def send(
        self,
        kv_indices: npt.NDArray[np.int32],
        state_indices: Optional[List[int]] = None,
    ) -> None:
        if state_indices is not None:
            self._mark_failed("shm_pinned PR1 does not support state transfer")
            return
        if self.kv_mgr.is_dummy_cp_rank:
            return

        try:
            if not self._ensure_session_ready():
                self._mark_failed("Session engine not initialized")
                return
            assert self.dst_kv_indices is not None

            logger.debug(
                "shm_pinned sender send start: session=%s room=%s curr_idx=%s chunk_pages=%s len(kv_indices)=%s",
                self.session_id,
                self.bootstrap_room,
                self.curr_idx,
                self.kv_mgr.chunk_pages,
                len(kv_indices),
            )

            if self.curr_idx + len(kv_indices) > len(self.dst_kv_indices):
                self._mark_failed("KV indices exceed destination buffer length")
                return

            src_groups, dst_groups = group_concurrent_contiguous(
                kv_indices.astype(np.int32),
                self.dst_kv_indices[self.curr_idx : self.curr_idx + len(kv_indices)],
            )

            if self.curr_idx == 0:
                self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Transferring)

            is_last = (self.curr_idx + len(kv_indices)) >= self.num_kv_indices
            if is_last and self._calculate_aux_bytes() > 0 and self.aux_index is None:
                self._mark_failed("Missing aux index for last chunk")
                return

            chunk_start = self.curr_idx
            max_pages = self.kv_mgr.chunk_pages
            for group_idx, (src_chunk, dst_chunk) in enumerate(
                zip(src_groups, dst_groups)
            ):
                group_len = len(src_chunk)
                for offset in range(0, group_len, max_pages):
                    sub_src = np.array(
                        src_chunk[offset : offset + max_pages],
                        dtype=np.int32,
                    )
                    sub_dst = np.array(
                        dst_chunk[offset : offset + max_pages],
                        dtype=np.int32,
                    )
                    is_last_sub = (
                        is_last
                        and group_idx == len(src_groups) - 1
                        and offset + len(sub_src) == group_len
                    )
                    self._send_chunk(
                        src_indices=sub_src,
                        dst_indices=sub_dst,
                        index_start=chunk_start + offset,
                        is_last=is_last_sub,
                    )
                chunk_start += group_len

            self.curr_idx += len(kv_indices)
            if is_last and self.session_id is not None:
                self.kv_mgr._pop_pending_request(
                    make_request_key(self.session_id, self.bootstrap_room)
                )
                self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Success)
                logger.debug(
                    "shm_pinned sender transfer complete: session=%s room=%s total_pages=%s",
                    self.session_id,
                    self.bootstrap_room,
                    self.curr_idx,
                )
        except Exception as e:
            self._mark_failed(str(e))

    def _send_chunk(
        self,
        src_indices: npt.NDArray[np.int32],
        dst_indices: npt.NDArray[np.int32],
        index_start: int,
        is_last: bool,
    ) -> None:
        if self.session_engine is None:
            raise RuntimeError("Session engine not initialized")

        slot_idx = None
        try:
            logger.debug(
                "shm_pinned sender waiting for free slot: session=%s room=%s index_start=%s index_len=%s is_last=%s",
                self.session_id,
                self.bootstrap_room,
                index_start,
                len(src_indices),
                is_last,
            )
            slot_idx = self.session_engine.wait_free()
            kv_bytes = self._calculate_chunk_bytes(len(src_indices))
            aux_bytes = self._calculate_aux_bytes()
            valid_bytes = kv_bytes + (aux_bytes if is_last else 0)
            if valid_bytes > self.session_engine.slot_bytes:
                raise RuntimeError(
                    f"Chunk bytes {valid_bytes} exceed slot size {self.session_engine.slot_bytes}"
                )

            self.session_engine.write_meta(
                slot_idx=slot_idx,
                room=self.bootstrap_room,
                index_start=index_start,
                index_len=len(src_indices),
                is_last=is_last,
                valid_bytes=valid_bytes,
                seqno=self.sent_chunks,
            )
            self._do_d2h_copy(slot_idx, src_indices)

            if is_last and aux_bytes > 0:
                self._copy_aux_to_slot(slot_idx, kv_bytes, self.aux_index)

            self.session_engine.post_ready(slot_idx)
            self.sent_chunks += 1
            logger.debug(
                "shm_pinned sender posted ready slot: session=%s room=%s slot=%s index_start=%s index_len=%s is_last=%s seqno=%s",
                self.session_id,
                self.bootstrap_room,
                slot_idx,
                index_start,
                len(src_indices),
                is_last,
                self.sent_chunks - 1,
            )
        except Exception:
            if slot_idx is not None:
                try:
                    self.session_engine.post_free(slot_idx)
                except Exception:
                    pass
            raise

    def _do_d2h_copy(
        self,
        slot_idx: int,
        src_indices: npt.NDArray[np.int32],
    ) -> None:
        if self.session_engine is None or src_indices.size == 0:
            return

        slot_ptr = self.session_engine.get_slot_data_ptr(slot_idx)
        offset = 0
        contiguous = np.all(np.diff(src_indices) == 1)

        for buf_ptr, item_len in zip(
            self.kv_mgr.kv_args.kv_data_ptrs,
            self.kv_mgr.kv_args.kv_item_lens,
        ):
            item_len = int(item_len)
            if contiguous:
                start_page = int(src_indices[0])
                total_bytes = item_len * int(src_indices.size)
                self.session_engine.cuda_memcpy(
                    dst_ptr=slot_ptr + offset,
                    src_ptr=buf_ptr + start_page * item_len,
                    num_bytes=total_bytes,
                )
                offset += total_bytes
            else:
                for j, page_idx in enumerate(src_indices.tolist()):
                    self.session_engine.cuda_memcpy(
                        dst_ptr=slot_ptr + offset + j * item_len,
                        src_ptr=buf_ptr + int(page_idx) * item_len,
                        num_bytes=item_len,
                    )
                offset += item_len * int(src_indices.size)

    def _calculate_chunk_bytes(self, num_indices: int) -> int:
        return int(num_indices) * int(sum(self.kv_mgr.kv_args.kv_item_lens))

    def _calculate_aux_bytes(self) -> int:
        return int(sum(getattr(self.kv_mgr.kv_args, "aux_item_lens", []) or []))

    def _copy_aux_to_slot(
        self,
        slot_idx: int,
        kv_bytes: int,
        aux_index: Optional[int],
    ) -> None:
        if self.session_engine is None:
            raise RuntimeError("Session engine not initialized")
        if aux_index is None:
            raise RuntimeError("Aux index is required for aux copy")

        slot_ptr = self.session_engine.get_slot_data_ptr(slot_idx)
        offset = int(kv_bytes)
        for aux_ptr, item_len in zip(
            self.kv_mgr.kv_args.aux_data_ptrs,
            self.kv_mgr.kv_args.aux_item_lens,
        ):
            item_len = int(item_len)
            ctypes.memmove(
                slot_ptr + offset,
                aux_ptr + item_len * int(aux_index),
                item_len,
            )
            offset += item_len

    def poll(self) -> KVPoll:
        return self.kv_mgr.get_status(self.bootstrap_room)

    def failure_exception(self) -> Exception:
        message = self.kv_mgr.get_failure_message(self.bootstrap_room)
        return RuntimeError(f"ShmPinned transfer failed: {message}")


class ShmPinnedKVReceiver(CommonKVReceiver):
    def __init__(
        self,
        mgr: ShmPinnedKVManager,
        bootstrap_addr: str,
        bootstrap_room: Optional[int] = None,
    ):
        super().__init__(mgr, bootstrap_addr, bootstrap_room)
        self.kv_mgr: ShmPinnedKVManager = mgr
        self.kv_indices: Optional[npt.NDArray[np.int32]] = None
        self.aux_index: Optional[int] = None
        self.session_id: Optional[str] = None
        self.transfer_timeout = 300.0
        self.transfer_start_time: Optional[float] = None

    def init(self, prefill_dp_rank: int) -> None:
        super().init(prefill_dp_rank)

    def send_metadata(
        self,
        kv_indices: npt.NDArray[np.int32],
        aux_index: Optional[int] = None,
        state_indices: Optional[List[int]] = None,
    ) -> None:
        if state_indices is not None:
            self.kv_mgr.record_failure(
                self.bootstrap_room,
                "shm_pinned PR1 does not support state transfer",
            )
            self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Failed)
            return
        if self.kv_mgr.engine is None:
            self.kv_mgr.record_failure(self.bootstrap_room, "Engine not initialized")
            self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Failed)
            return

        self.kv_indices = kv_indices
        self.aux_index = aux_index
        self.session_id = self.kv_mgr.engine.session_id
        self.transfer_start_time = time.time()
        total_pages = int(len(kv_indices))
        logger.debug(
            "shm_pinned receiver init: session=%s room=%s total_pages=%s aux_index=%s",
            self.session_id,
            self.bootstrap_room,
            total_pages,
            aux_index,
        )

        self.kv_mgr._add_pending_request(
            TransferRequest(
                room=self.bootstrap_room,
                session_id=self.session_id,
                dst_kv_indices=kv_indices,
                dst_aux_index=aux_index,
                total_pages=total_pages,
                received_bitmap=bytearray(total_pages),
            )
        )

        self._send_shm_info_to_prefill()
        self._send_transfer_req_to_prefill()
        self.kv_mgr.update_status(self.bootstrap_room, KVPoll.WaitingForInput)

    def _send_shm_info_to_prefill(self) -> None:
        if self.kv_mgr.engine is None or self.bootstrap_infos is None:
            return

        shm_info = self.kv_mgr.engine.export_info()
        shm_info.decode_host = self.kv_mgr.local_ip
        shm_info.decode_port = self.kv_mgr.rank_port
        logger.debug(
            "shm_pinned receiver sending SHM_INFO: session=%s room=%s decode=%s:%s",
            shm_info.session_id,
            self.bootstrap_room,
            shm_info.decode_host,
            shm_info.decode_port,
        )

        for bootstrap_info in self.bootstrap_infos:
            if bootstrap_info.get("is_dummy"):
                continue
            sock, lock = self._connect_to_bootstrap_server(bootstrap_info)
            with lock:
                sock.send_multipart(
                    [
                        b"SHM_INFO",
                        shm_info.session_id.encode("utf-8"),
                        json.dumps(shm_info.to_dict()).encode("utf-8"),
                    ]
                )

    def _send_transfer_req_to_prefill(self) -> None:
        if self.kv_indices is None or self.bootstrap_infos is None or self.session_id is None:
            return

        payload = self.kv_indices.astype(np.int32).tobytes()
        logger.debug(
            "shm_pinned receiver sending TRANSFER_REQ: session=%s room=%s total_pages=%s",
            self.session_id,
            self.bootstrap_room,
            len(self.kv_indices),
        )
        for bootstrap_info in self.bootstrap_infos:
            if bootstrap_info.get("is_dummy"):
                continue
            sock, lock = self._connect_to_bootstrap_server(bootstrap_info)
            with lock:
                sock.send_multipart(
                    [
                        b"TRANSFER_REQ",
                        str(self.bootstrap_room).encode("utf-8"),
                        self.session_id.encode("utf-8"),
                        payload,
                    ]
                )

    def poll(self) -> KVPoll:
        status = self.kv_mgr.get_status(self.bootstrap_room)
        if status == KVPoll.Bootstrapping and self.kv_indices is None:
            return KVPoll.WaitingForInput
        if status in (KVPoll.WaitingForInput, KVPoll.Transferring):
            if self.transfer_start_time is not None:
                elapsed = time.time() - self.transfer_start_time
                if elapsed > self.transfer_timeout:
                    reason = f"Transfer timeout after {elapsed:.1f}s"
                    self.kv_mgr.record_failure(self.bootstrap_room, reason)
                    self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Failed)
                    self.abort()
                    return KVPoll.Failed
        return status

    def failure_exception(self) -> Exception:
        message = self.kv_mgr.get_failure_message(self.bootstrap_room)
        return RuntimeError(f"ShmPinned receive failed: {message}")

    def clear(self) -> None:
        if self.session_id is None:
            return
        self.kv_mgr._pop_pending_request(
            make_request_key(self.session_id, self.bootstrap_room)
        )

    def abort(self) -> None:
        self.clear()
        self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Failed)
        if self.bootstrap_infos is None or self.session_id is None:
            return
        for bootstrap_info in self.bootstrap_infos:
            if bootstrap_info.get("is_dummy"):
                continue
            try:
                sock, lock = self._connect_to_bootstrap_server(bootstrap_info)
                with lock:
                    sock.send_multipart(
                        [
                            b"ABORT",
                            str(self.bootstrap_room).encode("utf-8"),
                            self.session_id.encode("utf-8"),
                        ]
                    )
            except Exception as e:
                logger.warning("Failed to send abort to prefill: %s", e)


class ShmPinnedKVBootstrapServer(CommonKVBootstrapServer):
    pass
