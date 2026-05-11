from __future__ import annotations

import dataclasses
import logging
import os
import struct
import threading
import time
import uuid
from typing import Dict, List, Optional, Tuple

import msgspec
import numpy as np
import numpy.typing as npt
import zmq
from mori.cpp import TransferStatus
from mori.io import (
    BackendType,
    EngineDesc,
    IOEngine,
    IOEngineConfig,
    MemoryDesc,
    MemoryLocationType,
    PollCqMode,
    RdmaBackendConfig,
    StatusCode,
)

from sglang.srt.disaggregation.base.conn import KVArgs, KVPoll
from sglang.srt.disaggregation.common.conn import (
    CommonKVBootstrapServer,
    CommonKVManager,
    CommonKVReceiver,
    CommonKVSender,
    KVTransferError,
)
from sglang.srt.disaggregation.common.utils import (
    AuxDataCodec,
    FastQueue,
    group_concurrent_contiguous,
    pack_int_lists,
    unpack_int_lists,
)
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.environ import envs
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils.network import NetworkAddress, get_local_ip_auto

logger = logging.getLogger(__name__)
MORI_GUARD = b"MoriMsgGuard"


def _normalize_state_indices_per_component(
    state_indices: Optional[List],
) -> Optional[List[Optional[npt.NDArray[np.int32]]]]:
    if state_indices is None:
        return None
    out: List[Optional[npt.NDArray[np.int32]]] = []
    for entry in state_indices:
        if entry is None:
            out.append(None)
        else:
            out.append(np.asarray(entry, dtype=np.int32).ravel())
    return out


def _pack_state_indices(
    state_indices: Optional[List[Optional[npt.NDArray[np.int32]]]],
) -> bytes:
    if not state_indices:
        return b""
    lists = [(arr.tolist() if arr is not None else []) for arr in state_indices]
    return pack_int_lists(lists, "i")


def _unpack_state_indices(buf: bytes) -> List[npt.NDArray[np.int32]]:
    if not buf:
        return []
    return [np.asarray(lst, dtype=np.int32) for lst in unpack_int_lists(buf, "i")]


def _pack_mem_desc_list(mems: List[MemoryDesc]) -> bytes:
    if not mems:
        return b""
    packed_descs = [mem.pack() for mem in mems]
    return msgspec.msgpack.encode(packed_descs)


def _unpack_mem_desc_list(blob: bytes) -> List[MemoryDesc]:
    if not blob:
        return []
    desc_blobs = msgspec.msgpack.decode(blob)
    return [MemoryDesc.unpack(b) for b in desc_blobs]


def _pack_mem_desc_lists(mems_per_comp: List[List[MemoryDesc]]) -> bytes:
    if not mems_per_comp:
        return b""
    return msgspec.msgpack.encode(
        [[mem.pack() for mem in comp] for comp in mems_per_comp]
    )


def _unpack_mem_desc_lists(blob: bytes) -> List[List[MemoryDesc]]:
    if not blob:
        return []
    nested = msgspec.msgpack.decode(blob)
    return [[MemoryDesc.unpack(b) for b in comp] for comp in nested]


@dataclasses.dataclass
class TransferInfo:
    room: int
    endpoint: str
    dst_port: int
    engine_key: str
    dst_kv_indices: npt.NDArray[np.int32]
    dst_aux_index: int
    dst_state_indices: List[npt.NDArray[np.int32]]
    required_dst_info_num: int
    is_dummy: bool
    # Number of tokens decode already holds in its radix cache; prefill should
    # only send pages beyond this prefix. None means the receiver did not
    # populate this field (older receiver or radix-cache feature off) -> treat
    # as 0 (no prefix hit, full send) for backward compatibility.
    decode_prefix_len: Optional[int] = None

    @classmethod
    def from_zmq(cls, payload: List[bytes]) -> TransferInfo:
        room = int(payload[0].decode("ascii"))
        endpoint = payload[1].decode("ascii")
        dst_port = int(payload[2].decode("ascii"))
        engine_key = payload[3].decode("ascii")

        if payload[4]:
            dst_kv_indices = np.frombuffer(payload[4], dtype=np.int32)
        else:
            dst_kv_indices = np.array([], dtype=np.int32)

        if payload[5]:
            dst_aux_index = int(payload[5].decode("ascii"))
        else:
            dst_aux_index = -1

        if len(payload) > 6 and payload[6]:
            dst_state_indices = _unpack_state_indices(payload[6])
        else:
            dst_state_indices = []

        required_dst_info_num = (
            int(payload[7].decode("ascii")) if len(payload) > 7 else 1
        )

        if len(payload) > 8 and payload[8]:
            decode_prefix_len: Optional[int] = int(payload[8].decode("ascii"))
        else:
            decode_prefix_len = None

        # A transfer is "dummy" only when the receiver does not need any
        # kv/aux/state delivered. When decode_prefix_len > 0 and the delta is
        # exactly zero (full prefix hit), dst_kv_indices is empty but aux is
        # still needed -> not dummy.
        is_dummy = (
            dst_kv_indices.size == 0 and dst_aux_index < 0 and not decode_prefix_len
        )
        return cls(
            room=room,
            endpoint=endpoint,
            dst_port=dst_port,
            engine_key=engine_key,
            dst_kv_indices=dst_kv_indices,
            dst_aux_index=dst_aux_index,
            dst_state_indices=dst_state_indices,
            required_dst_info_num=required_dst_info_num,
            is_dummy=is_dummy,
            decode_prefix_len=decode_prefix_len,
        )


@dataclasses.dataclass
class KVArgsRegisterInfo:
    endpoint: str
    dst_port: int
    engine_desc: EngineDesc
    dst_kv_mem_descs: List[MemoryDesc]
    dst_aux_mem_descs: List[MemoryDesc]
    dst_state_mem_descs: List[List[MemoryDesc]]
    gpu_id: int
    decode_tp_size: int
    decode_tp_rank: int
    dst_kv_item_len: int
    dst_state_item_lens: List[List[int]]
    dst_state_dim_per_tensor: List[List[int]]
    dst_kv_item_lens: List[int]

    @property
    def engine_key(self) -> str:
        return self.engine_desc.key

    @classmethod
    def from_zmq(cls, payload: List[bytes]) -> KVArgsRegisterInfo:
        endpoint = payload[1].decode("ascii")
        dst_port = int(payload[2].decode("ascii"))
        engine_desc = EngineDesc.unpack(payload[3])
        dst_kv_mem_descs = _unpack_mem_desc_list(payload[4])
        dst_aux_mem_descs = _unpack_mem_desc_list(payload[5])
        dst_state_mem_descs = _unpack_mem_desc_lists(payload[6])
        gpu_id = int(payload[7].decode("ascii"))
        decode_tp_size = int(payload[8].decode("ascii"))
        decode_tp_rank = int(payload[9].decode("ascii"))
        dst_kv_item_len = int(payload[10].decode("ascii"))
        dst_state_item_lens = (
            unpack_int_lists(payload[11], "I")
            if len(payload) > 11 and payload[11]
            else []
        )
        dst_state_dim_per_tensor = (
            unpack_int_lists(payload[12], "I")
            if len(payload) > 12 and payload[12]
            else []
        )
        # Per-layer KV item lens (payload[13]); falls back to broadcasting
        # the scalar dst_kv_item_len across every dst KV descriptor when
        # the sender is older and does not include this slot.
        dst_kv_item_lens = (
            list(struct.unpack(f"{len(payload[13]) // 4}I", payload[13]))
            if len(payload) > 13 and len(payload[13]) > 0
            else [dst_kv_item_len] * len(dst_kv_mem_descs)
        )
        return cls(
            endpoint=endpoint,
            dst_port=dst_port,
            engine_desc=engine_desc,
            dst_kv_mem_descs=dst_kv_mem_descs,
            dst_aux_mem_descs=dst_aux_mem_descs,
            dst_state_mem_descs=dst_state_mem_descs,
            gpu_id=gpu_id,
            decode_tp_size=decode_tp_size,
            decode_tp_rank=decode_tp_rank,
            dst_kv_item_len=dst_kv_item_len,
            dst_state_item_lens=dst_state_item_lens,
            dst_state_dim_per_tensor=dst_state_dim_per_tensor,
            dst_kv_item_lens=dst_kv_item_lens,
        )


@dataclasses.dataclass
class TPSliceConfig:
    page_size: int
    src_item_len: int
    dst_item_len: int
    bytes_per_token_src: int
    bytes_per_token_dst: int
    src_head_slice_offset: int
    dst_head_slice_offset: int
    heads_bytes_per_token_to_send: int


@dataclasses.dataclass(frozen=True)
class GroupedIndexPlan:
    src_starts: List[int]
    dst_starts: List[int]
    counts: List[int]

    @classmethod
    def from_groups(
        cls, src_groups: List[List[int]], dst_groups: List[List[int]]
    ) -> GroupedIndexPlan:
        if len(src_groups) != len(dst_groups):
            raise ValueError("Source and destination groups must have the same length")
        return cls(
            src_starts=[int(group[0]) for group in src_groups],
            dst_starts=[int(group[0]) for group in dst_groups],
            counts=[len(group) for group in src_groups],
        )

    def materialize(self, item_len: int) -> BatchTransferPlan:
        return BatchTransferPlan(
            local_offsets=[start * item_len for start in self.src_starts],
            remote_offsets=[start * item_len for start in self.dst_starts],
            sizes=[count * item_len for count in self.counts],
        )


@dataclasses.dataclass(frozen=True)
class BatchTransferPlan:
    local_offsets: List[int]
    remote_offsets: List[int]
    sizes: List[int]

    def empty(self) -> bool:
        return not self.sizes


@dataclasses.dataclass(frozen=True)
class TransferTarget:
    info: TransferInfo
    peer_info: KVArgsRegisterInfo


@dataclasses.dataclass
class _TransferChunk:
    sender: MoriKVSender
    kv_indices: npt.NDArray[np.int32]
    index_slice: slice
    is_last_chunk: bool
    aux_index: Optional[int]
    normalized_state: Optional[List[Optional[npt.NDArray[np.int32]]]]


class MoriKVManager(CommonKVManager):
    AUX_DATA_HEADER = b"AUX_DATA"

    def __init__(
        self,
        args: KVArgs,
        disaggregation_mode: DisaggregationMode,
        server_args: ServerArgs,
        is_mla_backend: Optional[bool] = False,
    ):
        super().__init__(args, disaggregation_mode, server_args, is_mla_backend)
        self.engine = self._init_engine()
        self.engine_desc = self.engine.get_engine_desc()
        self.kv_mem_descs: List[MemoryDesc] = []
        self.aux_mem_descs: List[MemoryDesc] = []
        self.state_mem_descs: List[List[MemoryDesc]] = []
        self.transfer_lock = threading.Lock()
        self._zmq_ctx = zmq.Context()
        self._socket_local = threading.local()
        self._send_aux_rdma = envs.SGLANG_MORI_SEND_AUX_RDMA.get()
        self._register_local_buffers()
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            self._num_shards = max(1, envs.SGLANG_MORI_TRANSFER_SHARDS.get())
            self._transfer_queues: List[FastQueue] = [
                FastQueue() for _ in range(self._num_shards)
            ]
            self._wait_poll_ms = envs.SGLANG_MORI_WAIT_POLL_MS.get()
            self._transfer_timeout_ms = envs.SGLANG_MORI_TRANSFER_TIMEOUT_MS.get()
            for shard, queue in enumerate(self._transfer_queues):
                threading.Thread(
                    target=self._transfer_worker,
                    args=(queue,),
                    daemon=True,
                    name=(
                        f"mori-xfer-dp{self.system_dp_rank}-"
                        f"tp{self.attn_tp_rank}-s{shard}"
                    ),
                ).start()
            self._start_bootstrap_thread()
        elif self.disaggregation_mode == DisaggregationMode.DECODE:
            self.room_to_bootstrap_addr: Dict[int, str] = {}
            self._start_decode_thread()

    def _init_engine(self) -> IOEngine:
        if self.kv_args.ib_device:
            os.environ["MORI_RDMA_DEVICES"] = self.kv_args.ib_device

        self.local_ip = get_local_ip_auto()
        config = IOEngineConfig(host=self.local_ip, port=0)

        engine_key = (
            f"io-{self.disaggregation_mode.value}-"
            f"dp{self.system_dp_rank}-tp{self.attn_tp_rank}-"
            f"pid{os.getpid()}-{self.local_ip}-"
            f"{uuid.uuid4().hex[:8]}"
        )

        engine = IOEngine(engine_key, config)
        poll_mode = PollCqMode.POLLING

        qp_per_transfer = envs.SGLANG_MORI_QP_PER_TRANSFER.get()
        post_batch_size = envs.SGLANG_MORI_POST_BATCH_SIZE.get()
        num_worker_threads = envs.SGLANG_MORI_NUM_WORKERS.get()

        rdma_cfg = RdmaBackendConfig(
            qp_per_transfer,
            post_batch_size,
            num_worker_threads,
            poll_mode,
            False,
        )
        engine.create_backend(BackendType.RDMA, rdma_cfg)
        actual_port = engine.get_engine_desc().port
        assert actual_port > 0, f"Failed to bind port for engine {engine_key}"
        logger.debug(
            "Initialized Mori IOEngine %s at %s:%s (qp_per_transfer=%s, workers=%s, poll_mode=%s)",
            engine_key,
            self.local_ip,
            actual_port,
            qp_per_transfer,
            num_worker_threads,
            poll_mode.name,
        )
        return engine

    def _register_local_buffers(self) -> None:
        for ptr, length in zip(self.kv_args.kv_data_ptrs, self.kv_args.kv_data_lens):
            mem_desc = self.engine.register_memory(
                ptr,
                length,
                self.kv_args.gpu_id,
                MemoryLocationType.GPU,
            )
            self.kv_mem_descs.append(mem_desc)
        for ptr, length in zip(self.kv_args.aux_data_ptrs, self.kv_args.aux_data_lens):
            desc = self.engine.register_memory(
                ptr,
                length,
                -1,
                MemoryLocationType.CPU,
            )
            self.aux_mem_descs.append(desc)
        for component_ptrs, component_lens in zip(
            self.kv_args.state_data_ptrs,
            getattr(self.kv_args, "state_data_lens", []),
        ):
            component_descs: List[MemoryDesc] = []
            for ptr, length in zip(component_ptrs, component_lens):
                desc = self.engine.register_memory(
                    ptr,
                    length,
                    self.kv_args.gpu_id,
                    MemoryLocationType.GPU,
                )
                component_descs.append(desc)
            self.state_mem_descs.append(component_descs)

    def update_status(self, bootstrap_room: int, status: KVPoll):
        current = self.request_status.get(bootstrap_room)
        if current is None:
            # Room not yet created or already cleared.
            # Only allow initial creation: Bootstrapping (normal) or
            # WaitingForInput (dummy CP rank, see CommonKVSender.__init__).
            if status not in (KVPoll.Bootstrapping, KVPoll.WaitingForInput):
                return
        elif current == KVPoll.Failed and status != KVPoll.Failed:
            # Failed is terminal — never overwrite with non-Failed.
            return
        super().update_status(bootstrap_room, status)

    def enqueue_transfer(self, task: _TransferChunk) -> None:
        self._transfer_queues[task.sender.bootstrap_room % self._num_shards].put(task)

    def _transfer_worker(self, queue: FastQueue) -> None:
        while True:
            task = queue.get()
            try:
                task.sender._run_chunk(task)
            except Exception as exc:
                failure_reason = f"transfer worker raised: {exc!r}"
                try:
                    logger.exception(
                        "Mori transfer worker failed for room %s",
                        task.sender.bootstrap_room,
                    )
                except Exception:
                    pass
                try:
                    task.sender._fail_from_worker(failure_reason)
                except Exception:
                    try:
                        logger.exception(
                            "Mori transfer worker failover failed for room %s",
                            task.sender.bootstrap_room,
                        )
                    except Exception:
                        pass

    def _connect_threadsafe(self, endpoint: str, is_ipv6: bool = False):
        """Thread-local ZMQ socket cache with shared Context.

        Each worker thread gets its own PUSH socket (ZMQ sockets are not
        thread-safe), but all sockets share a single process-level Context
        to avoid creating excessive I/O threads and TCP connections.
        """
        cache = getattr(self._socket_local, "socket_cache", None)
        if cache is None:
            cache = {}
            self._socket_local.socket_cache = cache
        if endpoint not in cache:
            sock = self._zmq_ctx.socket(zmq.PUSH)
            sock.setsockopt(zmq.SNDHWM, 0)
            sock.setsockopt(zmq.SNDTIMEO, 5000)
            sock.setsockopt(zmq.LINGER, 0)
            if is_ipv6:
                sock.setsockopt(zmq.IPV6, 1)
            sock.connect(endpoint)
            cache[endpoint] = sock
        return cache[endpoint]

    def _handle_register_message(self, payload: List[bytes]) -> None:
        try:
            register_info = KVArgsRegisterInfo.from_zmq(payload)
            self._add_remote_peer(register_info)
        except Exception:
            logger.exception("Failed to register remote peer")

    def _handle_transfer_message(self, payload: List[bytes]) -> None:
        try:
            transfer_info = TransferInfo.from_zmq(payload)
            with self.transfer_lock:
                # Accept metadata when room is not yet created (None) or
                # in Bootstrapping. Reject for active/terminal states where
                # the worker may already be using transfer_infos.
                # None is allowed because metadata can arrive from decode
                # before the prefill scheduler creates the MoriKVSender.
                current = self.request_status.get(transfer_info.room)
                if current is not None and current != KVPoll.Bootstrapping:
                    logger.debug(
                        "Ignoring stale transfer info for room %s (status=%s)",
                        transfer_info.room,
                        current,
                    )
                    return
                infos = self.transfer_infos.setdefault(transfer_info.room, {})
                infos[transfer_info.engine_key] = transfer_info

                if len(infos) >= transfer_info.required_dst_info_num:
                    # All decode peers reported their dst metadata; pick a
                    # non-None decode_prefix_len if any peer set it (they
                    # should all agree, but be defensive). 0 means "no
                    # prefix hit", which is the same as "feature off".
                    chosen_prefix_len = next(
                        (
                            info.decode_prefix_len
                            for info in infos.values()
                            if info.decode_prefix_len is not None
                        ),
                        0,
                    )
                    self.req_to_decode_prefix_len[transfer_info.room] = (
                        chosen_prefix_len
                    )
                    if chosen_prefix_len > 0:
                        # Surface incremental KV transfer at INFO so it's
                        # visible without bumping the global log level.
                        logger.info(
                            "MoriKV incremental: room=%s prefix_len=%s peers=%s",
                            transfer_info.room,
                            chosen_prefix_len,
                            len(infos),
                        )
                    else:
                        logger.debug(
                            "Bootstrap room %s got enough transfer info (%s), "
                            "decode_prefix_len=0",
                            transfer_info.room,
                            len(infos),
                        )
                    self.update_status(transfer_info.room, KVPoll.WaitingForInput)
        except Exception:
            logger.exception("Failed to parse transfer info message")

    def _validate_message(self, msg: List[bytes]) -> Optional[List[bytes]]:
        if not msg or msg[0] != MORI_GUARD:
            logger.warning("Received malformed bootstrap message")
            return None
        payload = msg[1:]
        if not payload:
            return None
        return payload

    def _start_bootstrap_thread(self) -> None:
        def bootstrap_worker():
            while True:
                try:
                    msg = self.server_socket.recv_multipart()
                    payload = self._validate_message(msg)
                    if payload is None:
                        continue
                    room = payload[0].decode("ascii")

                    if room == "None":
                        self._handle_register_message(payload)
                    else:
                        self._handle_transfer_message(payload)
                except Exception:
                    logger.exception("Bootstrap worker failed")

        threading.Thread(target=bootstrap_worker, daemon=True).start()

    def _cleanup_room_tracking(self, bootstrap_room: int) -> None:
        bootstrap_addr = self.room_to_bootstrap_addr.pop(bootstrap_room, None)
        if bootstrap_addr is not None:
            rooms = self.addr_to_rooms_tracker.get(bootstrap_addr)
            if rooms is not None:
                rooms.discard(bootstrap_room)
                if not rooms:
                    self.addr_to_rooms_tracker.pop(bootstrap_addr, None)

    def _start_decode_thread(self) -> None:
        def decode_worker():
            while True:
                try:
                    msg = self.server_socket.recv_multipart()
                    if msg and msg[0] == MoriKVManager.AUX_DATA_HEADER:
                        self._handle_aux_data(msg)
                        continue

                    if not msg or msg[0] != MORI_GUARD:
                        logger.warning(
                            "Received malformed status message on decode worker"
                        )
                        continue
                    payload = msg[1:]
                    if len(payload) < 3:
                        logger.warning("Incomplete status payload received")
                        continue
                    bootstrap_room = int(payload[0].decode("ascii"))
                    status_code = int(payload[1].decode("ascii"))
                    prefill_rank = int(payload[2].decode("ascii"))
                    failure_reason = (
                        payload[3].decode("utf-8")
                        if len(payload) > 3 and payload[3]
                        else None
                    )

                    if status_code == KVPoll.Success:
                        tracker = self.prefill_response_tracker[bootstrap_room]
                        tracker.add(prefill_rank)
                        expected = self.required_prefill_response_num_table.get(
                            bootstrap_room, 1
                        )
                        if len(tracker) >= expected:
                            self.prefill_response_tracker.pop(bootstrap_room, None)
                            self.update_status(bootstrap_room, KVPoll.Success)
                            self._cleanup_room_tracking(bootstrap_room)
                    elif status_code == KVPoll.Failed:
                        if failure_reason:
                            self.record_failure(bootstrap_room, failure_reason)
                        self.prefill_response_tracker.pop(bootstrap_room, None)
                        self.update_status(bootstrap_room, KVPoll.Failed)
                        self._cleanup_room_tracking(bootstrap_room)
                    else:
                        logger.warning(
                            "Unknown status code %s received for room %s",
                            status_code,
                            bootstrap_room,
                        )
                except Exception:
                    logger.exception("Decode status worker failed")

        threading.Thread(target=decode_worker, daemon=True).start()

    def _compute_prefill_unique_rank(self) -> int:
        """Unique id per prefill sender, encoding TP/PP/CP ranks.
        Must match Mooncake's formula so decode's response set size matches
        expected_response_num when multiple CP ranks participate."""
        return (
            self.attn_tp_rank * (self.pp_size * self.attn_cp_size)
            + self.pp_rank * self.attn_cp_size
            + self.attn_cp_rank
        )

    def notify_decode_status(
        self,
        infos: List[TransferInfo],
        bootstrap_room: int,
        status: KVPoll,
        failure_reason: Optional[str] = None,
    ) -> None:
        if not infos:
            return
        payload = [
            MORI_GUARD,
            str(bootstrap_room).encode("ascii"),
            str(int(status)).encode("ascii"),
            str(self._compute_prefill_unique_rank()).encode("ascii"),
            failure_reason.encode("utf-8") if failure_reason else b"",
        ]
        for info in infos:
            try:
                na = NetworkAddress(info.endpoint, info.dst_port)
                socket = self._connect_threadsafe(na.to_tcp(), is_ipv6=na.is_ipv6)
                socket.send_multipart(payload)
            except Exception:
                logger.exception(
                    "Failed to sync status %s to decode endpoint %s:%s for room %s",
                    status,
                    info.endpoint,
                    info.dst_port,
                    bootstrap_room,
                )

    def _add_remote_peer(self, register_info: KVArgsRegisterInfo) -> None:
        engine_key = register_info.engine_key
        if engine_key in self.decode_kv_args_table:
            logger.debug("Remote peer %s already registered. Skipping.", engine_key)
            return
        self.engine.register_remote_engine(register_info.engine_desc)
        self.decode_kv_args_table[engine_key] = register_info
        logger.debug(
            "Registered decode peer %s (%s:%s)",
            engine_key,
            register_info.endpoint,
            register_info.dst_port,
        )

    def _get_mha_mem_desc_slices(
        self, dst_mem_descs: List[MemoryDesc]
    ) -> tuple[
        List[MemoryDesc], List[MemoryDesc], List[MemoryDesc], List[MemoryDesc], int
    ]:
        src_descs = self.kv_mem_descs
        if not src_descs:
            raise RuntimeError("KV memory descriptors are empty on prefill side")

        num_local_layers = len(src_descs) // 2
        src_k_descs = src_descs[:num_local_layers]
        src_v_descs = src_descs[num_local_layers:]

        start_layer = self.kv_args.prefill_start_layer
        end_layer = start_layer + num_local_layers
        dst_total_layers = len(dst_mem_descs) // 2
        if len(dst_mem_descs) < 2 or end_layer > dst_total_layers:
            raise ValueError(
                "Destination KV descriptors do not match prefill pp configuration"
            )
        dst_k_descs = dst_mem_descs[start_layer:end_layer]
        dst_v_descs = dst_mem_descs[
            dst_total_layers + start_layer : dst_total_layers + end_layer
        ]
        return src_k_descs, src_v_descs, dst_k_descs, dst_v_descs, num_local_layers

    def _get_mla_mem_desc_slices(
        self, dst_mem_descs: List[MemoryDesc]
    ) -> tuple[List[MemoryDesc], List[MemoryDesc], int]:
        src_descs = self.kv_mem_descs
        num_local_layers = len(src_descs)
        start_layer = self.kv_args.prefill_start_layer
        end_layer = start_layer + num_local_layers
        if end_layer > len(dst_mem_descs):
            raise ValueError(
                "Destination MLA KV descriptors do not match prefill pp configuration"
            )
        dst_slice = dst_mem_descs[start_layer:end_layer]
        return src_descs, dst_slice, num_local_layers

    def _submit_batch_transfer_plan(
        self,
        src_desc: MemoryDesc,
        dst_desc: MemoryDesc,
        plan: BatchTransferPlan,
    ) -> List[TransferStatus]:
        if plan.empty():
            return []

        transfer_uid = self.engine.allocate_transfer_uid()

        statuses = self.engine.batch_write(
            [src_desc],
            [plan.local_offsets],
            [dst_desc],
            [plan.remote_offsets],
            [plan.sizes],
            [transfer_uid],
        )
        return statuses

    def _build_contiguous_transfer_plan(
        self, grouped_plan: GroupedIndexPlan, item_len: int
    ) -> BatchTransferPlan:
        # Reuse grouped indices across all layers/tensors that share the same item length.
        return grouped_plan.materialize(item_len)

    def _build_tp_slice_config(
        self,
        peer_info: KVArgsRegisterInfo,
        src_item_len: int,
        dst_item_len: int,
    ) -> TPSliceConfig:
        page_size = self.kv_args.page_size

        bytes_per_token_src = src_item_len // page_size
        bytes_per_token_dst = dst_item_len // page_size

        prefill_tp_size = self.attn_tp_size
        decode_tp_size = peer_info.decode_tp_size

        total_kv_heads = getattr(self.kv_args, "total_kv_head_num", 0)
        if total_kv_heads <= 0:
            total_kv_heads = self.kv_args.kv_head_num * prefill_tp_size

        src_heads_per_rank = max(1, total_kv_heads // prefill_tp_size)
        dst_heads_per_rank = max(1, total_kv_heads // decode_tp_size)

        bytes_per_head_slice = bytes_per_token_dst // dst_heads_per_rank
        if bytes_per_head_slice == 0:
            raise ValueError("Head slice size evaluates to zero")

        src_replication = max(1, prefill_tp_size // total_kv_heads)

        local_tp_rank = self.kv_args.engine_rank % prefill_tp_size
        dst_tp_rank = peer_info.decode_tp_rank % decode_tp_size

        if prefill_tp_size > decode_tp_size:
            src_head_start = 0
            num_heads_to_send = src_heads_per_rank
            unique_head_idx = local_tp_rank // src_replication
            dst_head_start = (unique_head_idx * src_heads_per_rank) % dst_heads_per_rank
        else:
            src_head_start = (dst_tp_rank * dst_heads_per_rank) % src_heads_per_rank
            num_heads_to_send = dst_heads_per_rank
            dst_head_start = 0

        src_head_slice_offset = src_head_start * bytes_per_head_slice
        dst_head_slice_offset = dst_head_start * bytes_per_head_slice
        heads_bytes_per_token = num_heads_to_send * bytes_per_head_slice

        if heads_bytes_per_token > bytes_per_token_dst:
            raise ValueError(
                "Slice size exceeds destination token capacity for TP slice transfer"
            )

        return TPSliceConfig(
            page_size=page_size,
            src_item_len=src_item_len,
            dst_item_len=dst_item_len,
            bytes_per_token_src=bytes_per_token_src,
            bytes_per_token_dst=bytes_per_token_dst,
            src_head_slice_offset=src_head_slice_offset,
            dst_head_slice_offset=dst_head_slice_offset,
            heads_bytes_per_token_to_send=heads_bytes_per_token,
        )

    def _build_tp_slice_transfer_plan(
        self,
        kv_indices: npt.NDArray[np.int32],
        dst_indices: npt.NDArray[np.int32],
        tp_cfg: TPSliceConfig,
    ) -> BatchTransferPlan:
        if kv_indices.size == 0 or dst_indices.size == 0:
            return BatchTransferPlan([], [], [])

        limit = min(kv_indices.size, dst_indices.size)
        if not limit:
            return BatchTransferPlan([], [], [])

        src_pages = kv_indices[:limit].astype(np.int64)
        dst_pages = dst_indices[:limit].astype(np.int64)
        token_slots = np.arange(tp_cfg.page_size, dtype=np.int64)

        src_page_bases = src_pages * tp_cfg.src_item_len
        dst_page_bases = dst_pages * tp_cfg.dst_item_len

        src_token_offsets = token_slots * tp_cfg.bytes_per_token_src
        dst_token_offsets = token_slots * tp_cfg.bytes_per_token_dst

        local_offsets = (
            (
                src_page_bases[:, np.newaxis]
                + src_token_offsets
                + tp_cfg.src_head_slice_offset
            )
            .flatten()
            .tolist()
        )
        remote_offsets = (
            (
                dst_page_bases[:, np.newaxis]
                + dst_token_offsets
                + tp_cfg.dst_head_slice_offset
            )
            .flatten()
            .tolist()
        )

        num_transfers = limit * tp_cfg.page_size
        sizes = [tp_cfg.heads_bytes_per_token_to_send] * num_transfers

        if not local_offsets:
            return BatchTransferPlan([], [], [])

        return BatchTransferPlan(
            local_offsets=local_offsets,
            remote_offsets=remote_offsets,
            sizes=sizes,
        )

    def send_kvcache(
        self,
        peer_info: KVArgsRegisterInfo,
        prefill_kv_indices: npt.NDArray[np.int32],
        dst_kv_indices: npt.NDArray[np.int32],
    ) -> List[TransferStatus]:
        grouped_plan = GroupedIndexPlan.from_groups(
            *group_concurrent_contiguous(
                prefill_kv_indices,
                dst_kv_indices,
            )
        )
        statuses: List[TransferStatus] = []
        src_kv_item_lens = self.kv_args.kv_item_lens
        dst_kv_item_lens = peer_info.dst_kv_item_lens
        start_layer = self.kv_args.prefill_start_layer

        def _layer_item_lens(src_idx: int, dst_idx: int) -> tuple[int, int]:
            src_len = src_kv_item_lens[src_idx]
            dst_len = dst_kv_item_lens[dst_idx] if dst_kv_item_lens else src_len
            if src_len != dst_len:
                raise ValueError(
                    "MoRI requires matching src/dst KV item lens per layer: "
                    f"src_idx={src_idx} ({src_len}) vs dst_idx={dst_idx} "
                    f"({dst_len})"
                )
            return src_len, dst_len

        if self.is_mla_backend:
            src_descs, dst_descs, layers_current_pp_stage = (
                self._get_mla_mem_desc_slices(peer_info.dst_kv_mem_descs)
            )
            for layer_id in range(layers_current_pp_stage):
                src_item_len, _ = _layer_item_lens(layer_id, start_layer + layer_id)
                layer_plan = self._build_contiguous_transfer_plan(
                    grouped_plan, src_item_len
                )
                statuses.extend(
                    self._submit_batch_transfer_plan(
                        src_descs[layer_id],
                        dst_descs[layer_id],
                        layer_plan,
                    )
                )
            return statuses

        (
            src_k_descs,
            src_v_descs,
            dst_k_descs,
            dst_v_descs,
            layers_current_pp_stage,
        ) = self._get_mha_mem_desc_slices(peer_info.dst_kv_mem_descs)
        # MHA KV item lens layout: K-layers first, then V-layers.
        # Source has num_local_layers per side; destination spans the full
        # decode-side model so K starts at start_layer and V starts at
        # dst_total_layers + start_layer.
        num_local_layers = layers_current_pp_stage
        dst_total_layers = len(peer_info.dst_kv_mem_descs) // 2
        k_src_off = 0
        v_src_off = num_local_layers
        k_dst_off = start_layer
        v_dst_off = dst_total_layers + start_layer

        if peer_info.decode_tp_size != self.attn_tp_size:
            for layer_id in range(layers_current_pp_stage):
                k_src_len, k_dst_len = _layer_item_lens(
                    k_src_off + layer_id, k_dst_off + layer_id
                )
                k_tp_cfg = self._build_tp_slice_config(peer_info, k_src_len, k_dst_len)
                k_slice_plan = self._build_tp_slice_transfer_plan(
                    prefill_kv_indices, dst_kv_indices, k_tp_cfg
                )
                statuses.extend(
                    self._submit_batch_transfer_plan(
                        src_k_descs[layer_id],
                        dst_k_descs[layer_id],
                        k_slice_plan,
                    )
                )
                v_src_len, v_dst_len = _layer_item_lens(
                    v_src_off + layer_id, v_dst_off + layer_id
                )
                v_tp_cfg = self._build_tp_slice_config(peer_info, v_src_len, v_dst_len)
                v_slice_plan = self._build_tp_slice_transfer_plan(
                    prefill_kv_indices, dst_kv_indices, v_tp_cfg
                )
                statuses.extend(
                    self._submit_batch_transfer_plan(
                        src_v_descs[layer_id],
                        dst_v_descs[layer_id],
                        v_slice_plan,
                    )
                )
            return statuses

        for layer_id in range(layers_current_pp_stage):
            k_src_len, _ = _layer_item_lens(k_src_off + layer_id, k_dst_off + layer_id)
            k_layer_plan = self._build_contiguous_transfer_plan(grouped_plan, k_src_len)
            statuses.extend(
                self._submit_batch_transfer_plan(
                    src_k_descs[layer_id],
                    dst_k_descs[layer_id],
                    k_layer_plan,
                )
            )
            v_src_len, _ = _layer_item_lens(v_src_off + layer_id, v_dst_off + layer_id)
            v_layer_plan = self._build_contiguous_transfer_plan(grouped_plan, v_src_len)
            statuses.extend(
                self._submit_batch_transfer_plan(
                    src_v_descs[layer_id],
                    dst_v_descs[layer_id],
                    v_layer_plan,
                )
            )
        return statuses

    def send_aux(
        self,
        peer_info: KVArgsRegisterInfo,
        prefill_aux_index: int,
        dst_aux_index: int,
        room: int,
    ) -> List[TransferStatus]:
        if self._send_aux_rdma:
            return self.send_aux_rdma(peer_info, prefill_aux_index, dst_aux_index, room)
        return self.send_aux_tcp(peer_info, prefill_aux_index, dst_aux_index, room)

    def send_aux_rdma(
        self,
        peer_info: KVArgsRegisterInfo,
        prefill_aux_index: int,
        dst_aux_index: int,
        room: int,
    ) -> List[TransferStatus]:
        if not self.aux_mem_descs or len(self.aux_mem_descs) != len(
            peer_info.dst_aux_mem_descs
        ):
            return self.send_aux_tcp(peer_info, prefill_aux_index, dst_aux_index, room)

        src_descs: List[MemoryDesc] = []
        dst_descs: List[MemoryDesc] = []
        local_offsets: List[List[int]] = []
        remote_offsets: List[List[int]] = []
        sizes: List[List[int]] = []
        uids = []
        for i in range(len(self.aux_mem_descs)):
            item_len = self.kv_args.aux_item_lens[i]
            src_descs.append(self.aux_mem_descs[i])
            dst_descs.append(peer_info.dst_aux_mem_descs[i])
            local_offsets.append([prefill_aux_index * item_len])
            remote_offsets.append([dst_aux_index * item_len])
            sizes.append([item_len])
            uids.append(self.engine.allocate_transfer_uid())
        return list(
            self.engine.batch_write(
                src_descs, local_offsets, dst_descs, remote_offsets, sizes, uids
            )
        )

    def send_aux_tcp(
        self,
        peer_info: KVArgsRegisterInfo,
        prefill_aux_index: int,
        dst_aux_index: int,
        room: int,
    ) -> List[TransferStatus]:
        for i in range(len(self.kv_args.aux_data_ptrs)):
            length = self.kv_args.aux_item_lens[i]
            src_addr = self.kv_args.aux_data_ptrs[i] + length * prefill_aux_index
            data = AuxDataCodec.serialize_data_from_buffer(src_addr, length)
            self._send_aux_data_to_endpoint(
                remote=peer_info.endpoint,
                dst_port=peer_info.dst_port,
                room=room,
                buffer_index=i,
                aux_index=dst_aux_index,
                data=data,
            )
        return []  # TCP path has no TransferStatus to poll

    def _send_aux_data_to_endpoint(
        self, remote, dst_port, room, buffer_index, aux_index, data
    ):
        na = NetworkAddress(remote, dst_port)
        socket = self._connect_threadsafe(na.to_tcp(), is_ipv6=na.is_ipv6)
        socket.send_multipart(
            [
                MoriKVManager.AUX_DATA_HEADER,
                str(room).encode("ascii"),
                str(buffer_index).encode("ascii"),
                str(aux_index).encode("ascii"),
                struct.pack(">I", len(data)),
                data,
            ]
        )

    def send_state(
        self,
        peer_info: KVArgsRegisterInfo,
        src_state_indices: List[npt.NDArray[np.int32]],
        dst_state_indices: List[npt.NDArray[np.int32]],
    ) -> List[TransferStatus]:
        # Guard: no local state tensors -> no-op (e.g. SWA layers=0 on this PP rank)
        if not self.state_mem_descs:
            return []

        state_types = self.kv_args.state_types
        if not state_types:
            raise RuntimeError(
                "PD state transfer failed: kv_args.state_types is empty but "
                "state_indices were provided"
            )

        if len(peer_info.dst_state_mem_descs) != len(self.state_mem_descs):
            raise RuntimeError(
                f"PD state transfer failed: state component count mismatch "
                f"(local={len(self.state_mem_descs)}, "
                f"remote={len(peer_info.dst_state_mem_descs)})"
            )

        src_state_item_lens = self.kv_args.state_item_lens
        src_state_dim_per_tensor = self.kv_args.state_dim_per_tensor

        statuses: List[TransferStatus] = []
        for i, st in enumerate(state_types):
            src_indices = src_state_indices[i] if i < len(src_state_indices) else None
            dst_indices = dst_state_indices[i] if i < len(dst_state_indices) else None
            if src_indices is None or src_indices.size == 0:
                continue
            if dst_indices is None or dst_indices.size == 0:
                continue

            src_descs = self.state_mem_descs[i]
            dst_descs = peer_info.dst_state_mem_descs[i]
            src_lens = src_state_item_lens[i] if i < len(src_state_item_lens) else []
            dst_lens = (
                peer_info.dst_state_item_lens[i]
                if i < len(peer_info.dst_state_item_lens)
                else []
            )
            src_dims = (
                src_state_dim_per_tensor[i] if i < len(src_state_dim_per_tensor) else []
            )
            dst_dims = (
                peer_info.dst_state_dim_per_tensor[i]
                if i < len(peer_info.dst_state_dim_per_tensor)
                else []
            )

            if st == "mamba":
                statuses.extend(
                    self._send_mamba_state(
                        peer_info,
                        src_indices,
                        dst_indices,
                        src_descs,
                        dst_descs,
                        src_lens,
                        dst_lens,
                        src_dims,
                        dst_dims,
                    )
                )
            elif st in ("swa", "dsa", "swa_ring"):
                statuses.extend(
                    self._send_swa_dsa_state(
                        peer_info,
                        src_indices,
                        dst_indices,
                        src_descs,
                        src_lens,
                        dst_descs,
                        st,
                    )
                )
            else:
                raise RuntimeError(f"PD state transfer failed: unknown state_type={st}")

        return statuses

    def _send_mamba_state(
        self,
        peer_info: KVArgsRegisterInfo,
        src_state_indices: npt.NDArray[np.int32],
        dst_state_indices: npt.NDArray[np.int32],
        src_state_mem_descs: List[MemoryDesc],
        dst_state_mem_descs: List[MemoryDesc],
        src_state_item_lens: List[int],
        dst_state_item_lens: List[int],
        src_state_dim_per_tensor: List[int],
        dst_state_dim_per_tensor: List[int],
    ) -> List[TransferStatus]:
        if src_state_indices.size != 1 or dst_state_indices.size != 1:
            raise RuntimeError(
                f"PD state transfer failed: mamba requires single state index, "
                f"got src={src_state_indices.size}, dst={dst_state_indices.size}"
            )

        tp_mismatch = peer_info.decode_tp_size != self.attn_tp_size

        # If dim info missing, silently degrade to whole-item copy (Mooncake compat)
        if tp_mismatch and (
            not src_state_dim_per_tensor or not dst_state_dim_per_tensor
        ):
            tp_mismatch = False

        if tp_mismatch:
            logger.warning_once(
                "Using Mamba state slice transfer for different TP sizes between prefill and decode. "
                f"Prefill attn_tp_size={self.attn_tp_size}, Decode attn_tp_size={peer_info.decode_tp_size}. "
                "Performance may be affected."
            )

        src_idx = int(src_state_indices[0])
        dst_idx = int(dst_state_indices[0])
        statuses: List[TransferStatus] = []

        local_tp_rank = self.kv_args.engine_rank % self.attn_tp_size
        dst_tp_rank = peer_info.decode_tp_rank % peer_info.decode_tp_size

        for i, src_desc in enumerate(src_state_mem_descs):
            dst_desc = dst_state_mem_descs[i]
            src_item_len = src_state_item_lens[i]

            if not tp_mismatch:
                # same-TP: whole item copy
                src_offset = src_idx * src_item_len
                dst_offset = dst_idx * src_item_len
                size = src_item_len
            else:
                # TP mismatch slice copy
                dst_item_len = dst_state_item_lens[i]
                src_dim = src_state_dim_per_tensor[i]
                dst_dim = dst_state_dim_per_tensor[i]

                src_bytes_per_dim = src_item_len // src_dim

                if self.attn_tp_size > peer_info.decode_tp_size:
                    src_dim_start = 0
                    num_dims_to_send = src_dim
                    writers_per_decode = self.attn_tp_size // peer_info.decode_tp_size
                    local_writer_idx = local_tp_rank % writers_per_decode
                    dst_dim_start = local_writer_idx * src_dim
                else:
                    src_dim_start = (dst_tp_rank * dst_dim) % src_dim
                    num_dims_to_send = dst_dim
                    dst_dim_start = 0

                dst_bytes_per_dim = dst_item_len // dst_dim
                src_dim_offset = src_dim_start * src_bytes_per_dim
                dst_dim_offset = dst_dim_start * dst_bytes_per_dim
                bytes_to_send = num_dims_to_send * src_bytes_per_dim

                src_offset = src_idx * src_item_len + src_dim_offset
                dst_offset = dst_idx * dst_item_len + dst_dim_offset
                size = bytes_to_send

            transfer_uid = self.engine.allocate_transfer_uid()
            batch_statuses = self.engine.batch_write(
                [src_desc],
                [[src_offset]],
                [dst_desc],
                [[dst_offset]],
                [[size]],
                [transfer_uid],
            )
            statuses.extend(batch_statuses)

        return statuses

    def _send_swa_dsa_state(
        self,
        peer_info: KVArgsRegisterInfo,
        src_state_indices: npt.NDArray[np.int32],
        dst_state_indices: npt.NDArray[np.int32],
        src_state_mem_descs: List[MemoryDesc],
        src_state_item_lens: List[int],
        dst_state_mem_descs: List[MemoryDesc],
        state_type: str,
    ) -> List[TransferStatus]:
        # TP mismatch check for non-MLA SWA
        if (
            state_type == "swa"
            and not self.is_mla_backend
            and peer_info.decode_tp_size != self.attn_tp_size
        ):
            raise RuntimeError(
                f"PD state transfer does not support TP-mismatched non-MLA SWA models "
                f"(prefill_tp_size={self.attn_tp_size}, decode_tp_size={peer_info.decode_tp_size})"
            )

        common_len = min(src_state_indices.size, dst_state_indices.size)
        if common_len == 0 and max(src_state_indices.size, dst_state_indices.size) > 0:
            raise RuntimeError(
                f"No overlapping state indices for state_type={state_type}"
            )
        if src_state_indices.size != dst_state_indices.size:
            # SWA_RING is positional: truncating silently misaligns rows and
            # corrupts KV, so fail loud. Paged swa/dsa tolerate a 1-page drift
            # -> keep truncation.
            if state_type == "swa_ring":
                raise RuntimeError(
                    "SWA_RING state index length mismatch: "
                    f"src={src_state_indices.size}, dst={dst_state_indices.size}"
                )
            logger.warning(
                "State index length mismatch for %s: src=%d dst=%d; truncating to common prefix=%d",
                state_type,
                src_state_indices.size,
                dst_state_indices.size,
                common_len,
            )
            src_state_indices = src_state_indices[:common_len]
            dst_state_indices = dst_state_indices[:common_len]

        # Group contiguous indices and issue per-tensor transfers
        grouped_plan = GroupedIndexPlan.from_groups(
            *group_concurrent_contiguous(src_state_indices, dst_state_indices)
        )

        statuses: List[TransferStatus] = []
        for i, src_desc in enumerate(src_state_mem_descs):
            dst_desc = dst_state_mem_descs[i]
            state_item_len = src_state_item_lens[i]

            statuses.extend(
                self._submit_batch_transfer_plan(
                    src_desc,
                    dst_desc,
                    self._build_contiguous_transfer_plan(grouped_plan, state_item_len),
                )
            )

        return statuses

    def _handle_aux_data(self, msg: List[bytes]):
        """Handle AUX_DATA messages received by the decode thread (legacy TCP path)."""
        room = int(msg[1].decode("ascii"))
        buffer_index = int(msg[2].decode("ascii"))
        aux_index = int(msg[3].decode("ascii"))
        data_length = struct.unpack(">I", msg[4])[0]
        data = msg[5]

        if len(data) != data_length:
            logger.error(f"AUX_DATA length mismatch for bootstrap_room {room}")
            return

        AuxDataCodec.deserialize_data_to_buffer(
            self.kv_args, buffer_index, aux_index, data
        )

    def add_transfer_request(
        self,
        bootstrap_room: int,
        kv_indices: npt.NDArray[np.int32],
        index_slice: slice,
        is_last_chunk: bool,
        aux_index: Optional[int] = None,
        state_indices: Optional[List[npt.NDArray[np.int32]]] = None,
    ) -> Tuple[List[TransferStatus], Optional[List[TransferInfo]]]:
        assert self.disaggregation_mode == DisaggregationMode.PREFILL

        if (
            bootstrap_room not in self.request_status
            or self.request_status.get(bootstrap_room) == KVPoll.Failed
        ):
            return [], None

        targets: List[TransferTarget] = []
        target_infos_snapshot: Optional[List[TransferInfo]] = None
        with self.transfer_lock:
            transfer_infos = self.transfer_infos.get(bootstrap_room)
            if not transfer_infos:
                reason = f"No transfer info found for bootstrap_room={bootstrap_room}"
                self.record_failure(bootstrap_room, reason)
                self.update_status(bootstrap_room, KVPoll.Failed)
                return [], None

            self.update_status(bootstrap_room, KVPoll.Transferring)
            for info in transfer_infos.values():
                peer_info = self.decode_kv_args_table.get(info.engine_key)
                if not peer_info:
                    reason = f"Peer info missing for engine {info.engine_key}"
                    self.record_failure(bootstrap_room, reason)
                    self.update_status(bootstrap_room, KVPoll.Failed)
                    return [], list(transfer_infos.values())
                targets.append(TransferTarget(info=info, peer_info=peer_info))
            if is_last_chunk:
                target_infos_snapshot = list(transfer_infos.values())

        result_statuses: List[TransferStatus] = []
        try:
            for target in targets:
                info = target.info
                peer_info = target.peer_info

                if not info.is_dummy:
                    dst_indices_chunk = info.dst_kv_indices[index_slice]
                    result_statuses.extend(
                        self.send_kvcache(peer_info, kv_indices, dst_indices_chunk)
                    )

                if (
                    is_last_chunk
                    and state_indices is not None
                    and not info.is_dummy
                    and self.state_mem_descs
                ):
                    result_statuses.extend(
                        self.send_state(
                            peer_info, state_indices, info.dst_state_indices
                        )
                    )

                if (
                    is_last_chunk
                    and aux_index is not None
                    and info.dst_aux_index >= 0
                    and self.pp_group.is_last_rank
                ):
                    result_statuses.extend(
                        self.send_aux(
                            peer_info, aux_index, info.dst_aux_index, bootstrap_room
                        )
                    )
        except Exception as e:
            reason = f"Transfer submission failed: {e}"
            with self.transfer_lock:
                self.record_failure(bootstrap_room, reason)
                self.update_status(bootstrap_room, KVPoll.Failed)
            logger.exception(
                "Mori KV transfer submission failed for bootstrap_room=%s",
                bootstrap_room,
            )
            return result_statuses, target_infos_snapshot

        return result_statuses, target_infos_snapshot


class MoriKVSender(CommonKVSender):
    def __init__(
        self,
        mgr: MoriKVManager,
        bootstrap_addr: str,
        bootstrap_room: int,
        dest_tp_ranks: List[int],
        pp_rank: int,
    ):
        super().__init__(mgr, bootstrap_addr, bootstrap_room, dest_tp_ranks, pp_rank)
        self.transfer_statuses: List[TransferStatus] = []
        self.pending_infos: Optional[List[TransferInfo]] = None
        self.conclude_state: Optional[KVPoll] = None
        self.status_notified = False
        self.init_time = time.time()
        self._notify_lock = threading.Lock()
        self._notified_status: Optional[KVPoll] = None
        self._notified_reason: Optional[str] = None

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

        normalized_state = (
            _normalize_state_indices_per_component(state_indices)
            if is_last_chunk
            else None
        )
        self._record_transfer_indices(kv_indices, state_indices)
        self.kv_mgr.enqueue_transfer(
            _TransferChunk(
                sender=self,
                kv_indices=kv_indices,
                index_slice=index_slice,
                is_last_chunk=is_last_chunk,
                aux_index=self.aux_index if is_last_chunk else None,
                normalized_state=normalized_state,
            )
        )
        self._maybe_finalize_if_room_failed()

    def _maybe_finalize_if_room_failed(self) -> None:
        if self.conclude_state is not None:
            return
        if self.kv_mgr.request_status.get(self.bootstrap_room) == KVPoll.Failed:
            self._finalize_failure()

    def _run_chunk(self, task: _TransferChunk) -> None:
        if self.conclude_state is not None:
            return
        if self.kv_mgr.request_status.get(self.bootstrap_room) == KVPoll.Failed:
            self._finalize_failure()
            return

        statuses, infos = self.kv_mgr.add_transfer_request(
            self.bootstrap_room,
            task.kv_indices,
            task.index_slice,
            task.is_last_chunk,
            aux_index=task.aux_index,
            state_indices=task.normalized_state,
        )
        self.transfer_statuses.extend(statuses)
        if infos is not None:
            self.pending_infos = infos

        if self.kv_mgr.request_status.get(self.bootstrap_room) == KVPoll.Failed:
            self._finalize_failure()
            return

        rc = self._wait_chunk(statuses)
        if self.conclude_state is not None:
            return
        if rc != StatusCode.SUCCESS:
            self._finalize_failure(self._collect_failure_reason())
            return
        if task.is_last_chunk:
            self._notify_decode(KVPoll.Success)
            with self._notify_lock:
                if self.conclude_state is None:
                    self.conclude_state = self._notified_status
                if self._notified_status == KVPoll.Success:
                    self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Success)

    def _wait_chunk(self, statuses: List[TransferStatus]) -> StatusCode:
        if not statuses:
            return StatusCode.SUCCESS

        start = time.perf_counter()
        sla_ms = self.kv_mgr._transfer_timeout_ms
        sla_tripped = False

        while True:
            rc = self.kv_mgr.engine.wait_all(
                statuses, timeout_ms=self.kv_mgr._wait_poll_ms
            )
            if rc != StatusCode.IN_PROGRESS:
                return rc
            if (
                sla_ms > 0
                and not sla_tripped
                and (time.perf_counter() - start) * 1000 >= sla_ms
            ):
                sla_tripped = True
                self._finalize_failure(f"KV transfer exceeded SLA {sla_ms}ms")

    def _fail_from_worker(self, reason: str) -> None:
        self._finalize_failure(reason)

    def poll(self) -> KVPoll:
        if self.conclude_state is not None:
            return self.conclude_state

        if self.bootstrap_room not in self.kv_mgr.request_status:
            sent_status, _ = self._finalize_failure()
            return sent_status

        status = self.kv_mgr.check_status(self.bootstrap_room)

        if status == KVPoll.Bootstrapping:
            elapsed = time.time() - self.init_time
            if elapsed >= self.kv_mgr.bootstrap_timeout:
                logger.warning_once(
                    "Some requests timed out when bootstrapping, "
                    "which means prefill instances fail to receive the KV indices from the decode instance of this request. "
                    "If a greater mean TTFT is acceptable, you can 'export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600' (10 minutes) to relax the timeout condition. "
                )
                reason = (
                    f"Request {self.bootstrap_room} timed out after {elapsed:.1f}s "
                    "in KVPoll.Bootstrapping"
                )
                sent_status, _ = self._finalize_failure(reason)
                return sent_status
            return status

        if status == KVPoll.Failed:
            sent_status, _ = self._finalize_failure()
            return sent_status

        if status == KVPoll.Success:
            self.conclude_state = KVPoll.Success
            return KVPoll.Success

        return status

    def _collect_failure_reason(self) -> str:
        for status in self.transfer_statuses:
            if status.Failed():
                return f"KV transfer failed: {status.Message()}"
        return "KV transfer failed due to unknown reason"

    def _terminalize_locked(
        self,
        status: KVPoll,
        reason: Optional[str] = None,
    ) -> Tuple[KVPoll, Optional[str], Optional[List[TransferInfo]]]:
        if self.status_notified:
            return self._notified_status, self._notified_reason, None

        if status == KVPoll.Success:
            with self.kv_mgr.failure_lock:
                recorded = self.kv_mgr.failure_records.get(self.bootstrap_room)
            if recorded is not None:
                status = KVPoll.Failed
                reason = recorded
            elif self.kv_mgr.request_status.get(self.bootstrap_room) == KVPoll.Failed:
                status = KVPoll.Failed
                reason = reason or "request marked Failed before notify"

        if status == KVPoll.Failed:
            with self.kv_mgr.failure_lock:
                self.kv_mgr.failure_records.setdefault(
                    self.bootstrap_room, reason or "KV transfer failed"
                )
            self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Failed)

        infos = self.pending_infos
        if infos is None:
            with self.kv_mgr.transfer_lock:
                room_infos = self.kv_mgr.transfer_infos.get(self.bootstrap_room)
                infos = list(room_infos.values()) if room_infos is not None else None

        self._notified_status = status
        self._notified_reason = reason
        self.status_notified = True
        return status, reason, infos

    def _notify_decode(
        self, status: KVPoll, failure_reason: Optional[str] = None
    ) -> Tuple[KVPoll, Optional[str]]:
        with self._notify_lock:
            emitted_status, emitted_reason, infos = self._terminalize_locked(
                status, failure_reason
            )
        if infos:
            self.kv_mgr.notify_decode_status(
                infos, self.bootstrap_room, emitted_status, emitted_reason
            )
        return emitted_status, emitted_reason

    def _finalize_failure(
        self, failure_reason: Optional[str] = None
    ) -> Tuple[KVPoll, Optional[str]]:
        if failure_reason is None:
            with self.kv_mgr.failure_lock:
                failure_reason = self.kv_mgr.failure_records.get(
                    self.bootstrap_room, "KV transfer failed"
                )
        sent_status, sent_reason = self._notify_decode(KVPoll.Failed, failure_reason)
        self.conclude_state = sent_status
        return sent_status, sent_reason

    def failure_exception(self):
        if self.conclude_state is None:
            self._finalize_failure()
        self.clear()
        with self.kv_mgr.failure_lock:
            failure_reason = self.kv_mgr.failure_records.pop(self.bootstrap_room, None)
        is_propagated = failure_reason is None
        if is_propagated:
            failure_reason = "KV transfer failed"
        raise KVTransferError(
            self.bootstrap_room, failure_reason, is_from_another_rank=is_propagated
        )

    def abort(self):
        self._finalize_failure("Aborted by AbortReq.")


class MoriKVReceiver(CommonKVReceiver):

    def __init__(
        self,
        mgr: MoriKVManager,
        bootstrap_addr: str,
        bootstrap_room: Optional[int] = None,
    ):
        super().__init__(mgr, bootstrap_addr, bootstrap_room)
        self.init_time: Optional[float] = None

    def init(
        self,
        prefill_dp_rank: int,
    ):
        super().init(prefill_dp_rank)
        if self.bootstrap_room is None:
            return
        self.kv_mgr.room_to_bootstrap_addr[self.bootstrap_room] = self.bootstrap_addr

    def _register_kv_args(self):
        if self.bootstrap_infos is None:
            return
        engine_desc_blob = self.kv_mgr.engine_desc.pack()
        packed_kv_descs = _pack_mem_desc_list(self.kv_mgr.kv_mem_descs)
        packed_aux_descs = _pack_mem_desc_list(self.kv_mgr.aux_mem_descs)
        packed_state_descs = _pack_mem_desc_lists(self.kv_mgr.state_mem_descs)
        gpu_id = str(self.kv_mgr.kv_args.gpu_id).encode("ascii")
        decode_tp_size = str(self.kv_mgr.attn_tp_size).encode("ascii")
        decode_tp_rank = str(self.kv_mgr.kv_args.engine_rank).encode("ascii")
        kv_item_len = str(self.kv_mgr.kv_args.kv_item_lens[0]).encode("ascii")
        packed_state_item_lens = pack_int_lists(
            self.kv_mgr.kv_args.state_item_lens, "I"
        )
        packed_state_dim_per_tensor = pack_int_lists(
            self.kv_mgr.kv_args.state_dim_per_tensor, "I"
        )
        packed_kv_item_lens = b"".join(
            struct.pack("I", item_len) for item_len in self.kv_mgr.kv_args.kv_item_lens
        )

        for bootstrap_info in self.bootstrap_infos:
            sock, lock = self._connect_to_bootstrap_server(bootstrap_info)
            with lock:
                sock.send_multipart(
                    [
                        MORI_GUARD,
                        "None".encode("ascii"),
                        self.kv_mgr.local_ip.encode("ascii"),
                        str(self.kv_mgr.rank_port).encode("ascii"),
                        engine_desc_blob,
                        packed_kv_descs,
                        packed_aux_descs,
                        packed_state_descs,
                        gpu_id,
                        decode_tp_size,
                        decode_tp_rank,
                        kv_item_len,
                        packed_state_item_lens,
                        packed_state_dim_per_tensor,
                        packed_kv_item_lens,
                    ]
                )

    def send_metadata(
        self,
        kv_indices: npt.NDArray[np.int32],
        aux_index: Optional[int] = None,
        state_indices: Optional[List] = None,
        decode_prefix_len: Optional[int] = None,
    ):
        if self.bootstrap_infos is None or self.bootstrap_room is None:
            return

        kv_indices_bytes = (
            np.asarray(kv_indices, dtype=np.int32).tobytes() if kv_indices.size else b""
        )
        aux_bytes = str(aux_index).encode("ascii") if aux_index is not None else b""
        normalized_state = _normalize_state_indices_per_component(state_indices)

        decode_prefix_bytes = (
            str(int(decode_prefix_len)).encode("ascii")
            if decode_prefix_len is not None and decode_prefix_len > 0
            else b""
        )

        for bootstrap_info in self.bootstrap_infos:
            sock, lock = self._connect_to_bootstrap_server(bootstrap_info)
            is_dummy = bootstrap_info.get("is_dummy", False)
            if not is_dummy and normalized_state is not None:
                state_bytes = _pack_state_indices(normalized_state)
            else:
                state_bytes = b""
            with lock:
                sock.send_multipart(
                    [
                        MORI_GUARD,
                        str(self.bootstrap_room).encode("ascii"),
                        self.kv_mgr.local_ip.encode("ascii"),
                        str(self.kv_mgr.rank_port).encode("ascii"),
                        self.kv_mgr.engine_desc.key.encode("ascii"),
                        kv_indices_bytes if not is_dummy else b"",
                        aux_bytes if not is_dummy else b"",
                        state_bytes,
                        str(self.required_dst_info_num).encode("ascii"),
                        decode_prefix_bytes,
                    ]
                )
        self.init_time = time.time()

    def poll(self) -> KVPoll:
        if self.conclude_state is not None:
            return self.conclude_state

        status = self.kv_mgr.check_status(self.bootstrap_room)
        if status in (KVPoll.Success, KVPoll.Failed):
            self.conclude_state = status
            return status

        if status == KVPoll.WaitingForInput:
            timeout_result = self._check_waiting_timeout()
            if timeout_result is not None:
                return timeout_result

        return status

    def clear(self) -> None:
        if self.bootstrap_room is None:
            return
        super().clear()
        self.kv_mgr._cleanup_room_tracking(self.bootstrap_room)

    def failure_exception(self):
        if self.conclude_state is None:
            self.conclude_state = KVPoll.Failed

        self.clear()
        with self.kv_mgr.failure_lock:
            failure_reason = self.kv_mgr.failure_records.pop(self.bootstrap_room, None)
        is_propagated = failure_reason is None
        if is_propagated:
            failure_reason = "KV transfer failed"
        raise KVTransferError(
            self.bootstrap_room, failure_reason, is_from_another_rank=is_propagated
        )

    def abort(self):
        if self.bootstrap_room is None:
            return
        super().abort()
        self.clear()


class MoriKVBootstrapServer(CommonKVBootstrapServer):
    pass
