from __future__ import annotations

import dataclasses
import logging
import os
import struct
import threading
import time
import uuid
from collections import defaultdict
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
from sglang.srt.disaggregation.common.staging_buffer import StagingBuffer
from sglang.srt.disaggregation.common.utils import (
    AuxDataCodec,
    DCPTokenTransferPlan,
    FastQueue,
    TransferKVChunk,
    build_dcp_token_transfer_plan,
    group_concurrent_contiguous,
    pack_int_lists,
    unpack_int_lists,
)
from sglang.srt.disaggregation.utils import (
    DisaggregationMode,
    resolve_dcp_dst_entry_indices,
)
from sglang.srt.environ import envs
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils.common import run_with_deadline
from sglang.srt.utils.network import NetworkAddress, get_local_ip_auto

logger = logging.getLogger(__name__)
MORI_GUARD = b"MoriMsgGuard"
MORI_DCP_GUARD = b"MoriDCPMsgGuard"
MORI_DCP_DRAIN_GUARD = b"MoriDCPDrainV1"
MORI_DCP_REGISTER_ACK = b"MoriDCPRegisterAck"
MORI_DCP_MIN_PACK_BUFFER_BYTES = 32 * 1024 * 1024
MORI_DCP_REGISTRATION_TIMEOUT_SECONDS = 30.0
MORI_ABORT_TOMBSTONE_LIMIT = 65536
_TAG_ABORT = b"ABORT"


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
    abort_generation: Optional[str] = None

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
        abort_generation = (
            payload[9].decode("ascii") if len(payload) > 9 and payload[9] else None
        )

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
            abort_generation=abort_generation,
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
    dst_kv_item_lens: List[int] = dataclasses.field(default_factory=list)
    dst_kv_layer_ids: List[int] = dataclasses.field(default_factory=list)
    dst_dcp_size: int = 1
    dst_dcp_rank: int = 0
    dcp_registration_id: Optional[str] = None
    supports_dcp_drain: bool = False
    requires_dcp_relayout: bool = False
    dcp_token_item_lens: Optional[List[int]] = None
    dcp_dst_region_indices: Optional[List[int]] = None

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
        dst_kv_item_lens = (
            list(struct.unpack(f"{len(payload[13]) // 8}Q", payload[13]))
            if len(payload) > 13 and payload[13]
            else [dst_kv_item_len] * len(dst_kv_mem_descs)
        )
        if len(dst_kv_item_lens) != len(dst_kv_mem_descs):
            raise ValueError(
                "dst_kv_item_lens length mismatch: "
                f"got {len(dst_kv_item_lens)}, expected {len(dst_kv_mem_descs)}"
            )
        dst_kv_layer_ids = (
            list(struct.unpack(f"{len(payload[14]) // 4}I", payload[14]))
            if len(payload) > 14 and payload[14]
            else []
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
            dst_kv_layer_ids=dst_kv_layer_ids,
            dst_dcp_size=(
                int(payload[15].decode("ascii"))
                if len(payload) > 15 and payload[15]
                else 1
            ),
            dst_dcp_rank=(
                int(payload[16].decode("ascii"))
                if len(payload) > 16 and payload[16]
                else 0
            ),
            dcp_registration_id=(
                payload[17].decode("ascii")
                if len(payload) > 17 and payload[17]
                else None
            ),
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
class MoriPackedDCPSource:
    mem_desc: MemoryDesc
    layer_offsets: List[int]
    token_indices: npt.NDArray[np.int64]


@dataclasses.dataclass(frozen=True)
class MoriPackedDCPRank:
    signature: Tuple[int, Tuple[int, ...], str, bytes]
    source: Optional[MoriPackedDCPSource]


class MoriKVSubmissionError(RuntimeError):
    def __init__(self, statuses: List[TransferStatus], cause: Exception):
        super().__init__(f"MORI KV transfer submission failed: {cause}")
        self.statuses = statuses


@dataclasses.dataclass(frozen=True)
class TransferTarget:
    info: TransferInfo
    peer_info: KVArgsRegisterInfo


class MoriKVManager(CommonKVManager):
    AUX_DATA_HEADER = b"AUX_DATA"

    # The bootstrap socket carries several message kinds, so the status message
    # is tagged. Mori has always shipped the failure reason with it.
    kv_status_msg_tag = MORI_GUARD
    kv_status_msg_carries_reason = True

    def __init__(
        self,
        args: KVArgs,
        disaggregation_mode: DisaggregationMode,
        server_args: ServerArgs,
        is_mla_backend: Optional[bool] = False,
    ):
        super().__init__(args, disaggregation_mode, server_args, is_mla_backend)
        self.requires_strict_deferred_release = (
            disaggregation_mode == DisaggregationMode.DECODE and self.dcp_size > 1
        )
        if self.requires_strict_deferred_release:
            self.enable_deferred_decode_kv_release = True
        self.engine = self._init_engine()
        self.engine_desc = self.engine.get_engine_desc()
        self.kv_mem_descs: List[MemoryDesc] = []
        self.aux_mem_descs: List[MemoryDesc] = []
        self.state_mem_descs: List[List[MemoryDesc]] = []
        self._dcp_pack_mem_descs: Dict[int, MemoryDesc] = {}
        self._dcp_pack_init_lock = threading.Lock()
        self._dcp_pack_dcp_size: Optional[int] = None
        self._dcp_pack_worker_count = 0
        self._dcp_pack_disabled_workers: set[int] = set()
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
            self._staging_outstanding = defaultdict(int)
            self._rooms_pending_clear: Dict[int, str] = {}
            self._room_generations: Dict[int, str] = {}
            self._room_owners: Dict[int, str] = {}
            self._retired_room_status: Dict[Tuple[int, str], KVPoll] = {}
            self._retired_room_failures: Dict[Tuple[int, str], str] = {}
            self._aborted_generations: Dict[Tuple[int, str], None] = {}
            self._room_lifecycle_lock = threading.Lock()
            for shard, queue in enumerate(self._transfer_queues):
                threading.Thread(
                    target=self._transfer_worker,
                    args=(queue, shard),
                    daemon=True,
                    name=(
                        f"mori-xfer-dp{self.system_dp_rank}-"
                        f"tp{self.attn_tp_rank}-s{shard}"
                    ),
                ).start()
            self._start_bootstrap_thread()
        elif self.disaggregation_mode == DisaggregationMode.DECODE:
            self._decode_room_generations: Dict[int, str] = {}
            self._dcp_registration_lock = threading.Lock()
            self._dcp_registration_events: Dict[str, threading.Event] = {}
            self._start_decode_thread()
            self._start_heartbeat_checker_thread()

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

        engine = run_with_deadline(
            lambda: IOEngine(engine_key, config),
            timeout_s=envs.SGLANG_DISAGGREGATION_ENGINE_INIT_TIMEOUT.get(),
            what=f"Mori IOEngine({engine_key!r}, host={self.local_ip!r})",
        )
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

    def _register_staging_memory(self, ptr: int, size: int) -> None:
        self._dcp_pack_mem_descs[ptr] = self.engine.register_memory(
            ptr,
            size,
            self.kv_args.gpu_id,
            MemoryLocationType.GPU,
        )

    def _init_dcp_pack_buffers_once(self, dcp_size: int) -> None:
        with self._dcp_pack_init_lock:
            if self._dcp_pack_dcp_size is not None:
                if self._dcp_pack_dcp_size != dcp_size:
                    raise RuntimeError(
                        "MORI DCP peers must use one dcp_size per manager, got "
                        f"{self._dcp_pack_dcp_size} and {dcp_size}"
                    )
                return
            if not self.kv_args.kv_item_lens:
                return

            from sglang.srt.disaggregation.common.dcp_pack import (
                dcp_pack_buffer_bytes_for_args,
            )

            buffer_bytes = max(
                dcp_pack_buffer_bytes_for_args(self.kv_args, dcp_size),
                MORI_DCP_MIN_PACK_BUFFER_BYTES,
            )
            budget_bytes = int(
                envs.SGLANG_MORI_DCP_PACK_BUFFER_BUDGET_GB.get() * 1024 * 1024 * 1024
            )
            self._dcp_pack_dcp_size = dcp_size
            self._dcp_pack_worker_count = max(
                0,
                min(
                    self._num_shards,
                    budget_bytes // buffer_bytes,
                ),
            )
            self._dcp_pack_buffers = [None] * self._num_shards
            if self._dcp_pack_worker_count == 0:
                logger.warning(
                    "MORI DCP pack buffer needs %.1f MB but budget is %.1f MB; "
                    "using per-token RDMA",
                    buffer_bytes / (1024 * 1024),
                    budget_bytes / (1024 * 1024),
                )
            else:
                logger.info(
                    "MORI DCP packing configured for %d/%d workers: %.1f MB "
                    "per buffer, %.1f MB budget",
                    self._dcp_pack_worker_count,
                    self._num_shards,
                    buffer_bytes / (1024 * 1024),
                    budget_bytes / (1024 * 1024),
                )

    def _get_or_init_dcp_pack_buffer(
        self, worker_index: int
    ) -> Optional[StagingBuffer]:
        if (
            self._dcp_pack_buffers is None
            or worker_index >= self._dcp_pack_worker_count
            or worker_index in self._dcp_pack_disabled_workers
        ):
            return None
        pack_buffer = self._dcp_pack_buffers[worker_index]
        if pack_buffer is not None:
            return pack_buffer

        with self._dcp_pack_init_lock:
            pack_buffer = self._dcp_pack_buffers[worker_index]
            if pack_buffer is not None:
                return pack_buffer
            try:
                from sglang.srt.disaggregation.common.dcp_pack import (
                    init_dcp_pack_buffers,
                )

                dcp_size = self._dcp_pack_dcp_size
                if dcp_size is None:
                    return None
                pack_buffer = init_dcp_pack_buffers(
                    self._register_staging_memory,
                    self.kv_args,
                    1,
                    dcp_size,
                    min_size_bytes=MORI_DCP_MIN_PACK_BUFFER_BYTES,
                )[0]
                self._dcp_pack_buffers[worker_index] = pack_buffer
                return pack_buffer
            except Exception:
                self._dcp_pack_disabled_workers.add(worker_index)
                logger.exception(
                    "MORI DCP pack allocation failed for worker %d; using "
                    "per-token RDMA",
                    worker_index,
                )
                return None

    def _retire_room_locked(self, room: int, owner: str) -> None:
        if self._room_owners.get(room) != owner:
            return
        status = self.request_status.pop(room, KVPoll.Failed)
        if status not in (KVPoll.Success, KVPoll.Failed):
            status = KVPoll.Failed
        self._retired_room_status[(room, owner)] = status
        failure_reason = self.failure_records.pop(room, None)
        if failure_reason is not None:
            self._retired_room_failures[(room, owner)] = failure_reason
        self.req_to_decode_prefix_len.pop(room, None)
        self.transfer_infos.pop(room, None)
        self._room_generations.pop(room, None)
        self._room_owners.pop(room, None)

    def activate_room_owner(self, room: int, owner: str) -> None:
        with self._room_lifecycle_lock:
            with self.failure_lock:
                with self.transfer_lock:
                    with self._deferred_ack_lock:
                        old_owner = self._room_owners.get(room)
                        if old_owner is not None and old_owner != owner:
                            old_status = self.request_status.get(room)
                            if self._staging_outstanding.get(
                                room, 0
                            ) > 0 or old_status not in (KVPoll.Success, KVPoll.Failed):
                                raise KVTransferError(
                                    room,
                                    f"Cannot reuse active MORI bootstrap room {room}",
                                )
                            self._retire_room_locked(room, old_owner)

                        self._room_owners[room] = owner
                        self._room_generations.pop(room, None)
                        self.req_to_decode_prefix_len.pop(room, None)
                        self.failure_records.pop(room, None)

                        infos = self.transfer_infos.get(room, {})
                        generations = {
                            info.abort_generation
                            for info in infos.values()
                            if info.abort_generation is not None
                        }
                        if len(generations) > 1:
                            self.transfer_infos.pop(room, None)
                            self.request_status[room] = KVPoll.Failed
                            self.failure_records[room] = (
                                "MORI decode ranks supplied inconsistent request epochs"
                            )
                            return

                        generation = next(iter(generations), None)
                        if generation is not None:
                            self._room_generations[room] = generation
                        if (
                            generation is not None
                            and (
                                room,
                                generation,
                            )
                            in self._aborted_generations
                        ):
                            self.transfer_infos.pop(room, None)
                            self.request_status[room] = KVPoll.Failed
                            self.failure_records[room] = (
                                "MORI request was aborted before the prefill sender activated"
                            )
                        elif infos:
                            ready = (
                                len(infos)
                                >= next(iter(infos.values())).required_dst_info_num
                            )
                            self.request_status[room] = (
                                KVPoll.WaitingForInput
                                if ready
                                else KVPoll.Bootstrapping
                            )
                        else:
                            self.request_status[room] = KVPoll.Bootstrapping
                        if generation is not None:
                            self._aborted_generations = {
                                entry: None
                                for entry in self._aborted_generations
                                if entry[0] != room
                            }

    def check_room_owner_status(self, room: int, owner: str) -> KVPoll:
        with self.transfer_lock:
            if self._room_owners.get(room) == owner:
                return self.request_status.get(room, KVPoll.Failed)
            return self._retired_room_status.get((room, owner), KVPoll.Failed)

    def pop_room_owner_failure(self, room: int, owner: str) -> Optional[str]:
        with self.failure_lock:
            with self.transfer_lock:
                if self._room_owners.get(room) == owner:
                    return self.failure_records.pop(room, None)
                return self._retired_room_failures.pop((room, owner), None)

    def clear_room_owner(self, room: int, owner: str) -> None:
        with self._room_lifecycle_lock:
            with self.failure_lock:
                with self.transfer_lock:
                    if self._room_owners.get(room) == owner:
                        self._retire_room_locked(room, owner)
                    self._retired_room_status.pop((room, owner), None)
                    self._retired_room_failures.pop((room, owner), None)

    def _snapshot_kv_status_extra_frames(
        self,
        bootstrap_room: int,
        targets: List[Tuple[str, int]],
    ) -> Dict[Tuple[str, int], List[bytes]]:
        with self.transfer_lock:
            generation = self._room_generations.get(bootstrap_room)
        if generation is None:
            return {}
        generation_frame = generation.encode("ascii")
        return {target: [generation_frame] for target in targets}

    @staticmethod
    def _status_targets(infos: List[TransferInfo]) -> List[Tuple[str, int]]:
        targets: List[Tuple[str, int]] = []
        for info in infos:
            if info.is_dummy:
                continue
            target = (info.endpoint, info.dst_port)
            if target not in targets:
                targets.append(target)
        return targets

    def _conclude_owned_transfer(
        self,
        room: int,
        status: KVPoll,
        room_owner: Optional[str],
        failure_reason: Optional[str] = None,
        target_infos: Optional[List[TransferInfo]] = None,
    ) -> Optional[KVPoll]:
        with self._room_lifecycle_lock:
            with self.transfer_lock:
                if room_owner is not None and self._room_owners.get(room) != room_owner:
                    return None
                infos = (
                    list(self.transfer_infos.get(room, {}).values())
                    if target_infos is None
                    else target_infos
                )
            return super().conclude_transfer(
                bootstrap_room=room,
                status=status,
                targets=self._status_targets(infos),
                failure_reason=failure_reason,
            )

    def _conclude_owned_failure(
        self,
        room: int,
        failure_reason: str,
        room_owner: Optional[str],
    ) -> Optional[KVPoll]:
        return self._conclude_owned_transfer(
            room,
            KVPoll.Failed,
            room_owner,
            failure_reason=failure_reason,
        )

    def _transfer_worker(self, queue: FastQueue, worker_index: int) -> None:
        while True:
            kv_chunk = queue.get()
            room = kv_chunk.room
            success_infos: Optional[List[TransferInfo]] = None
            try:
                success_infos = self._process_transfer_chunk(kv_chunk, worker_index)
            except Exception as exc:
                failure_reason = f"transfer worker raised: {exc!r}"
                try:
                    logger.exception(
                        "Mori transfer worker failed for room %s",
                        kv_chunk.room,
                    )
                except Exception:
                    pass
                try:
                    self._conclude_owned_failure(
                        kv_chunk.room, failure_reason, kv_chunk.room_owner
                    )
                except Exception:
                    try:
                        logger.exception(
                            "Mori transfer worker failover failed for room %s",
                            kv_chunk.room,
                        )
                    except Exception:
                        pass
            finally:
                with self._deferred_ack_lock:
                    self._staging_outstanding[room] -= 1
                    drained = self._staging_outstanding[room] <= 0
                if success_infos is not None:
                    self._conclude_owned_transfer(
                        room,
                        KVPoll.Success,
                        kv_chunk.room_owner,
                        target_infos=success_infos,
                    )
                if self.enable_deferred_decode_kv_release:
                    self._maybe_ack_drained_abort(room)
                if drained:
                    with self._deferred_ack_lock:
                        if self._staging_outstanding.get(room, 0) <= 0:
                            self._staging_outstanding.pop(room, None)
                    self._clear_room_after_drain(room)

    def _maybe_ack_drained_abort(self, room: int) -> None:
        with self._room_lifecycle_lock:
            with self.failure_lock:
                with self.transfer_lock:
                    with self._deferred_ack_lock:
                        if self._staging_outstanding.get(room, 0) <= 0:
                            owner = self._room_owners.get(room)
                            expected_generation = self._room_generations.get(room)
                            targets = self._deferred_ack_targets.get(room, ())
                            has_matching_target = any(
                                target[2] == expected_generation for target in targets
                            )
                            if (
                                owner is not None
                                and self.request_status.get(room) == KVPoll.Failed
                                and has_matching_target
                            ):
                                self._retire_room_locked(room, owner)
        super()._maybe_ack_drained_abort(room)

    def _defer_room_clear_if_outstanding(
        self, room: int, owner: Optional[str] = None
    ) -> bool:
        with self.transfer_lock:
            if owner is not None and self._room_owners.get(room) != owner:
                return False
            with self._deferred_ack_lock:
                if self._staging_outstanding.get(room, 0) <= 0:
                    return False
                if owner is not None:
                    self._rooms_pending_clear[room] = owner
            self.update_status(room, KVPoll.Failed)
            return True

    def _clear_room_after_drain(self, room: int) -> None:
        with self._room_lifecycle_lock:
            with self.failure_lock:
                with self.transfer_lock:
                    with self._deferred_ack_lock:
                        owner = self._rooms_pending_clear.get(room)
                        if self._staging_outstanding.get(room, 0) > 0 or owner is None:
                            return
                        self._rooms_pending_clear.pop(room, None)
                        if self._room_owners.get(room) == owner:
                            self._retire_room_locked(room, owner)
                        self._retired_room_status.pop((room, owner), None)
                        self._retired_room_failures.pop((room, owner), None)

    def _process_transfer_chunk(
        self, kv_chunk: TransferKVChunk, worker_index: int = 0
    ) -> Optional[List[TransferInfo]]:
        room = kv_chunk.room
        room_owner = kv_chunk.room_owner
        if self._should_skip_transfer(room, room_owner):
            return

        if kv_chunk.wait_event is not None:
            kv_chunk.wait_event.synchronize()

        if self._should_skip_transfer(room, room_owner):
            return

        try:
            statuses, target_infos = self._submit_kv_transfer(
                room,
                kv_chunk.prefill_kv_indices,
                kv_chunk.index_slice,
                kv_chunk.is_last_chunk,
                aux_index=kv_chunk.prefill_aux_index,
                state_indices=kv_chunk.state_indices,
                num_kv_tokens=kv_chunk.num_kv_tokens,
                worker_index=worker_index,
                room_owner=room_owner,
            )
        except MoriKVSubmissionError as exc:
            self._wait_transfer_completion(exc.statuses)
            raise

        failure_reason = self._wait_transfer_completion(statuses)
        if self._should_skip_transfer(room, room_owner):
            return
        if failure_reason is not None:
            self._conclude_owned_failure(room, failure_reason, room_owner)
            return None

        if kv_chunk.is_last_chunk:
            return target_infos if target_infos is not None else []
        return None

    def _should_skip_transfer(
        self, room: int, room_owner: Optional[str] = None
    ) -> bool:
        if (
            (room_owner is not None and self._room_owners.get(room) != room_owner)
            or room not in self.request_status
            or self.check_status(room) == KVPoll.Failed
        ):
            logger.debug(
                "Skipping chunk for room %s because it has already failed or been aborted",
                room,
            )
            return True
        return False

    def _wait_transfer_completion(
        self, statuses: List[TransferStatus]
    ) -> Optional[str]:
        if not statuses:
            return None

        start = time.perf_counter()
        sla_ms = self._transfer_timeout_ms

        while True:
            rc = self.engine.wait_all(statuses, timeout_ms=self._wait_poll_ms)
            if rc == StatusCode.SUCCESS:
                return None
            if rc != StatusCode.IN_PROGRESS:
                failure_reason = self._collect_transfer_failure_reason(statuses)
                self.engine.wait_all(statuses, timeout_ms=-1)
                return failure_reason
            if sla_ms > 0 and (time.perf_counter() - start) * 1000 >= sla_ms:
                timeout_reason = f"KV transfer exceeded SLA {sla_ms}ms"
                logger.error(
                    "%s; waiting for all in-flight MORI writes to drain",
                    timeout_reason,
                )
                self.engine.wait_all(statuses, timeout_ms=-1)
                return timeout_reason

    @staticmethod
    def _collect_transfer_failure_reason(statuses: List[TransferStatus]) -> str:
        for status in statuses:
            if status.Failed():
                return f"KV transfer failed: {status.Message()}"
        return "KV transfer failed due to unknown reason"

    def add_transfer_request(
        self,
        bootstrap_room: int,
        kv_indices: npt.NDArray[np.int32],
        index_slice: slice,
        is_last_chunk: bool,
        aux_index: Optional[int] = None,
        state_indices: Optional[List] = None,
        num_kv_tokens: Optional[int] = None,
        wait_event: Optional[object] = None,
        room_owner: Optional[str] = None,
    ) -> None:
        assert self.disaggregation_mode == DisaggregationMode.PREFILL
        assert not is_last_chunk or (is_last_chunk and aux_index is not None)

        with self.transfer_lock:
            if (
                bootstrap_room not in self.request_status
                or self.check_status(bootstrap_room) == KVPoll.Failed
                or (
                    room_owner is not None
                    and self._room_owners.get(bootstrap_room) != room_owner
                )
            ):
                logger.debug(
                    "Request with bootstrap_room=%s already failed", bootstrap_room
                )
                return

            if bootstrap_room not in self.transfer_infos:
                return

            infos = self.transfer_infos[bootstrap_room].values()
            requires_pack = self._dcp_pack_worker_count > 0 and any(
                (peer := self.decode_kv_args_table.get(info.engine_key)) is not None
                and peer.requires_dcp_relayout
                for info in infos
            )
            shard_idx = bootstrap_room % (
                self._dcp_pack_worker_count if requires_pack else self._num_shards
            )
            with self._deferred_ack_lock:
                self._staging_outstanding[bootstrap_room] += 1
        try:
            self._transfer_queues[shard_idx].put(
                TransferKVChunk(
                    room=bootstrap_room,
                    prefill_kv_indices=kv_indices,
                    index_slice=index_slice,
                    is_last_chunk=is_last_chunk,
                    prefill_aux_index=aux_index,
                    state_indices=state_indices,
                    num_kv_tokens=num_kv_tokens,
                    wait_event=wait_event,
                    room_owner=room_owner,
                )
            )
        except Exception:
            with self._deferred_ack_lock:
                self._staging_outstanding[bootstrap_room] -= 1
            if self.enable_deferred_decode_kv_release:
                self._maybe_ack_drained_abort(bootstrap_room)
            raise

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

    def _handle_register_message(
        self, payload: List[bytes], supports_dcp_drain: bool = False
    ) -> None:
        try:
            register_info = KVArgsRegisterInfo.from_zmq(payload)
            register_info.supports_dcp_drain = supports_dcp_drain
            if register_info.dst_dcp_size > 1 and not register_info.dcp_registration_id:
                raise RuntimeError(
                    "MORI DCP peer does not request a registration capability ACK"
                )
            self._add_remote_peer(register_info)
            if register_info.dcp_registration_id is not None:
                na = NetworkAddress(register_info.endpoint, register_info.dst_port)
                socket = self._connect_threadsafe(na.to_tcp(), is_ipv6=na.is_ipv6)
                socket.send_multipart(
                    [
                        MORI_DCP_REGISTER_ACK,
                        register_info.dcp_registration_id.encode("ascii"),
                    ]
                )
        except Exception:
            logger.exception("Failed to register remote peer")

    def _handle_transfer_message(self, payload: List[bytes]) -> None:
        try:
            transfer_info = TransferInfo.from_zmq(payload)
            peer_info = self.decode_kv_args_table.get(transfer_info.engine_key)
            requires_generation = peer_info is not None and peer_info.dst_dcp_size > 1
            if requires_generation and transfer_info.abort_generation is None:
                with self.transfer_lock:
                    has_owner = transfer_info.room in self._room_owners
                    if has_owner:
                        self.update_status(transfer_info.room, KVPoll.Failed)
                if has_owner:
                    self.record_failure(
                        transfer_info.room,
                        "MORI DCP metadata is missing its request epoch",
                    )
                logger.warning(
                    "Rejecting MORI DCP metadata without a request epoch for room %s",
                    transfer_info.room,
                )
                return

            with self.transfer_lock:
                if (
                    transfer_info.abort_generation is not None
                    and (
                        transfer_info.room,
                        transfer_info.abort_generation,
                    )
                    in self._aborted_generations
                ):
                    logger.debug(
                        "Ignoring aborted MORI metadata for room %s generation %s",
                        transfer_info.room,
                        transfer_info.abort_generation,
                    )
                    return

                expected_generation = self._room_generations.get(transfer_info.room)
                if (
                    expected_generation is not None
                    and transfer_info.abort_generation != expected_generation
                ):
                    logger.debug(
                        "Ignoring stale MORI metadata for room %s generation %s",
                        transfer_info.room,
                        transfer_info.abort_generation,
                    )
                    return

                current = self.request_status.get(transfer_info.room)
                if current is not None and current != KVPoll.Bootstrapping:
                    logger.debug(
                        "Ignoring stale transfer info for room %s (status=%s)",
                        transfer_info.room,
                        current,
                    )
                    return
                if (
                    transfer_info.room in self._room_owners
                    and expected_generation is None
                    and transfer_info.abort_generation is not None
                ):
                    self._room_generations[transfer_info.room] = (
                        transfer_info.abort_generation
                    )
                    self._aborted_generations = {
                        entry: None
                        for entry in self._aborted_generations
                        if entry[0] != transfer_info.room
                    }

                # Metadata may arrive before the prefill scheduler creates
                # the sender, so an owner-less room is intentionally valid.
                infos = self.transfer_infos.setdefault(transfer_info.room, {})
                infos[transfer_info.engine_key] = transfer_info

                if (
                    infos is not None
                    and len(infos) >= transfer_info.required_dst_info_num
                ):
                    self.resolve_kv_replica_factor(infos)
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
        if not msg or msg[0] not in (
            MORI_GUARD,
            MORI_DCP_GUARD,
            MORI_DCP_DRAIN_GUARD,
        ):
            logger.warning("Received malformed bootstrap message")
            return None
        payload = msg[1:]
        if not payload:
            return None
        return payload

    def _handle_abort_message(self, msg: List[bytes]) -> None:
        """Handle best-effort ABORT notifications from the decode side."""
        if len(msg) < 2:
            logger.warning("Malformed ABORT message: too few frames (%d)", len(msg))
            return

        try:
            bootstrap_room = int(msg[1].decode("ascii"))
            decode_ip = msg[2].decode("ascii") if len(msg) > 2 else None
            decode_port = int(msg[3].decode("ascii")) if len(msg) > 3 else None
            generation = msg[4].decode("ascii") if len(msg) > 4 and msg[4] else None
        except (ValueError, UnicodeDecodeError):
            logger.warning("Malformed ABORT message: invalid room field %r", msg[1])
            return

        with self.transfer_lock:
            current = self.request_status.get(bootstrap_room)
            room_owner = self._room_owners.get(bootstrap_room)
            expected_generation = self._room_generations.get(bootstrap_room)
            if expected_generation is None:
                info_generations = {
                    info.abort_generation
                    for info in self.transfer_infos.get(bootstrap_room, {}).values()
                    if info.abort_generation is not None
                }
                if len(info_generations) == 1:
                    expected_generation = next(iter(info_generations))
                    if room_owner is not None:
                        self._room_generations[bootstrap_room] = expected_generation

            stale_generation = (
                expected_generation is not None and generation != expected_generation
            )
            if not stale_generation and generation is not None:
                key = (bootstrap_room, generation)
                self._aborted_generations.pop(key, None)
                self._aborted_generations[key] = None
                if len(self._aborted_generations) > MORI_ABORT_TOMBSTONE_LIMIT:
                    oldest = next(iter(self._aborted_generations))
                    self._aborted_generations.pop(oldest)
            generation_matches = (
                expected_generation is not None and generation == expected_generation
            ) or (expected_generation is None and generation is None)
            room_active = room_owner is not None and current not in (
                None,
                KVPoll.Success,
                KVPoll.Failed,
            )
            if room_owner is None and generation_matches:
                self.transfer_infos.pop(bootstrap_room, None)
                self.request_status.pop(bootstrap_room, None)
                self.req_to_decode_prefix_len.pop(bootstrap_room, None)
            if room_active and generation_matches:
                self.update_status(bootstrap_room, KVPoll.Failed)

        if stale_generation:
            if decode_ip is not None and decode_port is not None:
                self.register_deferred_ack_target(
                    bootstrap_room, decode_ip, decode_port, generation
                )
                self._maybe_ack_drained_abort(bootstrap_room)
            return

        if room_active:
            logger.debug("Room %s marked Failed via ABORT from decode", bootstrap_room)
        else:
            logger.debug(
                "ABORT for inactive room %s; checking outstanding writes",
                bootstrap_room,
            )

        if (
            self.enable_deferred_decode_kv_release
            and decode_ip is not None
            and decode_port is not None
        ):
            self.register_deferred_ack_target(
                bootstrap_room, decode_ip, decode_port, generation
            )
            self._maybe_ack_drained_abort(bootstrap_room)

    def _start_bootstrap_thread(self) -> None:
        def bootstrap_worker():
            while True:
                try:
                    msg = self.server_socket.recv_multipart()
                    if not msg:
                        continue

                    tag = msg[0]
                    if tag == _TAG_ABORT:
                        self._handle_abort_message(msg)
                        continue

                    payload = self._validate_message(msg)
                    if payload is None:
                        continue
                    room = payload[0].decode("ascii")

                    if room == "None":
                        self._handle_register_message(
                            payload,
                            supports_dcp_drain=tag == MORI_DCP_DRAIN_GUARD,
                        )
                    else:
                        self._handle_transfer_message(payload)
                except Exception:
                    logger.exception("Bootstrap worker failed")

        threading.Thread(target=bootstrap_worker, daemon=True).start()

    def activate_decode_room_generation(self, room: int, generation: str) -> None:
        with self.transfer_lock:
            current = self._decode_room_generations.get(room)
            if current is not None and current != generation:
                raise KVTransferError(
                    room,
                    f"Cannot reuse active MORI decode bootstrap room {room}",
                )
            self._decode_room_generations[room] = generation

    def clear_decode_room_generation(self, room: int, generation: str) -> None:
        with self.transfer_lock:
            if self._decode_room_generations.get(room) == generation:
                self._decode_room_generations.pop(room, None)

    def _is_current_decode_generation(
        self, room: int, generation: Optional[str]
    ) -> bool:
        if not self.requires_strict_deferred_release:
            return True
        with self.transfer_lock:
            expected_generation = self._decode_room_generations.get(room)
        return expected_generation is not None and generation == expected_generation

    def _start_decode_thread(self) -> None:
        def decode_worker():
            while True:
                try:
                    msg = self.server_socket.recv_multipart()
                    if self._handle_dcp_registration_ack(msg):
                        continue
                    if self._handle_abort_ack(msg):
                        continue
                    if msg and msg[0] == MoriKVManager.AUX_DATA_HEADER:
                        self._handle_aux_data(msg)
                        continue

                    parsed = self.parse_kv_status_message(msg)
                    if parsed is None:
                        logger.warning(
                            "Received malformed status message on decode worker"
                        )
                        continue
                    room, status, prefill_rank, reason = parsed
                    generation = (
                        msg[5].decode("ascii") if len(msg) > 5 and msg[5] else None
                    )
                    if not self._is_current_decode_generation(room, generation):
                        logger.debug(
                            "Dropping stale MORI status for room %s generation %s",
                            room,
                            generation,
                        )
                        continue
                    self.apply_prefill_status(
                        bootstrap_room=room,
                        status=status,
                        prefill_rank=prefill_rank,
                        failure_reason=reason,
                    )
                except Exception:
                    logger.exception("Decode status worker failed")

        threading.Thread(target=decode_worker, daemon=True).start()

    def _handle_dcp_registration_ack(self, msg: List[bytes]) -> bool:
        if not msg or msg[0] != MORI_DCP_REGISTER_ACK:
            return False
        if len(msg) != 2 or not msg[1]:
            logger.warning("Malformed MORI DCP registration acknowledgement")
            return True
        try:
            registration_id = msg[1].decode("ascii")
        except UnicodeDecodeError:
            logger.warning("Malformed MORI DCP registration acknowledgement token")
            return True
        with self._dcp_registration_lock:
            event = self._dcp_registration_events.get(registration_id)
        if event is not None:
            event.set()
        else:
            logger.debug(
                "Dropping late MORI DCP registration acknowledgement %s",
                registration_id,
            )
        return True

    def _handle_abort_ack(self, msg: List[bytes]) -> bool:
        if not msg or msg[0] != b"ABORT_ACK":
            return False
        if self.enable_deferred_decode_kv_release and len(msg) >= 3:
            self.note_abort_ack(
                int(msg[1].decode("ascii")),
                int(msg[2].decode("ascii")),
                msg[3].decode("ascii") if len(msg) > 3 and msg[3] else None,
            )
        return True

    def _add_remote_peer(self, register_info: KVArgsRegisterInfo) -> None:
        engine_key = register_info.engine_key
        if engine_key in self.decode_kv_args_table:
            logger.debug("Remote peer %s already registered. Skipping.", engine_key)
            return
        if register_info.dst_dcp_size > 1:
            if not register_info.supports_dcp_drain:
                raise RuntimeError("MORI DCP peer does not advertise drain-ACK support")
            self.enable_deferred_decode_kv_release = True
        register_info.requires_dcp_relayout = self.requires_dcp_relayout(
            register_info.dst_dcp_size, register_info.dst_dcp_rank
        )
        if register_info.requires_dcp_relayout:
            if self.kv_args.kv_layer_ids or register_info.dst_kv_layer_ids:
                dst_indices = resolve_dcp_dst_entry_indices(
                    self.kv_args.kv_layer_ids,
                    register_info.dst_kv_layer_ids,
                    len(self.kv_mem_descs),
                    len(register_info.dst_kv_mem_descs),
                )
            else:
                _, dst_indices, _ = self.get_mla_kv_ptrs_with_pp(
                    list(range(len(self.kv_mem_descs))),
                    list(range(len(register_info.dst_kv_mem_descs))),
                )
            register_info.dcp_dst_region_indices = dst_indices
            register_info.dcp_token_item_lens = self.prepare_dcp_token_item_lens(
                [register_info.dst_kv_item_lens[index] for index in dst_indices],
                register_info.dst_dcp_size,
            )
            self._init_dcp_pack_buffers_once(register_info.dst_dcp_size)
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

        # Both peers expose the same PP-local layout. Their descriptor indices
        # are already aligned, so applying the Prefill rank's global layer
        # offset would incorrectly index into a local list.
        if len(src_descs) == len(dst_mem_descs):
            dst_k_descs = dst_mem_descs[:num_local_layers]
            dst_v_descs = dst_mem_descs[num_local_layers:]
            return (
                src_k_descs,
                src_v_descs,
                dst_k_descs,
                dst_v_descs,
                num_local_layers,
            )

        start_layer = self.kv_args.prefill_start_layer
        end_layer = start_layer + num_local_layers
        dst_total_layers = len(dst_mem_descs) // 2
        if len(dst_mem_descs) < 2 or end_layer > dst_total_layers:
            raise ValueError(
                "Destination KV descriptors do not match prefill pp configuration"
            )
        dst_k_descs = dst_mem_descs[start_layer:end_layer]
        if (
            num_local_layers < dst_total_layers
            and dst_total_layers % num_local_layers != 0
        ):
            # Decode has draft-model KV while Prefill has target-model KV only:
            # [K_main..., V_main..., draft_K..., draft_V...].
            multiplier_ratio = dst_total_layers // num_local_layers
            dst_v_offset = num_local_layers * multiplier_ratio
        else:
            dst_v_offset = dst_total_layers
        dst_v_descs = dst_mem_descs[
            dst_v_offset + start_layer : dst_v_offset + end_layer
        ]
        return src_k_descs, src_v_descs, dst_k_descs, dst_v_descs, num_local_layers

    def _get_mla_mem_desc_slices(
        self, dst_mem_descs: List[MemoryDesc]
    ) -> tuple[List[MemoryDesc], List[MemoryDesc], int]:
        src_descs = self.kv_mem_descs
        num_local_layers = len(src_descs)
        # Same-PP peers register matching local descriptor lists.
        if len(src_descs) == len(dst_mem_descs):
            return src_descs, dst_mem_descs, num_local_layers

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
        status_sink: Optional[List[TransferStatus]] = None,
    ) -> List[TransferStatus]:
        if plan.empty():
            return status_sink if status_sink is not None else []

        transfer_uid = self.engine.allocate_transfer_uid()

        statuses = list(
            self.engine.batch_write(
                [src_desc],
                [plan.local_offsets],
                [dst_desc],
                [plan.remote_offsets],
                [plan.sizes],
                [transfer_uid],
            )
        )
        if status_sink is not None:
            # Retain each status before a later submission can fail so the
            # worker drains every operation before reusing transfer memory.
            status_sink.extend(statuses)
        return statuses

    def _build_contiguous_transfer_plan(
        self, grouped_plan: GroupedIndexPlan, item_len: int
    ) -> BatchTransferPlan:
        # Reuse grouped indices across all layers/tensors that share the same item length.
        return grouped_plan.materialize(item_len)

    def _build_tp_slice_config(self, peer_info: KVArgsRegisterInfo) -> TPSliceConfig:
        page_size = self.kv_args.page_size

        src_item_len = self.kv_args.kv_item_lens[0]
        dst_item_len = peer_info.dst_kv_item_len

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
        status_sink: Optional[List[TransferStatus]] = None,
    ) -> List[TransferStatus]:
        grouped_plan = GroupedIndexPlan.from_groups(
            *group_concurrent_contiguous(
                prefill_kv_indices,
                dst_kv_indices,
            )
        )
        statuses = status_sink if status_sink is not None else []
        kv_item_len = self.kv_args.kv_item_lens[0]

        if self.is_mla_backend or self.is_hybrid_mla_backend:
            src_descs, dst_descs, layers_current_pp_stage = (
                self._get_mla_mem_desc_slices(peer_info.dst_kv_mem_descs)
            )
            for layer_id in range(layers_current_pp_stage):
                layer_plan = self._build_contiguous_transfer_plan(
                    grouped_plan, self.kv_args.kv_item_lens[layer_id]
                )
                self._submit_batch_transfer_plan(
                    src_descs[layer_id],
                    dst_descs[layer_id],
                    layer_plan,
                    status_sink=statuses,
                )
            return statuses

        (
            src_k_descs,
            src_v_descs,
            dst_k_descs,
            dst_v_descs,
            layers_current_pp_stage,
        ) = self._get_mha_mem_desc_slices(peer_info.dst_kv_mem_descs)

        if peer_info.decode_tp_size != self.attn_tp_size:
            tp_cfg = self._build_tp_slice_config(peer_info)
            slice_plan = self._build_tp_slice_transfer_plan(
                prefill_kv_indices, dst_kv_indices, tp_cfg
            )
            for layer_id in range(layers_current_pp_stage):
                self._submit_batch_transfer_plan(
                    src_k_descs[layer_id],
                    dst_k_descs[layer_id],
                    slice_plan,
                    status_sink=statuses,
                )
                self._submit_batch_transfer_plan(
                    src_v_descs[layer_id],
                    dst_v_descs[layer_id],
                    slice_plan,
                    status_sink=statuses,
                )
            return statuses

        layer_plan = self._build_contiguous_transfer_plan(grouped_plan, kv_item_len)
        for layer_id in range(layers_current_pp_stage):
            self._submit_batch_transfer_plan(
                src_k_descs[layer_id],
                dst_k_descs[layer_id],
                layer_plan,
                status_sink=statuses,
            )
            self._submit_batch_transfer_plan(
                src_v_descs[layer_id],
                dst_v_descs[layer_id],
                layer_plan,
                status_sink=statuses,
            )
        return statuses

    def _submit_dcp_part(
        self,
        src_descs: List[MemoryDesc],
        dst_descs: List[MemoryDesc],
        token_item_lens: List[int],
        src_token_indices: npt.NDArray[np.int64],
        dst_token_indices: npt.NDArray[np.int64],
        statuses: List[TransferStatus],
        packed_src: Optional[MoriPackedDCPSource] = None,
    ) -> None:
        if src_token_indices.size == 0:
            return
        if not (len(src_descs) == len(dst_descs) == len(token_item_lens)):
            raise RuntimeError(
                "MORI DCP transfer-part region count differs: "
                f"src={len(src_descs)}, dst={len(dst_descs)}, "
                f"item_lens={len(token_item_lens)}"
            )

        src_layer_offsets = [0] * len(src_descs)
        if packed_src is not None:
            if len(packed_src.layer_offsets) != len(src_descs):
                raise RuntimeError(
                    "MORI packed DCP source region count differs: "
                    f"packed={len(packed_src.layer_offsets)}, src={len(src_descs)}"
                )
            src_descs = [packed_src.mem_desc] * len(packed_src.layer_offsets)
            src_layer_offsets = packed_src.layer_offsets
            src_token_indices = packed_src.token_indices

        grouped_plan = GroupedIndexPlan.from_groups(
            *group_concurrent_contiguous(
                src_token_indices,
                dst_token_indices,
            )
        )
        for src_desc, dst_desc, token_item_len, src_layer_offset in zip(
            src_descs,
            dst_descs,
            token_item_lens,
            src_layer_offsets,
        ):
            transfer_plan = self._build_contiguous_transfer_plan(
                grouped_plan, token_item_len
            )
            if src_layer_offset:
                transfer_plan = dataclasses.replace(
                    transfer_plan,
                    local_offsets=[
                        src_layer_offset + offset
                        for offset in transfer_plan.local_offsets
                    ],
                )
            self._submit_batch_transfer_plan(
                src_desc,
                dst_desc,
                transfer_plan,
                status_sink=statuses,
            )

    def send_kvcache_dcp(
        self,
        peer_info: KVArgsRegisterInfo,
        plan: DCPTokenTransferPlan,
        packed_src: Optional[MoriPackedDCPSource] = None,
        status_sink: Optional[List[TransferStatus]] = None,
    ) -> List[TransferStatus]:
        statuses = status_sink if status_sink is not None else []
        if plan.empty():
            return statuses
        if (
            peer_info.dcp_dst_region_indices is None
            or peer_info.dcp_token_item_lens is None
        ):
            raise RuntimeError("MORI DCP peer transfer metadata is not initialized")

        dst_descs = [
            peer_info.dst_kv_mem_descs[index]
            for index in peer_info.dcp_dst_region_indices
        ]
        if not (
            len(self.kv_mem_descs)
            == len(dst_descs)
            == len(peer_info.dcp_token_item_lens)
        ):
            raise RuntimeError(
                "MORI DCP source/destination KV region count differs: "
                f"src={len(self.kv_mem_descs)}, dst={len(dst_descs)}, "
                f"item_lens={len(peer_info.dcp_token_item_lens)}"
            )

        num_draft = getattr(self.kv_args, "num_draft_entries", 0)
        num_target = len(self.kv_mem_descs) - num_draft
        if num_target < 0:
            raise RuntimeError(
                f"MORI DCP has num_draft_entries={num_draft} but only "
                f"{len(self.kv_mem_descs)} KV regions"
            )

        self._submit_dcp_part(
            self.kv_mem_descs[:num_target],
            dst_descs[:num_target],
            peer_info.dcp_token_item_lens[:num_target],
            plan.target_src_token_indices,
            plan.target_dst_token_indices,
            statuses,
            packed_src=packed_src,
        )
        if num_draft:
            self._submit_dcp_part(
                self.kv_mem_descs[num_target:],
                dst_descs[num_target:],
                peer_info.dcp_token_item_lens[num_target:],
                plan.draft_src_token_indices,
                plan.draft_dst_token_indices,
                statuses,
            )
        return statuses

    def _pack_dcp_rank_once(
        self,
        pack_buffer: Optional[StagingBuffer],
        peer_info: KVArgsRegisterInfo,
        src_token_indices: npt.NDArray[np.int64],
        packed_source_by_dcp_rank: Dict[int, MoriPackedDCPRank],
    ) -> Optional[MoriPackedDCPSource]:
        rank = peer_info.dst_dcp_rank
        token_item_lens = peer_info.dcp_token_item_lens
        assert token_item_lens is not None
        num_draft = getattr(self.kv_args, "num_draft_entries", 0)
        num_target = len(self.kv_args.kv_data_ptrs) - num_draft
        token_item_lens = token_item_lens[:num_target]
        signature = (
            peer_info.dst_dcp_size,
            tuple(token_item_lens),
            src_token_indices.dtype.str,
            src_token_indices.tobytes(),
        )
        if rank in packed_source_by_dcp_rank:
            cached = packed_source_by_dcp_rank[rank]
            if cached.signature == signature:
                return cached.source
            logger.warning(
                "MORI DCP rank %d has inconsistent source geometry within one "
                "chunk; using per-token RDMA for this target",
                rank,
            )
            return None
        if pack_buffer is None or src_token_indices.size == 0:
            packed_source_by_dcp_rank[rank] = MoriPackedDCPRank(signature, None)
            return None

        from sglang.srt.disaggregation.common.dcp_pack import try_pack_dcp_src

        rank_stride = pack_buffer.get_size() // peer_info.dst_dcp_size
        packed = try_pack_dcp_src(
            pack_buffer=pack_buffer,
            kv_data_ptrs=self.kv_args.kv_data_ptrs[:num_target],
            src_token_indices=src_token_indices,
            token_item_lens=token_item_lens,
            pack_offset_bytes=rank * rank_stride,
            pack_limit_bytes=(rank + 1) * rank_stride,
        )
        if packed is None:
            packed_source_by_dcp_rank[rank] = MoriPackedDCPRank(signature, None)
            return None

        pack_base = pack_buffer.get_ptr()
        pack_desc = self._dcp_pack_mem_descs.get(pack_base)
        if pack_desc is None:
            raise RuntimeError("MORI DCP pack buffer is not registered")
        packed_ptrs, packed_indices = packed
        packed_source = MoriPackedDCPSource(
            mem_desc=pack_desc,
            layer_offsets=[ptr - pack_base for ptr in packed_ptrs],
            token_indices=packed_indices,
        )
        packed_source_by_dcp_rank[rank] = MoriPackedDCPRank(signature, packed_source)
        return packed_source

    def send_aux(
        self,
        peer_info: KVArgsRegisterInfo,
        prefill_aux_index: int,
        dst_aux_index: int,
        room: int,
        generation: Optional[str] = None,
        status_sink: Optional[List[TransferStatus]] = None,
    ) -> List[TransferStatus]:
        if self._send_aux_rdma:
            return self.send_aux_rdma(
                peer_info,
                prefill_aux_index,
                dst_aux_index,
                room,
                generation,
                status_sink,
            )
        self.send_aux_tcp(
            peer_info,
            prefill_aux_index,
            dst_aux_index,
            room,
            generation,
        )
        return status_sink if status_sink is not None else []

    def send_aux_rdma(
        self,
        peer_info: KVArgsRegisterInfo,
        prefill_aux_index: int,
        dst_aux_index: int,
        room: int,
        generation: Optional[str] = None,
        status_sink: Optional[List[TransferStatus]] = None,
    ) -> List[TransferStatus]:
        if not self.aux_mem_descs or len(self.aux_mem_descs) != len(
            peer_info.dst_aux_mem_descs
        ):
            self.send_aux_tcp(
                peer_info,
                prefill_aux_index,
                dst_aux_index,
                room,
                generation,
            )
            return status_sink if status_sink is not None else []

        statuses = status_sink if status_sink is not None else []
        for i in range(len(self.aux_mem_descs)):
            item_len = self.kv_args.aux_item_lens[i]
            self._submit_batch_transfer_plan(
                self.aux_mem_descs[i],
                peer_info.dst_aux_mem_descs[i],
                BatchTransferPlan(
                    local_offsets=[prefill_aux_index * item_len],
                    remote_offsets=[dst_aux_index * item_len],
                    sizes=[item_len],
                ),
                status_sink=statuses,
            )
        return statuses

    def send_aux_tcp(
        self,
        peer_info: KVArgsRegisterInfo,
        prefill_aux_index: int,
        dst_aux_index: int,
        room: int,
        generation: Optional[str] = None,
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
                generation=generation,
            )
        return []  # TCP path has no TransferStatus to poll

    def _send_aux_data_to_endpoint(
        self, remote, dst_port, room, buffer_index, aux_index, data, generation=None
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
                generation.encode("ascii") if generation is not None else b"",
            ]
        )

    def send_state(
        self,
        peer_info: KVArgsRegisterInfo,
        src_state_indices: List[npt.NDArray[np.int32]],
        dst_state_indices: List[npt.NDArray[np.int32]],
        status_sink: Optional[List[TransferStatus]] = None,
    ) -> List[TransferStatus]:
        # Guard: no local state tensors -> no-op (e.g. SWA layers=0 on this PP rank)
        if not self.state_mem_descs:
            return status_sink if status_sink is not None else []

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

        statuses = status_sink if status_sink is not None else []
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
                if peer_info.decode_tp_size != self.attn_tp_size and 0 in src_dims:
                    raise RuntimeError(
                        "Replicated Mamba PD state transfer currently requires "
                        "matching prefill/decode attention TP sizes"
                    )
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
                    status_sink=statuses,
                )
            elif st in (
                "swa",
                "dsa",
                "qsa_pending",
                "qsa_compressed",
                "swa_ring",
                "c128_state",
                "minimax_index_k",
            ):
                self._send_swa_dsa_state(
                    peer_info,
                    src_indices,
                    dst_indices,
                    src_descs,
                    src_lens,
                    dst_descs,
                    st,
                    status_sink=statuses,
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
        status_sink: Optional[List[TransferStatus]] = None,
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
        statuses = status_sink if status_sink is not None else []

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

            self._submit_batch_transfer_plan(
                src_desc,
                dst_desc,
                BatchTransferPlan(
                    local_offsets=[src_offset],
                    remote_offsets=[dst_offset],
                    sizes=[size],
                ),
                status_sink=statuses,
            )

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
        status_sink: Optional[List[TransferStatus]] = None,
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
        if state_type in ("qsa_pending", "qsa_compressed", "minimax_index_k"):
            if self.pp_size is not None and self.pp_size > 1:
                # MORI registration does not exchange state_layer_ids. Compact
                # sparse-state lists therefore cannot be paired safely across
                # pipeline stages until that metadata is added to its protocol.
                raise RuntimeError(
                    f"MORI PD disaggregation requires PP=1 for {state_type}; "
                    "PP>1 needs peer state_layer_ids for global-layer descriptor "
                    "pairing."
                )
            if peer_info.decode_tp_size != self.attn_tp_size:
                raise RuntimeError(
                    f"PD disagg: heterogeneous TP not supported for {state_type} yet."
                )

        common_len = min(src_state_indices.size, dst_state_indices.size)
        if (
            state_type == "c128_state"
            and common_len == 0
            and src_state_indices.size == 0
            and dst_state_indices.size == 0
        ):
            return []
        if common_len == 0 and max(src_state_indices.size, dst_state_indices.size) > 0:
            raise RuntimeError(
                f"No overlapping state indices for state_type={state_type}"
            )
        if src_state_indices.size != dst_state_indices.size:
            # These components are position- or request-indexed: truncating
            # silently misaligns rows and corrupts KV. Paged swa/dsa tolerate
            # a 1-page drift -> keep truncation.
            if state_type in (
                "qsa_pending",
                "qsa_compressed",
                "swa_ring",
                "c128_state",
            ):
                raise RuntimeError(
                    f"{state_type.upper()} state index length mismatch: "
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

        statuses = status_sink if status_sink is not None else []
        for i, src_desc in enumerate(src_state_mem_descs):
            dst_desc = dst_state_mem_descs[i]
            state_item_len = src_state_item_lens[i]

            self._submit_batch_transfer_plan(
                src_desc,
                dst_desc,
                self._build_contiguous_transfer_plan(grouped_plan, state_item_len),
                status_sink=statuses,
            )

        return statuses

    def _handle_aux_data(self, msg: List[bytes]):
        """Handle AUX_DATA messages received by the decode thread (legacy TCP path)."""
        room = int(msg[1].decode("ascii"))
        buffer_index = int(msg[2].decode("ascii"))
        aux_index = int(msg[3].decode("ascii"))
        data_length = struct.unpack(">I", msg[4])[0]
        data = msg[5]
        generation = msg[6].decode("ascii") if len(msg) > 6 and msg[6] else None

        if not self._is_current_decode_generation(room, generation):
            logger.debug(
                "Dropping stale MORI AUX data for room %s generation %s",
                room,
                generation,
            )
            return

        if len(data) != data_length:
            logger.error(f"AUX_DATA length mismatch for bootstrap_room {room}")
            return

        AuxDataCodec.deserialize_data_to_buffer(
            self.kv_args, buffer_index, aux_index, data
        )

    def _submit_kv_transfer(
        self,
        bootstrap_room: int,
        kv_indices: npt.NDArray[np.int32],
        index_slice: slice,
        is_last_chunk: bool,
        aux_index: Optional[int] = None,
        state_indices: Optional[List[npt.NDArray[np.int32]]] = None,
        num_kv_tokens: Optional[int] = None,
        worker_index: Optional[int] = None,
        room_owner: Optional[str] = None,
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
            current = self.request_status.get(bootstrap_room)
            if (
                current is None
                or current == KVPoll.Failed
                or (
                    room_owner is not None
                    and self._room_owners.get(bootstrap_room) != room_owner
                )
            ):
                return [], None

            transfer_infos = self.transfer_infos.get(bootstrap_room)
            if not transfer_infos:
                raise RuntimeError(
                    f"No transfer info found for bootstrap_room={bootstrap_room}"
                )

            self.update_status(bootstrap_room, KVPoll.Transferring)
            for info in transfer_infos.values():
                peer_info = self.decode_kv_args_table.get(info.engine_key)
                if not peer_info:
                    raise RuntimeError(
                        f"Peer info missing for engine {info.engine_key}"
                    )
                targets.append(TransferTarget(info=info, peer_info=peer_info))
            if is_last_chunk:
                target_infos_snapshot = [
                    info for info in transfer_infos.values() if not info.is_dummy
                ]

        pack_buffer = (
            self._get_or_init_dcp_pack_buffer(worker_index)
            if worker_index is not None
            and any(target.peer_info.requires_dcp_relayout for target in targets)
            else None
        )
        result_statuses: List[TransferStatus] = []
        packed_source_by_dcp_rank: Dict[int, MoriPackedDCPRank] = {}
        try:
            for target in targets:
                info = target.info
                peer_info = target.peer_info

                if not info.is_dummy:
                    if peer_info.requires_dcp_relayout:
                        if num_kv_tokens is None:
                            raise ValueError("PD DCP transfer requires num_kv_tokens")
                        plan = build_dcp_token_transfer_plan(
                            kv_indices,
                            info.dst_kv_indices,
                            physical_page_size=self.kv_args.page_size,
                            dcp_size=peer_info.dst_dcp_size,
                            dcp_rank=peer_info.dst_dcp_rank,
                            src_page_offset=index_slice.start or 0,
                            decode_prefix_len=info.decode_prefix_len or 0,
                            num_kv_tokens=num_kv_tokens,
                        )
                        packed_src = (
                            self._pack_dcp_rank_once(
                                pack_buffer,
                                peer_info,
                                plan.target_src_token_indices,
                                packed_source_by_dcp_rank,
                            )
                            if pack_buffer is not None
                            else None
                        )
                        self.send_kvcache_dcp(
                            peer_info,
                            plan,
                            packed_src,
                            status_sink=result_statuses,
                        )
                    else:
                        dst_indices_chunk = info.dst_kv_indices[index_slice]
                        self.send_kvcache(
                            peer_info,
                            kv_indices,
                            dst_indices_chunk,
                            status_sink=result_statuses,
                        )

                if (
                    is_last_chunk
                    and state_indices is not None
                    and not info.is_dummy
                    and self.state_mem_descs
                ):
                    self.send_state(
                        peer_info,
                        state_indices,
                        info.dst_state_indices,
                        status_sink=result_statuses,
                    )

                if (
                    is_last_chunk
                    and aux_index is not None
                    and info.dst_aux_index >= 0
                    and self.pp_group.is_last_rank
                ):
                    self.send_aux(
                        peer_info,
                        aux_index,
                        info.dst_aux_index,
                        bootstrap_room,
                        info.abort_generation,
                        status_sink=result_statuses,
                    )
        except Exception as e:
            logger.exception(
                "Mori KV transfer submission failed for bootstrap_room=%s",
                bootstrap_room,
            )
            raise MoriKVSubmissionError(result_statuses, e) from e

        return result_statuses, target_infos_snapshot


class MoriKVSender(CommonKVSender):
    def __init__(
        self,
        mgr: MoriKVManager,
        bootstrap_addr: str,
        bootstrap_room: int,
        dest_tp_ranks: List[int],
        pp_rank: int,
        req_has_disagg_prefill_dp_rank: bool = False,
    ):
        self.room_owner = uuid.uuid4().hex
        mgr.activate_room_owner(bootstrap_room, self.room_owner)
        super().__init__(
            mgr,
            bootstrap_addr,
            bootstrap_room,
            dest_tp_ranks,
            pp_rank,
            req_has_disagg_prefill_dp_rank,
        )
        self.conclude_state: Optional[KVPoll] = None
        self.init_time = time.time()

    def send(
        self,
        kv_indices: npt.NDArray[np.int32],
        state_indices: Optional[List] = None,
        num_kv_tokens: Optional[int] = None,
    ):
        kv_indices, index_slice, is_last_chunk, should_skip = (
            self._prepare_send_indices(kv_indices, state_indices)
        )
        if should_skip:
            return

        transfer_state_indices = (
            None
            if self.kv_mgr._should_skip_cp_replicated_state_transfer()
            else state_indices
        )
        normalized_state = (
            _normalize_state_indices_per_component(transfer_state_indices)
            if is_last_chunk
            else None
        )
        self._record_transfer_indices(kv_indices, transfer_state_indices)
        wait_event = getattr(self, "_early_send_wait_event", None)
        self._early_send_wait_event = None

        if not is_last_chunk:
            self.kv_mgr.add_transfer_request(
                self.bootstrap_room,
                kv_indices,
                index_slice,
                False,
                num_kv_tokens=num_kv_tokens,
                wait_event=wait_event,
                room_owner=self.room_owner,
            )
        else:
            self.kv_mgr.add_transfer_request(
                self.bootstrap_room,
                kv_indices,
                index_slice,
                True,
                aux_index=self.aux_index,
                state_indices=normalized_state,
                num_kv_tokens=num_kv_tokens,
                wait_event=wait_event,
                room_owner=self.room_owner,
            )

    def poll(self) -> KVPoll:
        if self.conclude_state is not None:
            return self.conclude_state

        status = self.kv_mgr.check_room_owner_status(
            self.bootstrap_room, self.room_owner
        )
        if status == KVPoll.Bootstrapping:
            timeout_result = self._check_bootstrap_timeout()
            if timeout_result is not None:
                self.conclude_state = timeout_result
                return timeout_result
        if status in (KVPoll.Success, KVPoll.Failed):
            self.conclude_state = status
        return status

    def clear(self) -> None:
        if self.kv_mgr._defer_room_clear_if_outstanding(
            self.bootstrap_room, self.room_owner
        ):
            return
        self.kv_mgr.clear_room_owner(self.bootstrap_room, self.room_owner)

    def failure_exception(self):
        if self.conclude_state is None:
            self.conclude_state = KVPoll.Failed

        failure_reason = self.kv_mgr.pop_room_owner_failure(
            self.bootstrap_room, self.room_owner
        )
        self.clear()
        is_propagated = failure_reason is None
        if is_propagated:
            failure_reason = "KV transfer failed"
        raise KVTransferError(
            self.bootstrap_room, failure_reason, is_from_another_rank=is_propagated
        )


class MoriKVReceiver(CommonKVReceiver):
    def __init__(
        self,
        mgr: MoriKVManager,
        bootstrap_addr: str,
        bootstrap_room: Optional[int] = None,
        request_epoch: Optional[str] = None,
    ):
        generation_activated = False
        if mgr.requires_strict_deferred_release:
            if bootstrap_room is None or not isinstance(request_epoch, str):
                raise KVTransferError(
                    bootstrap_room, "MORI DCP receiver requires a request epoch"
                )
            mgr.activate_decode_room_generation(bootstrap_room, request_epoch)
            generation_activated = True
        try:
            super().__init__(mgr, bootstrap_addr, bootstrap_room)
        except Exception:
            if generation_activated:
                mgr.clear_decode_room_generation(bootstrap_room, request_epoch)
            raise
        if request_epoch is not None:
            self.abort_generation = request_epoch
        self.init_time: Optional[float] = None
        self.metadata_published = False

    def init(
        self,
        prefill_dp_rank: int,
    ):
        super().init(prefill_dp_rank)

    def _register_kv_args(self) -> bool:
        if self.bootstrap_infos is None:
            return False
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
        packed_kv_item_lens = struct.pack(
            f"{len(self.kv_mgr.kv_args.kv_item_lens)}Q",
            *self.kv_mgr.kv_args.kv_item_lens,
        )
        packed_kv_layer_ids = struct.pack(
            f"{len(self.kv_mgr.kv_args.kv_layer_ids)}I",
            *self.kv_mgr.kv_args.kv_layer_ids,
        )
        dst_dcp_size = str(self.kv_mgr.dcp_size).encode("ascii")
        dst_dcp_rank = str(self.kv_mgr.dcp_rank).encode("ascii")
        registration_guard = (
            MORI_DCP_DRAIN_GUARD if self.kv_mgr.dcp_size > 1 else MORI_GUARD
        )
        registration_waiters: Dict[str, threading.Event] = {}
        if self.kv_mgr.dcp_size > 1:
            registration_waiters = {
                uuid.uuid4().hex: threading.Event() for _ in self.bootstrap_infos
            }
            with self.kv_mgr._dcp_registration_lock:
                self.kv_mgr._dcp_registration_events.update(registration_waiters)
        registration_ids: List[Optional[str]] = (
            list(registration_waiters)
            if registration_waiters
            else [None] * len(self.bootstrap_infos)
        )

        try:
            for bootstrap_info, registration_id in zip(
                self.bootstrap_infos, registration_ids
            ):
                sock, lock = self._connect_to_bootstrap_server(bootstrap_info)
                frames = [
                    registration_guard,
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
                    packed_kv_layer_ids,
                    dst_dcp_size,
                    dst_dcp_rank,
                ]
                if registration_id is not None:
                    frames.append(registration_id.encode("ascii"))
                with lock:
                    sock.send_multipart(frames)

            deadline = time.monotonic() + min(
                float(self.kv_mgr.waiting_timeout),
                MORI_DCP_REGISTRATION_TIMEOUT_SECONDS,
            )
            for registration_id, event in registration_waiters.items():
                if not event.wait(max(0.0, deadline - time.monotonic())):
                    missing_count = sum(
                        not waiter.is_set() for waiter in registration_waiters.values()
                    )
                    raise TimeoutError(
                        "MORI DCP registration capability acknowledgement timed out "
                        f"for {missing_count} prefill rank(s)"
                    )
        except (zmq.ZMQError, TimeoutError) as exc:
            failure_reason = (
                "_register_kv_args to prefill "
                f"{bootstrap_info.get('rank_ip')}:{bootstrap_info.get('rank_port')} failed"
                if isinstance(exc, zmq.ZMQError)
                else str(exc)
            )
            self.kv_mgr.record_failure(self.bootstrap_room, failure_reason)
            self.conclude_state = KVPoll.Failed
            self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Failed)
            return False
        finally:
            if registration_waiters:
                with self.kv_mgr._dcp_registration_lock:
                    for registration_id in registration_waiters:
                        self.kv_mgr._dcp_registration_events.pop(registration_id, None)
        return True

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
        abort_generation_bytes = (
            self.abort_generation.encode("ascii")
            if self.kv_mgr.requires_strict_deferred_release
            else b""
        )

        for bootstrap_info in self.bootstrap_infos:
            is_dummy = bootstrap_info.get("is_dummy", False)
            if not is_dummy and normalized_state is not None:
                state_bytes = _pack_state_indices(normalized_state)
            else:
                state_bytes = b""
            try:
                sock, lock = self._connect_to_bootstrap_server(bootstrap_info)
                with lock:
                    self.metadata_published = True
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
                            abort_generation_bytes,
                        ]
                    )
            except zmq.ZMQError:
                self.invalidate_cached_bootstrap_infos()
                self.kv_mgr.record_failure(
                    self.bootstrap_room,
                    f"send_metadata to prefill {bootstrap_info.get('rank_ip')}:{bootstrap_info.get('rank_port')} failed",
                )
                self.conclude_state = KVPoll.Failed
                self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Failed)
                return
        self.init_time = time.time()

    def ensure_abort_notified_for_drain(self) -> bool:
        if not self.metadata_published:
            return False
        return super().ensure_abort_notified_for_drain()

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
        if self.kv_mgr.requires_strict_deferred_release:
            self.kv_mgr.clear_decode_room_generation(
                self.bootstrap_room, self.abort_generation
            )

    def failure_exception(self):
        if self.conclude_state is None:
            self.conclude_state = KVPoll.Failed

        with self.kv_mgr.failure_lock:
            failure_reason = self.kv_mgr.failure_records.pop(self.bootstrap_room, None)
        if not (self.kv_mgr.requires_strict_deferred_release and self.abort_notified):
            self.clear()
        is_propagated = failure_reason is None
        if is_propagated:
            failure_reason = "KV transfer failed"
        raise KVTransferError(
            self.bootstrap_room, failure_reason, is_from_another_rank=is_propagated
        )

    def abort(self):
        if self.bootstrap_room is None:
            return
        bootstrap_room = self.bootstrap_room
        super().abort()
        if not (self.kv_mgr.requires_strict_deferred_release and self.abort_notified):
            self.clear()
        with self.kv_mgr.failure_lock:
            self.kv_mgr.failure_records.pop(bootstrap_room, None)


class MoriKVBootstrapServer(CommonKVBootstrapServer):
    pass
