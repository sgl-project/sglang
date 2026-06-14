from __future__ import annotations

import concurrent.futures
import ctypes
import dataclasses
import logging
import os
import struct
import threading
import time
from collections import defaultdict
from typing import Dict, List, Optional, Union

import numpy as np
import numpy.typing as npt

from sglang.srt.disaggregation.base.conn import KVArgs, KVPoll
from sglang.srt.disaggregation.common.conn import (
    CommonKVBootstrapServer,
    CommonKVManager,
    CommonKVReceiver,
    CommonKVSender,
)
from sglang.srt.disaggregation.common.utils import (
    FastQueue,
    group_concurrent_contiguous,
)
from sglang.srt.disaggregation.p2p.transfer_engine import P2PTransferEngine
from sglang.srt.disaggregation.utils import (
    DisaggregationMode,
    filter_kv_indices_for_cp_rank,
)
from sglang.srt.environ import envs
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils.network import NetworkAddress

logger = logging.getLogger(__name__)


class P2PTransferError(Exception):
    def __init__(self, bootstrap_room: int, failure_reason: str):
        super().__init__(failure_reason)
        self.bootstrap_room = bootstrap_room
        self.failure_reason = failure_reason

    def __str__(self):
        return f"P2PTransferError(bootstrap_room={self.bootstrap_room}): {self.failure_reason}"


# prefill
@dataclasses.dataclass
class TransferKVChunk:
    room: int
    prefill_kv_indices: npt.NDArray[np.int32]
    index_slice: slice
    is_last: bool
    prefill_aux_index: Optional[int]
    state_indices: Optional[List[int]]


# decode
@dataclasses.dataclass
class TransferInfo:
    room: int
    endpoint: str
    dst_port: int
    p2p_session_id: str
    dst_kv_indices: npt.NDArray[np.int32]
    dst_aux_index: int
    dst_state_indices: List[int]
    required_dst_info_num: int
    is_dummy: bool

    @classmethod
    def from_zmq(cls, msg: List[bytes]):
        if msg[4] == b"" and msg[5] == b"":
            is_dummy = True
            dst_kv_indices = np.array([], dtype=np.int32)
            dst_aux_index = None
            dst_state_indices = []
        else:
            dst_kv_indices = np.frombuffer(msg[4], dtype=np.int32)
            dst_aux_index = int(msg[5].decode("ascii"))
            if msg[6] == b"":
                dst_state_indices = []
            else:
                dst_state_indices = list(np.frombuffer(msg[6], dtype=np.int32))
            is_dummy = False
        return cls(
            room=int(msg[0].decode("ascii")),
            endpoint=msg[1].decode("ascii"),
            dst_port=int(msg[2].decode("ascii")),
            p2p_session_id=msg[3].decode("ascii"),
            dst_kv_indices=dst_kv_indices,
            dst_aux_index=dst_aux_index,
            dst_state_indices=dst_state_indices,
            required_dst_info_num=int(msg[7].decode("ascii")),
            is_dummy=is_dummy,
        )


# decode
@dataclasses.dataclass
class KVArgsRegisterInfo:
    room: str
    endpoint: str
    dst_port: int
    p2p_session_id: str
    dst_kv_ptrs: List[bytes]
    dst_aux_ptrs: List[int]
    dst_state_data_ptrs: List[bytes]
    dst_tp_rank: int
    dst_attn_tp_size: int
    dst_kv_item_len: int
    # for mamba state different tp slice transfer
    dst_state_item_lens: List[int]
    dst_state_dim_per_tensor: List[int]
    dst_state_base_offsets: List[int]

    @classmethod
    def from_zmq(cls, msg: List[bytes], handle_size: int = 64):
        kv_handles_bytes = msg[4]
        if len(kv_handles_bytes) % handle_size != 0:
            raise ValueError(
                f"Invalid kv_handles_bytes length={len(kv_handles_bytes)}, "
                f"handle_size={handle_size}"
            )
        dst_kv_ptrs = [
            kv_handles_bytes[i * handle_size : (i + 1) * handle_size]
            for i in range(len(kv_handles_bytes) // handle_size)
        ]

        state_handles_bytes = msg[7] if len(msg) > 7 else b""
        if state_handles_bytes and len(state_handles_bytes) % handle_size != 0:
            raise ValueError(
                f"Invalid state_handles_bytes length={len(state_handles_bytes)}, "
                f"handle_size={handle_size}"
            )
        dst_state_data_ptrs = [
            state_handles_bytes[i * handle_size : (i + 1) * handle_size]
            for i in range(len(state_handles_bytes) // handle_size)
        ]

        return cls(
            room=str(msg[0].decode("ascii")),
            endpoint=msg[1].decode("ascii"),
            dst_port=int(msg[2].decode("ascii")),
            p2p_session_id=msg[3].decode("ascii"),
            dst_kv_ptrs=dst_kv_ptrs,
            dst_aux_ptrs=list(struct.unpack(f"{len(msg[5])//8}Q", msg[5])),
            dst_state_data_ptrs=dst_state_data_ptrs,
            dst_tp_rank=int(msg[8].decode("ascii")) if len(msg) > 8 else 0,
            dst_attn_tp_size=int(msg[9].decode("ascii")) if len(msg) > 9 else 0,
            dst_kv_item_len=int(msg[10].decode("ascii")) if len(msg) > 10 else 0,
            dst_state_item_lens=(
                list(struct.unpack(f"{len(msg[11])//4}I", msg[11]))
                if len(msg) > 11 and len(msg[11]) > 0
                else []
            ),
            dst_state_dim_per_tensor=(
                list(struct.unpack(f"{len(msg[12])//4}I", msg[12]))
                if len(msg) > 12 and len(msg[12]) > 0
                else []
            ),
            dst_state_base_offsets=(
                list(struct.unpack(f"{len(msg[13])//8}Q", msg[13]))
                if len(msg) > 13 and len(msg[13]) > 0
                else []
            ),
        )


class AuxDataCodec:
    """Handles serialization and deserialization of auxiliary data buffers"""

    @staticmethod
    def serialize_data_from_buffer(src_addr, data_length):
        """Serialize data from memory buffer to bytes"""
        buffer = (ctypes.c_byte * data_length).from_address(src_addr)
        return bytes(buffer)

    @staticmethod
    def deserialize_data_to_buffer(kv_args, buffer_index, aux_index, data):
        """Deserialize bytes into target memory buffer"""
        dst_aux_ptr = kv_args.aux_data_ptrs[buffer_index]
        item_len = kv_args.aux_item_lens[buffer_index]
        dst_addr = dst_aux_ptr + item_len * aux_index
        buffer = (ctypes.c_byte * len(data)).from_address(dst_addr)
        buffer[:] = data
        return


class P2PKVManager(CommonKVManager):
    AUX_DATA_HEADER = b"AUX_DATA"
    TRANSFER_POLL_INTERVAL = 0.005

    def __init__(
        self,
        args: KVArgs,
        disaggregation_mode: DisaggregationMode,
        server_args: ServerArgs,
        is_mla_backend: Optional[bool] = False,
    ):
        super().__init__(args, disaggregation_mode, server_args, is_mla_backend)

        self.state_handles = []
        self.state_base_offsets: List[int] = []

        self.handle_size = 64
        self.p2p_batch_limit = envs.SGLANG_P2P_BATCH_LIMIT.get()
        self.transfer_timeout = envs.SGLANG_P2P_TRANSFER_TIMEOUT.get()
        self.kv_transfer_stats: Dict[int, Dict[str, float | int]] = {}
        self.kv_transfer_stats_lock = threading.Lock()
        self.enable_kvcache_log = envs.SGLANG_KVCACHE_LOG.get()

        self.engine = P2PTransferEngine(self.local_ip, self.kv_args.gpu_id)
        self.decode_physical_gpu_ids = {}
        self.register_buffer_to_engine()

        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            self.start_prefill_thread()
            self.session_failures = defaultdict(int)
            self.failed_sessions = set()
            self.session_lock = threading.Lock()

            # Determine the number of threads to use for kv sender
            cpu_count = os.cpu_count() or 8
            transfer_thread_pool_size = (
                envs.SGLANG_DISAGGREGATION_THREAD_POOL_SIZE.get()
            )
            if transfer_thread_pool_size is None:
                transfer_thread_pool_size = min(max(4, int(0.5 * cpu_count) // 8), 12)

            transfer_queue_size = envs.SGLANG_DISAGGREGATION_QUEUE_SIZE.get()
            self.transfer_queues: List[FastQueue] = [
                FastQueue() for _ in range(transfer_queue_size)
            ]

            assert transfer_thread_pool_size >= transfer_queue_size, (
                f"The environment variable "
                f"SGLANG_DISAGGREGATION_THREAD_POOL_SIZE={transfer_thread_pool_size} "
                f"must be greater than or equal to "
                f"SGLANG_DISAGGREGATION_QUEUE_SIZE={transfer_queue_size}."
            )

            self.executors = [
                concurrent.futures.ThreadPoolExecutor(
                    transfer_thread_pool_size // transfer_queue_size
                )
                for _ in range(transfer_queue_size)
            ]

            for queue, executor in zip(self.transfer_queues, self.executors):
                threading.Thread(
                    target=self.transfer_worker,
                    args=(queue, executor),
                    daemon=True,
                ).start()

        elif self.disaggregation_mode == DisaggregationMode.DECODE:
            self.start_decode_thread()

    def register_buffer_to_engine(self):
        logger.info("Start registering KV buffers to P2P engines.")

        self.kv_handles = []
        self.state_handles = []
        self.state_base_offsets = []

        if self.disaggregation_mode != DisaggregationMode.DECODE:
            return

        # KV handles
        for ptr in self.kv_args.kv_data_ptrs:
            kv_handle = self.engine.register_buffer(int(ptr))
            self.kv_handles.append(kv_handle)

        # State handles
        state_ptrs = getattr(self.kv_args, "state_data_ptrs", []) or []

        if not state_ptrs:
            return

        registered_state_ptrs: List[int] = []

        for i, ptr in enumerate(state_ptrs):
            self.state_handles.append(self.engine.register_buffer(ptr))
            registered_state_ptrs.append(ptr)

        if len(registered_state_ptrs) != len(self.state_handles):
            raise RuntimeError(
                "registered_state_ptrs and state_handles length mismatch, "
                "this should never happen."
            )

        handle_min_ptr: Dict[bytes, int] = {}
        for ptr, handle in zip(registered_state_ptrs, self.state_handles):
            old = handle_min_ptr.get(handle)
            if old is None or ptr < old:
                handle_min_ptr[handle] = ptr

        self.state_base_offsets = [
            ptr - handle_min_ptr[handle]
            for ptr, handle in zip(registered_state_ptrs, self.state_handles)
        ]

    def _execute_batch_transfer(
        self,
        transfer_blocks: List[dict],
        transfer_type: str = "data",
    ) -> int:
        if not transfer_blocks:
            return 0

        batch_src_ptrs = [b["src_ptr"] for b in transfer_blocks]
        batch_src_devs = [b["src_dev"] for b in transfer_blocks]
        batch_dst_handles = [b["dst_handle"] for b in transfer_blocks]
        batch_dst_devs = [b["dst_dev"] for b in transfer_blocks]
        batch_offsets = [b["dst_offset"] for b in transfer_blocks]
        batch_lengths = [b["length"] for b in transfer_blocks]

        batch_limit = self.p2p_batch_limit
        timeout_s = self.transfer_timeout
        handles = []

        for i in range(0, len(batch_src_ptrs), batch_limit):
            j = i + batch_limit
            h = self.engine.transfer_many(
                src_ptrs=batch_src_ptrs[i:j],
                src_devs=batch_src_devs[i:j],
                dst_handles=batch_dst_handles[i:j],
                dst_devs=batch_dst_devs[i:j],
                dst_offsets=batch_offsets[i:j],
                lengths=batch_lengths[i:j],
            )
            handles.append(h)

        deadline = time.perf_counter() + timeout_s
        while True:
            if all(h.is_done() for h in handles):
                return 0
            if time.perf_counter() > deadline:
                logger.error(
                    f"P2P {transfer_type} transfer timeout after {timeout_s}s, "
                    f"{len(transfer_blocks)} blocks, {sum(batch_lengths)} bytes"
                )
                return 1
            time.sleep(self.TRANSFER_POLL_INTERVAL)

    def _get_physical_gpu_ids(self, p2p_session_id: str) -> tuple[int, int]:
        dst_physical_gpu_id = self.decode_physical_gpu_ids.get(p2p_session_id)
        if dst_physical_gpu_id is None:
            raise ValueError(f"Physical GPU ID not found for session {p2p_session_id}")

        src_physical_gpu_id = int(self.kv_args.gpu_id)
        dst_physical_gpu_id = int(dst_physical_gpu_id)
        return src_physical_gpu_id, dst_physical_gpu_id

    def _send_kvcache_generic(
        self,
        p2p_session_id: str,
        src_data_ptrs: List[int],
        dst_data_handles: List[Union[int, bytes]],
        item_lens: List[int],
        prefill_data_indices: npt.NDArray[np.int32],
        dst_data_indices: npt.NDArray[np.int32],
        executor: concurrent.futures.ThreadPoolExecutor,
        req: Optional[TransferInfo] = None,
        track_stats: bool = False,
        dst_data_base_offsets: Optional[List[int]] = None,
    ) -> int:
        try:
            src_physical_gpu_id, dst_physical_gpu_id = self._get_physical_gpu_ids(
                p2p_session_id
            )

            prefill_blocks, dst_blocks = group_concurrent_contiguous(
                prefill_data_indices, dst_data_indices
            )

            transfer_blocks = []
            total_tokens = 0
            start_time = time.perf_counter() if track_stats else None

            num_layers = min(len(src_data_ptrs), len(dst_data_handles), len(item_lens))

            for layer_id in range(num_layers):
                base_ptr = int(src_data_ptrs[layer_id])
                item_len = int(item_lens[layer_id])
                dst_handle = dst_data_handles[layer_id]
                dst_base_off = (
                    int(dst_data_base_offsets[layer_id])
                    if dst_data_base_offsets is not None
                    and layer_id < len(dst_data_base_offsets)
                    else 0
                )

                for prefill_block, dst_block in zip(prefill_blocks, dst_blocks):
                    if not prefill_block or not dst_block:
                        continue

                    first_src_block = int(prefill_block[0])
                    first_dst_block = int(dst_block[0])
                    length_bytes = len(prefill_block) * item_len

                    src_ptr = base_ptr + first_src_block * item_len
                    dst_off = dst_base_off + first_dst_block * item_len

                    transfer_blocks.append(
                        {
                            "src_ptr": src_ptr,
                            "src_dev": src_physical_gpu_id,
                            "dst_handle": dst_handle,
                            "dst_dev": dst_physical_gpu_id,
                            "dst_offset": dst_off,
                            "length": length_bytes,
                        }
                    )

                    if track_stats:
                        total_tokens += len(prefill_block)

            if not transfer_blocks:
                logger.debug("No transfer tasks to send.")
                return 0

            ret = self._execute_batch_transfer(transfer_blocks, "KV data")

            if ret == 0 and track_stats and req is not None and start_time is not None:
                total_bytes = sum(b["length"] for b in transfer_blocks)
                total_time_ms = (time.perf_counter() - start_time) * 1000.0
                with self.kv_transfer_stats_lock:
                    self.kv_transfer_stats[req.room] = {
                        "kv_bytes": int(total_bytes),
                        "kv_time_ms": float(total_time_ms),
                        "kv_tokens": int(total_tokens),
                    }

            return ret

        except ValueError as e:
            logger.error(f"P2P generic transfer failed: {e}")
            return 1
        except Exception as e:
            logger.exception(f"P2P generic transfer failed: {e}")
            return 1

    def send_kvcache(
        self,
        req: TransferInfo,
        p2p_session_id: str,
        prefill_kv_indices: npt.NDArray[np.int32],
        dst_kv_ptrs: List[bytes],
        dst_kv_indices: npt.NDArray[np.int32],
        executor: concurrent.futures.ThreadPoolExecutor,
    ):
        """
        Send KV cache data using the generic transfer function.
        """
        return self._send_kvcache_generic(
            p2p_session_id=p2p_session_id,
            src_data_ptrs=self.kv_args.kv_data_ptrs,
            dst_data_handles=dst_kv_ptrs,
            item_lens=self.kv_args.kv_item_lens,
            prefill_data_indices=prefill_kv_indices,
            dst_data_indices=dst_kv_indices,
            executor=executor,
            req=req,
            track_stats=True,
        )

    def send_aux(
        self,
        req: TransferInfo,
        prefill_aux_index: int,
        dst_aux_ptrs: list[int],
    ):
        prefill_aux_ptrs = self.kv_args.aux_data_ptrs
        prefill_aux_item_lens = self.kv_args.aux_item_lens

        for i in range(len(prefill_aux_ptrs)):
            length = prefill_aux_item_lens[i]
            src_addr = prefill_aux_ptrs[i] + length * prefill_aux_index
            data = AuxDataCodec.serialize_data_from_buffer(src_addr, length)

            self.send_aux_data_to_endpoint(
                remote=req.endpoint,
                dst_port=req.dst_port,
                room=req.room,
                buffer_index=i,
                aux_index=req.dst_aux_index,
                data=data,
            )
        return 0

    def send_aux_data_to_endpoint(
        self,
        remote: str,
        dst_port: int,
        room: int,
        buffer_index: int,
        aux_index: int,
        data: bytes,
    ):
        na = NetworkAddress(remote, dst_port)
        socket = self._connect(na.to_tcp(), is_ipv6=na.is_ipv6)

        socket.send_multipart(
            [
                P2PKVManager.AUX_DATA_HEADER,
                str(room).encode("ascii"),
                str(buffer_index).encode("ascii"),
                str(aux_index).encode("ascii"),
                struct.pack(">I", len(data)),
                data,
            ]
        )

    def _handle_aux_data(self, msg: List[bytes]):
        room = int(msg[1].decode("ascii"))
        buffer_index = int(msg[2].decode("ascii"))
        aux_index = int(msg[3].decode("ascii"))
        data_length = struct.unpack(">I", msg[4])[0]
        data = msg[5]

        if len(data) != data_length:
            logger.error(
                f"AUX_DATA length mismatch for bootstrap_room {room}: expected {data_length}, got {len(data)}"
            )
            return
        AuxDataCodec.deserialize_data_to_buffer(
            self.kv_args, buffer_index, aux_index, data
        )
        logger.debug(
            f"Received AUX_DATA for bootstrap_room {room} with length:{len(data)}"
        )

    def maybe_send_extra(
        self,
        req: TransferInfo,
        prefill_state_indices: List[int],
        dst_state_handles: List[Union[int, bytes]],
        executor: concurrent.futures.ThreadPoolExecutor,
        target_rank_registration_info: Optional[KVArgsRegisterInfo] = None,
    ):
        state_type = getattr(self.kv_args, "state_type", "none")
        if not prefill_state_indices or not dst_state_handles:
            return 0

        dst_state_base_offsets = (
            target_rank_registration_info.dst_state_base_offsets
            if target_rank_registration_info is not None
            else None
        )

        if state_type == "mamba":
            if len(prefill_state_indices) != 1:
                logger.error(
                    f"Invalid prefill_state_indices for mamba: {prefill_state_indices}"
                )
                return 1

            if not req.dst_state_indices or len(req.dst_state_indices) != 1:
                logger.error(
                    f"Invalid dst_state_indices for mamba, "
                    f"room={req.room}, p2p_session_id={req.p2p_session_id}, "
                    f"dst_state_indices={req.dst_state_indices}"
                )
                return 1
            if (
                target_rank_registration_info is not None
                and self.attn_tp_size != target_rank_registration_info.dst_attn_tp_size
            ):
                return self._send_mamba_state_slice(
                    req,
                    prefill_state_indices,
                    dst_state_handles,
                    target_rank_registration_info.dst_state_item_lens,
                    target_rank_registration_info.dst_state_dim_per_tensor,
                    target_rank_registration_info.dst_tp_rank,
                    target_rank_registration_info.dst_attn_tp_size,
                    dst_state_base_offsets,
                )
            else:
                return self._send_mamba_state(
                    req,
                    prefill_state_indices,
                    dst_state_handles,
                    (
                        target_rank_registration_info.dst_state_item_lens
                        if target_rank_registration_info is not None
                        else None
                    ),
                    dst_state_base_offsets,
                )

        elif state_type in ["swa", "nsa"]:
            # SWA and NSA hybrid models do not support different TP sizes yet
            if (
                target_rank_registration_info is not None
                and not self.is_mla_backend
                and self.attn_tp_size != target_rank_registration_info.dst_attn_tp_size
            ):
                raise RuntimeError(
                    f"PD Disaggregation does NOT support PD different TP sizes for "
                    f"non-MLA {state_type.upper()} hybrid models yet."
                )

            if len(prefill_state_indices) < len(req.dst_state_indices):
                logger.warning(
                    f"len(prefill_state_indices) = {len(prefill_state_indices)}, len(dst_state_indices) = {len(req.dst_state_indices)}"
                )
                prefill_state_indices = prefill_state_indices[
                    : len(req.dst_state_indices)
                ]

            return self._send_state_generic(
                req,
                prefill_state_indices,
                dst_state_handles,
                executor,
                dst_state_base_offsets,
            )

        else:
            return 0

    def _send_mamba_state(
        self,
        req: TransferInfo,
        prefill_state_indices: List[int],
        dst_state_handles: List[Union[int, bytes]],
        dst_state_item_lens: Optional[List[int]] = None,
        dst_state_base_offsets: Optional[List[int]] = None,
    ):
        assert len(prefill_state_indices) == 1, "Mamba should have single state index"

        pre_idx = int(prefill_state_indices[0])
        dst_idx = int(req.dst_state_indices[0])

        src_physical_gpu_id, dst_physical_gpu_id = self._get_physical_gpu_ids(
            req.p2p_session_id
        )

        transfer_blocks = []

        for i, dst_handle in enumerate(dst_state_handles):
            if i >= len(self.kv_args.state_data_ptrs):
                break

            src_item_len = int(self.kv_args.state_item_lens[i])
            dst_item_len = (
                int(dst_state_item_lens[i])
                if dst_state_item_lens is not None and i < len(dst_state_item_lens)
                else src_item_len
            )

            src_ptr = int(self.kv_args.state_data_ptrs[i]) + pre_idx * src_item_len
            dst_base_off = (
                int(dst_state_base_offsets[i])
                if dst_state_base_offsets is not None
                and i < len(dst_state_base_offsets)
                else 0
            )
            dst_offset = dst_base_off + dst_idx * dst_item_len

            transfer_blocks.append(
                {
                    "src_ptr": src_ptr,
                    "src_dev": src_physical_gpu_id,
                    "dst_handle": dst_handle,
                    "dst_dev": dst_physical_gpu_id,
                    "dst_offset": dst_offset,
                    "length": src_item_len,
                }
            )

        return self._execute_batch_transfer(transfer_blocks, "mamba state")

    def _send_mamba_state_slice(
        self,
        req: TransferInfo,
        prefill_mamba_index: List[int],
        dst_state_handles: List[Union[int, bytes]],
        dst_state_item_lens: List[int],
        dst_state_dim_per_tensor: List[int],
        dst_tp_rank: int,
        dst_attn_tp_size: int,
        dst_state_base_offsets: Optional[List[int]] = None,
    ):
        logger.warning(
            f"Using Mamba state slice transfer for different TP sizes between prefill "
            f"and decode. Prefill attn_tp_size={self.attn_tp_size}, "
            f"Decode attn_tp_size={dst_attn_tp_size}. Performance may be affected."
        )
        assert len(prefill_mamba_index) == 1, "Mamba should have single state index"

        src_physical_gpu_id, dst_physical_gpu_id = self._get_physical_gpu_ids(
            req.p2p_session_id
        )

        prefill_state_data_ptrs = self.kv_args.state_data_ptrs
        prefill_state_item_lens = self.kv_args.state_item_lens
        src_state_dim_per_tensor = getattr(self.kv_args, "state_dim_per_tensor", [])
        pre_idx = int(prefill_mamba_index[0])

        if not src_state_dim_per_tensor or not dst_state_dim_per_tensor:
            return self._send_mamba_state(
                req,
                prefill_mamba_index,
                dst_state_handles,
                dst_state_item_lens=dst_state_item_lens,
                dst_state_base_offsets=dst_state_base_offsets,
            )

        local_tp_rank_in_group = self.kv_args.engine_rank % self.attn_tp_size
        dst_tp_rank_in_group = dst_tp_rank % dst_attn_tp_size

        transfer_blocks = []
        for i, dst_handle in enumerate(dst_state_handles):
            if i >= len(prefill_state_data_ptrs):
                break

            src_item_len = prefill_state_item_lens[i]
            dst_item_len = (
                dst_state_item_lens[i] if i < len(dst_state_item_lens) else src_item_len
            )
            src_dim = (
                src_state_dim_per_tensor[i] if i < len(src_state_dim_per_tensor) else 1
            )
            dst_dim = (
                dst_state_dim_per_tensor[i] if i < len(dst_state_dim_per_tensor) else 1
            )

            src_bytes_per_dim = src_item_len // src_dim if src_dim > 0 else src_item_len
            dst_bytes_per_dim = dst_item_len // dst_dim if dst_dim > 0 else dst_item_len

            if self.attn_tp_size > dst_attn_tp_size:
                # Multiple prefill ranks -> 1 decode rank
                src_dim_start = 0
                num_dims_to_send = src_dim
                writers_per_decode = self.attn_tp_size // dst_attn_tp_size
                local_writer_idx = local_tp_rank_in_group % writers_per_decode
                dst_dim_start = local_writer_idx * src_dim
                length = num_dims_to_send * src_bytes_per_dim
            else:
                # 1 prefill rank -> multiple decode ranks
                src_dim_start = (dst_tp_rank_in_group * dst_dim) % src_dim
                num_dims_to_send = dst_dim
                dst_dim_start = 0
                length = num_dims_to_send * dst_bytes_per_dim

            src_ptr = (
                int(prefill_state_data_ptrs[i])
                + pre_idx * src_item_len
                + src_dim_start * src_bytes_per_dim
            )

            dst_state_idx = int(req.dst_state_indices[0])

            dst_base_off = (
                int(dst_state_base_offsets[i])
                if dst_state_base_offsets is not None
                and i < len(dst_state_base_offsets)
                else 0
            )
            dst_offset = (
                dst_base_off
                + dst_state_idx * dst_item_len
                + dst_dim_start * dst_bytes_per_dim
            )

            transfer_blocks.append(
                {
                    "src_ptr": src_ptr,
                    "src_dev": src_physical_gpu_id,
                    "dst_handle": dst_handle,
                    "dst_dev": dst_physical_gpu_id,
                    "dst_offset": dst_offset,
                    "length": length,
                }
            )

        return self._execute_batch_transfer(transfer_blocks, "mamba state slice")

    def _send_state_generic(
        self,
        req: TransferInfo,
        prefill_state_indices: List[int],
        dst_state_handles: List[Union[int, bytes]],
        executor: concurrent.futures.ThreadPoolExecutor,
        dst_state_base_offsets: Optional[List[int]] = None,
    ):
        pre_idx = np.array(prefill_state_indices, dtype=np.int32)
        dst_idx = np.array(req.dst_state_indices or [], dtype=np.int32)

        if len(pre_idx) == 0 or len(dst_idx) == 0:
            return 0

        return self._send_kvcache_generic(
            p2p_session_id=req.p2p_session_id,
            src_data_ptrs=self.kv_args.state_data_ptrs,
            dst_data_handles=dst_state_handles,
            item_lens=self.kv_args.state_item_lens,
            prefill_data_indices=pre_idx,
            dst_data_indices=dst_idx,
            executor=executor,
            req=None,
            track_stats=False,
            dst_data_base_offsets=dst_state_base_offsets,
        )

    def sync_status_to_decode_endpoint(
        self, remote: str, dst_port: int, room: int, status: int, prefill_rank: int
    ):
        na = NetworkAddress(remote, dst_port)
        self._connect(na.to_tcp(), is_ipv6=na.is_ipv6).send_multipart(
            [
                str(room).encode("ascii"),
                str(status).encode("ascii"),
                str(prefill_rank).encode("ascii"),
            ]
        )

    def transfer_worker(
        self, queue: FastQueue, executor: concurrent.futures.ThreadPoolExecutor
    ):
        while True:
            try:
                kv_chunk: TransferKVChunk = queue.get()
                reqs_to_be_processed = (
                    self.transfer_infos[kv_chunk.room].values()
                    if kv_chunk.room in self.transfer_infos
                    else []
                )
                polls = []
                dst_ranks_infos = []
                prefill_unique_rank = (
                    self.attn_tp_rank * (self.pp_size * self.attn_cp_size)
                    + self.pp_rank * self.attn_cp_size
                    + self.attn_cp_rank
                )

                for req in reqs_to_be_processed:
                    if not req.is_dummy:
                        with self.session_lock:
                            if req.p2p_session_id in self.failed_sessions:
                                self.record_failure(
                                    kv_chunk.room,
                                    f"Decode instance could be dead, remote p2p session "
                                    f"{req.p2p_session_id} is not alive",
                                )
                                self.update_status(kv_chunk.room, KVPoll.Failed)
                                self.sync_status_to_decode_endpoint(
                                    req.endpoint,
                                    req.dst_port,
                                    req.room,
                                    KVPoll.Failed,
                                    prefill_unique_rank,
                                )
                                break

                        chunked_dst_kv_indice = req.dst_kv_indices[kv_chunk.index_slice]

                        if len(chunked_dst_kv_indice) < len(
                            kv_chunk.prefill_kv_indices
                        ):
                            kv_chunk.prefill_kv_indices = kv_chunk.prefill_kv_indices[
                                : len(chunked_dst_kv_indice)
                            ]
                            logger.warning(
                                f"len(chunked_dst_kv_indice) = {len(chunked_dst_kv_indice)}, "
                                f"len(kv_chunk.prefill_kv_indices) = {len(kv_chunk.prefill_kv_indices)}"
                            )

                        target_rank_registration_info: KVArgsRegisterInfo = (
                            self.decode_kv_args_table[req.p2p_session_id]
                        )

                        if self.is_mla_backend or (
                            self.attn_tp_size
                            == target_rank_registration_info.dst_attn_tp_size
                        ):
                            ret = self.send_kvcache(
                                req,
                                req.p2p_session_id,
                                kv_chunk.prefill_kv_indices,
                                target_rank_registration_info.dst_kv_ptrs,
                                chunked_dst_kv_indice,
                                executor,
                            )
                        else:
                            ret = self.send_kvcache_slice(
                                req,
                                req.p2p_session_id,
                                kv_chunk.prefill_kv_indices,
                                target_rank_registration_info.dst_kv_ptrs,
                                chunked_dst_kv_indice,
                                target_rank_registration_info.dst_tp_rank,
                                target_rank_registration_info.dst_attn_tp_size,
                                target_rank_registration_info.dst_kv_item_len,
                                executor,
                            )

                        if ret != 0:
                            with self.kv_transfer_stats_lock:
                                self.kv_transfer_stats.pop(kv_chunk.room, None)
                            with self.session_lock:
                                self.session_failures[req.p2p_session_id] += 1
                                if self.session_failures[req.p2p_session_id] >= 1:
                                    self.failed_sessions.add(req.p2p_session_id)
                                    logger.error(
                                        f"Session {req.p2p_session_id} failed."
                                    )
                            self.record_failure(
                                kv_chunk.room,
                                f"Failed to send kv chunk of {kv_chunk.room} to "
                                f"{NetworkAddress(req.endpoint, req.dst_port).to_host_port_str()}",
                            )
                            self.update_status(kv_chunk.room, KVPoll.Failed)
                            self.sync_status_to_decode_endpoint(
                                req.endpoint,
                                req.dst_port,
                                req.room,
                                KVPoll.Failed,
                                prefill_unique_rank,
                            )
                            break

                        if kv_chunk.is_last:
                            state_ret = 0
                            if kv_chunk.state_indices is not None:
                                state_ret = self.maybe_send_extra(
                                    req,
                                    kv_chunk.state_indices,
                                    target_rank_registration_info.dst_state_data_ptrs,
                                    executor,
                                    target_rank_registration_info,
                                )

                            if state_ret != 0:
                                with self.kv_transfer_stats_lock:
                                    self.kv_transfer_stats.pop(kv_chunk.room, None)

                                with self.session_lock:
                                    self.session_failures[req.p2p_session_id] += 1
                                    if self.session_failures[req.p2p_session_id] >= 1:
                                        self.failed_sessions.add(req.p2p_session_id)
                                        logger.error(
                                            f"Session {req.p2p_session_id} failed during state transfer."
                                        )

                                self.record_failure(
                                    kv_chunk.room,
                                    f"Failed to send state chunk of {kv_chunk.room} to "
                                    f"{NetworkAddress(req.endpoint, req.dst_port).to_host_port_str()}",
                                )
                                self.update_status(kv_chunk.room, KVPoll.Failed)
                                self.sync_status_to_decode_endpoint(
                                    req.endpoint,
                                    req.dst_port,
                                    req.room,
                                    KVPoll.Failed,
                                    prefill_unique_rank,
                                )
                                break

                            ret = self.send_aux(
                                req,
                                kv_chunk.prefill_aux_index,
                                target_rank_registration_info.dst_aux_ptrs,
                            )
                            polls.append(True if ret == 0 else False)
                            dst_ranks_infos.append(
                                (req.endpoint, req.dst_port, req.room)
                            )

                            # Only sync status when all the dst ranks have received the kvcache
                            if len(polls) == req.required_dst_info_num:
                                status = KVPoll.Success if all(polls) else KVPoll.Failed
                                self.update_status(req.room, status)
                                for endpoint, dst_port, room in dst_ranks_infos:
                                    self.sync_status_to_decode_endpoint(
                                        endpoint,
                                        dst_port,
                                        room,
                                        status,
                                        prefill_unique_rank,
                                    )

                            with self.kv_transfer_stats_lock:
                                stats = self.kv_transfer_stats.pop(kv_chunk.room, None)

                            if self.enable_kvcache_log and stats is not None:
                                kv_bytes = stats.get("kv_bytes", 0)
                                kv_time_ms = stats.get("kv_time_ms", 0.0)
                                kv_tokens = stats.get("kv_tokens", 0)
                                logger.info(
                                    f"[KVCACHE_TRANSFER] room={kv_chunk.room} "
                                    f"bytes={kv_bytes} tokens={kv_tokens} "
                                    f"time_ms={kv_time_ms:.2f}"
                                )
                    else:
                        # Dummy request means the decode instance is not used,
                        # so its status can be marked as success directly
                        if kv_chunk.is_last and req.room in self.request_status:
                            self.update_status(req.room, KVPoll.Success)

                if (
                    kv_chunk.room not in self.request_status
                    or self.check_status(kv_chunk.room) == KVPoll.Success
                ):
                    if kv_chunk.room in self.transfer_infos:
                        self.transfer_infos.pop(kv_chunk.room)

            except Exception as e:
                raise RuntimeError(
                    f"Transfer thread failed because of {e}. "
                    f"Prefill instance with bootstrap_port={self.bootstrap_port} is dead."
                )

    def start_prefill_thread(self):
        def bootstrap_thread():
            while True:
                waiting_req_bytes = self.server_socket.recv_multipart()
                room = waiting_req_bytes[0].decode("ascii")
                p2p_session_id = waiting_req_bytes[3].decode("ascii")

                if room == "None":
                    physical_gpu_id = int(waiting_req_bytes[6].decode("ascii"))
                    self.decode_physical_gpu_ids[p2p_session_id] = physical_gpu_id

                    # KV handles
                    handles_bytes = waiting_req_bytes[4]
                    if len(handles_bytes) % self.handle_size != 0:
                        logger.error(
                            f"Invalid handles_bytes length={len(handles_bytes)} "
                            f"(handle_size={self.handle_size})"
                        )
                        continue

                    num_kv_handles = len(handles_bytes) // self.handle_size
                    dst_kv_ptrs = [
                        handles_bytes[i * self.handle_size : (i + 1) * self.handle_size]
                        for i in range(num_kv_handles)
                    ]

                    for kv_handle in dst_kv_ptrs:
                        try:
                            self.engine.register_d_handle(kv_handle)
                        except Exception as e:
                            logger.exception(f"register_d_handle failed: {e}")

                    # AUX raw ptrs
                    aux_ptrs = list(
                        struct.unpack(
                            f"{len(waiting_req_bytes[5]) // 8}Q",
                            waiting_req_bytes[5],
                        )
                    )

                    # State handles
                    state_handles_bytes = (
                        waiting_req_bytes[7] if len(waiting_req_bytes) > 7 else b""
                    )
                    dst_state_ptrs: List[bytes] = []
                    if state_handles_bytes:
                        if len(state_handles_bytes) % self.handle_size != 0:
                            logger.error(
                                f"Invalid state_handles_bytes length={len(state_handles_bytes)} "
                                f"(handle_size={self.handle_size})"
                            )
                        else:
                            num_state_handles = (
                                len(state_handles_bytes) // self.handle_size
                            )
                            dst_state_ptrs = [
                                state_handles_bytes[
                                    i * self.handle_size : (i + 1) * self.handle_size
                                ]
                                for i in range(num_state_handles)
                            ]

                            seen_state_handles = set()
                            for h in dst_state_ptrs:
                                if h in seen_state_handles:
                                    continue
                                seen_state_handles.add(h)

                                try:
                                    self.engine.register_d_handle(h)
                                except Exception as e:
                                    logger.exception(
                                        f"register_d_handle(State) failed: {e}"
                                    )

                    dst_tp_rank = (
                        int(waiting_req_bytes[8].decode("ascii"))
                        if len(waiting_req_bytes) > 8
                        else 0
                    )
                    dst_attn_tp_size = (
                        int(waiting_req_bytes[9].decode("ascii"))
                        if len(waiting_req_bytes) > 9
                        else 0
                    )
                    dst_kv_item_len = (
                        int(waiting_req_bytes[10].decode("ascii"))
                        if len(waiting_req_bytes) > 10
                        else 0
                    )

                    dst_state_item_lens = (
                        list(
                            struct.unpack(
                                f"{len(waiting_req_bytes[11])//4}I",
                                waiting_req_bytes[11],
                            )
                        )
                        if len(waiting_req_bytes) > 11
                        and len(waiting_req_bytes[11]) > 0
                        else []
                    )
                    dst_state_dim_per_tensor = (
                        list(
                            struct.unpack(
                                f"{len(waiting_req_bytes[12])//4}I",
                                waiting_req_bytes[12],
                            )
                        )
                        if len(waiting_req_bytes) > 12
                        and len(waiting_req_bytes[12]) > 0
                        else []
                    )
                    dst_state_base_offsets = (
                        list(
                            struct.unpack(
                                f"{len(waiting_req_bytes[13])//8}Q",
                                waiting_req_bytes[13],
                            )
                        )
                        if len(waiting_req_bytes) > 13
                        and len(waiting_req_bytes[13]) > 0
                        else []
                    )

                    if dst_state_base_offsets and (
                        len(dst_state_base_offsets) != len(dst_state_ptrs)
                    ):
                        logger.error(
                            f"dst_state_base_offsets length mismatch: "
                            f"len(dst_state_base_offsets)={len(dst_state_base_offsets)}, "
                            f"len(dst_state_ptrs)={len(dst_state_ptrs)}"
                        )

                    self.decode_kv_args_table[p2p_session_id] = KVArgsRegisterInfo(
                        room=room,
                        endpoint=waiting_req_bytes[1].decode("ascii"),
                        dst_port=int(waiting_req_bytes[2].decode("ascii")),
                        p2p_session_id=p2p_session_id,
                        dst_kv_ptrs=dst_kv_ptrs,
                        dst_aux_ptrs=aux_ptrs,
                        dst_state_data_ptrs=dst_state_ptrs,
                        dst_tp_rank=dst_tp_rank,
                        dst_attn_tp_size=dst_attn_tp_size,
                        dst_kv_item_len=dst_kv_item_len,
                        dst_state_item_lens=dst_state_item_lens,
                        dst_state_dim_per_tensor=dst_state_dim_per_tensor,
                        dst_state_base_offsets=dst_state_base_offsets,
                    )

                    with self.session_lock:
                        if p2p_session_id in self.failed_sessions:
                            self.failed_sessions.remove(p2p_session_id)
                        if p2p_session_id in self.session_failures:
                            del self.session_failures[p2p_session_id]

                    logger.debug(f"Register KVArgs from {p2p_session_id} successfully")
                    continue

                else:
                    required_dst_info_num = int(waiting_req_bytes[7].decode("ascii"))
                    room = int(waiting_req_bytes[0].decode("ascii"))
                    if room not in self.transfer_infos:
                        self.transfer_infos[room] = {}

                    self.transfer_infos[room][p2p_session_id] = TransferInfo.from_zmq(
                        waiting_req_bytes
                    )
                    if len(self.transfer_infos[room]) == required_dst_info_num:
                        self.update_status(room, KVPoll.WaitingForInput)

        threading.Thread(target=bootstrap_thread).start()

    def start_decode_thread(self):
        def decode_thread():
            while True:
                msg = self.server_socket.recv_multipart()
                if msg[0] == P2PKVManager.AUX_DATA_HEADER:
                    self._handle_aux_data(msg)
                    continue

                bootstrap_room, status, prefill_rank = msg
                status = int(status.decode("ascii"))
                bootstrap_room = int(bootstrap_room.decode("ascii"))
                prefill_rank = int(prefill_rank.decode("ascii"))

                if status == KVPoll.Success:
                    if bootstrap_room in self.request_status:
                        self.prefill_response_tracker[bootstrap_room].add(prefill_rank)
                        expected_response_num = (
                            self.required_prefill_response_num_table[bootstrap_room]
                        )
                        arrived_response_num = len(
                            self.prefill_response_tracker[bootstrap_room]
                        )
                        if arrived_response_num == expected_response_num:
                            self.update_status(bootstrap_room, KVPoll.Success)
                elif status == KVPoll.Failed:
                    self.record_failure(
                        bootstrap_room,
                        "Failed to get kvcache from prefill instance, it might be dead",
                    )
                self.update_status(bootstrap_room, status)

        def heartbeat_checker():
            while True:
                time.sleep(self.heartbeat_interval)
                with self.connection_lock:
                    addresses = list(self.prefill_info_table.keys())

                for bootstrap_addr in addresses:
                    session = None
                    try:
                        with self.session_pool_lock:
                            session = self.session_pool[bootstrap_addr]
                        response = session.get(
                            f"http://{bootstrap_addr}/health",
                            timeout=(2, 3),
                            headers={"Connection": "keep-alive"},
                        )
                        if response.status_code == 200:
                            self.heartbeat_failures[bootstrap_addr] = 0

                            current_rooms = self.addr_to_rooms_tracker[
                                bootstrap_addr
                            ].copy()

                            for bootstrap_room in current_rooms:
                                if bootstrap_room not in self.request_status:
                                    self.addr_to_rooms_tracker[bootstrap_addr].discard(
                                        bootstrap_room
                                    )
                        else:
                            logger.info(
                                f"Attempting to reconnect to {bootstrap_addr}..."
                            )
                            self.heartbeat_failures[bootstrap_addr] = (
                                self.heartbeat_failures.get(bootstrap_addr, 0) + 1
                            )
                            with self.session_pool_lock:
                                if bootstrap_addr in self.session_pool:
                                    del self.session_pool[bootstrap_addr]
                    except Exception:
                        logger.info(f"Attempting to reconnect to {bootstrap_addr}...")
                        self.heartbeat_failures[bootstrap_addr] = (
                            self.heartbeat_failures.get(bootstrap_addr, 0) + 1
                        )

                    if (
                        self.heartbeat_failures.get(bootstrap_addr, 0)
                        >= self.max_failures
                    ):
                        self._handle_node_failure(bootstrap_addr)
                        with self.session_pool_lock:
                            if bootstrap_addr in self.session_pool:
                                del self.session_pool[bootstrap_addr]

        threading.Thread(target=decode_thread).start()
        threading.Thread(target=heartbeat_checker).start()

    def add_transfer_request(
        self,
        bootstrap_room: int,
        kv_indices: npt.NDArray[np.int32],
        index_slice: slice,
        is_last: bool,
        aux_index: Optional[int] = None,
        state_indices: Optional[List[int]] = None,
    ):
        assert self.disaggregation_mode == DisaggregationMode.PREFILL
        assert not is_last or (is_last and aux_index is not None)

        if (
            bootstrap_room not in self.request_status
            or self.check_status(bootstrap_room) == KVPoll.Failed
        ):
            logger.debug(
                "Request with bootstrap_room=%s already failed", bootstrap_room
            )
            return

        if bootstrap_room not in self.transfer_infos:
            return

        dst_infos = self.transfer_infos[bootstrap_room].keys()
        session_port_sum = sum(int(session.rsplit(":", 1)[1]) for session in dst_infos)
        shard_idx = session_port_sum % len(self.transfer_queues)

        kv_indices_copy = np.array(kv_indices, dtype=np.int32, copy=True)
        state_indices_copy = list(state_indices) if state_indices is not None else None

        self.transfer_queues[shard_idx].put(
            TransferKVChunk(
                room=bootstrap_room,
                prefill_kv_indices=kv_indices_copy,
                index_slice=index_slice,
                is_last=is_last,
                prefill_aux_index=aux_index,
                state_indices=state_indices_copy,
            )
        )

    def get_session_id(self):
        return self.engine.get_session_id()

    def _handle_node_failure(self, failed_bootstrap_addr):
        with self.connection_lock:
            keys_to_remove = [
                k for k in self.connection_pool if k.startswith(failed_bootstrap_addr)
            ]
            for k in keys_to_remove:
                del self.connection_pool[k]

            possible_affected_rooms = self.addr_to_rooms_tracker.get(
                failed_bootstrap_addr, []
            )
            self.prefill_info_table.pop(failed_bootstrap_addr, None)
            self.addr_to_rooms_tracker.pop(failed_bootstrap_addr, None)

        affected_rooms = []
        for room in possible_affected_rooms:
            if (
                room in self.request_status
                and self.check_status(room) != KVPoll.Success
            ):
                self.record_failure(
                    room,
                    f"Losing connection with prefill instance (bootstrap_addr: {failed_bootstrap_addr})",
                )
                self.update_status(room, KVPoll.Failed)
                affected_rooms.append(room)
        logger.error(
            f"Losing connection with prefill instance (bootstrap_addr: {failed_bootstrap_addr}), affected {len(affected_rooms)} requests"
        )


class P2PKVSender(CommonKVSender):
    def __init__(
        self,
        mgr: P2PKVManager,
        bootstrap_addr: str,
        bootstrap_room: int,
        dest_tp_ranks: List[int],
        pp_rank: int,
    ):
        super().__init__(mgr, bootstrap_addr, bootstrap_room, dest_tp_ranks, pp_rank)
        self.conclude_state = None
        self.init_time = time.time()

    def send(
        self,
        kv_indices: npt.NDArray[np.int32],
        state_indices: Optional[List[int]] = None,
    ):
        index_slice = slice(self.curr_idx, self.curr_idx + len(kv_indices))
        self.curr_idx += len(kv_indices)
        is_last = self.curr_idx == self.num_kv_indices

        # Special handling for cp
        if self.kv_mgr.enable_all_cp_ranks_for_transfer:
            kv_indices, index_slice = filter_kv_indices_for_cp_rank(
                self.kv_mgr,
                kv_indices,
                index_slice,
            )
        elif self.kv_mgr.is_dummy_cp_rank:
            if not is_last:
                return
            else:
                self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Success)
                return

        if not is_last:
            self.kv_mgr.add_transfer_request(
                self.bootstrap_room, kv_indices, index_slice, False
            )
        else:
            self.kv_mgr.add_transfer_request(
                self.bootstrap_room,
                kv_indices,
                index_slice,
                True,
                aux_index=self.aux_index,
                state_indices=state_indices,
            )

    def poll(self) -> KVPoll:
        if self.conclude_state is None:
            status = self.kv_mgr.check_status(self.bootstrap_room)
            if status in (KVPoll.Success, KVPoll.Failed):
                self.conclude_state = status
            elif status == KVPoll.Bootstrapping:
                if self.init_time is not None:
                    now = time.time()
                    elapsed = now - self.init_time
                    if elapsed >= self.kv_mgr.bootstrap_timeout:
                        logger.warning_once(
                            "Some requests timed out when bootstrapping, "
                            "which means prefill instances fail to receive the KV indices from the decode instance of this request. "
                            "If a greater mean TTFT is acceptable, you can 'export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600' (10 minutes) to relax the timeout condition. "
                        )
                        self.kv_mgr.record_failure(
                            self.bootstrap_room,
                            f"Request {self.bootstrap_room} timed out after {elapsed:.1f}s in KVPoll.Bootstrapping",
                        )
                        self.conclude_state = KVPoll.Failed
                        return KVPoll.Failed

            return status
        else:
            return self.conclude_state

    def clear(self) -> None:
        if self.bootstrap_room in self.kv_mgr.request_status:
            self.kv_mgr.request_status.pop(self.bootstrap_room)

    def failure_exception(self):
        if self.conclude_state is None:
            self.conclude_state = KVPoll.Failed

        self.clear()

        with self.kv_mgr.failure_lock:
            failure_reason = self.kv_mgr.failure_records.pop(
                self.bootstrap_room, "P2P transfer failed due to an unknown reason"
            )
        raise P2PTransferError(self.bootstrap_room, failure_reason)

    def abort(self):
        self.kv_mgr.record_failure(
            self.bootstrap_room,
            "Aborted by AbortReq.",
        )
        # Explicitly set the status to failure since this request has been aborted
        self.conclude_state = KVPoll.Failed


class P2PKVReceiver(CommonKVReceiver):
    def __init__(
        self,
        mgr: P2PKVManager,
        bootstrap_addr: str,
        bootstrap_room: Optional[int] = None,
    ):
        self.session_id = mgr.get_session_id()
        self.init_time = None
        super().__init__(mgr, bootstrap_addr, bootstrap_room)

    def _register_kv_args(self):
        tp_rank = self.kv_mgr.kv_args.engine_rank
        physical_gpu_id = (
            self.kv_mgr.server_args.base_gpu_id
            + tp_rank * self.kv_mgr.server_args.gpu_id_step
        )

        packed_kv_handles = b"".join(self.kv_mgr.kv_handles)
        packed_state_handles = b"".join(self.kv_mgr.state_handles)
        packed_aux_data_ptrs = b"".join(
            struct.pack("Q", ptr) for ptr in self.kv_mgr.kv_args.aux_data_ptrs
        )

        dst_tp_rank = str(tp_rank).encode("ascii")
        dst_attn_tp_size = str(self.kv_mgr.attn_tp_size).encode("ascii")
        kv_item_len = self.kv_mgr.kv_args.kv_item_lens[0]
        dst_kv_item_len = str(kv_item_len).encode("ascii")
        gpu_id_str = str(physical_gpu_id)

        state_item_lens = getattr(self.kv_mgr.kv_args, "state_item_lens", []) or []
        packed_state_item_lens = (
            b"".join(struct.pack("I", length) for length in state_item_lens)
            if state_item_lens
            else b""
        )

        state_dim_per_tensor = (
            getattr(self.kv_mgr.kv_args, "state_dim_per_tensor", []) or []
        )
        packed_state_dim_per_tensor = (
            b"".join(struct.pack("I", dim) for dim in state_dim_per_tensor)
            if state_dim_per_tensor
            else b""
        )

        if self.kv_mgr.state_handles and (
            len(self.kv_mgr.state_handles) != len(self.kv_mgr.state_base_offsets)
        ):
            raise RuntimeError(
                "state_handles and state_base_offsets length mismatch: "
                f"{len(self.kv_mgr.state_handles)} vs "
                f"{len(self.kv_mgr.state_base_offsets)}"
            )

        packed_state_base_offsets = (
            b"".join(
                struct.pack("Q", int(offset))
                for offset in self.kv_mgr.state_base_offsets
            )
            if self.kv_mgr.state_base_offsets
            else b""
        )

        for bootstrap_info in self.bootstrap_infos:
            sock, lock = self._connect_to_bootstrap_server(bootstrap_info)
            with lock:
                sock.send_multipart(
                    [
                        "None".encode("ascii"),
                        self.kv_mgr.local_ip.encode("ascii"),
                        str(self.kv_mgr.rank_port).encode("ascii"),
                        self.session_id.encode("ascii"),
                        packed_kv_handles,
                        packed_aux_data_ptrs,
                        gpu_id_str.encode("ascii"),
                        packed_state_handles,
                        dst_tp_rank,
                        dst_attn_tp_size,
                        dst_kv_item_len,
                        packed_state_item_lens,
                        packed_state_dim_per_tensor,
                        packed_state_base_offsets,
                    ]
                )

    def init(
        self,
        prefill_dp_rank: int,
    ):
        super().init(prefill_dp_rank)

    def send_metadata(
        self,
        kv_indices: npt.NDArray[np.int32],
        aux_index: Optional[int] = None,
        state_indices: Optional[List[int]] = None,
    ):
        if self.bootstrap_infos is None:
            self.kv_mgr.record_failure(
                self.bootstrap_room,
                f"Could not fetch prefill parallel info from bootstrap_addr: {self.bootstrap_addr}",
            )
            self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Failed)
            return

        for bootstrap_info in self.bootstrap_infos:
            sock, lock = self._connect_to_bootstrap_server(bootstrap_info)
            is_dummy = bootstrap_info["is_dummy"]

            with lock:
                sock.send_multipart(
                    [
                        str(self.bootstrap_room).encode("ascii"),
                        self.kv_mgr.local_ip.encode("ascii"),
                        str(self.kv_mgr.rank_port).encode("ascii"),
                        self.session_id.encode("ascii"),
                        kv_indices.tobytes() if not is_dummy else b"",
                        str(aux_index).encode("ascii") if not is_dummy else b"",
                        (
                            np.array(
                                state_indices,
                                dtype=np.int32,
                            ).tobytes()
                            if not is_dummy and state_indices is not None
                            else b""
                        ),
                        str(self.required_dst_info_num).encode("ascii"),
                    ]
                )
        self.init_time = time.time()

    def poll(self) -> KVPoll:
        if self.conclude_state is None:
            status = self.kv_mgr.check_status(self.bootstrap_room)
            if status in (KVPoll.Success, KVPoll.Failed):
                self.conclude_state = status
            elif status == KVPoll.WaitingForInput:
                if self.init_time is not None:
                    now = time.time()
                    elapsed = now - self.init_time
                    if elapsed >= self.kv_mgr.waiting_timeout:
                        logger.warning_once(
                            "Some requests fail to receive KV Cache transfer done signal after bootstrapping. "
                            "If a greater mean TTFT is acceptable, you can 'export SGLANG_DISAGGREGATION_WAITING_TIMEOUT=600' (10 minutes) to relax the timeout condition. "
                        )
                        self.kv_mgr.record_failure(
                            self.bootstrap_room,
                            f"Request {self.bootstrap_room} timed out after {elapsed:.1f}s in KVPoll.WaitingForInput",
                        )
                        self.conclude_state = KVPoll.Failed
                        return KVPoll.Failed

            return status
        else:
            return self.conclude_state

    def clear(self) -> None:
        if self.bootstrap_room in self.kv_mgr.request_status:
            self.kv_mgr.request_status.pop(self.bootstrap_room)

        if self.bootstrap_room in self.kv_mgr.required_prefill_response_num_table:
            self.kv_mgr.required_prefill_response_num_table.pop(self.bootstrap_room)

        if self.bootstrap_room in self.kv_mgr.prefill_response_tracker:
            self.kv_mgr.prefill_response_tracker.pop(self.bootstrap_room)

    def failure_exception(self):
        if self.conclude_state is None:
            self.conclude_state = KVPoll.Failed

        self.clear()

        with self.kv_mgr.failure_lock:
            failure_reason = self.kv_mgr.failure_records.pop(
                self.bootstrap_room, "P2P transfer failed due to an unknown reason"
            )
        raise P2PTransferError(self.bootstrap_room, failure_reason)

    def abort(self):
        self.kv_mgr.record_failure(
            self.bootstrap_room,
            "Aborted by AbortReq.",
        )
        # Explicitly set the status to failure since this request has been aborted
        self.conclude_state = KVPoll.Failed


class P2PKVBootstrapServer(CommonKVBootstrapServer):
    pass
