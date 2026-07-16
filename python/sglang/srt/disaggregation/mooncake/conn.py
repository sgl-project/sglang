from __future__ import annotations

import concurrent.futures
import dataclasses
import json
import logging
import os
import struct
import threading
import time
from collections import defaultdict
from typing import List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
from prometheus_client import Counter

from sglang.srt.disaggregation.base.conn import KVArgs, KVPoll, StateType
from sglang.srt.disaggregation.common.conn import (
    CommonKVBootstrapServer,
    CommonKVManager,
    CommonKVReceiver,
    CommonKVSender,
    KVTransferError,
)
from sglang.srt.disaggregation.common.staging_handler import (
    DecodeStagingContext,
    PrefillStagingContext,
    StagingRegisterInfo,
    StagingTransferInfo,
)
from sglang.srt.disaggregation.common.utils import (
    AuxDataCodec,
    FastQueue,
    TransferKVChunk,
    group_concurrent_contiguous,
    pack_int_lists,
    unpack_int_lists,
)
from sglang.srt.disaggregation.mooncake.utils import (
    check_mooncake_custom_mem_pool_enabled,
)
from sglang.srt.disaggregation.utils import (
    DisaggregationMode,
    compute_mamba_state_slice_blocks,
)
from sglang.srt.distributed.parallel_state import get_mooncake_transfer_engine
from sglang.srt.environ import envs
from sglang.srt.observability.mooncake_trace import (
    MooncakeRequestStage,
    mooncake_trace_func,
    mooncake_trace_slice,
)
from sglang.srt.observability.trace import (
    TraceNullContext,
    TraceReqContext,
    trace_set_thread_info,
)
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils.network import NetworkAddress

logger = logging.getLogger(__name__)

FAILED_SESSION_RECOVERIES = Counter(
    "sglang:failed_session_recoveries_total",
    "Number of mooncake_session_ids un-blacklisted via probe.",
)


# decode
@dataclasses.dataclass
class TransferInfo:
    room: int
    endpoint: str
    dst_port: int
    mooncake_session_id: str
    dst_kv_indices: npt.NDArray[np.int32]
    dst_aux_index: int
    dst_state_indices: List[List[int]]  # parallel to receiver's state_types
    required_dst_info_num: int
    is_dummy: bool
    decode_prefix_len: Optional[int] = None
    spec_metadata: Optional[dict] = None
    # Note: always put the optional staging field at the final (it will be set through 'STAGING_RSP' pkg when needed)
    staging: Optional[StagingTransferInfo] = None

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
            dst_state_indices = unpack_int_lists(msg[6], "i")
            is_dummy = False
        return cls(
            room=int(msg[0].decode("ascii")),
            endpoint=msg[1].decode("ascii"),
            dst_port=int(msg[2].decode("ascii")),
            mooncake_session_id=msg[3].decode("ascii"),
            dst_kv_indices=dst_kv_indices,
            dst_aux_index=dst_aux_index,
            dst_state_indices=dst_state_indices,
            required_dst_info_num=int(msg[7].decode("ascii")),
            is_dummy=is_dummy,
            decode_prefix_len=(
                int(msg[8].decode("ascii")) if len(msg) > 8 and msg[8] != b"" else None
            ),
            spec_metadata=(
                json.loads(msg[9].decode("utf-8"))
                if len(msg) > 9 and msg[9] != b""
                else None
            ),
        )


# decode
@dataclasses.dataclass
class KVArgsRegisterInfo:
    room: str
    endpoint: str
    dst_port: int
    mooncake_session_id: str
    dst_kv_ptrs: list[int]
    dst_aux_ptrs: list[int]
    dst_state_data_ptrs: List[List[int]]  # parallel to state_types (same below)
    dst_tp_rank: int
    dst_attn_tp_size: int
    dst_kv_item_len: int
    # for mamba state different tp slice transfer
    dst_state_item_lens: List[List[int]]
    dst_state_dim_per_tensor: List[List[int]]
    # Note: always put the staging field at the final (since the staging field is optional and contains multiple inputs)
    staging: Optional[StagingRegisterInfo] = None

    @classmethod
    def from_zmq(cls, msg: List[bytes]):
        return cls(
            room=str(msg[0].decode("ascii")),
            endpoint=msg[1].decode("ascii"),
            dst_port=int(msg[2].decode("ascii")),
            mooncake_session_id=msg[3].decode("ascii"),
            dst_kv_ptrs=list(struct.unpack(f"{len(msg[4])//8}Q", msg[4])),
            dst_aux_ptrs=list(struct.unpack(f"{len(msg[5])//8}Q", msg[5])),
            dst_state_data_ptrs=unpack_int_lists(msg[6], "Q"),
            dst_tp_rank=int(msg[7].decode("ascii")),
            dst_attn_tp_size=int(msg[8].decode("ascii")),
            dst_kv_item_len=int(msg[9].decode("ascii")),
            dst_state_item_lens=(
                unpack_int_lists(msg[10], "I") if len(msg) > 10 else []
            ),
            dst_state_dim_per_tensor=(
                unpack_int_lists(msg[11], "I") if len(msg) > 11 else []
            ),
            # Note: always put the staging field at the final
            staging=StagingRegisterInfo.from_zmq_fields(msg, 12),
        )


class MooncakeKVManager(CommonKVManager):
    AUX_DATA_HEADER = b"AUX_DATA"

    def __init__(
        self,
        args: KVArgs,
        disaggregation_mode: DisaggregationMode,
        server_args: ServerArgs,
        is_mla_backend: Optional[bool] = False,
    ):
        super().__init__(args, disaggregation_mode, server_args, is_mla_backend)
        self.init_engine()
        self.register_buffer_to_engine()
        self.enable_staging = envs.SGLANG_DISAGG_STAGING_BUFFER.get()
        self.enable_trace = server_args.enable_trace
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            self.start_prefill_thread()
            self.session_failures = defaultdict(int)
            self.failed_sessions = set()
            self.session_lock = threading.Lock()
            self.dspark_hidden_done_rooms = set()
            self.dspark_hidden_done_lock = threading.Lock()
            # Determine the number of threads to use for kv sender
            cpu_count = os.cpu_count()
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
                f"The environment variable SGLANG_DISAGGREGATION_THREAD_POOL_SIZE={transfer_thread_pool_size} must be "
                f"greater than or equal to SGLANG_DISAGGREGATION_QUEUE_SIZE={transfer_queue_size}."
            )
            self.executors = [
                concurrent.futures.ThreadPoolExecutor(
                    transfer_thread_pool_size // transfer_queue_size
                )
                for _ in range(transfer_queue_size)
            ]
            self.enable_custom_mem_pool, self.custom_mem_pool_type = (
                check_mooncake_custom_mem_pool_enabled()
            )
            self._staging_ctx = PrefillStagingContext() if self.enable_staging else None
            if self.enable_staging:
                self._init_staging_buffers(len(self.transfer_queues))
            for i, (queue, executor) in enumerate(
                zip(self.transfer_queues, self.executors)
            ):
                threading.Thread(
                    target=self.transfer_worker,
                    args=(
                        queue,
                        executor,
                        (
                            self._staging_ctx.buffers[i]
                            if self.enable_staging and self._staging_ctx.buffers
                            else None
                        ),
                        i,
                    ),
                    daemon=True,
                ).start()
            self.enable_failed_session_probe = (
                envs.SGLANG_ENABLE_FAILED_SESSION_PROBE.get()
            )
            if self.enable_failed_session_probe:
                self.failed_session_probe_interval = (
                    envs.SGLANG_FAILED_SESSION_PROBE_INTERVAL_S.get()
                )
                self._failed_session_probe_shutdown = threading.Event()
                threading.Thread(
                    target=self._failed_session_probe_loop,
                    name="MooncakeFailedSessionProbe",
                    daemon=True,
                ).start()
        elif self.disaggregation_mode == DisaggregationMode.DECODE:
            self._staging_ctx = DecodeStagingContext() if self.enable_staging else None
            if self.enable_staging:
                self._init_staging_allocator()
                self._staging_handler = None
                self._chunk_writer_counts: dict = defaultdict(lambda: defaultdict(list))
            self.start_decode_thread()

    def mark_dspark_hidden_done(
        self,
        bootstrap_room: int,
        state_indices: Optional[List] = None,
    ) -> None:
        if not hasattr(self, "dspark_hidden_done_rooms"):
            return
        with self.dspark_hidden_done_lock:
            room = int(bootstrap_room)
            if room in self.dspark_hidden_done_rooms:
                return
            self.dspark_hidden_done_rooms.add(room)
        pool = getattr(self, "dspark_hidden_pool", None)
        state_idx = self._dspark_hidden_state_index()
        if (
            pool is not None
            and state_indices is not None
            and state_idx is not None
            and state_idx < len(state_indices)
        ):
            indices = state_indices[state_idx]
            if indices is not None and len(indices) > 0:
                pool.free([int(idx) for idx in indices])

    def pop_dspark_hidden_done(self, bootstrap_room: int) -> bool:
        if not hasattr(self, "dspark_hidden_done_rooms"):
            return False
        with self.dspark_hidden_done_lock:
            room = int(bootstrap_room)
            if room not in self.dspark_hidden_done_rooms:
                return False
            self.dspark_hidden_done_rooms.remove(room)
            return True

    def init_engine(self):
        self.engine = get_mooncake_transfer_engine()

    def register_buffer_to_engine(self):
        # Batch register KV data buffers
        if self.kv_args.kv_data_ptrs and self.kv_args.kv_data_lens:
            self.engine.batch_register(
                self.kv_args.kv_data_ptrs, self.kv_args.kv_data_lens
            )

        # Batch register auxiliary data buffers
        if self.kv_args.aux_data_ptrs and self.kv_args.aux_data_lens:
            self.engine.batch_register(
                self.kv_args.aux_data_ptrs, self.kv_args.aux_data_lens
            )

        for ptrs, lens in zip(
            self.kv_args.state_data_ptrs, self.kv_args.state_data_lens
        ):
            if ptrs and lens:
                self.engine.batch_register(ptrs, lens)

    def deregister_buffer_to_engine(self):
        if self.kv_args.kv_data_ptrs:
            self.engine.batch_deregister(self.kv_args.kv_data_ptrs)

        if self.kv_args.aux_data_ptrs:
            self.engine.batch_deregister(self.kv_args.aux_data_ptrs)

        for ptrs in self.kv_args.state_data_ptrs or []:
            if ptrs:
                self.engine.batch_deregister(ptrs)

        if hasattr(self, "connection_pool"):
            with self.connection_lock:
                self.connection_pool.clear()

    # ------------------------------------------------------------------
    # Staging buffer methods (all delegate to staging_handler.py)
    # ------------------------------------------------------------------

    def register_staging_room_bootstrap(self, room, bootstrap_infos, receiver):
        self._staging_ctx.room_bootstrap[room] = bootstrap_infos
        self._staging_ctx.room_receivers[room] = receiver

    def set_kv_buffer_tensors(self, k_buffers: list, v_buffers: list, page_size: int):
        self.kv_buffer_tensors = {
            "k_buffers": k_buffers,
            "v_buffers": v_buffers,
            "page_size": page_size,
        }

    def _init_staging_buffers(self, count: int):
        from sglang.srt.disaggregation.common.staging_handler import (
            init_staging_buffers,
        )

        self._staging_ctx.buffers = init_staging_buffers(
            lambda ptr, size: self.engine.batch_register([ptr], [size]),
            self.kv_args,
            count,
        )
        self.kv_buffer_tensors = None

    def _init_staging_allocator(self):
        from sglang.srt.disaggregation.common.staging_handler import (
            init_staging_allocator,
        )

        self._staging_ctx.allocator = init_staging_allocator(
            lambda ptr, size: self.engine.batch_register([ptr], [size]),
            self.kv_args,
        )
        self.kv_buffer_tensors = None

    def _handle_staging_req(self, msg):
        from sglang.srt.disaggregation.common.staging_handler import (
            handle_staging_req,
        )

        room = int(msg[1].decode("ascii"))
        session_id = msg[4].decode("ascii")
        handler = self._staging_handler
        assert (
            handler is not None
        ), "STAGING_REQ received before staging handler initialized"
        decode_req = handler._room_to_decode_req.get(room)
        if decode_req is None:
            logger.warning(
                "STAGING_REQ received for unregistered room=%s, skipping",
                room,
            )
            return
        prefill_tp = decode_req.kv_receiver.prefill_info.attn_tp_size
        handle_staging_req(
            msg,
            self._staging_ctx.allocator,
            self.kv_args,
            self.attn_tp_size,
            prefill_tp,
            getattr(self, "kv_buffer_tensors", None),
            self._staging_ctx.room_receivers,
            self._staging_ctx.room_bootstrap,
        )

        receiver = self._staging_ctx.room_receivers.get(room)
        if receiver is not None:
            handler.register_wm_subscriber(receiver, session_id)

    def _is_watermark_ready(
        self, session_id: str, alloc_round: int, alloc_end: int
    ) -> bool:
        from sglang.srt.disaggregation.common.staging_handler import (
            is_watermark_ready,
        )

        return is_watermark_ready(self._staging_ctx, session_id, alloc_round, alloc_end)

    def _try_create_staging_strategy(self, staging_buffer):
        if not self.enable_staging or self.kv_buffer_tensors is None:
            return None
        from sglang.srt.disaggregation.common.staging_handler import (
            PrefillStagingStrategy,
        )

        return PrefillStagingStrategy(self, staging_buffer)

    def _send_chunk_ready(self, req, chunk_idx, kv_chunk, prefill_unique_rank):
        """Notify decode that a non-last staging chunk RDMA is complete."""
        try:
            na = NetworkAddress(req.endpoint, req.dst_port)
            self._connect(
                na.to_tcp(),
                is_ipv6=na.is_ipv6,
            ).send_multipart(
                [
                    b"CHUNK_READY",
                    str(req.room).encode("ascii"),
                    str(chunk_idx).encode("ascii"),
                    str(kv_chunk.index_slice.start).encode("ascii"),
                    str(len(kv_chunk.prefill_kv_indices)).encode("ascii"),
                    req.mooncake_session_id.encode("ascii"),
                    str(prefill_unique_rank).encode("ascii"),
                ]
            )
        except Exception:
            pass

    def _do_staging_transfer(
        self,
        staging_strategy,
        kv_chunk,
        req,
        target_info,
        chunked_dst_kv_indice,
        executor,
        queue,
        prefill_unique_rank,
    ):
        """Execute staging transfer for one chunk. Returns (ret, deferred).

        Handles readiness check, transfer, fallback, and CHUNK_READY notification.
        deferred=True means caller should re-enqueue and break.
        """
        _tp = self.attn_tp_rank
        ready, chunk_idx, c_offset, _, _ = staging_strategy.check_ready(
            req,
            kv_chunk.index_slice.start,
            len(kv_chunk.prefill_kv_indices),
        )
        if not ready:
            from sglang.srt.disaggregation.common.staging_buffer import StagingAllocator

            if c_offset == StagingAllocator.ALLOC_OVERSIZED:
                raise RuntimeError(
                    f"[Staging] Chunk staging allocation permanently failed: "
                    f"chunk exceeds ring buffer total size (room={kv_chunk.room}). "
                    f"Increase SGLANG_DISAGG_STAGING_POOL_SIZE_MB."
                )
            queue.put(kv_chunk)
            return (-1, True)

        ret = staging_strategy.transfer(
            req.mooncake_session_id,
            kv_chunk.prefill_kv_indices,
            target_info.staging.base_ptr + c_offset,
            target_info.staging.total_size - c_offset,
            target_info,
        )
        if ret == -1:
            logger.warning(
                f"[Staging][tp{_tp}] Falling back to per-token slice path "
                f"(room={kv_chunk.room})"
            )
            ret = self.send_kvcache_slice(
                req.mooncake_session_id,
                kv_chunk.prefill_kv_indices,
                target_info.dst_kv_ptrs,
                chunked_dst_kv_indice,
                target_info.dst_tp_rank,
                target_info.dst_attn_tp_size,
                target_info.dst_kv_item_len,
                executor,
            )
        elif ret == 0 and not kv_chunk.is_last_chunk:
            self._send_chunk_ready(req, chunk_idx, kv_chunk, prefill_unique_rank)
        return (ret, False)

    def _prefetch_staging_reqs(self, room: int):
        if not self.enable_staging or self.kv_buffer_tensors is None:
            return

        room_infos = self.transfer_infos.get(room, {})
        needs_staging = any(
            not tinfo.is_dummy
            and self.decode_kv_args_table.get(tinfo.mooncake_session_id) is not None
            and self.decode_kv_args_table[tinfo.mooncake_session_id].dst_attn_tp_size
            != self.attn_tp_size
            for tinfo in room_infos.values()
        )
        if not needs_staging:
            return

        from sglang.srt.disaggregation.common.staging_handler import (
            prefetch_staging_reqs,
        )

        prefetch_staging_reqs(
            room,
            self.transfer_infos,
            self.kv_buffer_tensors,
            self.server_args.chunked_prefill_size,
            self._staging_ctx.prefetch_requested,
            self._staging_ctx.prefetch_sockets,
        )

    def send_kvcache_staged(
        self,
        mooncake_session_id: str,
        prefill_kv_indices: npt.NDArray[np.int32],
        dst_staging_ptr: int,
        dst_staging_size: int,
        dst_tp_rank: int,
        dst_attn_tp_size: int,
        dst_kv_item_len: int,
        staging_buffer=None,
    ) -> int:
        """Transfer KV cache via staging buffers (gather -> bulk RDMA -> scatter on decode)."""
        from sglang.srt.disaggregation.common.staging_buffer import (
            compute_head_slice_params,
            compute_staging_layout,
            resolve_total_kv_heads,
        )

        if self.kv_buffer_tensors is None or staging_buffer is None:
            return -1

        k_buffers = self.kv_buffer_tensors["k_buffers"]
        v_buffers = self.kv_buffer_tensors["v_buffers"]
        page_size = self.kv_buffer_tensors["page_size"]
        num_layers = len(k_buffers)
        head_dim = k_buffers[0].shape[-1]
        dtype_size = k_buffers[0].element_size()

        total_kv_heads = resolve_total_kv_heads(self.kv_args, self.attn_tp_size)

        local_tp_rank = self.kv_args.engine_rank % self.attn_tp_size
        src_head_start, num_heads_to_send, _, _ = compute_head_slice_params(
            self.attn_tp_size,
            dst_attn_tp_size,
            local_tp_rank,
            dst_tp_rank,
            total_kv_heads,
        )

        num_tokens = len(prefill_kv_indices) * page_size
        per_layer_bytes = num_tokens * num_heads_to_send * head_dim * dtype_size
        per_rank_bytes = per_layer_bytes * num_layers * 2

        num_writers, writer_rank_bytes, total_staging_needed = compute_staging_layout(
            self.attn_tp_size,
            dst_attn_tp_size,
            dst_tp_rank,
            total_kv_heads,
            num_tokens,
            head_dim * dtype_size,
            num_layers,
        )
        writer_idx = local_tp_rank % num_writers if num_writers > 1 else 0
        rank_offset = sum(writer_rank_bytes[:writer_idx])

        if not staging_buffer.fits(per_rank_bytes):
            logger.warning(
                f"Prefill staging too small for {per_rank_bytes} bytes, falling back"
            )
            return -1
        if dst_staging_size < total_staging_needed:
            logger.warning(
                f"Decode staging too small: need {total_staging_needed} bytes "
                f"({num_writers if self.attn_tp_size > dst_attn_tp_size else 1} writers "
                f"x {per_rank_bytes} bytes/rank), have {dst_staging_size}, falling back"
            )
            return -1

        from sglang.srt.disaggregation.common.staging_buffer import (
            gather_all_layers_to_staging,
        )

        gather_all_layers_to_staging(
            k_buffers,
            v_buffers,
            prefill_kv_indices,
            staging_buffer,
            src_head_start,
            num_heads_to_send,
            page_size,
            self.kv_args.gpu_id,
        )

        dst_write_ptr = dst_staging_ptr + rank_offset
        ret = self._transfer_data(
            mooncake_session_id,
            [(staging_buffer.get_ptr(), dst_write_ptr, per_rank_bytes)],
        )
        if ret != 0:
            raise RuntimeError(
                f"[Staging] Bulk RDMA transfer failed with ret={ret}. "
                f"src_ptr=0x{staging_buffer.get_ptr():x}, "
                f"dst_ptr=0x{dst_write_ptr:x}, size={per_rank_bytes}. "
                f"The decode staging buffer may not be properly registered."
            )
        return ret

    def _transfer_data(self, mooncake_session_id, transfer_blocks):
        if not transfer_blocks:
            return 0

        src_addrs, dst_addrs, lengths = zip(*transfer_blocks)
        return self.engine.batch_transfer_sync(
            mooncake_session_id, list(src_addrs), list(dst_addrs), list(lengths)
        )

    def _send_kvcache_generic(
        self,
        mooncake_session_id: str,
        src_data_ptrs: list[int],
        dst_data_ptrs: list[int],
        item_lens: list[int],
        prefill_data_indices: npt.NDArray[np.int32],
        dst_data_indices: npt.NDArray[np.int32],
        executor: concurrent.futures.ThreadPoolExecutor,
        state_type: Optional[StateType] = None,
        force_flat: bool = False,
    ) -> int:
        """
        Generic KV cache transfer supporting both MHA and MLA architectures.
        This method is used by both send_kvcache (full pool) and maybe_send_extra.

        ``force_flat`` uses the MLA-style flat (single-buffer-per-layer) layout
        even on a non-MLA backend, for K-only state buffers (e.g. MiniMax sparse
        index) whose per-layer list must not be half-split into K/V.
        """
        # Group by indices for optimization
        prefill_kv_blocks, dst_kv_blocks = group_concurrent_contiguous(
            prefill_data_indices, dst_data_indices
        )

        layers_params = None

        # Decode pp size should be equal to prefill pp size or 1
        if self.is_mla_backend or self.is_hybrid_mla_backend or force_flat:
            src_kv_ptrs, dst_kv_ptrs, layers_current_pp_stage = (
                self.get_mla_kv_ptrs_with_pp(src_data_ptrs, dst_data_ptrs, state_type)
            )
            layers_params = [
                (
                    src_kv_ptrs[layer_id],
                    dst_kv_ptrs[layer_id],
                    item_lens[layer_id],
                )
                for layer_id in range(layers_current_pp_stage)
            ]
        else:
            src_k_ptrs, src_v_ptrs, dst_k_ptrs, dst_v_ptrs, layers_current_pp_stage = (
                self.get_mha_kv_ptrs_with_pp(src_data_ptrs, dst_data_ptrs)
            )
            # item_lens structure: [k_layer0, k_layer1, ..., k_layerN, v_layer0, v_layer1, ..., v_layerN]
            # Use correct item lengths for K and V separately
            if layers_current_pp_stage > len(dst_k_ptrs):
                logger.error(
                    "Prefill transfer kvcache error, layers_current_pp_stage is out of range: "
                    f"layers_current_pp_stage={layers_current_pp_stage}, len(dst_k_ptrs)={len(dst_k_ptrs)}"
                )
                return -1
            layers_params = [
                (
                    src_k_ptrs[layer_id],
                    dst_k_ptrs[layer_id],
                    item_lens[layer_id],  # K item length
                )
                for layer_id in range(layers_current_pp_stage)
            ] + [
                (
                    src_v_ptrs[layer_id],
                    dst_v_ptrs[layer_id],
                    item_lens[layers_current_pp_stage + layer_id],  # V item length
                )
                for layer_id in range(layers_current_pp_stage)
            ]
        assert layers_params is not None

        def set_transfer_blocks(
            src_ptr: int, dst_ptr: int, item_len: int
        ) -> List[Tuple[int, int, int]]:
            transfer_blocks = []
            for prefill_index, decode_index in zip(prefill_kv_blocks, dst_kv_blocks):
                src_addr = src_ptr + int(prefill_index[0]) * item_len
                dst_addr = dst_ptr + int(decode_index[0]) * item_len
                length = item_len * len(prefill_index)
                transfer_blocks.append((src_addr, dst_addr, length))
            return transfer_blocks

        # Worker function for processing a single layer
        def process_layer(src_ptr: int, dst_ptr: int, item_len: int) -> int:
            transfer_blocks = set_transfer_blocks(src_ptr, dst_ptr, item_len)
            return self._transfer_data(mooncake_session_id, transfer_blocks)

        # Worker function for processing all layers in a batch
        def process_layers(layers_params: List[Tuple[int, int, int]]) -> int:
            transfer_blocks = []
            for src_ptr, dst_ptr, item_len in layers_params:
                transfer_blocks.extend(set_transfer_blocks(src_ptr, dst_ptr, item_len))
            return self._transfer_data(mooncake_session_id, transfer_blocks)

        if self.enable_custom_mem_pool:
            futures = [
                executor.submit(
                    process_layer,
                    src_ptr,
                    dst_ptr,
                    item_len,
                )
                for (src_ptr, dst_ptr, item_len) in layers_params
            ]
            for future in concurrent.futures.as_completed(futures):
                status = future.result()
                if status != 0:
                    for f in futures:
                        f.cancel()
                    return status
            return 0
        else:
            # Combining all layers' params in one batch transfer is more efficient
            # compared to using multiple threads
            return process_layers(layers_params)

    def send_kvcache(
        self,
        mooncake_session_id: str,
        prefill_kv_indices: npt.NDArray[np.int32],
        dst_kv_ptrs: list[int],
        dst_kv_indices: npt.NDArray[np.int32],
        executor: concurrent.futures.ThreadPoolExecutor,
    ):
        return self._send_kvcache_generic(
            mooncake_session_id=mooncake_session_id,
            src_data_ptrs=self.kv_args.kv_data_ptrs,
            dst_data_ptrs=dst_kv_ptrs,
            item_lens=self.kv_args.kv_item_lens,
            prefill_data_indices=prefill_kv_indices,
            dst_data_indices=dst_kv_indices,
            executor=executor,
        )

    def send_kvcache_slice(
        self,
        mooncake_session_id: str,
        prefill_kv_indices: npt.NDArray[np.int32],
        dst_kv_ptrs: list[int],
        dst_kv_indices: npt.NDArray[np.int32],
        dst_tp_rank: int,
        dst_attn_tp_size: int,
        dst_kv_item_len: int,
        executor: concurrent.futures.ThreadPoolExecutor,
    ):
        """
        Sends KV cache slices from this Prefill rank to a target Decode rank,
        supporting generic M-to-N TP size configurations.

        NOTE: This implementation calls the transfer engine for each token slot within
        each page to ensure correctness for any page_size and head-slicing configuration.
        This may introduce performance overhead (increased TTFT) for long sequences.
        """
        # Extract configuration
        local_tp_rank_in_group = self.kv_args.engine_rank % self.attn_tp_size
        src_kv_item_len = self.kv_args.kv_item_lens[0]
        dst_tp_rank_in_group = dst_tp_rank % dst_attn_tp_size
        page_size = self.kv_args.page_size

        # Use total KV head count (not per-rank) for correct head distribution.
        # Per-rank kv_head_num is max(1, total//tp) which loses info when total < tp.
        total_kv_heads = getattr(self.kv_args, "total_kv_head_num", 0)
        if total_kv_heads <= 0:
            total_kv_heads = self.kv_args.kv_head_num * self.attn_tp_size

        src_heads_per_rank = max(1, total_kv_heads // self.attn_tp_size)
        dst_heads_per_rank = max(1, total_kv_heads // dst_attn_tp_size)
        bytes_per_head_slice_to_send = (
            dst_kv_item_len // page_size // dst_heads_per_rank
        )

        # GQA replication: how many prefill ranks share the same KV head
        src_replication = max(1, self.attn_tp_size // total_kv_heads)

        # Determine slicing parameters based on TP configuration
        if self.attn_tp_size > dst_attn_tp_size:
            # Send KVCache from multiple prefill instances to 1 decode instance
            src_head_start_offset = 0
            num_heads_to_send = src_heads_per_rank
            unique_head_idx = local_tp_rank_in_group // src_replication
            dst_head_start_offset = (
                unique_head_idx * src_heads_per_rank
            ) % dst_heads_per_rank
        else:
            # Send KVCache from 1 prefill instance to multiple decode instances
            # GQA replication (total_kv_heads < dst_attn_tp_size): consecutive decode
            # ranks share one KV head (QKVParallelLinear: tp_rank // num_kv_head_replicas),
            # so map by integer division NOT modulo or ranks 1..r-1 fetch the wrong head.
            dst_replication = max(1, dst_attn_tp_size // total_kv_heads)
            unique_dst_head_idx = dst_tp_rank_in_group // dst_replication
            src_head_start_offset = (
                unique_dst_head_idx * dst_heads_per_rank
            ) % src_heads_per_rank
            num_heads_to_send = dst_heads_per_rank
            dst_head_start_offset = 0

        src_k_ptrs, src_v_ptrs, dst_k_ptrs, dst_v_ptrs, layers_current_pp_stage = (
            self.get_mha_kv_ptrs_with_pp(self.kv_args.kv_data_ptrs, dst_kv_ptrs)
        )

        # Calculate precise byte offset and length for the sub-slice within the token
        src_head_slice_offset = src_head_start_offset * bytes_per_head_slice_to_send
        dst_head_slice_offset = dst_head_start_offset * bytes_per_head_slice_to_send
        heads_bytes_per_token_to_send = num_heads_to_send * bytes_per_head_slice_to_send

        # Sanity check: The data sub-slice to be sent should fit into the dst buffer.
        # This means heads_bytes_per_token_to_send <= (dst_kv_item_len // page_size)
        if heads_bytes_per_token_to_send > (dst_kv_item_len // page_size):
            logger.error(
                f"[{mooncake_session_id}] slice size ({heads_bytes_per_token_to_send}) exceeds "
                f"target token slot size ({dst_kv_item_len // page_size})"
            )
            return -1

        prefill_page_indices = prefill_kv_indices.reshape(-1, 1).astype(np.int64)
        decode_page_indices = dst_kv_indices.reshape(-1, 1).astype(np.int64)
        tokens_per_page = np.arange(page_size, dtype=np.int64).reshape(1, -1)
        bytes_per_token_on_prefill = src_kv_item_len // page_size
        bytes_per_token_on_decode = dst_kv_item_len // page_size
        src_token_slot_offsets = (
            tokens_per_page * bytes_per_token_on_prefill + src_head_slice_offset
        )
        dst_token_slot_offsets = (
            tokens_per_page * bytes_per_token_on_decode + dst_head_slice_offset
        )

        def process_layer_tp_aware(src_layer_ptr, dst_layer_ptr):
            src_page_base_addrs = src_layer_ptr + prefill_page_indices * src_kv_item_len
            dst_page_base_addrs = dst_layer_ptr + decode_page_indices * dst_kv_item_len
            src_slice_addrs = src_page_base_addrs + src_token_slot_offsets
            dst_slice_addrs = dst_page_base_addrs + dst_token_slot_offsets

            src_addr_list = src_slice_addrs.reshape(-1).tolist()
            if not src_addr_list:
                # Nothing to transfer for this layer.
                return 0
            dst_addr_list = dst_slice_addrs.reshape(-1).tolist()
            total_slices = len(src_addr_list)
            length_list = [heads_bytes_per_token_to_send] * total_slices
            return self.engine.batch_transfer_sync(
                mooncake_session_id, src_addr_list, dst_addr_list, length_list
            )

        futures = []
        for i in range(layers_current_pp_stage):
            futures.append(
                executor.submit(process_layer_tp_aware, src_k_ptrs[i], dst_k_ptrs[i])
            )
        for i in range(layers_current_pp_stage):
            futures.append(
                executor.submit(process_layer_tp_aware, src_v_ptrs[i], dst_v_ptrs[i])
            )

        for future in concurrent.futures.as_completed(futures):
            status = future.result()
            if status != 0:
                for f in futures:
                    f.cancel()
                return status

        return 0

    def send_aux(
        self,
        req: TransferInfo,
        prefill_aux_index: int,
        dst_aux_ptrs: list[int],
    ):
        # TODO(shangming): Fix me when nvlink_transport of Mooncake is bug-free
        if (
            self.enable_custom_mem_pool and self.custom_mem_pool_type == "NVLINK"
        ) or envs.SGLANG_MOONCAKE_SEND_AUX_TCP.get():
            return self.send_aux_tcp(req, prefill_aux_index, dst_aux_ptrs)

        transfer_blocks = []
        prefill_aux_ptrs = self.kv_args.aux_data_ptrs
        prefill_aux_item_lens = self.kv_args.aux_item_lens

        for i, dst_aux_ptr in enumerate(dst_aux_ptrs):
            length = prefill_aux_item_lens[i]
            src_addr = prefill_aux_ptrs[i] + length * prefill_aux_index
            dst_addr = dst_aux_ptrs[i] + length * req.dst_aux_index
            transfer_blocks.append((src_addr, dst_addr, length))

        return self._transfer_data(req.mooncake_session_id, transfer_blocks)

    def send_aux_tcp(
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
                MooncakeKVManager.AUX_DATA_HEADER,
                str(room).encode("ascii"),
                str(buffer_index).encode("ascii"),
                str(aux_index).encode("ascii"),
                struct.pack(">I", len(data)),
                data,
            ]
        )

    def _handle_aux_data(self, msg: List[bytes]):
        """Handle AUX_DATA messages received by the decode thread."""
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

        logger.debug(
            f"Received AUX_DATA for bootstrap_room {room} with length:{len(data)}"
        )

    def _get_dsa_cache_transfer_skip_flags(
        self, info: Optional[KVArgsRegisterInfo]
    ) -> Tuple[bool, bool]:
        skip_kv = False
        skip_state = False
        if not self.is_hybrid_mla_backend:
            return skip_kv, skip_state

        if info is not None and self.attn_tp_size > info.dst_attn_tp_size:
            sub_rank = (self.kv_args.engine_rank % self.attn_tp_size) % (
                self.attn_tp_size // info.dst_attn_tp_size
            )
            if sub_rank != 0:
                skip_kv = True
                skip_state = True

        if (
            self.attn_cp_size > 1
            and self.attn_cp_rank != 0
            and not self.server_args.enable_dsa_cache_layer_split
        ):
            skip_state = True

        return skip_kv, skip_state

    def maybe_send_extra(
        self,
        req: TransferInfo,
        prefill_state_indices: List,
        executor: concurrent.futures.ThreadPoolExecutor,
        target_rank_registration_info: Optional[KVArgsRegisterInfo] = None,
    ):
        rc = 0
        state_types = getattr(self.kv_args, "state_types", [])
        for i, st in enumerate(state_types):
            indices = (
                prefill_state_indices[i] if i < len(prefill_state_indices) else None
            )
            if indices is None:
                continue
            src_data_ptrs = self.kv_args.state_data_ptrs[i]
            src_item_lens = self.kv_args.state_item_lens[i]
            src_dim_per_tensor = (
                self.kv_args.state_dim_per_tensor[i]
                if i < len(self.kv_args.state_dim_per_tensor)
                else []
            )
            src_conv_shard_groups = getattr(self.kv_args, "state_conv_shard_groups", [])
            src_conv_shard_groups = (
                src_conv_shard_groups[i] if i < len(src_conv_shard_groups) else []
            )
            if target_rank_registration_info is not None:
                dst_data_ptrs = (
                    target_rank_registration_info.dst_state_data_ptrs[i]
                    if i < len(target_rank_registration_info.dst_state_data_ptrs)
                    else []
                )
                dst_item_lens = (
                    target_rank_registration_info.dst_state_item_lens[i]
                    if i < len(target_rank_registration_info.dst_state_item_lens)
                    else []
                )
                dst_dim_per_tensor = (
                    target_rank_registration_info.dst_state_dim_per_tensor[i]
                    if i < len(target_rank_registration_info.dst_state_dim_per_tensor)
                    else []
                )
            else:
                dst_data_ptrs, dst_item_lens, dst_dim_per_tensor = [], [], []
            dst_indices = (
                req.dst_state_indices[i] if i < len(req.dst_state_indices) else []
            )

            if st == StateType.MAMBA:
                if (
                    target_rank_registration_info is not None
                    and self.attn_tp_size
                    != target_rank_registration_info.dst_attn_tp_size
                ):
                    rc = (
                        self._send_mamba_state_slice(
                            req,
                            indices,
                            src_data_ptrs,
                            src_item_lens,
                            src_dim_per_tensor,
                            dst_data_ptrs,
                            dst_indices,
                            dst_item_lens,
                            dst_dim_per_tensor,
                            target_rank_registration_info.dst_tp_rank,
                            target_rank_registration_info.dst_attn_tp_size,
                            src_conv_shard_groups,
                        )
                        or rc
                    )
                else:
                    rc = (
                        self._send_mamba_state(
                            req,
                            indices,
                            src_data_ptrs,
                            src_item_lens,
                            dst_data_ptrs,
                            dst_indices,
                        )
                        or rc
                    )
            elif st in (
                StateType.SWA,
                StateType.DSA,
                StateType.SWA_RING,
                StateType.C128_STATE,
            ):
                if (
                    target_rank_registration_info is not None
                    and not self.is_mla_backend
                    and self.attn_tp_size
                    != target_rank_registration_info.dst_attn_tp_size
                ):
                    raise RuntimeError(
                        f"PD Disaggregation does NOT support PD different TP sizes for non-MLA {st.upper()} hybrid models yet."
                    )
                src_indices = list(indices)
                dst_indices_local = list(dst_indices)
                if (
                    st == StateType.C128_STATE
                    and len(src_indices) == 0
                    and len(dst_indices_local) == 0
                ):
                    continue
                if len(src_indices) != len(dst_indices_local):
                    # These components are position- or request-indexed:
                    # truncating silently misaligns rows and corrupts KV.
                    # Paged SWA/DSA tolerate a 1-page drift -> keep the
                    # lenient truncation below.
                    if st in (
                        StateType.SWA_RING,
                        StateType.C128_STATE,
                    ):
                        raise RuntimeError(
                            f"{st.upper()} state index length mismatch: "
                            f"prefill={len(src_indices)}, dst={len(dst_indices_local)}"
                        )
                    logger.warning(
                        f"len(prefill_state_indices) = {len(src_indices)}, len(dst_state_indices) = {len(dst_indices_local)}"
                    )
                    if len(src_indices) > len(dst_indices_local):
                        src_indices = src_indices[: len(dst_indices_local)]
                    else:
                        dst_indices_local = dst_indices_local[: len(src_indices)]
                if st == StateType.DSPARK_HIDDEN and dynamic_dst:
                    row_chunks = dynamic_dst.get("row_chunks") or [
                        {"row_start": 0, "row_len": len(src_indices)}
                    ]
                    for row_chunk in row_chunks:
                        row_start = int(row_chunk.get("row_start", 0))
                        row_len = int(row_chunk.get("row_len", 0))
                        if row_len <= 0:
                            continue
                        row_end = row_start + row_len
                        if row_start < 0 or row_end > len(src_indices):
                            raise RuntimeError(
                                "Invalid DSpark hidden row chunk: "
                                f"room={req.room}, row_start={row_start}, "
                                f"row_len={row_len}, row_count={len(src_indices)}"
                            )
                        if "ptr" in row_chunk:
                            chunk_dst_data_ptrs = [int(row_chunk["ptr"])]
                            chunk_dst_indices = list(range(row_len))
                        else:
                            chunk_dst_data_ptrs = dst_data_ptrs
                            chunk_dst_indices = dst_indices_local[row_start:row_end]
                        rc = (
                            self._send_kvcache_generic(
                                mooncake_session_id=req.mooncake_session_id,
                                src_data_ptrs=src_data_ptrs,
                                dst_data_ptrs=chunk_dst_data_ptrs,
                                item_lens=src_item_lens,
                                prefill_data_indices=np.array(
                                    src_indices[row_start:row_end], dtype=np.int32
                                ),
                                dst_data_indices=np.array(
                                    chunk_dst_indices, dtype=np.int32
                                ),
                                executor=executor,
                                state_type=st,
                            )
                            or rc
                        )
                    continue
                rc = (
                    self._send_kvcache_generic(
                        mooncake_session_id=req.mooncake_session_id,
                        src_data_ptrs=src_data_ptrs,
                        dst_data_ptrs=dst_data_ptrs,
                        item_lens=src_item_lens,
                        prefill_data_indices=np.array(src_indices, dtype=np.int32),
                        dst_data_indices=np.array(dst_indices_local, dtype=np.int32),
                        executor=executor,
                        state_type=st,
                    )
                    or rc
                )
            elif st == StateType.MINIMAX_INDEX_K:
                # Equal-TP / PP=1 only. Sub-pools are compacted sparse-layer
                # lists, so PP>1 mis-slices and heterogeneous TP is unsupported.
                if self.pp_size is not None and self.pp_size > 1:
                    raise RuntimeError(
                        "PD disagg: PP>1 not supported for MiniMax sparse index yet."
                    )
                if (
                    target_rank_registration_info is not None
                    and self.attn_tp_size
                    != target_rank_registration_info.dst_attn_tp_size
                ):
                    raise RuntimeError(
                        "PD disagg: heterogeneous TP not supported for MiniMax "
                        "sparse index yet."
                    )
                src_indices = list(indices)
                dst_indices_local = list(dst_indices)
                if len(src_indices) > len(dst_indices_local):
                    src_indices = src_indices[: len(dst_indices_local)]
                elif len(src_indices) < len(dst_indices_local):
                    dst_indices_local = dst_indices_local[: len(src_indices)]
                rc = (
                    self._send_kvcache_generic(
                        mooncake_session_id=req.mooncake_session_id,
                        src_data_ptrs=src_data_ptrs,
                        dst_data_ptrs=dst_data_ptrs,
                        item_lens=src_item_lens,
                        prefill_data_indices=np.array(src_indices, dtype=np.int32),
                        dst_data_indices=np.array(dst_indices_local, dtype=np.int32),
                        executor=executor,
                        force_flat=True,
                    )
                    or rc
                )
        return rc

    def _dspark_hidden_state_index(self) -> Optional[int]:
        for idx, state_type in enumerate(getattr(self.kv_args, "state_types", [])):
            if state_type == StateType.DSPARK_HIDDEN:
                return idx
        return None

    def _has_dspark_hidden_state(self, state_indices: Optional[List]) -> bool:
        idx = self._dspark_hidden_state_index()
        if idx is None or not state_indices or idx >= len(state_indices):
            return False
        indices = state_indices[idx]
        return indices is not None and len(indices) > 0

    def _without_dspark_hidden_state(
        self, state_indices: Optional[List]
    ) -> Optional[List]:
        idx = self._dspark_hidden_state_index()
        if idx is None or not state_indices or idx >= len(state_indices):
            return state_indices
        ret = list(state_indices)
        ret[idx] = None
        return ret

    def _send_dspark_hidden_packet(
        self,
        req: TransferInfo,
        prefill_state_indices: List,
        packet_idx: int,
        executor: concurrent.futures.ThreadPoolExecutor,
    ) -> Tuple[int, bool]:
        state_idx = self._dspark_hidden_state_index()
        if state_idx is None or state_idx >= len(prefill_state_indices):
            return 0, True
        indices = prefill_state_indices[state_idx]
        if indices is None:
            return 0, True
        src_indices = np.asarray(indices, dtype=np.int32)
        dynamic_dst = (req.spec_metadata or {}).get("pp_slice", {}).get("dynamic_dst")
        if not dynamic_dst:
            return 0, True
        row_chunks = dynamic_dst.get("row_chunks") or []
        if packet_idx >= len(row_chunks):
            return 0, True

        row_chunk = row_chunks[packet_idx]
        row_start = int(row_chunk.get("row_start", 0))
        row_len = int(row_chunk.get("row_len", 0))
        if row_len <= 0:
            return 0, packet_idx + 1 >= len(row_chunks)
        row_end = row_start + row_len
        if row_start < 0 or row_end > len(src_indices):
            raise RuntimeError(
                "Invalid DSpark hidden packet: "
                f"room={req.room}, packet_idx={packet_idx}, "
                f"row_start={row_start}, row_len={row_len}, "
                f"row_count={len(src_indices)}"
            )

        src_data_ptrs = self.kv_args.state_data_ptrs[state_idx]
        dst_data_ptrs = [int(row_chunk.get("ptr", dynamic_dst.get("ptr", 0)))]
        item_lens = [int(dynamic_dst["item_len"])]
        rc = self._send_kvcache_generic(
            mooncake_session_id=req.mooncake_session_id,
            src_data_ptrs=src_data_ptrs,
            dst_data_ptrs=dst_data_ptrs,
            item_lens=item_lens,
            prefill_data_indices=src_indices[row_start:row_end],
            dst_data_indices=np.arange(row_len, dtype=np.int32),
            executor=executor,
            state_type=StateType.DSPARK_HIDDEN,
        )
        return rc, packet_idx + 1 >= len(row_chunks)

    def _send_mamba_state(
        self,
        req: TransferInfo,
        prefill_mamba_index: list,
        src_state_data_ptrs: list[int],
        src_state_item_lens: list[int],
        dst_state_data_ptrs: list[int],
        dst_mamba_index: list,
    ):
        assert len(prefill_mamba_index) == 1, "Mamba should have single state index"

        transfer_blocks = []
        for i, dst_state_ptr in enumerate(dst_state_data_ptrs):
            length = src_state_item_lens[i]
            src_addr = src_state_data_ptrs[i] + length * int(prefill_mamba_index[0])
            dst_addr = dst_state_ptr + length * int(dst_mamba_index[0])
            transfer_blocks.append((src_addr, dst_addr, length))

        return self._transfer_data(req.mooncake_session_id, transfer_blocks)

    def _send_mamba_state_slice(
        self,
        req: TransferInfo,
        prefill_mamba_index: list,
        src_state_data_ptrs: list[int],
        src_state_item_lens: list[int],
        src_state_dim_per_tensor: list[int],
        dst_state_data_ptrs: list[int],
        dst_mamba_index: list,
        dst_state_item_lens: list[int],
        dst_state_dim_per_tensor: list[int],
        dst_tp_rank: int,
        dst_attn_tp_size: int,
        src_state_conv_shard_groups: list = None,
    ):
        """Transfer Mamba states with TP slice support.

        Mamba state layout:
        - conv_state: [num_layers, size+1, conv_dim/tp, conv_kernel-1]
        - temporal_state: [num_layers, size+1, num_heads/tp, head_dim, state_size]

        The 3rd dimension is sliced by TP. When prefill and decode have different
        attn_tp_size, we slice the state accordingly. GDN conv_state is the
        concatenation [query | key | value] with each sub-block head-sharded
        independently, so on the scatter path it is sliced per sub-block via
        ``src_state_conv_shard_groups`` (see compute_mamba_state_slice_blocks).
        """
        logger.warning_once(
            "Using Mamba state slice transfer for different TP sizes between prefill and decode. "
            f"Prefill attn_tp_size={self.attn_tp_size}, Decode attn_tp_size={dst_attn_tp_size}. "
            "Performance may be affected."
        )
        assert len(prefill_mamba_index) == 1, "Mamba should have single state index"

        # If no dimension info available, fall back to regular transfer
        if not src_state_dim_per_tensor or not dst_state_dim_per_tensor:
            return self._send_mamba_state(
                req,
                prefill_mamba_index,
                src_state_data_ptrs,
                src_state_item_lens,
                dst_state_data_ptrs,
                dst_mamba_index,
            )

        local_tp_rank_in_group = self.kv_args.engine_rank % self.attn_tp_size
        dst_tp_rank_in_group = dst_tp_rank % dst_attn_tp_size

        transfer_blocks = []
        for i, dst_state_ptr in enumerate(dst_state_data_ptrs):
            src_item_len = src_state_item_lens[i]
            dst_item_len = dst_state_item_lens[i]
            src_dim = src_state_dim_per_tensor[i]
            dst_dim = dst_state_dim_per_tensor[i]

            # item_len = dim * trailing_dims_size, so trailing_dims_size = item_len / dim
            src_bytes_per_dim = src_item_len // src_dim
            dst_bytes_per_dim = dst_item_len // dst_dim

            conv_shard_groups = (
                src_state_conv_shard_groups[i]
                if src_state_conv_shard_groups and i < len(src_state_conv_shard_groups)
                else None
            )
            # One block for single-axis states; three (q/k/v) for GDN conv_state
            # on the scatter path.
            for (
                src_dim_start,
                dst_dim_start,
                num_dims_to_send,
            ) in compute_mamba_state_slice_blocks(
                src_dim=src_dim,
                dst_dim=dst_dim,
                src_attn_tp_size=self.attn_tp_size,
                dst_attn_tp_size=dst_attn_tp_size,
                dst_tp_rank_in_group=dst_tp_rank_in_group,
                local_tp_rank_in_group=local_tp_rank_in_group,
                conv_shard_groups=conv_shard_groups,
            ):
                src_dim_offset = src_dim_start * src_bytes_per_dim
                dst_dim_offset = dst_dim_start * dst_bytes_per_dim
                bytes_to_send = num_dims_to_send * src_bytes_per_dim

                src_addr = (
                    src_state_data_ptrs[i]
                    + src_item_len * int(prefill_mamba_index[0])
                    + src_dim_offset
                )
                dst_addr = (
                    dst_state_ptr
                    + dst_item_len * int(dst_mamba_index[0])
                    + dst_dim_offset
                )

                transfer_blocks.append((src_addr, dst_addr, bytes_to_send))

        return self._transfer_data(req.mooncake_session_id, transfer_blocks)

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
        self,
        queue: FastQueue,
        executor: concurrent.futures.ThreadPoolExecutor,
        staging_buffer=None,
        worker_index=0,
    ):
        staging_strategy = None
        if self.enable_trace:
            trace_set_thread_info(
                f"mooncake transfer worker {worker_index}",
                tp_rank=self.attn_tp_rank,
                dp_rank=self.attn_dp_rank,
            )

        while True:
            try:
                kv_chunk: TransferKVChunk = queue.get()
                if self.enable_trace:
                    kv_chunk.trace_ctx.rebuild_thread_context()
                    kv_chunk.trace_ctx.trace_slice_start(
                        MooncakeRequestStage.MOONCAKE_WORKER_SEND.stage_name,
                        MooncakeRequestStage.MOONCAKE_WORKER_SEND.level,
                    )
                current_status = self.request_status.get(kv_chunk.room)
                if current_status is None or current_status == KVPoll.Failed:
                    logger.debug(
                        f"Skipping chunk for room {kv_chunk.room} because it has already failed or been aborted"
                    )
                    if (
                        kv_chunk.is_last_chunk
                        and not kv_chunk.dspark_hidden_sent
                        and self._has_dspark_hidden_state(kv_chunk.state_indices)
                    ):
                        self.mark_dspark_hidden_done(
                            kv_chunk.room,
                            kv_chunk.state_indices,
                        )
                    if self.enable_trace:
                        kv_chunk.trace_ctx.trace_slice_end(
                            MooncakeRequestStage.MOONCAKE_WORKER_SEND.stage_name,
                            MooncakeRequestStage.MOONCAKE_WORKER_SEND.level,
                            thread_finish_flag=True,
                        )
                    continue
                if kv_chunk.source_event is not None:
                    kv_chunk.source_event.synchronize()
                    kv_chunk.source_event = None

                if (
                    self.enable_staging
                    and staging_strategy is None
                    and staging_buffer is not None
                ):
                    staging_strategy = self._try_create_staging_strategy(staging_buffer)
                reqs_to_be_processed = (
                    self.transfer_infos[kv_chunk.room].values()
                    if kv_chunk.room in self.transfer_infos
                    else []
                )
                polls = []
                dst_ranks_infos = []
                dspark_hidden_expected = 0
                dspark_hidden_done_count = 0
                # Unique id per prefill sender so decode's response set size matches expected_response_num.
                prefill_unique_rank = (
                    self.attn_tp_rank * (self.pp_size * self.attn_cp_size)
                    + self.pp_rank * self.attn_cp_size
                    + self.attn_cp_rank
                )
                # When staging transfer is not yet ready (watermark/allocation pending),
                # the chunk is re-enqueued and we break out of the req loop to retry later.
                staging_deferred = False
                dspark_hidden_deferred = False
                dspark_hidden_failed = False
                if (
                    kv_chunk.is_last_chunk
                    and not kv_chunk.dspark_hidden_sent
                    and kv_chunk.state_indices
                    and self._has_dspark_hidden_state(kv_chunk.state_indices)
                ):
                    for req in reqs_to_be_processed:
                        if req.is_dummy:
                            continue
                        target_rank_registration_info = self.decode_kv_args_table[
                            req.mooncake_session_id
                        ]
                        _, skip_state = self._get_dsa_cache_transfer_skip_flags(
                            target_rank_registration_info
                        )
                        if not skip_state:
                            dspark_hidden_expected += 1

                    for req in reqs_to_be_processed:
                        if req.is_dummy:
                            continue
                        with self.session_lock:
                            session_failed = (
                                req.mooncake_session_id in self.failed_sessions
                            )
                        if session_failed:
                            continue
                        target_rank_registration_info = self.decode_kv_args_table[
                            req.mooncake_session_id
                        ]
                        _, skip_state = self._get_dsa_cache_transfer_skip_flags(
                            target_rank_registration_info
                        )
                        if skip_state:
                            continue
                        ret, dspark_hidden_done = self._send_dspark_hidden_packet(
                            req,
                            kv_chunk.state_indices,
                            kv_chunk.dspark_hidden_packet_idx,
                            executor,
                        )
                        if ret != 0:
                            with self.session_lock:
                                self.session_failures[
                                    req.mooncake_session_id
                                ] += 1
                                if (
                                    self.session_failures[req.mooncake_session_id]
                                    >= 1
                                ):
                                    self.failed_sessions.add(
                                        req.mooncake_session_id
                                    )
                                    logger.error(
                                        f"Session {req.mooncake_session_id} failed."
                                    )
                            self.record_failure(
                                kv_chunk.room,
                                "Failed to send DSpark hidden packet "
                                f"{kv_chunk.dspark_hidden_packet_idx} of "
                                f"{kv_chunk.room} to "
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
                            self.mark_dspark_hidden_done(
                                kv_chunk.room,
                                kv_chunk.state_indices,
                            )
                            dspark_hidden_failed = True
                            break
                        if not dspark_hidden_done:
                            dspark_hidden_deferred = True
                            continue
                        dspark_hidden_done_count += 1

                    if dspark_hidden_failed:
                        continue
                    current_status = self.request_status.get(kv_chunk.room)
                    if (
                        dspark_hidden_expected > 0
                        and dspark_hidden_done_count == dspark_hidden_expected
                        and current_status is not None
                        and current_status != KVPoll.Failed
                    ):
                        kv_chunk.dspark_hidden_sent = True
                        self.mark_dspark_hidden_done(
                            kv_chunk.room,
                            kv_chunk.state_indices,
                        )
                    if dspark_hidden_deferred:
                        kv_chunk.dspark_hidden_packet_idx += 1
                        queue.put(kv_chunk)
                        continue

                for req in reqs_to_be_processed:
                    start_ts = time.perf_counter()
                    if not req.is_dummy:
                        # Early exit if the request has failed
                        with self.session_lock:
                            if req.mooncake_session_id in self.failed_sessions:
                                self.record_failure(
                                    kv_chunk.room,
                                    f"Decode instance could be dead, remote mooncake session {req.mooncake_session_id} is not alive",
                                )
                                self.update_status(kv_chunk.room, KVPoll.Failed)
                                self.sync_status_to_decode_endpoint(
                                    req.endpoint,
                                    req.dst_port,
                                    req.room,
                                    KVPoll.Failed,
                                    prefill_unique_rank,
                                )
                                if (
                                    kv_chunk.is_last_chunk
                                    and not kv_chunk.dspark_hidden_sent
                                    and self._has_dspark_hidden_state(
                                        kv_chunk.state_indices
                                    )
                                ):
                                    self.mark_dspark_hidden_done(
                                        kv_chunk.room,
                                        kv_chunk.state_indices,
                                    )
                                break

                        chunked_dst_kv_indice = req.dst_kv_indices[kv_chunk.index_slice]

                        # NOTE: This is temporarily a workaround to deal with the case where the prefill_kv_indices
                        # is mismatched with the dst_kv_indices when page size > 1, this should never happen.
                        if len(chunked_dst_kv_indice) < len(
                            kv_chunk.prefill_kv_indices
                        ):
                            logger.warning(
                                f"len(chunked_dst_kv_indice) = {len(chunked_dst_kv_indice)}, len(kv_chunk.prefill_kv_indices) = {len(kv_chunk.prefill_kv_indices)}"
                            )
                            kv_chunk.prefill_kv_indices = kv_chunk.prefill_kv_indices[
                                : len(chunked_dst_kv_indice)
                            ]

                        target_rank_registration_info: KVArgsRegisterInfo = (
                            self.decode_kv_args_table[req.mooncake_session_id]
                        )
                        skip_kv, skip_state = self._get_dsa_cache_transfer_skip_flags(
                            target_rank_registration_info
                        )
                        if kv_chunk.kv_sent:
                            ret = 0
                        elif len(kv_chunk.prefill_kv_indices) == 0 or skip_kv:
                            ret = 0
                        else:
                            if (
                                self.is_mla_backend
                                or self.is_hybrid_mla_backend
                                or self.attn_tp_size
                                == target_rank_registration_info.dst_attn_tp_size
                            ):
                                ret = self.send_kvcache(
                                    req.mooncake_session_id,
                                    kv_chunk.prefill_kv_indices,
                                    target_rank_registration_info.dst_kv_ptrs,
                                    chunked_dst_kv_indice,
                                    executor,
                                )
                            elif (
                                self.enable_staging
                                and staging_strategy is not None
                                and target_rank_registration_info.staging is not None
                            ):
                                ret, deferred = self._do_staging_transfer(
                                    staging_strategy,
                                    kv_chunk,
                                    req,
                                    target_rank_registration_info,
                                    chunked_dst_kv_indice,
                                    executor,
                                    queue,
                                    prefill_unique_rank,
                                )
                                if deferred:
                                    staging_deferred = True
                                    # Chunk re-enqueued; stop processing remaining reqs for this chunk
                                    break
                            else:
                                ret = self.send_kvcache_slice(
                                    req.mooncake_session_id,
                                    kv_chunk.prefill_kv_indices,
                                    target_rank_registration_info.dst_kv_ptrs,
                                    chunked_dst_kv_indice,
                                    target_rank_registration_info.dst_tp_rank,
                                    target_rank_registration_info.dst_attn_tp_size,
                                    target_rank_registration_info.dst_kv_item_len,
                                    executor,
                                )
                        if ret != 0:
                            with self.session_lock:
                                self.session_failures[req.mooncake_session_id] += 1
                                # Failures should never happen if the session is not dead, if the session fails once, mark it as failed
                                if self.session_failures[req.mooncake_session_id] >= 1:
                                    self.failed_sessions.add(req.mooncake_session_id)
                                    logger.error(
                                        f"Session {req.mooncake_session_id} failed."
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

                        if kv_chunk.is_last_chunk:
                            has_dspark_hidden = (
                                kv_chunk.state_indices
                                and not kv_chunk.dspark_hidden_sent
                                and not skip_state
                                and self._has_dspark_hidden_state(kv_chunk.state_indices)
                            )
                            if has_dspark_hidden:
                                dspark_hidden_expected += 1
                                ret, dspark_hidden_done = (
                                    self._send_dspark_hidden_packet(
                                        req,
                                        kv_chunk.state_indices,
                                        kv_chunk.dspark_hidden_packet_idx,
                                        executor,
                                    )
                                )
                                if ret != 0:
                                    with self.session_lock:
                                        self.session_failures[
                                            req.mooncake_session_id
                                        ] += 1
                                        if (
                                            self.session_failures[
                                                req.mooncake_session_id
                                            ]
                                            >= 1
                                        ):
                                            self.failed_sessions.add(
                                                req.mooncake_session_id
                                            )
                                            logger.error(
                                                f"Session {req.mooncake_session_id} failed."
                                            )
                                    self.record_failure(
                                        kv_chunk.room,
                                        "Failed to send DSpark hidden packet "
                                        f"{kv_chunk.dspark_hidden_packet_idx} of "
                                        f"{kv_chunk.room} to "
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
                                    self.mark_dspark_hidden_done(
                                        kv_chunk.room,
                                        kv_chunk.state_indices,
                                    )
                                    break
                                if not dspark_hidden_done:
                                    dspark_hidden_deferred = True
                                    continue
                                dspark_hidden_done_count += 1

                            if kv_chunk.state_indices and not skip_state:
                                self.maybe_send_extra(
                                    req,
                                    self._without_dspark_hidden_state(
                                        kv_chunk.state_indices
                                    ),
                                    executor,
                                    target_rank_registration_info,
                                )

                            # Only the last chunk we need to send the aux data
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
                    else:
                        # Dummy request means the decode instance is not used, so its status can be marked as success directly
                        # Dummy request does not need to sync status to decode endpoint
                        if kv_chunk.is_last_chunk and req.room in self.request_status:
                            self.update_status(req.room, KVPoll.Success)

                    if self.enable_trace:
                        mooncake_trace_slice(
                            kv_chunk.trace_ctx,
                            MooncakeRequestStage.MOONCAKE_WORKER_SEND_SESSION,
                            start_ts,
                        )

                if self.enable_trace:
                    kv_chunk.trace_ctx.trace_slice_end(
                        MooncakeRequestStage.MOONCAKE_WORKER_SEND.stage_name,
                        MooncakeRequestStage.MOONCAKE_WORKER_SEND.level,
                        thread_finish_flag=True,
                    )

                if staging_deferred:
                    continue
                current_status = self.request_status.get(kv_chunk.room)
                if current_status is not None and current_status != KVPoll.Failed:
                    kv_chunk.kv_sent = True
                if (
                    kv_chunk.is_last_chunk
                    and not kv_chunk.dspark_hidden_sent
                    and dspark_hidden_expected > 0
                    and dspark_hidden_done_count == dspark_hidden_expected
                    and current_status is not None
                    and current_status != KVPoll.Failed
                ):
                    self.mark_dspark_hidden_done(
                        kv_chunk.room,
                        kv_chunk.state_indices,
                    )
                if dspark_hidden_deferred:
                    kv_chunk.dspark_hidden_packet_idx += 1
                    queue.put(kv_chunk)
                    continue

                current_status = self.request_status.get(kv_chunk.room)
                if kv_chunk.room not in self.request_status or current_status == KVPoll.Success:
                    if kv_chunk.room in self.transfer_infos:
                        self.transfer_infos.pop(kv_chunk.room)
                    self.req_to_decode_prefix_len.pop(kv_chunk.room, None)

            except Exception as e:
                # NOTE(shangming): Remove this when we make sure the transfer thread is bug-free
                raise RuntimeError(
                    f"Transfer thread failed because of {e}. Prefill instance with bootstrap_port={self.bootstrap_port} is dead."
                )

    def start_prefill_thread(self):
        def bootstrap_thread():
            """This thread recvs pre-alloc notification from the decode engine"""
            # KVPoll.Bootstrapping -> KVPoll.WaitingForInput
            while True:
                waiting_req_bytes = self.server_socket.recv_multipart()
                room = waiting_req_bytes[0].decode("ascii")
                # Staging: decode reports consumption watermark back to prefill
                if room == "WATERMARK":
                    from sglang.srt.disaggregation.common.staging_handler import (
                        handle_watermark_msg,
                    )

                    handle_watermark_msg(self._staging_ctx, waiting_req_bytes)
                    continue
                # Staging: decode replies with allocated staging offset
                if room == "STAGING_RSP":
                    from sglang.srt.disaggregation.common.staging_handler import (
                        handle_staging_rsp,
                    )

                    handle_staging_rsp(waiting_req_bytes, self.transfer_infos)
                    continue
                # Decode-side abort notification: mark room as failed and ACK
                if room == "ABORT":
                    room_to_be_aborted = int(waiting_req_bytes[1].decode("ascii"))
                    decode_ip = waiting_req_bytes[2].decode("ascii")
                    decode_port = int(waiting_req_bytes[3].decode("ascii"))
                    # No need to abort the room if it has already succeeded
                    if (
                        room_to_be_aborted in self.request_status
                        and self.check_status(room_to_be_aborted) != KVPoll.Success
                    ):
                        self.update_status(room_to_be_aborted, KVPoll.Failed)
                        logger.debug(
                            f"Received abort notification for room {room_to_be_aborted}, "
                            f"marked as Failed"
                        )
                    else:
                        logger.debug(
                            f"Received abort notification for room {room_to_be_aborted}, "
                            f"ignoring (already completed or unknown)"
                        )
                    # Send ACK back to decode endpoint
                    try:
                        na = NetworkAddress(decode_ip, decode_port)
                        self._connect(na.to_tcp(), is_ipv6=na.is_ipv6).send_multipart(
                            [
                                b"ABORT_ACK",
                                str(room_to_be_aborted).encode("ascii"),
                            ]
                        )
                        logger.debug(
                            f"Sent ABORT_ACK for room {room_to_be_aborted} to "
                            f"{decode_ip}:{decode_port}"
                        )
                    except Exception as e:
                        logger.debug(
                            f"Failed to send ABORT_ACK for room {room_to_be_aborted}: {e}"
                        )
                    continue
                mooncake_session_id = waiting_req_bytes[3].decode("ascii")
                if room == "None":
                    self.decode_kv_args_table[mooncake_session_id] = (
                        KVArgsRegisterInfo.from_zmq(waiting_req_bytes)
                    )
                    with self.session_lock:
                        if mooncake_session_id in self.failed_sessions:
                            self.failed_sessions.remove(mooncake_session_id)
                        if mooncake_session_id in self.session_failures:
                            del self.session_failures[mooncake_session_id]
                    logger.debug(
                        f"Register KVArgs from {mooncake_session_id} successfully"
                    )
                    continue
                else:
                    required_dst_info_num = int(waiting_req_bytes[7].decode("ascii"))
                    room = int(room)
                    if room not in self.transfer_infos:
                        self.transfer_infos[room] = {}

                    transfer_info = TransferInfo.from_zmq(waiting_req_bytes)
                    self.transfer_infos[room][mooncake_session_id] = transfer_info
                    # NOTE: after bootstrapping we can mark the req as waiting for input
                    if len(self.transfer_infos[room]) == required_dst_info_num:
                        self.resolve_kv_replica_factor(self.transfer_infos[room])
                        self.req_to_decode_prefix_len[room] = next(
                            (
                                info.decode_prefix_len
                                for info in self.transfer_infos[room].values()
                                if info.decode_prefix_len is not None
                            ),
                            0,
                        )
                        dspark_meta = next(
                            (
                                info.spec_metadata
                                for info in self.transfer_infos[room].values()
                                if info.spec_metadata
                                and info.spec_metadata.get("dspark_hidden")
                            ),
                            None,
                        )
                        if dspark_meta:
                            self.req_to_dspark_hidden_meta[room] = dspark_meta
                        self.update_status(room, KVPoll.WaitingForInput)

        threading.Thread(target=bootstrap_thread).start()

    def start_decode_thread(self):
        def decode_thread():
            while True:
                msg = self.server_socket.recv_multipart()
                if msg[0] == MooncakeKVManager.AUX_DATA_HEADER:
                    self._handle_aux_data(msg)
                    continue

                # Staging: prefill notifies a chunk written to staging buffer
                if msg[0] == b"CHUNK_READY":
                    room = int(msg[1].decode("ascii"))
                    chunk_idx = int(msg[2].decode("ascii"))
                    page_start = int(msg[3].decode("ascii"))
                    num_pages = int(msg[4].decode("ascii"))
                    session_id = msg[5].decode("ascii")
                    handler = self._staging_handler
                    assert (
                        handler is not None
                    ), "CHUNK_READY received before staging handler initialized"
                    handler.handle_chunk_arrived(
                        room,
                        chunk_idx,
                        page_start,
                        num_pages,
                        session_id,
                        self._chunk_writer_counts,
                    )
                    continue

                # Staging: prefill pre-requests staging allocation before forward
                if msg[0] == b"STAGING_REQ":
                    self._handle_staging_req(msg)
                    continue

                # Prefill acknowledges abort notification
                if msg[0] == b"ABORT_ACK":
                    # TODO(shangming): use this info to implement the deferred release mechanism if needed
                    ack_aborted_room = int(msg[1].decode("ascii"))
                    logger.debug(f"Received ABORT_ACK for room {ack_aborted_room}")
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
                        logger.debug(
                            "DSPARK_HIDDEN_DECODE_STATUS_ACK room=%s "
                            "prefill_rank=%s arrived=%s expected=%s "
                            "status=%s",
                            bootstrap_room,
                            prefill_rank,
                            arrived_response_num,
                            expected_response_num,
                            status,
                        )
                        if arrived_response_num == expected_response_num:
                            if self.enable_staging:
                                handler = self._staging_handler
                                if handler.is_staging_room(bootstrap_room):
                                    handler.submit_last_scatter_async(bootstrap_room)
                                self._chunk_writer_counts.pop(bootstrap_room, None)
                            self.update_status(bootstrap_room, KVPoll.Success)
                    else:
                        logger.debug(
                            "DSPARK_HIDDEN_DECODE_STATUS_ACK_IGNORED room=%s "
                            "prefill_rank=%s status=%s reason=missing_request_status",
                            bootstrap_room,
                            prefill_rank,
                            status,
                        )
                elif status == KVPoll.Failed:
                    self.record_failure(
                        bootstrap_room,
                        "Failed to get kvcache from prefill instance, it might be dead",
                    )
                    self.update_status(bootstrap_room, status)

        threading.Thread(target=decode_thread).start()
        self._start_heartbeat_checker_thread()

    def add_transfer_request(
        self,
        bootstrap_room: int,
        kv_indices: npt.NDArray[np.int32],
        index_slice: slice,
        is_last_chunk: bool,
        aux_index: Optional[int] = None,
        state_indices: Optional[List] = None,
        trace_ctx: Optional[Union[TraceReqContext, TraceNullContext]] = None,
        source_event=None,
    ):
        assert self.disaggregation_mode == DisaggregationMode.PREFILL
        assert not is_last_chunk or (is_last_chunk and aux_index is not None)

        if (
            bootstrap_room not in self.request_status
            or self.check_status(bootstrap_room) == KVPoll.Failed
        ):
            logger.debug(
                "Request with bootstrap_room=%s already failed", bootstrap_room
            )
            return

        if bootstrap_room not in self.transfer_infos:
            # This means that the current rank is a dummy rank for this request,
            # and it has already been marked as success, so there is no need to
            # add further chunks into the transfer queue.
            return

        # NOTE(shangming): sharding according to the dst_infos to make sure
        # requests with the same dst_sessions will be added into the same
        # queue, which enables early abort with failed sessions.
        dst_infos = self.transfer_infos[bootstrap_room].keys()
        session_port_sum = sum(int(session.rsplit(":", 1)[1]) for session in dst_infos)
        shard_idx = session_port_sum % len(self.transfer_queues)

        if trace_ctx is None:
            trace_ctx = TraceNullContext()

        self.transfer_queues[shard_idx].put(
            TransferKVChunk(
                room=bootstrap_room,
                prefill_kv_indices=kv_indices,
                index_slice=index_slice,
                is_last_chunk=is_last_chunk,
                prefill_aux_index=aux_index,
                state_indices=state_indices,
                source_event=source_event,
                trace_ctx=trace_ctx,
            )
        )

    def get_session_id(self):
        return self.engine.get_session_id()

    def _on_heartbeat_success(self, bootstrap_addr: str):
        current_rooms = self.addr_to_rooms_tracker[bootstrap_addr].copy()
        for bootstrap_room in current_rooms:
            # Remove KVPoll.Success requests from the tracker
            if bootstrap_room not in self.request_status:
                self.addr_to_rooms_tracker[bootstrap_addr].discard(bootstrap_room)

    def _run_one_probe_pass(self) -> None:
        with self.session_lock:
            snapshot = list(self.failed_sessions)
        for session_id in snapshot:
            send_probe = getattr(self.engine, "send_probe", None)
            if send_probe is None:
                rc = -1
            else:
                try:
                    rc = send_probe(session_id)
                except Exception as e:
                    logger.warning("send_probe(%s) raised: %s", session_id, e)
                    continue
            if rc == 0:
                with self.session_lock:
                    was_blacklisted = session_id in self.failed_sessions
                    self.failed_sessions.discard(session_id)
                    self.session_failures.pop(session_id, None)
                if was_blacklisted:
                    logger.info(
                        "Session %s recovered via probe; un-blacklisted",
                        session_id,
                    )
                    FAILED_SESSION_RECOVERIES.inc()
            else:
                logger.debug("Probe still failing for %s (rc=%d)", session_id, rc)

    def _failed_session_probe_loop(self) -> None:
        logger.info(
            "Starting failed-session probe loop (interval=%.1fs)",
            self.failed_session_probe_interval,
        )
        while not self._failed_session_probe_shutdown.wait(
            self.failed_session_probe_interval
        ):
            self._run_one_probe_pass()


class MooncakeKVSender(CommonKVSender):

    def __init__(
        self,
        mgr: MooncakeKVManager,
        bootstrap_addr: str,
        bootstrap_room: int,
        dest_tp_ranks: List[int],
        pp_rank: int,
        req_has_disagg_prefill_dp_rank: bool = False,
    ):
        super().__init__(
            mgr,
            bootstrap_addr,
            bootstrap_room,
            dest_tp_ranks,
            pp_rank,
            req_has_disagg_prefill_dp_rank,
        )
        self.conclude_state = None
        self.init_time = time.time()
        self._source_event = None
        self._init_trace_ctx()

    def set_source_event(self, source_event) -> None:
        self._source_event = source_event

    @mooncake_trace_func(MooncakeRequestStage.MOONCAKE_SEND)
    def send(
        self,
        kv_indices: npt.NDArray[np.int32],
        state_indices: Optional[List] = None,
    ):
        kv_indices, index_slice, is_last_chunk, should_skip = (
            self._prepare_send_indices(kv_indices, state_indices)
        )
        if should_skip:
            self._source_event = None
            return

        if not is_last_chunk:
            self.kv_mgr.add_transfer_request(
                self.bootstrap_room,
                kv_indices,
                index_slice,
                False,
                trace_ctx=self.trace_ctx.copy_for_thread(),
            )
        else:
            source_event = self._source_event
            self._source_event = None
            self.kv_mgr.add_transfer_request(
                self.bootstrap_room,
                kv_indices,
                index_slice,
                True,
                aux_index=self.aux_index,
                state_indices=state_indices,
                source_event=source_event,
                trace_ctx=self.trace_ctx.copy_for_thread(),
            )
        self._record_transfer_indices(kv_indices, state_indices)

    def poll(self) -> KVPoll:
        if self.conclude_state is None:
            status = self.kv_mgr.check_status(self.bootstrap_room)
            if status in (KVPoll.Success, KVPoll.Failed):
                self.conclude_state = status
                self.trace_ctx.trace_req_finish()
            elif status == KVPoll.Bootstrapping:
                timeout_result = self._check_bootstrap_timeout()
                if timeout_result is not None:
                    return timeout_result

            return status
        else:
            return self.conclude_state

    def failure_exception(self):
        # Explicitly set the status to failure since this request has failed in another rank
        if self.conclude_state is None:
            self.conclude_state = KVPoll.Failed

        self.clear()

        with self.kv_mgr.failure_lock:
            failure_reason = self.kv_mgr.failure_records.pop(self.bootstrap_room, None)
        is_propagated = failure_reason is None
        if is_propagated:
            failure_reason = "Failed due to an unknown reason from another rank"
        raise KVTransferError(
            self.bootstrap_room, failure_reason, is_from_another_rank=is_propagated
        )

    def _init_trace_ctx(self):
        if self.kv_mgr.enable_trace:
            self.trace_ctx = TraceReqContext(
                rid=str(hex(self.bootstrap_room)),
                bootstrap_room=self.bootstrap_room,
                role="Sender",
                module_name="mooncake",
            )
            if not self.trace_ctx.tracing_enable:
                self.trace_ctx = TraceNullContext()
        else:
            self.trace_ctx = TraceNullContext()

        self.trace_ctx.trace_req_start()

    def abort(self):
        super().abort()
        self.trace_ctx.abort(abort_info={"reason": "Aborted"})
        self.trace_ctx.trace_req_finish()


class MooncakeKVReceiver(CommonKVReceiver):
    def __init__(
        self,
        mgr: MooncakeKVManager,
        bootstrap_addr: str,
        bootstrap_room: Optional[int] = None,
    ):
        self.session_id = mgr.get_session_id()
        self.init_time = None
        super().__init__(mgr, bootstrap_addr, bootstrap_room)

    def _register_kv_args(self):
        for bootstrap_info in self.bootstrap_infos:
            packed_kv_data_ptrs = b"".join(
                struct.pack("Q", ptr) for ptr in self.kv_mgr.kv_args.kv_data_ptrs
            )
            packed_aux_data_ptrs = b"".join(
                struct.pack("Q", ptr) for ptr in self.kv_mgr.kv_args.aux_data_ptrs
            )
            packed_state_data_ptrs = pack_int_lists(
                self.kv_mgr.kv_args.state_data_ptrs, "Q"
            )
            packed_state_item_lens = pack_int_lists(
                self.kv_mgr.kv_args.state_item_lens, "I"
            )
            packed_state_dim_per_tensor = pack_int_lists(
                getattr(self.kv_mgr.kv_args, "state_dim_per_tensor", []) or [], "I"
            )
            # Note(shangming): No need to add pp rank here since decode pp size should be equal to prefill pp size or 1
            tp_rank = self.kv_mgr.kv_args.engine_rank
            kv_item_len = self.kv_mgr.kv_args.kv_item_lens[0]
            dst_tp_rank = str(tp_rank).encode("ascii")
            dst_attn_tp_size = str(self.kv_mgr.attn_tp_size).encode("ascii")
            dst_kv_item_len = str(kv_item_len).encode("ascii")
            if (
                self.kv_mgr.enable_staging
                and self.kv_mgr._staging_ctx.allocator is not None
            ):
                _alloc = self.kv_mgr._staging_ctx.allocator
                packed_staging_base_ptr = struct.pack("Q", _alloc.get_base_ptr())
                staging_total_size_str = str(_alloc.get_total_size()).encode("ascii")
            else:
                packed_staging_base_ptr = b""
                staging_total_size_str = b""

            sock, lock = self._connect_to_bootstrap_server(bootstrap_info)
            with lock:
                sock.send_multipart(
                    [
                        "None".encode("ascii"),
                        self.kv_mgr.local_ip.encode("ascii"),
                        str(self.kv_mgr.rank_port).encode("ascii"),
                        self.session_id.encode("ascii"),
                        packed_kv_data_ptrs,
                        packed_aux_data_ptrs,
                        packed_state_data_ptrs,
                        dst_tp_rank,
                        dst_attn_tp_size,
                        dst_kv_item_len,
                        packed_state_item_lens,
                        packed_state_dim_per_tensor,
                        packed_staging_base_ptr,
                        staging_total_size_str,
                    ]
                )

    def send_metadata(
        self,
        kv_indices: npt.NDArray[np.int32],
        aux_index: Optional[int] = None,
        state_indices: Optional[List] = None,
        decode_prefix_len: Optional[int] = None,
        spec_metadata: Optional[dict] = None,
    ):
        if self.bootstrap_infos is None:
            self.kv_mgr.record_failure(
                self.bootstrap_room,
                f"Could not fetch prefill parallel info from bootstrap_addr: {self.bootstrap_addr}",
            )
            self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Failed)
            return

        if (
            self.kv_mgr.enable_staging
            and self.kv_mgr._staging_ctx.allocator is not None
        ):
            self.chunk_staging_infos = []
            self.kv_mgr.register_staging_room_bootstrap(
                self.bootstrap_room, self.bootstrap_infos, self
            )

        for bootstrap_info in self.bootstrap_infos:
            sock, lock = self._connect_to_bootstrap_server(bootstrap_info)
            is_dummy = bootstrap_info["is_dummy"]
            local_state_indices = state_indices
            local_spec_metadata = spec_metadata
            if spec_metadata and spec_metadata.get("pp_slices"):
                pp_rank_value = bootstrap_info.get(
                    "target_pp_rank", bootstrap_info.get("pp_rank")
                )
                if pp_rank_value is None:
                    raise RuntimeError(
                        "DSpark PP hidden metadata requires target_pp_rank in "
                        f"bootstrap_info, got keys={sorted(bootstrap_info.keys())}"
                    )
                pp_rank = int(pp_rank_value)
                pp_slice = spec_metadata["pp_slices"].get(str(pp_rank))
                if pp_slice is None:
                    raise RuntimeError(
                        "DSpark PP hidden metadata is missing slice for "
                        f"target_pp_rank={pp_rank}, available_pp_slices="
                        f"{sorted(spec_metadata['pp_slices'].keys())}"
                    )
                local_spec_metadata = {
                    **spec_metadata,
                    "target_pp_rank": int(pp_rank),
                    "pp_slice": pp_slice,
                }
                local_state_indices = list(
                    state_indices
                    if state_indices is not None
                    else [None] * len(self.kv_mgr.kv_args.state_types)
                )
                for idx, state_type in enumerate(self.kv_mgr.kv_args.state_types):
                    if state_type == StateType.DSPARK_HIDDEN:
                        local_state_indices[idx] = pp_slice.get("dst_indices", [])
                        break

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
                            pack_int_lists(local_state_indices, "i")
                            if not is_dummy and local_state_indices
                            else b""
                        ),
                        str(self.required_dst_info_num).encode("ascii"),
                        str(decode_prefix_len or 0).encode("ascii"),
                        (
                            json.dumps(local_spec_metadata).encode("utf-8")
                            if local_spec_metadata
                            else b""
                        ),
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
        if self.conclude_state is None:
            self.conclude_state = KVPoll.Failed

        self.clear()

        with self.kv_mgr.failure_lock:
            failure_reason = self.kv_mgr.failure_records.pop(self.bootstrap_room, None)
        is_propagated = failure_reason is None
        if is_propagated:
            failure_reason = "Failed due to an unknown reason from another rank"
        raise KVTransferError(
            self.bootstrap_room, failure_reason, is_from_another_rank=is_propagated
        )


class MooncakeKVBootstrapServer(CommonKVBootstrapServer):
    pass
