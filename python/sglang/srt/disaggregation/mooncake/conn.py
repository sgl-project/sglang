from __future__ import annotations

import concurrent.futures
import dataclasses
import logging
import os
import struct
import threading
import time
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

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
    filter_kv_indices_for_cp_rank,
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

# Sentinels for `add_transfer_request` fields that don't apply to the
# draft (MTP/EAGLE NEXTN) KV chunk: draft is not part of the main
# layer-pipeline group sequence and never carries a `total_layer_groups`
# value (the trailing aux finalizer in `send()` carries that). Stored on
# `TransferKVChunk` for observability / log readability; never consumed
# for routing or watermark decisions.
DRAFT_LAYER_GROUP_ID = -1
DRAFT_TOTAL_LAYER_GROUPS_SENTINEL = -1


@dataclasses.dataclass
class _RoomLayerPipelineProgress:
    """Per-room watermark state for layer-pipelined KV transfer.

    Lifetime: created on first chunk arrival for a room (gated by
    `layer_pipeline_progress_lock`), removed by
    `_clear_layer_pipeline_progress` on terminal status (Success /
    Failed) or on the three failure-exit paths in the transfer worker.

    The watermark closes — and the room moves to KVPoll.Success —
    iff every dst rank has reported both its full chunk count AND its
    aux finalizer.
    """

    # Count of layer-group chunks acknowledged delivered, keyed by
    # (mooncake_session_id, dst_pp_rank). A chunk is counted when the
    # transfer_worker calls `_record_chunk_done` post-RDMA. Required to
    # reach `total_chunks_expected` for every dst before Success fires.
    chunks_done_per_dst: Dict[Tuple[str, int], int] = dataclasses.field(
        default_factory=dict
    )
    # Aux finalizer status per dst rank. Populated by `_record_aux_sent`.
    # Each dst must report exactly once before Success can fire — both
    # CP0 (real aux RDMA) and non-CP0 ranks (status-only, skip_aux_rdma)
    # call this so the per-rank watermark slot closes uniformly.
    aux_results_per_dst: Dict[Tuple[str, int], int] = dataclasses.field(
        default_factory=dict
    )
    # Set by the trailing aux finalizer's `total_chunks_in_request`
    # field on the sender side. The watermark sync loop compares each
    # dst's `chunks_done_per_dst` count against this value. None until
    # the finalizer has been enqueued — chunk arrivals that race the
    # finalizer get deferred at sync-check time.
    total_chunks_expected: Optional[int] = None
    # How many dst ranks must report before Success fires. Set once at
    # bootstrap from the request's dst-rank set (CP, TP, PP fan-out).
    required_dst_info_num: Optional[int] = None
    # One-shot guard so `_maybe_sync_success_locked` fires
    # `KVPoll.Success` exactly once per room, even if late-arriving
    # chunks call the sync path again post-close.
    success_synced: bool = False


@dataclasses.dataclass
class _LayerPipelineRequestDispatch:
    room: int
    sender: "MooncakeKVSender"
    page_indices: npt.NDArray[np.int32]
    index_slice: slice
    state_indices: Optional[npt.NDArray[np.int32]] = None


def split_layer_groups(
    num_layers: int, layer_group_size: int
) -> List[Tuple[int, int]]:
    if num_layers <= 0:
        return []
    if layer_group_size <= 0 or layer_group_size >= num_layers:
        return [(0, num_layers)]
    return [
        (start, min(start + layer_group_size, num_layers))
        for start in range(0, num_layers, layer_group_size)
    ]


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
    layer_pipeline_enabled: bool = False
    layer_group_size: int = 0
    kv_dtype: str = "auto"
    # Decode-side local main KV layer count (logical layers, excludes
    # draft tail). ``None`` when the dst side did not advertise it
    # (legacy / no-draft) — receivers fall back to inferring from total
    # length, which is unsafe when the two sides disagree on draft
    # presence; see ``get_state_ptrs_with_pp``.
    dst_num_main_kv_layers: Optional[int] = None
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
            layer_pipeline_enabled=(
                msg[12].decode("ascii") == "1" if len(msg) > 12 else False
            ),
            layer_group_size=(
                int(msg[13].decode("ascii"))
                if len(msg) > 13 and len(msg[13]) > 0
                else 0
            ),
            kv_dtype=msg[14].decode("ascii") if len(msg) > 14 else "auto",
            dst_num_main_kv_layers=(
                int(msg[17].decode("ascii"))
                if len(msg) > 17 and len(msg[17]) > 0
                else None
            ),
            # Note: always put the staging field at the final
            staging=StagingRegisterInfo.from_zmq_fields(msg, 15),
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
            # Reserved for the staging-path GPU copy; unused on the current
            # RDMA path (mooncake transfer is CPU-side).
            try:
                import torch as _torch
                self.transfer_stream = (
                    _torch.cuda.Stream() if _torch.cuda.is_available() else None
                )
            except ImportError:
                self.transfer_stream = None
            # LP-only state. Kept off the LP-disabled path so the runtime
            # matches the pre-LP build: no watermark dict/lock, no hook
            # timing counters, no LP metrics buffers. Helpers that touch
            # these fields no-op when they are absent.
            if self.layer_pipeline_enabled:
                # Per-room watermark state for layer-pipeline chunks. Tracks
                # per-dst-rank chunk completion + aux send so Success fires
                # regardless of completion order. Cleared on Success/Failed.
                self.layer_pipeline_progress: Dict[int, _RoomLayerPipelineProgress] = {}
                self.layer_pipeline_progress_lock = threading.Lock()
                self._hook_timing_total_ns: int = 0
                self._hook_timing_fire_count: int = 0
                self._hook_timing_dispatch_count: int = 0
                self._HOOK_TIMING_LOG_EVERY_FIRES: int = 200
                self._hook_instrumentation_logged_init: bool = False
                # When True, hook skips add_transfer_request and sender's
                # hook_expected branch short-circuits to Success without RDMA.
                # Prefill-side diagnostic only; decode side will fail.
                self._lp_hook_noop_fake_success: bool = (
                    envs.SGLANG_DISAGG_LAYER_PIPELINE_HOOK_NOOP.get()
                )
                if self._lp_hook_noop_fake_success:
                    logger.warning(
                        "[layer-pipeline] HOOK_NOOP fake-success ENABLED. "
                        "Prefill batches will short-circuit to Success without "
                        "any RDMA submit; decode side WILL fail. NEVER set in "
                        "production."
                    )
                # Periodic LP-chunk metrics: bumped by transfer worker on each
                # successful chunk, snapshot+reset by pop_layer_pipeline_metrics.
                self._lp_metrics_lock = threading.Lock()
                self._lp_chunks_total: int = 0
                self._lp_chunks_periodic: int = 0
                self._lp_chunk_ms_samples: List[float] = []
                # Bound buffer memory if the scheduler snapshot loop stalls;
                # the Counter is unaffected.
                self._LP_SAMPLE_BUFFER_CAP: int = 4096
            # SGLANG_DISAGG_KV_HASH_VERIFY: accumulate per-room prefill
            # src page_indices (dedup on tobytes), emit one log line per
            # room when watermark closes to Success. NEVER enable in
            # production — adds D2H memcpy per layer.
            self._kv_hash_lock = threading.Lock()
            self._kv_hash_prefill_chunks: Dict[int, List[Any]] = {}
            self._kv_hash_prefill_seen: Dict[int, set] = {}
            self._kv_hash_prefill_layer_ranges: Dict[int, List[Tuple[int, int]]] = {}
            self._kv_hash_layout_logged_prefill: bool = False
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
            # SGLANG_DISAGG_KV_HASH_VERIFY: stash dst page indices on
            # send_metadata; emit decode-side hash on first Success poll.
            self._kv_hash_lock = threading.Lock()
            self._kv_hash_decode_indices: Dict[int, Any] = {}
            self._kv_hash_decode_emitted: set = set()
            self._kv_hash_layout_logged_decode: bool = False
            self.start_decode_thread()

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

    def _room_supports_layer_pipeline(self, room: int) -> bool:
        if not self.layer_pipeline_enabled:
            return False
        for req in self.transfer_infos.get(room, {}).values():
            if req.is_dummy:
                continue
            target_info = self.decode_kv_args_table.get(req.mooncake_session_id)
            if target_info is None:
                return False
            if not target_info.layer_pipeline_enabled:
                return False
            if (
                target_info.layer_group_size > 0
                and target_info.layer_group_size != self.layer_group_size
            ):
                return False
            if target_info.kv_dtype != self.server_args.kv_cache_dtype:
                return False
        return True

    def _maybe_warn_lp_handshake_mismatch(
        self,
        mooncake_session_id: str,
        target_info: "KVArgsRegisterInfo",
    ) -> None:
        """Fail-loud (once per session) when prefill and decode disagree
        on layer-pipeline settings that silently disable LP.

        `_room_supports_layer_pipeline` covers the same checks but only
        in the hot path — and returning False there just falls back to
        the legacy fan-out without surfacing the misconfiguration.
        Operators tuning `--disagg-layer-group-size` would never see
        that their setting is being silently overridden.
        """
        if not self.layer_pipeline_enabled:
            # Prefill has LP off — nothing to warn about even if decode
            # has it on; the prefill side is authoritative.
            return
        if not getattr(target_info, "layer_pipeline_enabled", False):
            logger.warning(
                "Layer-pipeline handshake mismatch with decode session "
                "%s: prefill has --enable-disagg-layer-pipeline but the "
                "decode side does NOT. LP is disabled for transfers to "
                "this session; KV transfer falls back to the legacy "
                "single-chunk fan-out.",
                mooncake_session_id,
            )
            return
        dec_group_size = getattr(target_info, "layer_group_size", 0)
        if dec_group_size > 0 and dec_group_size != self.layer_group_size:
            logger.warning(
                "Layer-pipeline group-size mismatch with decode session "
                "%s: prefill --disagg-layer-group-size=%d vs decode "
                "--disagg-layer-group-size=%d. LP is disabled for "
                "transfers to this session; both sides must agree on "
                "the group size for LP to engage.",
                mooncake_session_id,
                self.layer_group_size,
                dec_group_size,
            )
            return
        dec_kv_dtype = getattr(target_info, "kv_dtype", None)
        if dec_kv_dtype and dec_kv_dtype != self.server_args.kv_cache_dtype:
            logger.warning(
                "KV dtype mismatch with decode session %s: prefill "
                "kv_cache_dtype=%r vs decode kv_cache_dtype=%r. LP is "
                "disabled for transfers to this session.",
                mooncake_session_id,
                self.server_args.kv_cache_dtype,
                dec_kv_dtype,
            )

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
                dst_num_main=target_info.dst_num_main_kv_layers,
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
        force_flat: bool = False,
        dst_num_main: Optional[int] = None,
    ) -> int:
        """
        Generic KV cache transfer supporting both MHA and MLA architectures.
        This method is used by both send_kvcache (full pool) and maybe_send_extra.

        ``force_flat`` uses the MLA-style flat (single-buffer-per-layer) layout
        even on a non-MLA backend, for K-only state buffers (e.g. MiniMax sparse
        index) whose per-layer list must not be half-split into K/V.
        ``dst_num_main`` is the decode-side advertised main KV layer
        count (from registration); needed so the KV helpers can slice
        dst's main and draft regions independently when src has a
        draft tail (DSA + EAGLE + PP > 1). ``None`` for legacy decode
        or non-draft deployments — see helper docstrings for the
        fallback behavior.
        """
        # Group by indices for optimization
        prefill_kv_blocks, dst_kv_blocks = group_concurrent_contiguous(
            prefill_data_indices, dst_data_indices
        )

        layers_params = None

        num_main_local = getattr(
            self.kv_args, "prefill_num_main_kv_layers", None
        )
        # Decode pp size should be equal to prefill pp size or 1
        if self.is_mla_backend or force_flat:
            src_kv_ptrs, dst_kv_ptrs, layers_current_pp_stage = (
                self.get_mla_kv_ptrs_with_pp(
                    src_data_ptrs,
                    dst_data_ptrs,
                    num_main_local=num_main_local,
                    dst_num_main=dst_num_main,
                )
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
                self.get_mha_kv_ptrs_with_pp(
                    src_data_ptrs,
                    dst_data_ptrs,
                    num_main_local=num_main_local,
                    dst_num_main=dst_num_main,
                )
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

        _b1_trace_on = (
            envs.SGLANG_DISAGG_KV_HASH_VERIFY.get()
            and not getattr(self, "_kv_hash_rdma_trace_logged", False)
            and layers_params
            and len(self.kv_args.kv_data_ptrs) > 0
            and int(layers_params[-1][0]) == int(self.kv_args.kv_data_ptrs[-1])
        )

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
            ret = 0
        else:
            # Combining all layers' params in one batch transfer is more efficient
            # compared to using multiple threads
            ret = process_layers(layers_params)

        if _b1_trace_on:
            try:
                self._kv_hash_rdma_trace_logged = True
                last_src_ptr, last_dst_ptr, last_item_len = layers_params[-1]
                first_blk = (
                    set_transfer_blocks(last_src_ptr, last_dst_ptr, last_item_len)[0]
                    if prefill_kv_blocks and dst_kv_blocks
                    else (0, 0, 0)
                )
                logger.info(
                    "[KV_HASH_DBG] RDMA_TRACE_ONE_SHOT side=prefill "
                    "submit_layer_count=%d last_layer_src_ptr=0x%x "
                    "last_layer_dst_ptr=0x%x item_len=%d "
                    "first_block_src=0x%x first_block_dst=0x%x first_block_len=%d "
                    "prefill_first_page=%s decode_first_page=%s rdma_ret=%d",
                    len(layers_params),
                    int(last_src_ptr),
                    int(last_dst_ptr),
                    int(last_item_len),
                    int(first_blk[0]),
                    int(first_blk[1]),
                    int(first_blk[2]),
                    int(prefill_kv_blocks[0][0]) if prefill_kv_blocks else -1,
                    int(dst_kv_blocks[0][0]) if dst_kv_blocks else -1,
                    int(ret),
                )
            except Exception as exc:
                logger.warning("[KV_HASH_DBG] RDMA_TRACE log failed: %r", exc)
        return ret

    def _send_kvcache_layer_group(
        self,
        mooncake_session_id: str,
        src_data_ptrs: list[int],
        dst_data_ptrs: list[int],
        item_lens: list[int],
        prefill_data_indices: npt.NDArray[np.int32],
        dst_data_indices: npt.NDArray[np.int32],
        executor: concurrent.futures.ThreadPoolExecutor,
        layer_range: Optional[Tuple[int, int]],
        dst_num_main: Optional[int] = None,
    ) -> int:
        if layer_range is None:
            return self._send_kvcache_generic(
                mooncake_session_id=mooncake_session_id,
                src_data_ptrs=src_data_ptrs,
                dst_data_ptrs=dst_data_ptrs,
                item_lens=item_lens,
                prefill_data_indices=prefill_data_indices,
                dst_data_indices=dst_data_indices,
                executor=executor,
                dst_num_main=dst_num_main,
            )

        layer_start, layer_end = layer_range
        if layer_start < 0 or layer_end <= layer_start:
            logger.error("Invalid layer transfer range: %s", layer_range)
            return -1

        num_main_local = getattr(
            self.kv_args, "prefill_num_main_kv_layers", None
        )
        if self.is_mla_backend:
            src_kv_ptrs, dst_kv_ptrs, layers_current_pp_stage = (
                self.get_mla_kv_ptrs_with_pp(
                    src_data_ptrs,
                    dst_data_ptrs,
                    num_main_local=num_main_local,
                    dst_num_main=dst_num_main,
                )
            )
            if layer_end > layers_current_pp_stage:
                logger.error(
                    "Layer transfer range %s exceeds current PP stage layers %s",
                    layer_range,
                    layers_current_pp_stage,
                )
                return -1
            src_data_ptrs = src_kv_ptrs[layer_start:layer_end]
            dst_data_ptrs = dst_kv_ptrs[layer_start:layer_end]
            item_lens = item_lens[layer_start:layer_end]
        else:
            src_k_ptrs, src_v_ptrs, dst_k_ptrs, dst_v_ptrs, layers_current_pp_stage = (
                self.get_mha_kv_ptrs_with_pp(
                    src_data_ptrs,
                    dst_data_ptrs,
                    num_main_local=num_main_local,
                    dst_num_main=dst_num_main,
                )
            )
            if layer_end > layers_current_pp_stage:
                logger.error(
                    "Layer transfer range %s exceeds current PP stage layers %s",
                    layer_range,
                    layers_current_pp_stage,
                )
                return -1
            # `item_lens` is expected to be the [K_0..K_{N-1}, V_0..V_{N-1}]
            # concatenated layout produced by MHATokenToKVPool. The slicing below
            # depends on this contiguous K-then-V layout — guard against silent
            # corruption if the layout ever changes (issue #8).
            assert len(item_lens) == 2 * layers_current_pp_stage, (
                f"MHA item_lens layout assumption violated: "
                f"len(item_lens)={len(item_lens)} expected={2 * layers_current_pp_stage}"
            )
            src_data_ptrs = (
                src_k_ptrs[layer_start:layer_end]
                + src_v_ptrs[layer_start:layer_end]
            )
            dst_data_ptrs = (
                dst_k_ptrs[layer_start:layer_end]
                + dst_v_ptrs[layer_start:layer_end]
            )
            item_lens = (
                item_lens[layer_start:layer_end]
                + item_lens[
                    layers_current_pp_stage + layer_start : layers_current_pp_stage
                    + layer_end
                ]
            )

        return self._send_kvcache_generic(
            mooncake_session_id=mooncake_session_id,
            src_data_ptrs=src_data_ptrs,
            dst_data_ptrs=dst_data_ptrs,
            item_lens=item_lens,
            prefill_data_indices=prefill_data_indices,
            dst_data_indices=dst_data_indices,
            executor=executor,
            dst_num_main=dst_num_main,
        )

    def send_kvcache(
        self,
        mooncake_session_id: str,
        prefill_kv_indices: npt.NDArray[np.int32],
        dst_kv_ptrs: list[int],
        dst_kv_indices: npt.NDArray[np.int32],
        executor: concurrent.futures.ThreadPoolExecutor,
        layer_range: Optional[Tuple[int, int]] = None,
        dst_num_main: Optional[int] = None,
    ):
        return self._send_kvcache_layer_group(
            mooncake_session_id=mooncake_session_id,
            src_data_ptrs=self.kv_args.kv_data_ptrs,
            dst_data_ptrs=dst_kv_ptrs,
            item_lens=self.kv_args.kv_item_lens,
            prefill_data_indices=prefill_kv_indices,
            dst_data_indices=dst_kv_indices,
            executor=executor,
            layer_range=layer_range,
            dst_num_main=dst_num_main,
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
        layer_range: Optional[Tuple[int, int]] = None,
        dst_num_main: Optional[int] = None,
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
            src_head_start_offset = (
                dst_tp_rank_in_group * dst_heads_per_rank
            ) % src_heads_per_rank
            num_heads_to_send = dst_heads_per_rank
            dst_head_start_offset = 0

        src_k_ptrs, src_v_ptrs, dst_k_ptrs, dst_v_ptrs, layers_current_pp_stage = (
            self.get_mha_kv_ptrs_with_pp(
                self.kv_args.kv_data_ptrs,
                dst_kv_ptrs,
                num_main_local=getattr(
                    self.kv_args, "prefill_num_main_kv_layers", None
                ),
                dst_num_main=dst_num_main,
            )
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

        if layer_range is None:
            layer_start, layer_end = 0, layers_current_pp_stage
        else:
            layer_start, layer_end = layer_range
            if layer_start < 0 or layer_end > layers_current_pp_stage:
                logger.error(
                    "Layer transfer range %s exceeds current PP stage layers %s",
                    layer_range,
                    layers_current_pp_stage,
                )
                return -1

        futures = []
        for i in range(layer_start, layer_end):
            futures.append(
                executor.submit(process_layer_tp_aware, src_k_ptrs[i], dst_k_ptrs[i])
            )
        for i in range(layer_start, layer_end):
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

    def maybe_send_extra(
        self,
        req: TransferInfo,
        prefill_state_indices: List,
        executor: concurrent.futures.ThreadPoolExecutor,
        target_rank_registration_info: Optional[KVArgsRegisterInfo] = None,
        layer_range: Optional[Tuple[int, int]] = None,
        dst_state_index_slice: Optional[slice] = None,
    ):
        rc = 0
        state_types = getattr(self.kv_args, "state_types", [])
        if (
            layer_range is not None
            and not (
                isinstance(prefill_state_indices, list)
                and len(prefill_state_indices) == len(state_types)
                and (
                    len(prefill_state_indices) == 0
                    or prefill_state_indices[0] is None
                    or isinstance(prefill_state_indices[0], (list, np.ndarray))
                )
            )
        ):
            prefill_state_indices = [
                prefill_state_indices if st == StateType.DSA else None
                for st in state_types
            ]
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
            if dst_state_index_slice is not None and st == StateType.DSA:
                dst_indices = dst_indices[dst_state_index_slice]

            if st == StateType.MAMBA:
                if layer_range is not None:
                    logger.warning(
                        "maybe_send_extra: layer_range=%s ignored for mamba state",
                        layer_range,
                    )
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
            elif st in (StateType.SWA, StateType.DSA, StateType.SWA_RING):
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
                if len(src_indices) != len(dst_indices_local):
                    # SWA_RING is positional: truncating silently misaligns rows
                    # and corrupts KV, so fail loud. DSA state under layer
                    # pipeline is also positional with the per-layer-group
                    # transfer slices; a large mismatch means the KV and state
                    # index coordinate systems diverged and truncation would
                    # silently corrupt the indexer state.
                    if st == StateType.SWA_RING or (
                        st == StateType.DSA
                        and getattr(self, "layer_pipeline_enabled", False)
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
                transfer_src_ptrs = src_data_ptrs
                transfer_dst_ptrs = dst_data_ptrs
                transfer_item_lens = src_item_lens
                if layer_range is not None and st is StateType.DSA:
                    # Only DSA wires LP state shipping in production today
                    # (see `_draft_layer_range` + the hook's state dispatch),
                    # and only DSA's state layout is validated for the
                    # main-vs-draft PP split that `get_state_ptrs_with_pp`
                    # performs (one ptr per main KV layer + optional draft
                    # tail). SWA / SWA_RING state has a different layout
                    # (e.g. unified-KV ring slots), so keep them on the
                    # legacy contiguous path until they grow LP coverage
                    # with their own alignment tests.
                    #
                    # Align dst to the prefill PP slice BEFORE the local
                    # layer_range slice. `layer_range` is in local
                    # coordinates (the LP hook subtracts
                    # `prefill_start_layer` before computing the group
                    # boundaries); without this alignment, a prefill
                    # PP rank > 0 against a decode full-model dst would
                    # silently pick dst layers [0, len(src)) instead of
                    # [prefill_start_layer, prefill_start_layer+len(src)).
                    #
                    # `num_main_local` lets the helper detect a draft
                    # tail in src (`setup_state_kv_args` appends draft
                    # state ptrs to main) so the helper slices main and
                    # draft portions independently. For a PP rank with
                    # `prefill_num_main_kv_layers` < `len(src)`, the
                    # draft layer is `dst[dst_num_main]`, NOT
                    # `dst[start_layer + local_main_count]` (which is
                    # the next stage's main layer in the dst list).
                    num_main_local = getattr(
                        self.kv_args, "prefill_num_main_kv_layers", None
                    )
                    # Decode advertises its own main count on registration
                    # so the helper can fail loud on cross-side draft
                    # disagreement (e.g. prefill has draft but decode
                    # doesn't, or vice versa). ``None`` ⇒ legacy decode
                    # didn't ship the field — fall back to length-based
                    # inference inside the helper.
                    dst_num_main = (
                        target_rank_registration_info.dst_num_main_kv_layers
                        if target_rank_registration_info is not None
                        else None
                    )
                    (
                        transfer_src_ptrs,
                        transfer_dst_ptrs,
                        transfer_item_lens,
                    ) = self.get_state_ptrs_with_pp(
                        transfer_src_ptrs,
                        transfer_dst_ptrs,
                        transfer_item_lens,
                        num_main_local=num_main_local,
                        dst_num_main=dst_num_main,
                    )
                    layer_start, layer_end = layer_range
                    if layer_start < 0 or layer_end <= layer_start:
                        logger.error(
                            "maybe_send_extra: invalid layer_range=%s", layer_range
                        )
                        return -1
                    if layer_end > len(transfer_src_ptrs):
                        logger.error(
                            "maybe_send_extra: layer_end=%s exceeds state ptr count=%s",
                            layer_end,
                            len(transfer_src_ptrs),
                        )
                        return -1
                    transfer_src_ptrs = transfer_src_ptrs[layer_start:layer_end]
                    transfer_dst_ptrs = transfer_dst_ptrs[layer_start:layer_end]
                    transfer_item_lens = transfer_item_lens[layer_start:layer_end]
                rc = (
                    self._send_kvcache_generic(
                        mooncake_session_id=req.mooncake_session_id,
                        src_data_ptrs=transfer_src_ptrs,
                        dst_data_ptrs=transfer_dst_ptrs,
                        item_lens=transfer_item_lens,
                        prefill_data_indices=np.array(src_indices, dtype=np.int32),
                        dst_data_indices=np.array(dst_indices_local, dtype=np.int32),
                        executor=executor,
                        dst_num_main=(
                            target_rank_registration_info.dst_num_main_kv_layers
                            if target_rank_registration_info is not None
                            else None
                        ),
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
    ):
        """Transfer Mamba states with TP slice support.

        Mamba state layout:
        - conv_state: [num_layers, size+1, conv_dim/tp, conv_kernel-1]
        - temporal_state: [num_layers, size+1, num_heads/tp, head_dim, state_size]

        The 3rd dimension is sliced by TP. When prefill and decode have different
        attn_tp_size, we need to slice the state accordingly.
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

            if self.attn_tp_size > dst_attn_tp_size:
                # Multiple prefill ranks send to 1 decode rank
                src_dim_start = 0
                num_dims_to_send = src_dim
                writers_per_decode = self.attn_tp_size // dst_attn_tp_size
                local_writer_idx = local_tp_rank_in_group % writers_per_decode
                dst_dim_start = local_writer_idx * src_dim
            else:
                # 1 prefill rank sends to multiple decode ranks
                src_dim_start = (dst_tp_rank_in_group * dst_dim) % src_dim
                num_dims_to_send = dst_dim
                dst_dim_start = 0

            src_dim_offset = src_dim_start * src_bytes_per_dim
            dst_dim_offset = dst_dim_start * dst_bytes_per_dim
            bytes_to_send = num_dims_to_send * src_bytes_per_dim

            src_addr = (
                src_state_data_ptrs[i]
                + src_item_len * int(prefill_mamba_index[0])
                + src_dim_offset
            )
            dst_addr = (
                dst_state_ptr + dst_item_len * int(dst_mamba_index[0]) + dst_dim_offset
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

    # ------------------------------------------------------------------
    # Per-(room, dst-rank) chunk + aux completion tracking. The helpers
    # below own all reads/writes of `self.layer_pipeline_progress`
    # under a single lock.
    # ------------------------------------------------------------------

    def _record_chunk_done(
        self,
        room: int,
        dst: Tuple[str, int],
        required_dst_info_num: int,
        prefill_unique_rank: int,
    ) -> None:
        """Bump the per-dst chunk counter after a successful KV transfer.

        May trigger Success sync if this completion brings the room over
        the watermark line.
        """
        with self.layer_pipeline_progress_lock:
            prog = self.layer_pipeline_progress.setdefault(
                room, _RoomLayerPipelineProgress()
            )
            prog.required_dst_info_num = required_dst_info_num
            prog.chunks_done_per_dst[dst] = prog.chunks_done_per_dst.get(dst, 0) + 1
            sync = self._maybe_sync_success_locked(room, prog, prefill_unique_rank)
        # ZMQ notify must happen OUTSIDE the lock.
        if sync is not None:
            self._dispatch_sync_outside_lock(room, prefill_unique_rank, *sync)

    def _record_aux_sent(
        self,
        room: int,
        dst: Tuple[str, int],
        total_chunks_in_request: Optional[int],
        required_dst_info_num: int,
        aux_ret: int,
        prefill_unique_rank: int,
    ) -> None:
        """Record that aux/state RDMA finished for one (room, dst_rank) pair.

        `aux_ret == 0` = success; non-zero drives final status to Failed
        once all dst ranks have reported.
        """
        with self.layer_pipeline_progress_lock:
            prog = self.layer_pipeline_progress.setdefault(
                room, _RoomLayerPipelineProgress()
            )
            prog.required_dst_info_num = required_dst_info_num
            if total_chunks_in_request is not None:
                # All dst ranks carry the same total from the sender.
                prog.total_chunks_expected = total_chunks_in_request
            prog.aux_results_per_dst[dst] = aux_ret
            sync = self._maybe_sync_success_locked(room, prog, prefill_unique_rank)
        # ZMQ notify must happen OUTSIDE the lock.
        if sync is not None:
            self._dispatch_sync_outside_lock(room, prefill_unique_rank, *sync)

    def _maybe_sync_success_locked(
        self,
        room: int,
        prog: _RoomLayerPipelineProgress,
        prefill_unique_rank: int,
    ) -> Optional[Tuple[int, List[Tuple[str, int]]]]:
        """Decide whether the watermark is satisfied; return the snapshot the
        caller should dispatch OUTSIDE the lock, or `None` if no sync fires.

        MUST be called with `layer_pipeline_progress_lock` held. Idempotent
        via `prog.success_synced` — at most one non-None snapshot per room.
        The actual ZMQ notify must run outside the lock so a blocked decode
        endpoint cannot stall progress on other rooms.
        """
        if prog.success_synced:
            return None
        if (
            prog.total_chunks_expected is None
            or prog.required_dst_info_num is None
        ):
            return None
        if len(prog.aux_results_per_dst) < prog.required_dst_info_num:
            return None
        # All required dst ranks have aux'd. Verify chunk counts too.
        for dst, _aux_ret in prog.aux_results_per_dst.items():
            if (
                prog.chunks_done_per_dst.get(dst, 0)
                < prog.total_chunks_expected
            ):
                return None
        any_aux_failed = any(r != 0 for r in prog.aux_results_per_dst.values())
        final_status = KVPoll.Failed if any_aux_failed else KVPoll.Success
        prog.success_synced = True
        # Snapshot endpoints inside the lock; caller dispatches outside.
        return final_status, list(prog.aux_results_per_dst.keys())

    def _dispatch_sync_outside_lock(
        self,
        room: int,
        prefill_unique_rank: int,
        final_status: int,
        endpoints: List[Tuple[str, int]],
    ) -> None:
        """Apply final status + notify all decode endpoints.

        MUST be called with `layer_pipeline_progress_lock` NOT held.
        `endpoints` is already a snapshot taken under the lock.
        """
        if envs.SGLANG_DISAGG_KV_HASH_VERIFY.get():
            if final_status == KVPoll.Success:
                self._kv_hash_emit_prefill(room)
            else:
                self._kv_hash_drop(room)
        self.update_status(room, final_status)
        for endpoint, dst_port in endpoints:
            self.sync_status_to_decode_endpoint(
                endpoint,
                dst_port,
                room,
                final_status,
                prefill_unique_rank,
            )
        self._clear_layer_pipeline_progress(room)

    def _clear_layer_pipeline_progress(self, room: int) -> None:
        """Drop a room's watermark state. Safe to call on rooms that never
        registered any progress (no-op). Also a no-op when LP is disabled
        and the watermark dict was never allocated.
        """
        progress = getattr(self, "layer_pipeline_progress", None)
        if progress is not None:
            with self.layer_pipeline_progress_lock:
                progress.pop(room, None)
        if envs.SGLANG_DISAGG_KV_HASH_VERIFY.get():
            self._kv_hash_drop(room)

    def _record_layer_group_metric(self, enqueue_ns: int) -> None:
        """Record one successful LP-chunk RDMA completion for metrics.

        Non-LP chunks carry `enqueue_ns=0` and are silently skipped.
        Protected by `_lp_metrics_lock` to serialize with snapshot+reset.
        """
        if enqueue_ns <= 0:
            return
        metrics_lock = getattr(self, "_lp_metrics_lock", None)
        if metrics_lock is None:
            return
        elapsed_ms = (time.monotonic_ns() - enqueue_ns) / 1_000_000.0
        with metrics_lock:
            self._lp_chunks_total += 1
            self._lp_chunks_periodic += 1
            if len(self._lp_chunk_ms_samples) < self._LP_SAMPLE_BUFFER_CAP:
                self._lp_chunk_ms_samples.append(elapsed_ms)

    def _log_lp_kv_byte_hash(self, kv_chunk: "TransferKVChunk") -> None:
        """Env-gated CRC32 fingerprint of sample bytes per layer.

        Called after `event.synchronize()` and before mooncake RDMA submit,
        only when `SGLANG_DISAGG_LAYER_PIPELINE_HASH_LOG=1`. MLA only.
        NEVER enable in production — D2H memcpy breaks zero-copy.
        """
        try:
            import ctypes
            import zlib

            if not self.is_mla_backend or kv_chunk.layer_range is None:
                return
            cudart = self._get_cudart_lib()
            if cudart is None:
                return
            layer_start, layer_end = kv_chunk.layer_range
            src_kv_ptrs, _, layers_pp = self.get_mla_kv_ptrs_with_pp(
                self.kv_args.kv_data_ptrs, []
            )
            item_lens = self.kv_args.kv_item_lens
            page_indices = kv_chunk.prefill_kv_indices
            if len(page_indices) == 0 or layer_end > layers_pp:
                return
            # Sample 3 pages each from FRONT + MIDDLE + BACK (deduped).
            n = len(page_indices)
            sample_idx = sorted(set([
                0,
                min(1, n - 1),
                min(2, n - 1),
                n // 2,
                min(n // 2 + 1, n - 1),
                min(n // 2 + 2, n - 1),
                n - 3 if n >= 3 else 0,
                n - 2 if n >= 2 else 0,
                n - 1,
            ]))
            sampled_pages = [int(page_indices[i]) for i in sample_idx]
            sample_bytes = 32
            per_layer = []
            for layer_id in range(layer_start, layer_end):
                if layer_id >= len(src_kv_ptrs) or layer_id >= len(item_lens):
                    continue
                src_ptr = int(src_kv_ptrs[layer_id])
                item_len = int(item_lens[layer_id])
                if item_len <= 0:
                    continue
                page_hashes = []
                for pidx in sampled_pages:
                    addr = src_ptr + int(pidx) * item_len
                    buf = (ctypes.c_uint8 * sample_bytes)()
                    err = cudart.cudaMemcpy(
                        ctypes.cast(buf, ctypes.c_void_p),
                        ctypes.c_void_p(addr),
                        ctypes.c_size_t(sample_bytes),
                        ctypes.c_int(2),  # cudaMemcpyDeviceToHost
                    )
                    if err != 0:
                        page_hashes.append(f"err{err}")
                    else:
                        page_hashes.append(f"{zlib.crc32(bytes(buf)):08x}")
                per_layer.append(f"L{layer_id}:[{','.join(page_hashes)}]")
            logger.info(
                "[LP_HASH] room=%s layer_range=(%s,%s) pages_sample=%s "
                "sample_bytes=%d hashes=%s",
                kv_chunk.room,
                layer_start,
                layer_end,
                sampled_pages,
                sample_bytes,
                " ".join(per_layer),
            )
        except Exception as exc:
            logger.warning("[LP_HASH] failed: %r", exc)

    # ------------------------------------------------------------------
    # KV_HASH_VERIFY (SGLANG_DISAGG_KV_HASH_VERIFY): per-request CRC32
    # sampling for offline prefill-vs-decode diff. MLA only.
    # ------------------------------------------------------------------

    _KV_HASH_SAMPLE_BYTES = 32

    @staticmethod
    def _kv_hash_sample_indices(n: int) -> List[int]:
        if n <= 0:
            return []
        cand = [
            0,
            min(1, n - 1),
            min(2, n - 1),
            n // 2,
            min(n // 2 + 1, n - 1),
            min(n // 2 + 2, n - 1),
            n - 3 if n >= 3 else 0,
            n - 2 if n >= 2 else 0,
            n - 1,
        ]
        return sorted(set(cand))

    def _kv_hash_record_prefill_chunk(
        self,
        room: int,
        prefill_kv_indices,
        layer_range: Optional[Tuple[int, int]] = None,
    ) -> None:
        """Accumulate per-room src page_indices for prefill-side hash.

        Same chunk arrives once per dst-session; dedup on tobytes so each
        chunk's pages count exactly once. `layer_range` is appended on
        every call (outside the dedup) so single-chunk LP with multiple
        layer-groups still reports every group fire.
        """
        try:
            if prefill_kv_indices is None or len(prefill_kv_indices) == 0:
                return
            key = bytes(prefill_kv_indices.tobytes())
            with self._kv_hash_lock:
                if layer_range is not None:
                    self._kv_hash_prefill_layer_ranges.setdefault(
                        room, []
                    ).append(tuple(layer_range))
                seen = self._kv_hash_prefill_seen.setdefault(room, set())
                if key in seen:
                    return
                seen.add(key)
                self._kv_hash_prefill_chunks.setdefault(room, []).append(
                    prefill_kv_indices
                )
        except Exception as exc:
            logger.warning("[KV_HASH_REQ] prefill record failed: %r", exc)

    def _kv_hash_emit_prefill(self, room: int) -> None:
        """Pop accumulated chunks and emit the per-room prefill hash line.

        Called only when the watermark closes to Success.
        """
        try:
            import numpy as _np

            with self._kv_hash_lock:
                chunks = self._kv_hash_prefill_chunks.pop(room, None)
                self._kv_hash_prefill_seen.pop(room, None)
                layer_ranges = self._kv_hash_prefill_layer_ranges.pop(room, None)
            if not chunks:
                return
            full_indices = _np.concatenate(chunks)
            if layer_ranges:
                logger.info(
                    "[KV_HASH_DBG] side=prefill rank=%d room=%s "
                    "layer_ranges_seen=%s",
                    getattr(self.kv_args, "engine_rank", -1),
                    room,
                    layer_ranges,
                )
            self._log_kv_hash_req("prefill", room, full_indices)
        except Exception as exc:
            logger.warning("[KV_HASH_REQ] prefill emit failed: %r", exc)

    def _kv_hash_drop(self, room: int) -> None:
        """Drop prefill-side hash accumulator for a room (failure / cleanup)."""
        with self._kv_hash_lock:
            self._kv_hash_prefill_chunks.pop(room, None)
            self._kv_hash_prefill_seen.pop(room, None)
            self._kv_hash_prefill_layer_ranges.pop(room, None)

    def _kv_hash_save_decode_indices(self, room: int, kv_indices) -> None:
        """Stash dst page indices for decode-side hash emit on Success poll."""
        try:
            if kv_indices is None or len(kv_indices) == 0:
                return
            with self._kv_hash_lock:
                # First non-dummy registration wins; all bootstrap infos for
                # a single send_metadata call carry the SAME dst indices.
                if room in self._kv_hash_decode_indices:
                    return
                self._kv_hash_decode_indices[room] = kv_indices
        except Exception as exc:
            logger.warning("[KV_HASH_REQ] decode save failed: %r", exc)

    def _kv_hash_emit_decode_once(self, room: int) -> None:
        """Emit decode-side hash once per room. Idempotent via `_emitted` set."""
        try:
            with self._kv_hash_lock:
                if room in self._kv_hash_decode_emitted:
                    return
                indices = self._kv_hash_decode_indices.pop(room, None)
                if indices is None:
                    return
                self._kv_hash_decode_emitted.add(room)
            self._log_kv_hash_req("decode", room, indices)
        except Exception as exc:
            logger.warning("[KV_HASH_REQ] decode emit failed: %r", exc)

    def _kv_hash_clear_decode(self, room: int) -> None:
        """Drop decode-side hash state on receiver clear()."""
        with self._kv_hash_lock:
            self._kv_hash_decode_indices.pop(room, None)
            self._kv_hash_decode_emitted.discard(room)

    def _log_kv_hash_req(self, side: str, room: int, page_indices) -> None:
        """Sample CRC32 of 32 bytes at 9 positions in `page_indices` for
        every layer in this PP-stage's `kv_data_ptrs`, log one line.

        MLA backend only; MHA double-pointer layout not implemented.
        """
        try:
            import ctypes
            import zlib

            if not self.is_mla_backend:
                logger.info(
                    "[KV_HASH_REQ] side=%s room=%s skipped (non-MLA backend)",
                    side, room,
                )
                return
            cudart = self._get_cudart_lib()
            if cudart is None:
                return
            n = len(page_indices)
            sample_idx = self._kv_hash_sample_indices(n)
            if not sample_idx:
                return
            sampled_pages = [int(page_indices[i]) for i in sample_idx]
            kv_ptrs = self.kv_args.kv_data_ptrs
            kv_data_lens = getattr(self.kv_args, "kv_data_lens", []) or []
            item_lens = self.kv_args.kv_item_lens
            num_layers = len(kv_ptrs)
            sample_bytes = self._KV_HASH_SAMPLE_BYTES
            rank = getattr(self.kv_args, "engine_rank", -1)
            log_layout = False
            with self._kv_hash_lock:
                if side == "prefill" and not self._kv_hash_layout_logged_prefill:
                    self._kv_hash_layout_logged_prefill = True
                    log_layout = True
                elif side == "decode" and not self._kv_hash_layout_logged_decode:
                    self._kv_hash_layout_logged_decode = True
                    log_layout = True
            if log_layout:
                layout_summary = []
                for L in range(num_layers):
                    p = int(kv_ptrs[L]) if L < len(kv_ptrs) else 0
                    dlen = int(kv_data_lens[L]) if L < len(kv_data_lens) else 0
                    ilen = int(item_lens[L]) if L < len(item_lens) else 0
                    max_pages = dlen // ilen if ilen > 0 else 0
                    layout_summary.append(
                        f"L{L}:(ptr=0x{p:x},data_len={dlen},item_len={ilen},"
                        f"max_pages={max_pages})"
                    )
                logger.info(
                    "[KV_HASH_DBG] side=%s rank=%d ONE_SHOT_LAYOUT "
                    "num_kv_layers=%d layers=%s",
                    side, rank, num_layers, " ".join(layout_summary),
                )
            per_layer = []
            for layer_id in range(num_layers):
                if layer_id >= len(item_lens):
                    continue
                src_ptr = int(kv_ptrs[layer_id])
                item_len = int(item_lens[layer_id])
                if item_len <= 0:
                    continue
                page_hashes = []
                for pidx in sampled_pages:
                    addr = src_ptr + int(pidx) * item_len
                    buf = (ctypes.c_uint8 * sample_bytes)()
                    err = cudart.cudaMemcpy(
                        ctypes.cast(buf, ctypes.c_void_p),
                        ctypes.c_void_p(addr),
                        ctypes.c_size_t(sample_bytes),
                        ctypes.c_int(2),  # cudaMemcpyDeviceToHost
                    )
                    if err != 0:
                        page_hashes.append(f"err{err}")
                    else:
                        page_hashes.append(f"{zlib.crc32(bytes(buf)):08x}")
                per_layer.append(f"L{layer_id}:[{','.join(page_hashes)}]")
            logger.info(
                "[KV_HASH_REQ] side=%s rank=%d room=%s num_pages=%d "
                "num_layers=%d sample_idx=%s sample_pages=%s sample_bytes=%d "
                "hashes=%s",
                side,
                rank,
                room,
                n,
                num_layers,
                sample_idx,
                sampled_pages,
                sample_bytes,
                " ".join(per_layer),
            )
            state_ptrs = getattr(self.kv_args, "state_data_ptrs", []) or []
            state_item_lens = getattr(self.kv_args, "state_item_lens", []) or []
            state_type_str = getattr(self.kv_args, "state_type", "none")
            if state_ptrs and state_type_str == "nsa":
                per_layer_state = []
                num_state_layers = len(state_ptrs)
                for layer_id in range(num_state_layers):
                    if layer_id >= len(state_item_lens):
                        continue
                    sptr = int(state_ptrs[layer_id])
                    sitem_len = int(state_item_lens[layer_id])
                    if sitem_len <= 0:
                        continue
                    page_hashes = []
                    for pidx in sampled_pages:
                        addr = sptr + int(pidx) * sitem_len
                        sbuf = (ctypes.c_uint8 * sample_bytes)()
                        err = cudart.cudaMemcpy(
                            ctypes.cast(sbuf, ctypes.c_void_p),
                            ctypes.c_void_p(addr),
                            ctypes.c_size_t(sample_bytes),
                            ctypes.c_int(2),
                        )
                        if err != 0:
                            page_hashes.append(f"err{err}")
                        else:
                            page_hashes.append(
                                f"{zlib.crc32(bytes(sbuf)):08x}"
                            )
                    per_layer_state.append(
                        f"L{layer_id}:[{','.join(page_hashes)}]"
                    )
                logger.info(
                    "[STATE_HASH_REQ] side=%s rank=%d room=%s "
                    "num_pages=%d num_state_layers=%d sample_idx=%s "
                    "sample_pages=%s sample_bytes=%d hashes=%s",
                    side,
                    rank,
                    room,
                    n,
                    num_state_layers,
                    sample_idx,
                    sampled_pages,
                    sample_bytes,
                    " ".join(per_layer_state),
                )
        except Exception as exc:
            logger.warning("[KV_HASH_REQ] %s emit failed: %r", side, exc)

    _cudart_lib = None
    _cudart_lib_init = False

    @classmethod
    def _get_cudart_lib(cls):
        """Load libcudart via ctypes (torch.cuda.cudart() doesn't expose
        cudaMemcpy). Cached after first successful load. Returns None on
        platforms without libcudart — hash logging silently skipped.
        """
        if cls._cudart_lib_init:
            return cls._cudart_lib
        cls._cudart_lib_init = True
        try:
            import ctypes.util

            name = ctypes.util.find_library("cudart") or "libcudart.so"
            lib = ctypes.CDLL(name)
            lib.cudaMemcpy.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_size_t,
                ctypes.c_int,
            ]
            lib.cudaMemcpy.restype = ctypes.c_int
            cls._cudart_lib = lib
        except Exception as exc:
            logger.warning("[LP_HASH] libcudart load failed: %r", exc)
            cls._cudart_lib = None
        return cls._cudart_lib

    def pop_layer_pipeline_metrics(self) -> Tuple[int, List[float]]:
        """Snapshot+reset periodic LP metrics for the scheduler's stats loop.

        Lifetime total `_lp_chunks_total` is preserved across calls.
        Returns `(chunk_delta, samples_ms)`.
        """
        metrics_lock = getattr(self, "_lp_metrics_lock", None)
        if metrics_lock is None:
            return 0, []
        with metrics_lock:
            delta = self._lp_chunks_periodic
            samples = self._lp_chunk_ms_samples
            self._lp_chunks_periodic = 0
            self._lp_chunk_ms_samples = []
        return delta, samples

    def make_layer_pipeline_hook(
        self,
        dispatch: List["_LayerPipelineRequestDispatch"],
    ) -> Callable[[int, "ForwardBatch"], None]:
        """Build the per-batch forward hook the scheduler attaches to
        `ForwardBatch.layer_pipeline_hook`.

        Fires from `RadixAttention.forward` once per layer; only does work
        on layer-group boundaries (or the last main layer), at which point
        it records a CUDA event on the compute stream and enqueues one
        `TransferKVChunk` per dispatch entry. The trailing aux/state chunk
        is still owned by `MooncakeKVSender.send`.
        """
        group_size = self.layer_group_size
        if group_size <= 0:
            # Misconfigured group_size; degrade safely to legacy path.
            return lambda layer_id, _fb: None
        # Dummy CP rank: sender.send early-returns, so hook fires would
        # advance the chunk counter without a finalizer and deadlock the
        # receiver-side watermark.
        if getattr(self, "is_dummy_cp_rank", False):
            return lambda layer_id, _fb: None
        import torch as _torch

        num_layers = self.local_num_kv_layers()
        # Hard-fail when group_size exceeds this PP stage's layer count:
        # split_layer_groups would collapse to a single group and silently
        # offer zero overlap.
        if group_size > num_layers:
            raise ValueError(
                f"--disagg-layer-group-size={group_size} exceeds the "
                f"current PP stage's KV layer count ({num_layers}). "
                f"Pick a value in [1, {num_layers}] — typically a "
                f"divisor of num_layers_per_pp_stage so groups are "
                f"evenly sized. With PP > 1, num_layers_per_pp_stage = "
                f"total_kv_layers / pp_size, NOT the global model's "
                f"layer count."
            )
        # When a draft (MTP/EAGLE NEXTN) model is loaded, its KV pool ptrs
        # are appended to the main pool's ptrs. The main model's forward
        # only iterates main layers, so pin `last_layer_idx` to the LAST
        # MAIN layer; the draft KV ships via `send_draft_kv` after the
        # draft forward completes.
        num_main_kv_layers = getattr(
            self.kv_args, "prefill_num_main_kv_layers", None
        )
        if num_main_kv_layers is None or num_main_kv_layers > num_layers:
            num_main_kv_layers = num_layers  # safe default: no draft
        last_layer_idx = num_main_kv_layers - 1
        # Snapshot the PP-stage offset into closure scope so the hook can
        # convert global `RadixAttention.layer_id` to the local index used
        # by `kv_data_ptrs`.
        prefill_start_layer = self.kv_args.prefill_start_layer
        # Ceil division so non-multiple num_main_kv_layers reports the
        # correct count (e.g. 10/4 = 3 groups).
        total_layer_groups = (
            num_main_kv_layers + group_size - 1
        ) // group_size
        if not getattr(self, "_lp_layer_geom_logged", False):
            _groups_dbg = split_layer_groups(num_main_kv_layers, group_size)
            logger.info(
                "[layer-pipeline GEOM] num_kv_layers=%d "
                "num_main_kv_layers=%d num_draft_kv_layers=%d "
                "group_size=%d total_layer_groups=%d "
                "prefill_start_layer=%d last_main_group=%s main_groups=%s "
                "draft_sent_via=send_draft_kv",
                num_layers, num_main_kv_layers,
                num_layers - num_main_kv_layers,
                group_size, total_layer_groups,
                prefill_start_layer,
                _groups_dbg[-1] if _groups_dbg else None,
                _groups_dbg,
            )
            self._lp_layer_geom_logged = True

        # Snapshot CP layer-shard ownership state into closure scope so
        # each fire pays only a local-int branch. In layer-shard mode
        # each CP rank owns layer groups whose id mod attn_cp_size
        # equals attn_cp_rank; non-owner fires skip both enqueue and
        # the sender counter bump (so empty-owner ranks reach send()
        # with _hook_enqueued_chunks=0 and take the empty-finalizer path).
        # Defensive `getattr` so callers that mock the manager surface
        # (older unit tests) keep working — missing helper ⇒ shard off.
        use_layer_cp_shard = getattr(
            self, "use_layer_cp_shard_for_transfer", lambda: False
        )()
        cp_size_for_shard = (
            getattr(self, "attn_cp_size", 1) if use_layer_cp_shard else 1
        )
        cp_rank_for_shard = (
            getattr(self, "attn_cp_rank", 0) if use_layer_cp_shard else 0
        )

        # Snapshot diagnostic env-vars into closure scope so each fire
        # pays only a local-bool branch (not os.environ.get).
        instrumentation_timing = envs.SGLANG_DISAGG_LAYER_PIPELINE_HOOK_TIMING.get()
        instrumentation_noop = envs.SGLANG_DISAGG_LAYER_PIPELINE_HOOK_NOOP.get()
        if (instrumentation_timing or instrumentation_noop) and not getattr(
            self, "_hook_instrumentation_logged_init", False
        ):
            logger.warning(
                "[layer-pipeline] hook diagnostic ENABLED — "
                "TIMING=%s NOOP=%s. NEVER enable in production.",
                instrumentation_timing,
                instrumentation_noop,
            )
            self._hook_instrumentation_logged_init = True

        def _hook(layer_id: int, _fb: "ForwardBatch") -> None:
            # `RadixAttention.forward` is shared with decode and draft
            # forwards; bail on non-extend batches as a safety net.
            if not _fb.forward_mode.is_extend():
                return
            # `RadixAttention.layer_id` is global; subtract the PP stage
            # offset to get the local index into `kv_data_ptrs`.
            local_id = layer_id - prefill_start_layer
            # Drop fires from PP foreign layers and from draft slots: the
            # draft pool's trailing ptrs are never iterated by main
            # RadixAttention.
            if local_id < 0 or local_id >= num_main_kv_layers:
                return
            is_group_boundary = (local_id + 1) % group_size == 0
            is_last = local_id == last_layer_idx  # = num_main_kv_layers - 1
            if not is_group_boundary and not is_last:
                return
            layer_end = local_id + 1
            if is_last and not is_group_boundary:
                # Final main partial group (num_main_kv_layers % group_size
                # != 0). Use the canonical splitter so layer_start lines
                # up with prior boundaries.
                groups = split_layer_groups(num_main_kv_layers, group_size)
                layer_start, _ = groups[-1]
            else:
                layer_start = layer_end - group_size
            # Env-gated KV visibility reconfirm: forces an all-streams
            # barrier at the hook site. NEVER set in production.
            if envs.SGLANG_DISAGG_LAYER_PIPELINE_VERIFY_KV.get():
                _torch.cuda.synchronize()
                logger.info(
                    "[layer-pipeline VERIFY_KV] hook fire local_id=%s "
                    "layer_range=(%s,%s) dispatches=%s",
                    local_id,
                    layer_start,
                    layer_end,
                    len(dispatch),
                )
            ev = _torch.cuda.Event() if not instrumentation_noop else None
            if ev is not None:
                ev.record()
            layer_group_id = local_id // group_size
            # Layer-shard CP ownership: only the owner CP rank enqueues
            # this group's KV (+ matching NSA state). Non-owner ranks skip
            # both add_transfer_request AND the per-sender chunk counter
            # bump so sender.send's `_chunks_sent = hook_enqueued + 1`
            # accounting stays per-rank correct.
            if (
                cp_size_for_shard > 1
                and layer_group_id % cp_size_for_shard != cp_rank_for_shard
            ):
                return
            t_start_ns = (
                time.perf_counter_ns() if instrumentation_timing else 0
            )
            for entry in dispatch:
                if not instrumentation_noop:
                    self.add_transfer_request(
                        bootstrap_room=entry.room,
                        kv_indices=entry.page_indices,
                        index_slice=entry.index_slice,
                        is_last_chunk=False,            # KV-only — aux via sender.send
                        aux_index=None,
                        # Per-LP state indices for NSA hybrid models;
                        # None for non-NSA or scheduler opt-out.
                        state_indices=entry.state_indices,
                        layer_group_id=layer_group_id,
                        layer_range=(layer_start, layer_end),
                        total_layer_groups=total_layer_groups,
                        total_chunks_in_request=None,   # set on aux chunk only
                        transfer_event=ev,
                    )
                # Sender owns the running count so its trailing aux chunk
                # can declare `total_chunks_in_request` correctly. MUST
                # advance even in NOOP mode so the fail-loud check passes.
                entry.sender._hook_enqueued_chunks += 1
            if instrumentation_timing:
                elapsed_ns = time.perf_counter_ns() - t_start_ns
                self._hook_timing_total_ns += elapsed_ns
                self._hook_timing_fire_count += 1
                self._hook_timing_dispatch_count += len(dispatch)
                if (
                    self._hook_timing_fire_count
                    % self._HOOK_TIMING_LOG_EVERY_FIRES
                    == 0
                ):
                    avg_per_fire_us = (
                        self._hook_timing_total_ns
                        / self._hook_timing_fire_count
                        / 1000.0
                    )
                    avg_per_dispatch_us = (
                        self._hook_timing_total_ns
                        / max(self._hook_timing_dispatch_count, 1)
                        / 1000.0
                    )
                    logger.info(
                        "[layer-pipeline] hook timing: "
                        "fires=%d dispatches=%d total=%dms "
                        "avg/fire=%.1fus avg/dispatch=%.1fus "
                        "noop_mode=%s",
                        self._hook_timing_fire_count,
                        self._hook_timing_dispatch_count,
                        self._hook_timing_total_ns // 1_000_000,
                        avg_per_fire_us,
                        avg_per_dispatch_us,
                        instrumentation_noop,
                    )

        return _hook

    def make_layer_pipeline_hook_for_reqs(
        self,
        reqs_with_indices: List[
            Tuple[
                "MooncakeKVSender",
                npt.NDArray[np.int32],
                slice,
                Optional[npt.NDArray[np.int32]],
            ]
        ],
    ) -> Optional[Callable[[int, "ForwardBatch"], None]]:
        """Scheduler entry point: filter senders for layer-pipeline
        eligibility, mark them, and build the hook closure.

        Each tuple is `(sender, page_indices, index_slice, state_indices)`
        for one request. `state_indices` is None when state ships one-shot
        via the aux finalizer. Three-tuples (no `state_indices`) are also
        accepted for back-compat.

        Returns `None` when no req qualifies — caller must NOT install a
        hook in that case.
        """
        if not self.layer_pipeline_enabled or self.is_dummy_cp_rank:
            return None
        dispatches: List[_LayerPipelineRequestDispatch] = []
        for tup in reqs_with_indices:
            if len(tup) == 3:
                sender, page_indices, index_slice = tup
                state_indices = None
            else:
                sender, page_indices, index_slice, state_indices = tup
            if len(page_indices) == 0:
                # Already filtered upstream, but keep the guard so a
                # future caller that forgets it cannot wedge the watermark.
                continue
            layer_groups = sender._layer_groups_for_send()
            if len(layer_groups) == 1 and layer_groups[0] is None:
                # Keep scheduler hook eligibility exactly aligned with
                # sender.send's legacy fallback gate. This covers short
                # requests below min_prefill_len, staging fallback, room
                # capability mismatches, and group_size <= 0.
                continue
            # Contract with `MooncakeKVSender.send`: this flag means the
            # next `send()` for this room MUST take the aux-only path
            # and will fail loudly if no hook fire bumped the counter.
            sender._hook_handled_in_current_send = True
            # Tells the aux finalizer to skip state (already shipped per
            # layer-group by the hook), avoiding a double-send.
            sender._hook_handles_state = state_indices is not None
            if state_indices is not None:
                # Latch the persistent variant so the empty-last-chunk
                # aux-only fallback (in send()) knows state was hook-shipped
                # in some prior round even after the per-send flag is cleared.
                sender._hook_handles_state_persistent = True
            # Layer-shard mode: empty-owner CP ranks legitimately enqueue
            # zero hook chunks. send() uses this flag to relax the
            # "hook fired but counter didn't move" contract guard.
            sender._hook_layer_shard_active = getattr(
                self, "use_layer_cp_shard_for_transfer", lambda: False
            )()
            dispatches.append(
                _LayerPipelineRequestDispatch(
                    room=sender.bootstrap_room,
                    sender=sender,
                    page_indices=page_indices,
                    index_slice=index_slice,
                    state_indices=state_indices,
                )
            )
        if not dispatches:
            return None
        return self.make_layer_pipeline_hook(dispatches)

    def transfer_worker(
        self,
        queue: FastQueue,
        executor: concurrent.futures.ThreadPoolExecutor,
        staging_buffer=None,
        worker_index=0,
    ):
        staging_strategy = None
        trace_enabled = getattr(self, "enable_trace", False)
        if trace_enabled:
            trace_set_thread_info(
                f"mooncake transfer worker {worker_index}",
                tp_rank=self.attn_tp_rank,
                dp_rank=self.attn_dp_rank,
            )

        while True:
            try:
                kv_chunk: TransferKVChunk = queue.get()
                if trace_enabled:
                    kv_chunk.trace_ctx.rebuild_thread_context()
                    kv_chunk.trace_ctx.trace_slice_start(
                        MooncakeRequestStage.MOONCAKE_WORKER_SEND.stage_name,
                        MooncakeRequestStage.MOONCAKE_WORKER_SEND.level,
                    )

                _status = self.request_status.get(kv_chunk.room)
                if _status is None or _status in (KVPoll.Failed, KVPoll.Success):
                    logger.debug(
                        f"Skipping chunk for room {kv_chunk.room} because it has already failed or been aborted"
                    )
                    self._clear_layer_pipeline_progress(kv_chunk.room)
                    if trace_enabled:
                        kv_chunk.trace_ctx.trace_slice_end(
                            MooncakeRequestStage.MOONCAKE_WORKER_SEND.stage_name,
                            MooncakeRequestStage.MOONCAKE_WORKER_SEND.level,
                            thread_finish_flag=True,
                        )
                    continue

                # If the forward hook recorded an event after the layer
                # group's KV write, block this worker until the GPU has
                # committed those bytes. `event.synchronize()` is the
                # right barrier because mooncake's RDMA call is CPU-side
                # — `event.wait(stream=...)` would not gate it.
                if kv_chunk.transfer_event is not None:
                    try:
                        kv_chunk.transfer_event.synchronize()
                    except Exception as exc:
                        logger.error(
                            "transfer_event.synchronize() failed "
                            "for room=%s: %s. Marking room failed and "
                            "continuing — other rooms on this worker are "
                            "unaffected.",
                            kv_chunk.room,
                            exc,
                        )
                        self.record_failure(
                            kv_chunk.room,
                            f"layer-pipeline event sync failed: {exc!r}",
                        )
                        self.update_status(kv_chunk.room, KVPoll.Failed)
                        self._clear_layer_pipeline_progress(kv_chunk.room)
                        if trace_enabled:
                            kv_chunk.trace_ctx.trace_slice_end(
                                MooncakeRequestStage.MOONCAKE_WORKER_SEND.stage_name,
                                MooncakeRequestStage.MOONCAKE_WORKER_SEND.level,
                                thread_finish_flag=True,
                            )
                        continue
                # Env-gated KV byte-hash sample logging, after GPU sync and
                # before RDMA submit. LP KV chunks only.
                if envs.SGLANG_DISAGG_LAYER_PIPELINE_HASH_LOG.get():
                    self._log_lp_kv_byte_hash(kv_chunk)
                # Aux-only finalizer: hook already enqueued every
                # layer-group KV chunk; sender.send queues a single
                # trailing aux/state chunk. Skip the KV-send paths.
                is_aux_only_chunk = (
                    kv_chunk.layer_range is None
                    and len(kv_chunk.prefill_kv_indices) == 0
                    and kv_chunk.is_last_chunk
                )
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
                # Unique id per prefill sender so decode's response set size matches expected_response_num.
                prefill_unique_rank = (
                    self.attn_tp_rank * (self.pp_size * self.attn_cp_size)
                    + self.pp_rank * self.attn_cp_size
                    + self.attn_cp_rank
                )
                # When staging transfer is not yet ready (watermark/allocation pending),
                # the chunk is re-enqueued and we break out of the req loop to retry later.
                staging_deferred = False
                # LP-off restores the pre-LP completion path: per-chunk
                # success accumulated locally and synced once every dst
                # rank has received KV + aux, with no watermark dict.
                # The LP watermark helpers run only when LP is enabled.
                use_lp_completion = self.layer_pipeline_enabled
                polls: List[bool] = []
                dst_ranks_infos: List[Tuple[str, int, int]] = []
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
                                # Drop watermark state so a stale
                                # `success_synced=False` entry cannot
                                # resurrect a Success notification on
                                # later chunks for the same room.
                                self._clear_layer_pipeline_progress(kv_chunk.room)
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
                        if is_aux_only_chunk:
                            # Hook owns all KV bytes; pretend KV send
                            # succeeded so the chunk-count watermark
                            # advances (aux-only is the (N+1)-th chunk
                            # accounted for by the sender's total).
                            ret = 0
                        elif self.is_mla_backend or (
                            self.attn_tp_size
                            == target_rank_registration_info.dst_attn_tp_size
                        ):
                            ret = self.send_kvcache(
                                req.mooncake_session_id,
                                kv_chunk.prefill_kv_indices,
                                target_rank_registration_info.dst_kv_ptrs,
                                chunked_dst_kv_indice,
                                executor,
                                layer_range=kv_chunk.layer_range,
                                dst_num_main=target_rank_registration_info.dst_num_main_kv_layers,
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
                                layer_range=kv_chunk.layer_range,
                                dst_num_main=target_rank_registration_info.dst_num_main_kv_layers,
                            )

                        # Per-LP-chunk state pipeline: when the hook
                        # dispatched state alongside KV, ship the state
                        # slice right after KV using the same layer_range.
                        if (
                            ret == 0
                            and kv_chunk.layer_range is not None
                            and kv_chunk.state_indices is not None
                        ):
                            ret = self.maybe_send_extra(
                                req,
                                kv_chunk.state_indices,
                                executor,
                                target_rank_registration_info,
                                layer_range=kv_chunk.layer_range,
                                # NSA state shares page numbering with KV,
                                # so the same slice that indexes
                                # `dst_kv_indices` indexes the matching
                                # `dst_state_indices` slice.
                                dst_state_index_slice=kv_chunk.index_slice,
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
                            self._clear_layer_pipeline_progress(kv_chunk.room)
                            break

                        # Chunk RDMA succeeded for this dst rank; may
                        # trigger Success sync if this completion crosses
                        # the watermark (LP-only; legacy finalizes below).
                        if use_lp_completion:
                            self._record_chunk_done(
                                kv_chunk.room,
                                (req.endpoint, req.dst_port),
                                req.required_dst_info_num,
                                prefill_unique_rank,
                            )
                            if envs.SGLANG_DISAGG_KV_HASH_VERIFY.get():
                                self._kv_hash_record_prefill_chunk(
                                    kv_chunk.room,
                                    kv_chunk.prefill_kv_indices,
                                    kv_chunk.layer_range,
                                )
                            # Only LP chunks contribute (legacy / aux-only
                            # carry enqueue_ns=0, filtered inside the helper).
                            self._record_layer_group_metric(kv_chunk.enqueue_ns)

                        if kv_chunk.is_last_chunk:
                            if kv_chunk.state_indices:
                                self.maybe_send_extra(
                                    req,
                                    kv_chunk.state_indices,
                                    executor,
                                    target_rank_registration_info,
                                )

                            # Only the last chunk we need to send the aux data
                            # Layer-shard CP optimization: aux content is
                            # identical across CP ranks (per-request, not
                            # per-page) and CP0 is the single writer.
                            # Non-CP0 ranks still call `_record_aux_sent`
                            # so their per-rank watermark closes, but skip
                            # send_aux to avoid N-fold metadata writes.
                            if kv_chunk.skip_aux_rdma:
                                aux_ret = 0
                            else:
                                aux_ret = self.send_aux(
                                    req,
                                    kv_chunk.prefill_aux_index,
                                    target_rank_registration_info.dst_aux_ptrs,
                                )
                            if use_lp_completion:
                                # `_record_aux_sent` emits Success/Failed
                                # sync once every dst rank's aux is in AND
                                # its chunk count reaches
                                # `total_chunks_in_request`.
                                self._record_aux_sent(
                                    kv_chunk.room,
                                    (req.endpoint, req.dst_port),
                                    kv_chunk.total_chunks_in_request,
                                    req.required_dst_info_num,
                                    aux_ret,
                                    prefill_unique_rank,
                                )
                            else:
                                # Legacy path: a single full-layer chunk per
                                # request, so finalize once all dst ranks have
                                # reported — identical to the pre-LP sync timing.
                                polls.append(True if aux_ret == 0 else False)
                                dst_ranks_infos.append(
                                    (req.endpoint, req.dst_port, req.room)
                                )
                                if len(polls) == req.required_dst_info_num:
                                    status = (
                                        KVPoll.Success if all(polls) else KVPoll.Failed
                                    )
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

                    if trace_enabled:
                        mooncake_trace_slice(
                            kv_chunk.trace_ctx,
                            MooncakeRequestStage.MOONCAKE_WORKER_SEND_SESSION,
                            start_ts,
                        )

                if trace_enabled:
                    kv_chunk.trace_ctx.trace_slice_end(
                        MooncakeRequestStage.MOONCAKE_WORKER_SEND.stage_name,
                        MooncakeRequestStage.MOONCAKE_WORKER_SEND.level,
                        thread_finish_flag=True,
                    )

                if staging_deferred:
                    continue

                if (
                    kv_chunk.room not in self.request_status
                    or self.check_status(kv_chunk.room) == KVPoll.Success
                ):
                    if kv_chunk.room in self.transfer_infos:
                        self.transfer_infos.pop(kv_chunk.room)
                    if hasattr(self, "req_to_decode_prefix_len"):
                        self.req_to_decode_prefix_len.pop(kv_chunk.room, None)
                    # Drop watermark state on terminal status.
                    self._clear_layer_pipeline_progress(kv_chunk.room)

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
                    # Loud-fail on LP misconfiguration at handshake time.
                    # `_room_supports_layer_pipeline` silently falls back to
                    # legacy fan-out on size mismatches; surface it here
                    # ONCE per registering session so a bad deployment is
                    # visible in startup logs.
                    self._maybe_warn_lp_handshake_mismatch(
                        mooncake_session_id,
                        self.decode_kv_args_table[mooncake_session_id],
                    )
                    logger.debug(
                        f"Register KVArgs from {mooncake_session_id} successfully"
                    )
                    continue
                else:
                    required_dst_info_num = int(waiting_req_bytes[7].decode("ascii"))
                    room = int(room)
                    if room not in self.transfer_infos:
                        self.transfer_infos[room] = {}

                    self.transfer_infos[room][mooncake_session_id] = (
                        TransferInfo.from_zmq(waiting_req_bytes)
                    )
                    # NOTE: after bootstrapping we can mark the req as waiting for input
                    if len(self.transfer_infos[room]) == required_dst_info_num:
                        self.req_to_decode_prefix_len[room] = next(
                            (
                                info.decode_prefix_len
                                for info in self.transfer_infos[room].values()
                                if info.decode_prefix_len is not None
                            ),
                            0,
                        )
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
                        if arrived_response_num == expected_response_num:
                            if self.enable_staging:
                                handler = self._staging_handler
                                if handler.is_staging_room(bootstrap_room):
                                    handler.submit_last_scatter_async(bootstrap_room)
                                self._chunk_writer_counts.pop(bootstrap_room, None)
                            self.update_status(bootstrap_room, KVPoll.Success)
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
        layer_group_id: int = 0,
        layer_range: Optional[Tuple[int, int]] = None,
        total_layer_groups: int = 1,
        total_chunks_in_request: Optional[int] = None,
        transfer_event: Optional["torch.cuda.Event"] = None,
        skip_aux_rdma: bool = False,
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
                trace_ctx=trace_ctx,
                layer_group_id=layer_group_id,
                layer_range=layer_range,
                total_layer_groups=total_layer_groups,
                kv_dtype=self.server_args.kv_cache_dtype,
                total_chunks_in_request=total_chunks_in_request,
                transfer_event=transfer_event,
                # Stamped only on LP chunks; legacy / aux-only carry 0
                # so the metric doesn't double-count.
                enqueue_ns=(time.monotonic_ns() if layer_range is not None else 0),
                skip_aux_rdma=skip_aux_rdma,
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
    ):
        super().__init__(mgr, bootstrap_addr, bootstrap_room, dest_tp_ranks, pp_rank)
        self.conclude_state = None
        self.init_time = time.time()
        self._init_trace_ctx()
        # Total chunks enqueued for this request across its whole lifecycle.
        # The last enqueued chunk carries this value so the receiver knows
        # how many completions to wait for before Success. Survives across
        # `send()` calls (chunked-prefill). A request runs in exactly one
        # mode: legacy fan-out (per-call increment) OR hook-driven (set
        # once to `_hook_enqueued_chunks + 1`).
        self._chunks_sent: int = 0
        # Bumped by the forward hook on every KV-only chunk it enqueues.
        # The trailing aux finalizer uses it to compute
        # `total_chunks_in_request = _hook_enqueued_chunks + 1`. No lock:
        # hook fires on the compute thread, `send()` on the scheduler
        # thread AFTER forward returns — they never overlap.
        self._hook_enqueued_chunks: int = 0
        # Per-`send()` mode signal set by the scheduler ahead of forward
        # when it built a hook dispatch entry. `send()` consumes (and
        # resets) it on entry; `_hook_chunks_at_last_send` snapshots the
        # counter so we can fail loudly when hook mode was dispatched but
        # the hook never fired this round.
        self._hook_handled_in_current_send: bool = False
        self._hook_chunks_at_last_send: int = 0
        # Per-`send()` flag for the NSA state buffer: when True the aux
        # finalizer passes `state_indices=None` (per-layer-group state was
        # already shipped by the hook); when False the legacy aux
        # finalizer ships the one-shot state.
        self._hook_handles_state: bool = False
        # Persistent (never auto-cleared) variant of `_hook_handles_state`:
        # latches True once the hook ever shipped state for this sender.
        # Read by the empty-last-chunk aux-only finalizer, where the
        # per-`send()` flag is no longer available.
        self._hook_handles_state_persistent: bool = False
        # Per-`send()` flag set when scheduler installed the LP hook under
        # CP layer-shard mode. Empty-owner CP ranks legitimately see
        # _hook_enqueued_chunks unchanged across send() calls; the default
        # contract guard would mistake that for a missed fire.
        self._hook_layer_shard_active: bool = False

    def _layer_groups_for_send(self) -> List[Optional[Tuple[int, int]]]:
        # Gate by the request-level prefill length (in page units) so
        # chunked-prefill cannot push individual chunks below
        # `min_prefill_len` and force fallback.
        prefill_len = self.num_kv_indices * self.kv_mgr.kv_args.page_size
        # The runtime staging branch in `transfer_worker` only fires for
        # non-MLA + TP-mismatched dst ranks, so MLA can safely use LP even
        # when `enable_staging=True`. For non-MLA + matched-TP + staging
        # the staging path is registered but unused per-dst at runtime; we
        # still fall back conservatively because we don't yet know each
        # dst's `dst_attn_tp_size` here.
        staging_path_active = (
            self.kv_mgr.enable_staging and not self.kv_mgr.is_mla_backend
        )
        if (
            not self.kv_mgr.layer_pipeline_enabled
            or staging_path_active
            or not self.kv_mgr._room_supports_layer_pipeline(self.bootstrap_room)
            or self.kv_mgr.layer_group_size <= 0
            or prefill_len < self.kv_mgr.layer_pipeline_min_prefill_len
        ):
            return [None]
        return split_layer_groups(
            self.kv_mgr.local_num_kv_layers(), self.kv_mgr.layer_group_size
        )

    def _draft_layer_range(self) -> Optional[Tuple[int, int]]:
        """Return `(num_main, num_total)` if a draft (MTP/EAGLE NEXTN) KV
        pool is appended to `kv_data_ptrs`, else `None`.

        Units are **logical layers** to match `prefill_num_main_kv_layers`
        (which `prefill.py:148` already divides by 2 for MHA) and the
        `layer_range` semantics. Using raw `len(kv_data_ptrs)` would
        double-count MHA's K/V split and yield an out-of-bounds
        `(num_main, 2*num_main)` for MHA-no-draft.
        """
        num_total = self.kv_mgr.local_num_kv_layers()
        num_main = getattr(
            self.kv_mgr.kv_args, "prefill_num_main_kv_layers", None
        )
        if num_main is None or num_main >= num_total or num_main < 0:
            return None
        return (num_main, num_total)

    def send_draft_kv(
        self,
        kv_indices: npt.NDArray[np.int32],
    ) -> None:
        """Ship the draft (MTP/EAGLE NEXTN) KV slice for this token chunk.

        Called by the scheduler after `forward_batch_generation` returns
        (so the draft forward has written its KV bytes) and before
        `sender.send()`. The main LP hook does not cover draft layers.

        Enqueues one TransferKVChunk with
        `layer_range=(num_main, num_total)`, sharing `kv_indices` with
        main. Counts toward `_hook_enqueued_chunks` so the trailing aux
        finalizer's chunk-total accounting stays correct.

        No-op when: kv_indices is empty (empty-last-chunk under chunked
        prefill), no draft pool, LP path inactive, or dummy CP rank.
        """
        # Empty-last-chunk: chunked prefill's page-alignment leftover
        # produces a final send() with no new pages. Calling here would
        # enqueue a phantom draft chunk and break the aux-only finalizer
        # contract. Defend at the source.
        if len(kv_indices) == 0:
            return
        draft_range = self._draft_layer_range()
        if draft_range is None:
            return
        if self.kv_mgr.is_dummy_cp_rank:
            # Dummy CP rank short-circuits all transfer; draft must match
            # or watermark accounting diverges from live transfer.
            return
        # If LP is not active for this request the legacy fan-out path
        # will ship all layers at once — do not pre-empt with a
        # draft-only chunk.
        layer_groups = self._layer_groups_for_send()
        lp_active = not (len(layer_groups) == 1 and layer_groups[0] is None)
        if not lp_active:
            return
        # CP layer-shard mode: treat draft as the (N_main_groups)-th group
        # and route it to a single owner CP rank. Main groups use local
        # layer ids; draft layers sit AFTER all main layers so
        # `draft_group_id == ceil(num_main / group_size)`. Non-owner ranks
        # skip both enqueue and counter bump.
        use_layer_cp_shard = self.kv_mgr.use_layer_cp_shard_for_transfer()
        if use_layer_cp_shard:
            num_main_kv_layers = getattr(
                self.kv_mgr.kv_args, "prefill_num_main_kv_layers", None
            )
            if num_main_kv_layers is None or num_main_kv_layers <= 0:
                # No main layer info ⇒ cannot compute a deterministic draft
                # owner. Fall through to page-shard behavior: EVERY CP rank
                # ships the same draft chunk. This is N-fold redundant RDMA
                # write to the same dst buffer (correct, since draft KV is
                # per-request, not per-page), but the receiver-side
                # watermark still closes because every rank reports its
                # chunk. We accept the redundancy over the alternative
                # (skipping draft entirely on non-CP0), which would diverge
                # from the main-LP layer-shard accounting.
                pass
            else:
                group_size = self.kv_mgr.layer_group_size
                total_main_layer_groups = (
                    num_main_kv_layers + group_size - 1
                ) // group_size
                draft_group_id = total_main_layer_groups
                if (
                    draft_group_id % self.kv_mgr.attn_cp_size
                    != self.kv_mgr.attn_cp_rank
                ):
                    return
        # Match sender.send's index_slice contract WITHOUT advancing
        # curr_idx — the following sender.send(kv_indices) advances it.
        index_slice = slice(self.curr_idx, self.curr_idx + len(kv_indices))
        # CP filter: when all_cp_ranks transfer is enabled with page shard,
        # restrict to this rank's share. Layer-shard mode keeps the full
        # page set (only one CP rank ships draft).
        if (
            self.kv_mgr.enable_all_cp_ranks_for_transfer
            and not use_layer_cp_shard
        ):
            kv_indices, index_slice = filter_kv_indices_for_cp_rank(
                self.kv_mgr,
                kv_indices,
                index_slice,
            )
            if len(kv_indices) == 0:
                return
        # Record an event so the worker synchronizes before RDMA, in case
        # draft_fwd's KV writes haven't yet committed to the pool.
        ev = None
        try:
            import torch as _torch

            if _torch.cuda.is_available():
                ev = _torch.cuda.Event()
                ev.record()
        except ImportError:
            ev = None
        self._hook_enqueued_chunks += 1
        # Ship draft state alongside KV: state and KV share page numbering;
        # transfer_worker's LP path will chain
        # maybe_send_extra(state_indices=kv_indices, layer_range=draft_range).
        self.kv_mgr.add_transfer_request(
            self.bootstrap_room,
            kv_indices,
            index_slice,
            is_last_chunk=False,
            aux_index=None,
            state_indices=kv_indices,
            layer_group_id=DRAFT_LAYER_GROUP_ID,
            layer_range=draft_range,
            total_layer_groups=DRAFT_TOTAL_LAYER_GROUPS_SENTINEL,
            total_chunks_in_request=None,
            transfer_event=ev,
        )

    @mooncake_trace_func(MooncakeRequestStage.MOONCAKE_SEND)
    def send(
        self,
        kv_indices: npt.NDArray[np.int32],
        state_indices: Optional[List] = None,
    ):
        # The read-and-clear below is the FIRST thing send() does. Once
        # cleared, the flag cannot leak into a subsequent send() on this
        # sender — even if any later step raises. The scheduler resets
        # the flag every round via `make_layer_pipeline_hook_for_reqs`.
        # `_hook_chunks_at_last_send` is snapshotted in the `finally`
        # below; harmless on top-of-send exceptions (try never entered →
        # no update; next round's contract guard accommodates the
        # accumulated count since hook fires that DID issue RDMA stay
        # reflected in `_hook_enqueued_chunks`).
        hook_expected = self._hook_handled_in_current_send
        self._hook_handled_in_current_send = False
        hook_handled_state = self._hook_handles_state
        self._hook_handles_state = False
        hook_layer_shard = self._hook_layer_shard_active
        self._hook_layer_shard_active = False

        use_layer_cp_shard = self.kv_mgr.use_layer_cp_shard_for_transfer()
        index_slice = slice(self.curr_idx, self.curr_idx + len(kv_indices))
        self.curr_idx += len(kv_indices)
        is_last_chunk = self.curr_idx == self.num_kv_indices

        # Special handling for cp
        if (
            self.kv_mgr.enable_all_cp_ranks_for_transfer
            and not use_layer_cp_shard
        ):
            # Page-shard mode: every CP rank's send() filters its share of
            # pages at the top so both hook and legacy branches see the
            # partitioned indices. Layer-shard mode skips this: the hook
            # already enqueued full pages per owned group, and rewriting
            # `index_slice` here would leak a stale shard rectangle into
            # the aux finalizer / any future state code. It also must not
            # call filter_kv_indices_for_cp_rank because that helper
            # assumes a contiguous page range.
            kv_indices, index_slice = filter_kv_indices_for_cp_rank(
                self.kv_mgr,
                kv_indices,
                index_slice,
            )
        elif self.kv_mgr.is_dummy_cp_rank:
            if not is_last_chunk:
                return
            else:
                self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Success)
                return

        try:
            if hook_expected:
                # Layer-pipeline path: the hook already streamed every
                # layer group's KV bytes. Only the very last token chunk
                # of the whole request needs the trailing aux/state
                # finalizer.
                if self._hook_enqueued_chunks <= self._hook_chunks_at_last_send:
                    # Layer-shard mode: empty-owner CP ranks legitimately
                    # never bumped the counter (they own zero layer
                    # groups this round). The aux-only finalizer still
                    # needs to fire so the receiver's chunk-count
                    # watermark closes.
                    if not hook_layer_shard:
                        # Fail-loud: scheduler told us to expect hook-driven
                        # enqueue this round, but no hook fire bumped the
                        # counter. Refusing to emit a malformed aux finalizer.
                        raise RuntimeError(
                            f"Layer-pipeline contract violated for room "
                            f"{self.bootstrap_room}: scheduler dispatched hook "
                            f"mode (set _hook_handled_in_current_send=True) but "
                            f"no hook fire enqueued KV "
                            f"(_hook_enqueued_chunks={self._hook_enqueued_chunks}, "
                            f"baseline={self._hook_chunks_at_last_send}). "
                            f"Refusing to fall through — would corrupt "
                            f"receiver-side watermark accounting."
                        )
                if not is_last_chunk:
                    return
                # Hook NOOP mode skipped add_transfer_request; mirror
                # is_dummy_cp_rank by marking Success directly. Decode
                # side will fail; documented for this dev tool.
                if self.kv_mgr._lp_hook_noop_fake_success:
                    self.kv_mgr.update_status(
                        self.bootstrap_room, KVPoll.Success
                    )
                    return
                self._chunks_sent = self._hook_enqueued_chunks + 1
                # When the hook shipped state per layer-group, the aux
                # finalizer MUST pass state_indices=None or decode
                # receives two overlapping writes (last-writer wins).
                aux_state_indices = None if hook_handled_state else state_indices
                # Layer-shard CP optimization: only CP0 issues the actual
                # aux RDMA (aux content is identical across CP ranks,
                # per-request not per-page). Other CP ranks still call
                # _record_aux_sent so their per-rank watermark closes,
                # but skipping the RDMA avoids N-fold metadata writes.
                skip_aux_rdma = (
                    use_layer_cp_shard and self.kv_mgr.attn_cp_rank != 0
                )
                self.kv_mgr.add_transfer_request(
                    self.bootstrap_room,
                    # Empty kv_indices + layer_range=None signals
                    # "aux-only finalizer" to the worker.
                    kv_indices=np.array([], dtype=np.int32),
                    index_slice=index_slice,
                    is_last_chunk=True,
                    aux_index=self.aux_index,
                    state_indices=aux_state_indices,
                    layer_group_id=0,
                    layer_range=None,
                    total_layer_groups=1,
                    total_chunks_in_request=self._chunks_sent,
                    transfer_event=None,
                    trace_ctx=self.trace_ctx.copy_for_thread(),
                    skip_aux_rdma=skip_aux_rdma,
                )
                return
        finally:
            # Snapshot AFTER both branches so the next `send()`'s
            # baseline reflects everything enqueued so far.
            self._hook_chunks_at_last_send = self._hook_enqueued_chunks

        # Empty-last-chunk LP fallthrough: chunked prefill can produce a
        # final send() whose token range was fully covered by prior
        # chunks (page-alignment leftover ⇒ end_idx == start_idx). The
        # scheduler's hook skipped this round, so `hook_expected` is
        # False and `kv_indices` is empty — but prior rounds DID
        # hook-enqueue real KV chunks. The legacy fan-out below would
        # emit N empty layer-group chunks with a stale
        # `total_chunks_in_request = N`, clobbering the receiver-side
        # watermark (which already counted `_hook_enqueued_chunks` real
        # arrivals). Emit a single aux-only finalizer instead, with the
        # correct cumulative chunk total.
        if (
            is_last_chunk
            and len(kv_indices) == 0
            and self._hook_enqueued_chunks > 0
            and self._layer_groups_for_send() != [None]
        ):
            self._chunks_sent = self._hook_enqueued_chunks + 1
            aux_state_indices = (
                None if self._hook_handles_state_persistent else state_indices
            )
            skip_aux_rdma = (
                use_layer_cp_shard and self.kv_mgr.attn_cp_rank != 0
            )
            self.kv_mgr.add_transfer_request(
                self.bootstrap_room,
                kv_indices=np.array([], dtype=np.int32),
                index_slice=index_slice,
                is_last_chunk=True,
                aux_index=self.aux_index,
                state_indices=aux_state_indices,
                layer_group_id=0,
                layer_range=None,
                total_layer_groups=1,
                total_chunks_in_request=self._chunks_sent,
                transfer_event=None,
                trace_ctx=self.trace_ctx.copy_for_thread(),
                skip_aux_rdma=skip_aux_rdma,
            )
            return

        # Legacy path: hook is disabled (or this request didn't qualify
        # for layer pipeline at scheduler time). Enqueue the full N×M
        # layer-group fan-out exactly as before.
        layer_groups = self._layer_groups_for_send()
        for layer_group_id, layer_range in enumerate(layer_groups):
            is_last_layer_group = layer_group_id == len(layer_groups) - 1
            chunk_is_last = is_last_chunk and is_last_layer_group
            self._chunks_sent += 1
            self.kv_mgr.add_transfer_request(
                self.bootstrap_room,
                kv_indices,
                index_slice,
                chunk_is_last,
                aux_index=self.aux_index if chunk_is_last else None,
                state_indices=state_indices if chunk_is_last else None,
                trace_ctx=self.trace_ctx.copy_for_thread(),
                layer_group_id=layer_group_id,
                layer_range=layer_range,
                total_layer_groups=len(layer_groups),
                # Only the very last enqueued chunk carries the running
                # total so the worker can finalize deterministically.
                total_chunks_in_request=self._chunks_sent if chunk_is_last else None,
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
        if getattr(self.kv_mgr, "enable_trace", False):
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
            layer_pipeline_enabled = (
                b"1" if self.kv_mgr.layer_pipeline_enabled else b"0"
            )
            layer_group_size = str(self.kv_mgr.layer_group_size).encode("ascii")
            kv_dtype = self.kv_mgr.server_args.kv_cache_dtype.encode("ascii")
            # Decode-side local main KV layer count (excludes draft tail);
            # prefill receives it as ``dst_num_main_kv_layers`` to fail
            # loud when the cross-side draft layout disagrees. Empty ⇒
            # legacy decode / no draft pool (receivers fall back to
            # inferring from total length).
            num_main_kv_layers_attr = getattr(
                self.kv_mgr.kv_args, "num_main_kv_layers", None
            )
            num_main_kv_layers_str = (
                str(num_main_kv_layers_attr).encode("ascii")
                if num_main_kv_layers_attr is not None
                else b""
            )

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
                        layer_pipeline_enabled,
                        layer_group_size,
                        kv_dtype,
                        packed_staging_base_ptr,
                        staging_total_size_str,
                        num_main_kv_layers_str,
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
        state_indices: Optional[List] = None,
        decode_prefix_len: Optional[int] = None,
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
                            pack_int_lists(state_indices, "i")
                            if not is_dummy and state_indices
                            else b""
                        ),
                        str(self.required_dst_info_num).encode("ascii"),
                        str(decode_prefix_len or 0).encode("ascii"),
                    ]
                )
        self.init_time = time.time()
        # Stash dst page indices for the decode-side hash emit on first
        # Success poll. Save once per room.
        if envs.SGLANG_DISAGG_KV_HASH_VERIFY.get():
            self.kv_mgr._kv_hash_save_decode_indices(
                self.bootstrap_room, kv_indices
            )

    def poll(self) -> KVPoll:
        if self.conclude_state is not None:
            return self.conclude_state

        status = self.kv_mgr.check_status(self.bootstrap_room)
        if status in (KVPoll.Success, KVPoll.Failed):
            self.conclude_state = status
            if envs.SGLANG_DISAGG_KV_HASH_VERIFY.get():
                if status == KVPoll.Success:
                    self.kv_mgr._kv_hash_emit_decode_once(self.bootstrap_room)
                else:
                    self.kv_mgr._kv_hash_clear_decode(self.bootstrap_room)
        elif status == KVPoll.WaitingForInput:
            timeout_result = self._check_waiting_timeout()
            if timeout_result is not None:
                return timeout_result

        return status

    def failure_exception(self):
        if self.conclude_state is None:
            self.conclude_state = KVPoll.Failed

        if envs.SGLANG_DISAGG_KV_HASH_VERIFY.get():
            self.kv_mgr._kv_hash_clear_decode(self.bootstrap_room)
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
