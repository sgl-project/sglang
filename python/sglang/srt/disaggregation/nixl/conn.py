from __future__ import annotations

import dataclasses
import json
import logging
import struct
import threading
import time
import uuid
from collections import defaultdict
from typing import TYPE_CHECKING, Dict, List, Optional, Set

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from sglang.srt.disaggregation.common.staging_handler import StagingTransferInfo

from sglang.srt.disaggregation.base.conn import KVArgs, KVPoll, StateType
from sglang.srt.disaggregation.common.conn import (
    CommonKVBootstrapServer,
    CommonKVManager,
    CommonKVReceiver,
    CommonKVSender,
)
from sglang.srt.disaggregation.common.staging_handler import StagingRegisterInfo
from sglang.srt.disaggregation.common.utils import (
    FastQueue,
    group_concurrent_contiguous,
    pack_int_lists,
    unpack_int_lists,
)
from sglang.srt.disaggregation.utils import (
    DisaggregationMode,
    filter_kv_indices_for_cp_rank,
)
from sglang.srt.environ import envs
from sglang.srt.server_args import ServerArgs

try:
    from nixl._bindings import (
        nixlBackendError,
        nixlCancelledError,
        nixlRemoteDisconnectError,
    )

    _NIXL_TRANSPORT_ERRORS = (
        nixlRemoteDisconnectError,
        nixlBackendError,
        nixlCancelledError,
    )
except ImportError:
    _NIXL_TRANSPORT_ERRORS = (RuntimeError,)

logger = logging.getLogger(__name__)

GUARD = "NixlMsgGuard".encode("ascii")


@dataclasses.dataclass
class TransferInfo:
    """Contains indices for a transfer, sent by KVReceiver. Received by prefill bootstrap thread."""

    room: int
    endpoint: str
    dst_port: int
    agent_name: str
    dst_kv_indices: npt.NDArray[np.int32]
    dst_aux_index: int
    required_dst_info_num: int
    dst_state_indices: List[List[int]]
    decode_prefix_len: Optional[int] = None  # for decode radix cache
    # NOTE: optional staging field; populated via STAGING_RSP. Keep at the
    # end so positional construction in from_zmq() continues to work.
    staging: Optional["StagingTransferInfo"] = None

    def is_dummy(self):
        # A transfer is "dummy" only for CP non-authoritative ranks.
        # When dst_kv_indices is empty due to a decode-side radix cache
        # full hit (decode_prefix_len > 0), the transfer is NOT dummy --
        # aux/state data still needs to be sent.
        if self.dst_kv_indices.size == 0 and self.decode_prefix_len:
            return False
        return self.dst_kv_indices.size == 0

    @classmethod
    def from_zmq(cls, msg: List[bytes]):
        dst_state_indices = (
            unpack_int_lists(msg[7], "i") if len(msg) > 7 and msg[7] != b"" else []
        )

        return cls(
            room=int(msg[0].decode("ascii")),
            endpoint=msg[1].decode("ascii"),
            dst_port=int(msg[2].decode("ascii")),
            agent_name=msg[3].decode("ascii"),
            dst_kv_indices=np.frombuffer(msg[4], dtype=np.int32),
            dst_aux_index=int(msg[5].decode("ascii")),
            required_dst_info_num=int(msg[6].decode("ascii")),
            dst_state_indices=dst_state_indices,
            decode_prefix_len=(
                int(msg[8].decode("ascii")) if len(msg) > 8 and msg[8] != b"" else None
            ),  # hacky just add it into the message that will be sent
        )


@dataclasses.dataclass
class TransferKVChunk:
    room: int
    prefill_kv_indices: npt.NDArray[np.int32]
    index_slice: slice
    is_last: bool
    chunk_id: int
    prefill_aux_index: Optional[int]
    state_indices: Optional[List]


@dataclasses.dataclass
class KVArgsRegisterInfo:
    """Contains base pointers and other info which only needs to be sent once by KVReceiver. Received by prefill bootstrap thread."""

    room: str
    endpoint: str
    dst_port: int
    agent_name: str
    agent_metadata: bytes
    dst_kv_ptrs: list[int]
    dst_aux_ptrs: list[int]
    dst_state_data_ptrs: List[List[int]]
    gpu_id: int
    decode_tp_size: int
    decode_tp_rank: int
    dst_kv_item_len: int
    dst_state_item_lens: List[List[int]] = dataclasses.field(default_factory=list)
    dst_state_dim_per_tensor: List[List[int]] = dataclasses.field(default_factory=list)
    # Keep last: optional, parsed from a variable-length tail of the ZMQ
    # frame in from_zmq() below, so positional construction stays stable.
    staging: Optional["StagingRegisterInfo"] = None

    @classmethod
    def from_zmq(cls, msg: List[bytes]):
        dst_state_data_ptrs = (
            unpack_int_lists(msg[7], "Q") if len(msg) > 7 and msg[7] != b"" else []
        )
        dst_state_item_lens = (
            unpack_int_lists(msg[12], "I") if len(msg) > 12 and len(msg[12]) > 0 else []
        )
        dst_state_dim_per_tensor = (
            unpack_int_lists(msg[13], "I") if len(msg) > 13 and len(msg[13]) > 0 else []
        )

        return cls(
            room=str(msg[0].decode("ascii")),
            endpoint=msg[1].decode("ascii"),
            dst_port=int(msg[2].decode("ascii")),
            agent_name=msg[3].decode("ascii"),
            agent_metadata=msg[4],
            dst_kv_ptrs=list(struct.unpack(f"{len(msg[5]) // 8}Q", msg[5])),
            dst_aux_ptrs=list(struct.unpack(f"{len(msg[6]) // 8}Q", msg[6])),
            dst_state_data_ptrs=dst_state_data_ptrs,
            gpu_id=int(msg[8].decode("ascii")),
            decode_tp_size=int(msg[9].decode("ascii")),
            decode_tp_rank=int(msg[10].decode("ascii")),
            dst_kv_item_len=int(msg[11].decode("ascii")),
            dst_state_item_lens=dst_state_item_lens,
            dst_state_dim_per_tensor=dst_state_dim_per_tensor,
            staging=StagingRegisterInfo.from_zmq_fields(msg, 14),
        )


@dataclasses.dataclass
class TransferStatus:
    """Used by KV Receiver to know when a transfer is done."""

    # KV chunks received per pp_rank: {pp_rank: set of chunk_ids}
    received_kvs_per_pp: Dict[int, Set[int]] = dataclasses.field(
        default_factory=lambda: defaultdict(set)
    )
    # Expected chunk count per pp_rank (set when is_last=True): {pp_rank: expected_count}
    expected_kvs_per_pp: Dict[int, int] = dataclasses.field(default_factory=dict)
    # Number of PP ranks expected to send data.
    num_pp_ranks_expected: Optional[int] = None
    # Whether aux data has been received.
    received_aux: bool = False
    # PP ranks that have sent state data (state is layer-specific, each PP rank sends its portion).
    received_state_per_pp: Set[int] = dataclasses.field(default_factory=set)
    # Whether state data is expected (set based on state_type).
    expects_state: bool = False
    # Mark as failed
    is_failure: bool = False

    def is_done(self):
        if self.is_failure:
            return True
        if self.num_pp_ranks_expected is None or not self.received_aux:
            return False
        # If state data is expected, check all PP ranks have sent it
        if (
            self.expects_state
            and len(self.received_state_per_pp) < self.num_pp_ranks_expected
        ):
            return False
        # All PP ranks must have reported their expected count
        if len(self.expected_kvs_per_pp) < self.num_pp_ranks_expected:
            return False
        # Each PP rank must have received all expected chunks
        for pp_rank, expected in self.expected_kvs_per_pp.items():
            if len(self.received_kvs_per_pp[pp_rank]) != expected:
                return False
        return True

    def is_failed(self):
        return self.is_failure


class NixlKVManager(CommonKVManager):
    def __init__(
        self,
        args: KVArgs,
        disaggregation_mode: DisaggregationMode,
        server_args: ServerArgs,
        is_mla_backend: Optional[bool] = False,
    ):
        super().__init__(args, disaggregation_mode, server_args, is_mla_backend)
        try:
            from nixl._api import nixl_agent, nixl_agent_config
        except ImportError as e:
            raise ImportError(
                "Please install NIXL by following the instructions at "
                "https://github.com/ai-dynamo/nixl/blob/main/README.md "
                "to run SGLang with NixlTransferEngine."
            ) from e

        backend = envs.SGLANG_DISAGGREGATION_NIXL_BACKEND.get()
        num_threads = 8 if disaggregation_mode == DisaggregationMode.PREFILL else 0
        backend_params = json.loads(
            envs.SGLANG_DISAGGREGATION_NIXL_BACKEND_PARAMS.get()
        )
        if not isinstance(backend_params, dict) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in backend_params.items()
        ):
            raise ValueError(
                "SGLANG_DISAGGREGATION_NIXL_BACKEND_PARAMS must be a JSON object "
                "with string keys and string values"
            )
        agent_config = nixl_agent_config(backends=[], num_threads=num_threads)
        self.agent = nixl_agent(str(uuid.uuid4()), agent_config)
        if num_threads > 0:
            # TODO: Remove this once NIXL passes thread parameters from
            # nixl_agent_config to explicitly-created backends.
            if backend == "UCX" or backend == "OBJ":
                backend_params.setdefault("num_threads", str(num_threads))
            elif backend == "GDS_MT":
                backend_params.setdefault("thread_count", str(num_threads))
            elif backend == "UCCL":
                backend_params.setdefault("num_cpus", str(num_threads))
        self.agent.create_backend(backend, backend_params)

        available_plugins = self.agent.get_plugin_list()
        if backend not in available_plugins:
            raise ValueError(
                f"NIXL backend '{backend}' not found. Available: {available_plugins}. "
                f"Please install the required NIXL plugin or choose from: {available_plugins}"
            )
        logger.info(f"NIXL KVManager initialized with backend: {backend}")

        self.register_buffer_to_engine()

        self.enable_staging = envs.SGLANG_DISAGG_STAGING_BUFFER.get()
        self.kv_buffer_tensors = None

        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            transfer_queue_size = envs.SGLANG_DISAGGREGATION_QUEUE_SIZE.get()
            self.transfer_queues: List[FastQueue] = [
                FastQueue() for _ in range(transfer_queue_size)
            ]
            self.exceptions: Dict[int, Exception] = {}
            # Mirror mooncake: one staging buffer per worker queue, all
            # built before workers spawn so each worker owns a private
            # buffer (no cross-worker contention on the staging ring).
            if self.enable_staging:
                self._init_staging_prefill_ctx()
                self._init_staging_buffers(len(self.transfer_queues))
            for i, queue in enumerate(self.transfer_queues):
                staging_buffer = (
                    self._staging_ctx.buffers[i]
                    if self.enable_staging and self._staging_ctx.buffers
                    else None
                )
                threading.Thread(
                    target=self.transfer_worker,
                    args=(queue, staging_buffer),
                    daemon=True,
                ).start()
            self._start_bootstrap_thread()
        elif self.disaggregation_mode == DisaggregationMode.DECODE:
            self.transfer_statuses: Dict[int, TransferStatus] = defaultdict(
                TransferStatus
            )
            if self.enable_staging:
                self._init_staging_decode_ctx()
                self._staging_handler = None
                self._chunk_writer_counts: dict = defaultdict(lambda: defaultdict(list))
                self._start_decode_staging_thread()
            self._start_heartbeat_checker_thread()
        else:
            raise ValueError(
                f"Unsupported DisaggregationMode: {self.disaggregation_mode}"
            )

    def _init_staging_prefill_ctx(self):
        from sglang.srt.disaggregation.common.staging_handler import (
            PrefillStagingContext,
        )

        self._staging_ctx = PrefillStagingContext()

    def _init_staging_decode_ctx(self):
        from sglang.srt.disaggregation.common.staging_handler import (
            DecodeStagingContext,
        )

        self._staging_ctx = DecodeStagingContext()
        self._init_staging_allocator()

    def _init_staging_buffers(self, count: int):
        from sglang.srt.disaggregation.common.staging_handler import (
            init_staging_buffers,
        )

        gpu_id = self.kv_args.gpu_id
        self._staging_ctx.buffers = init_staging_buffers(
            lambda ptr, size: self._register_staging_memory(ptr, size, gpu_id),
            self.kv_args,
            count,
        )

    def _init_staging_allocator(self):
        from sglang.srt.disaggregation.common.staging_handler import (
            init_staging_allocator,
        )

        gpu_id = self.kv_args.gpu_id
        self._staging_ctx.allocator = init_staging_allocator(
            lambda ptr, size: self._register_staging_memory(ptr, size, gpu_id),
            self.kv_args,
        )

    def _register_staging_memory(self, ptr: int, size: int, gpu_id: int):
        """Register a staging buffer with the NIXL agent."""
        addrs = [(ptr, size, gpu_id, "")]
        descs = self.agent.register_memory(addrs, "VRAM")
        if not descs:
            raise RuntimeError(
                f"NIXL memory registration failed for staging buffer "
                f"(ptr=0x{ptr:x}, size={size})"
            )

    def set_kv_buffer_tensors(self, k_buffers: list, v_buffers: list, page_size: int):
        # NOTE: matches mooncake behavior -- staging buffers are now
        # created in __init__ (per-worker), independent of the kv
        # tensors. This setter only stashes the tensor metadata used by
        # send_kvcache_staged().
        self.kv_buffer_tensors = {
            "k_buffers": k_buffers,
            "v_buffers": v_buffers,
            "page_size": page_size,
        }

    def register_staging_room_bootstrap(self, room, bootstrap_infos, receiver):
        self._staging_ctx.room_bootstrap[room] = bootstrap_infos
        self._staging_ctx.room_receivers[room] = receiver

    def _is_watermark_ready(
        self, agent_name: str, alloc_round: int, alloc_end: int
    ) -> bool:
        from sglang.srt.disaggregation.common.staging_handler import (
            is_watermark_ready,
        )

        return is_watermark_ready(self._staging_ctx, agent_name, alloc_round, alloc_end)

    def _start_decode_staging_thread(self):
        """Start a thread on the decode side to recv STAGING_REQ from prefill via ZMQ."""

        def decode_staging_thread():
            while True:
                msg = self.server_socket.recv_multipart()
                if msg[0] == b"STAGING_REQ":
                    self._handle_staging_req(msg)
                    continue
                logger.warning(
                    "decode_staging_thread: unexpected message tag %s",
                    msg[0][:20],
                )

        threading.Thread(target=decode_staging_thread, daemon=True).start()

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

    def _prefetch_staging_reqs(self, room: int):
        """Send STAGING_REQ for all chunks before the prefill forward starts.

        Idempotent per room: the first call for a given room does the full
        fan-out (one STAGING_REQ per chunk per peer); subsequent calls return
        immediately. This lets the caller invoke this on every chunk without
        depending on a chunk_id == 0 sentinel.
        """
        if not self.enable_staging or self.kv_buffer_tensors is None:
            return
        if room in self._staging_ctx.prefetched_rooms:
            return

        room_infos = self.transfer_infos.get(room, {})
        needs_staging = any(
            not tinfo.is_dummy()
            and tinfo.agent_name in self.decode_kv_args_table
            and self.decode_kv_args_table[tinfo.agent_name].decode_tp_size
            != self.attn_tp_size
            for tinfo in room_infos.values()
        )
        if not needs_staging:
            # Mark anyway so we don't re-evaluate the predicate every chunk.
            self._staging_ctx.prefetched_rooms.add(room)
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
        self._staging_ctx.prefetched_rooms.add(room)

    def _start_heartbeat_checker_thread(self):
        """
        Start the heartbeat checker thread for Decode worker.
        TODO (smor): unite nixl heartbeat checker with mooncake's.
        """

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

        threading.Thread(target=heartbeat_checker, daemon=True).start()

    def _handle_node_failure(self, failed_bootstrap_addr):
        """Handle failure of a prefill node."""
        with self.connection_lock:
            keys_to_remove = [
                k for k in self.connection_pool if k.startswith(failed_bootstrap_addr)
            ]
            for k in keys_to_remove:
                del self.connection_pool[k]
            self.prefill_info_table.pop(failed_bootstrap_addr, None)

            possible_affected_rooms = self.addr_to_rooms_tracker.get(
                failed_bootstrap_addr, []
            )
            self.addr_to_rooms_tracker.pop(failed_bootstrap_addr, None)

        # Mark all pending transfers associated with the failed node as failed
        affected_rooms = []
        for room in possible_affected_rooms:
            if (
                room in self.transfer_statuses
                and not self.transfer_statuses[room].is_done()
            ):
                # Mark the transfer as failed
                self.transfer_statuses[room].is_failure = True
                affected_rooms.append(room)

        logger.error(
            f"Lost connection with prefill instance (bootstrap_addr: {failed_bootstrap_addr}), "
            f"{len(affected_rooms)} transfers affected"
        )
        for room in possible_affected_rooms:
            logger.error(f"Let room {room} be failed due to prefill down")
            self.update_status(room, KVPoll.Failed)

    def check_status(self, bootstrap_room: int):
        return self.request_status.get(bootstrap_room, KVPoll.WaitingForInput)

    def transfer_worker(self, queue: FastQueue, staging_buffer=None):
        # Per-worker staging strategy: lazy-created on first chunk so we
        # see kv_buffer_tensors (set by ModelRunner after engine init).
        # Never cache on self -- multiple workers would race the ring.
        staging_strategy = None

        while True:
            kv_chunk: TransferKVChunk = queue.get()
            room = kv_chunk.room
            try:
                if self.check_status(room) == KVPoll.Failed:
                    continue

                assert room in self.transfer_infos

                # Lazily build a per-worker staging strategy bound to this
                # worker's private staging buffer (matches mooncake).
                if (
                    self.enable_staging
                    and staging_strategy is None
                    and staging_buffer is not None
                ):
                    staging_strategy = self._try_create_staging_strategy(staging_buffer)

                self.update_status(room, KVPoll.Transferring)

                reqs_to_be_processed = list(self.transfer_infos[room].values())
                handles: List = []

                # Set when staging allocation/watermark is not yet ready and
                # the chunk has been re-enqueued. We then break out of the
                # per-req loop and `continue` the worker main loop without
                # touching room status -- the next pop will retry.
                staging_deferred = False

                for req in reqs_to_be_processed:
                    assert room == req.room
                    if req.is_dummy():
                        continue

                    assert req.agent_name in self.decode_kv_args_table
                    dst_info = self.decode_kv_args_table[req.agent_name]
                    decode_tp_size = dst_info.decode_tp_size

                    # Skip KV RDMA transfer when there are no pages to send
                    # (e.g., decode-side radix cache matched the entire prefix).
                    # Aux data is still sent below when is_last=True.
                    if len(kv_chunk.prefill_kv_indices) > 0:
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

                        # Decide which kv send path to use:
                        #   1. Staging (heterogeneous TP, both sides have
                        #      registered staging, watermark/alloc ready)
                        #   2. send_kvcache (MLA or homogeneous TP)
                        #   3. send_kvcache_slice (heterogeneous TP fallback,
                        #      or staging hard-failed for this chunk)
                        use_staging = (
                            self.enable_staging
                            and staging_strategy is not None
                            and not self.is_mla_backend
                            and decode_tp_size != self.attn_tp_size
                            and dst_info.staging is not None
                        )

                        kv_xfer_handle = None
                        if use_staging:
                            kv_xfer_handle, deferred = self._do_staging_transfer(
                                staging_strategy,
                                kv_chunk,
                                req,
                                dst_info,
                                queue,
                            )
                            if deferred:
                                # Chunk re-enqueued; stop processing remaining
                                # reqs for this chunk and let the worker loop
                                # pick it up again on the next pop.
                                staging_deferred = True
                                break
                            # kv_xfer_handle is None here means staging
                            # send_kvcache_staged() returned None (e.g.
                            # decode buffer too small) -- fall through to
                            # the slice path below.

                        if kv_xfer_handle is None:
                            notif = (
                                f"{req.room}_kv_{kv_chunk.chunk_id}"
                                f"_{int(kv_chunk.is_last)}_{self.kv_args.engine_rank}"
                            )
                            if self.is_mla_backend or (
                                decode_tp_size == self.attn_tp_size
                            ):
                                kv_xfer_handle = self.send_kvcache(
                                    req.agent_name,
                                    kv_chunk.prefill_kv_indices,
                                    dst_info.dst_kv_ptrs,
                                    chunked_dst_kv_indice,
                                    dst_info.gpu_id,
                                    notif,
                                )
                            else:
                                kv_xfer_handle = self.send_kvcache_slice(
                                    req.agent_name,
                                    kv_chunk.prefill_kv_indices,
                                    dst_info.dst_kv_ptrs,
                                    chunked_dst_kv_indice,
                                    dst_info.gpu_id,
                                    notif,
                                    prefill_tp_size=self.attn_tp_size,
                                    decode_tp_size=decode_tp_size,
                                    decode_tp_rank=dst_info.decode_tp_rank,
                                    dst_kv_item_len=dst_info.dst_kv_item_len,
                                )

                        handles.append(kv_xfer_handle)

                    if kv_chunk.is_last and kv_chunk.state_indices:
                        dst_info = self.decode_kv_args_table[req.agent_name]
                        state_xfer_handles = self.maybe_send_extra(
                            req.agent_name,
                            kv_chunk.state_indices,
                            dst_info.dst_state_data_ptrs,
                            req.dst_state_indices,
                            dst_info.gpu_id,
                            f"{req.room}_state_{self.kv_args.engine_rank}",
                            decode_tp_size,
                            decode_tp_rank=dst_info.decode_tp_rank,
                            dst_state_item_lens=dst_info.dst_state_item_lens,
                            dst_state_dim_per_tensor=dst_info.dst_state_dim_per_tensor,
                        )
                        handles.extend(h for h in state_xfer_handles if h is not None)

                        if kv_chunk.prefill_aux_index is None:
                            raise RuntimeError("Missing aux index for last chunk")
                        # When no KV pages were sent (decode-side cache hit),
                        # encode pp_rank in aux notif so receiver can mark
                        # expected_kvs_per_pp[pp_rank] = 0.
                        if len(kv_chunk.prefill_kv_indices) == 0:
                            aux_notif = (
                                f"{req.room}_aux_nokv_{self.kv_args.engine_rank}"
                            )
                        else:
                            aux_notif = f"{req.room}_aux"
                        aux_xfer_handle = self.send_aux(
                            req.agent_name,
                            kv_chunk.prefill_aux_index,
                            dst_info.dst_aux_ptrs,
                            req.dst_aux_index,
                            aux_notif,
                        )
                        handles.append(aux_xfer_handle)

                if staging_deferred:
                    # Chunk has been re-enqueued; do not advance status.
                    continue

                while handles:
                    states = [self.agent.check_xfer_state(h) for h in handles]
                    if any(s == "ERR" for s in states):
                        raise RuntimeError(f"NIXL transfer encountered ERR room={room}")
                    if all(s == "DONE" for s in states):
                        break
                    time.sleep(0)

                if kv_chunk.is_last:
                    self.update_status(room, KVPoll.Success)
                    # Drop per-room state on Success (parity with mooncake
                    # transfer_worker; staging prefetch sets are NIXL-only).
                    self.transfer_infos.pop(room, None)
                    self.req_to_decode_prefix_len.pop(room, None)
                    if self.enable_staging and self._staging_ctx is not None:
                        self._staging_ctx.prefetched_rooms.discard(room)
                        self._staging_ctx.prefetch_requested = {
                            k
                            for k in self._staging_ctx.prefetch_requested
                            if k[0] != room
                        }
                else:
                    self.update_status(room, KVPoll.Transferring)
            except Exception as e:
                # Catch all exceptions to prevent silently killing this
                # worker thread, but still propagate via failure_exception().
                if isinstance(e, _NIXL_TRANSPORT_ERRORS):
                    logger.warning(f"NIXL transport error for room {room}: {e}")
                else:
                    logger.exception(
                        f"Unexpected transfer worker error for room {room}"
                    )
                self.exceptions[room] = e
                self.record_failure(room, str(e))
                self.update_status(room, KVPoll.Failed)

    def register_buffer_to_engine(self):
        kv_addrs = []
        for kv_data_ptr, kv_data_len in zip(
            self.kv_args.kv_data_ptrs, self.kv_args.kv_data_lens
        ):
            kv_addrs.append((kv_data_ptr, kv_data_len, self.kv_args.gpu_id, ""))
        self.kv_descs = self.agent.register_memory(kv_addrs, "VRAM")
        logger.debug(f"Register kv tensors, len(kv_addr)= {len(kv_addrs)}")
        if not self.kv_descs:
            raise Exception("NIXL memory registration failed for kv tensors")
        aux_addrs = []
        for aux_data_ptr, aux_data_len in zip(
            self.kv_args.aux_data_ptrs, self.kv_args.aux_data_lens
        ):
            aux_addrs.append((aux_data_ptr, aux_data_len, 0, ""))
        self.aux_descs = self.agent.register_memory(aux_addrs, "DRAM")
        logger.debug(f"Register aux tensors, len(aux_addrs)= {len(aux_addrs)}")
        if not self.aux_descs:
            raise Exception("NIXL memory registration failed for aux tensors")

        state_addrs = []
        for comp_ptrs, comp_lens in zip(
            self.kv_args.state_data_ptrs or [],
            self.kv_args.state_data_lens or [],
        ):
            for state_data_ptr, state_data_len in zip(comp_ptrs, comp_lens):
                if state_data_ptr == 0 or state_data_len == 0:
                    continue
                state_addrs.append(
                    (state_data_ptr, state_data_len, self.kv_args.gpu_id, "")
                )
        if state_addrs:
            self.state_descs = self.agent.register_memory(state_addrs, "VRAM")
            logger.debug(
                f"Register state tensors, len(state_addrs)= {len(state_addrs)}"
            )
            if not self.state_descs:
                raise Exception("NIXL memory registration failed for state tensors")

    def _add_remote_peer(self, decode_kv_args: KVArgsRegisterInfo):
        agent_name = decode_kv_args.agent_name
        if agent_name in self.decode_kv_args_table:
            logger.info(f"Peer {agent_name} was already registered, ignoring.")
            return
        self.decode_kv_args_table[agent_name] = decode_kv_args
        self.agent.add_remote_agent(decode_kv_args.agent_metadata)

    def _send_kvcache_generic(
        self,
        peer_name: str,
        src_data_ptrs: list[int],
        dst_data_ptrs: list[int],
        item_lens: list[int],
        prefill_data_indices: npt.NDArray[np.int32],
        dst_data_indices: npt.NDArray[np.int32],
        dst_gpu_id: int,
        notif: str,
    ):
        """Generic KV cache transfer supporting both MHA and MLA architectures.
        Used by both send_kvcache and maybe_send_extra."""
        # Convert pointer lists to np.uint64 arrays up front.
        # torch.int exceeds np.int64 range on Intel XPU (addresses have bit 63 set, e.g.
        # 0xffff81ab54e01000). Casting here prevents overflow when these values
        # are later used in numpy arithmetic.
        src_data_ptrs = np.array(src_data_ptrs, dtype=np.uint64)
        dst_data_ptrs = np.array(dst_data_ptrs, dtype=np.uint64)
        item_lens = np.array(item_lens, dtype=np.uint64)

        # group by indices
        prefill_kv_blocks, dst_kv_blocks = group_concurrent_contiguous(
            prefill_data_indices, dst_data_indices
        )

        logger.debug(f"sending kvcache to {peer_name} with notif {notif}")
        # Make descs
        if self.is_mla_backend:
            src_kv_ptrs, dst_kv_ptrs, layers_current_pp_stage = (
                self.get_mla_kv_ptrs_with_pp(src_data_ptrs, dst_data_ptrs)
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

            layers_params = [
                (
                    src_k_ptrs[layer_id],
                    dst_k_ptrs[layer_id],
                    item_lens[layer_id],
                )
                for layer_id in range(layers_current_pp_stage)
            ] + [
                (
                    src_v_ptrs[layer_id],
                    dst_v_ptrs[layer_id],
                    item_lens[layer_id],
                )
                for layer_id in range(layers_current_pp_stage)
            ]

        src_addrs = []
        src_lens = []
        dst_addrs = []
        dst_lens = []

        # Precompute block starts/lengths to reduce Python-level loops.
        prefill_starts = np.fromiter(
            (block[0] for block in prefill_kv_blocks), dtype=np.uint64
        )
        dst_starts = np.fromiter((block[0] for block in dst_kv_blocks), dtype=np.uint64)
        block_lens = np.fromiter(
            (len(block) for block in prefill_kv_blocks), dtype=np.uint64
        )

        for src_ptr, dst_ptr, item_len in layers_params:
            lengths = item_len * block_lens
            src_addrs.append(src_ptr + prefill_starts * item_len)
            src_lens.append(lengths)
            dst_addrs.append(dst_ptr + dst_starts * item_len)
            dst_lens.append(lengths)

        def make_req_array(addr_chunks, len_chunks, gpu):
            if not addr_chunks:
                return np.empty((0, 3), dtype=np.uint64)
            flat_addrs = np.concatenate(addr_chunks).astype(np.uint64, copy=False)
            flat_lens = np.concatenate(len_chunks).astype(np.uint64, copy=False)
            return np.column_stack(
                (
                    flat_addrs,
                    flat_lens,
                    np.full_like(flat_addrs, gpu, dtype=np.uint64),
                )
            )

        src_reqs = make_req_array(src_addrs, src_lens, self.kv_args.gpu_id)
        dst_reqs = make_req_array(dst_addrs, dst_lens, dst_gpu_id)

        logger.debug(
            f"len(src_addrs): before group: {len(prefill_data_indices)}, after group: {len(src_addrs)}"
        )
        src_descs = self.agent.get_xfer_descs(src_reqs, "VRAM")
        dst_descs = self.agent.get_xfer_descs(dst_reqs, "VRAM")
        # Transfer data
        xfer_handle = self.agent.initialize_xfer(
            "WRITE",
            src_descs,
            dst_descs,
            peer_name,
            notif.encode("ascii"),  # type: ignore
        )
        if not xfer_handle:
            raise Exception("KVSender failed to create transfer")
        state = self.agent.transfer(xfer_handle)
        if state == "ERR":
            raise Exception("KVSender failed to post transfer")
        return xfer_handle

    def send_kvcache(
        self,
        peer_name: str,
        prefill_kv_indices: npt.NDArray[np.int32],
        dst_kv_ptrs: list[int],
        dst_kv_indices: npt.NDArray[np.int32],
        dst_gpu_id: int,
        notif: str,
    ):
        return self._send_kvcache_generic(
            peer_name=peer_name,
            src_data_ptrs=self.kv_args.kv_data_ptrs,
            dst_data_ptrs=dst_kv_ptrs,
            item_lens=self.kv_args.kv_item_lens,
            prefill_data_indices=prefill_kv_indices,
            dst_data_indices=dst_kv_indices,
            dst_gpu_id=dst_gpu_id,
            notif=notif,
        )

    def send_kvcache_slice(
        self,
        peer_name: str,
        prefill_kv_indices: npt.NDArray[np.int32],
        dst_kv_ptrs: list[int],
        dst_kv_indices: npt.NDArray[np.int32],
        dst_gpu_id: int,
        notif: str,
        prefill_tp_size: int,
        decode_tp_size: int,
        decode_tp_rank: int,
        dst_kv_item_len: int,
    ):
        # Get configuration from kv_args
        local_tp_rank_in_group = self.kv_args.engine_rank % prefill_tp_size
        dst_tp_rank_in_group = decode_tp_rank % decode_tp_size

        src_kv_item_len = self.kv_args.kv_item_lens[0]
        page_size = self.kv_args.page_size

        # Use total KV head count (not per-rank) for correct head distribution.
        # Per-rank kv_head_num is max(1, total//tp) which loses info when total < tp.
        total_kv_heads = getattr(self.kv_args, "total_kv_head_num", 0)
        if total_kv_heads <= 0:
            total_kv_heads = self.kv_args.kv_head_num * prefill_tp_size

        src_heads_per_rank = max(1, total_kv_heads // prefill_tp_size)
        dst_heads_per_rank = max(1, total_kv_heads // decode_tp_size)

        bytes_per_head_slice_to_send = (
            dst_kv_item_len // page_size // dst_heads_per_rank
        )

        # GQA replication: how many prefill ranks share the same KV head
        src_replication = max(1, prefill_tp_size // total_kv_heads)

        # Determine which heads to send
        if prefill_tp_size > decode_tp_size:
            # Multiple prefill ranks to one decode rank
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

        # torch.int exceeds np.int64 range on Intel XPU (addresses have bit 63 set, e.g.
        # 0xffff81ab54e01000). Use np.uint64 to prevent overflow on XPU.
        kv_data_ptrs = np.array(self.kv_args.kv_data_ptrs, dtype=np.uint64)
        dst_kv_ptrs = np.array(dst_kv_ptrs, dtype=np.uint64)
        src_k_ptrs, src_v_ptrs, dst_k_ptrs, dst_v_ptrs, layers_current_pp_stage = (
            self.get_mha_kv_ptrs_with_pp(kv_data_ptrs, dst_kv_ptrs)
        )
        # Calculate precise byte offset and length for the sub-slice within the token
        src_head_slice_offset = src_head_start_offset * bytes_per_head_slice_to_send
        dst_head_slice_offset = dst_head_start_offset * bytes_per_head_slice_to_send
        heads_bytes_per_token_to_send = num_heads_to_send * bytes_per_head_slice_to_send

        src_dst_ptr_pairs = [
            (
                src_k_ptrs[layer_id],
                dst_k_ptrs[layer_id],
            )
            for layer_id in range(layers_current_pp_stage)
        ] + [
            (
                src_v_ptrs[layer_id],
                dst_v_ptrs[layer_id],
            )
            for layer_id in range(layers_current_pp_stage)
        ]

        prefill_indices = np.asarray(prefill_kv_indices, dtype=np.uint64)
        dst_indices = np.asarray(dst_kv_indices, dtype=np.uint64)
        bytes_per_token_prefill = src_kv_item_len // page_size
        bytes_per_token_decode = dst_kv_item_len // page_size
        token_offsets = np.arange(page_size, dtype=np.uint64)

        src_addrs = []
        dst_addrs = []

        for src_ptr, dst_ptr in src_dst_ptr_pairs:
            src_page_bases = src_ptr + prefill_indices * src_kv_item_len
            dst_page_bases = dst_ptr + dst_indices * dst_kv_item_len

            src_all = (
                src_page_bases[:, None]
                + token_offsets[None, :] * bytes_per_token_prefill
                + src_head_slice_offset
            ).ravel()
            dst_all = (
                dst_page_bases[:, None]
                + token_offsets[None, :] * bytes_per_token_decode
                + dst_head_slice_offset
            ).ravel()

            src_addrs.append(src_all)
            dst_addrs.append(dst_all)

        def make_req_array(addr_chunks, size, gpu):
            if not addr_chunks:
                return np.empty((0, 3), dtype=np.uint64)
            flat_addrs = np.concatenate(addr_chunks).astype(np.uint64, copy=False)
            return np.column_stack(
                (
                    flat_addrs,
                    np.full_like(flat_addrs, size, dtype=np.uint64),
                    np.full_like(flat_addrs, gpu, dtype=np.uint64),
                )
            )

        src_reqs = make_req_array(
            src_addrs, heads_bytes_per_token_to_send, self.kv_args.gpu_id
        )
        dst_reqs = make_req_array(dst_addrs, heads_bytes_per_token_to_send, dst_gpu_id)

        # Use NIXL agent for transfer
        src_descs = self.agent.get_xfer_descs(src_reqs, "VRAM")
        dst_descs = self.agent.get_xfer_descs(dst_reqs, "VRAM")

        xfer_handle = self.agent.initialize_xfer(
            "WRITE", src_descs, dst_descs, peer_name, notif.encode("ascii")
        )
        if not xfer_handle:
            raise Exception("Failed to create sliced KV transfer")

        state = self.agent.transfer(xfer_handle)
        if state == "ERR":
            raise Exception("Failed to post sliced KV transfer")

        return xfer_handle

    def send_kvcache_staged(
        self,
        peer_name: str,
        prefill_kv_indices: npt.NDArray[np.int32],
        dst_staging_ptr: int,
        dst_staging_size: int,
        dst_gpu_id: int,
        dst_tp_rank: int,
        dst_attn_tp_size: int,
        dst_kv_item_len: int,
        notif: str,
        staging_buffer=None,
    ):
        """Transfer KV cache via staging buffers (gather -> bulk RDMA -> scatter on decode)."""
        from sglang.srt.disaggregation.common.staging_buffer import (
            compute_head_slice_params,
            compute_staging_layout,
            gather_all_layers_to_staging,
            resolve_total_kv_heads,
        )

        if self.kv_buffer_tensors is None or staging_buffer is None:
            return None

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
            return None
        if dst_staging_size < total_staging_needed:
            logger.warning(
                f"Decode staging too small: need {total_staging_needed} bytes, "
                f"have {dst_staging_size}, falling back"
            )
            return None

        # gather_all_layers_to_staging() runs the gather kernel on its own
        # dedicated stream and synchronizes that stream before returning, so
        # the staging buffer is fully populated and visible to the NIC by the
        # time we post the RDMA WRITE below. No extra sync needed (matches
        # mooncake's send_kvcache_staged behavior).
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
        src_reqs = np.array(
            [[staging_buffer.get_ptr(), per_rank_bytes, self.kv_args.gpu_id]],
            dtype=np.int64,
        )
        dst_reqs = np.array(
            [[dst_write_ptr, per_rank_bytes, dst_gpu_id]], dtype=np.int64
        )

        src_descs = self.agent.get_xfer_descs(src_reqs, "VRAM")
        dst_descs = self.agent.get_xfer_descs(dst_reqs, "VRAM")

        xfer_handle = self.agent.initialize_xfer(
            "WRITE", src_descs, dst_descs, peer_name, notif.encode("ascii")
        )
        if not xfer_handle:
            raise RuntimeError(
                f"[Staging] Failed to create NIXL bulk transfer "
                f"(src=0x{staging_buffer.get_ptr():x}, dst=0x{dst_write_ptr:x}, "
                f"size={per_rank_bytes})"
            )
        state = self.agent.transfer(xfer_handle)
        if state == "ERR":
            raise RuntimeError("[Staging] NIXL bulk transfer failed to post")
        return xfer_handle

    def _try_create_staging_strategy(self, staging_buffer):
        """Create a per-worker PrefillStagingStrategy bound to ``staging_buffer``.

        Returns ``None`` if staging is disabled or kv tensors not yet set.
        Caller is expected to keep the returned strategy as a worker-local
        variable; never cache on ``self`` (multiple workers would race on
        the underlying staging ring buffer).
        """
        if not self.enable_staging or self.kv_buffer_tensors is None:
            return None
        from sglang.srt.disaggregation.common.staging_handler import (
            PrefillStagingStrategy,
        )

        return PrefillStagingStrategy(self, staging_buffer)

    def _do_staging_transfer(
        self,
        staging_strategy,
        kv_chunk: "TransferKVChunk",
        req: "TransferInfo",
        dst_info: "KVArgsRegisterInfo",
        queue: FastQueue,
    ):
        """Attempt staging transfer for one chunk. Returns (xfer_handle, deferred).

        Mirrors mooncake._do_staging_transfer semantics:
          - staging not ready (watermark/alloc pending) -> ``queue.put(kv_chunk)``
            re-enqueue the chunk and return ``(None, True)``. Caller should
            ``break`` out of the per-req loop and ``continue`` the worker
            main loop without updating room status -- the chunk will be
            retried on the next pop.
          - oversized chunk (will never fit) -> raise RuntimeError.
          - staging successfully posted -> return ``(handle, False)``. The
            caller appends the handle to the per-chunk handle list and
            busy-polls it to DONE alongside other handles.
          - send_kvcache_staged returned None (decode buffer too small,
            kv_buffer_tensors missing, etc.) -> return ``(None, False)``,
            signalling the caller to fall back to send_kvcache_slice.
        """
        page_start = kv_chunk.index_slice.start
        num_pages = len(kv_chunk.prefill_kv_indices)

        ready, chunk_idx, c_offset, _, _ = staging_strategy.check_ready(
            req, page_start, num_pages, session_id=req.agent_name
        )
        if not ready:
            from sglang.srt.disaggregation.common.staging_buffer import (
                StagingAllocator,
            )

            if c_offset == StagingAllocator.ALLOC_OVERSIZED:
                raise RuntimeError(
                    f"[Staging] Chunk staging allocation permanently failed: "
                    f"chunk exceeds ring buffer total size "
                    f"(room={kv_chunk.room}). Increase "
                    f"SGLANG_DISAGG_STAGING_POOL_SIZE_MB."
                )
            queue.put(kv_chunk)
            return (None, True)

        notif_tag = (
            f"{req.room}_stg_{kv_chunk.chunk_id}_{int(kv_chunk.is_last)}"
            f"_{self.kv_args.engine_rank}_{chunk_idx}"
            f"_{page_start}_{num_pages}_{req.agent_name}"
        )
        handle = self.send_kvcache_staged(
            req.agent_name,
            kv_chunk.prefill_kv_indices,
            dst_info.staging.base_ptr + c_offset,
            dst_info.staging.total_size - c_offset,
            dst_info.gpu_id,
            dst_info.decode_tp_rank,
            dst_info.decode_tp_size,
            dst_info.dst_kv_item_len,
            notif_tag,
            staging_buffer=staging_strategy.staging_buffer,
        )
        return (handle, False)

    def send_aux(
        self,
        peer_name: str,
        prefill_aux_index: int,
        dst_aux_ptrs: list[int],
        dst_aux_index: int,
        notif: str,
    ):
        src_addrs = []
        dst_addrs = []

        prefill_aux_ptrs = self.kv_args.aux_data_ptrs
        prefill_aux_item_lens = self.kv_args.aux_item_lens

        for i, _ in enumerate(dst_aux_ptrs):
            length = prefill_aux_item_lens[i]
            src_addr = prefill_aux_ptrs[i] + length * prefill_aux_index
            dst_addr = dst_aux_ptrs[i] + length * dst_aux_index
            src_addrs.append((src_addr, length, 0))
            dst_addrs.append((dst_addr, length, 0))

        src_descs = self.agent.get_xfer_descs(src_addrs, "DRAM")
        dst_descs = self.agent.get_xfer_descs(dst_addrs, "DRAM")
        # Transfer data
        xfer_handle = self.agent.initialize_xfer(
            "WRITE",
            src_descs,
            dst_descs,
            peer_name,
            notif.encode("ascii"),  # type: ignore
        )
        if not xfer_handle:
            raise Exception("KVSender failed to create transfer")
        state = self.agent.transfer(xfer_handle)
        if state == "ERR":
            raise Exception("KVSender failed to post transfer")
        return xfer_handle

    def _send_mamba_state(
        self,
        peer_name: str,
        prefill_state_indices: List[int],
        src_state_data_ptrs: list[int],
        src_state_item_lens: list[int],
        dst_state_data_ptrs: list[int],
        dst_state_indices: List[int],
        dst_gpu_id: int,
        notif: str,
    ):
        """Transfer Mamba states via RDMA."""
        assert len(prefill_state_indices) == 1, "Mamba should have single state index"
        assert len(dst_state_indices) == len(
            prefill_state_indices
        ), "State indices count mismatch between Prefill and Decode"

        src_addrs = []
        dst_addrs = []

        for i, dst_state_ptr in enumerate(dst_state_data_ptrs):
            length = src_state_item_lens[i]
            if length == 0 or src_state_data_ptrs[i] == 0 or dst_state_ptr == 0:
                continue
            src_addr = src_state_data_ptrs[i] + length * int(prefill_state_indices[0])
            dst_addr = dst_state_ptr + length * int(dst_state_indices[0])
            src_addrs.append((src_addr, length, self.kv_args.gpu_id))
            dst_addrs.append((dst_addr, length, dst_gpu_id))

        src_descs = self.agent.get_xfer_descs(src_addrs, "VRAM")
        dst_descs = self.agent.get_xfer_descs(dst_addrs, "VRAM")

        xfer_handle = self.agent.initialize_xfer(
            "WRITE",
            src_descs,
            dst_descs,
            peer_name,
            notif.encode("ascii"),
        )
        if not xfer_handle:
            raise Exception("Failed to create Mamba state transfer")
        state = self.agent.transfer(xfer_handle)
        if state == "ERR":
            raise Exception("Failed to post Mamba state transfer")
        return xfer_handle

    def _send_mamba_state_slice(
        self,
        peer_name: str,
        prefill_state_indices: List[int],
        src_state_data_ptrs: list[int],
        src_state_item_lens: list[int],
        src_state_dim_per_tensor: list[int],
        dst_state_data_ptrs: list[int],
        dst_state_indices: List[int],
        dst_state_item_lens: list[int],
        dst_state_dim_per_tensor: list[int],
        dst_gpu_id: int,
        notif: str,
        decode_tp_size: int,
        decode_tp_rank: int,
    ):
        """Transfer Mamba states with TP slice support via RDMA.

        When prefill and decode have different attn_tp_size, we slice the
        TP-sharded dimension (3rd dim) of conv_state and temporal_state
        accordingly, mirroring Mooncake's _send_mamba_state_slice.
        """
        logger.warning_once(
            "Using Mamba state slice transfer for different TP sizes. "
            f"Prefill attn_tp_size={self.attn_tp_size}, "
            f"Decode attn_tp_size={decode_tp_size}."
        )
        assert len(prefill_state_indices) == 1, "Mamba should have single state index"

        if not src_state_dim_per_tensor or not dst_state_dim_per_tensor:
            return self._send_mamba_state(
                peer_name,
                prefill_state_indices,
                src_state_data_ptrs,
                src_state_item_lens,
                dst_state_data_ptrs,
                dst_state_indices,
                dst_gpu_id,
                notif,
            )

        local_tp_rank_in_group = self.kv_args.engine_rank % self.attn_tp_size
        dst_tp_rank_in_group = decode_tp_rank % decode_tp_size

        src_addrs = []
        dst_addrs = []

        for i, dst_state_ptr in enumerate(dst_state_data_ptrs):
            src_item_len = src_state_item_lens[i]
            dst_item_len = dst_state_item_lens[i]
            if src_item_len == 0 or src_state_data_ptrs[i] == 0 or dst_state_ptr == 0:
                continue
            src_dim = src_state_dim_per_tensor[i]
            dst_dim = dst_state_dim_per_tensor[i]

            src_bytes_per_dim = src_item_len // src_dim
            dst_bytes_per_dim = dst_item_len // dst_dim

            if self.attn_tp_size > decode_tp_size:
                src_dim_start = 0
                num_dims_to_send = src_dim
                writers_per_decode = self.attn_tp_size // decode_tp_size
                local_writer_idx = local_tp_rank_in_group % writers_per_decode
                dst_dim_start = local_writer_idx * src_dim
            else:
                src_dim_start = (dst_tp_rank_in_group * dst_dim) % src_dim
                num_dims_to_send = dst_dim
                dst_dim_start = 0

            src_dim_offset = src_dim_start * src_bytes_per_dim
            dst_dim_offset = dst_dim_start * dst_bytes_per_dim
            bytes_to_send = num_dims_to_send * src_bytes_per_dim

            src_addr = (
                src_state_data_ptrs[i]
                + src_item_len * int(prefill_state_indices[0])
                + src_dim_offset
            )
            dst_addr = (
                dst_state_ptr
                + dst_item_len * int(dst_state_indices[0])
                + dst_dim_offset
            )
            src_addrs.append((src_addr, bytes_to_send, self.kv_args.gpu_id))
            dst_addrs.append((dst_addr, bytes_to_send, dst_gpu_id))

        src_descs = self.agent.get_xfer_descs(src_addrs, "VRAM")
        dst_descs = self.agent.get_xfer_descs(dst_addrs, "VRAM")

        xfer_handle = self.agent.initialize_xfer(
            "WRITE",
            src_descs,
            dst_descs,
            peer_name,
            notif.encode("ascii"),
        )
        if not xfer_handle:
            raise Exception("Failed to create Mamba state slice transfer")
        state = self.agent.transfer(xfer_handle)
        if state == "ERR":
            raise Exception("Failed to post Mamba state slice transfer")
        return xfer_handle

    def maybe_send_extra(
        self,
        peer_name: str,
        prefill_state_indices: List[List[int]],
        dst_state_data_ptrs: List[List[int]],
        dst_state_indices: List[List[int]],
        dst_gpu_id: int,
        notif: str,
        decode_tp_size: int,
        decode_tp_rank: int = 0,
        dst_state_item_lens: List[List[int]] | None = None,
        dst_state_dim_per_tensor: List[List[int]] | None = None,
    ):
        """Send state per hybrid component, dispatching by state_type[i]."""
        state_types = getattr(self.kv_args, "state_types", []) or []
        src_state_data_ptrs = self.kv_args.state_data_ptrs or []
        src_state_item_lens = self.kv_args.state_item_lens or []
        src_state_dim_per_tensor = (
            getattr(self.kv_args, "state_dim_per_tensor", []) or []
        )
        dst_state_item_lens = dst_state_item_lens or []
        dst_state_dim_per_tensor = dst_state_dim_per_tensor or []

        handles = []
        for i, st in enumerate(state_types):
            src_indices = (
                prefill_state_indices[i] if i < len(prefill_state_indices) else None
            )
            if src_indices is None or len(src_indices) == 0:
                continue
            src_ptrs = src_state_data_ptrs[i] if i < len(src_state_data_ptrs) else []
            src_lens = src_state_item_lens[i] if i < len(src_state_item_lens) else []
            src_dims = (
                src_state_dim_per_tensor[i] if i < len(src_state_dim_per_tensor) else []
            )
            dst_ptrs = dst_state_data_ptrs[i] if i < len(dst_state_data_ptrs) else []
            dst_indices = dst_state_indices[i] if i < len(dst_state_indices) else []
            dst_lens = dst_state_item_lens[i] if i < len(dst_state_item_lens) else []
            dst_dims = (
                dst_state_dim_per_tensor[i] if i < len(dst_state_dim_per_tensor) else []
            )
            comp_notif = f"{notif}_{i}"

            if st == StateType.MAMBA:
                if self.attn_tp_size != decode_tp_size:
                    h = self._send_mamba_state_slice(
                        peer_name,
                        src_indices,
                        src_ptrs,
                        src_lens,
                        src_dims,
                        dst_ptrs,
                        dst_indices,
                        dst_lens,
                        dst_dims,
                        dst_gpu_id,
                        comp_notif,
                        decode_tp_size,
                        decode_tp_rank,
                    )
                else:
                    h = self._send_mamba_state(
                        peer_name,
                        src_indices,
                        src_ptrs,
                        src_lens,
                        dst_ptrs,
                        dst_indices,
                        dst_gpu_id,
                        comp_notif,
                    )
            elif st in (StateType.SWA, StateType.NSA):
                if not self.is_mla_backend and self.attn_tp_size != decode_tp_size:
                    raise RuntimeError(
                        f"PD Disaggregation does NOT support PD different TP sizes for non-MLA {st.upper()} hybrid models yet."
                    )
                if len(src_indices) != len(dst_indices):
                    raise RuntimeError(
                        f"State index length mismatch at component {i}: "
                        f"prefill={len(src_indices)}, dst={len(dst_indices)}"
                    )
                h = self._send_kvcache_generic(
                    peer_name=peer_name,
                    src_data_ptrs=src_ptrs,
                    dst_data_ptrs=dst_ptrs,
                    item_lens=src_lens,
                    prefill_data_indices=np.array(src_indices, dtype=np.int32),
                    dst_data_indices=np.array(dst_indices, dtype=np.int32),
                    dst_gpu_id=dst_gpu_id,
                    notif=comp_notif,
                )
            else:
                raise RuntimeError(
                    f"PD Disaggregation via NIXL does NOT support {st} hybrid models yet."
                )
            if h is not None:
                handles.append(h)
        return handles

    def add_transfer_request(
        self,
        bootstrap_room: int,
        kv_indices: npt.NDArray[np.int32],
        index_slice: slice,
        is_last: bool,
        chunk_id: int,
        aux_index: Optional[int] = None,
        state_indices: Optional[List] = None,
    ):
        assert self.disaggregation_mode == DisaggregationMode.PREFILL
        assert not is_last or (is_last and aux_index is not None)

        # Prefetch STAGING_REQ to decode before enqueueing so decode has
        # already allocated staging by the time the worker picks up the
        # chunk. Internally a no-op when staging is disabled or no peer
        # in this room needs heterogeneous-TP staging.
        if self.enable_staging:
            self._prefetch_staging_reqs(bootstrap_room)

        # Transfer is async: just enqueue the chunk; the per-queue worker
        # (transfer_worker) does the actual gather + RDMA. Routing by
        # ``room % N`` keeps every chunk of a given room on the same
        # worker -- and therefore on the same private staging buffer --
        # which is required for the staging ring's offset/watermark
        # state machine to advance correctly.
        shard_idx = bootstrap_room % len(self.transfer_queues)
        self.transfer_queues[shard_idx].put(
            TransferKVChunk(
                room=bootstrap_room,
                prefill_kv_indices=kv_indices,
                index_slice=index_slice,
                is_last=is_last,
                chunk_id=chunk_id,
                prefill_aux_index=aux_index,
                state_indices=state_indices,
            )
        )
        return None

    def update_transfer_status(self):
        # Process notifications from received transfers.
        notif_map = self.agent.get_new_notifs()
        for peer_name, messages in notif_map.items():
            for msg in messages:
                # Notification tag layouts (underscore-separated):
                #   kv:    {room}_kv_{chunk_id}_{is_last}_{pp_rank}             -> 5 fields
                #   stg:   {room}_stg_{chunk_id}_{is_last}_{pp_rank}_{chunk_idx}
                #          _{page_start}_{num_pages}_{agent_name}               -> 9 fields
                #   aux:   {room}_aux                                           -> 2 fields
                #   state: {room}_state_{pp_rank}                               -> 3 fields
                # maxsplit=8 keeps everything past the 8th underscore in the
                # last component, so agent_name (which may itself contain
                # underscores) lands intact in components[8] for the stg path.
                components = msg.decode("ascii").split("_", 8)
                room = int(components[0])
                tag = components[1]
                if tag == "kv":
                    chunk_id = int(components[2])
                    is_last = bool(int(components[3]))
                    pp_rank = int(components[4]) if len(components) > 4 else 0
                    self._track_kv_arrival(room, chunk_id, is_last, pp_rank)
                elif tag == "stg":
                    self._handle_stg_notification(components, room)
                elif tag == "aux":
                    # main's "nokv" marker (decode-side radix cache hit):
                    # mark expected_kvs_per_pp[pp_rank] = 0 for this rank.
                    self._handle_aux_notification(room, components)
                elif tag == "state":
                    pp_rank = int(components[2]) if len(components) > 2 else 0
                    self.transfer_statuses[room].received_state_per_pp.add(pp_rank)

    def _handle_stg_notification(self, components, room: int):
        """Handle a staging RDMA notification tag.

        Format: {room}_stg_{chunk_id}_{is_last}_{pp_rank}_{chunk_idx}_{page_start}_{num_pages}_{agent_name}
        """
        chunk_id = int(components[2])
        is_last = bool(int(components[3]))
        pp_rank = int(components[4])
        chunk_idx = int(components[5])
        page_start = int(components[6])
        num_pages = int(components[7])
        agent_name = components[8] if len(components) > 8 else ""
        self._track_kv_arrival(room, chunk_id, is_last, pp_rank)
        self._handle_staging_chunk_arrived(
            room, chunk_idx, page_start, num_pages, agent_name
        )

    def _handle_aux_notification(self, room: int, components: List[str]):
        """Handle an aux notification and trigger last scatter if staging is complete.

        Notification tag layouts:
          aux:         {room}_aux                              -> 2 fields
          aux (nokv):  {room}_aux_nokv_{pp_rank}               -> 4 fields
                       (decode-side radix cache hit; this pp_rank sent
                       no KV pages, so expected_kvs_per_pp[pp_rank] = 0)
        """
        self.transfer_statuses[room].received_aux = True
        # main's "nokv" marker (decode-side radix cache hit, see #19746).
        if len(components) > 3 and components[2] == "nokv":
            pp_rank = int(components[3])
            self.transfer_statuses[room].expected_kvs_per_pp[pp_rank] = 0
        if self.transfer_statuses[room].num_pp_ranks_expected is None:
            self.transfer_statuses[room].num_pp_ranks_expected = (
                self.required_prefill_response_num_table.get(room, 1)
            )
        if (
            self.enable_staging
            and self._staging_handler is not None
            and self._staging_handler.is_staging_room(room)
        ):
            self._maybe_submit_last_scatter(room)

    def _track_kv_arrival(self, room: int, chunk_id: int, is_last: bool, pp_rank: int):
        """Update transfer status tracking for a kv chunk arrival."""
        self.transfer_statuses[room].received_kvs_per_pp[pp_rank].add(chunk_id)
        if is_last:
            self.transfer_statuses[room].expected_kvs_per_pp[pp_rank] = chunk_id + 1
            if self.transfer_statuses[room].num_pp_ranks_expected is None:
                self.transfer_statuses[room].num_pp_ranks_expected = (
                    self.required_prefill_response_num_table.get(room, 1)
                )
            if (
                self.enable_staging
                and self._staging_handler is not None
                and self._staging_handler.is_staging_room(room)
            ):
                self._maybe_submit_last_scatter(room)

    def _handle_staging_chunk_arrived(
        self,
        room: int,
        chunk_idx: int,
        page_start: int,
        num_pages: int,
        agent_name: str,
    ):
        """Process a staging chunk arrival via RDMA notification."""
        handler = self._staging_handler
        if handler is None:
            return
        handler.handle_chunk_arrived(
            room,
            chunk_idx,
            page_start,
            num_pages,
            agent_name,
            self._chunk_writer_counts,
        )

    def _maybe_submit_last_scatter(self, room: int):
        """Check if all kv+aux transfers are done and submit last scatter if so."""
        status = self.transfer_statuses.get(room)
        if status is None:
            return
        if not status.received_aux:
            return
        if status.num_pp_ranks_expected is None:
            return
        if len(status.expected_kvs_per_pp) < status.num_pp_ranks_expected:
            return
        for pp_rank, expected in status.expected_kvs_per_pp.items():
            if len(status.received_kvs_per_pp[pp_rank]) != expected:
                return
        handler = self._staging_handler
        if handler is not None and handler.is_staging_room(room):
            handler.submit_last_scatter_async(room)
            self._chunk_writer_counts.pop(room, None)

    def check_transfer_done(self, room: int):
        if room not in self.transfer_statuses:
            return False
        return self.transfer_statuses[room].is_done()

    def _start_bootstrap_thread(self):
        def bootstrap_thread():
            """This thread recvs transfer info from the decode engine"""
            while True:
                waiting_req_bytes = self.server_socket.recv_multipart()
                logger.debug(
                    f"Received multipart with total byte size {sum(len(x) for x in waiting_req_bytes)}"
                )

                # Staging: decode reports consumption watermark back to prefill
                if waiting_req_bytes[0] == b"WATERMARK":
                    if self.enable_staging:
                        from sglang.srt.disaggregation.common.staging_handler import (
                            handle_watermark_msg,
                        )

                        handle_watermark_msg(self._staging_ctx, waiting_req_bytes)
                    continue

                # Staging: decode replies with allocated staging offset
                if waiting_req_bytes[0] == b"STAGING_RSP":
                    if self.enable_staging:
                        from sglang.srt.disaggregation.common.staging_handler import (
                            handle_staging_rsp,
                        )

                        handle_staging_rsp(waiting_req_bytes, self.transfer_infos)
                    continue

                assert (
                    waiting_req_bytes[0] == GUARD
                ), f"First message should be {GUARD}. Foreign traffic?"
                waiting_req_bytes = waiting_req_bytes[1:]
                room = waiting_req_bytes[0].decode("ascii")
                agent_name = waiting_req_bytes[3].decode("ascii")
                if room == "None":
                    # Register new peer and save KV base pointers.
                    self._add_remote_peer(
                        KVArgsRegisterInfo.from_zmq(waiting_req_bytes)
                    )
                    logger.debug(f"Register KVArgs from {agent_name} successfully")
                    continue
                room = int(room)
                if room not in self.transfer_infos:
                    self.transfer_infos[room] = {}
                self.transfer_infos[room][agent_name] = TransferInfo.from_zmq(
                    waiting_req_bytes
                )
                required_dst_info_num = self.transfer_infos[room][
                    agent_name
                ].required_dst_info_num
                logger.debug(f"got info {room=} {agent_name=} {required_dst_info_num=}")
                if len(self.transfer_infos[room]) == required_dst_info_num:
                    self.req_to_decode_prefix_len[room] = next(
                        (
                            info.decode_prefix_len
                            for info in self.transfer_infos[room].values()
                            if info.decode_prefix_len is not None
                        ),
                        0,
                    )
                    logger.debug(f"{room=} is bootstrapped")
                    self.update_status(room, KVPoll.WaitingForInput)

        threading.Thread(target=bootstrap_thread).start()


class NixlKVSender(CommonKVSender):
    def __init__(
        self,
        mgr: NixlKVManager,
        bootstrap_addr: str,
        bootstrap_room: int,
        dest_tp_ranks: List[int],
        pp_rank: int,
    ):
        super().__init__(mgr, bootstrap_addr, bootstrap_room, dest_tp_ranks, pp_rank)
        self.has_sent = False
        self.chunk_id = 0
        self._send_failed = False
        self._send_error: Optional[Exception] = None
        self._transfer_start_time: Optional[float] = None

    def pop_decode_prefix_len(self) -> int:
        return self.kv_mgr.req_to_decode_prefix_len.pop(self.bootstrap_room, 0)

    def should_send_kv_chunk(self, num_pages: int, last_chunk: bool) -> bool:
        return num_pages > 0 or last_chunk

    def send(
        self,
        kv_indices: npt.NDArray[np.int32],
        state_indices: Optional[List] = None,
    ):
        if self._send_failed:
            return

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

        if self._transfer_start_time is None and (
            len(kv_indices) > 0 or state_indices is not None
        ):
            self._transfer_start_time = time.perf_counter()

        self.kv_mgr.add_transfer_request(
            self.bootstrap_room,
            kv_indices,
            index_slice,
            is_last,
            self.chunk_id,
            self.aux_index,
            state_indices,
        )
        self._record_transfer_indices(kv_indices, state_indices)
        self.chunk_id += 1
        if is_last:
            self.has_sent = True

    def poll(self) -> KVPoll:
        if self._send_failed:
            return KVPoll.Failed  # type: ignore
        status = self.kv_mgr.check_status(self.bootstrap_room)
        if (
            status == KVPoll.Success
            and self._transfer_start_time is not None
            and self._transfer_metric.transfer_latency_s is None
        ):
            self._transfer_metric.transfer_latency_s = (
                time.perf_counter() - self._transfer_start_time
            )
        return status

    def clear(self):
        super().clear()

    def failure_exception(self):
        if self._send_error is not None:
            raise self._send_error
        exc = self.kv_mgr.exceptions.pop(self.bootstrap_room, None)
        if exc is not None:
            raise exc
        raise RuntimeError("NIXL KVSender Exception")


class NixlKVReceiver(CommonKVReceiver):
    def __init__(
        self,
        mgr: NixlKVManager,
        bootstrap_addr: str,
        bootstrap_room: Optional[int] = None,
    ):
        self.started_transfer = False
        super().__init__(mgr, bootstrap_addr, bootstrap_room)
        self.init_time = None

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
            logger.error(
                f"Could not fetch prefill parallel info from bootstrap_addr: {self.bootstrap_addr}",
            )
            self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Failed)
            return

        # Register staging room bootstrap info for staging handler
        if (
            self.kv_mgr.enable_staging
            and self.kv_mgr._staging_ctx.allocator is not None
        ):
            self.chunk_staging_infos = []
            self.kv_mgr.register_staging_room_bootstrap(
                self.bootstrap_room, self.bootstrap_infos, self
            )

        for bootstrap_info in self.bootstrap_infos:
            logger.debug(
                f"Fetched bootstrap info: {bootstrap_info} for engine rank: {self.kv_mgr.kv_args.engine_rank}"
            )
            sock, lock = self._connect_to_bootstrap_server(bootstrap_info)
            is_dummy = bootstrap_info["is_dummy"]
            logger.debug(
                f"Sending to prefill server with bootstrap room {self.bootstrap_room} {is_dummy=}"
            )
            packed_state_indices = (
                pack_int_lists(
                    [(idx if idx is not None else []) for idx in state_indices], "i"
                )
                if not is_dummy and state_indices is not None
                else b""
            )
            with lock:
                sock.send_multipart(
                    [
                        GUARD,
                        str(self.bootstrap_room).encode("ascii"),
                        self.kv_mgr.local_ip.encode("ascii"),
                        str(self.kv_mgr.rank_port).encode("ascii"),
                        self.kv_mgr.agent.name.encode("ascii"),
                        kv_indices.tobytes() if not is_dummy else b"",
                        str(aux_index).encode("ascii"),
                        str(self.required_dst_info_num).encode("ascii"),
                        packed_state_indices,
                        str(decode_prefix_len or 0).encode("ascii"),
                    ]
                )

        # Mark that we expect state data if state_indices was provided
        if state_indices is not None:
            self.kv_mgr.transfer_statuses[self.bootstrap_room].expects_state = True

        self.started_transfer = True
        self.init_time = time.time()

    def poll(self) -> KVPoll:
        if self.conclude_state is not None:
            return self.conclude_state
        status = self.kv_mgr.check_status(self.bootstrap_room)
        if status in (KVPoll.Success, KVPoll.Failed):
            self.conclude_state = status
            return status
        if not self.started_transfer:
            return status

        now = time.time()
        elapsed = now - self.init_time

        if elapsed >= self.kv_mgr.waiting_timeout:
            logger.error(f"Request {self.bootstrap_room} waiting_timeout")
            self.kv_mgr.record_failure(
                self.bootstrap_room,
                f"Request {self.bootstrap_room} timed out after {elapsed:.1f}s in KVPoll.WaitingForInput",
            )
            self.conclude_state = KVPoll.Failed
            return KVPoll.Failed

        self.kv_mgr.update_transfer_status()
        if self.kv_mgr.check_transfer_done(self.bootstrap_room):  # type: ignore
            self.kv_mgr.addr_to_rooms_tracker[self.bootstrap_addr].discard(
                self.bootstrap_room
            )
            # Check if the transfer failed
            if self.kv_mgr.transfer_statuses[self.bootstrap_room].is_failed():
                self.conclude_state = KVPoll.Failed
                logger.error(
                    f"Transfer for room {self.bootstrap_room} failed due to node failure"
                )
            else:
                self.conclude_state = KVPoll.Success
            del self.kv_mgr.transfer_statuses[self.bootstrap_room]
            return self.conclude_state  # type: ignore
        return KVPoll.WaitingForInput  # type: ignore

    def _register_kv_args(self):
        for bootstrap_info in self.bootstrap_infos:
            sock, lock = self._connect_to_bootstrap_server(bootstrap_info)
            packed_kv_data_ptrs = b"".join(
                struct.pack("Q", ptr) for ptr in self.kv_mgr.kv_args.kv_data_ptrs
            )
            packed_aux_data_ptrs = b"".join(
                struct.pack("Q", ptr) for ptr in self.kv_mgr.kv_args.aux_data_ptrs
            )
            packed_state_data_ptrs = pack_int_lists(
                self.kv_mgr.kv_args.state_data_ptrs or [], "Q"
            )
            packed_state_item_lens = pack_int_lists(
                self.kv_mgr.kv_args.state_item_lens or [], "I"
            )
            packed_state_dim_per_tensor = pack_int_lists(
                getattr(self.kv_mgr.kv_args, "state_dim_per_tensor", []) or [], "I"
            )

            # Include staging allocator metadata if available
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

            with lock:
                sock.send_multipart(
                    [
                        GUARD,
                        "None".encode("ascii"),
                        self.kv_mgr.local_ip.encode("ascii"),
                        str(self.kv_mgr.rank_port).encode("ascii"),
                        self.kv_mgr.agent.name.encode("ascii"),
                        self.kv_mgr.agent.get_agent_metadata(),
                        packed_kv_data_ptrs,
                        packed_aux_data_ptrs,
                        packed_state_data_ptrs,
                        str(self.kv_mgr.kv_args.gpu_id).encode("ascii"),
                        str(self.kv_mgr.attn_tp_size).encode("ascii"),
                        str(self.kv_mgr.kv_args.engine_rank).encode("ascii"),
                        str(self.kv_mgr.kv_args.kv_item_lens[0]).encode("ascii"),
                        packed_state_item_lens,
                        packed_state_dim_per_tensor,
                        packed_staging_base_ptr,
                        staging_total_size_str,
                    ]
                )

    def failure_exception(self):
        raise RuntimeError("NIXL KVReceiver Exception")


class NixlKVBootstrapServer(CommonKVBootstrapServer):
    pass
