from __future__ import annotations

import dataclasses
import logging
import os
import struct
import threading
import time
import uuid
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import numpy.typing as npt
import requests
import torch

from sglang.srt.disaggregation.base.conn import KVArgs, KVPoll
from sglang.srt.disaggregation.common.conn import (
    CommonKVBootstrapServer,
    CommonKVManager,
    CommonKVReceiver,
    CommonKVSender,
)
from sglang.srt.disaggregation.common.utils import group_concurrent_contiguous
from sglang.srt.disaggregation.nixl.pinned_buffer_pool import PinnedBufferPool
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.environ import envs
from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)

# Default staging buffer size for Triton KV transfer (256MB)
DEFAULT_TRITON_STAGING_BUFFER_SIZE_MB = 256.0


def _import_triton_kv_transfer():
    """Lazily import Triton KV transfer functions to avoid import errors when not used."""
    try:
        from sglang.srt.layers.attention.triton_ops.kv_transfer import (
            gather_kv_to_pinned_all_layers,
            scatter_kv_with_staging_all_layers,
        )

        return gather_kv_to_pinned_all_layers, scatter_kv_with_staging_all_layers
    except ImportError as e:
        logger.warning(f"[TRITON-KV] Failed to import Triton KV transfer: {e}")
        return None, None


GUARD = "NixlMsgGuard".encode("ascii")

# Set SGLANG_NIXL_DEBUG_CHECKSUM=1 to enable KV transfer checksum validation.
# Prefill logs a checksum after gather; decode logs one before scatter.
# Mismatches indicate corruption between gather and scatter.
_NIXL_DEBUG_CHECKSUM = os.environ.get("SGLANG_NIXL_DEBUG_CHECKSUM", "0") == "1"


def _kv_checksum(buf: torch.Tensor, label: str, room: int) -> int:
    """
    Compute and log a diagnostic checksum of a pinned CPU KV buffer.

    Views the buffer as int16, samples ~1024 evenly-spaced elements, and sums
    them.  Returns the 32-bit wrapped sum.  Logs at WARNING level so it is
    visible without changing the log level.

    Detects:
    - All-zero buffer  → transfer never wrote / wrong address
    - Value mismatch   → data written to wrong slot / premature pool release
    - Count of zeros   → partially-written transfer
    """
    flat = buf.view(torch.int16)
    n = flat.numel()
    step = max(1, n // 1024)
    sample = flat[::step].to(torch.int32)
    total = int(sample.sum().item())
    checksum = total & 0xFFFFFFFF
    num_zeros = int((sample == 0).sum().item())
    # Log first 4 and last 4 raw int16 values as a sanity peek
    head_vals = flat[:4].tolist()
    tail_vals = flat[-4:].tolist()
    logger.warning(
        f"[KV-CKSUM] {label} room={room} "
        f"checksum=0x{checksum:08x} "
        f"sampled={len(sample)} zeros={num_zeros}/{len(sample)} "
        f"head={head_vals} tail={tail_vals}"
    )
    return checksum


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
    dst_state_indices: List[int]
    # Per-request allocated pinned buffer address and size (for concurrent-safe CPU buffer transfers)
    dst_pinned_ptr: int = 0
    dst_pinned_size: int = 0

    def is_dummy(self):
        return self.dst_kv_indices.size == 0

    @classmethod
    def from_zmq(cls, msg: List[bytes]):
        # Parse state_indices from msg[7] if present
        if len(msg) > 7 and msg[7] != b"":
            dst_state_indices = list(np.frombuffer(msg[7], dtype=np.int32))
        else:
            dst_state_indices = []

        # Parse per-request pinned buffer info from msg[8]/msg[9] if present
        dst_pinned_ptr = int(msg[8].decode("ascii")) if len(msg) > 8 else 0
        dst_pinned_size = int(msg[9].decode("ascii")) if len(msg) > 9 else 0

        return cls(
            room=int(msg[0].decode("ascii")),
            endpoint=msg[1].decode("ascii"),
            dst_port=int(msg[2].decode("ascii")),
            agent_name=msg[3].decode("ascii"),
            dst_kv_indices=np.frombuffer(msg[4], dtype=np.int32),
            dst_aux_index=int(msg[5].decode("ascii")),
            required_dst_info_num=int(msg[6].decode("ascii")),
            dst_state_indices=dst_state_indices,
            dst_pinned_ptr=dst_pinned_ptr,
            dst_pinned_size=dst_pinned_size,
        )


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
    dst_state_data_ptrs: list[int]
    gpu_id: int
    decode_tp_size: int
    decode_tp_rank: int
    dst_kv_item_len: int
    # For Triton KV transfer: pinned CPU buffer address and size
    dst_pinned_ptr: int = 0
    dst_pinned_size: int = 0

    @classmethod
    def from_zmq(cls, msg: List[bytes]):
        # Parse state_data_ptrs from msg[7] if present
        if len(msg) > 7 and msg[7] != b"":
            dst_state_data_ptrs = list(struct.unpack(f"{len(msg[7]) // 8}Q", msg[7]))
        else:
            dst_state_data_ptrs = []

        dst_pinned_ptr = int(msg[12].decode("ascii"))
        dst_pinned_size = int(msg[13].decode("ascii"))

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
            dst_pinned_ptr=dst_pinned_ptr,
            dst_pinned_size=dst_pinned_size,
        )


@dataclasses.dataclass
class TransferStatus:
    """Used by KV Receiver to know when a transfer is done."""

    # KV chunks received per sender: {sender_key: set of chunk_ids}
    # sender_key is the NIXL peer_name, which uniquely identifies each prefill TP rank
    received_kvs_per_sender: Dict[str, Set[int]] = dataclasses.field(
        default_factory=lambda: defaultdict(set)
    )
    # Expected chunk count per sender (set when is_last=True): {sender_key: expected_count}
    expected_kvs_per_sender: Dict[str, int] = dataclasses.field(default_factory=dict)
    # Number of senders expected to send data.
    num_senders_expected: Optional[int] = None
    # Whether aux data has been received.
    received_aux: bool = False
    # Senders that have sent state data.
    received_state_per_sender: Set[str] = dataclasses.field(default_factory=set)
    # Whether state data is expected (set based on state_type).
    expects_state: bool = False
    # Mark as failed
    is_failure: bool = False

    def is_done(self):
        if self.is_failure:
            return True
        if self.num_senders_expected is None or not self.received_aux:
            return False
        # If state data is expected, check all senders have sent it
        if (
            self.expects_state
            and len(self.received_state_per_sender) < self.num_senders_expected
        ):
            return False
        # All senders must have reported their expected count
        if len(self.expected_kvs_per_sender) < self.num_senders_expected:
            return False
        # Each sender must have received all expected chunks
        for sender_key, expected in self.expected_kvs_per_sender.items():
            if len(self.received_kvs_per_sender[sender_key]) != expected:
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
        agent_config = nixl_agent_config(
            backends=[backend],
            num_threads=(8 if disaggregation_mode == DisaggregationMode.PREFILL else 0),
        )
        self.agent = nixl_agent(str(uuid.uuid4()), agent_config)

        available_plugins = self.agent.get_plugin_list()
        if backend not in available_plugins:
            raise ValueError(
                f"NIXL backend '{backend}' not found. Available: {available_plugins}. "
                f"Please install the required NIXL plugin or choose from: {available_plugins}"
            )
        logger.info(f"NIXL KVManager initialized with backend: {backend}")

        # Store CPU buffer transfer configuration
        self.nixl_use_cpu_buffer = getattr(server_args, "nixl_use_cpu_buffer", False)
        self.triton_staging_buffer: Optional[torch.Tensor] = None
        self._pinned_pool: Optional[PinnedBufferPool] = None
        self.triton_pinned_descs = None
        self._server_args = server_args

        # Initialize Triton transfer infrastructure if enabled
        if self.nixl_use_cpu_buffer:
            self._init_triton_transfer_buffers()

        self.register_buffer_to_engine()

        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            # Per-request KV index accumulation for CPU buffer chunked-prefill fix.
            # Maps bootstrap_room -> list of kv_index arrays (one per send() chunk).
            # All chunks are combined into a single gather+NIXL write on is_last,
            # preventing each chunk from overwriting the previous chunk's offset in
            # the destination (decode) CPU buffer.
            self._cpu_pending_kv: Dict[int, List] = {}
            self._start_bootstrap_thread()
        elif self.disaggregation_mode == DisaggregationMode.DECODE:
            self.transfer_statuses: Dict[int, TransferStatus] = defaultdict(
                TransferStatus
            )
            # Deferred pool releases: (cuda_event, pool_offset) pairs waiting for
            # the scatter kernel to finish before the pinned region can be reused.
            self._pending_pool_releases: List[Tuple[torch.cuda.Event, int]] = []
            self._start_heartbeat_checker_thread()
        else:
            raise ValueError(
                f"Unsupported DisaggregationMode: {self.disaggregation_mode}"
            )

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

        # Register state/extra pool data buffers if present
        if self.kv_args.state_data_ptrs and self.kv_args.state_data_lens:
            state_addrs = []
            for state_data_ptr, state_data_len in zip(
                self.kv_args.state_data_ptrs, self.kv_args.state_data_lens
            ):
                state_addrs.append(
                    (state_data_ptr, state_data_len, self.kv_args.gpu_id, "")
                )
            self.state_descs = self.agent.register_memory(state_addrs, "VRAM")
            logger.debug(
                f"Register state tensors, len(state_addrs)= {len(state_addrs)}"
            )
            if not self.state_descs:
                raise Exception("NIXL memory registration failed for state tensors")

        # Register shared pinned buffer pool with NIXL if enabled
        if self.nixl_use_cpu_buffer and self._pinned_pool is not None:
            self.triton_pinned_descs = self._pinned_pool.register_with_nixl(self.agent)

    def _init_triton_transfer_buffers(self):
        """Initialize GPU staging buffer and shared pinned buffer pool for Triton KV transfer."""
        # Get dtype from KV cache buffers (supports fp8, fp16, bf16)
        k_buffers = self.kv_args.k_buffers
        if k_buffers is not None and len(k_buffers) > 0:
            kv_dtype = k_buffers[0].dtype
            kv_elem_bytes = k_buffers[0].element_size()
        else:
            # Fallback to bfloat16 if k_buffers not available yet
            kv_dtype = torch.bfloat16
            kv_elem_bytes = 2
            logger.warning(
                "[TRITON-KV] k_buffers not available, falling back to bfloat16. "
                "This may cause issues if KV cache uses a different dtype (e.g., fp8)."
            )

        # Allocate GPU staging buffer (fixed size, 256MB by default)
        staging_size_bytes = int(DEFAULT_TRITON_STAGING_BUFFER_SIZE_MB * 1e6)
        staging_elements = staging_size_bytes // kv_elem_bytes
        self.triton_staging_buffer = torch.empty(
            staging_elements, dtype=kv_dtype, device=f"cuda:{self.kv_args.gpu_id}"
        )

        # Get or create shared pinned buffer pool for this GPU
        pinned_size_bytes = int(
            getattr(self._server_args, "nixl_cpu_buffer_size_gb", 16.0) * 1e9
        )
        self._pinned_pool = PinnedBufferPool.get_or_create(
            gpu_id=self.kv_args.gpu_id,
            dtype=kv_dtype,
            total_size_bytes=pinned_size_bytes,
        )

        logger.info(
            f"[TRITON-KV] Initialized transfer buffers: "
            f"staging={self.triton_staging_buffer.nbytes / 1e6:.2f}MB (GPU), "
            f"shared_pinned_pool={pinned_size_bytes / 1e9:.2f}GB (CPU)"
        )

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
            (block[0] for block in prefill_kv_blocks), dtype=np.int64
        )
        dst_starts = np.fromiter((block[0] for block in dst_kv_blocks), dtype=np.int64)
        block_lens = np.fromiter(
            (len(block) for block in prefill_kv_blocks), dtype=np.int64
        )

        for src_ptr, dst_ptr, item_len in layers_params:
            lengths = item_len * block_lens
            src_addrs.append(src_ptr + prefill_starts * item_len)
            src_lens.append(lengths)
            dst_addrs.append(dst_ptr + dst_starts * item_len)
            dst_lens.append(lengths)

        def make_req_array(addr_chunks, len_chunks, gpu):
            if not addr_chunks:
                return np.empty((0, 3), dtype=np.int64)
            flat_addrs = np.concatenate(addr_chunks)
            flat_lens = np.concatenate(len_chunks)
            return np.column_stack(
                (
                    flat_addrs,
                    flat_lens,
                    np.full_like(flat_addrs, gpu),
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
        num_kv_heads = self.kv_args.kv_head_num

        # Calculate head distribution
        src_heads_per_rank = num_kv_heads
        dst_heads_per_rank = num_kv_heads * prefill_tp_size // decode_tp_size

        src_kv_item_len = self.kv_args.kv_item_lens[0]
        page_size = self.kv_args.page_size

        bytes_per_head_slice_to_send = (
            dst_kv_item_len // page_size // dst_heads_per_rank
        )

        # Determine which heads to send
        if prefill_tp_size > decode_tp_size:
            # Multiple prefill ranks to one decode rank
            src_head_start_offset = 0
            num_heads_to_send = src_heads_per_rank
            dst_head_start_offset = local_tp_rank_in_group * src_heads_per_rank
        else:
            # Send KVCache from 1 prefill instance to multiple decode instances
            src_head_start_offset = (
                dst_tp_rank_in_group * dst_heads_per_rank
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

        prefill_indices = np.asarray(prefill_kv_indices, dtype=np.int64)
        dst_indices = np.asarray(dst_kv_indices, dtype=np.int64)
        bytes_per_token_prefill = src_kv_item_len // page_size
        bytes_per_token_decode = dst_kv_item_len // page_size
        token_offsets = np.arange(page_size, dtype=np.int64)

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
                return np.empty((0, 3), dtype=np.int64)
            flat_addrs = np.concatenate(addr_chunks)
            return np.column_stack(
                (
                    flat_addrs,
                    np.full_like(flat_addrs, size),
                    np.full_like(flat_addrs, gpu),
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

    def _expand_pages_to_slots(
        self,
        page_indices: npt.NDArray[np.int32],
        page_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Expand page indices to slot indices (each page has page_size slots)."""
        pages = torch.from_numpy(page_indices).to(device, dtype=torch.int64)
        offsets = torch.arange(page_size, device=device, dtype=torch.int64)
        return (pages.unsqueeze(1) * page_size + offsets).flatten()

    def send_kvcache_triton(
        self,
        peer_name: str,
        prefill_kv_indices: npt.NDArray[np.int32],
        dst_pinned_ptr: int,
        dst_pinned_size: int,
        notif: str,
        head_start: int = 0,
        num_heads_to_send: int = None,
        dst_head_offset: int = 0,
    ):
        """
        Send KV cache using Triton gather kernel + single NIXL transfer.

        This method:
        1. Allocates a region from the shared pinned buffer pool
        2. Uses gather_kv_to_pinned_all_layers to collect scattered KV data into the region
        3. Records a CUDA event and returns (event, post_fn)

        The caller should poll event.query() and call post_fn() when the event fires.
        post_fn() initiates the NIXL transfer and returns (handles, pool_allocations).
        """
        gather_kv_all_layers, _ = _import_triton_kv_transfer()
        if gather_kv_all_layers is None:
            raise RuntimeError(
                "[TRITON-KV] Triton KV transfer not available. "
                "Make sure triton is installed."
            )

        if self.kv_args.k_buffers is None or self.kv_args.v_buffers is None:
            raise RuntimeError(
                "[TRITON-KV] k_buffers and v_buffers must be set in KVArgs "
                "when using Triton KV transfer."
            )

        if self._pinned_pool is None:
            raise RuntimeError(
                "[TRITON-KV] Pinned buffer pool not initialized."
            )

        k_buffers = self.kv_args.k_buffers
        v_buffers = self.kv_args.v_buffers
        num_layers = len(k_buffers)
        num_heads = k_buffers[0].shape[1]
        head_dim = k_buffers[0].shape[2]
        device = k_buffers[0].device

        if num_heads_to_send is None:
            num_heads_to_send = num_heads - head_start

        # Convert page indices to slot indices
        page_size = self.kv_args.page_size
        slot_indices_tensor = self._expand_pages_to_slots(
            prefill_kv_indices, page_size, device
        ).to(torch.int32)
        num_tokens = len(slot_indices_tensor)

        # Calculate transfer size
        bytes_per_element = k_buffers[0].element_size()
        transfer_elements = num_layers * 2 * num_tokens * num_heads_to_send * head_dim
        transfer_bytes = transfer_elements * bytes_per_element

        # Allocate region from shared pinned buffer pool
        src_offset, buffer_region = self._pinned_pool.allocate(transfer_bytes)

        logger.debug(
            f"[TRITON-KV] send_kvcache_triton: {num_tokens} tokens, {num_layers} layers, "
            f"heads [{head_start}:{head_start + num_heads_to_send}], "
            f"transfer_size={transfer_bytes / 1e6:.2f}MB, pool_offset={src_offset}"
        )

        # Create pointer tensors (cached for reuse)
        if not hasattr(self, '_k_data_ptrs') or self._k_data_ptrs is None:
            self._k_data_ptrs = torch.tensor(
                [x.data_ptr() for x in k_buffers], dtype=torch.uint64, device=device
            )
            self._v_data_ptrs = torch.tensor(
                [x.data_ptr() for x in v_buffers], dtype=torch.uint64, device=device
            )
            self._src_slot_stride = k_buffers[0].stride(0)
            self._src_head_stride = k_buffers[0].stride(1)

        # Gather KV data to allocated region using single-kernel Triton (device->host)
        gather_kv_all_layers(
            k_data_ptrs=self._k_data_ptrs,
            v_data_ptrs=self._v_data_ptrs,
            slot_indices=slot_indices_tensor,
            pinned_output=buffer_region,
            head_start=head_start,
            num_heads_to_gather=num_heads_to_send,
            num_layers=num_layers,
            head_dim=head_dim,
            src_slot_stride=self._src_slot_stride,
            src_head_stride=self._src_head_stride,
            kv_elem_bytes=bytes_per_element,
        )

        # Record CUDA event — poll() will call post_fn() once event.query() is True,
        # ensuring the gather kernel has written all data to pinned memory before NIXL reads it.
        event = torch.cuda.Event()
        event.record()

        # Capture variables needed by post_fn
        head_stride_bytes = num_layers * 2 * num_tokens * head_dim * bytes_per_element
        dst_offset = dst_head_offset * head_stride_bytes
        buf_ptr = buffer_region.data_ptr()
        pool_ref = self._pinned_pool

        def post_fn():
            if dst_pinned_ptr == 0:
                pool_ref.release(src_offset)
                raise RuntimeError(
                    f"[TRITON-KV] Invalid dst_pinned_ptr=0 for {peer_name}."
                )

            # Checksum the gather output AFTER the CUDA event has fired,
            # confirming the gather kernel completed before NIXL reads it.
            room_for_log = int(notif.split("_")[0])
            logger.warning(
                f"[DBG-NIXL-WRITE] room={room_for_log} "
                f"src=0x{buf_ptr:x} dst=0x{dst_pinned_ptr + dst_offset:x} "
                f"size={transfer_bytes}"
            )
            if _NIXL_DEBUG_CHECKSUM:
                _kv_checksum(
                    buffer_region,
                    f"PREFILL-AFTER-GATHER peer={peer_name}",
                    room_for_log,
                )

            src_addrs = [(buf_ptr, transfer_bytes, 0)]
            dst_addrs = [(dst_pinned_ptr + dst_offset, transfer_bytes, 0)]

            src_descs = self.agent.get_xfer_descs(src_addrs, "DRAM")
            dst_descs = self.agent.get_xfer_descs(dst_addrs, "DRAM")

            xfer_handle = self.agent.initialize_xfer(
                "WRITE", src_descs, dst_descs, peer_name, notif.encode("ascii")
            )
            if not xfer_handle:
                pool_ref.release(src_offset)
                raise Exception("[TRITON-KV] Failed to create Triton KV transfer")

            state = self.agent.transfer(xfer_handle)
            if state == "ERR":
                pool_ref.release(src_offset)
                raise Exception("[TRITON-KV] Failed to post Triton KV transfer")

            return [xfer_handle], [(pool_ref, src_offset)]

        return event, post_fn

    def _send_kvcache_triton_batched(
        self,
        requests: List[tuple],
        prefill_kv_indices: npt.NDArray[np.int32],
        total_heads: int,
    ):
        """
        Batched KV transfer: ONE gather of ALL heads, then slice buffer for parallel NIXL transfers.

        Args:
            requests: List of (agent_name, dst_pinned_ptr, dst_pinned_size, notif, head_start, num_heads)
            prefill_kv_indices: Page indices to transfer
            total_heads: Total number of KV heads on this prefill rank

        Returns:
            Tuple of (event, post_fn) where post_fn() initiates all NIXL transfers and
            returns (handles, pool_allocations).
        """
        gather_kv_all_layers, _ = _import_triton_kv_transfer()
        if gather_kv_all_layers is None:
            raise RuntimeError("[TRITON-KV] Triton KV transfer not available.")

        if self.kv_args.k_buffers is None or self.kv_args.v_buffers is None:
            raise RuntimeError("[TRITON-KV] k_buffers and v_buffers must be set.")

        if self._pinned_pool is None:
            raise RuntimeError("[TRITON-KV] Pinned buffer pool not initialized.")

        k_buffers = self.kv_args.k_buffers
        v_buffers = self.kv_args.v_buffers
        num_layers = len(k_buffers)
        head_dim = k_buffers[0].shape[2]
        device = k_buffers[0].device

        # Convert page indices to slot indices
        page_size = self.kv_args.page_size
        slot_indices_tensor = self._expand_pages_to_slots(
            prefill_kv_indices, page_size, device
        ).to(torch.int32)
        num_tokens = len(slot_indices_tensor)

        # Calculate total buffer size for ALL heads
        bytes_per_element = k_buffers[0].element_size()
        total_transfer_bytes = num_layers * 2 * num_tokens * total_heads * head_dim * bytes_per_element

        # Allocate ONE buffer from pool for all heads
        src_offset, buffer_region = self._pinned_pool.allocate(total_transfer_bytes)

        # Create pointer tensors (cached for reuse)
        if not hasattr(self, '_k_data_ptrs') or self._k_data_ptrs is None:
            self._k_data_ptrs = torch.tensor(
                [x.data_ptr() for x in k_buffers], dtype=torch.uint64, device=device
            )
            self._v_data_ptrs = torch.tensor(
                [x.data_ptr() for x in v_buffers], dtype=torch.uint64, device=device
            )
            self._src_slot_stride = k_buffers[0].stride(0)
            self._src_head_stride = k_buffers[0].stride(1)

        # ONE gather of ALL heads
        gather_kv_all_layers(
            k_data_ptrs=self._k_data_ptrs,
            v_data_ptrs=self._v_data_ptrs,
            slot_indices=slot_indices_tensor,
            pinned_output=buffer_region,
            head_start=0,
            num_heads_to_gather=total_heads,
            num_layers=num_layers,
            head_dim=head_dim,
            src_slot_stride=self._src_slot_stride,
            src_head_stride=self._src_head_stride,
            kv_elem_bytes=bytes_per_element,
        )

        # Record CUDA event — poll() will call post_fn() once event.query() is True,
        # ensuring the gather kernel has written all data to pinned memory before NIXL reads it.
        event = torch.cuda.Event()
        event.record()

        # Capture variables needed by post_fn
        head_stride_bytes = num_layers * 2 * num_tokens * head_dim * bytes_per_element
        buf_data_ptr = buffer_region.data_ptr()
        pool_ref = self._pinned_pool

        def post_fn():
            handles = []
            # Checksum the FULL gather buffer once (after the CUDA event fires)
            if _NIXL_DEBUG_CHECKSUM and requests:
                first_notif = requests[0][3]
                room_for_log = int(first_notif.split("_")[0])
                _kv_checksum(
                    buffer_region,
                    f"PREFILL-BATCHED-AFTER-GATHER nreqs={len(requests)}",
                    room_for_log,
                )
            for agent_name, dst_pinned_ptr, dst_pinned_size, notif, head_start, num_heads in requests:
                src_slice_ptr = buf_data_ptr + head_start * head_stride_bytes
                slice_bytes = num_heads * head_stride_bytes

                if dst_pinned_ptr == 0:
                    pool_ref.release(src_offset)
                    raise RuntimeError(
                        f"[TRITON-KV-BATCHED] Invalid dst_pinned_ptr=0 for {agent_name}."
                    )

                src_addrs = [(src_slice_ptr, slice_bytes, 0)]
                dst_addrs = [(dst_pinned_ptr, slice_bytes, 0)]

                src_descs = self.agent.get_xfer_descs(src_addrs, "DRAM")
                dst_descs = self.agent.get_xfer_descs(dst_addrs, "DRAM")

                xfer_handle = self.agent.initialize_xfer(
                    "WRITE", src_descs, dst_descs, agent_name, notif.encode("ascii")
                )
                if not xfer_handle:
                    pool_ref.release(src_offset)
                    raise Exception(f"[TRITON-KV-BATCHED] Failed to create transfer to {agent_name}")

                state = self.agent.transfer(xfer_handle)
                if state == "ERR":
                    pool_ref.release(src_offset)
                    raise Exception(f"[TRITON-KV-BATCHED] Failed to post transfer to {agent_name}")

                handles.append(xfer_handle)

            return handles, [(pool_ref, src_offset)]

        return event, post_fn

    def scatter_received_kv(
        self,
        kv_indices: npt.NDArray[np.int32],
        head_start: int = 0,
        num_heads_received: int = None,
        pinned_buffer: Optional[torch.Tensor] = None,
    ):
        """
        Scatter received KV data from pinned buffer to KV cache.

        Called on the receiver side after NIXL transfer completes.
        """
        _, scatter_kv_all_layers = _import_triton_kv_transfer()
        if scatter_kv_all_layers is None:
            raise RuntimeError("[TRITON-KV] Triton KV transfer not available.")

        if self.kv_args.k_buffers is None or self.kv_args.v_buffers is None:
            raise RuntimeError("[TRITON-KV] k_buffers and v_buffers must be set.")

        if self._pinned_pool is None:
            raise RuntimeError("[TRITON-KV] Pinned buffer pool not initialized.")

        k_buffers = self.kv_args.k_buffers
        v_buffers = self.kv_args.v_buffers
        num_layers = len(k_buffers)
        num_heads = k_buffers[0].shape[1]
        head_dim = k_buffers[0].shape[2]
        device = k_buffers[0].device

        if num_heads_received is None:
            num_heads_received = num_heads - head_start

        # Convert page indices to slot indices
        page_size = self.kv_args.page_size
        slot_indices_tensor = self._expand_pages_to_slots(
            kv_indices, page_size, device
        ).to(torch.int32)
        num_tokens = len(slot_indices_tensor)

        bytes_per_element = k_buffers[0].element_size()

        # Create pointer tensors (cached for reuse)
        if not hasattr(self, '_k_data_ptrs') or self._k_data_ptrs is None:
            self._k_data_ptrs = torch.tensor(
                [x.data_ptr() for x in k_buffers], dtype=torch.uint64, device=device
            )
            self._v_data_ptrs = torch.tensor(
                [x.data_ptr() for x in v_buffers], dtype=torch.uint64, device=device
            )
            self._dst_slot_stride = k_buffers[0].stride(0)
            self._dst_head_stride = k_buffers[0].stride(1)

        # Scatter from the per-request allocated region (or whole pool as fallback) to KV cache.
        # No CPU sync needed: the scatter kernel runs on the default CUDA stream, and the
        # subsequent model forward pass also runs on that stream, so GPU stream ordering
        # guarantees the scatter completes before the forward reads the KV cache.
        input_buffer = pinned_buffer if pinned_buffer is not None else self._pinned_pool.buffer
        scatter_kv_all_layers(
            pinned_input=input_buffer,
            k_data_ptrs=self._k_data_ptrs,
            v_data_ptrs=self._v_data_ptrs,
            slot_indices=slot_indices_tensor,
            head_start=head_start,
            num_heads_to_scatter=num_heads_received,
            num_layers=num_layers,
            head_dim=head_dim,
            dst_slot_stride=self._dst_slot_stride,
            dst_head_stride=self._dst_head_stride,
            kv_elem_bytes=bytes_per_element,
        )

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

        prefill_state_data_ptrs = self.kv_args.state_data_ptrs
        prefill_state_item_lens = self.kv_args.state_item_lens

        for i, dst_state_ptr in enumerate(dst_state_data_ptrs):
            length = prefill_state_item_lens[i]
            src_addr = prefill_state_data_ptrs[i] + length * int(
                prefill_state_indices[0]
            )
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

    def maybe_send_extra(
        self,
        peer_name: str,
        prefill_state_indices: List[int],
        dst_state_data_ptrs: list[int],
        dst_state_indices: List[int],
        dst_gpu_id: int,
        notif: str,
        decode_tp_size: int,
    ):
        """Send state or extra pool data with type-specific handling."""
        state_type = getattr(self.kv_args, "state_type", "none")

        if state_type == "mamba":
            if self.attn_tp_size != decode_tp_size:
                raise RuntimeError(
                    "PD Disaggregation does NOT support PD different TP sizes for hybrid mamba models yet."
                )
            return self._send_mamba_state(
                peer_name,
                prefill_state_indices,
                dst_state_data_ptrs,
                dst_state_indices,
                dst_gpu_id,
                notif,
            )
        elif state_type in ["swa", "nsa"]:
            if not self.is_mla_backend and self.attn_tp_size != decode_tp_size:
                raise RuntimeError(
                    f"PD Disaggregation does NOT support PD different TP sizes for non-MLA {state_type.upper()} hybrid models yet."
                )
            if len(prefill_state_indices) != len(dst_state_indices):
                raise RuntimeError(
                    f"State index length mismatch: prefill={len(prefill_state_indices)}, "
                    f"dst={len(dst_state_indices)}"
                )
            return self._send_kvcache_generic(
                peer_name=peer_name,
                src_data_ptrs=self.kv_args.state_data_ptrs,
                dst_data_ptrs=dst_state_data_ptrs,
                item_lens=self.kv_args.state_item_lens,
                prefill_data_indices=np.array(prefill_state_indices, dtype=np.int32),
                dst_data_indices=np.array(dst_state_indices, dtype=np.int32),
                dst_gpu_id=dst_gpu_id,
                notif=notif,
            )
        else:
            if state_type != "none":
                raise RuntimeError(
                    f"PD Disaggregation via NIXL does NOT support {state_type} hybrid models yet."
                )
            return None

    def add_transfer_request(
        self,
        bootstrap_room: int,
        kv_indices: npt.NDArray[np.int32],
        index_slice: slice,
        is_last: bool,
        chunk_id: int,
        aux_index: Optional[int] = None,
        state_indices: Optional[List[int]] = None,
    ):
        """
        Add a transfer request for KV cache data.

        Returns:
            Tuple of (handles, pool_allocations) where:
            - handles: List of NIXL transfer handles
            - pool_allocations: List of (pool, offset) tuples for later release
        """
        assert self.disaggregation_mode == DisaggregationMode.PREFILL
        assert not is_last or (is_last and aux_index is not None)

        reqs_to_be_processed = self.transfer_infos[bootstrap_room].values()
        handles = []
        pool_allocations = []
        pending_posts = []

        # Filter out dummy requests for CPU buffer batched path detection
        active_reqs = [req for req in reqs_to_be_processed if not req.is_dummy()]

        # Detect batched Triton case: prefill_tp < decode_tp with multiple destinations
        if active_reqs:
            first_decode_info = self.decode_kv_args_table.get(active_reqs[0].agent_name)
            if first_decode_info:
                prefill_tp_size = self.attn_tp_size
                decode_tp_size = first_decode_info.decode_tp_size

                use_batched = (
                    self.nixl_use_cpu_buffer
                    and prefill_tp_size < decode_tp_size
                    and all(
                        self.decode_kv_args_table[r.agent_name].dst_pinned_ptr != 0
                        for r in active_reqs
                    )
                    and self.kv_args.k_buffers is not None
                    and self.kv_args.v_buffers is not None
                    and not self.is_mla_backend
                )

                if use_batched:
                    # Collect batch request info
                    num_kv_heads = self.kv_args.kv_head_num
                    total_prefill_heads = num_kv_heads * prefill_tp_size
                    heads_per_decode_rank = total_prefill_heads // decode_tp_size
                    # Decode ranks that connect to this prefill rank are grouped in
                    # a contiguous block. Use the relative rank within that block so
                    # head_start stays in [0, num_kv_heads).
                    decode_per_prefill = decode_tp_size // prefill_tp_size

                    batch_requests = []
                    for req in active_reqs:
                        decode_info = self.decode_kv_args_table[req.agent_name]
                        decode_tp_rank = decode_info.decode_tp_rank % decode_tp_size
                        relative_decode_rank = decode_tp_rank % decode_per_prefill
                        head_start = relative_decode_rank * heads_per_decode_rank
                        logger.debug(
                            f"[MIXED-TP-BATCHED] prefill_tp={prefill_tp_size}, "
                            f"decode_tp={decode_tp_size}, decode_tp_rank={decode_tp_rank}, "
                            f"decode_per_prefill={decode_per_prefill}, "
                            f"relative_decode_rank={relative_decode_rank}, "
                            f"head_start={head_start}, heads_per_decode={heads_per_decode_rank}"
                        )
                        notif = f"{req.room}_kv_{chunk_id}_{int(is_last)}_{self.kv_args.pp_rank}"
                        # Use per-request allocated ptr if available, else fall back to static pool start
                        effective_dst_ptr = req.dst_pinned_ptr if req.dst_pinned_ptr != 0 else decode_info.dst_pinned_ptr
                        effective_dst_size = req.dst_pinned_size if req.dst_pinned_size != 0 else decode_info.dst_pinned_size
                        batch_requests.append((
                            req.agent_name,
                            effective_dst_ptr,
                            effective_dst_size,
                            notif,
                            head_start,
                            heads_per_decode_rank,
                        ))

                    batch_event, batch_post_fn = self._send_kvcache_triton_batched(
                        batch_requests, kv_indices, num_kv_heads
                    )
                    pending_posts.append((batch_event, batch_post_fn))

                    # Handle aux data separately
                    if is_last:
                        for req in active_reqs:
                            assert aux_index is not None
                            decode_info = self.decode_kv_args_table[req.agent_name]
                            aux_xfer_handle = self.send_aux(
                                req.agent_name,
                                aux_index,
                                decode_info.dst_aux_ptrs,
                                req.dst_aux_index,
                                f"{req.room}_aux",
                            )
                            handles.append(aux_xfer_handle)

                    if is_last:
                        del self.transfer_infos[bootstrap_room]

                    return handles, pool_allocations, pending_posts

        for req in reqs_to_be_processed:
            assert bootstrap_room == req.room
            if req.is_dummy():
                continue

            chunked_dst_kv_indice = req.dst_kv_indices[index_slice]
            assert len(chunked_dst_kv_indice) == len(kv_indices)
            assert req.agent_name in self.decode_kv_args_table

            notif = f"{req.room}_kv_{chunk_id}_{int(is_last)}_{self.kv_args.pp_rank}"
            decode_info = self.decode_kv_args_table[req.agent_name]
            decode_tp_size = decode_info.decode_tp_size

            # Check if CPU buffer Triton transfer is enabled and supported
            prefill_tp_size = self.attn_tp_size
            use_cpu_buffer = (
                self.nixl_use_cpu_buffer
                and decode_info.dst_pinned_ptr != 0
                and self.kv_args.k_buffers is not None
                and self.kv_args.v_buffers is not None
                and not self.is_mla_backend
            )

            kv_xfer_handle = None
            if use_cpu_buffer and prefill_tp_size >= decode_tp_size:
                # Triton CPU buffer path for same-TP or prefill_tp > decode_tp.
                #
                # Bug fix: chunked prefill sends kv_indices in multiple send()
                # calls (chunk_id 0, 1, ...).  Without accumulation, each chunk
                # computes dst_offset using its own num_tokens, so chunk N
                # overwrites chunk N-1 at the same destination offset, leaving
                # the second half of the decode CPU buffer as zeros.
                #
                # Fix: accumulate all per-chunk kv_indices and issue a single
                # gather + NIXL WRITE only when is_last=True, covering the full
                # N_total-token buffer in one shot.  Decode always receives
                # exactly one KV notification per prefill TP rank (chunk_id=0,
                # is_last=True), so its is_done() logic is unaffected.
                num_kv_heads = self.kv_args.kv_head_num
                local_tp_rank = self.kv_args.engine_rank % prefill_tp_size

                if prefill_tp_size > decode_tp_size:
                    head_start = 0
                    num_heads_to_send = num_kv_heads
                    # Use the rank relative to the decode bucket so dst_head_offset
                    # stays within [0, num_kv_heads * prefill_ranks_per_decode).
                    prefill_ranks_per_decode = prefill_tp_size // decode_tp_size
                    dst_head_offset = (local_tp_rank % prefill_ranks_per_decode) * num_kv_heads
                    logger.debug(
                        f"[MIXED-TP] prefill_tp={prefill_tp_size}, decode_tp={decode_tp_size}, "
                        f"local_tp_rank={local_tp_rank}, num_kv_heads={num_kv_heads}, "
                        f"prefill_ranks_per_decode={prefill_ranks_per_decode}, "
                        f"dst_head_offset={dst_head_offset}"
                    )
                else:
                    head_start = 0
                    num_heads_to_send = num_kv_heads
                    dst_head_offset = 0

                # Accumulate kv_indices across chunks for this request.
                pending_kv = self._cpu_pending_kv.setdefault(bootstrap_room, [])
                pending_kv.append(kv_indices)

                if is_last:
                    # Combine all accumulated chunks into one contiguous array.
                    all_kv_indices = (
                        np.concatenate(pending_kv) if len(pending_kv) > 1
                        else pending_kv[0]
                    )
                    del self._cpu_pending_kv[bootstrap_room]

                    # Use per-request allocated ptr if available, else fall back to static pool start
                    effective_dst_ptr = req.dst_pinned_ptr if req.dst_pinned_ptr != 0 else decode_info.dst_pinned_ptr
                    effective_dst_size = req.dst_pinned_size if req.dst_pinned_size != 0 else decode_info.dst_pinned_size

                    # Always use chunk_id=0 / is_last=True for the CPU buffer
                    # path: we emit exactly one KV notification per request.
                    cpu_notif = f"{req.room}_kv_0_1_{self.kv_args.pp_rank}"
                    kv_event, kv_post_fn = self.send_kvcache_triton(
                        peer_name=req.agent_name,
                        prefill_kv_indices=all_kv_indices,
                        dst_pinned_ptr=effective_dst_ptr,
                        dst_pinned_size=effective_dst_size,
                        notif=cpu_notif,
                        head_start=head_start,
                        num_heads_to_send=num_heads_to_send,
                        dst_head_offset=dst_head_offset,
                    )
                    pending_posts.append((kv_event, kv_post_fn))
                # else: not the last chunk — accumulate only, defer send.
            elif self.is_mla_backend or (decode_tp_size == self.attn_tp_size):
                kv_xfer_handle = self.send_kvcache(
                    req.agent_name,
                    kv_indices,
                    decode_info.dst_kv_ptrs,
                    chunked_dst_kv_indice,
                    decode_info.gpu_id,
                    notif,
                )
            else:
                kv_xfer_handle = self.send_kvcache_slice(
                    req.agent_name,
                    kv_indices,
                    decode_info.dst_kv_ptrs,
                    chunked_dst_kv_indice,
                    decode_info.gpu_id,
                    notif,
                    prefill_tp_size=self.attn_tp_size,
                    decode_tp_size=decode_tp_size,
                    decode_tp_rank=decode_info.decode_tp_rank,
                    dst_kv_item_len=decode_info.dst_kv_item_len,
                )

            if kv_xfer_handle is not None:
                handles.append(kv_xfer_handle)
            # Only the last chunk we need to send the aux data.
            if is_last:
                if state_indices is not None:
                    dst_info = self.decode_kv_args_table[req.agent_name]
                    state_xfer_handle = self.maybe_send_extra(
                        req.agent_name,
                        state_indices,
                        dst_info.dst_state_data_ptrs,
                        req.dst_state_indices,
                        dst_info.gpu_id,
                        f"{req.room}_state_{self.kv_args.pp_rank}",
                        decode_tp_size,
                    )
                    if state_xfer_handle is not None:
                        handles.append(state_xfer_handle)

                assert aux_index is not None
                aux_xfer_handle = self.send_aux(
                    req.agent_name,
                    aux_index,
                    decode_info.dst_aux_ptrs,
                    req.dst_aux_index,
                    f"{req.room}_aux",
                )
                handles.append(aux_xfer_handle)
        if is_last:
            del self.transfer_infos[bootstrap_room]
        return handles, pool_allocations, pending_posts

    def _drain_deferred_pool_releases(self) -> None:
        """Release pinned-buffer regions whose scatter kernels have completed.

        Checks each pending (CUDA event, pool offset) pair.  If the event has
        fired (GPU kernel done), the pool region is released immediately so it
        can be reused by the next NIXL transfer.  Pending entries whose events
        have *not* yet fired are kept for the next call.

        This is called both from ``update_transfer_status()`` (normal poll path)
        and from ``NixlKVReceiver.init()`` *before* blocking on pool allocation,
        to avoid a deadlock where the allocator waits for space that would only
        be freed after ``poll()`` runs on an already-allocated request.
        """
        if not self._pending_pool_releases or self._pinned_pool is None:
            return
        remaining = []
        for event, offset in self._pending_pool_releases:
            if event.query():
                self._pinned_pool.release(offset)
            else:
                remaining.append((event, offset))
        self._pending_pool_releases = remaining

    def update_transfer_status(self):
        # Drain deferred pinned-buffer releases from completed scatter kernels.
        self._drain_deferred_pool_releases()

        # Process notifications from received transfers.
        notif_map = self.agent.get_new_notifs()
        for peer_name, messages in notif_map.items():
            # Use peer_name as the unique sender key. This correctly handles
            # mixed TP where multiple prefill TP ranks (each with a unique
            # NIXL agent/peer_name) send to the same decode rank.
            for msg in messages:
                components = msg.decode("ascii").split("_", 4)
                room = int(components[0])
                if components[1] == "kv":
                    chunk_id = int(components[2])
                    is_last = bool(int(components[3]))
                    sender_key = peer_name
                    # Track received chunks per sender
                    self.transfer_statuses[room].received_kvs_per_sender[
                        sender_key
                    ].add(chunk_id)
                    if is_last:
                        # Record expected chunk count for this sender
                        self.transfer_statuses[room].expected_kvs_per_sender[
                            sender_key
                        ] = (chunk_id + 1)
                        # Set num_senders_expected from table (or default to 1)
                        if self.transfer_statuses[room].num_senders_expected is None:
                            self.transfer_statuses[room].num_senders_expected = (
                                self.required_prefill_response_num_table.get(room, 1)
                            )
                elif components[1] == "aux":
                    self.transfer_statuses[room].received_aux = True
                elif components[1] == "state":
                    sender_key = peer_name
                    self.transfer_statuses[room].received_state_per_sender.add(sender_key)

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
        self.xfer_handles = []
        # Track pool allocations for release when transfer completes
        self._pool_allocations: List[tuple] = []
        # Pending (event, post_fn) pairs: NIXL not yet posted, waiting for gather kernel
        self._pending_posts: List[tuple] = []
        self.has_sent = False
        self.chunk_id = 0

    def send(
        self,
        kv_indices: npt.NDArray[np.int32],
        state_indices: Optional[List[int]] = None,
    ):
        index_slice = slice(self.curr_idx, self.curr_idx + len(kv_indices))
        self.curr_idx += len(kv_indices)
        is_last = self.curr_idx == self.num_kv_indices

        new_xfer_handles, new_pool_allocations, new_pending_posts = (
            self.kv_mgr.add_transfer_request(
                self.bootstrap_room,
                kv_indices,
                index_slice,
                is_last,
                self.chunk_id,
                self.aux_index,
                state_indices,
            )
        )
        self.xfer_handles.extend(new_xfer_handles)
        self._pool_allocations.extend(new_pool_allocations)
        self._pending_posts.extend(new_pending_posts)
        self.chunk_id += 1
        if is_last:
            self.has_sent = True
            del self.kv_mgr.request_status[self.bootstrap_room]

    def poll(self) -> KVPoll:
        if not self.has_sent:
            return self.kv_mgr.check_status(self.bootstrap_room)

        # Drain pending gather events: once a CUDA event fires, post the NIXL transfer.
        if self._pending_posts:
            remaining = []
            for event, post_fn in self._pending_posts:
                if event.query():
                    new_handles, new_allocs = post_fn()
                    self.xfer_handles.extend(new_handles)
                    self._pool_allocations.extend(new_allocs)
                else:
                    remaining.append((event, post_fn))
            self._pending_posts = remaining
            if self._pending_posts:
                return KVPoll.WaitingForInput  # type: ignore

        if not self.xfer_handles:
            return KVPoll.WaitingForInput  # type: ignore

        states = [self.kv_mgr.agent.check_xfer_state(x) for x in self.xfer_handles]
        if all([x == "DONE" for x in states]):
            # Release pool allocations now that all transfers are complete
            for pool, offset in self._pool_allocations:
                pool.release(offset)
            self._pool_allocations.clear()
            return KVPoll.Success  # type: ignore
        if any([x == "ERR" for x in states]):
            # Release pool allocations on error too
            for pool, offset in self._pool_allocations:
                pool.release(offset)
            self._pool_allocations.clear()
            raise Exception("KVSender transfer encountered an error.")
        return KVPoll.WaitingForInput  # type: ignore

    def failure_exception(self):
        raise RuntimeError("NIXL KVSender Exception")


class NixlKVReceiver(CommonKVReceiver):
    def __init__(
        self,
        mgr: NixlKVManager,
        bootstrap_addr: str,
        bootstrap_room: Optional[int] = None,
        prefill_dp_rank: Optional[int] = None,
    ):
        self.started_transfer = False
        self.conclude_state = None
        super().__init__(mgr, bootstrap_addr, bootstrap_room, prefill_dp_rank)

        # Track this room with its bootstrap address for heartbeat monitoring
        if hasattr(self.kv_mgr, "addr_to_rooms_tracker"):
            self.kv_mgr.addr_to_rooms_tracker[self.bootstrap_addr].add(
                self.bootstrap_room
            )
        self.init_time = None
        # Store kv_indices for Triton scatter after transfer completes
        self._triton_kv_indices: Optional[npt.NDArray[np.int32]] = None
        self._triton_scatter_done = False
        # Per-request pinned buffer allocation on the receive side
        self._recv_pool_offset: Optional[int] = None
        self._recv_pool_buffer_view: Optional[torch.Tensor] = None

    def init(
        self,
        kv_indices: npt.NDArray[np.int32],
        aux_index: Optional[int] = None,
        state_indices: Optional[List[int]] = None,
    ):
        if self.bootstrap_infos is None:
            logger.error(
                f"Could not fetch prefill parallel info from bootstrap_addr: {self.bootstrap_addr}",
            )
            self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Failed)
            return

        # For CPU buffer transfers, allocate a per-request region from the receive pool.
        # Each concurrent request gets its own region so NIXL writes don't overwrite each other.
        # We send the allocated ptr to the prefill so it writes to our unique offset.
        recv_pinned_ptr = 0
        recv_pinned_size = 0
        if (
            self.kv_mgr.nixl_use_cpu_buffer
            and self.kv_mgr._pinned_pool is not None
            and self.kv_mgr.kv_args.k_buffers is not None
        ):
            k_buffers = self.kv_mgr.kv_args.k_buffers
            num_layers = len(k_buffers)
            num_heads = k_buffers[0].shape[1]
            head_dim = k_buffers[0].shape[2]
            bytes_per_element = k_buffers[0].element_size()
            num_tokens = len(kv_indices) * self.kv_mgr.kv_args.page_size
            recv_pinned_size = (
                num_layers * 2 * num_tokens * num_heads * head_dim * bytes_per_element
            )
            # Drain any deferred releases from completed scatter kernels before
            # allocating, so we don't block if a previous request's scatter has
            # already finished but its pool region hasn't been freed yet.
            self.kv_mgr._drain_deferred_pool_releases()
            recv_offset, recv_buffer_view = self.kv_mgr._pinned_pool.allocate(
                recv_pinned_size
            )
            self._recv_pool_offset = recv_offset
            self._recv_pool_buffer_view = recv_buffer_view
            recv_pinned_ptr = self.kv_mgr._pinned_pool.buffer.data_ptr() + recv_offset
            logger.warning(
                f"[DBG-ALLOC] room={self.bootstrap_room} offset={recv_offset} "
                f"ptr=0x{recv_pinned_ptr:x} size={recv_pinned_size}"
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
                        (
                            np.array(state_indices, dtype=np.int32).tobytes()
                            if not is_dummy and state_indices is not None
                            else b""
                        ),
                        str(recv_pinned_ptr).encode("ascii"),
                        str(recv_pinned_size).encode("ascii"),
                    ]
                )

        # Mark that we expect state data if state_indices was provided
        if state_indices is not None:
            self.kv_mgr.transfer_statuses[self.bootstrap_room].expects_state = True

        self.started_transfer = True
        self.init_time = time.time()

        # Store kv_indices for Triton scatter after transfer completes
        if self.kv_mgr.nixl_use_cpu_buffer:
            self._triton_kv_indices = kv_indices.copy()
            self._triton_scatter_done = False

    def poll(self) -> KVPoll:
        if self.conclude_state is not None:
            return self.conclude_state
        status = self.kv_mgr.check_status(self.bootstrap_room)
        if status in (KVPoll.Success, KVPoll.Failed):
            self.conclude_state = status
            return status
        if not self.started_transfer:
            return KVPoll.WaitingForInput  # type: ignore

        now = time.time()
        elapsed = now - self.init_time

        if elapsed >= self.kv_mgr.waiting_timeout:
            logger.error(f"Request {self.bootstrap_room} waiting_timeout")
            self.kv_mgr.record_failure(
                self.bootstrap_room,
                f"Request {self.bootstrap_room} timed out after {elapsed:.1f}s in KVPoll.WaitingForInput",
            )
            self._release_recv_pool()
            self.conclude_state = KVPoll.Failed
            return KVPoll.Failed

        self.kv_mgr.update_transfer_status()
        if self.kv_mgr.check_transfer_done(self.bootstrap_room):  # type: ignore
            self.kv_mgr.addr_to_rooms_tracker[self.bootstrap_addr].discard(
                self.bootstrap_room
            )
            # Check if the transfer failed
            if self.kv_mgr.transfer_statuses[self.bootstrap_room].is_failed():
                self._release_recv_pool()
                self.conclude_state = KVPoll.Failed
                logger.error(
                    f"Transfer for room {self.bootstrap_room} failed due to node failure"
                )
            else:
                # For CPU buffer transfer, scatter received data from pinned buffer to GPU KV cache
                if (
                    self.kv_mgr.nixl_use_cpu_buffer
                    and self._triton_kv_indices is not None
                    and not self._triton_scatter_done
                ):
                    try:
                        # Checksum the receive buffer AFTER the NIXL notification
                        # confirms the transfer is done, but BEFORE the scatter
                        # kernel reads it.  Compare to the prefill-side checksum
                        # in the prefill log for the same room to detect:
                        #   - all-zero buffer  → NIXL write never reached this buffer
                        #   - value mismatch   → wrong buffer address / pool aliasing
                        #   - partial zeros    → transfer not fully written yet
                        buf_ptr = self._recv_pool_buffer_view.data_ptr() if self._recv_pool_buffer_view is not None else 0
                        logger.warning(
                            f"[DBG-SCATTER] room={self.bootstrap_room} "
                            f"buf_ptr=0x{buf_ptr:x} offset={self._recv_pool_offset}"
                        )
                        if _NIXL_DEBUG_CHECKSUM and self._recv_pool_buffer_view is not None:
                            _kv_checksum(
                                self._recv_pool_buffer_view,
                                "DECODE-BEFORE-SCATTER",
                                self.bootstrap_room,
                            )
                        self.kv_mgr.scatter_received_kv(
                            kv_indices=self._triton_kv_indices,
                            head_start=0,
                            num_heads_received=None,
                            pinned_buffer=self._recv_pool_buffer_view,
                        )
                        self._triton_scatter_done = True
                        # Defer pool release until scatter kernel completes on GPU.
                        # Record a CUDA event now (after kernel launch) and hand the
                        # offset to the manager's deferred-release list.  The pool
                        # region must not be reused until the GPU kernel has finished
                        # reading from it, otherwise a concurrent NIXL write could
                        # corrupt the buffer before the scatter is done.
                        self._defer_recv_pool_release()
                    except Exception as e:
                        logger.error(
                            f"[TRITON-KV] Scatter failed: room={self.bootstrap_room}, error={e}"
                        )
                        self._release_recv_pool()
                        self.conclude_state = KVPoll.Failed
                        del self.kv_mgr.transfer_statuses[self.bootstrap_room]
                        return KVPoll.Failed

                self.conclude_state = KVPoll.Success
            del self.kv_mgr.transfer_statuses[self.bootstrap_room]
            return self.conclude_state  # type: ignore
        return KVPoll.WaitingForInput  # type: ignore

    def _release_recv_pool(self):
        """Release the per-request pinned buffer allocation immediately."""
        if self._recv_pool_offset is not None and self.kv_mgr._pinned_pool is not None:
            self.kv_mgr._pinned_pool.release(self._recv_pool_offset)
            self._recv_pool_offset = None
            self._recv_pool_buffer_view = None

    def _defer_recv_pool_release(self):
        """Defer pinned-buffer release until the scatter kernel finishes on GPU.

        Records a CUDA event on the current stream immediately after the scatter
        kernel launch and hands the (event, offset) pair to the manager's
        deferred-release list.  The pool region is freed in
        ``update_transfer_status()`` once ``event.query()`` returns True.
        """
        if self._recv_pool_offset is None:
            return
        event = torch.cuda.Event()
        event.record()
        self.kv_mgr._pending_pool_releases.append((event, self._recv_pool_offset))
        self._recv_pool_offset = None
        self._recv_pool_buffer_view = None

    def _register_kv_args(self):
        for bootstrap_info in self.bootstrap_infos:
            sock, lock = self._connect_to_bootstrap_server(bootstrap_info)
            packed_kv_data_ptrs = b"".join(
                struct.pack("Q", ptr) for ptr in self.kv_mgr.kv_args.kv_data_ptrs
            )
            packed_aux_data_ptrs = b"".join(
                struct.pack("Q", ptr) for ptr in self.kv_mgr.kv_args.aux_data_ptrs
            )
            packed_state_data_ptrs = b"".join(
                struct.pack("Q", ptr) for ptr in self.kv_mgr.kv_args.state_data_ptrs
            )

            # Get pinned buffer info for CPU buffer KV transfer
            pinned_ptr = 0
            pinned_size = 0
            if (
                self.kv_mgr.nixl_use_cpu_buffer
                and self.kv_mgr._pinned_pool is not None
            ):
                pinned_ptr, pinned_size = self.kv_mgr._pinned_pool.get_buffer_info()

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
                        str(self.kv_mgr.kv_args.decode_tp_size).encode("ascii"),
                        str(self.kv_mgr.kv_args.engine_rank).encode("ascii"),
                        str(self.kv_mgr.kv_args.kv_item_lens[0]).encode("ascii"),
                        str(pinned_ptr).encode("ascii"),
                        str(pinned_size).encode("ascii"),
                    ]
                )

    def failure_exception(self):
        raise RuntimeError("NIXL KVReceiver Exception")


class NixlKVBootstrapServer(CommonKVBootstrapServer):
    pass
