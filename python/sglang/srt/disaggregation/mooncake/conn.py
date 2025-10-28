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
from typing import Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import requests
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
    group_concurrent_contiguous,
)
from sglang.srt.disaggregation.mooncake.transfer_engine import MooncakeTransferEngine
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import (
    format_tcp_address,
    get_bool_env_var,
    get_int_env_var,
    is_valid_ipv6_address,
)

logger = logging.getLogger(__name__)


class KVTransferError(Exception):
    def __init__(self, bootstrap_room: int, failure_reason: str):
        super().__init__(failure_reason)
        self.bootstrap_room = bootstrap_room
        self.failure_reason = failure_reason

    def __str__(self):
        return f"KVTransferError(bootstrap_room={self.bootstrap_room}): {self.failure_reason}"


# Multimodal Embedding Error
class EmbeddingTransferError(Exception):
    def __init__(self, bootstrap_room: int, failure_reason: str):
        super().__init__(failure_reason)
        self.bootstrap_room = bootstrap_room
        self.failure_reason = failure_reason

    def __str__(self):
        return f"EmbeddingTransferError(bootstrap_room={self.bootstrap_room}): {self.failure_reason}"


# prefill
@dataclasses.dataclass
class TransferKVChunk:
    room: int
    prefill_kv_indices: npt.NDArray[np.int32]
    index_slice: slice
    is_last: bool
    prefill_aux_index: Optional[int]


# embedding (multimodal)
@dataclasses.dataclass
class TransferEmbeddingChunk:
    room: int
    embedding_indices: List[int]  # Source embedding indices
    is_last: bool
    total_tokens: int  # Total number of tokens to transfer


# decode
@dataclasses.dataclass
class TransferInfo:
    room: int
    endpoint: str
    dst_port: int
    mooncake_session_id: str
    dst_kv_indices: npt.NDArray[np.int32]
    dst_aux_index: int
    required_dst_info_num: int
    is_dummy: bool

    @classmethod
    def from_zmq(cls, msg: List[bytes]):
        if msg[4] == b"" and msg[5] == b"":
            is_dummy = True
            dst_kv_indices = np.array([], dtype=np.int32)
            dst_aux_index = None
        else:
            dst_kv_indices = np.frombuffer(msg[4], dtype=np.int32)
            dst_aux_index = int(msg[5].decode("ascii"))
            is_dummy = False
        return cls(
            room=int(msg[0].decode("ascii")),
            endpoint=msg[1].decode("ascii"),
            dst_port=int(msg[2].decode("ascii")),
            mooncake_session_id=msg[3].decode("ascii"),
            dst_kv_indices=dst_kv_indices,
            dst_aux_index=dst_aux_index,
            required_dst_info_num=int(msg[6].decode("ascii")),
            is_dummy=is_dummy,
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
    dst_tp_rank: int
    dst_attn_tp_size: int
    dst_kv_item_len: int

    @classmethod
    def from_zmq(cls, msg: List[bytes]):
        return cls(
            room=str(msg[0].decode("ascii")),
            endpoint=msg[1].decode("ascii"),
            dst_port=int(msg[2].decode("ascii")),
            mooncake_session_id=msg[3].decode("ascii"),
            dst_kv_ptrs=list(struct.unpack(f"{len(msg[4])//8}Q", msg[4])),
            dst_aux_ptrs=list(struct.unpack(f"{len(msg[5])//8}Q", msg[5])),
            dst_tp_rank=int(msg[6].decode("ascii")),
            dst_attn_tp_size=int(msg[7].decode("ascii")),
            dst_kv_item_len=int(msg[8].decode("ascii")),
        )


# language (multimodal)
@dataclasses.dataclass
class TransferEmbeddingInfo:
    room: int
    endpoint: str
    dst_port: int
    mooncake_session_id: str
    dst_embedding_indices: List[int]
    required_dst_info_num: int
    sent_tokens: int = 0  # Number of tokens already sent (for resume transfer)
    allocated_tokens: int = 0  # Number of tokens allocated by Language side
    # For resume: need to store original embedding data to retrigger transfer
    src_embedding_indices: List[int] = (
        None  # Source embedding indices (from Embedding side)
    )
    total_tokens: int = 0  # Total tokens to transfer (from Embedding side)
    resume_ready: bool = False  # Whether ready for resume transfer

    @classmethod
    def from_zmq(cls, msg: List[bytes]):
        # Parse embedding_indices from message
        # Format: comma-separated list of embedding indices
        dst_embedding_indices_str = msg[4].decode("ascii")
        if dst_embedding_indices_str:
            dst_embedding_indices = [
                int(x) for x in dst_embedding_indices_str.split(",")
            ]
        else:
            dst_embedding_indices = []

        required_dst_info_num = int(msg[5].decode("ascii"))

        # Parse allocated_tokens (always present in init message, msg[6])
        # For resume messages, msg[6] is sent_tokens, msg[7] is allocated_tokens
        allocated_tokens = 0
        sent_tokens = 0

        if len(msg) >= 7:
            # Check if this is a resume message (has 8 fields) or init message (has 7 fields)
            if len(msg) >= 8:
                # Resume message: msg[6] = sent_tokens, msg[7] = allocated_tokens
                sent_tokens = int(msg[6].decode("ascii"))
                allocated_tokens = int(msg[7].decode("ascii"))
            else:
                # Init message: msg[6] = allocated_tokens
                allocated_tokens = int(msg[6].decode("ascii"))

        return cls(
            room=int(msg[0].decode("ascii")),
            endpoint=msg[1].decode("ascii"),
            dst_port=int(msg[2].decode("ascii")),
            mooncake_session_id=msg[3].decode("ascii"),
            dst_embedding_indices=dst_embedding_indices,
            required_dst_info_num=required_dst_info_num,
            sent_tokens=sent_tokens,
            allocated_tokens=allocated_tokens,
        )


@dataclasses.dataclass
class EmbeddingArgsRegisterInfo:
    room: str
    endpoint: str
    dst_port: int
    mooncake_session_id: str
    dst_embedding_ptrs: list[int]

    @classmethod
    def from_zmq(cls, msg: List[bytes]):
        return cls(
            room=str(msg[0].decode("ascii")),
            endpoint=msg[1].decode("ascii"),
            dst_port=int(msg[2].decode("ascii")),
            mooncake_session_id=msg[3].decode("ascii"),
            dst_embedding_ptrs=list(struct.unpack(f"{len(msg[4])//8}Q", msg[4])),
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


class MooncakeKVManager(CommonKVManager):
    AUX_DATA_HEADER = b"AUX_DATA"

    def __init__(
        self,
        args: KVArgs,
        disaggregation_mode: DisaggregationMode,
        server_args: ServerArgs,
        is_mla_backend: Optional[bool] = False,
        is_multimodal: bool = False,  # Support multimodal embedding/language mode
    ):
        self.is_multimodal = is_multimodal
        super().__init__(args, disaggregation_mode, server_args, is_mla_backend)
        self.init_engine()
        self.register_buffer_to_engine()

        # Check if this is sender mode (PREFILL or ENCODE)
        self.is_sender_mode = (
            self.disaggregation_mode == DisaggregationMode.PREFILL
            or (is_multimodal and self.disaggregation_mode == DisaggregationMode.ENCODE)
        )

        if self.is_sender_mode:
            self.start_prefill_thread()
            self.session_failures = defaultdict(int)
            self.failed_sessions = set()
            self.session_lock = threading.Lock()
            # Determine the number of threads to use for kv sender
            cpu_count = os.cpu_count()
            transfer_thread_pool_size = get_int_env_var(
                "SGLANG_DISAGGREGATION_THREAD_POOL_SIZE",
                min(max(4, int(0.75 * cpu_count) // 8), 12),
            )
            transfer_queue_size = get_int_env_var("SGLANG_DISAGGREGATION_QUEUE_SIZE", 4)
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
            for queue, executor in zip(self.transfer_queues, self.executors):
                threading.Thread(
                    target=self.transfer_worker, args=(queue, executor), daemon=True
                ).start()
            # If a timeout happens on the prefill side, it means prefill instances
            # fail to receive the KV indices from the decode instance of this request.
            # These timeout requests should be aborted to release the tree cache.
            self.bootstrap_timeout = get_int_env_var(
                "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT", 300
            )

            self.enable_custom_mem_pool = get_bool_env_var(
                "SGLANG_MOONCAKE_CUSTOM_MEM_POOL", "false"
            )

        # Check if this is receiver mode (DECODE or LANGUAGE)
        self.is_receiver_mode = (
            self.disaggregation_mode == DisaggregationMode.DECODE
            or (
                is_multimodal
                and self.disaggregation_mode == DisaggregationMode.LANGUAGE
            )
        )

        if self.is_receiver_mode:
            self.heartbeat_failures = {}
            self.session_pool = defaultdict(requests.Session)
            self.session_pool_lock = threading.Lock()
            self.addr_to_rooms_tracker = defaultdict(set)
            # Both DECODE and LANGUAGE modes need prefill_response_tracker
            # (LANGUAGE receives from ENCODE, similar to DECODE receiving from PREFILL)
            self.prefill_response_tracker: Dict[int, Set[int]] = defaultdict(set)
            # Heartbeat interval should be at least 2 seconds
            self.heartbeat_interval = max(
                float(os.getenv("SGLANG_DISAGGREGATION_HEARTBEAT_INTERVAL", 5.0)), 2.0
            )
            # Heartbeat failure should be at least 1
            self.max_failures = max(
                get_int_env_var("SGLANG_DISAGGREGATION_HEARTBEAT_MAX_FAILURE", 2), 1
            )
            self.start_decode_thread()
            # If a timeout happens on the decode side, it means decode instances
            # fail to receive the KV Cache transfer done signal after bootstrapping.
            # These timeout requests should be aborted to release the tree cache.
            self.waiting_timeout = get_int_env_var(
                "SGLANG_DISAGGREGATION_WAITING_TIMEOUT", 300
            )

        self.failure_records: Dict[int, str] = {}
        self.failure_lock = threading.Lock()

    def init_engine(self):
        self.engine = MooncakeTransferEngine(
            hostname=self.local_ip,
            gpu_id=self.kv_args.gpu_id,
            ib_device=self.kv_args.ib_device,
        )

    def register_buffer_to_engine(self):
        # For multimodal mode, only register aux_data (embeddings)
        if self.is_multimodal:
            # Only register aux data buffers for embedding/language mode
            if self.kv_args.aux_data_ptrs and self.kv_args.aux_data_lens:
                self.engine.batch_register(
                    self.kv_args.aux_data_ptrs, self.kv_args.aux_data_lens
                )
        else:
            # For KV mode, register both kv_data and aux_data
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

    def _transfer_data(self, mooncake_session_id, transfer_blocks):
        if not transfer_blocks:
            return 0

        src_addrs, dst_addrs, lengths = zip(*transfer_blocks)
        return self.engine.batch_transfer_sync(
            mooncake_session_id, list(src_addrs), list(dst_addrs), list(lengths)
        )

    def send_embedding(
        self,
        mooncake_session_id: str,
        embedding_indices: List[int],
        dst_embedding_ptrs: list[int],
        dst_embedding_indices: List[int],
        total_tokens: int,
        block_size: int,
        sent_tokens: int = 0,
        allocated_tokens: int = None,
    ):
        """Send embedding data using block-based transfer (multimodal mode).

        Args:
            mooncake_session_id: Session ID for transfer
            embedding_indices: Source embedding indices
            dst_embedding_ptrs: Destination buffer pointers
            dst_embedding_indices: Destination embedding indices
            total_tokens: Total number of tokens to transfer
            block_size: Number of tokens per block
            sent_tokens: Number of tokens already sent (for resume transfer)
            allocated_tokens: Number of tokens allocated by Language side

        Returns:
            Tuple of (ret, is_partial):
                ret: 0 if all transfers succeed, 1 otherwise
                is_partial: True if this is a partial transfer (more data remaining)
        """
        # Validate block_size consistency
        if allocated_tokens is not None:
            expected_block_size = allocated_tokens // len(dst_embedding_indices)
            if expected_block_size != block_size:
                raise ValueError(
                    f"Block size mismatch: Embedding side uses {block_size}, "
                    f"but Language side allocated {allocated_tokens} tokens "
                    f"for {len(dst_embedding_indices)} blocks "
                    f"(implies block_size={expected_block_size})"
                )
        else:
            # Backward compatibility: if no allocated_tokens, calculate from block count
            allocated_tokens = len(dst_embedding_indices) * block_size

        # Calculate remaining tokens and determine if this is a partial transfer
        remaining_tokens = total_tokens - sent_tokens

        if remaining_tokens > allocated_tokens:
            # Need partial transfer
            logger.debug(
                f"Partial transfer: remaining={remaining_tokens} > "
                f"allocated={allocated_tokens}. Will transfer {allocated_tokens} tokens."
            )
            tokens_to_send = allocated_tokens
            is_partial = True
        else:
            # Can transfer all remaining tokens
            tokens_to_send = remaining_tokens
            is_partial = False

        # Calculate required dst blocks
        dst_blocks_needed = (tokens_to_send + block_size - 1) // block_size

        # Validate dst buffer is sufficient
        if dst_blocks_needed > len(dst_embedding_indices):
            raise ValueError(
                f"Insufficient dst blocks: need {dst_blocks_needed} blocks "
                f"for {tokens_to_send} tokens, but only have {len(dst_embedding_indices)} blocks"
            )

        # Calculate source block range based on sent_tokens
        start_block = sent_tokens // block_size
        embedding_indices_to_send = embedding_indices[
            start_block : start_block + dst_blocks_needed
        ]
        dst_embedding_indices = dst_embedding_indices[:dst_blocks_needed]

        src_addrs = []
        dst_addrs = []
        lengths = []

        tokens_transferred = 0

        for block_idx, (src_block_idx, dst_block_idx) in enumerate(
            zip(embedding_indices_to_send, dst_embedding_indices)
        ):
            # Calculate tokens in this block
            remaining_in_transfer = tokens_to_send - tokens_transferred
            tokens_in_block = min(block_size, remaining_in_transfer)

            if tokens_in_block <= 0:
                break

            # Transfer each buffer type within the block
            for buffer_type_idx in range(len(self.kv_args.aux_item_lens)):
                embedding_item_len = self.kv_args.aux_item_lens[buffer_type_idx]

                # Calculate chunk size based on buffer type and tokens_in_block
                # For aux_datas, only transfer in first block of initial transfer
                if buffer_type_idx == 3:  # aux_datas
                    if sent_tokens == 0 and block_idx == 0:
                        chunk_size = embedding_item_len  # Transfer full aux_datas
                    else:
                        continue  # Skip aux_datas for resume or other blocks
                else:
                    # For embeddings, fill_ids, mrope_positions: scale by tokens_in_block
                    # embedding_item_len already contains the full block size
                    # We need to transfer only tokens_in_block portion
                    chunk_size = (embedding_item_len * tokens_in_block) // block_size

                # Calculate source address: base_ptr + src_block_idx * item_len
                embedding_addr = (
                    self.kv_args.aux_data_ptrs[buffer_type_idx]
                    + src_block_idx * embedding_item_len
                )

                # Calculate destination address: base_ptr + dst_block_idx * item_len
                dst_embedding_addr = (
                    dst_embedding_ptrs[buffer_type_idx]
                    + dst_block_idx * embedding_item_len
                )

                src_addrs.append(embedding_addr)
                dst_addrs.append(dst_embedding_addr)
                lengths.append(chunk_size)

            tokens_transferred += tokens_in_block

        ret = self.engine.batch_transfer_sync(
            mooncake_session_id, src_addrs, dst_addrs, lengths
        )

        return ret, is_partial

    def send_kvcache(
        self,
        mooncake_session_id: str,
        prefill_kv_indices: npt.NDArray[np.int32],
        dst_kv_ptrs: list[int],
        dst_kv_indices: npt.NDArray[np.int32],
        executor: concurrent.futures.ThreadPoolExecutor,
    ):
        # Group by indices
        prefill_kv_blocks, dst_kv_blocks = group_concurrent_contiguous(
            prefill_kv_indices, dst_kv_indices
        )

        layers_params = None

        # pp is not supported on the decode side yet
        if self.is_mla_backend:
            src_kv_ptrs, dst_kv_ptrs, layers_current_pp_stage = (
                self.get_mla_kv_ptrs_with_pp(self.kv_args.kv_data_ptrs, dst_kv_ptrs)
            )
            kv_item_len = self.kv_args.kv_item_lens[0]
            layers_params = [
                (
                    src_kv_ptrs[layer_id],
                    dst_kv_ptrs[layer_id],
                    kv_item_len,
                )
                for layer_id in range(layers_current_pp_stage)
            ]
        else:
            src_k_ptrs, src_v_ptrs, dst_k_ptrs, dst_v_ptrs, layers_current_pp_stage = (
                self.get_mha_kv_ptrs_with_pp(self.kv_args.kv_data_ptrs, dst_kv_ptrs)
            )
            kv_item_len = self.kv_args.kv_item_lens[0]
            layers_params = [
                (
                    src_k_ptrs[layer_id],
                    dst_k_ptrs[layer_id],
                    kv_item_len,
                )
                for layer_id in range(layers_current_pp_stage)
            ] + [
                (
                    src_v_ptrs[layer_id],
                    dst_v_ptrs[layer_id],
                    kv_item_len,
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
        else:
            # Combining all layers' params in one batch transfer is more efficient
            # compared to using multiple threads
            return process_layers(layers_params)

        return 0

    def send_kvcache_slice(
        self,
        mooncake_session_id: str,
        prefill_kv_indices: npt.NDArray[np.int64],
        dst_kv_ptrs: list[int],
        dst_kv_indices: npt.NDArray[np.int64],
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
        num_kv_heads = self.kv_args.kv_head_num
        num_layers = len(self.kv_args.kv_data_ptrs)
        page_size = self.kv_args.page_size

        # Calculate head distribution
        src_heads_per_rank = num_kv_heads
        dst_heads_per_rank = num_kv_heads * self.attn_tp_size // dst_attn_tp_size
        bytes_per_head_slice_to_send = (
            dst_kv_item_len // page_size // dst_heads_per_rank
        )

        # Determine slicing parameters based on TP configuration
        if self.attn_tp_size > dst_attn_tp_size:
            # Send KVCache from multiple prefill instances to 1 decode instance
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

        # Sanity check: The data sub-slice to be sent should fit into the dst buffer.
        # This means heads_bytes_per_token_to_send <= (dst_kv_item_len // page_size)
        if heads_bytes_per_token_to_send > (dst_kv_item_len // page_size):
            logger.error(
                f"[{mooncake_session_id}] slice size ({heads_bytes_per_token_to_send}) exceeds "
                f"target token slot size ({dst_kv_item_len // page_size})"
            )
            return -1

        layers_params = [
            (
                src_k_ptrs[layer_id],
                dst_k_ptrs[layer_id],
                src_kv_item_len,
                dst_kv_item_len,
                src_head_slice_offset,
                dst_head_slice_offset,
                heads_bytes_per_token_to_send,
            )
            for layer_id in range(layers_current_pp_stage)
        ] + [
            (
                src_v_ptrs[layer_id],
                dst_v_ptrs[layer_id],
                src_kv_item_len,
                dst_kv_item_len,
                src_head_slice_offset,
                dst_head_slice_offset,
                heads_bytes_per_token_to_send,
            )
            for layer_id in range(layers_current_pp_stage)
        ]

        def process_layer_tp_aware(layer_params):
            (
                src_ptr,
                dst_ptr,
                src_item_len,
                dst_item_len,
                src_head_slice_offset,
                dst_head_slice_offset,
                heads_bytes_per_token_to_send,
            ) = layer_params
            src_addr_list = []
            dst_addr_list = []
            length_list = []

            # Calculate strides for a single token slot
            bytes_per_token_on_prefill = src_item_len // page_size
            bytes_per_token_on_decode = dst_item_len // page_size

            for i in range(len(prefill_kv_indices)):
                prefill_page_idx = int(prefill_kv_indices[i])
                decode_page_idx = int(dst_kv_indices[i])

                # Get the starting addresses for the current src and dst pages
                src_page_start_addr = src_ptr + prefill_page_idx * src_item_len
                dst_page_start_addr = dst_ptr + decode_page_idx * dst_item_len

                # Iterate through each valid token slot within the current page
                for token_slot_in_page in range(page_size):
                    # Calculate the start address of the current token slot
                    src_token_slot_start_addr = (
                        src_page_start_addr
                        + token_slot_in_page * bytes_per_token_on_prefill
                    )
                    dst_token_slot_start_addr = (
                        dst_page_start_addr
                        + token_slot_in_page * bytes_per_token_on_decode
                    )

                    # Calculate final src and dst addresses by applying head-slice offsets
                    src_slice_addr = src_token_slot_start_addr + src_head_slice_offset
                    dst_slice_addr = dst_token_slot_start_addr + dst_head_slice_offset

                    src_addr_list.append(src_slice_addr)
                    dst_addr_list.append(dst_slice_addr)
                    length_list.append(heads_bytes_per_token_to_send)

            return self.engine.batch_transfer_sync(
                mooncake_session_id, src_addr_list, dst_addr_list, length_list
            )

        futures = [
            executor.submit(
                process_layer_tp_aware,
                layer_params,
            )
            for layer_params in layers_params
        ]

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
        if self.enable_custom_mem_pool:
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
        socket = self._connect(
            format_tcp_address(remote, dst_port), is_ipv6=is_valid_ipv6_address(remote)
        )

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

    def sync_status_to_receiver_endpoint(
        self, remote: str, dst_port: int, room: int, status: int, sender_rank: int = -1
    ):
        self._connect(
            format_tcp_address(remote, dst_port), is_ipv6=is_valid_ipv6_address(remote)
        ).send_multipart(
            [
                str(room).encode("ascii"),
                str(status).encode("ascii"),
                str(sender_rank).encode("ascii"),
            ]
        )

    def transfer_worker(
        self, queue: FastQueue, executor: concurrent.futures.ThreadPoolExecutor
    ):
        while True:
            try:
                # Get chunk from queue (could be KV or Embedding)
                chunk = queue.get()

                # Determine if this is multimodal mode
                if self.is_multimodal:
                    # Multimodal Embedding transfer
                    self._transfer_worker_embedding(chunk, executor)
                else:
                    # KV cache transfer
                    self._transfer_worker_kv(chunk, executor)

            except Exception as e:
                mode_name = "Embedding" if self.is_multimodal else "Prefill"
                raise RuntimeError(
                    f"Transfer thread failed because of {e}. "
                    f"{mode_name} instance with bootstrap_port={self.bootstrap_port} is dead."
                )

    def _transfer_worker_kv(
        self, kv_chunk: TransferKVChunk, executor: concurrent.futures.ThreadPoolExecutor
    ):
        """KV cache transfer worker"""
        reqs_to_be_processed = (
            self.transfer_infos[kv_chunk.room].values()
            if kv_chunk.room in self.transfer_infos
            else []
        )
        polls = []
        dst_ranks_infos = []
        local_rank = self.attn_tp_rank * self.pp_size + self.pp_rank
        for req in reqs_to_be_processed:
            if not req.is_dummy:
                # Early exit if the request has failed
                with self.session_lock:
                    if req.mooncake_session_id in self.failed_sessions:
                        self.record_failure(
                            kv_chunk.room,
                            f"Decode instance could be dead, remote mooncake session {req.mooncake_session_id} is not alive",
                        )
                        self.update_status(kv_chunk.room, KVPoll.Failed)
                        self.sync_status_to_receiver_endpoint(
                            req.endpoint,
                            req.dst_port,
                            req.room,
                            KVPoll.Failed,
                            local_rank,
                        )
                        break

                chunked_dst_kv_indice = req.dst_kv_indices[kv_chunk.index_slice]

                # NOTE: This is temporarily a workaround to deal with the case where the prefill_kv_indices
                # is mismatched with the dst_kv_indices when page size > 1, this should never happen.
                if len(chunked_dst_kv_indice) < len(kv_chunk.prefill_kv_indices):
                    logger.warning(
                        f"len(chunked_dst_kv_indice) = {len(chunked_dst_kv_indice)}, len(kv_chunk.prefill_kv_indices) = {len(kv_chunk.prefill_kv_indices)}"
                    )
                    kv_chunk.prefill_kv_indices = kv_chunk.prefill_kv_indices[
                        : len(chunked_dst_kv_indice)
                    ]

                target_rank_registration_info: KVArgsRegisterInfo = (
                    self.decode_kv_args_table[req.mooncake_session_id]
                )
                if self.is_mla_backend or (
                    self.attn_tp_size == target_rank_registration_info.dst_attn_tp_size
                ):
                    ret = self.send_kvcache(
                        req.mooncake_session_id,
                        kv_chunk.prefill_kv_indices,
                        target_rank_registration_info.dst_kv_ptrs,
                        chunked_dst_kv_indice,
                        executor,
                    )
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
                            logger.error(f"Session {req.mooncake_session_id} failed.")
                    self.record_failure(
                        kv_chunk.room,
                        f"Failed to send kv chunk of {kv_chunk.room} to {req.endpoint}:{req.dst_port}",
                    )
                    self.update_status(kv_chunk.room, KVPoll.Failed)
                    self.sync_status_to_receiver_endpoint(
                        req.endpoint,
                        req.dst_port,
                        req.room,
                        KVPoll.Failed,
                        local_rank,
                    )
                    break

                if kv_chunk.is_last:
                    if self.pp_group.is_last_rank:
                        # Only the last chunk we need to send the aux data
                        ret = self.send_aux(
                            req,
                            kv_chunk.prefill_aux_index,
                            target_rank_registration_info.dst_aux_ptrs,
                        )
                    polls.append(True if ret == 0 else False)
                    dst_ranks_infos.append((req.endpoint, req.dst_port, req.room))

                    # Only sync status when all the dst ranks have received the kvcache
                    if len(polls) == req.required_dst_info_num:
                        status = KVPoll.Success if all(polls) else KVPoll.Failed
                        self.update_status(req.room, status)
                        for endpoint, dst_port, room in dst_ranks_infos:
                            self.sync_status_to_receiver_endpoint(
                                endpoint, dst_port, room, status, local_rank
                            )
            else:
                # Dummy request means the decode instance is not used, so its status can be marked as success directly
                # Dummy request does not need to sync status to decode endpoint
                if kv_chunk.is_last and req.room in self.request_status:
                    self.update_status(req.room, KVPoll.Success)

        if (
            kv_chunk.room not in self.request_status
            or self.check_status(kv_chunk.room) == KVPoll.Success
        ):
            if kv_chunk.room in self.transfer_infos:
                self.transfer_infos.pop(kv_chunk.room)

    def _transfer_worker_embedding(
        self,
        embedding_chunk: TransferEmbeddingChunk,
        executor: concurrent.futures.ThreadPoolExecutor,
    ):
        """Embedding transfer worker (with Resume Transfer support)"""
        reqs_to_be_processed = (
            self.transfer_infos[embedding_chunk.room].values()
            if embedding_chunk.room in self.transfer_infos
            else []
        )
        polls = []
        dst_ranks_infos = []
        local_rank = self.attn_tp_rank * self.pp_size + self.pp_rank

        for req in reqs_to_be_processed:
            # Early exit if the request has failed
            with self.session_lock:
                if req.mooncake_session_id in self.failed_sessions:
                    self.record_failure(
                        embedding_chunk.room,
                        f"Language instance could be dead, remote mooncake session {req.mooncake_session_id} is not alive",
                    )
                    self.update_status(embedding_chunk.room, KVPoll.Failed)
                    self.sync_status_to_receiver_endpoint(
                        req.endpoint,
                        req.dst_port,
                        req.room,
                        KVPoll.Failed,
                        local_rank,
                    )
                    break

            # Save source embedding info for potential resume
            if req.src_embedding_indices is None:
                req.src_embedding_indices = embedding_chunk.embedding_indices
                req.total_tokens = embedding_chunk.total_tokens

            # Block-based transfer
            # Calculate block_size from aux_item_lens
            # aux_item_lens[1] is for fill_ids per block
            # block_size = aux_item_lens[1] / fill_ids.itemsize
            # Assuming fill_ids is int32 (4 bytes)
            block_size = self.kv_args.aux_item_lens[1] // 4

            # Get sent_tokens and allocated_tokens from transfer_info
            sent_tokens = req.sent_tokens
            allocated_tokens = req.allocated_tokens

            ret, is_partial = self.send_embedding(
                req.mooncake_session_id,
                embedding_chunk.embedding_indices,
                self.decode_kv_args_table[req.mooncake_session_id].dst_embedding_ptrs,
                req.dst_embedding_indices,
                embedding_chunk.total_tokens,
                block_size,
                sent_tokens,
                allocated_tokens,
            )

            if ret != 0:
                with self.session_lock:
                    self.session_failures[req.mooncake_session_id] += 1
                    # Failures should never happen if the session is not dead, if the session fails once, mark it as failed
                    if self.session_failures[req.mooncake_session_id] >= 1:
                        self.failed_sessions.add(req.mooncake_session_id)
                        logger.error(f"Session {req.mooncake_session_id} failed.")
                    logger.error(
                        f"Session {req.mooncake_session_id} failed with {embedding_chunk.room=};{req.endpoint=};{req.dst_port=};{req.room=}"
                    )
                self.record_failure(
                    embedding_chunk.room,
                    f"Failed to send embedding chunk of {embedding_chunk.room} to {req.endpoint}:{req.dst_port}",
                )
                self.update_status(embedding_chunk.room, KVPoll.Failed)
                self.sync_status_to_receiver_endpoint(
                    req.endpoint, req.dst_port, req.room, KVPoll.Failed, local_rank
                )
                break

            # Update sent_tokens after successful transfer
            tokens_sent = min(
                embedding_chunk.total_tokens - sent_tokens, allocated_tokens
            )
            req.sent_tokens += tokens_sent

            polls.append(True if ret == 0 else False)
            dst_ranks_infos.append((req.endpoint, req.dst_port, req.room))

            # Only sync status when all the dst ranks have received the embedding data
            if len(polls) == req.required_dst_info_num:
                if is_partial:
                    # Partial transfer complete, waiting for resume
                    status = KVPoll.Transferring if all(polls) else KVPoll.Failed
                else:
                    # Complete transfer done
                    status = KVPoll.Success if all(polls) else KVPoll.Failed

                self.update_status(req.room, status)
                for endpoint, dst_port, room in dst_ranks_infos:
                    self.sync_status_to_receiver_endpoint(
                        endpoint, dst_port, room, status, local_rank
                    )

        if (
            embedding_chunk.room not in self.request_status
            or self.check_status(embedding_chunk.room) == KVPoll.Success
        ):
            if embedding_chunk.room in self.transfer_infos:
                self.transfer_infos.pop(embedding_chunk.room)

    def start_prefill_thread(self):
        self._bind_server_socket()

        def bootstrap_thread():
            """This thread recvs pre-alloc notification from the decode/language engine"""
            # KVPoll.Bootstrapping -> KVPoll.WaitingForInput
            while True:
                waiting_req_bytes = self.server_socket.recv_multipart()
                room = waiting_req_bytes[0].decode("ascii")
                mooncake_session_id = waiting_req_bytes[3].decode("ascii")
                if room == "None":
                    # Register KV/Embedding Args
                    if self.is_multimodal:
                        self.decode_kv_args_table[mooncake_session_id] = (
                            EmbeddingArgsRegisterInfo.from_zmq(waiting_req_bytes)
                        )
                    else:
                        self.decode_kv_args_table[mooncake_session_id] = (
                            KVArgsRegisterInfo.from_zmq(waiting_req_bytes)
                        )
                    with self.session_lock:
                        if mooncake_session_id in self.failed_sessions:
                            self.failed_sessions.remove(mooncake_session_id)
                        if mooncake_session_id in self.session_failures:
                            del self.session_failures[mooncake_session_id]
                    logger.debug(
                        f"Register {'Embedding' if self.is_multimodal else 'KV'}Args from {mooncake_session_id} successfully"
                    )
                    continue
                else:
                    room = int(room)
                    if self.is_multimodal:
                        # Multimodal Embedding mode: handle resume transfer
                        # Check if this is a resume request (8 fields) or init request (7 fields)
                        is_resume = len(waiting_req_bytes) >= 8

                        if is_resume:
                            # Resume request: update existing transfer_info and trigger transfer
                            if (
                                room in self.transfer_infos
                                and mooncake_session_id in self.transfer_infos[room]
                            ):
                                transfer_info = TransferEmbeddingInfo.from_zmq(
                                    waiting_req_bytes
                                )
                                req = self.transfer_infos[room][mooncake_session_id]

                                # Update the existing transfer_info with resume data
                                req.sent_tokens = transfer_info.sent_tokens
                                req.allocated_tokens = transfer_info.allocated_tokens
                                req.dst_embedding_indices = (
                                    transfer_info.dst_embedding_indices
                                )

                                logger.debug(
                                    f"Resume transfer for room={room}, sent_tokens={transfer_info.sent_tokens}, "
                                    f"allocated_tokens={transfer_info.allocated_tokens}"
                                )

                                req.resume_ready = True
                                # Check if all sessions are ready for resume (similar to init logic)
                                all_dst_ranks_ready = all(
                                    dst_req.resume_ready
                                    for dst_req in self.transfer_infos[room].values()
                                )

                                # Only trigger resume transfer when all dst ranks are ready
                                if all_dst_ranks_ready:
                                    if (
                                        req.src_embedding_indices is not None
                                        and req.total_tokens > 0
                                    ):
                                        # Calculate which queue to use (same as add_transfer_request)
                                        dst_infos = self.transfer_infos[room].keys()
                                        session_port_sum = sum(
                                            int(session.split(":")[1])
                                            for session in dst_infos
                                        )
                                        shard_idx = session_port_sum % len(
                                            self.transfer_queues
                                        )

                                        # Add resume transfer chunk to queue (only once for all sessions)
                                        self.transfer_queues[shard_idx].put(
                                            TransferEmbeddingChunk(
                                                room=room,
                                                embedding_indices=req.src_embedding_indices,
                                                is_last=True,  # Resume is always the last part
                                                total_tokens=req.total_tokens,
                                            )
                                        )

                                        logger.debug(
                                            f"Resume transfer triggered: room={room}, "
                                            f"queue_idx={shard_idx}, src_blocks={len(req.src_embedding_indices)}, "
                                            f"all {len(self.transfer_infos[room])} sessions ready"
                                        )
                                        for dst_req in self.transfer_infos[
                                            room
                                        ].values():
                                            dst_req.resume_ready = False  # Reset for potential future resumes
                                    else:
                                        logger.error(
                                            f"Cannot trigger resume: missing src_embedding_indices or total_tokens "
                                            f"for room={room} session={mooncake_session_id}"
                                        )
                                else:
                                    logger.debug(
                                        f"Waiting for all dst ranks to be ready for resume: room={room}, "
                                        f"ready={sum(dst_req.resume_ready for dst_req in self.transfer_infos[room].values())}/{len(self.transfer_infos[room])}"
                                    )
                            else:
                                logger.error(
                                    f"Cannot resume: room={room} session={mooncake_session_id} not found in transfer_infos"
                                )
                        else:
                            # Init request: create new transfer_info
                            required_dst_info_num = int(
                                waiting_req_bytes[5].decode("ascii")
                            )

                            if room not in self.transfer_infos:
                                self.transfer_infos[room] = {}

                            self.transfer_infos[room][mooncake_session_id] = (
                                TransferEmbeddingInfo.from_zmq(waiting_req_bytes)
                            )
                            # NOTE: after bootstrapping we can mark the req as waiting for input
                            if len(self.transfer_infos[room]) == required_dst_info_num:
                                self.update_status(room, KVPoll.WaitingForInput)
                    else:
                        required_dst_info_num = int(
                            waiting_req_bytes[6].decode("ascii")
                        )
                        if room not in self.transfer_infos:
                            self.transfer_infos[room] = {}

                        self.transfer_infos[room][mooncake_session_id] = (
                            TransferInfo.from_zmq(waiting_req_bytes)
                        )
                        # NOTE: after bootstrapping we can mark the req as waiting for input
                        if len(self.transfer_infos[room]) == required_dst_info_num:
                            self.update_status(room, KVPoll.WaitingForInput)

        threading.Thread(target=bootstrap_thread).start()

    def start_decode_thread(self):
        self._bind_server_socket()

        def decode_thread():
            while True:
                msg = self.server_socket.recv_multipart()
                if msg[0] == MooncakeKVManager.AUX_DATA_HEADER:
                    self._handle_aux_data(msg)
                    continue

                (bootstrap_room, status, prefill_rank) = msg
                status = int(status.decode("ascii"))
                bootstrap_room = int(bootstrap_room.decode("ascii"))
                prefill_rank = int(prefill_rank.decode("ascii"))

                if status in [KVPoll.Success, KVPoll.Transferring]:
                    if bootstrap_room in self.request_status:
                        self.prefill_response_tracker[bootstrap_room].add(prefill_rank)
                        expected_response_num = (
                            self.required_prefill_response_num_table[bootstrap_room]
                        )
                        arrived_response_num = len(
                            self.prefill_response_tracker[bootstrap_room]
                        )
                        if arrived_response_num == expected_response_num:
                            self.update_status(bootstrap_room, status)
                elif status == KVPoll.Failed:
                    self.record_failure(
                        bootstrap_room,
                        f"Failed to get kvcache from prefill instance, it might be dead",
                    )
                    self.update_status(bootstrap_room, status)

        def heartbeat_checker():
            while True:
                time.sleep(self.heartbeat_interval)
                with self.connection_lock:
                    addresses = list(self.prefill_dp_size_table.keys())

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
                                # Remove KVPoll.Success requests from the tracker
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
        kv_indices: Optional[npt.NDArray[np.int32]] = None,
        index_slice: Optional[slice] = None,
        is_last: bool = True,
        aux_index: Optional[int] = None,
        # Embedding-specific parameters
        embedding_indices: Optional[List[int]] = None,
        total_tokens: Optional[int] = None,
        block_size: Optional[int] = None,
    ):
        assert self.is_sender_mode
        assert (
            not is_last
            or (is_last and aux_index is not None)
            or (is_last and embedding_indices is not None)
        )

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

        if self.is_multimodal:
            # Prevent duplicate transfer: if already in Transferring or Success state, skip
            current_status = self.check_status(bootstrap_room)
            if current_status in [KVPoll.Transferring, KVPoll.Success]:
                logger.debug(
                    f"Skip duplicate transfer for room={bootstrap_room}, status={current_status}"
                )
                return

        # NOTE(shangming): sharding according to the dst_infos to make sure
        # requests with the same dst_sessions will be added into the same
        # queue, which enables early abort with failed sessions.
        dst_infos = self.transfer_infos[bootstrap_room].keys()
        session_port_sum = sum(int(session.rsplit(":", 1)[1]) for session in dst_infos)
        shard_idx = session_port_sum % len(self.transfer_queues)

        if self.is_multimodal:
            transfer_chunk = TransferEmbeddingChunk(
                room=bootstrap_room,
                embedding_indices=embedding_indices,
                is_last=is_last,
                total_tokens=total_tokens,
            )
        else:
            transfer_chunk = TransferKVChunk(
                room=bootstrap_room,
                prefill_kv_indices=kv_indices,
                index_slice=index_slice,
                is_last=is_last,
                prefill_aux_index=aux_index,
            )

        self.transfer_queues[shard_idx].put(transfer_chunk)

    def check_status(self, bootstrap_room: int):
        return self.request_status[bootstrap_room]

    def update_status(self, bootstrap_room: int, status: KVPoll):
        if bootstrap_room not in self.request_status:
            self.request_status[bootstrap_room] = status
        else:
            # NOTE: status is only allowed to be incremented unless it is KVPoll.Failed
            if status == KVPoll.Failed:
                self.request_status[bootstrap_room] = KVPoll.Failed
            else:
                self.request_status[bootstrap_room] = max(
                    self.request_status[bootstrap_room], status
                )

    def record_failure(self, bootstrap_room: int, failure_reason: str):
        with self.failure_lock:
            self.failure_records[bootstrap_room] = failure_reason

    def get_session_id(self):
        return self.engine.get_session_id()

    def _handle_node_failure(self, failed_bootstrap_addr):
        with self.connection_lock:
            keys_to_remove = [
                k for k in self.connection_pool if k.startswith(failed_bootstrap_addr)
            ]
            for k in keys_to_remove:
                del self.connection_pool[k]
            if failed_bootstrap_addr in self.prefill_attn_tp_size_table:
                del self.prefill_attn_tp_size_table[failed_bootstrap_addr]
            if failed_bootstrap_addr in self.prefill_dp_size_table:
                del self.prefill_dp_size_table[failed_bootstrap_addr]
            if failed_bootstrap_addr in self.prefill_pp_size_table:
                del self.prefill_pp_size_table[failed_bootstrap_addr]

            possible_affected_rooms = self.addr_to_rooms_tracker.get(
                failed_bootstrap_addr, []
            )
            if failed_bootstrap_addr in self.addr_to_rooms_tracker:
                del self.addr_to_rooms_tracker[failed_bootstrap_addr]

        # Report the requests associated with the failed bootstrap addr and mark their status as KVPoll.Failed
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
            f"Losing connection with prefill instance (bootstrap_addr: {failed_bootstrap_addr}), {len(affected_rooms)} requests affected"
        )


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

        # For multimodal mode
        if mgr.is_multimodal:
            self.embedding_indices = None

    def init(
        self,
        num_kv_indices: Optional[int] = None,
        aux_index: Optional[int] = None,
        # Embedding-specific parameters
        embedding_indices: Optional[List[int]] = None,
    ):
        """Initialize sender for KV or Embedding mode"""
        if self.kv_mgr.is_multimodal:
            # Multimodal Embedding mode
            self.embedding_indices = embedding_indices
            self.init_time = time.time()
        else:
            # KV mode
            super().init(num_kv_indices, aux_index)

    def send(
        self,
        kv_indices: npt.NDArray[np.int32],
    ):
        index_slice = slice(self.curr_idx, self.curr_idx + len(kv_indices))
        self.curr_idx += len(kv_indices)
        is_last = self.curr_idx == self.num_kv_indices

        if not is_last:
            self.kv_mgr.add_transfer_request(
                self.bootstrap_room,
                kv_indices,
                index_slice,
                False,
            )
        else:
            self.kv_mgr.add_transfer_request(
                self.bootstrap_room,
                kv_indices,
                index_slice,
                True,
                aux_index=self.aux_index,
            )

    def send_embedding(
        self,
        embedding_indices: List[int] = None,
        last_chunk: bool = True,
        total_tokens: int = None,
        block_size: int = None,
    ):
        """Send embedding data to language instances (multimodal mode)"""
        if not self.kv_mgr.is_multimodal:
            raise ValueError("send_embedding only available in multimodal mode")

        self.kv_mgr.add_transfer_request(
            self.bootstrap_room,
            embedding_indices=embedding_indices,
            is_last=last_chunk,
            total_tokens=total_tokens,
            block_size=block_size,
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

        # defense code for multimodal mode
        if (
            self.kv_mgr.is_multimodal
            and self.bootstrap_room in self.kv_mgr.transfer_infos
        ):
            self.kv_mgr.transfer_infos.pop(self.bootstrap_room)

    def failure_exception(self):
        # Explicitly set the status to failure since this request has failed in another rank
        if self.conclude_state is None:
            self.conclude_state = KVPoll.Failed

        self.clear()

        with self.kv_mgr.failure_lock:
            failure_reason = self.kv_mgr.failure_records.pop(
                self.bootstrap_room, "Failed due to an unknown reason from another rank"
            )
        if self.kv_mgr.is_multimodal:
            raise EmbeddingTransferError(self.bootstrap_room, failure_reason)
        else:
            raise KVTransferError(self.bootstrap_room, failure_reason)

    def abort(self):
        self.kv_mgr.record_failure(
            self.bootstrap_room,
            "Aborted by AbortReq.",
        )
        # Explicitly set the status to failure since this request has been aborted
        self.conclude_state = KVPoll.Failed


class MooncakeKVReceiver(CommonKVReceiver):
    _ctx = zmq.Context()
    _socket_cache = {}
    _socket_locks = {}
    _global_lock = threading.Lock()

    def __init__(
        self,
        mgr: MooncakeKVManager,
        bootstrap_addr: str,
        bootstrap_room: Optional[int] = None,
        prefill_dp_rank: Optional[int] = None,
    ):
        self.session_id = mgr.get_session_id()
        self.conclude_state = None
        self.init_time = None
        super().__init__(mgr, bootstrap_addr, bootstrap_room, prefill_dp_rank)

        self.kv_mgr.addr_to_rooms_tracker[self.bootstrap_addr].add(self.bootstrap_room)
        self.kv_mgr.update_status(self.bootstrap_room, KVPoll.WaitingForInput)

    def _get_bootstrap_info_from_server(
        self, engine_rank, target_dp_group, target_pp_rank
    ):
        """Fetch the bootstrap info from the bootstrap server."""
        try:
            url = f"http://{self.bootstrap_addr}/route?engine_rank={engine_rank}&target_dp_group={target_dp_group}&target_pp_rank={target_pp_rank}"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                bootstrap_info = response.json()
                return bootstrap_info
            else:
                logger.error(
                    f"Failed to get prefill/encode server info: {response.status_code}, {response.text}"
                )
                return None
        except Exception as e:
            logger.error(f"Error fetching prefill info from bootstrap: {e}")
            return None

    def _register_kv_args(self):
        """Register KV or Embedding args to bootstrap server"""
        for bootstrap_info in self.bootstrap_infos:
            # Build message info based on mode
            messages = [
                "None".encode("ascii"),
                self.kv_mgr.local_ip.encode("ascii"),
                str(self.kv_mgr.rank_port).encode("ascii"),
                self.session_id.encode("ascii"),
            ]

            if self.kv_mgr.is_multimodal:
                # Multimodal Embedding mode: only send aux_data_ptrs (embedding data)
                packed_embedding_data_ptrs = b"".join(
                    struct.pack("Q", ptr) for ptr in self.kv_mgr.kv_args.aux_data_ptrs
                )
                messages.append(packed_embedding_data_ptrs)
            else:
                # KV mode: send kv_data_ptrs, aux_data_ptrs, and metadata
                packed_kv_data_ptrs = b"".join(
                    struct.pack("Q", ptr) for ptr in self.kv_mgr.kv_args.kv_data_ptrs
                )
                packed_aux_data_ptrs = b"".join(
                    struct.pack("Q", ptr) for ptr in self.kv_mgr.kv_args.aux_data_ptrs
                )
                # Note(shangming): No need to add pp rank here since pp is not supported on the decode side yet
                tp_rank = self.kv_mgr.kv_args.engine_rank
                kv_item_len = self.kv_mgr.kv_args.kv_item_lens[0]

                messages.extend(
                    [
                        packed_kv_data_ptrs,
                        packed_aux_data_ptrs,
                        str(tp_rank).encode("ascii"),
                        str(self.kv_mgr.attn_tp_size).encode("ascii"),
                        str(kv_item_len).encode("ascii"),
                    ]
                )

            # Send the message
            sock, lock = self._connect_to_bootstrap_server(bootstrap_info)
            with lock:
                sock.send_multipart(messages)

    def init(
        self,
        kv_indices: Optional[npt.NDArray[np.int32]] = None,
        aux_index: Optional[int] = None,
        # Embedding-specific parameters
        embedding_indices: Optional[List[int]] = None,
        allocated_tokens: Optional[int] = None,
    ):
        """Initialize receiver for KV or Embedding mode"""
        for bootstrap_info in self.bootstrap_infos:
            messages = [
                str(self.bootstrap_room).encode("ascii"),
                self.kv_mgr.local_ip.encode("ascii"),
                str(self.kv_mgr.rank_port).encode("ascii"),
                self.session_id.encode("ascii"),
            ]

            if self.kv_mgr.is_multimodal:
                # Multimodal Embedding mode
                embedding_indices_str = (
                    ",".join(str(idx) for idx in embedding_indices)
                    if embedding_indices is not None
                    else ""
                )

                # Calculate allocated_tokens if not provided
                if allocated_tokens is None and embedding_indices is not None:
                    # block_size = aux_item_lens[1] / fill_ids.itemsize (assuming int32 = 4 bytes)
                    block_size = self.kv_mgr.kv_args.aux_item_lens[1] // 4
                    allocated_tokens = len(embedding_indices) * block_size

                messages.extend(
                    [
                        embedding_indices_str.encode("ascii"),
                        str(self.required_dst_info_num).encode("ascii"),
                        str(allocated_tokens).encode("ascii"),
                    ]
                )
            else:
                is_dummy = bootstrap_info["is_dummy"]
                messages.append(
                    [
                        kv_indices.tobytes() if not is_dummy else b"",
                        str(aux_index).encode("ascii") if not is_dummy else b"",
                        str(self.required_dst_info_num).encode("ascii"),
                    ]
                )

            sock, lock = self._connect_to_bootstrap_server(bootstrap_info)
            with lock:
                sock.send_multipart(messages)
        self.init_time = time.time()

    def resume_transfer(
        self,
        embedding_indices: List[int],
        sent_tokens: int,
        allocated_tokens: int,
    ):
        """Resume transfer with new allocation after partial transfer (Embedding mode only)"""
        if not self.kv_mgr.is_multimodal:
            raise ValueError("resume_transfer only available in multimodal mode")

        embedding_indices_str = ",".join(str(idx) for idx in embedding_indices)

        for bootstrap_info in self.bootstrap_infos:
            sock, lock = self._connect_to_bootstrap_server(bootstrap_info)
            with lock:
                sock.send_multipart(
                    [
                        str(self.bootstrap_room).encode("ascii"),
                        self.kv_mgr.local_ip.encode("ascii"),
                        str(self.kv_mgr.rank_port).encode("ascii"),
                        self.session_id.encode("ascii"),
                        embedding_indices_str.encode("ascii"),  # New allocation
                        str(self.required_dst_info_num).encode("ascii"),
                        str(sent_tokens).encode("ascii"),  # Resume marker
                        str(allocated_tokens).encode("ascii"),  # New allocation size
                    ]
                )

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

        if not self.kv_mgr.is_multimodal:
            if self.bootstrap_room in self.kv_mgr.required_prefill_response_num_table:
                self.kv_mgr.required_prefill_response_num_table.pop(self.bootstrap_room)

            if self.bootstrap_room in self.kv_mgr.prefill_response_tracker:
                self.kv_mgr.prefill_response_tracker.pop(self.bootstrap_room)

    def failure_exception(self):
        # Explicitly set the status to failure since this request has failed in another rank
        if self.conclude_state is None:
            self.conclude_state = KVPoll.Failed

        self.clear()

        with self.kv_mgr.failure_lock:
            failure_reason = self.kv_mgr.failure_records.pop(
                self.bootstrap_room, "Failed due to an unknown reason from another rank"
            )
        if self.kv_mgr.is_multimodal:
            raise EmbeddingTransferError(self.bootstrap_room, failure_reason)
        else:
            raise KVTransferError(self.bootstrap_room, failure_reason)

    def abort(self):
        self.kv_mgr.record_failure(
            self.bootstrap_room,
            "Aborted by AbortReq.",
        )
        # Explicitly set the status to failure since this request has been aborted
        self.conclude_state = KVPoll.Failed


class MooncakeKVBootstrapServer(CommonKVBootstrapServer):
    pass


# ============================================================================
# Multimodal Embedding/Language Support (Aliases to KV classes with is_multimodal=True)
# ============================================================================

# Embedding Manager (just use KVManager with is_multimodal=True)
MooncakeEmbeddingManager = MooncakeKVManager

# Embedding Sender
MooncakeEmbeddingSender = MooncakeKVSender

# Embedding Receiver
MooncakeEmbeddingReceiver = MooncakeKVReceiver

# Embedding Bootstrap Server (same as KV)
MooncakeEmbeddingBootstrapServer = MooncakeKVBootstrapServer
