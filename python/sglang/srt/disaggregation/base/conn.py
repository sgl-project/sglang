from __future__ import annotations

import dataclasses
import enum
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from sglang.srt.server_args import ServerArgs

if TYPE_CHECKING:
    from sglang.srt.disaggregation.utils import DisaggregationMode


class StateType(str, enum.Enum):
    MAMBA = "mamba"
    QSA_PENDING = "qsa_pending"
    QSA_COMPRESSED = "qsa_compressed"
    SWA = "swa"
    DSA = "dsa"
    # DSA kpool-compress tail: one per-request ring row. The indices encode
    # only the live subrange of that row for the current open pool.
    DSA_TAIL = "dsa_tail"
    MINIMAX_INDEX_K = "minimax_index_k"
    # DeepSeek-V4 unified_kv SWA ring: addressed per-row by ring slot
    # (req_pool_idx * ring_stride + pos % ring_stride), needs its own component.
    SWA_RING = "swa_ring"
    # DeepSeek-V4 request-scoped compression state; preserve the legacy wire value.
    DSV4_REQUEST_STATE = "c128_state"
    # A block-scaled KV dtype keeps its per-block scales in buffers parallel to
    # K/V, one component per sub-pool so each carries the index payload of the
    # KV it describes (whole sequence for full attention, window for SWA).
    BLOCK_SCALE = "block_scale"
    BLOCK_SCALE_SWA = "block_scale_swa"


@dataclasses.dataclass
class KVTransferMetric:
    # Backends that cannot isolate transfer latency can leave this as None.
    transfer_latency_s: float | None = None
    # Backends that cannot isolate allocation wait latency can leave this as None.
    alloc_latency_s: float | None = None
    transfer_total_bytes: int | None = None


class KVArgs:
    engine_rank: int
    kv_data_ptrs: list[int]
    kv_data_lens: list[int]
    kv_item_lens: list[int]
    kv_layer_ids: list[int]
    kv_cache_dtype_str: str
    aux_data_ptrs: list[int]
    aux_data_lens: list[int]
    aux_item_lens: list[int]
    state_types: list[StateType]
    state_data_ptrs: list[list[int]]
    state_data_lens: list[list[int]]
    state_item_lens: list[list[int]]
    state_layer_ids: list[list[int]]
    # Per-tensor TP slice dim, used when prefill/decode attn_tp_size differ.
    state_dim_per_tensor: list[list[int]]
    # Number of rows before the slice axis in each per-slot state tensor.
    state_slice_outer_counts: list[list[int]]
    is_hybrid_mla_backend: bool
    # Per-tensor conv sub-block dims (GDN: [key_dim, key_dim, value_dim]) so the
    # scatter transfer can slice each independently head-sharded sub-block; None
    # per tensor when the single contiguous slice already matches the layout.
    state_conv_shard_groups: list[list[list[int] | None]]
    ib_device: str
    gpu_id: int
    kv_head_num: int
    total_kv_head_num: int
    page_size: int
    # for system dp
    system_dp_rank: int
    # Local Rust /route registry port; None on scheduler ranks without a listener.
    rust_http_port: int | None
    # for pp prefill
    pp_rank: int
    prefill_start_layer: int
    # Absolute end layer (exclusive) for this prefill PP stage. Needed to
    # reconstruct PP sub-ranges when kv_data_ptrs does not use a flat
    # layer-indexed layout (e.g. DeepSeek V4's buffer-type-organized flat
    # list).
    prefill_end_layer: int | None
    # For DeepSeek V4 (and other compressed-MLA) memory pools only.
    # Full-model compression ratio per layer (entries are 0/4/128). Used by
    # the connection layer to slice the buffer-type-organized flat list in a
    # PP-aware manner.
    mla_compression_ratios: list[int] | None
    # Only used of npu, for kv buf groups
    kv_buf_groups: int
    # Only used of npu, for decode total kv layers
    hidden_kv_layers: int
    # Only used of npu, for decode total kv layers
    draft_kv_layers: int
    num_draft_entries: int = 0


class KVPoll:
    Failed = 0
    Bootstrapping = 1
    WaitingForInput = 2
    Transferring = 3
    Success = 4


class BaseKVManager(ABC):
    """Base class for managing transfer states"""

    enable_deferred_decode_kv_release: bool = False

    @abstractmethod
    def __init__(
        self,
        args: KVArgs,
        disaggregation_mode: DisaggregationMode,
        server_args: ServerArgs,
        is_mla_backend: bool | None = False,
    ): ...

    @abstractmethod
    def register_to_bootstrap(self):
        """Register prefill server info to the bootstrap server."""
        ...


class BaseKVSender(ABC):
    @abstractmethod
    def __init__(
        self,
        mgr: BaseKVManager,
        bootstrap_addr: str,
        bootstrap_room: int,
        dest_tp_ranks: list[int],
        pp_rank: int,
        req_has_disagg_prefill_dp_rank: bool = False,
    ): ...

    @abstractmethod
    def init(self, num_kv_indices: int, aux_index: int | None = None):
        """
        Set req's index metadata locally or notify the decoder server about the kv indices length and aux index.
        """
        ...

    @abstractmethod
    def send(
        self,
        kv_indices: npt.NDArray[np.int32],
        state_indices: list | None = None,
        num_kv_tokens: int | None = None,
    ):
        """
        Send the kv cache at the given kv indices and the extra cache/state at the given indices to the decoder server.
        """
        ...

    def pop_decode_prefix_len(self) -> int:
        return 0

    def should_send_kv_chunk(self, num_pages: int, last_chunk: bool) -> bool:
        return num_pages > 0

    @abstractmethod
    def get_transfer_metric(self) -> KVTransferMetric:
        """Return backend-specific transfer metrics for this sender."""
        ...

    @abstractmethod
    def poll(self) -> KVPoll:
        """
        Check the status of the kv cache transfer.
        """
        ...

    @abstractmethod
    def failure_exception(self):
        """
        Raise an exception if the kv cache transfer fails.
        """
        ...

    def clear(self):
        """
        Clear any internal states.
        """

    def abort(self):
        """
        Abort the current transfer.
        """


class BaseKVReceiver(ABC):
    @abstractmethod
    def __init__(
        self,
        mgr: BaseKVManager,
        bootstrap_addr: str,
        bootstrap_room: int | None = None,
    ): ...

    @abstractmethod
    def init(
        self,
        prefill_dp_rank: int,
    ):
        """
        Resolve bootstrap metadata and mark the receiver ready for transfer metadata.
        """
        ...

    @abstractmethod
    def send_metadata(
        self,
        kv_indices: npt.NDArray[np.int32],
        aux_index: int | None = None,
        state_indices: list | None = None,
        decode_prefix_len: int | None = None,
    ):
        """
        Notify the prefill server about the kv indices, aux index, and state_indices.
        """
        ...

    @abstractmethod
    def poll(self) -> KVPoll:
        """
        Check the status of the kv cache transfer.
        """
        ...

    @abstractmethod
    def failure_exception(self):
        """
        Raise an exception if the kv cache transfer fails.
        """
        ...

    def clear(self):
        """
        Clear any internal states.
        """

    def abort(self):
        """
        Abort the current transfer.
        """


class BaseKVBootstrapServer(ABC):
    @abstractmethod
    def __init__(self, host: str, port: int): ...
