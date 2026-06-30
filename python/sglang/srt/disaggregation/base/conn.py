from __future__ import annotations

import dataclasses
import enum
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Optional

import numpy as np
import numpy.typing as npt

from sglang.srt.server_args import ServerArgs

if TYPE_CHECKING:
    from sglang.srt.disaggregation.utils import DisaggregationMode


class StateType(str, enum.Enum):
    MAMBA = "mamba"
    SWA = "swa"
    DSA = "dsa"
    MINIMAX_INDEX_K = "minimax_index_k"
    # DeepSeek-V4 unified_kv SWA ring: addressed per-row by ring slot
    # (req_pool_idx * ring_stride + pos % ring_stride), needs its own component.
    SWA_RING = "swa_ring"


@dataclasses.dataclass
class KVTransferMetric:
    # Backends that cannot isolate transfer latency can leave this as None.
    transfer_latency_s: Optional[float] = None
    # Backends that cannot isolate allocation wait latency can leave this as None.
    alloc_latency_s: Optional[float] = None
    transfer_total_bytes: Optional[int] = None


class KVArgs:
    engine_rank: int
    kv_data_ptrs: List[int]
    kv_data_lens: List[int]
    kv_item_lens: List[int]
    aux_data_ptrs: List[int]
    aux_data_lens: List[int]
    aux_item_lens: List[int]
    state_types: List[StateType]
    state_data_ptrs: List[List[int]]
    state_data_lens: List[List[int]]
    state_item_lens: List[List[int]]
    # Per-tensor TP slice dim, used when prefill/decode attn_tp_size differ.
    state_dim_per_tensor: List[List[int]]
    ib_device: str
    ib_traffic_class: str
    gpu_id: int
    kv_head_num: int
    total_kv_head_num: int
    page_size: int
    # for system dp
    system_dp_rank: int
    # for pp prefill
    pp_rank: int
    prefill_start_layer: int
    # Absolute end layer (exclusive) for this PP stage, needed to reconstruct
    # PP sub-ranges when kv_data_ptrs isn't flat layer-indexed (e.g. DeepSeek V4).
    prefill_end_layer: Optional[int]
    # Compressed-MLA pools only. Per-layer compression ratio (0/4/128) for
    # PP-aware slicing of the buffer-type-organized flat list.
    mla_compression_ratios: Optional[List[int]]
    # Only used of npu, for kv buf groups
    kv_buf_groups: int
    # Only used of npu, for decode total kv layers
    total_kv_layers: int
    # Main-model KV layer count when draft (MTP/EAGLE) ptrs are appended, so
    # the LP hook's is_last fires on the last main layer, not past it.
    # None ⇒ no draft, len(kv_data_ptrs) is all main.
    prefill_num_main_kv_layers: Optional[int] = None
    # Local main-model KV layer count (excludes draft tail). Equals
    # ``prefill_num_main_kv_layers`` on prefill; on decode it holds the
    # full-model main count so the connection layer can fail-loud when the
    # cross-side layout doesn't match. None ⇒ no draft pool on this side.
    num_main_kv_layers: Optional[int] = None


class KVPoll:
    Failed = 0
    Bootstrapping = 1
    WaitingForInput = 2
    Transferring = 3
    Success = 4


class BaseKVManager(ABC):
    """Base class for managing transfer states"""

    @abstractmethod
    def __init__(
        self,
        args: KVArgs,
        disaggregation_mode: DisaggregationMode,
        server_args: ServerArgs,
        is_mla_backend: Optional[bool] = False,
    ): ...

    @abstractmethod
    def register_to_bootstrap(self):
        """Register prefill server info to the bootstrap server."""
        ...

    def make_layer_pipeline_hook_for_reqs(self, reqs_with_indices):
        """Build a per-batch forward hook for layer-pipelined KV transfer.

        Returns ``None`` by default — non-Mooncake backends opt out.
        ``MooncakeKVManager`` overrides this. The caller
        (``Scheduler.build_layer_pipeline_hook``) MUST NOT install a hook
        when this returns ``None``.
        """
        return None


class BaseKVSender(ABC):
    @abstractmethod
    def __init__(
        self,
        mgr: BaseKVManager,
        bootstrap_addr: str,
        bootstrap_room: int,
        dest_tp_ranks: List[int],
        pp_rank: int,
    ): ...

    @abstractmethod
    def init(self, num_kv_indices: int, aux_index: Optional[int] = None):
        """
        Set req's index metadata locally or notify the decoder server about the kv indices length and aux index.
        """
        ...

    @abstractmethod
    def send(
        self,
        kv_indices: npt.NDArray[np.int32],
        state_indices: Optional[List] = None,
    ):
        """
        Send the kv cache at the given kv indices and the extra cache/state at the given indices to the decoder server.
        """
        ...

    def pop_decode_prefix_len(self) -> int:
        return 0

    def send_draft_kv(self, kv_indices: npt.NDArray[np.int32]) -> None:
        """Ship the draft (MTP/EAGLE NEXTN) KV slice for this token chunk.

        Default no-op — backends without a separate draft pool or layer-pipelined
        transfer don't need it. ``MooncakeKVSender`` overrides this. Callers may
        invoke it unconditionally; the default keeps the call duck-typing-free.
        """
        return

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


class BaseKVReceiver(ABC):
    @abstractmethod
    def __init__(
        self,
        mgr: BaseKVManager,
        bootstrap_addr: str,
        bootstrap_room: Optional[int] = None,
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
        aux_index: Optional[int] = None,
        state_indices: Optional[List] = None,
        decode_prefix_len: Optional[int] = None,
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
        pass

    def abort(self):
        """
        Abort the current transfer.
        """
        pass


class BaseKVBootstrapServer(ABC):
    @abstractmethod
    def __init__(self, host: str, port: int): ...
