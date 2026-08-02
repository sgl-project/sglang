"""Backend-independent contracts for post-hoc KV-cache sparsity."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional

import torch

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


class KVStateAction(str, Enum):
    """What happens to KV entries excluded by a policy."""

    VISIBILITY_ONLY = "visibility_only"
    EVICT = "evict"
    HIERARCHICAL = "hierarchical"


class Granularity(str, Enum):
    """Logical unit returned by a sparsity policy."""

    TOKEN = "token"
    PAGE = "page"


class SelectionEvidence(str, Enum):
    """Runtime evidence used to make a sparse selection."""

    POSITION = "position"
    KV_DERIVED = "kv_derived"
    CURRENT_QUERY = "current_query"
    ATTENTION_HISTORY = "attention_history"


class DecisionScope(str, Enum):
    """Lifetime of a policy decision."""

    REQUEST = "request"
    STEP = "step"
    LAYER = "layer"


@dataclass(frozen=True)
class SparsityCapabilities:
    """Explicit policy requirements used for compatibility validation."""

    state_action: KVStateAction
    granularity: Granularity
    evidence: SelectionEvidence
    scope: DecisionScope
    requires_request_state: bool = False
    supports_cuda_graph: bool = False


@dataclass(frozen=True)
class RequestIdentity:
    """Generation-aware identity for a reusable request-pool slot."""

    pool_index: int
    generation: int


@dataclass(frozen=True)
class SelectionContext:
    """Logical inputs available to a sparsity policy."""

    query: Optional[torch.Tensor]
    layer_id: int
    req_pool_indices: torch.Tensor
    seq_lens: torch.Tensor
    key: Optional[torch.Tensor] = None
    value: Optional[torch.Tensor] = None
    output: Optional[torch.Tensor] = None
    forward_batch: Optional[ForwardBatch] = None
    attention_layer: Optional[RadixAttention] = None
    request_generations: Optional[torch.Tensor] = None
    metadata: Any = None


@dataclass(frozen=True)
class SelectionResult:
    """A backend-independent sparse view over request-relative KV positions.

    ``logical_indices`` is padded to a fixed capacity with ``-1``. For PAGE
    selections, ``visible_kv_lens`` carries the exact token count exposed to
    attention, including a possible partial final page.
    """

    granularity: Granularity
    logical_indices: torch.Tensor
    valid_lengths: torch.Tensor
    visible_kv_lens: torch.Tensor
    sparse_mask: torch.Tensor
    layer_id: Optional[int] = None
    max_visible_kv_len: Optional[int] = None

    def __post_init__(self) -> None:
        if self.logical_indices.ndim != 2:
            raise ValueError("logical_indices must have shape [batch, capacity]")
        batch_size = self.logical_indices.shape[0]
        for name, tensor in (
            ("valid_lengths", self.valid_lengths),
            ("visible_kv_lens", self.visible_kv_lens),
            ("sparse_mask", self.sparse_mask),
        ):
            if tensor.ndim != 1 or tensor.shape[0] != batch_size:
                raise ValueError(f"{name} must have shape [batch]")
        if self.sparse_mask.dtype != torch.bool:
            raise TypeError("sparse_mask must use torch.bool")
        if self.logical_indices.dtype not in (torch.int32, torch.int64):
            raise TypeError("logical_indices must use an integer dtype")
        if self.valid_lengths.dtype not in (torch.int32, torch.int64):
            raise TypeError("valid_lengths must use an integer dtype")
        if self.visible_kv_lens.dtype not in (torch.int32, torch.int64):
            raise TypeError("visible_kv_lens must use an integer dtype")
        if not (
            self.logical_indices.device
            == self.valid_lengths.device
            == self.visible_kv_lens.device
            == self.sparse_mask.device
        ):
            raise ValueError("all SelectionResult tensors must share a device")
        if self.max_visible_kv_len is not None and (
            not isinstance(self.max_visible_kv_len, int)
            or isinstance(self.max_visible_kv_len, bool)
            or self.max_visible_kv_len <= 0
        ):
            raise ValueError("max_visible_kv_len must be a positive integer")

    @property
    def capacity(self) -> int:
        return self.logical_indices.shape[1]
