# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Base types for context parallel strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Tuple

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


class ContextParallelStrategyKind(IntEnum):
    """Context parallel strategy identifiers."""

    NONE = 0
    ZIGZAG = 1
    INTERLEAVE = 2

    @classmethod
    def from_string(cls, value: str) -> "ContextParallelStrategyKind":
        if value == "zigzag":
            return cls.ZIGZAG
        if value == "interleave":
            return cls.INTERLEAVE
        raise ValueError(
            f"Unknown cp_strategy={value!r}; expected one of "
            "{'zigzag', 'interleave'}"
        )

    @property
    def cli_value(self) -> str:
        return {
            ContextParallelStrategyKind.NONE: "none",
            ContextParallelStrategyKind.ZIGZAG: "zigzag",
            ContextParallelStrategyKind.INTERLEAVE: "interleave",
        }[self]


class CPAttentionBackendKind(IntEnum):
    """Attention backend calling convention used by CP strategy dispatch."""

    FLASH_ATTENTION = 0


@dataclass
class BaseContextParallelMetadata:
    total_seq_lens: int = 0
    bs: int = 1


class ContextParallelStrategy(ABC):
    """Owns process-wide policy for one context parallel layout."""

    name: str
    kind: ContextParallelStrategyKind

    def __init__(self, cp_size: int):
        self.cp_size = cp_size

    @property
    def cp_rank(self) -> int:
        from sglang.srt.layers.dp_attention import get_attention_cp_rank

        return get_attention_cp_rank()

    @property
    def per_layer_attn_cp_comm(self) -> bool:
        return _is_dsa_active()

    @abstractmethod
    def can_apply(self, num_tokens: int, forward_batch: "ForwardBatch") -> bool:
        """Return True if this strategy can shard the current forward."""

    @abstractmethod
    def build_metadata(
        self,
        num_tokens: int,
        seqs_len: Optional[List[int]],
        extend_seqs_len: Optional[List[int]] = None,
    ) -> BaseContextParallelMetadata:
        """Build per-forward metadata for this strategy."""

    @abstractmethod
    def shard_tokens(self, x: Any, forward_batch: "ForwardBatch") -> Any:
        """Shard a token-major payload to the current CP rank."""

    @abstractmethod
    def shard_positions(self, positions: Any, forward_batch: "ForwardBatch") -> Any:
        """Shard position payloads to the current CP rank."""

    @abstractmethod
    def gather_tokens(
        self,
        x: Any,
        forward_batch: "ForwardBatch",
        stream: Optional[Any] = None,
    ) -> Any:
        """Gather rank-local token payloads back to full token order."""

    @abstractmethod
    def gather_kv_cache(
        self,
        x: Any,
        forward_batch: "ForwardBatch",
        stream: Optional[Any] = None,
    ) -> Any:
        """Gather rank-local KV payloads back to full token order."""

    def shard_per_request(
        self,
        extend_seqs_cpu: List[int],
        extend_seqs: Any,
    ) -> Tuple[List[int], Any, List[int], Any]:
        raise NotImplementedError(
            f"{self.name} strategy does not support per-request sharding"
        )

    def split_before_forward(
        self,
        forward_batch: "ForwardBatch",
        input_ids: Optional[Any],
        positions: Any,
        input_embeds: Optional[Any] = None,
    ) -> Optional[Any]:
        """Shard model inputs before model.forward in CP-v2 paths."""
        if input_ids is not None:
            forward_batch.cp_v2_input_ids = self.shard_tokens(input_ids, forward_batch)
        forward_batch.positions = self.shard_positions(positions, forward_batch)
        if input_embeds is not None:
            return self.shard_tokens(input_embeds, forward_batch)
        return None

    @abstractmethod
    def run_attention(
        self,
        q: Any,
        forward_batch: "ForwardBatch",
        device: Any,
        attn_fn: Callable[[Any, Any, Any, int], Any],
        attention_backend: CPAttentionBackendKind = CPAttentionBackendKind.FLASH_ATTENTION,
    ) -> Any:
        """Dispatch CP attention using the selected backend convention."""

    @abstractmethod
    def materialize_full_kv(
        self,
        forward_batch: "ForwardBatch",
        layer: Any,
        k: Any,
        v: Any,
    ) -> None:
        """Write full-layout K/V to the backend cache if needed."""

    def reindex_attn_metadata(self, core_attn_metadata: Any) -> None:
        """Optional attention metadata rewrite for strategies that need it."""
        return None


def _is_dsa_active() -> bool:
    from sglang.srt.server_args import get_global_server_args

    sa = get_global_server_args()
    return bool(
        getattr(sa, "enable_prefill_cp", False)
        and getattr(sa, "_is_dsa_model_arch", False)
    )
