# Copyright 2023-2026 SGLang Team
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

"""Base types and process-wide helpers for context parallel strategies.

The strategy implementation is split across:

* ``base.py``: base ABC, base metadata dataclass, enums, and singleton helpers.
* ``zigzag.py``: former in-seq-split strategy and zigzag metadata.
* ``interleave.py``: former round-robin-split strategy and interleave metadata.
* ``utils.py``: public re-exports for import convenience.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Tuple

from sglang.srt.runtime_context import get_parallel

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.server_args import ServerArgs


class ContextParallelStrategyKind(IntEnum):
    """Context parallel strategy identifiers."""

    NONE = 0
    ZIGZAG = 1
    INTERLEAVE = 2

    @classmethod
    def from_string(cls, value: str) -> ContextParallelStrategyKind:
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
    DSA = 1

    @classmethod
    def from_string(cls, value: str) -> CPAttentionBackendKind:
        if value in ("fa3", "fa4", "flashinfer"):
            return cls.FLASH_ATTENTION
        if value in ("dsa"):
            return cls.DSA
        raise ValueError(
            f"Unsupported attention_backend={value!r} for CP strategy; expected one "
            "of {'fa3', 'fa4', 'flashinfer', 'dsa'}"
        )


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

        return get_parallel().attn_cp_rank

    @property
    def per_layer_attn_cp_comm(self) -> bool:
        return _is_dsa_active()

    @abstractmethod
    def can_apply(self, num_tokens: int, forward_batch: ForwardBatch) -> bool:
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
    def shard_hidden_states(self, x: Any, forward_batch: ForwardBatch) -> Any:
        """Shard hidden states to the current CP rank, usually at the first layer."""

    @abstractmethod
    def shard_position_ids(self, positions: Any, forward_batch: ForwardBatch) -> Any:
        """Shard KV-cache slot position IDs for each token to the current CP rank."""

    @abstractmethod
    def gather_hidden_states(
        self,
        x: Any,
        forward_batch: ForwardBatch,
        stream: Optional[Any] = None,
    ) -> Any:
        """Gather rank-local hidden states, usually at the last layer."""

    @abstractmethod
    def gather_kv_cache(
        self,
        x: Any,
        forward_batch: ForwardBatch,
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

    def shard_local_tokens(self, input_: Any) -> Any:
        """Shard an arbitrary per-token tensor/list to this CP rank's local tokens."""
        raise NotImplementedError(
            f"{self.name} strategy does not support local-token sharding"
        )

    def materialize_full_indexer_k_cache(
        self, key: Any, forward_batch: ForwardBatch
    ) -> Any:
        """Gather the rank-local DSA indexer key back to full token order.

        Only DSA-capable strategies (interleave) override this; others reject it.
        """
        raise NotImplementedError(
            f"{self.name} strategy does not support DSA indexer key gather"
        )

    def all_gather_dsa_trtllm_fp8_kv(
        self, forward_batch: ForwardBatch, k: Any, k_rope: Any
    ) -> Any:
        """All-gather FP8 KV for the TRT-LLM MLA backend back to full token order.

        Only DSA-capable strategies (interleave) override this; others reject it.
        """
        raise NotImplementedError(
            f"{self.name} strategy does not support DSA trtllm FP8 KV gather"
        )

    def split_before_forward(
        self,
        forward_batch: ForwardBatch,
        input_ids: Optional[Any],
        positions: Any,
        input_embeds: Optional[Any] = None,
    ) -> Optional[Any]:
        """Shard model inputs before model.forward in CP-v2 paths."""
        if input_ids is not None:
            forward_batch.cp_v2_input_ids = self.shard_hidden_states(
                input_ids, forward_batch
            )
        forward_batch.positions = self.shard_position_ids(positions, forward_batch)
        if input_embeds is not None:
            return self.shard_hidden_states(input_embeds, forward_batch)
        return None

    @abstractmethod
    def run_attention(
        self,
        q: Any,
        forward_batch: ForwardBatch,
        device: Any,
        attn_fn: Callable[[Any, Any, Any, int], Any],
        attention_backend: CPAttentionBackendKind = CPAttentionBackendKind.FLASH_ATTENTION,
    ) -> Any:
        """Dispatch CP attention using the selected backend convention."""

    @abstractmethod
    def materialize_full_kv(
        self,
        forward_batch: ForwardBatch,
        layer: Any = None,
        k: Any = None,
        v: Any = None,
        swa_loc: Optional[Any] = None,
        **kwargs,
    ) -> Any:
        """Write full-layout K/V to the backend cache if needed."""

    def reindex_attn_metadata(self, core_attn_metadata: Any) -> None:
        """Optional attention metadata rewrite for strategies that need it."""
        return None


def _is_dsa_active() -> bool:
    from sglang.srt.runtime_context import get_server_args

    sa = get_server_args()
    return bool(
        getattr(sa, "enable_prefill_cp", False)
        and getattr(sa, "_is_dsa_model_arch", False)
    )


_STRATEGY: Optional[ContextParallelStrategy] = None


def init_cp_strategy(server_args: ServerArgs) -> None:
    """Bind the configured CP strategy for this process."""
    global _STRATEGY

    if not getattr(server_args, "enable_prefill_cp", False):
        _STRATEGY = None
        return

    cp_size = getattr(server_args, "attn_cp_size", 1)
    if cp_size <= 1:
        _STRATEGY = None
        return

    kind = ContextParallelStrategyKind.from_string(server_args.cp_strategy)
    if kind == ContextParallelStrategyKind.ZIGZAG:
        from sglang.srt.layers.cp.zigzag import ZigzagCPStrategy

        _STRATEGY = ZigzagCPStrategy(cp_size=cp_size)
    elif kind == ContextParallelStrategyKind.INTERLEAVE:
        from sglang.srt.layers.cp.interleave import InterleaveCPStrategy

        _STRATEGY = InterleaveCPStrategy(cp_size=cp_size)
    else:
        raise ValueError(
            f"Unsupported cp_strategy kind {kind} for "
            f"cp_strategy={server_args.cp_strategy!r}"
        )


def get_cp_strategy() -> Optional[ContextParallelStrategy]:
    """Return the configured strategy, initializing lazily on first call.

    Subprocesses re-import this module with ``_STRATEGY = None`` and never
    re-run ``ServerArgs.__post_init__`` because the pickled instance bypasses
    ``__init__``. Lazy init lets worker processes recover the singleton from
    global server args.
    """
    global _STRATEGY

    if _STRATEGY is None:
        from sglang.srt.runtime_context import get_server_args

        try:
            server_args = get_server_args()
        except ValueError:
            return None
        if server_args is not None and getattr(server_args, "enable_prefill_cp", False):
            init_cp_strategy(server_args)
    return _STRATEGY


def get_cp_strategy_kind() -> ContextParallelStrategyKind:
    strategy = get_cp_strategy()
    if strategy is None:
        return ContextParallelStrategyKind.NONE
    return strategy.kind


def is_cp_enabled() -> bool:
    return get_cp_strategy() is not None


def is_zigzag() -> bool:
    return get_cp_strategy_kind() == ContextParallelStrategyKind.ZIGZAG


def is_interleave() -> bool:
    return get_cp_strategy_kind() == ContextParallelStrategyKind.INTERLEAVE
