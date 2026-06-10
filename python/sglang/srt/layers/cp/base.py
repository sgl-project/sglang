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

"""Base types for context-parallel strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple, Union

import torch

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


class ContextParallelStrategyKind(IntEnum):
    """Context-parallel strategy identifiers.

    The enum is used for internal dispatch and type checks. CLI and serialized
    server args keep using strings for compatibility.
    """

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
            f"Unknown cp_strategy={value!r}; expected one of {{'zigzag', 'interleave'}}"
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
    # Aggregate sum of extend_seq_lens across the batch.
    total_seq_lens: int = 0
    bs: int = 1


class ContextParallelStrategy(ABC):
    """Owns per-mode CP policy for one process.

    A single instance is constructed at server start. Per-forward request state
    lives on ``forward_batch.attn_cp_metadata`` via :meth:`build_metadata`.
    """

    #: Strategy short name, used in logs and as the CLI value.
    name: str
    kind: ContextParallelStrategyKind

    #: True when the strategy expects each layer to run with the model body in
    #: SCATTERED layout and communicate hidden states over the attn_cp group.
    @property
    def per_layer_attn_cp_comm(self) -> bool:
        return _is_dsa_active()

    def __init__(self, cp_size: int):
        self.cp_size = cp_size
        # cp_rank is resolved lazily on first use — the cp group is built only
        # after distributed init, which happens after server_args validation.

    @property
    def cp_rank(self) -> int:
        from sglang.srt.layers.dp_attention import get_attention_cp_rank

        return get_attention_cp_rank()

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
        """Produce the CP metadata payload for one forward."""

    @abstractmethod
    def shard_tokens(
        self,
        x: Union[torch.Tensor, List, Tuple],
        forward_batch: "ForwardBatch",
    ) -> torch.Tensor:
        """Slice ``[L, ...]`` along dim 0 down to this rank."""

    @abstractmethod
    def shard_positions(
        self, positions: torch.Tensor, forward_batch: "ForwardBatch"
    ) -> torch.Tensor:
        """Slice positions along the last dim."""

    @abstractmethod
    def gather_tokens(
        self,
        x: torch.Tensor,
        forward_batch: "ForwardBatch",
        stream: Optional[torch.cuda.Stream] = None,
    ) -> torch.Tensor:
        """Allgather per-rank tokens into original token order."""

    @abstractmethod
    def gather_kv_cache(
        self,
        x: torch.Tensor,
        forward_batch: "ForwardBatch",
        stream: Optional[torch.cuda.Stream] = None,
    ) -> torch.Tensor:
        """Gather variant for multi-dim KV cache tensors."""

    def shard_per_request(
        self,
        extend_seqs_cpu: List[int],
        extend_seqs: torch.Tensor,
    ) -> Tuple[List[int], torch.Tensor, List[int], torch.Tensor]:
        """For interleave: split per-request lengths across CP ranks."""
        raise NotImplementedError(
            f"{self.name} strategy does not support per-request sharding"
        )

    def split_before_forward(
        self,
        forward_batch: "ForwardBatch",
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        input_embeds: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """Shard model inputs before ``model.forward`` in the CP-v2 path.

        The full ``input_ids`` tensor is preserved for logits/logprob
        bookkeeping. Rank-local ids are stashed on ``forward_batch`` for model
        internals that need token ids in the same layout as hidden states.
        ``input_embeds`` is returned because it is passed through kwargs rather
        than stored on ``ForwardBatch``.
        """
        if input_ids is not None:
            forward_batch.cp_v2_input_ids = self.shard_tokens(input_ids, forward_batch)
            spec_info = getattr(forward_batch, "spec_info", None)
            spec_hidden_states = getattr(spec_info, "hidden_states", None)
            if (
                spec_hidden_states is not None
                and spec_hidden_states.shape[0] == input_ids.shape[0]
            ):
                spec_info.hidden_states = self.shard_tokens(
                    spec_hidden_states, forward_batch
                )
        forward_batch.positions = self.shard_positions(positions, forward_batch)
        if input_embeds is not None:
            return self.shard_tokens(input_embeds, forward_batch)
        return None

    @abstractmethod
    def run_attention(
        self,
        q: torch.Tensor,
        forward_batch: "ForwardBatch",
        device: torch.device,
        attn_fn: Callable[
            [torch.Tensor, torch.Tensor, torch.Tensor, int], torch.Tensor
        ],
        attention_backend: CPAttentionBackendKind = CPAttentionBackendKind.FLASH_ATTENTION,
    ) -> torch.Tensor:
        """Dispatch CP attention using the selected backend calling convention.

        The current implementation supports FlashAttention-style callbacks. If
        another backend needs different metadata, add a new
        ``CPAttentionBackendKind`` branch in each concrete strategy instead of
        widening the callback contract implicitly.
        """

    @abstractmethod
    def materialize_full_kv(
        self,
        forward_batch: "ForwardBatch",
        layer,
        k: torch.Tensor,
        v: torch.Tensor,
        swa_loc: Optional[torch.Tensor] = None,
    ) -> None:
        """Write the full-sequence K/V into every rank's pool."""

    def reindex_attn_metadata(self, core_attn_metadata) -> None:
        """Optional attention metadata rewrite for strided strategies."""
        return None


def _is_dsa_active() -> bool:
    """Return whether unified prefill CP is active for a DSA architecture."""
    from sglang.srt.server_args import get_global_server_args

    sa = get_global_server_args()
    return bool(
        getattr(sa, "enable_prefill_cp", False)
        and getattr(sa, "_is_dsa_model_arch", False)
    )
