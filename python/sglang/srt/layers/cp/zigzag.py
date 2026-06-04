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

"""Zigzag context parallel strategy shell."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

from sglang.srt.layers.cp.base import (
    BaseContextParallelMetadata,
    ContextParallelStrategy,
    ContextParallelStrategyKind,
    CPAttentionBackendKind,
)


@dataclass
class ZigzagContextParallelMetadata(BaseContextParallelMetadata):
    # Layout lists have length bs * cp_segment_num (= bs * 2 * cp_size).
    split_list: Optional[List[int]] = None
    zigzag_index: Optional[List[int]] = None
    cp_reverse_index: Optional[List[int]] = None
    reverse_split_len: Optional[List[int]] = None

    # Per-rank aggregate lists have length cp_size.
    per_rank_actual_token: Optional[List[int]] = None
    max_rank_len: Optional[List[int]] = None

    # Per-sequence FlashAttention tensors (shape [bs] or [bs + 1]).
    kv_len_prev_tensor: Optional[Any] = None
    kv_len_next_tensor: Optional[Any] = None
    actual_seq_q_prev_tensor: Optional[Any] = None
    actual_seq_q_next_tensor: Optional[Any] = None
    cu_seqlens_q_prev_tensor: Optional[Any] = None
    cu_seqlens_q_next_tensor: Optional[Any] = None

    # Scalars derived from the per-sequence lists above.
    total_q_prev_tokens: int = 0
    total_q_next_tokens: int = 0
    max_seqlen_q_prev: int = 0
    max_seqlen_q_next: int = 0

    # Per-sequence CPU lists, useful for indexers and diagnostics.
    kv_len_prev_list: Optional[List[int]] = None
    kv_len_next_list: Optional[List[int]] = None
    actual_seq_q_prev_list: Optional[List[int]] = None
    actual_seq_q_next_list: Optional[List[int]] = None


ContextParallelMetadata = ZigzagContextParallelMetadata


class ZigzagCPStrategy(ContextParallelStrategy):
    name = "zigzag"
    kind = ContextParallelStrategyKind.ZIGZAG

    def can_apply(self, num_tokens: int, forward_batch) -> bool:
        if self.cp_size <= 1 or num_tokens < self.cp_size * 2:
            return False
        forward_mode = getattr(forward_batch, "forward_mode", None)
        return forward_mode is None or forward_mode.is_context_parallel_extend()

    def build_metadata(
        self,
        num_tokens: int,
        seqs_len: Optional[List[int]],
        extend_seqs_len: Optional[List[int]] = None,
    ) -> ZigzagContextParallelMetadata:
        return ZigzagContextParallelMetadata(
            total_seq_lens=sum(extend_seqs_len or seqs_len or [num_tokens]),
            bs=len(extend_seqs_len or seqs_len or [num_tokens]),
        )

    def shard_tokens(self, x: Any, forward_batch) -> Any:
        raise NotImplementedError("Zigzag token sharding will land in a follow-up PR")

    def shard_positions(self, positions: Any, forward_batch) -> Any:
        raise NotImplementedError(
            "Zigzag position sharding will land in a follow-up PR"
        )

    def gather_tokens(self, x: Any, forward_batch, stream: Optional[Any] = None) -> Any:
        raise NotImplementedError("Zigzag token gather will land in a follow-up PR")

    def gather_kv_cache(
        self, x: Any, forward_batch, stream: Optional[Any] = None
    ) -> Any:
        raise NotImplementedError("Zigzag KV gather will land in a follow-up PR")

    def run_attention(
        self,
        q: Any,
        forward_batch,
        device: Any,
        attn_fn,
        attention_backend: CPAttentionBackendKind = CPAttentionBackendKind.FLASH_ATTENTION,
    ) -> Any:
        raise NotImplementedError(
            "Zigzag attention dispatch will land in a follow-up PR"
        )

    def materialize_full_kv(self, forward_batch, layer: Any, k: Any, v: Any) -> None:
        raise NotImplementedError(
            "Zigzag KV materialization will land in a follow-up PR"
        )
