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

"""Interleave context parallel strategy shell.

For ``cp_size = 4``, each rank owns every fourth token:

    dp_attn_tp0: token0, token4, token8,  token12, token16, ...
    dp_attn_tp1: token1, token5, token9,  token13, token17, ...
    dp_attn_tp2: token2, token6, token10, token14, token18, ...
    dp_attn_tp3: token3, token7, token11, token15, token19, ...

After all-gather, tokens are restored to the original order:

    token0, token1, token2, token3, token4, token5, token6, token7, ...
"""

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
class InterleaveContextParallelMetadata(BaseContextParallelMetadata):
    """Interleave has no per-forward zigzag permutation payload."""


class InterleaveCPStrategy(ContextParallelStrategy):
    name = "interleave"
    kind = ContextParallelStrategyKind.INTERLEAVE

    def can_apply(self, num_tokens: int, forward_batch) -> bool:
        if self.cp_size <= 1 or num_tokens < self.cp_size:
            return False
        forward_mode = getattr(forward_batch, "forward_mode", None)
        return forward_mode is None or forward_mode.is_context_parallel_extend()

    def build_metadata(
        self,
        num_tokens: int,
        seqs_len: Optional[List[int]],
        extend_seqs_len: Optional[List[int]] = None,
    ) -> InterleaveContextParallelMetadata:
        return InterleaveContextParallelMetadata(
            total_seq_lens=sum(extend_seqs_len or seqs_len or [num_tokens]),
            bs=len(extend_seqs_len or seqs_len or [num_tokens]),
        )

    def shard_hidden_states(self, x: Any, forward_batch) -> Any:
        raise NotImplementedError(
            "Interleave hidden-state sharding will land in a follow-up PR"
        )

    def shard_position_ids(self, positions: Any, forward_batch) -> Any:
        raise NotImplementedError(
            "Interleave position-id sharding will land in a follow-up PR"
        )

    def gather_hidden_states(
        self, x: Any, forward_batch, stream: Optional[Any] = None
    ) -> Any:
        raise NotImplementedError(
            "Interleave hidden-state gather will land in a follow-up PR"
        )

    def gather_kv_cache(
        self, x: Any, forward_batch, stream: Optional[Any] = None
    ) -> Any:
        raise NotImplementedError("Interleave KV gather will land in a follow-up PR")

    def run_attention(
        self,
        q: Any,
        forward_batch,
        device: Any,
        attn_fn,
        attention_backend: CPAttentionBackendKind = CPAttentionBackendKind.FLASH_ATTENTION,
    ) -> Any:
        raise NotImplementedError(
            "Interleave attention dispatch will land in a follow-up PR"
        )

    def materialize_full_kv(self, forward_batch, layer: Any, k: Any, v: Any) -> None:
        raise NotImplementedError(
            "Interleave KV materialization will land in a follow-up PR"
        )
