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

"""Zigzag context-parallel strategy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

import torch

from sglang.srt.layers.cp.base import (
    BaseContextParallelMetadata,
    ContextParallelStrategy,
    ContextParallelStrategyKind,
    CPAttentionBackendKind,
)

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


@dataclass
class CPAttnSlice:
    """One Q chunk produced by :meth:`ZigzagCPStrategy.iter_attn_slices`."""

    q: torch.Tensor
    actual_seq_q: int
    cache_seqlens_tensor: torch.Tensor
    actual_seq_q_tensor: torch.Tensor


@dataclass
class ZigzagContextParallelMetadata(BaseContextParallelMetadata):
    # Layout lists have length bs * cp_segment_num (= bs * 2 * cp_size).
    split_list: List[int] = None
    zigzag_index: List[int] = None
    cp_reverse_index: List[int] = None
    reverse_split_len: List[int] = None

    # Per-rank-aggregate lists have length cp_size.
    # max_rank_len is a list of cp_size copies of max(per_rank_actual_token),
    # kept as a list for torch.split() bucket sizes.
    per_rank_actual_token: List[int] = None
    max_rank_len: List[int] = None

    # Per-sequence FlashAttention tensors (shape [bs] or [bs+1]).
    kv_len_prev_tensor: torch.Tensor = None  # [bs] int32 CUDA
    kv_len_next_tensor: torch.Tensor = None  # [bs] int32 CUDA
    actual_seq_q_prev_tensor: torch.Tensor = None  # [bs] int32 CUDA
    actual_seq_q_next_tensor: torch.Tensor = None  # [bs] int32 CUDA
    cu_seqlens_q_prev_tensor: torch.Tensor = None  # [bs+1] int32 CUDA
    cu_seqlens_q_next_tensor: torch.Tensor = None  # [bs+1] int32 CUDA

    # Scalars derived from the per-sequence lists above.
    total_q_prev_tokens: int = 0
    total_q_next_tokens: int = 0
    max_seqlen_q_prev: int = 0
    max_seqlen_q_next: int = 0

    # Per-seq CPU lists (useful for NSA indexer and diagnostics).
    kv_len_prev_list: List[int] = None
    kv_len_next_list: List[int] = None
    actual_seq_q_prev_list: List[int] = None
    actual_seq_q_next_list: List[int] = None


# Compatibility name used by existing backends/tests. The legacy metadata
# shape is the zigzag shape; interleave call sites should use the concrete
# InterleaveContextParallelMetadata type.
ContextParallelMetadata = ZigzagContextParallelMetadata


class ZigzagCPStrategy(ContextParallelStrategy):
    """In-seq-split strategy.

    The sequence is cut into ``2 * cp_size`` blocks. Rank ``r`` gets blocks
    ``r`` and ``2 * cp_size - 1 - r`` and runs attention twice, once for each
    half, to keep causal attention cost balanced.
    """

    name = "zigzag"
    kind = ContextParallelStrategyKind.ZIGZAG

    def can_apply(self, num_tokens: int, forward_batch: "ForwardBatch") -> bool:
        # Delegate to the ported free function so the bs>1 per-sequence guards
        # match the legacy implementation exactly.
        from sglang.srt.layers.cp.utils import can_cp_split

        return can_cp_split(num_tokens, self.cp_size, forward_batch)

    def build_metadata(
        self,
        num_tokens: int,
        seqs_len: Optional[List[int]],
        extend_seqs_len: Optional[List[int]] = None,
    ) -> ZigzagContextParallelMetadata:
        from sglang.srt.layers.cp.utils import prepare_context_parallel_metadata

        if extend_seqs_len is None:
            extend_seqs_len = seqs_len
        return prepare_context_parallel_metadata(
            num_tokens,
            self.cp_rank,
            self.cp_size,
            seqs_len,
            extend_seqs_len=extend_seqs_len,
        )

    def shard_tokens(self, x, forward_batch):
        from sglang.srt.layers.cp.utils import cp_split_and_rebuild_data

        return cp_split_and_rebuild_data(forward_batch, x)

    def shard_positions(self, positions, forward_batch):
        from sglang.srt.layers.cp.utils import cp_split_and_rebuild_position

        return cp_split_and_rebuild_position(forward_batch, positions)

    def gather_tokens(self, x, forward_batch, stream=None):
        from sglang.srt.layers.cp.utils import cp_all_gather_rerange_output

        s = stream if stream is not None else torch.cuda.current_stream()
        return cp_all_gather_rerange_output(x, self.cp_size, forward_batch, s)

    def gather_kv_cache(self, x, forward_batch, stream=None):
        from sglang.srt.layers.cp.utils import cp_all_gather_rerange_kv_cache

        s = stream if stream is not None else torch.cuda.current_stream()
        return cp_all_gather_rerange_kv_cache(x, self.cp_size, forward_batch, s)

    def run_attention(
        self,
        q,
        forward_batch,
        device,
        attn_fn,
        attention_backend=CPAttentionBackendKind.FLASH_ATTENTION,
    ):
        if attention_backend != CPAttentionBackendKind.FLASH_ATTENTION:
            raise NotImplementedError(
                f"{self.name} CP does not support {attention_backend=}"
            )

        from sglang.srt.layers.cp.utils import cp_attn_forward_extend

        return cp_attn_forward_extend(forward_batch, q, device, attn_fn)

    def iter_attn_slices(self, q, forward_batch):
        cp_meta = forward_batch.attn_cp_metadata
        q_prev = q[: cp_meta.total_q_prev_tokens]
        q_next = q[cp_meta.total_q_prev_tokens :]
        return [
            CPAttnSlice(
                q=q_prev,
                actual_seq_q=cp_meta.max_seqlen_q_prev,
                cache_seqlens_tensor=cp_meta.kv_len_prev_tensor,
                actual_seq_q_tensor=cp_meta.cu_seqlens_q_prev_tensor,
            ),
            CPAttnSlice(
                q=q_next,
                actual_seq_q=cp_meta.max_seqlen_q_next,
                cache_seqlens_tensor=cp_meta.kv_len_next_tensor,
                actual_seq_q_tensor=cp_meta.cu_seqlens_q_next_tensor,
            ),
        ]

    def materialize_full_kv(self, forward_batch, layer, k, v, swa_loc=None):
        from sglang.srt.layers.cp.utils import cp_allgather_and_save_kv_cache

        cp_allgather_and_save_kv_cache(
            forward_batch, layer, k, v, self.cp_size, swa_loc=swa_loc
        )
