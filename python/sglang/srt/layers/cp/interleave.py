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

"""Interleave context parallel strategy.

For ``cp_size = 4``, each rank owns every fourth token:

    cp0: token0, token4, token8,  token12, token16, ...
    cp1: token1, token5, token9,  token13, token17, ...
    cp2: token2, token6, token10, token14, token18, ...
    cp3: token3, token7, token11, token15, token19, ...

After all-gather, tokens are restored to the original order:

    token0, token1, token2, token3, token4, token5, token6, token7, ...
"""

from __future__ import annotations

from bisect import bisect_right
from contextlib import nullcontext
from dataclasses import dataclass
from itertools import accumulate
from typing import Any, List, Optional

import torch
import torch.nn.functional as F

from sglang.srt.distributed import get_attn_cp_group
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.layers.cp.base import (
    BaseContextParallelMetadata,
    ContextParallelStrategy,
    ContextParallelStrategyKind,
    CPAttentionBackendKind,
)
from sglang.srt.layers.dp_attention import is_allocation_symmetric
from sglang.srt.mem_cache.memory_pool import KVWriteLoc
from sglang.srt.model_executor.forward_context import get_token_to_kv_pool
from sglang.srt.runtime_context import get_parallel


@dataclass
class InterleaveContextParallelMetadata(BaseContextParallelMetadata):
    """Per-forward round-robin layout and causal attention metadata."""

    local_token_indices: Optional[Any] = None
    restore_indices: Optional[Any] = None
    restore_real_indices: Optional[Any] = None
    per_rank_actual_token: Optional[List[int]] = None
    per_rank_real_token: Optional[List[int]] = None
    max_rank_len: Optional[List[int]] = None
    max_rank_real_len: Optional[List[int]] = None
    query_request_indices: Optional[Any] = None
    query_cache_lengths: Optional[Any] = None
    cu_seqlens_q: Optional[Any] = None
    local_real_tokens: int = 0
    total_padded_tokens: int = 0


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
        extend_seqs_len = [
            int(x) for x in (extend_seqs_len or seqs_len or [num_tokens])
        ]
        real_num_tokens = sum(extend_seqs_len)
        if num_tokens < real_num_tokens:
            raise ValueError(
                f"Interleave CP received num_tokens={num_tokens}, smaller than "
                f"the real extend token count {real_num_tokens}."
            )

        if seqs_len is not None and len(seqs_len) == len(extend_seqs_len):
            prefix_lens = [
                max(int(seq_len) - extend_len, 0)
                for seq_len, extend_len in zip(seqs_len, extend_seqs_len)
            ]
        else:
            prefix_lens = [0] * len(extend_seqs_len)

        try:
            from sglang.srt.runtime_context import get_server_args

            device = torch.device(get_server_args().device)
        except Exception:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        per_rank_indices = [
            list(range(rank, num_tokens, self.cp_size)) for rank in range(self.cp_size)
        ]
        per_rank_real_indices = [
            [index for index in indices if index < real_num_tokens]
            for indices in per_rank_indices
        ]
        per_rank_actual_token = [len(indices) for indices in per_rank_indices]
        per_rank_real_token = [len(indices) for indices in per_rank_real_indices]
        max_rank_len = [max(per_rank_actual_token, default=0)] * self.cp_size
        max_rank_real_len = [max(per_rank_real_token, default=0)] * self.cp_size

        rank_major_indices = [
            index for indices in per_rank_indices for index in indices
        ]
        rank_major_real_indices = [
            index for indices in per_rank_real_indices for index in indices
        ]
        restore_indices = sorted(
            range(len(rank_major_indices)), key=rank_major_indices.__getitem__
        )
        restore_real_indices = sorted(
            range(len(rank_major_real_indices)),
            key=rank_major_real_indices.__getitem__,
        )

        local_indices = per_rank_indices[self.cp_rank]
        local_real_indices = per_rank_real_indices[self.cp_rank]
        seq_ends = list(accumulate(extend_seqs_len))
        request_indices: List[int] = []
        cache_lengths: List[int] = []
        for token_index in local_real_indices:
            request_index = bisect_right(seq_ends, token_index)
            seq_start = 0 if request_index == 0 else seq_ends[request_index - 1]
            request_indices.append(request_index)
            cache_lengths.append(
                prefix_lens[request_index] + token_index - seq_start + 1
            )

        local_real_tokens = len(local_real_indices)
        return InterleaveContextParallelMetadata(
            local_token_indices=torch.tensor(
                local_indices, device=device, dtype=torch.long
            ),
            restore_indices=torch.tensor(
                restore_indices, device=device, dtype=torch.long
            ),
            restore_real_indices=torch.tensor(
                restore_real_indices, device=device, dtype=torch.long
            ),
            per_rank_actual_token=per_rank_actual_token,
            per_rank_real_token=per_rank_real_token,
            max_rank_len=max_rank_len,
            max_rank_real_len=max_rank_real_len,
            query_request_indices=torch.tensor(
                request_indices, device=device, dtype=torch.long
            ),
            query_cache_lengths=torch.tensor(
                cache_lengths, device=device, dtype=torch.int32
            ),
            cu_seqlens_q=torch.arange(
                local_real_tokens + 1, device=device, dtype=torch.int32
            ),
            local_real_tokens=local_real_tokens,
            total_padded_tokens=num_tokens,
            total_seq_lens=real_num_tokens,
            bs=len(extend_seqs_len),
        )

    def shard_hidden_states(self, x: Any, forward_batch) -> Any:
        indices = forward_batch.attn_cp_metadata.local_token_indices.to(x.device)
        return x.index_select(0, indices)

    def shard_position_ids(self, positions: Any, forward_batch) -> Any:
        indices = forward_batch.attn_cp_metadata.local_token_indices.to(
            positions.device
        )
        return positions.index_select(-1, indices)

    def gather_hidden_states(
        self, x: Any, forward_batch, stream: Optional[Any] = None
    ) -> Any:
        return self._all_gather_reorganized(x, forward_batch, real_tokens_only=False)

    def gather_kv_cache(
        self, x: Any, forward_batch, stream: Optional[Any] = None
    ) -> Any:
        del stream
        real_tokens = forward_batch.attn_cp_metadata.local_real_tokens
        return self._all_gather_reorganized(
            x[:real_tokens], forward_batch, real_tokens_only=True
        )

    def get_supported_attention_backend(self):
        return [CPAttentionBackendKind.FLASH_ATTENTION]

    def run_attention(
        self,
        q: Any,
        forward_batch,
        device: Any,
        attn_fn,
        attention_backend: CPAttentionBackendKind = CPAttentionBackendKind.FLASH_ATTENTION,
    ) -> Any:
        assert (
            attention_backend in self.get_supported_attention_backend()
        ), f"{self.name} CP does not support {attention_backend=}"
        meta = forward_batch.attn_cp_metadata
        real_q = q[: meta.local_real_tokens]
        result = attn_fn(
            real_q,
            meta.cu_seqlens_q.to(q.device),
            meta.query_cache_lengths.to(q.device),
            1,
            meta.query_request_indices.to(q.device),
        )
        if result.shape[0] == q.shape[0]:
            return result
        if result.shape[0] > q.shape[0]:
            raise ValueError(
                f"Interleave attention returned {result.shape[0]} rows for "
                f"a local shard with {q.shape[0]} rows."
            )
        padding = result.new_zeros((q.shape[0] - result.shape[0], *result.shape[1:]))
        return torch.cat([result, padding], dim=0)

    def materialize_full_kv(
        self, forward_batch, layer: Any, k: Any, v: Any, swa_loc: Optional[Any] = None
    ) -> None:
        cache_loc = (
            forward_batch.out_cache_loc
            if not layer.is_cross_attention
            else forward_batch.encoder_out_cache_loc
        )
        real_num_tokens = forward_batch.attn_cp_metadata.total_seq_lens
        key_cache_full = self.gather_kv_cache(k.contiguous(), forward_batch)
        value_cache_full = self.gather_kv_cache(v.contiguous(), forward_batch)
        get_token_to_kv_pool().set_kv_buffer(
            layer,
            KVWriteLoc(cache_loc[:real_num_tokens], swa_loc),
            key_cache_full,
            value_cache_full,
            layer.k_scale,
            layer.v_scale,
        )

    def _all_gather_reorganized(
        self, x: torch.Tensor, forward_batch, *, real_tokens_only: bool
    ) -> torch.Tensor:
        meta = forward_batch.attn_cp_metadata
        counts = (
            meta.per_rank_real_token if real_tokens_only else meta.per_rank_actual_token
        )
        max_len = (
            meta.max_rank_real_len[0] if real_tokens_only else meta.max_rank_len[0]
        )
        restore_indices = (
            meta.restore_real_indices if real_tokens_only else meta.restore_indices
        )

        pad_size = max_len - x.shape[0]
        if pad_size < 0:
            raise ValueError(
                f"Interleave CP local tensor has {x.shape[0]} rows, exceeding "
                f"the metadata maximum {max_len}."
            )
        if pad_size:
            padding = [0, 0] * (x.ndim - 1) + [0, pad_size]
            x = F.pad(x, padding)

        group = get_parallel().attn_cp_group
        ctx = (
            use_symmetric_memory(group, disabled=not is_allocation_symmetric())
            if x.is_cuda
            else nullcontext()
        )
        with ctx:
            gathered = torch.empty(
                max_len * self.cp_size,
                *x.shape[1:],
                dtype=x.dtype,
                device=x.device,
            )
        get_attn_cp_group().all_gather_into_tensor(gathered, x)

        rank_chunks = torch.split(gathered, max_len, dim=0)
        rank_major = torch.cat(
            [chunk[:count] for chunk, count in zip(rank_chunks, counts)], dim=0
        )
        return rank_major.index_select(0, restore_indices.to(rank_major.device))
