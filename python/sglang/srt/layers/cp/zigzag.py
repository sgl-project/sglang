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

"""Zigzag context parallel strategy shell.

For ``cp_size = 4``, each sequence is split into ``2 * cp_size`` blocks. Each
rank owns one early block and one late block:

    cp0: block0, block7
    cp1: block1, block6
    cp2: block2, block5
    cp3: block3, block4

After all-gather, the blocks are reranged back to their original order:

    block0 | block7 | block1 | block6 | block2 | block5 | block3 | block4
      -> block0 | block1 | block2 | block3 | block4 | block5 | block6 | block7
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from itertools import accumulate
from typing import Any, List, Optional

import torch
import torch.nn.functional as F

from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.layers.cp.base import (
    BaseContextParallelMetadata,
    ContextParallelStrategy,
    ContextParallelStrategyKind,
    CPAttentionBackendKind,
)
from sglang.srt.layers.dp_attention import (
    is_allocation_symmetric,
)
from sglang.srt.mem_cache.memory_pool import KVWriteLoc
from sglang.srt.model_executor.forward_context import get_token_to_kv_pool
from sglang.srt.runtime_context import get_parallel


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

# CP-v2 is kept for page-sized-or-larger prefill chunks. Smaller chunks stay on
# the legacy path, but uneven long chunks are supported by the zigzag metadata's
# per-rank padding.
MIN_ZIGZAG_CP_V2_TOKENS = 128


class ZigzagCPStrategy(ContextParallelStrategy):
    name = "zigzag"
    kind = ContextParallelStrategyKind.ZIGZAG

    def can_apply(self, num_tokens: int, forward_batch) -> bool:
        min_cp_tokens = max(MIN_ZIGZAG_CP_V2_TOKENS, self.cp_size * 2)
        if self.cp_size <= 1 or num_tokens < min_cp_tokens:
            return False
        forward_mode = getattr(forward_batch, "forward_mode", None)
        if forward_mode is not None and not forward_mode.is_context_parallel_extend():
            return False

        extend_lens = getattr(forward_batch, "extend_seq_lens_cpu", None)
        if extend_lens is None:
            return True

        return all(int(length) >= min_cp_tokens for length in extend_lens)

    def build_metadata(
        self,
        num_tokens: int,
        seqs_len: Optional[List[int]],
        extend_seqs_len: Optional[List[int]] = None,
    ) -> ZigzagContextParallelMetadata:
        if extend_seqs_len is None:
            extend_seqs_len = seqs_len or [num_tokens]
        extend_seqs_len = [int(x) for x in extend_seqs_len]

        pad_len = int(num_tokens) - sum(extend_seqs_len)
        if pad_len > 0:
            extend_seqs_len[-1] += pad_len
            if seqs_len is not None and len(seqs_len) == len(extend_seqs_len):
                seqs_len = list(seqs_len)
                seqs_len[-1] += pad_len

        bs = len(extend_seqs_len)
        cp_segment_num = self.cp_size * 2
        if seqs_len is not None and len(seqs_len) == bs:
            prefix_offsets = [
                max(int(seqs_len[i]) - extend_seqs_len[i], 0) for i in range(bs)
            ]
        else:
            prefix_offsets = [0] * bs

        # TODO: move these per-request layout/index computations to a Triton
        # kernel if Python-side metadata construction becomes a bottleneck.
        per_seq_block_sizes: List[List[int]] = []
        split_list: List[int] = []
        for length in extend_seqs_len:
            base = length // cp_segment_num
            rem = length % cp_segment_num
            block_sizes = [
                base + 1 if block_id < rem else base
                for block_id in range(cp_segment_num)
            ]
            per_seq_block_sizes.append(block_sizes)
            split_list.extend(block_sizes)

        per_rank_actual_token = []
        for rank in range(self.cp_size):
            per_rank_actual_token.append(
                sum(
                    block_sizes[rank] + block_sizes[cp_segment_num - 1 - rank]
                    for block_sizes in per_seq_block_sizes
                )
            )
        max_rank_len = [max(per_rank_actual_token)] * self.cp_size

        cp_rank = self.cp_rank
        zigzag_index = list(
            range(cp_rank, cp_rank + bs * cp_segment_num, cp_segment_num)
        ) + list(
            range(
                cp_segment_num - cp_rank - 1,
                bs * cp_segment_num,
                cp_segment_num,
            )
        )

        cp_reverse_index: List[int] = []
        for batch_id in range(bs):
            cp_reverse_index.extend(
                list(range(batch_id, cp_segment_num * bs, 2 * bs))
                + list(
                    range(
                        (cp_segment_num - 1) * bs + batch_id,
                        0,
                        -2 * bs,
                    )
                )
            )

        reverse_split_len: List[int] = []
        for rank in range(self.cp_size):
            for batch_id in range(bs):
                reverse_split_len.append(per_seq_block_sizes[batch_id][rank])
            for batch_id in range(bs):
                reverse_split_len.append(
                    per_seq_block_sizes[batch_id][cp_segment_num - 1 - rank]
                )

        kv_len_prev_list: List[int] = []
        kv_len_next_list: List[int] = []
        actual_seq_q_prev_list: List[int] = []
        actual_seq_q_next_list: List[int] = []
        for batch_id, block_sizes in enumerate(per_seq_block_sizes):
            kv_len_prev_list.append(
                prefix_offsets[batch_id] + sum(block_sizes[: cp_rank + 1])
            )
            kv_len_next_list.append(
                prefix_offsets[batch_id] + sum(block_sizes[: cp_segment_num - cp_rank])
            )
            actual_seq_q_prev_list.append(block_sizes[cp_rank])
            actual_seq_q_next_list.append(block_sizes[cp_segment_num - cp_rank - 1])

        from sglang.srt.runtime_context import get_server_args

        try:
            device = torch.device(get_server_args().device)
        except Exception:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cu_prev = [0] + list(accumulate(actual_seq_q_prev_list))
        cu_next = [0] + list(accumulate(actual_seq_q_next_list))

        total_seq_lens = sum(extend_seqs_len)
        assert len(split_list) == bs * cp_segment_num
        assert sum(split_list) == total_seq_lens
        assert len(zigzag_index) == 2 * bs
        assert len(cp_reverse_index) == bs * cp_segment_num
        assert sorted(cp_reverse_index) == list(range(bs * cp_segment_num))
        assert sum(per_rank_actual_token) == total_seq_lens

        return ZigzagContextParallelMetadata(
            split_list=split_list,
            zigzag_index=zigzag_index,
            cp_reverse_index=cp_reverse_index,
            reverse_split_len=reverse_split_len,
            per_rank_actual_token=per_rank_actual_token,
            max_rank_len=max_rank_len,
            kv_len_prev_tensor=torch.tensor(
                kv_len_prev_list, device=device, dtype=torch.int32
            ),
            kv_len_next_tensor=torch.tensor(
                kv_len_next_list, device=device, dtype=torch.int32
            ),
            actual_seq_q_prev_tensor=torch.tensor(
                actual_seq_q_prev_list, device=device, dtype=torch.int32
            ),
            actual_seq_q_next_tensor=torch.tensor(
                actual_seq_q_next_list, device=device, dtype=torch.int32
            ),
            cu_seqlens_q_prev_tensor=torch.tensor(
                cu_prev, device=device, dtype=torch.int32
            ),
            cu_seqlens_q_next_tensor=torch.tensor(
                cu_next, device=device, dtype=torch.int32
            ),
            total_q_prev_tokens=cu_prev[-1],
            total_q_next_tokens=cu_next[-1],
            max_seqlen_q_prev=(
                max(actual_seq_q_prev_list) if actual_seq_q_prev_list else 0
            ),
            max_seqlen_q_next=(
                max(actual_seq_q_next_list) if actual_seq_q_next_list else 0
            ),
            kv_len_prev_list=kv_len_prev_list,
            kv_len_next_list=kv_len_next_list,
            actual_seq_q_prev_list=actual_seq_q_prev_list,
            actual_seq_q_next_list=actual_seq_q_next_list,
            total_seq_lens=total_seq_lens,
            bs=bs,
        )

    def shard_hidden_states(self, x: Any, forward_batch) -> Any:
        chunks = torch.split(x, forward_batch.attn_cp_metadata.split_list, dim=0)
        return torch.cat(
            [chunks[i] for i in forward_batch.attn_cp_metadata.zigzag_index], dim=0
        )

    def shard_position_ids(self, positions: Any, forward_batch) -> Any:
        chunks = torch.split(
            positions, forward_batch.attn_cp_metadata.split_list, dim=-1
        )
        return torch.cat(
            [chunks[i] for i in forward_batch.attn_cp_metadata.zigzag_index], dim=-1
        )

    def gather_hidden_states(
        self, x: Any, forward_batch, stream: Optional[Any] = None
    ) -> Any:
        gathered = self._all_gather_reorganized(x, forward_batch, stream)
        chunks = torch.split(
            gathered, forward_batch.attn_cp_metadata.reverse_split_len, dim=0
        )
        return torch.cat(
            [chunks[i] for i in forward_batch.attn_cp_metadata.cp_reverse_index], dim=0
        )

    def gather_kv_cache(
        self, x: Any, forward_batch, stream: Optional[Any] = None
    ) -> Any:
        gathered = self._all_gather_reorganized(x, forward_batch, stream)
        chunks = torch.split(
            gathered, forward_batch.attn_cp_metadata.reverse_split_len, dim=0
        )
        return torch.cat(
            [chunks[i] for i in forward_batch.attn_cp_metadata.cp_reverse_index], dim=0
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
        q_prev = q[: meta.total_q_prev_tokens]
        q_next = q[meta.total_q_prev_tokens :]

        result_prev = attn_fn(
            q_prev,
            meta.cu_seqlens_q_prev_tensor,
            meta.kv_len_prev_tensor,
            meta.max_seqlen_q_prev,
        )
        result_next = attn_fn(
            q_next,
            meta.cu_seqlens_q_next_tensor,
            meta.kv_len_next_tensor,
            meta.max_seqlen_q_next,
        )
        return torch.cat([result_prev, result_next], dim=0)

    def materialize_full_kv(
        self, forward_batch, layer: Any, k: Any, v: Any, swa_loc: Optional[Any] = None
    ) -> None:
        cache_loc = (
            forward_batch.out_cache_loc
            if not layer.is_cross_attention
            else forward_batch.encoder_out_cache_loc
        )
        key_cache_full = self.gather_kv_cache(
            k.contiguous(), forward_batch, torch.cuda.current_stream()
        )
        value_cache_full = self.gather_kv_cache(
            v.contiguous(), forward_batch, torch.cuda.current_stream()
        )
        get_token_to_kv_pool().set_kv_buffer(
            layer,
            KVWriteLoc(cache_loc, swa_loc),
            key_cache_full,
            value_cache_full,
            layer.k_scale,
            layer.v_scale,
        )

    def _all_gather_reorganized(self, x: torch.Tensor, forward_batch, stream):
        meta = forward_batch.attn_cp_metadata
        max_len = meta.max_rank_len[0]
        pad_size = max_len - x.shape[0]
        if pad_size > 0:
            padding = [0, 0] * (x.ndim - 1) + [0, pad_size]
            x = F.pad(x, padding, mode="constant", value=0)

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
                device=x.device,
                dtype=x.dtype,
            )
        if len(set(meta.per_rank_actual_token)) > 1:
            group.barrier()
        group.cp_all_gather_into_tensor_async(gathered, x, stream)

        chunks = torch.split(gathered, meta.max_rank_len, dim=0)
        return torch.cat(
            [
                chunks[rank][:per_rank_len]
                for rank, per_rank_len in enumerate(meta.per_rank_actual_token)
            ],
            dim=0,
        )
