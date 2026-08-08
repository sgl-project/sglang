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

"""Contiguous context parallel strategy.

Each sequence is split into ``cp_size`` contiguous blocks; rank ``r`` owns
block ``r`` of every sequence:

    cp0: block0        cp1: block1        cp2: block2        cp3: block3

This is the layout hybrid linear-attention (KDA) models need: the recurrent
state hand-off assumes each rank holds one contiguous span per sequence, so
no relayout is required around the (majority) linear layers. Dense/MLA layers
keep the all-gather-KV attention path; relative to zigzag they trade causal
load balance (later ranks attend to longer histories) for zero relayout —
the right trade when most layers are linear.

Block boundaries use the same integer split as the KDA op layer
(``build_cp_shard_layout``): block r covers ``[L*r//W, L*(r+1)//W)``. Keeping
the formulas identical is what lets the KDA backend derive its shard geometry
independently and stay consistent with the runner's sharding.
"""

from __future__ import annotations

from itertools import accumulate
from typing import Any, List, Optional

import torch

from sglang.srt.layers.cp.base import (
    ContextParallelStrategyKind,
    CPAttentionBackendKind,
)
from sglang.srt.layers.cp.zigzag import (
    ZigzagContextParallelMetadata,
    ZigzagCPStrategy,
)
from sglang.srt.runtime_context import get_device


class ContiguousCPStrategy(ZigzagCPStrategy):
    """Contiguous per-sequence sharding on top of zigzag's collectives.

    The shard/gather/materialize machinery of ``ZigzagCPStrategy`` is fully
    index-list-driven, so this strategy only redefines the layout metadata
    (one block per rank per sequence) and the attention dispatch (a single
    varlen call instead of zigzag's early/late block pair).
    """

    name = "contiguous"
    kind = ContextParallelStrategyKind.CONTIGUOUS

    def can_apply(self, num_tokens: int, forward_batch) -> bool:
        if self.cp_size <= 1 or num_tokens < self.cp_size:
            return False
        forward_mode = getattr(forward_batch, "forward_mode", None)
        if forward_mode is not None and not forward_mode.is_context_parallel_extend():
            return False

        extend_lens = getattr(forward_batch, "extend_seq_lens_cpu", None)
        if extend_lens is None:
            return True
        # Every rank must hold >= 1 token of every sequence: the dense
        # attention path does not handle zero-length local queries (the KDA
        # op layer's empty-shard compaction covers only the linear side).
        return all(int(length) >= self.cp_size for length in extend_lens)

    def get_supported_attention_backend(self):
        return [CPAttentionBackendKind.FLASH_ATTENTION]

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
        cp_size = self.cp_size
        if seqs_len is not None and len(seqs_len) == bs:
            prefix_offsets = [
                max(int(seqs_len[i]) - extend_seqs_len[i], 0) for i in range(bs)
            ]
        else:
            prefix_offsets = [0] * bs

        # Same boundaries as the KDA op layer's build_cp_shard_layout.
        per_seq_block_sizes: List[List[int]] = []
        split_list: List[int] = []
        for length in extend_seqs_len:
            block_sizes = [
                (length * (r + 1)) // cp_size - (length * r) // cp_size
                for r in range(cp_size)
            ]
            per_seq_block_sizes.append(block_sizes)
            split_list.extend(block_sizes)

        per_rank_actual_token = [
            sum(block_sizes[rank] for block_sizes in per_seq_block_sizes)
            for rank in range(cp_size)
        ]
        max_rank_len = [max(per_rank_actual_token)] * cp_size

        cp_rank = self.cp_rank
        # Blocks are numbered seq-major (seq0: 0..W-1, seq1: W..2W-1, ...).
        zigzag_index = [seq * cp_size + cp_rank for seq in range(bs)]

        # After the all-gather, chunks arrive rank-major: rank0's blocks for
        # every sequence, then rank1's, ... Reassemble to seq-major order.
        reverse_split_len = [
            per_seq_block_sizes[seq][rank]
            for rank in range(cp_size)
            for seq in range(bs)
        ]
        cp_reverse_index = [
            rank * bs + seq for seq in range(bs) for rank in range(cp_size)
        ]

        # Single-call attention geometry: this rank's queries per sequence
        # attend to the prefix plus every block up to and including its own.
        kv_len_list = [
            prefix_offsets[seq] + sum(per_seq_block_sizes[seq][: cp_rank + 1])
            for seq in range(bs)
        ]
        actual_seq_q_list = [per_seq_block_sizes[seq][cp_rank] for seq in range(bs)]

        try:
            device = torch.device(get_device().device)
        except Exception:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cu_q = [0] + list(accumulate(actual_seq_q_list))
        cu_kv = [0] + list(accumulate(kv_len_list))

        total_seq_lens = sum(extend_seqs_len)
        assert sum(split_list) == total_seq_lens
        assert sum(per_rank_actual_token) == total_seq_lens
        assert sorted(cp_reverse_index) == list(range(bs * cp_size))

        empty_i32 = torch.zeros(1, device=device, dtype=torch.int32)
        return ZigzagContextParallelMetadata(
            split_list=split_list,
            zigzag_index=zigzag_index,
            cp_reverse_index=cp_reverse_index,
            reverse_split_len=reverse_split_len,
            per_rank_actual_token=per_rank_actual_token,
            max_rank_len=max_rank_len,
            # Single-call geometry lives in the "prev" family; "next" is empty.
            kv_len_prev_tensor=torch.tensor(
                kv_len_list, device=device, dtype=torch.int32
            ),
            kv_len_next_tensor=empty_i32,
            cu_seqlens_kv_prev_tensor=torch.tensor(
                cu_kv, device=device, dtype=torch.int32
            ),
            cu_seqlens_kv_next_tensor=empty_i32,
            actual_seq_q_prev_tensor=torch.tensor(
                actual_seq_q_list, device=device, dtype=torch.int32
            ),
            actual_seq_q_next_tensor=empty_i32,
            cu_seqlens_q_prev_tensor=torch.tensor(
                cu_q, device=device, dtype=torch.int32
            ),
            cu_seqlens_q_next_tensor=empty_i32,
            total_q_prev_tokens=cu_q[-1],
            total_q_next_tokens=0,
            max_seqlen_q_prev=max(actual_seq_q_list) if actual_seq_q_list else 0,
            max_seqlen_q_next=0,
            kv_len_prev_list=kv_len_list,
            kv_len_next_list=[],
            actual_seq_q_prev_list=actual_seq_q_list,
            actual_seq_q_next_list=[],
            total_seq_lens=total_seq_lens,
            bs=bs,
        )

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
        logical_tokens = meta.total_q_prev_tokens
        result = attn_fn(
            q[:logical_tokens],
            meta.cu_seqlens_q_prev_tensor,
            meta.kv_len_prev_tensor,
            meta.max_seqlen_q_prev,
        )

        pad_size = q.shape[0] - logical_tokens
        assert pad_size >= 0
        if pad_size > 0:
            result = torch.cat(
                [result, result.new_zeros(pad_size, *result.shape[1:])], dim=0
            )
        return result
