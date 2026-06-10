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

"""Interleave context-parallel strategy."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import accumulate
from typing import TYPE_CHECKING, List, Tuple, Union

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
class InterleaveContextParallelMetadata(BaseContextParallelMetadata):
    per_rank_actual_token: List[int] = None
    max_rank_len: List[int] = None
    full_q_lens_cpu: List[int] = None
    local_q_lens_cpu: List[int] = None
    full_cu_seqlens_q: torch.Tensor = None
    full_cache_seqlens: torch.Tensor = None
    full_max_seqlen_q: int = 0
    local_cu_seqlens_q: torch.Tensor = None
    local_max_seqlen_q: int = 0


class InterleaveCPStrategy(ContextParallelStrategy):
    """Round-robin-split strategy.

    Token ``i`` goes to rank ``i % cp_size``. Each rank runs attention once on
    its strided slice. This strategy supports multi-batch prefill.
    """

    name = "interleave"
    kind = ContextParallelStrategyKind.INTERLEAVE

    def can_apply(self, num_tokens: int, forward_batch: "ForwardBatch") -> bool:
        # Interleave gathers rank-local tensors with all_gather_into_tensor, so
        # every CP rank must receive the same number of tokens.
        return (
            num_tokens > 0
            and num_tokens >= self.cp_size
            and num_tokens % self.cp_size == 0
            and self.cp_size > 1
            and forward_batch.forward_mode.is_context_parallel_extend()
        )

    def build_metadata(
        self,
        num_tokens: int,
        seqs_len: List[int] | None,
        extend_seqs_len: List[int] | None = None,
    ) -> InterleaveContextParallelMetadata:
        lens = extend_seqs_len if extend_seqs_len is not None else seqs_len
        full_q_lens_cpu = [int(x) for x in lens] if lens is not None else [num_tokens]
        full_cache_lens_cpu = (
            [int(x) for x in seqs_len]
            if seqs_len is not None and len(seqs_len) == len(full_q_lens_cpu)
            else list(full_q_lens_cpu)
        )
        pad_len = int(num_tokens) - sum(full_q_lens_cpu)
        if pad_len > 0:
            full_q_lens_cpu[-1] += pad_len
            full_cache_lens_cpu[-1] += pad_len

        local_q_lens_cpu = _interleave_q_lens_cpu(
            full_q_lens_cpu, self.cp_size, self.cp_rank
        )
        per_rank_actual_token = [
            sum(_interleave_q_lens_cpu(full_q_lens_cpu, self.cp_size, r))
            for r in range(self.cp_size)
        ]
        max_rank_len = [max(per_rank_actual_token)] * self.cp_size
        full_cu_q = [0] + list(accumulate(full_q_lens_cpu))
        local_cu_q = [0] + list(accumulate(local_q_lens_cpu))

        return InterleaveContextParallelMetadata(
            per_rank_actual_token=per_rank_actual_token,
            max_rank_len=max_rank_len,
            full_q_lens_cpu=full_q_lens_cpu,
            local_q_lens_cpu=local_q_lens_cpu,
            full_cu_seqlens_q=torch.tensor(full_cu_q, device="cuda", dtype=torch.int32),
            full_cache_seqlens=torch.tensor(
                full_cache_lens_cpu, device="cuda", dtype=torch.int32
            ),
            full_max_seqlen_q=max(full_q_lens_cpu) if full_q_lens_cpu else 0,
            local_cu_seqlens_q=torch.tensor(
                local_cu_q, device="cuda", dtype=torch.int32
            ),
            local_max_seqlen_q=max(local_q_lens_cpu) if local_q_lens_cpu else 0,
            total_seq_lens=int(num_tokens),
            bs=len(full_q_lens_cpu),
        )

    def shard_tokens(self, x, forward_batch):
        return _strided_take(x, self.cp_rank, self.cp_size)

    def shard_positions(self, positions, forward_batch):
        return _strided_take(positions, self.cp_rank, self.cp_size)

    def gather_tokens(self, x, forward_batch, stream=None):
        from sglang.srt.distributed.device_communicators.pynccl_allocator import (
            use_symmetric_memory,
        )
        from sglang.srt.layers.dp_attention import (
            attn_cp_all_gather_into_tensor,
            get_attention_cp_group,
            is_allocation_symmetric,
        )

        stream = stream if stream is not None else torch.cuda.current_stream()
        with torch.cuda.stream(stream):
            with use_symmetric_memory(
                get_attention_cp_group(), disabled=not is_allocation_symmetric()
            ):
                out = x.new_empty((x.shape[0] * self.cp_size, *x.shape[1:]))
            attn_cp_all_gather_into_tensor(out, x)
            out_shape = out.shape
            out = (
                out.view(self.cp_size, -1, *out_shape[1:])
                .transpose(0, 1)
                .reshape(out_shape)
            )
        torch.cuda.current_stream().wait_stream(stream)
        return out

    def gather_kv_cache(self, x, forward_batch, stream=None):
        return self.gather_tokens(x, forward_batch, stream)

    def shard_per_request(self, extend_seqs_cpu, extend_seqs):
        return _interleave_split_q_seqs(
            extend_seqs_cpu, extend_seqs, self.cp_size, self.cp_rank
        )

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

        cp_meta = forward_batch.attn_cp_metadata
        # FA's causal kvcache path assumes query tokens are a contiguous suffix.
        # Interleave tokens are strided, so run FA on full query order and then
        # shard the result back to this rank.
        q_full = self.gather_tokens(q, forward_batch)
        result_full = attn_fn(
            q_full,
            cp_meta.full_cu_seqlens_q,
            cp_meta.full_cache_seqlens,
            cp_meta.full_max_seqlen_q,
        )
        return self.shard_tokens(result_full, forward_batch)

    def materialize_full_kv(self, forward_batch, layer, k, v, swa_loc=None):
        from sglang.srt.mem_cache.memory_pool import KVWriteLoc
        from sglang.srt.model_executor.forward_context import get_token_to_kv_pool

        cache_loc = (
            forward_batch.out_cache_loc
            if not layer.is_cross_attention
            else forward_batch.encoder_out_cache_loc
        )
        k = k.contiguous()
        v = v.contiguous()
        get_token_to_kv_pool().set_kv_buffer(
            layer,
            KVWriteLoc(cache_loc, swa_loc),
            self.gather_kv_cache(k, forward_batch),
            self.gather_kv_cache(v, forward_batch),
            layer.k_scale,
            layer.v_scale,
        )

    def reindex_attn_metadata(self, core_attn_metadata) -> None:
        if hasattr(core_attn_metadata, "apply_cp_reindex"):
            core_attn_metadata.apply_cp_reindex()
        if hasattr(core_attn_metadata, "init_flashmla_related"):
            core_attn_metadata.init_flashmla_related()


def _strided_take(x: Union[torch.Tensor, List, Tuple], rank: int, world: int):
    if isinstance(x, tuple):
        return tuple(x[i] for i in range(rank, len(x), world))
    if isinstance(x, list):
        return [x[i] for i in range(rank, len(x), world)]

    n = len(x)
    if n % world != 0:
        cur_len = n // world + (n % world > rank)
        if cur_len == 0:
            return x.new_empty(0, *x.shape[1:])
        indices = torch.arange(rank, n, world, device=x.device)
        return x[indices]
    return x.view(-1, world, *x.shape[1:])[:, rank].contiguous()


def _interleave_split_q_seqs_cpu(
    extend_seqs_cpu: List[int], cp_size: int, cp_rank: int
) -> Tuple[List[int], List[int]]:
    q_seqs = _interleave_q_lens_cpu(extend_seqs_cpu, cp_size, cp_rank)
    bs_idx_cpu = [i for i, x in enumerate(q_seqs) if x > 0]
    ret_q_lens_cpu = [q for q in q_seqs if q > 0]
    return ret_q_lens_cpu, bs_idx_cpu


def _interleave_q_lens_cpu(
    extend_seqs_cpu: List[int], cp_size: int, cp_rank: int
) -> List[int]:
    extra_seq = 0
    q_seqs = []
    for cur_len in extend_seqs_cpu:
        cur_len += extra_seq
        cur_seq = cur_len // cp_size + int(cur_len % cp_size > cp_rank)
        q_seqs.append(cur_seq)
        extra_seq = cur_len - cur_seq * cp_size
    return q_seqs


def _interleave_split_q_seqs(extend_seqs_cpu, extend_seqs, cp_size: int, cp_rank: int):
    """Compute per-rank ``q_seqs`` and the ``bs_idx`` mask.

    The CPU pass is mirrored by a Triton kernel for GPU tensors.
    """
    from sglang.srt.layers.attention.dsa.utils import (
        dsa_cp_round_robin_split_q_seqs_kernel,
    )

    ret_q_lens_cpu, bs_idx_cpu = _interleave_split_q_seqs_cpu(
        extend_seqs_cpu, cp_size, cp_rank
    )
    ret_q_lens = torch.empty(
        (len(bs_idx_cpu),), device=extend_seqs.device, dtype=extend_seqs.dtype
    )
    bs_idx = torch.empty(
        (len(bs_idx_cpu),), device=extend_seqs.device, dtype=torch.int32
    )
    dsa_cp_round_robin_split_q_seqs_kernel[(1,)](
        extend_seqs, ret_q_lens, bs_idx, len(extend_seqs), cp_size, cp_rank
    )
    return ret_q_lens_cpu, ret_q_lens, bs_idx_cpu, bs_idx
