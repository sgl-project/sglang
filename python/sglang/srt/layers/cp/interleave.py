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

    cp0: token0, token4, token8,  token12, token16, ...
    cp1: token1, token5, token9,  token13, token17, ...
    cp2: token2, token6, token10, token14, token18, ...
    cp3: token3, token7, token11, token15, token19, ...

After all-gather, tokens are restored to the original order:

    token0, token1, token2, token3, token4, token5, token6, token7, ...
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

import torch

from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.layers.cp.base import (
    BaseContextParallelMetadata,
    ContextParallelStrategy,
    ContextParallelStrategyKind,
    CPAttentionBackendKind,
)
from sglang.srt.layers.cp.padding import pad_local_rows
from sglang.srt.layers.dp_attention import (
    attn_cp_all_gather_into_tensor,
    is_allocation_symmetric,
)
from sglang.srt.runtime_context import get_parallel


def attn_cp_interleave_gather(hidden_states: torch.Tensor):
    """Gather equal padded interleave shards in rank order, not token order.

    Size from the actual shard: the DP scratch length can already describe a
    shard when dense FFNs run over TP, and is not a CP collective's output size.
    """
    parallel = get_parallel()
    with use_symmetric_memory(
        parallel.attn_cp_group, disabled=not is_allocation_symmetric()
    ):
        gathered = hidden_states.new_empty(
            (hidden_states.shape[0] * parallel.attn_cp_size, *hidden_states.shape[1:])
        )
    attn_cp_all_gather_into_tensor(gathered, hidden_states.contiguous())
    return gathered


@dataclass
class InterleaveContextParallelMetadata(BaseContextParallelMetadata):
    per_rank_actual_token: Optional[List[int]] = None
    max_rank_len: Optional[List[int]] = None
    per_rank_logical_token: Optional[List[int]] = None
    # Tail row -> packed all-gather slot; local tail metadata rows include padding.
    gather_index: Optional[torch.Tensor] = None
    local_index: Optional[torch.Tensor] = None
    moe_local_token_count: Optional[torch.Tensor] = None


def is_interleave_extend(forward_batch):
    return forward_batch.forward_mode.is_context_parallel_extend() and isinstance(
        forward_batch.attn_cp_metadata, InterleaveContextParallelMetadata
    )


def cp_interleave_to_sequence_order(hidden_states, forward_batch):
    """Restore sequence order after CP attention and before linear attention.

    With CP2 and seven tokens, the boundary all-gathers the interleaved shards
    as ``[t0, t2, t4, t6, t1, t3, t5, pad]``. This returns
    ``[t0, t1, t2, t3, t4, t5, t6]`` for sequence-dependent compute.

    The residual and mHC coefficients do not move: only the already-read mixer
    input is reordered. Remove physical CP padding before the recurrent kernel.
    """
    if not is_interleave_extend(forward_batch):
        return hidden_states
    metadata = forward_batch.attn_cp_metadata
    if metadata.gather_index is not None:
        return hidden_states.index_select(0, metadata.gather_index)
    size = get_parallel().attn_cp_size
    rows = hidden_states.shape[0] // size
    return (
        hidden_states.reshape(size, rows, *hidden_states.shape[1:])
        .transpose(0, 1)
        .flatten(0, 1)[: metadata.total_seq_lens]
        .contiguous()
    )


def cp_sequence_to_interleave_order(hidden_states, forward_batch, gathered_rows):
    """Restore interleave order after linear attention and before CP attention.

    With CP2 and seven tokens, ``[t0, t1, t2, t3, t4, t5, t6]`` becomes
    ``[t0, t2, t4, t6, t1, t3, t5, 0]``. These padded rank-major rows match
    the boundary's reduce-scatter and the local residual/mHC rows. The reorder
    itself performs no communication or reduction.
    """
    if not is_interleave_extend(forward_batch):
        return hidden_states
    metadata = forward_batch.attn_cp_metadata
    padded = hidden_states.new_zeros((gathered_rows, *hidden_states.shape[1:]))
    if metadata.gather_index is not None:
        return padded.index_copy_(0, metadata.gather_index, hidden_states)
    padded[: hidden_states.shape[0]] = hidden_states
    size = get_parallel().attn_cp_size
    return (
        padded.reshape(-1, size, *hidden_states.shape[1:])
        .transpose(0, 1)
        .flatten(0, 1)
        .contiguous()
    )


class InterleaveCPStrategy(ContextParallelStrategy):
    name = "interleave"
    kind = ContextParallelStrategyKind.INTERLEAVE

    def moe_num_token_non_padded(self, forward_batch):
        """Mask physical CP padding before the dispatch/combine all-to-alls.

        Attention-TP localization does not split the count over CP ranks.
        Interleave's valid rows form a prefix of each padded local shard.
        """
        metadata = forward_batch.attn_cp_metadata
        if metadata.moe_local_token_count is None:
            lengths = metadata.per_rank_logical_token or metadata.per_rank_actual_token
            metadata.moe_local_token_count = torch.tensor(
                lengths[self.cp_rank],
                dtype=torch.int32,
                device=forward_batch.input_ids.device,
            )
        return metadata.moe_local_token_count

    def can_apply(self, num_tokens: int, forward_batch) -> bool:
        if not forward_batch.forward_mode.is_context_parallel_extend():
            return False
        cp_size = self.cp_size
        seq_len = sum(forward_batch.extend_seq_lens_cpu)
        return seq_len > 0 and seq_len >= cp_size and cp_size > 1

    def build_metadata(
        self,
        num_tokens: int,
        seqs_len: Optional[List[int]],
        extend_seqs_len: Optional[List[int]] = None,
    ) -> InterleaveContextParallelMetadata:
        if extend_seqs_len is None:
            extend_seqs_len = seqs_len or [num_tokens]
        extend_seqs_len = [int(x) for x in extend_seqs_len]

        pad_len = int(num_tokens) - sum(extend_seqs_len)
        if pad_len > 0:
            extend_seqs_len[-1] += pad_len

        total_seq_lens = sum(extend_seqs_len)
        base_len, extra = divmod(total_seq_lens, self.cp_size)
        per_rank_actual_token = [
            base_len + (rank < extra) for rank in range(self.cp_size)
        ]

        return InterleaveContextParallelMetadata(
            per_rank_actual_token=per_rank_actual_token,
            max_rank_len=[max(per_rank_actual_token)] * self.cp_size,
            total_seq_lens=total_seq_lens,
            bs=len(extend_seqs_len),
        )

    def shard_hidden_states(self, x: Any, forward_batch) -> Any:
        metadata = forward_batch.attn_cp_metadata
        local_x = self._interleave_shard(x[: metadata.total_seq_lens])
        return pad_local_rows(local_x, metadata, dim=0)

    def shard_position_ids(self, positions: Any, forward_batch) -> Any:
        metadata = forward_batch.attn_cp_metadata
        local_positions = self._interleave_shard(positions[: metadata.total_seq_lens])
        return pad_local_rows(local_positions, metadata, dim=0)

    def _interleave_shard(self, input_: Any) -> Any:
        cp_size = self.cp_size
        cp_rank = self.cp_rank
        if isinstance(input_, (tuple, list)):
            indices = range(cp_rank, len(input_), cp_size)
            return input_[indices]

        tokens = len(input_)
        if tokens % cp_size != 0:
            cur_len = tokens // cp_size + (tokens % cp_size > cp_rank)
            if cur_len == 0:
                return input_.new_empty(0, *input_.shape[1:])
            indices = torch.arange(cp_rank, tokens, cp_size, device=input_.device)
            return input_[indices]

        return input_.view(-1, cp_size, *input_.shape[1:])[:, cp_rank].contiguous()

    def local_q_indices(self, num_tokens: int, forward_batch) -> Any:
        device = getattr(getattr(forward_batch, "input_ids", None), "device", None)
        if device is None:
            device = torch.device("cpu")
        return torch.arange(
            self.cp_rank, int(num_tokens), self.cp_size, device=device, dtype=torch.long
        )

    def shard_local_tokens(self, input_: Any) -> Any:
        return self._interleave_shard(input_)

    def shard_per_request(
        self,
        extend_seqs_cpu: List[int],
        extend_seqs: Any,
    ):
        """Build device outputs in the shared kernel to keep the split graph-safe."""
        from sglang.kernels.ops.attention.dsa.cp_split import (
            dsa_cp_interleave_q_seqs_kernel,
        )

        cp_size = self.cp_size
        cp_rank = self.cp_rank

        extra_seq = 0
        q_lens_cpu: List[int] = []
        for cur_len in extend_seqs_cpu:
            cur_len += extra_seq
            cur_seq = cur_len // cp_size + int(cur_len % cp_size > cp_rank)
            q_lens_cpu.append(cur_seq)
            extra_seq = cur_len - cur_seq * cp_size
        bs_idx_cpu = [i for i, q_len in enumerate(q_lens_cpu) if q_len > 0]
        q_lens_cpu = [q_len for q_len in q_lens_cpu if q_len > 0]

        q_lens = torch.empty(
            (len(bs_idx_cpu),), device=extend_seqs.device, dtype=extend_seqs.dtype
        )
        bs_idx = torch.empty(
            (len(bs_idx_cpu),), device=extend_seqs.device, dtype=torch.int32
        )
        dsa_cp_interleave_q_seqs_kernel[(1,)](
            extend_seqs, q_lens, bs_idx, len(extend_seqs), cp_size, cp_rank
        )
        return q_lens_cpu, q_lens, bs_idx_cpu, bs_idx

    def gather_hidden_states(
        self, x: Any, forward_batch, stream: Optional[Any] = None
    ) -> Any:
        return self._gather_interleaved_tensor(x, forward_batch)

    def gather_kv_cache(
        self, x: Any, forward_batch, stream: Optional[Any] = None
    ) -> Any:
        return self._gather_interleaved_tensor(x, forward_batch)

    def _gather_interleaved_tensor(self, x: Any, forward_batch) -> Any:
        metadata = getattr(forward_batch, "attn_cp_metadata", None)
        if metadata is None:
            raise RuntimeError("Interleave CP gather requires attn_cp_metadata.")

        total_tokens = int(metadata.total_seq_lens)
        if total_tokens < 0:
            raise RuntimeError(
                f"Invalid interleave CP total_seq_lens={total_tokens}; expected >= 0."
            )

        logical_rank_lens = (
            metadata.per_rank_logical_token or metadata.per_rank_actual_token
        )
        local_logical_len = logical_rank_lens[self.cp_rank]
        if x.shape[0] < local_logical_len:
            raise RuntimeError(
                "Interleave CP gather received an unexpected local token count: "
                f"rank={self.cp_rank}, got={x.shape[0]}, "
                f"expected_at_least={local_logical_len}, "
                f"total={total_tokens}, cp_size={self.cp_size}."
            )

        physical_rank_len = max(metadata.per_rank_actual_token)
        if physical_rank_len == 0:
            return x.new_empty((0, *x.shape[1:]))

        padded_x = x.new_zeros((physical_rank_len, *x.shape[1:]))
        padded_x[:local_logical_len] = x[:local_logical_len]

        with use_symmetric_memory(
            get_parallel().attn_cp_group, disabled=not is_allocation_symmetric()
        ):
            gathered = x.new_empty((self.cp_size * physical_rank_len, *x.shape[1:]))
        attn_cp_all_gather_into_tensor(gathered, padded_x.contiguous())

        if metadata.gather_index is not None:
            return gathered.index_select(0, metadata.gather_index)

        # Equal per-rank lengths: one interleave copy restores the original
        # token order; cheaper than the index_select fallback below.
        actual = metadata.per_rank_actual_token
        if total_tokens == self.cp_size * physical_rank_len and all(
            int(n) == physical_rank_len for n in actual
        ):
            return (
                gathered.view(self.cp_size, physical_rank_len, *x.shape[1:])
                .transpose(0, 1)
                .reshape(total_tokens, *x.shape[1:])
            )

        flat_indices = torch.arange(total_tokens, device=x.device)
        gather_indices = (
            flat_indices % self.cp_size
        ) * physical_rank_len + flat_indices // self.cp_size
        return gathered.index_select(0, gather_indices)

    def get_supported_attention_backend(self):
        return [CPAttentionBackendKind.DSA, CPAttentionBackendKind.FLASH_ATTENTION]

    def materialize_full_indexer_k_cache(self, key: Any, forward_batch) -> Any:
        return self.gather_kv_cache(
            key.contiguous(), forward_batch, torch.cuda.current_stream()
        )

    def run_attention(
        self,
        q: Any,
        forward_batch,
        device: Any,
        attn_fn,
        attention_backend: CPAttentionBackendKind = CPAttentionBackendKind.FLASH_ATTENTION,
        **kwargs,
    ) -> Any:
        assert attention_backend == CPAttentionBackendKind.FLASH_ATTENTION
        # One logical sequence per query gives FA's bottom-right causal mask
        # the true query position. Treating a strided shard as a contiguous
        # request suffix would expose future keys and shift the SWA window.
        num_tokens = sum(forward_batch.extend_seq_lens_cpu)
        indices = self.local_q_indices(num_tokens, forward_batch)
        ends = forward_batch.extend_seq_lens.cumsum(0)
        request_indices = torch.searchsorted(ends, indices, right=True)
        starts = ends - forward_batch.extend_seq_lens
        lengths = (
            forward_batch.extend_prefix_lens[request_indices]
            + indices
            - starts[request_indices]
            + 1
        ).to(torch.int32)
        num_queries = indices.shape[0]
        cu_q = torch.arange(num_queries + 1, dtype=torch.int32, device=device)
        result = attn_fn(
            q[:num_queries],
            cu_q,
            lengths,
            1,
            request_indices=request_indices,
        )
        # Physical collective padding must never become an attention query.
        pad_size = q.shape[0] - num_queries
        assert pad_size >= 0
        if pad_size:
            result = torch.cat(
                [result, result.new_zeros(pad_size, *result.shape[1:])], dim=0
            )
        return result

    def all_gather_dsa_trtllm_fp8_kv(self, forward_batch, k: Any, k_rope: Any) -> Any:
        kv_lora_rank = k.shape[-1]
        qk_rope_head_dim = k_rope.shape[-1]
        kv_dtype = k.dtype
        # Pack → gather in raw bytes to avoid dtype issues with FP8
        kv = torch.cat((k, k_rope), dim=-1).view(torch.uint8)
        kv = self.gather_kv_cache(
            kv.contiguous(), forward_batch, torch.cuda.current_stream()
        ).view(kv_dtype)
        return kv.split((kv_lora_rank, qk_rope_head_dim), dim=-1)

    def materialize_full_kv(
        self,
        forward_batch,
        layer: Any = None,
        k: Any = None,
        v: Any = None,
        swa_loc: Optional[Any] = None,
    ) -> Any:
        from sglang.srt.mem_cache.memory_pool import KVWriteLoc
        from sglang.srt.model_executor.forward_context import get_token_to_kv_pool

        k_dim, v_dim = k.shape[-1], v.shape[-1]
        full_k, full_v = self.gather_kv_cache(
            torch.cat([k, v], dim=-1).contiguous(), forward_batch
        ).split([k_dim, v_dim], dim=-1)
        get_token_to_kv_pool().set_kv_buffer(
            layer,
            KVWriteLoc.for_layer(forward_batch, layer, swa_loc=swa_loc),
            full_k.contiguous(),
            full_v.contiguous(),
            layer.k_scale,
            layer.v_scale,
        )

    def materialize_full_mla_kv(
        self,
        forward_batch,
        layer: Any,
        k_nope: Any,
        k_rope: Any,
    ) -> Any:
        kv_lora_rank = k_nope.shape[-1]
        latent_cache = torch.cat([k_nope, k_rope], dim=-1).squeeze(1)
        full_latent = self.gather_kv_cache(
            latent_cache.contiguous(), forward_batch, torch.cuda.current_stream()
        )
        k_nope = full_latent[..., :kv_lora_rank].unsqueeze(1)
        k_rope = full_latent[..., kv_lora_rank:].unsqueeze(1)
        return k_nope, k_rope


def interleave_rows_per_request(
    extend_lens: List[int], cp_rank: int, cp_size: int
) -> List[int]:
    """Rows of each request a CP rank holds: global token index congruent to cp_rank."""
    counts, start = [], 0
    for n in extend_lens:
        end = start + n
        counts.append((end - 1 - cp_rank) // cp_size - (start - 1 - cp_rank) // cp_size)
        start = end
    return counts
