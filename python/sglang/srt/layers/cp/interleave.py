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


@dataclass
class InterleaveContextParallelMetadata(BaseContextParallelMetadata):
    per_rank_actual_token: Optional[List[int]] = None
    max_rank_len: Optional[List[int]] = None
    per_rank_logical_token: Optional[List[int]] = None


class InterleaveCPStrategy(ContextParallelStrategy):
    name = "interleave"
    kind = ContextParallelStrategyKind.INTERLEAVE

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
        """Split tokens evenly by the rule ``token_idx % cp_size``."""
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

        # for torch device tensor
        return input_.view(-1, cp_size, *input_.shape[1:])[:, cp_rank].contiguous()

    def shard_local_tokens(self, input_: Any) -> Any:
        """Round-robin split a per-token tensor/list to this CP rank's local tokens."""
        return self._interleave_shard(input_)

    def shard_per_request(
        self,
        extend_seqs_cpu: List[int],
        extend_seqs: Any,
    ):
        """Round-robin per-request Q-length split across CP ranks.

        Distributes each request's tokens by ``token_idx % cp_size`` and returns
        this rank's per-request lengths (zeros dropped) plus the indices of the
        requests that keep at least one token:
        ``(q_lens_cpu, q_lens, bs_idx_cpu, bs_idx)``. The ``_cpu`` lists are built
        on host; ``q_lens`` / ``bs_idx`` are produced on-device by the shared
        ``dsa_cp_round_robin_split_q_seqs_kernel`` so the split stays graph-safe.
        """
        from sglang.kernels.ops.attention.dsa.cp_split import (
            dsa_cp_round_robin_split_q_seqs_kernel,
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
        dsa_cp_round_robin_split_q_seqs_kernel[(1,)](
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

        flat_indices = torch.arange(total_tokens, device=x.device)
        gather_indices = (
            flat_indices % self.cp_size
        ) * physical_rank_len + flat_indices // self.cp_size
        return gathered.index_select(0, gather_indices)

    def get_supported_attention_backend(self):
        return [CPAttentionBackendKind.DSA]

    def materialize_full_indexer_k_cache(self, key: Any, forward_batch) -> Any:
        """CP-v2 DSA indexer hook: all-gather the rank-local indexer key."""
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
        # No-op: run_attention is the FlashAttention/zigzag dispatch hook.
        # Interleave serves the DSA backend, which runs attention itself.
        return None

    def all_gather_dsa_trtllm_fp8_kv(self, forward_batch, k: Any, k_rope: Any) -> Any:
        """All-gather FP8 KV for the TRT-LLM MLA backend under CP-v2.

        Concatenates k and k_rope along the last dim, gathers across CP ranks
        in full interleave order, then splits back into (k, k_rope).
        """
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
        **kwargs,
    ) -> Any:
        """CP-v2 DSA MLA: rebuild the full-sequence latent KV from rank-local shards."""
        latent_cache = kwargs["latent_cache"]
        k_nope = kwargs["k_nope"]
        k_pe = kwargs["k_pe"]
        kv_lora_rank = kwargs["kv_lora_rank"]
        latent_cache[..., :kv_lora_rank] = k_nope.squeeze(1)
        latent_cache[..., kv_lora_rank:] = k_pe.squeeze(1)
        full_latent = self.gather_kv_cache(
            latent_cache.contiguous(), forward_batch, torch.cuda.current_stream()
        )
        k_nope = full_latent[..., :kv_lora_rank].unsqueeze(1)
        k_pe = full_latent[..., kv_lora_rank:].unsqueeze(1)
        return k_nope, k_pe
