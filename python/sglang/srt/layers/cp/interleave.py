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
from sglang.srt.layers.dp_attention import (
    attn_cp_all_gather_into_tensor,
    get_attention_cp_group,
    is_allocation_symmetric,
)


@dataclass
class InterleaveContextParallelMetadata(BaseContextParallelMetadata):
    """Interleave has no per-forward zigzag permutation payload."""


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

        return InterleaveContextParallelMetadata(
            total_seq_lens=sum(extend_seqs_len),
            bs=len(extend_seqs_len),
        )

    def shard_hidden_states(self, x: Any, forward_batch) -> Any:
        return self._round_robin_shard(x)

    def shard_position_ids(self, positions: Any, forward_batch) -> Any:
        return self._round_robin_shard(positions)

    def _round_robin_shard(self, input_: Any) -> Any:
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

        base_len, extra = divmod(total_tokens, self.cp_size)
        per_rank_lens = [
            base_len + (1 if rank < extra else 0) for rank in range(self.cp_size)
        ]
        local_len = per_rank_lens[self.cp_rank]
        if x.shape[0] != local_len:
            raise RuntimeError(
                "Interleave CP gather received an unexpected local token count: "
                f"rank={self.cp_rank}, got={x.shape[0]}, expected={local_len}, "
                f"total={total_tokens}, cp_size={self.cp_size}."
            )

        max_rank_len = base_len + (1 if extra else 0)
        if max_rank_len == 0:
            return x.new_empty((0, *x.shape[1:]))

        if local_len < max_rank_len:
            padded_x = x.new_zeros((max_rank_len, *x.shape[1:]))
            padded_x[:local_len] = x
        else:
            padded_x = x.contiguous()

        with use_symmetric_memory(
            get_attention_cp_group(), disabled=not is_allocation_symmetric()
        ):
            gathered = x.new_empty((self.cp_size * max_rank_len, *x.shape[1:]))
        attn_cp_all_gather_into_tensor(gathered, padded_x.contiguous())

        if extra == 0:
            out_shape = gathered.shape
            return (
                gathered.view(self.cp_size, max_rank_len, *out_shape[1:])
                .transpose(0, 1)
                .reshape(total_tokens, *out_shape[1:])
            )

        flat_indices = torch.arange(total_tokens, device=x.device)
        gather_indices = (
            flat_indices % self.cp_size
        ) * max_rank_len + flat_indices // self.cp_size
        return gathered.index_select(0, gather_indices)

    def get_supported_attention_backend(self):
        return [CPAttentionBackendKind.DSA]

    def run_indexer(
        self, indexer, q_lora: Any, x: Any, positions: Any, forward_batch
    ) -> Any:
        """CP-v2 indexer dispatch: _get_q_k_bf16 (skips AllGather) + strategy gathers Key.

        Returns (query, full_key, weights_raw) for forward_cuda to continue with topk.
        """
        stream = torch.cuda.current_stream()
        query, local_key, weights_raw = indexer._get_q_k_bf16(
            q_lora,
            x,
            positions,
            enable_dual_stream=False,
            forward_batch=forward_batch,
        )
        # Strategy does the AllGather
        full_key = self.gather_kv_cache(local_key.contiguous(), forward_batch, stream)
        return query, full_key, weights_raw

    def run_attention(
        self,
        q: Any,
        forward_batch,
        device: Any,
        attn_fn,
        attention_backend: CPAttentionBackendKind = CPAttentionBackendKind.FLASH_ATTENTION,
        **kwargs,
    ) -> Any:
        pass

    def materialize_full_kv(
        self,
        forward_batch,
        attention_backend: CPAttentionBackendKind = CPAttentionBackendKind.DSA,
        **kwargs,
    ):
        """
        Interleave CP currently only supports the DSA attention backend, whose MLA
        path writes the KV cache itself, so the gathered latent is returned to the
        caller instead of being stored here.

        Expected kwargs: ``latent_cache``, ``k_nope``, ``k_pe``, ``kv_lora_rank``.
        """
        if attention_backend != CPAttentionBackendKind.DSA:
            raise NotImplementedError(
                "Interleave CP materialize_full_kv only supports the DSA backend."
            )
        latent_cache = kwargs["latent_cache"]
        k_nope = kwargs["k_nope"]
        k_pe = kwargs["k_pe"]
        kv_lora_rank = kwargs["kv_lora_rank"]
        # pack (k_nope, k_pe) -> latent_cache
        latent_cache[..., :kv_lora_rank] = k_nope.squeeze(1)
        latent_cache[..., kv_lora_rank:] = k_pe.squeeze(1)
        # AllGather across CP ranks
        full_latent = self.gather_kv_cache(
            latent_cache.contiguous(), forward_batch, torch.cuda.current_stream()
        )
        # unpack full latent -> (k_nope, k_pe)
        k_nope = full_latent[..., :kv_lora_rank].unsqueeze(1)
        k_pe = full_latent[..., kv_lora_rank:].unsqueeze(1)
        return k_nope, k_pe
