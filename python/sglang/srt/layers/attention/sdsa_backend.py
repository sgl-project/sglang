# Copyright 2026 SGLang Team
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
"""SDSA sparse attention for Nemotron Shortcut, on FlashInfer's SDSA kernels.

An indexer selects each query's Top-K keys (``index_topk``, by the plain dot
product of its one-head index query with the cached index keys), and an MQA
core attends to them:

1. ``flashinfer.sdsa.indexer_scores``: FP32 scores against ``[pages, 64, d]``
   index keys, one query row per group.
2. ``flashinfer.top_k_page_table_transform``: top-k positions mapped through
   ``req_to_token`` to KV slots (-1 padded).
3. ``trtllm_batch_decode_with_kv_cache(sparse_top_k=...)``: token-sparse MQA,
   one query row per request over one-token KV pages.

With at most ``index_topk`` causal keys every key is kept, so this is exact for
short contexts too. Decode always takes the sparse path (one graph-safe
program); extend runs the dense TRT-LLM path unless some row sees more keys.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import msgspec
import torch

from sglang.srt.layers.attention.trtllm_mha_backend import TRTLLMHAAttnBackend
from sglang.srt.mem_cache.memory_pool import KVWriteLoc, window_write_locs
from sglang.srt.mem_cache.sdsa_kv_pool import SDSA_INDEX_PAGE, SdsaTokenToKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.runtime_context import get_disagg, get_memory, get_parallel

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner

# Bound on the FP32 score matrix of one sparse-prefill chunk; arbitrary.
_SCORE_CHUNK_BYTES = 1 << 30


class SdsaRows(msgspec.Struct, frozen=True):
    """One query row per token: its request (row of the batch's page tables)
    and its number of causal keys."""

    requests: torch.Tensor  # int32 [rows]
    context_lens: torch.Tensor  # int32 [rows]


def create_sdsa_backend(runner: ModelRunner, full_attn_backend):
    """Full-attention backend for a Nemotron Shortcut hybrid: SDSA when the model has
    an index_topk, else the configured one."""
    from sglang.srt.configs.nemotron_shortcut import sdsa_index_head_dim

    if sdsa_index_head_dim(runner.model_config.hf_config) is None:
        return full_attn_backend
    if not isinstance(full_attn_backend, TRTLLMHAAttnBackend):
        raise ValueError(
            "Nemotron Shortcut SDSA attention runs on FlashInfer's TRT-LLM kernels; "
            f"use --attention-backend trtllm_mha (got {type(full_attn_backend).__name__})."
        )
    # The index-key cache is not part of PD transfer or HiCache offload.
    if get_disagg().disaggregation_mode != "null":
        raise NotImplementedError("SDSA does not support PD disaggregation yet.")
    if get_memory().enable_hierarchical_cache:
        raise NotImplementedError("SDSA does not support --enable-hierarchical-cache.")
    return SdsaAttnBackend(runner)


class SdsaAttnBackend(TRTLLMHAAttnBackend):
    """TRT-LLM MHA backend whose layers attend to their SDSA top-k keys."""

    def __init__(self, model_runner: ModelRunner):
        super().__init__(model_runner)
        model_config = model_runner.model_config
        # The sparse path reads the cache as unscaled model-dtype pages.
        if self.data_type != model_config.dtype:
            raise NotImplementedError(
                f"SDSA needs a {model_config.dtype} KV cache, got {self.data_type}."
            )
        # Top-k slots index one-token pages of a single KV head.
        num_kv_heads = model_config.get_num_kv_heads(get_parallel().attn_tp_size)
        if num_kv_heads != 1:
            raise NotImplementedError(
                f"SDSA needs one KV head per rank, got {num_kv_heads}."
            )
        self.index_topk = model_config.hf_text_config.index_topk
        if self.index_topk % 4 != 0:
            raise ValueError(
                f"SDSA index_topk must be a multiple of 4, got {self.index_topk}"
            )
        pool = model_runner.token_to_kv_pool
        if not isinstance(pool, SdsaTokenToKVPool):
            raise TypeError(
                f"SDSA needs an SdsaTokenToKVPool, got {type(pool).__name__}"
            )
        self.index_pool = pool
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        # Score columns covered by the 64-key page table of a req_to_token row.
        self.score_width = (
            -(-self.req_to_token.shape[1] // SDSA_INDEX_PAGE) * SDSA_INDEX_PAGE
        )
        self.extend_rows: Optional[SdsaRows] = None
        self.extend_is_sparse = False

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------
    def init_forward_metadata(self, forward_batch: ForwardBatch):
        super().init_forward_metadata(forward_batch)
        mode = forward_batch.forward_mode
        self.extend_rows = None
        self.extend_is_sparse = False
        if not mode.is_extend() or mode.is_target_verify():
            return
        max_len = max(forward_batch.seq_lens_cpu.tolist(), default=0)
        self.extend_is_sparse = max_len > self.index_topk
        if self.extend_is_sparse:
            self.extend_rows = self._extend_rows(forward_batch)

    @staticmethod
    def _extend_rows(forward_batch: ForwardBatch) -> SdsaRows:
        batch = torch.arange(
            forward_batch.batch_size,
            dtype=torch.int32,
            device=forward_batch.seq_lens.device,
        )
        requests = torch.repeat_interleave(batch, forward_batch.extend_seq_lens)
        # Positions are token indices (NoPE), so position + 1 keys are causal.
        context_lens = (forward_batch.positions + 1).to(torch.int32)
        return SdsaRows(requests=requests, context_lens=context_lens)

    @staticmethod
    def _decode_rows(forward_batch: ForwardBatch) -> SdsaRows:
        seq_lens = forward_batch.seq_lens
        return SdsaRows(
            requests=torch.arange(
                seq_lens.shape[0], dtype=torch.int32, device=seq_lens.device
            ),
            context_lens=seq_lens.to(torch.int32),
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        *,
        sdsa_index_q: torch.Tensor,
        sdsa_index_k: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        if save_kv_cache:
            self._store(layer, forward_batch, k, v, sdsa_index_k)
        return self._sparse_attention(
            q, layer, forward_batch, sdsa_index_q, self._decode_rows(forward_batch)
        )

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        *,
        sdsa_index_q: torch.Tensor,
        sdsa_index_k: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        if forward_batch.forward_mode.is_target_verify():
            raise NotImplementedError("SDSA does not support speculative verify yet.")
        if not self.extend_is_sparse:
            # Every row sees at most index_topk keys: dense attention is exact.
            if save_kv_cache:
                self.index_pool.set_index_k_buffer(
                    layer.layer_id, forward_batch.out_cache_loc, sdsa_index_k
                )
            return super().forward_extend(
                q, k, v, layer, forward_batch, save_kv_cache, **kwargs
            )
        if save_kv_cache:
            self._store(layer, forward_batch, k, v, sdsa_index_k)
        return self._sparse_attention(
            q, layer, forward_batch, sdsa_index_q, self.extend_rows
        )

    def _store(self, layer, forward_batch, k, v, index_k) -> None:
        # The current tokens are causal keys of their own rows: write them first.
        self.kv_layers.layer(layer.layer_id).store(
            layer,
            KVWriteLoc.for_batch(
                forward_batch,
                window_locs=window_write_locs(
                    self.runner_ids, self.forward_metadata.swa_out_cache_loc
                ),
            ),
            k,
            v,
            *self._kv_write_scales(layer),
        )
        self.index_pool.set_index_k_buffer(
            layer.layer_id, forward_batch.out_cache_loc, index_k
        )

    def _sparse_attention(
        self,
        q: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        index_q: torch.Tensor,
        rows: SdsaRows,
    ) -> torch.Tensor:
        from flashinfer import top_k_page_table_transform
        from flashinfer.decode import trtllm_batch_decode_with_kv_cache
        from flashinfer.sdsa import indexer_scores

        num_rows = q.shape[0]
        num_heads, head_dim = layer.tp_q_head_num, layer.head_dim
        q = q.reshape(num_rows, num_heads, head_dim)
        k_cache, v_cache = self.kv_layers.read(layer.layer_id).kv()
        # One-token HND pages of the single KV head.
        kv_cache = (
            k_cache.view(-1, 1, 1, head_dim),
            v_cache.view(-1, 1, 1, head_dim),
        )
        index_keys = self.index_pool.get_index_k_buffer(layer.layer_id).view(
            -1, SDSA_INDEX_PAGE, index_q.shape[-1]
        )
        token_table = self.req_to_token[forward_batch.req_pool_indices]
        page_table = token_table[:, ::SDSA_INDEX_PAGE] // SDSA_INDEX_PAGE
        topk = self.index_topk
        bmm1_scale, bmm2_scale = self._get_bmm_scales(layer, 1.0)

        out = q.new_empty(num_rows, num_heads * head_dim)
        chunk = max(1, min(num_rows, _SCORE_CHUNK_BYTES // (4 * self.score_width)))
        for start in range(0, num_rows, chunk):
            end = min(num_rows, start + chunk)
            requests = rows.requests[start:end]
            context_lens = rows.context_lens[start:end]
            scores = torch.empty(
                (end - start, self.score_width), dtype=torch.float32, device=q.device
            )
            indexer_scores(
                index_q[start:end].contiguous(),
                index_keys,
                page_table,
                context_lens,
                requests=requests,
                scores=scores,
            )
            slots = top_k_page_table_transform(
                scores,
                token_table,
                context_lens,
                topk,
                row_to_batch=requests,
                dsa_graph_safe=True,
            )
            o = trtllm_batch_decode_with_kv_cache(
                query=q[start:end],
                kv_cache=kv_cache,
                workspace_buffer=self.workspace_buffer,
                block_tables=torch.stack([slots, slots], dim=1),
                seq_lens=context_lens.clamp(max=topk),
                max_seq_len=topk,
                bmm1_scale=bmm1_scale,
                bmm2_scale=bmm2_scale,
                out_dtype=q.dtype,
                kv_layout="HND",
                backend="trtllm-gen",
                uses_shared_paged_kv_idx=False,
                sparse_top_k=topk,
            )
            out[start:end] = o.reshape(end - start, -1)
        return out
