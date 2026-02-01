from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)

BLOCK_SIZE = 64


@dataclass
class HpcAttentionMetadata:
    cache_seqlens_int32: torch.Tensor = None  # [bs] int32
    max_seq_len_q: int = 1
    cu_seqlens_q: torch.Tensor = None  # [bs+1] int32
    page_table: torch.Tensor = None  # [bs, max_blocks] int32


class HpcAttentionBackend(AttentionBackend):
    def __init__(self, model_runner: ModelRunner):
        super().__init__()

        self.page_size = model_runner.server_args.page_size
        assert self.page_size == BLOCK_SIZE, (
            f"HPC attention backend requires page_size={BLOCK_SIZE}, "
            f"got {self.page_size}"
        )

        self.device = model_runner.device
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.forward_metadata: HpcAttentionMetadata = None

    def _build_page_table(self, forward_batch: ForwardBatch, max_seq_len_k: int):
        """Convert SGLang token-level req_to_token into HPC block_ids.

        SGLang stores flat slot indices in req_to_token. HPC needs a page table
        of shape [bs, max_blocks] containing physical page IDs. Since pages are
        contiguous blocks of page_size slots, slot_id // page_size = page_id.
        """
        bs = forward_batch.batch_size
        device = forward_batch.seq_lens.device

        max_num_blocks = (max_seq_len_k + BLOCK_SIZE - 1) // BLOCK_SIZE

        # Sample one token position per block: 0, 64, 128, ...
        sample_positions = torch.arange(
            0, max_num_blocks * BLOCK_SIZE, BLOCK_SIZE, device=device
        )

        # Get the flat slot indices at those positions
        raw = self.req_to_token[forward_batch.req_pool_indices][
            :, sample_positions
        ]  # [bs, max_num_blocks]

        # Convert flat slot indices to page IDs
        block_ids = (raw // BLOCK_SIZE).to(torch.int32)
        return block_ids

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        bs = forward_batch.batch_size
        device = forward_batch.seq_lens.device

        if forward_batch.forward_mode.is_decode_or_idle():
            cache_seqlens = forward_batch.seq_lens.to(torch.int32)
            max_seq_len_k = int(forward_batch.seq_lens.max().item())

            cu_seqlens_q = torch.arange(
                0, bs + 1, dtype=torch.int32, device=device
            )
            max_seq_len_q = 1

            page_table = self._build_page_table(forward_batch, max_seq_len_k)

            self.forward_metadata = HpcAttentionMetadata(
                cache_seqlens_int32=cache_seqlens,
                max_seq_len_q=max_seq_len_q,
                cu_seqlens_q=cu_seqlens_q,
                page_table=page_table,
            )

        elif forward_batch.forward_mode.is_extend():
            cache_seqlens = forward_batch.seq_lens.to(torch.int32)
            max_seq_len_k = int(forward_batch.seq_lens.max().item())

            # Build cu_seqlens_q from extend_seq_lens
            cu_seqlens_q = torch.zeros(
                bs + 1, dtype=torch.int32, device=device
            )
            cu_seqlens_q[1:] = torch.cumsum(
                forward_batch.extend_seq_lens, dim=0
            ).to(torch.int32)
            max_seq_len_q = int(forward_batch.extend_seq_lens.max().item())

            page_table = self._build_page_table(forward_batch, max_seq_len_k)

            self.forward_metadata = HpcAttentionMetadata(
                cache_seqlens_int32=cache_seqlens,
                max_seq_len_q=max_seq_len_q,
                cu_seqlens_q=cu_seqlens_q,
                page_table=page_table,
            )

        else:
            raise NotImplementedError(
                f"HPC attention backend does not support forward mode: "
                f"{forward_batch.forward_mode}"
            )

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
    ):
        import hpc

        if save_kv_cache:
            forward_batch.token_to_kv_pool.set_kv_buffer(
                layer, forward_batch.out_cache_loc, k, v
            )

        metadata = self.forward_metadata
        tp_q_head_num = layer.tp_q_head_num
        head_dim = layer.head_dim
        v_head_dim = layer.v_head_dim

        # Reshape Q to [total_tokens, q_heads, head_dim]
        q_3d = q.contiguous().view(-1, tp_q_head_num, head_dim)

        # Get KV cache and reshape to [num_pages, page_size, kv_heads, dim]
        key_cache = forward_batch.token_to_kv_pool.get_key_buffer(
            layer.layer_id
        )
        value_cache = forward_batch.token_to_kv_pool.get_value_buffer(
            layer.layer_id
        )
        key_cache = key_cache.view(-1, BLOCK_SIZE, key_cache.shape[-2], head_dim)
        value_cache = value_cache.view(
            -1, BLOCK_SIZE, value_cache.shape[-2], v_head_dim
        )

        o = hpc.attention_with_kvcache_prefill_bf16(
            q=q_3d,
            kcache=key_cache,
            vcache=value_cache,
            cu_seqlens_q=metadata.cu_seqlens_q,
            block_ids=metadata.page_table,
            seqlens_kvcache=metadata.cache_seqlens_int32,
            max_seqlens_q=metadata.max_seq_len_q,
        )

        return o.view(-1, tp_q_head_num * v_head_dim)

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
    ):
        import hpc

        if save_kv_cache:
            forward_batch.token_to_kv_pool.set_kv_buffer(
                layer, forward_batch.out_cache_loc, k, v
            )

        metadata = self.forward_metadata
        tp_q_head_num = layer.tp_q_head_num
        head_dim = layer.head_dim
        v_head_dim = layer.v_head_dim

        # Reshape Q to [bs, q_heads, head_dim]
        q_3d = q.contiguous().view(-1, tp_q_head_num, head_dim)

        # Get KV cache and reshape to [num_pages, page_size, kv_heads, dim]
        key_cache = forward_batch.token_to_kv_pool.get_key_buffer(
            layer.layer_id
        )
        value_cache = forward_batch.token_to_kv_pool.get_value_buffer(
            layer.layer_id
        )
        key_cache = key_cache.view(-1, BLOCK_SIZE, key_cache.shape[-2], head_dim)
        value_cache = value_cache.view(
            -1, BLOCK_SIZE, value_cache.shape[-2], v_head_dim
        )

        o = hpc.attention_decode_bf16(
            q=q_3d,
            kcache=key_cache,
            vcache=value_cache,
            block_ids=metadata.page_table,
            num_seq_kvcache=metadata.cache_seqlens_int32,
            new_kv_included=True,
            splitk=True,
        )

        return o.view(-1, tp_q_head_num * v_head_dim)
