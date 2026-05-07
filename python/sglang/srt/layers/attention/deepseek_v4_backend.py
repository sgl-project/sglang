from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Optional

import torch
import torch.nn.functional as F

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.dsv4_sparse_attention import dsv4_sparse_attn
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner


@dataclass
class DeepseekV4XPUAttentionMetadata:
    page_table: torch.Tensor
    cache_seqlens_int32: torch.Tensor
    cu_seqlens_q: torch.Tensor
    cu_seqlens_k: torch.Tensor
    max_seq_len_q: int
    max_seq_len_k: int


def _gather_kv_from_page_table(
    kv_cache: torch.Tensor,
    page_table: torch.Tensor,
    batch_idx: int,
    kv_len: int,
    page_size: int,
) -> torch.Tensor:
    positions = torch.arange(kv_len, device=page_table.device)
    if page_size == 1:
        block_ids = page_table[batch_idx, :kv_len].to(torch.long)
        offsets = torch.zeros_like(block_ids)
    else:
        block_ids = page_table[batch_idx, positions // page_size].to(torch.long)
        offsets = positions % page_size
    return kv_cache[block_ids, offsets.to(torch.long)]


class DeepseekV4Backend(AttentionBackend):
    """Intel XPU fallback backend for DeepSeek V4 compressed attention.

    This backend keeps the DeepSeek V4 model-facing API from the compressed
    backend (`compress_ratio` and `attn_sink`) while using the torch reference
    sparse attention implementation available on XPU.
    """

    def __init__(self, model_runner: ModelRunner):
        super().__init__()
        self.device = model_runner.device
        self.page_size = model_runner.page_size
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.forward_metadata: Optional[DeepseekV4XPUAttentionMetadata] = None

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        seq_lens = forward_batch.seq_lens.to(torch.int32)
        seq_lens_cpu = forward_batch.seq_lens_cpu
        assert seq_lens_cpu is not None

        batch_size = forward_batch.batch_size
        device = seq_lens.device
        max_seq_len_k = seq_lens_cpu.max().item()
        cu_seqlens_k = F.pad(torch.cumsum(seq_lens, dim=0, dtype=torch.int32), (1, 0))
        page_table = forward_batch.req_to_token_pool.req_to_token[
            forward_batch.req_pool_indices, :max_seq_len_k
        ]

        if forward_batch.forward_mode.is_decode_or_idle():
            cu_seqlens_q = torch.arange(
                0, batch_size + 1, dtype=torch.int32, device=device
            )
            max_seq_len_q = 1
        elif forward_batch.forward_mode.is_extend_or_draft_extend_or_mixed():
            if (
                forward_batch.extend_prefix_lens_cpu is not None
                and any(forward_batch.extend_prefix_lens_cpu)
            ) or forward_batch.forward_mode == ForwardMode.DRAFT_EXTEND:
                cu_seqlens_q = F.pad(
                    torch.cumsum(
                        forward_batch.extend_seq_lens, dim=0, dtype=torch.int32
                    ),
                    (1, 0),
                )
                max_seq_len_q = max(forward_batch.extend_seq_lens_cpu)
            else:
                cu_seqlens_q = cu_seqlens_k
                max_seq_len_q = max_seq_len_k
        else:
            raise NotImplementedError(
                f"DeepseekV4Backend does not support {forward_batch.forward_mode=}"
            )

        if self.page_size > 1:
            strided_indices = torch.arange(
                0, page_table.shape[1], self.page_size, device=device
            )
            page_table = page_table[:, strided_indices] // self.page_size

        self.forward_metadata = DeepseekV4XPUAttentionMetadata(
            page_table=page_table,
            cache_seqlens_int32=seq_lens,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seq_len_q=max_seq_len_q,
            max_seq_len_k=max_seq_len_k,
        )

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info,
    ):
        raise NotImplementedError("DeepseekV4Backend on XPU does not support graphs")

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info,
        seq_lens_cpu: Optional[torch.Tensor],
    ):
        raise NotImplementedError("DeepseekV4Backend on XPU does not support graphs")

    def get_cuda_graph_seq_len_fill_value(self):
        return 1

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        *,
        compress_ratio: Literal[0, 4, 128] = 0,
        attn_sink: Optional[torch.Tensor] = None,
        topk_indices: Optional[torch.Tensor] = None,
        **_,
    ) -> torch.Tensor:
        if forward_batch.forward_mode.is_idle():
            return q.new_empty(q.shape[0], layer.tp_q_head_num, layer.v_head_dim)

        if attn_sink is None:
            raise ValueError("DeepSeek V4 sparse attention requires attention sinks")
        if k is not v:
            raise ValueError("DeepSeek V4 attention expects shared k and v states")

        self._save_kv_cache(k, layer, forward_batch, save_kv_cache)
        key_cache, value_cache = self._get_kv_cache(layer, forward_batch)
        return self._forward_dsv4_sparse_attn(
            q=q,
            key_cache=key_cache,
            value_cache=value_cache,
            topk_indices=topk_indices,
            sinks=attn_sink,
            compress_ratio=compress_ratio,
            layer=layer,
        )

    def _save_kv_cache(
        self,
        kv: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool,
    ):
        if not save_kv_cache:
            return

        kv = kv.contiguous().view(-1, layer.tp_k_head_num, layer.head_dim)
        forward_batch.token_to_kv_pool.set_kv_buffer(
            layer,
            forward_batch.out_cache_loc,
            kv,
            kv,
            layer.k_scale,
            layer.v_scale,
        )

    def _get_kv_cache(
        self, layer: RadixAttention, forward_batch: ForwardBatch
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key_cache, value_cache = forward_batch.token_to_kv_pool.get_kv_buffer(
            layer.layer_id
        )
        key_cache = key_cache.view(
            -1, self.page_size, layer.tp_k_head_num, layer.head_dim
        )
        value_cache = value_cache.view(
            -1, self.page_size, layer.tp_v_head_num, layer.v_head_dim
        )
        return key_cache, value_cache

    def _forward_dsv4_sparse_attn(
        self,
        q: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        topk_indices: Optional[torch.Tensor],
        sinks: torch.Tensor,
        compress_ratio: Literal[0, 4, 128],
        layer: RadixAttention,
    ) -> torch.Tensor:
        assert self.forward_metadata is not None
        metadata = self.forward_metadata
        q = q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim)
        output = q.new_empty((q.shape[0], layer.tp_q_head_num, layer.v_head_dim))

        for batch_idx in range(metadata.cache_seqlens_int32.numel()):
            q_start = int(metadata.cu_seqlens_q[batch_idx].item())
            q_end = int(metadata.cu_seqlens_q[batch_idx + 1].item())
            kv_len = int(metadata.cache_seqlens_int32[batch_idx].item())
            if q_end == q_start:
                continue

            key_states = _gather_kv_from_page_table(
                key_cache, metadata.page_table, batch_idx, kv_len, self.page_size
            ).unsqueeze(0)
            value_states = _gather_kv_from_page_table(
                value_cache, metadata.page_table, batch_idx, kv_len, self.page_size
            ).unsqueeze(0)
            topk_idxs = self._get_topk_indices_for_batch(
                topk_indices=topk_indices,
                batch_idx=batch_idx,
                q_start=q_start,
                q_end=q_end,
                kv_len=kv_len,
                compress_ratio=compress_ratio,
                device=q.device,
            )

            output[q_start:q_end] = dsv4_sparse_attn(
                q[q_start:q_end].unsqueeze(0),
                key_states,
                value_states,
                sinks,
                topk_idxs,
                layer.scaling,
            ).squeeze(0)

        return output

    @staticmethod
    def _get_topk_indices_for_batch(
        *,
        topk_indices: Optional[torch.Tensor],
        batch_idx: int,
        q_start: int,
        q_end: int,
        kv_len: int,
        compress_ratio: Literal[0, 4, 128],
        device: torch.device,
    ) -> torch.Tensor:
        if topk_indices is not None:
            if topk_indices.dim() == 3:
                return topk_indices[batch_idx : batch_idx + 1]
            return topk_indices[q_start:q_end].unsqueeze(0)

        if compress_ratio == 0:
            selected_indices = torch.arange(kv_len, device=device, dtype=torch.int32)
        else:
            selected_indices = torch.arange(
                compress_ratio - 1,
                kv_len,
                compress_ratio,
                device=device,
                dtype=torch.int32,
            )
            if selected_indices.numel() == 0:
                selected_indices = torch.full(
                    (1,), -1, device=device, dtype=torch.int32
                )

        return selected_indices.view(1, 1, -1).expand(1, q_end - q_start, -1)