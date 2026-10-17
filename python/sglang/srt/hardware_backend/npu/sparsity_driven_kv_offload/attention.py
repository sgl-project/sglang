"""Ascend attention path backed by sparsity-driven KV offload."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch
import torch_npu

from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.manager import (
    normalize_batch_topk_indices,
)
from sglang.srt.layers.attention.dsa.utils import is_dsa_enable_prefill_cp

if TYPE_CHECKING:
    from sglang.srt.hardware_backend.npu.attention.ascend_backend import (
        AscendAttnBackend,
    )
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


def _get_sparse_kv_manager(backend: AscendAttnBackend):
    if backend.sparse_kv_manager is None:
        raise RuntimeError(
            "Sparsity-driven KV offload is disabled or was not initialized."
        )
    return backend.sparse_kv_manager


def _expand_dsa_sparse_indices(topk_indices: torch.Tensor) -> torch.Tensor:
    """Expand [T, K] to [T, 1, K] for NPU sparse attention."""
    if topk_indices.dim() == 2:
        return topk_indices.unsqueeze(-2)
    return topk_indices


def forward_sparsity_driven_kv_offload(
    backend: AscendAttnBackend,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    layer: RadixAttention,
    forward_batch: ForwardBatch,
    save_kv_cache: bool = True,
    q_rope: Optional[torch.Tensor] = None,
    k_rope: Optional[torch.Tensor] = None,
    topk_indices: Optional[torch.Tensor] = None,
):
    """Run sparse attention using host-offloaded compact MLA KV."""
    del v
    if q_rope is None or k_rope is None or topk_indices is None:
        raise ValueError(
            "Sparsity-driven KV offload requires q_rope, k_rope, and topk_indices."
        )

    is_prefill = forward_batch.forward_mode.is_extend_without_speculative()

    q_nope, q_pe = q, q_rope
    k_nope = k.view(-1, layer.tp_k_head_num, backend.kv_lora_rank).contiguous()
    k_pe = k_rope.view(-1, layer.tp_k_head_num, backend.qk_rope_head_dim).contiguous()
    sparse_kv_manager = _get_sparse_kv_manager(backend)
    stream = torch.npu.current_stream(backend.device)

    decode_offload_done = None
    if save_kv_cache:
        if forward_batch.forward_mode.is_decode():
            decode_offload_done = sparse_kv_manager.offload_v2_decode_async(
                k_nope, k_pe, layer, forward_batch, stream
            )
        else:
            sparse_kv_manager.offload_v2(k_nope, k_pe, layer, forward_batch, stream)

    if is_prefill:
        if backend.forward_metadata.actual_seq_lengths_q is not None:
            actual_seq_qlen = backend.forward_metadata.actual_seq_lengths_q
        else:
            actual_seq_qlen = torch.cumsum(forward_batch.extend_seq_lens, dim=0)
    elif backend.forward_metadata.actual_seq_lengths_q is None:
        if (
            forward_batch.forward_mode.is_draft_extend_v2()
            or forward_batch.forward_mode.is_target_verify()
        ):
            actual_seq_qlen = (
                torch.arange(
                    backend.speculative_num_draft_tokens,
                    backend.speculative_num_draft_tokens + q.shape[0],
                    backend.speculative_num_draft_tokens,
                    dtype=torch.int32,
                )
                .to(q.device)
                .to(torch.int32)
            )
        else:
            actual_seq_qlen = (
                torch.arange(1, q.shape[0] + 1).to(q.device).to(torch.int32)
            )
    else:
        actual_seq_qlen = backend.forward_metadata.actual_seq_lengths_q

    if backend.forward_metadata.actual_seq_lengths_kv is not None:
        actual_seq_lengths_kv = backend.forward_metadata.actual_seq_lengths_kv
    elif backend.forward_metadata.seq_lens_cpu_int is not None:
        actual_seq_lengths_kv = backend.forward_metadata.seq_lens_cpu_int
    else:
        actual_seq_lengths_kv = backend.forward_metadata.seq_lens

    if (
        is_prefill
        and is_dsa_enable_prefill_cp()
        and forward_batch.attn_cp_metadata is not None
    ):
        attn_out = backend.do_cp_balance_attn(
            q_nope,
            k_nope,
            q_pe,
            k_pe,
            topk_indices,
            layer,
            actual_seq_qlen,
            actual_seq_lengths_kv,
        )
    elif forward_batch.forward_mode.is_decode():
        batch_size = forward_batch.batch_size
        num_kv_heads = layer.tp_k_head_num
        num_query_heads = layer.tp_q_head_num
        nope_head_dim = backend.kv_lora_rank
        rope_head_dim = backend.qk_rope_head_dim

        topk_2d_input = normalize_batch_topk_indices(topk_indices)
        effective_topk_length = topk_2d_input.shape[1]
        selected_kv_length = sparse_kv_manager.sparse_context_len
        if effective_topk_length <= 0:
            raise RuntimeError("SFA BSND compact path expects a positive top-k length.")
        if effective_topk_length > selected_kv_length:
            raise RuntimeError(
                "DSA top-k length exceeds sparse attention window: "
                f"topk_len={effective_topk_length}, "
                f"sparse_context_len={selected_kv_length}."
            )
        if effective_topk_length == selected_kv_length:
            topk_2d = topk_2d_input.contiguous()
        else:
            topk_2d = torch.full(
                (batch_size, selected_kv_length),
                -1,
                dtype=topk_2d_input.dtype,
                device=topk_2d_input.device,
            )
            topk_2d[:, :effective_topk_length] = topk_2d_input
            topk_2d = topk_2d.contiguous()

        assert num_kv_heads == 1, (
            "FIA_v2 MLA selected KV path expects KV_N == 1, "
            f"got num_kv_heads={num_kv_heads}"
        )

        padded_query_heads = q_nope.numel() // (batch_size * nope_head_dim)
        assert padded_query_heads >= num_query_heads, (
            "query head count mismatch: "
            f"padded_query_heads={padded_query_heads}, "
            f"num_query_heads={num_query_heads}"
        )

        # Materialize the current top-k for attention. The manager chooses
        # whether to retain LRU slots or replace the cache with this window.
        selected_kv_buffer = torch.zeros(
            (
                batch_size,
                selected_kv_length,
                num_kv_heads,
                nope_head_dim + rope_head_dim,
            ),
            dtype=k.dtype,
            device=backend.device,
        )
        refill_plan, topk_valid, valid_topk_counts = (
            sparse_kv_manager.materialize_selected_kv(
                layer,
                forward_batch,
                topk_2d,
                selected_kv_buffer,
                stream,
                host_kv_ready_event=decode_offload_done,
            )
        )

        # Both copies are complete here. Metadata update overlaps preparation
        # below without consuming selected_kv_buffer.

        actual_seq_lengths_kv = (
            valid_topk_counts.clamp(min=1, max=selected_kv_length)
            .to(device=q_nope.device, dtype=torch.int32)
            .contiguous()
        )
        actual_seq_lengths_query = sparse_kv_manager._decode_query_seq_lengths[
            :batch_size
        ]

        compact_valid = topk_valid.view(batch_size, 1, 1, selected_kv_length)
        sparse_indices = torch.where(
            compact_valid,
            sparse_kv_manager._compact_sparse_indices,
            sparse_kv_manager._invalid_sparse_indices,
        ).contiguous()

        empty_rows = (valid_topk_counts == 0).view(batch_size, 1, 1)
        sparse_indices[:, :, :, 0] = torch.where(
            empty_rows,
            sparse_kv_manager._zero_sparse_index[:batch_size],
            sparse_indices[:, :, :, 0],
        )

        q_nope_sfa = q_nope.view(
            batch_size, 1, padded_query_heads, nope_head_dim
        ).contiguous()
        q_rope_sfa = q_pe.view(
            batch_size, 1, padded_query_heads, rope_head_dim
        ).contiguous()

        selected_k_nope, selected_k_rope = selected_kv_buffer.split(
            [nope_head_dim, rope_head_dim], dim=-1
        )
        k_nope_sfa = selected_k_nope.contiguous()
        k_rope_sfa = selected_k_rope.contiguous()

        assert q_nope_sfa.shape == (
            batch_size,
            1,
            padded_query_heads,
            nope_head_dim,
        )
        assert q_rope_sfa.shape == (
            batch_size,
            1,
            padded_query_heads,
            rope_head_dim,
        )
        assert k_nope_sfa.shape == (
            batch_size,
            selected_kv_length,
            num_kv_heads,
            nope_head_dim,
        )
        assert k_rope_sfa.shape == (
            batch_size,
            selected_kv_length,
            num_kv_heads,
            rope_head_dim,
        )

        # The manager orders refill and metadata before attention on this stream.
        sparse_kv_manager.refill_selected_kv(
            layer=layer,
            selected_kv_buffer=selected_kv_buffer,
            plan=refill_plan,
            stream=stream,
        )
        ret = torch_npu.npu_sparse_flash_attention(
            q_nope_sfa,
            k_nope_sfa,
            k_nope_sfa,
            sparse_indices,
            layer.scaling,
            actual_seq_lengths_query=actual_seq_lengths_query,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
            query_rope=q_rope_sfa,
            key_rope=k_rope_sfa,
            sparse_block_size=1,
            layout_query="BSND",
            layout_kv="BSND",
            sparse_mode=0,
            attention_mode=2,
            return_softmax_lse=False,
        )

        attn_out = ret[0] if isinstance(ret, tuple) else ret
        attn_out = attn_out[:, :, :num_query_heads, :].reshape(
            batch_size, num_query_heads * nope_head_dim
        )
    else:
        if is_prefill:
            k_nope_sfa, k_pe_sfa = sparse_kv_manager.get_forward_kv(
                layer, forward_batch, stream
            )
            forward_actual_seq_lengths_kv = torch.cumsum(forward_batch.seq_lens, dim=0)
        else:
            k_nope_sfa, k_pe_sfa = k_nope, k_pe
            forward_actual_seq_lengths_kv = actual_seq_lengths_kv

        topk_indices = _expand_dsa_sparse_indices(topk_indices)
        attn_out, _, _ = torch_npu.npu_sparse_flash_attention(
            query=q_nope,
            key=k_nope_sfa,
            value=k_nope_sfa,
            query_rope=q_pe,
            key_rope=k_pe_sfa,
            sparse_indices=topk_indices,
            scale_value=layer.scaling,
            actual_seq_lengths_query=actual_seq_qlen.to(
                device=q_nope.device, dtype=torch.int32
            ),
            actual_seq_lengths_kv=forward_actual_seq_lengths_kv.to(
                device=q_nope.device, dtype=torch.int32
            ),
            sparse_block_size=1,
            layout_query="TND",
            layout_kv="TND",
            sparse_mode=3,
            attention_mode=2,
            return_softmax_lse=False,
        )

    return attn_out
