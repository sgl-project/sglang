"""Ascend attention path backed by sparsity-driven KV offload."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch
import torch_npu

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

    if save_kv_cache:
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
        selected_kv_length = 2048
        num_kv_heads = layer.tp_k_head_num
        num_query_heads = layer.tp_q_head_num
        nope_head_dim = backend.kv_lora_rank
        rope_head_dim = backend.qk_rope_head_dim

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

        selected_kv = torch.zeros(
            (
                batch_size,
                selected_kv_length,
                num_kv_heads,
                nope_head_dim + rope_head_dim,
            ),
            dtype=k.dtype,
            device=backend.device,
        )
        sparse_kv_manager.prefetch(
            layer, forward_batch, topk_indices, selected_kv, stream
        )
        selected_k_nope, selected_k_rope = selected_kv.split(
            [nope_head_dim, rope_head_dim], dim=-1
        )

        topk_2d = topk_indices
        if topk_2d.dim() == 3:
            topk_2d = topk_2d[:, 0, :]
        elif topk_2d.dim() == 4:
            topk_2d = topk_2d[:, 0, 0, :]
        elif topk_2d.dim() != 2:
            raise RuntimeError(
                "SFA BSND compact path expects topk rank 2/3/4, " f"got {topk_2d.dim()}"
            )
        topk_2d = topk_2d[:, :selected_kv_length].contiguous()
        topk_length = topk_2d.shape[1]

        topk_valid = topk_2d >= 0
        if forward_batch.seq_lens is not None:
            valid_rows = (forward_batch.seq_lens[:batch_size] > 0).view(batch_size, 1)
            topk_valid = topk_valid & valid_rows

        actual_seq_lengths_kv = (
            topk_valid.sum(dim=1)
            .clamp(min=1, max=topk_length)
            .to(device=q_nope.device, dtype=torch.int32)
            .contiguous()
        )
        actual_seq_lengths_query = torch.ones(
            batch_size, dtype=torch.int32, device=q_nope.device
        ).contiguous()

        compact_indices = (
            torch.arange(topk_length, device=q_nope.device, dtype=torch.int32)
            .view(1, 1, 1, topk_length)
            .expand(batch_size, 1, num_kv_heads, topk_length)
            .clone()
        )
        compact_valid = topk_valid.view(batch_size, 1, 1, topk_length).expand(
            batch_size, 1, num_kv_heads, topk_length
        )
        sparse_indices = torch.where(
            compact_valid,
            compact_indices,
            torch.full_like(compact_indices, -1),
        ).contiguous()

        empty_rows = (topk_valid.sum(dim=1) == 0).view(batch_size, 1, 1)
        sparse_indices[:, :, :, 0] = torch.where(
            empty_rows.expand(batch_size, 1, num_kv_heads),
            torch.zeros(
                (batch_size, 1, num_kv_heads),
                dtype=torch.int32,
                device=q_nope.device,
            ),
            sparse_indices[:, :, :, 0],
        )

        q_nope_sfa = q_nope.view(
            batch_size, 1, padded_query_heads, nope_head_dim
        ).contiguous()
        q_rope_sfa = q_pe.view(
            batch_size, 1, padded_query_heads, rope_head_dim
        ).contiguous()
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
