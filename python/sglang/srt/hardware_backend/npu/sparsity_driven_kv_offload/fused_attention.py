"""Target decode/verify always consumes native LIM plans with CopySFA."""

from __future__ import annotations

from typing import Optional

import torch
import torch_npu

from sglang.srt.layers.attention.dsa.utils import is_dsa_enable_prefill_cp
from sglang.srt.model_executor.forward_mode import ForwardMode


def forward_fused_sparse_kv_offload(
    backend,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    layer,
    forward_batch,
    save_kv_cache: bool = True,
    q_rope: Optional[torch.Tensor] = None,
    k_rope: Optional[torch.Tensor] = None,
    topk_indices: Optional[torch.Tensor] = None,
):
    """Publish current KV before fused attention."""
    del v
    manager = backend.sparse_kv_manager
    if q.shape[0] == 0:
        return q.new_empty((0, layer.tp_q_head_num * backend.kv_lora_rank))
    is_prefill = forward_batch.forward_mode.is_extend_without_speculative()
    plan = None if is_prefill else backend.forward_metadata.sparse_kv_lim_plan
    k_nope = k.view(-1, layer.tp_k_head_num, backend.kv_lora_rank).contiguous()
    k_pe = k_rope.view(-1, layer.tp_k_head_num, backend.qk_rope_head_dim).contiguous()
    current_inputs = manager.offload_v2(k_nope, k_pe, layer, forward_batch)
    if not is_prefill:
        rows = forward_batch.batch_size * plan.query_width
        result = manager.copy_sfa_attention(
            layer,
            q[:rows].view(rows, -1, backend.kv_lora_rank),
            q_rope[:rows].view(rows, -1, backend.qk_rope_head_dim),
            plan,
            current_inputs,
        )
        if q.shape[0] > rows:
            return torch.nn.functional.pad(result, (0, 0, 0, q.shape[0] - rows))
        return result

    actual_q = backend.forward_metadata.actual_seq_lengths_q
    if actual_q is None:
        actual_q = torch.cumsum(forward_batch.extend_seq_lens, dim=0)
    actual_kv = backend.forward_metadata.actual_seq_lengths_kv
    if actual_kv is None:
        actual_kv = forward_batch.seq_lens
    if is_dsa_enable_prefill_cp() and forward_batch.attn_cp_metadata is not None:
        return backend.do_cp_balance_attn(
            q, k_nope, q_rope, k_pe, topk_indices, layer, actual_q, actual_kv
        )
    gather_plan = None
    if not backend.graph_mode and forward_batch.forward_mode == ForwardMode.EXTEND:
        gather_plan = backend.forward_metadata.sparse_kv_fused_prefill_plan
        if gather_plan is None:
            gather_plan = manager.prepare_forward_kv_plan(
                forward_batch, current=current_inputs
            )
            backend.forward_metadata.sparse_kv_fused_prefill_plan = gather_plan
    k_nope, k_pe = manager.get_forward_kv(
        layer, forward_batch, plan=gather_plan, current=current_inputs
    )
    if topk_indices.ndim == 2:
        topk_indices = topk_indices.unsqueeze(-2)
    result, _, _ = torch_npu.npu_sparse_flash_attention(
        query=q,
        key=k_nope,
        value=k_nope,
        query_rope=q_rope,
        key_rope=k_pe,
        sparse_indices=topk_indices,
        scale_value=layer.scaling,
        actual_seq_lengths_query=actual_q.to(device=q.device, dtype=torch.int32),
        actual_seq_lengths_kv=torch.cumsum(forward_batch.seq_lens, dim=0).to(
            device=q.device, dtype=torch.int32
        ),
        sparse_block_size=1,
        layout_query="TND",
        layout_kv="TND",
        sparse_mode=3,
        attention_mode=2,
        return_softmax_lse=False,
    )
    return result
