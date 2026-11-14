"""Ascend collectives and LSE merge for MLA decode context parallelism."""

from __future__ import annotations

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.dcp.comm import cp_lse_ag_out_rs_mla_npu, dcp_a2a_lse_reduce
from sglang.srt.runtime_context import get_parallel


def get_mla_dcp_backend(backend):
    """Find the full-attention backend behind TBO and hybrid dispatchers."""
    return getattr(backend, "full_attn_backend", backend)


def forward_mla_attention_npu(
    module,
    backend,
    forward_batch,
    q_nope_out,
    k_nope,
    q_pe,
    k_pe,
    **attn_kwargs,
):
    """Run local attention, merge DCP partials, and restore this TP rank's heads.

    DSA folds its softmax-stat conversion into the compact Triton A2A pack.
    """
    use_dcp = backend.uses_dcp_partial_attention(forward_batch)
    attn = module.attn_mqa_for_dcp_decode if use_dcp else module.attn_mqa
    if use_dcp:
        attn_kwargs["return_softmax_lse"] = True
    attn_output = attn(
        q_nope_out,
        k_nope,
        k_nope,
        forward_batch,
        q_rope=q_pe,
        k_rope=k_pe,
        **attn_kwargs,
    )
    if use_dcp:
        attn_output, attn_stats = attn_output
        attn_output = attn_output.view(
            -1,
            module.num_local_heads * backend.dcp_size,
            module.kv_lora_rank,
        )
        group = get_parallel().dcp_group
        if backend.use_dsa:
            softmax_max, softmax_sum = attn_stats
            mode = forward_batch.forward_mode
            fuse_stats = (
                envs.SGLANG_NPU_DSA_DCP_FUSED_A2A.get()
                and (mode.is_decode() or mode.is_target_verify())
                and attn_output.dtype == torch.bfloat16
                and attn_output.is_contiguous()
                and attn_output.shape[-1] == 512
                and softmax_max.dtype == softmax_sum.dtype == torch.float32
            )
            if not fuse_stats:
                lse = softmax_max.float() + torch.log(softmax_sum.float())
                lse = lse.permute(1, 0, 2).reshape(lse.shape[1], -1)
                attn_output = cp_lse_ag_out_rs_mla_npu(attn_output, lse, group)
                return attn_output.view(-1, module.num_local_heads, module.kv_lora_rank)
            lse_or_max = softmax_max.permute(1, 0, 2).reshape(attn_output.shape[:2])
            softmax_sum = softmax_sum.permute(1, 0, 2).reshape(attn_output.shape[:2])
        else:
            lse_or_max, softmax_sum = attn_stats, None
        attn_output = dcp_a2a_lse_reduce(
            attn_output,
            lse_or_max,
            group,
            is_lse_base_on_e=True,
            comm_backend="a2a",
            softmax_sum=softmax_sum,
        )
    return attn_output.view(-1, module.num_local_heads, module.kv_lora_rank)
