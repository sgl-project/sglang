"""Ascend collectives and LSE merge for MLA decode context parallelism."""

from __future__ import annotations

import torch

from sglang.srt.layers.dcp.comm import (
    all_gather_q_for_mla_decode,
    cp_lse_ag_out_rs_mla_npu,
    dcp_a2a_lse_reduce,
)
from sglang.srt.runtime_context import get_parallel


def get_mla_dcp_backend(backend):
    """Find the full-attention backend behind TBO and hybrid dispatchers."""
    backend = getattr(backend, "primary", backend)
    return getattr(backend, "full_attn_backend", backend)


def gather_mla_dcp_q_npu(backend, forward_batch, q_nope_out, q_pe):
    """Gather the query heads required by rank-local MLA or DSA attention."""
    if backend.is_dcp_mla_decode_phase(forward_batch):
        q_nope_out, q_pe = all_gather_q_for_mla_decode(q_nope_out, q_pe)
        # The common gather returns transposed views; Ascend kernels consume
        # contiguous token-major queries.
        return q_nope_out.contiguous(), q_pe.contiguous()
    return q_nope_out, q_pe


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

    Device-specific arguments stay with the caller. DSA retains its Ascend
    attention-update merge; dense MLA retains the compact Triton A2A merge.
    """
    use_dcp = backend.is_dcp_mla_decode_phase(forward_batch)
    attn = module.attn_mqa_for_dcp_decode if use_dcp else module.attn_mqa
    if use_dcp and not backend.use_dsa:
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
        attn_output, lse = attn_output
        attn_output = attn_output.view(
            -1,
            module.num_local_heads * backend.dcp_size,
            module.kv_lora_rank,
        )
        if backend.use_dsa:
            attn_output = cp_lse_ag_out_rs_mla_npu(
                attn_output, lse, get_parallel().dcp_group
            )
        else:
            attn_output = dcp_a2a_lse_reduce(
                attn_output,
                lse,
                get_parallel().dcp_group,
                is_lse_base_on_e=True,
                comm_backend="a2a",
            )
    return attn_output.view(-1, module.num_local_heads, module.kv_lora_rank)


def mask_empty_mla_dcp_shards_npu(
    partial_output: torch.Tensor,
    partial_lse: torch.Tensor,
    local_seq_lens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Represent an empty local KV shard as the online-softmax identity.

    Lengths may be per request (decode) or per query (target verify). A verify
    query can have no visible local KV even when a later query in that request
    has one. FIA's empty-row output must not enter the cross-rank LSE merge.
    """
    num_rows = partial_output.shape[0]
    num_lengths = local_seq_lens.numel()
    if num_lengths == 0:
        return partial_output, partial_lse
    rows_per_length = num_rows // num_lengths
    valid = (local_seq_lens > 0).repeat_interleave(rows_per_length).view(-1, 1, 1)
    return (
        torch.where(valid, partial_output, torch.zeros_like(partial_output)),
        torch.where(
            valid,
            partial_lse,
            torch.full_like(partial_lse, float("-inf")),
        ),
    )
