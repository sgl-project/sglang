"""Ascend collectives and LSE merge for MLA decode context parallelism."""

from __future__ import annotations

from sglang.srt.layers.dcp.comm import (
    cp_lse_ag_out_rs_mla_npu,
    dcp_a2a_lse_reduce,
)
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

    Device-specific arguments stay with the caller. DSA retains its Ascend
    attention-update merge; dense MLA retains the compact Triton A2A merge.
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
        # An idle DCP group has no partials to exchange.
        elif attn_output.shape[0] > 0:
            attn_output = dcp_a2a_lse_reduce(
                attn_output,
                lse,
                get_parallel().dcp_group,
                is_lse_base_on_e=True,
                comm_backend="a2a",
            )
    return attn_output.view(-1, module.num_local_heads, module.kv_lora_rank)
