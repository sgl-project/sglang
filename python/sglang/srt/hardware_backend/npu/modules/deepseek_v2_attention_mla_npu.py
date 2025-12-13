from typing import TYPE_CHECKING

import torch
import torch_npu

from sglang.srt.hardware_backend.npu.attention.mla_preprocess import (
    NPUFusedMLAPreprocess,
    is_fia_nz,
    is_mla_preprocess_enabled,
)
from sglang.srt.layers.attention.nsa.utils import (
    cp_split_and_rebuild_position,
    enable_prefill_cp,
)
from sglang.srt.layers.communicator import get_attn_tp_context

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.models.deepseek_v2 import DeepseekV2AttentionMLA
    from sglang.srt.utils import BumpAllocator


# region MHA
def forward_mha_prepare_npu(
    m: "DeepseekV2AttentionMLA",
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    forward_batch: "ForwardBatch",
    zero_allocator: "BumpAllocator",
):
    if m.q_lora_rank is not None:
        q, latent_cache = (
            get_attn_tp_context()
            .fetch_qkv_latent()
            .split(
                [m.q_lora_rank, m.kv_lora_rank + m.qk_rope_head_dim],
                dim=-1,
            )
        )

        # NSA Indexer: cache quantized keys, auto-skip topk for sequences <= nsa_index_topk

        if m.use_nsa:
            q_lora = m.q_a_layernorm(q)
            q = m.q_b_proj(q_lora)[0].view(-1, m.num_local_heads, m.qk_head_dim)
            _ = m.indexer(
                x=hidden_states,
                q_lora=q_lora,
                positions=positions,
                forward_batch=forward_batch,
                layer_id=m.layer_id,
                return_indices=False,
            )

        else:
            q = m.q_a_layernorm(q)
            q = m.q_b_proj(q)[0].view(-1, m.num_local_heads, m.qk_head_dim)

    else:
        q = m.q_proj(hidden_states)[0].view(-1, m.num_local_heads, m.qk_head_dim)
        latent_cache = m.kv_a_proj_with_mqa(hidden_states)[0]

    _, q_pe = q.split([m.qk_nope_head_dim, m.qk_rope_head_dim], dim=-1)
    kv_a, _ = latent_cache.split([m.kv_lora_rank, m.qk_rope_head_dim], dim=-1)
    latent_cache = latent_cache.unsqueeze(1)

    B, S = q.shape[0], 1
    cos, sin = m.rotary_emb.get_cos_sin_cache(
        positions, hidden_states.dtype, offsets=None
    )
    q_pe = torch_npu.npu_interleave_rope(
        q_pe.reshape(B, -1, S, m.qk_rope_head_dim),
        cos,
        sin,
    )
    q_pe = q_pe.reshape(B, -1, m.qk_rope_head_dim)

    ckv_cache, k_rope_cache = forward_batch.token_to_kv_pool.get_kv_buffer(m.layer_id)
    _, _, k_pe, kv_a = torch_npu.npu_kv_rmsnorm_rope_cache(
        latent_cache.view(-1, 1, 1, m.kv_lora_rank + m.qk_rope_head_dim),  # bnsd
        m.kv_a_layernorm.weight,
        cos,
        sin,
        forward_batch.out_cache_loc.to(torch.int64),
        k_rope_cache,
        ckv_cache,
        k_rope_scale=None,
        c_kv_scale=None,
        k_rope_offset=None,
        c_kv_offset=None,
        epsilon=m.kv_a_layernorm.variance_epsilon,
        cache_mode="PA_NZ" if is_fia_nz() else "PA_BNSD",
        is_output_kv=True,
    )  # adapter NZ

    k_pe = k_pe.reshape(B, -1, m.qk_rope_head_dim)

    q[..., m.qk_nope_head_dim :] = q_pe

    kv = m.kv_b_proj(kv_a)[0]
    kv = kv.view(-1, m.num_local_heads, m.qk_nope_head_dim + m.v_head_dim)
    k_nope = kv[..., : m.qk_nope_head_dim]
    v = kv[..., m.qk_nope_head_dim :]

    k = m._concat_and_cast_mha_k(k_nope, k_pe, forward_batch)
    return q, k, v, forward_batch


def forward_mha_core_npu(
    m: "DeepseekV2AttentionMLA",
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    forward_batch: "ForwardBatch",
) -> torch.Tensor:
    attn_output = m.attn_mha(q, k, v, forward_batch, save_kv_cache=False)
    attn_output = attn_output.reshape(-1, m.num_local_heads * m.v_head_dim)
    output, _ = m.o_proj(attn_output)
    return output


# endregion


# region MLA
def forward_mla_prepare_npu(
    m: "DeepseekV2AttentionMLA",
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    forward_batch: "ForwardBatch",
    zero_allocator: "BumpAllocator",
):
    if is_mla_preprocess_enabled():
        if not hasattr(m, "mla_preprocess"):
            m.mla_preprocess = NPUFusedMLAPreprocess(
                m.fused_qkv_a_proj_with_mqa,
                m.q_a_layernorm,
                m.kv_a_layernorm,
                m.q_b_proj,
                m.w_kc,
                m.rotary_emb,
                m.layer_id,
                m.num_local_heads,
                m.qk_nope_head_dim,
                m.qk_rope_head_dim,
                m.quant_config,
            )
        (
            q_pe,
            k_pe,
            q_nope_out,
            k_nope,
            forward_batch,
            zero_allocator,
            positions,
        ) = m.mla_preprocess.forward(
            positions, hidden_states, forward_batch, zero_allocator
        )
        topk_indices = None
    else:
        q_lora = None
        if m.q_lora_rank is not None:
            q, latent_cache = (
                get_attn_tp_context()
                .fetch_qkv_latent()
                .split(
                    [m.q_lora_rank, m.kv_lora_rank + m.qk_rope_head_dim],
                    dim=-1,
                )
            )
            k_nope = latent_cache[..., : m.kv_lora_rank]

            q = m.q_a_layernorm(q)
            k_nope = m.kv_a_layernorm(k_nope)

            # q_lora needed by indexer
            if m.use_nsa:
                q_lora = q

            k_nope = k_nope.unsqueeze(1)
            q = m.q_b_proj(q)[0].view(-1, m.num_local_heads, m.qk_head_dim)
        else:
            q = m.q_proj(hidden_states)[0].view(-1, m.num_local_heads, m.qk_head_dim)
            latent_cache = m.kv_a_proj_with_mqa(hidden_states)[0]
            k_nope = latent_cache[..., : m.kv_lora_rank]
            k_nope = m.kv_a_layernorm(k_nope).unsqueeze(1)

        q_nope, q_pe = q.split([m.qk_nope_head_dim, m.qk_rope_head_dim], dim=-1)
        k_pe = latent_cache[..., m.kv_lora_rank :].unsqueeze(1)

        q_nope_out = torch.bmm(q_nope.transpose(0, 1), m.w_kc)

        q_nope_out = q_nope_out.transpose(0, 1)

        if enable_prefill_cp(forward_batch, m.nsa_enable_prefill_cp):
            positions = cp_split_and_rebuild_position(forward_batch, positions)

        q_pe, k_pe = m.rotary_emb(positions, q_pe, k_pe)

        if enable_prefill_cp(forward_batch, m.nsa_enable_prefill_cp):
            # support allgather+rerrange
            k_nope, k_pe = m.rebuild_cp_kv_cache(
                latent_cache, forward_batch, k_nope, k_pe
            )
        topk_indices = None
        if q_lora is not None:
            topk_indices = m.indexer(
                x=hidden_states,
                q_lora=q_lora,
                positions=positions,
                forward_batch=forward_batch,
                layer_id=m.layer_id,
            )

    return (
        q_pe,
        k_pe,
        q_nope_out,
        k_nope,
        forward_batch,
        zero_allocator,
        positions,
        topk_indices,
    )


def forward_mla_core_npu(
    m: "DeepseekV2AttentionMLA",
    q_pe: torch.Tensor,
    k_pe: torch.Tensor,
    q_nope_out: torch.Tensor,
    k_nope: torch.Tensor,
    forward_batch: "ForwardBatch",
    zero_allocator: "BumpAllocator",
    positions: torch.Tensor,
    topk_indices: torch.Tensor,
) -> torch.Tensor:
    attn_output = m.attn_mqa(
        q_nope_out,
        k_nope,
        k_nope,
        forward_batch,
        q_rope=q_pe,
        k_rope=k_pe,
        **(dict(topk_indices=topk_indices) if topk_indices is not None else {}),
    )

    attn_output = attn_output.view(-1, m.num_local_heads, m.kv_lora_rank)

    attn_bmm_output = torch.empty(
        (attn_output.shape[0], m.num_local_heads, m.v_head_dim),
        dtype=attn_output.dtype,
        device=attn_output.device,
    )

    attn_output = attn_output.contiguous()
    torch.ops.npu.batch_matmul_transpose(attn_output, m.w_vc, attn_bmm_output)

    attn_bmm_output = attn_bmm_output.reshape(-1, m.num_local_heads * m.v_head_dim)
    output, _ = m.o_proj(attn_bmm_output)

    return output


# endregion


# region DSA
def forward_dsa_prepare_npu(
    m: "DeepseekV2AttentionMLA",
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    forward_batch: "ForwardBatch",
    zero_allocator: "BumpAllocator",
):
    if is_mla_preprocess_enabled() and forward_batch.forward_mode.is_decode():
        if not hasattr(m, "mla_preprocess"):
            m.mla_preprocess = NPUFusedMLAPreprocess(
                m.fused_qkv_a_proj_with_mqa,
                m.q_a_layernorm,
                m.kv_a_layernorm,
                m.q_b_proj,
                m.w_kc,
                m.rotary_emb,
                m.layer_id,
                m.num_local_heads,
                m.qk_nope_head_dim,
                m.qk_rope_head_dim,
                m.quant_config,
            )
        mla_event = torch.npu.Event()
        mla_event.record()
        with torch.npu.stream(m.alt_stream):
            torch.npu.current_stream().wait_event(mla_event)
            (
                q_pe,
                k_pe,
                q_nope_out,
                k_nope,
                forward_batch,
                zero_allocator,
                positions,
            ) = m.mla_preprocess.forward(
                positions, hidden_states, forward_batch, zero_allocator
            )
        fused_qkv_a_proj_out = m.fused_qkv_a_proj_with_mqa(hidden_states)[0]
        q, _ = fused_qkv_a_proj_out.split(
            [m.q_lora_rank, m.kv_lora_rank + m.qk_rope_head_dim], dim=-1
        )
        q_lora = m.q_a_layernorm(q)
        torch.npu.current_stream().wait_stream(m.alt_stream)
    else:
        fused_qkv_a_proj_out = m.fused_qkv_a_proj_with_mqa(hidden_states)[0]
        q, latent_cache = fused_qkv_a_proj_out.split(
            [m.q_lora_rank, m.kv_lora_rank + m.qk_rope_head_dim], dim=-1
        )

        # overlap qk norm
        q = m.q_a_layernorm(q)

        q_lora = q.clone()  # required for topk_indices

        m.alt_stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(m.alt_stream):
            q = m.q_b_proj(q_lora)[0].view(-1, m.num_local_heads, m.qk_head_dim)
            q.record_stream(m.alt_stream)
            q_event = m.alt_stream.record_event()

        k_nope, k_pe = latent_cache.unsqueeze(1).split(
            [m.kv_lora_rank, m.qk_rope_head_dim], dim=-1
        )
        k_nope = m.kv_a_layernorm(k_nope).unsqueeze(1)
        torch.npu.current_stream().wait_event(q_event)

        q_nope, q_pe = q.split([m.qk_nope_head_dim, m.qk_rope_head_dim], dim=-1)

        q_nope_out = torch.bmm(q_nope.transpose(0, 1), m.w_kc)

        q_nope_out = q_nope_out.transpose(0, 1)

        if enable_prefill_cp(forward_batch, m.nsa_enable_prefill_cp):
            positions = cp_split_and_rebuild_position(forward_batch, positions)

        q_pe, k_pe = m.rotary_emb(positions, q_pe, k_pe)

        if enable_prefill_cp(forward_batch, m.nsa_enable_prefill_cp):
            # support allgather+rerrange
            k_nope, k_pe = m.rebuild_cp_kv_cache(
                latent_cache, forward_batch, k_nope, k_pe
            )

    topk_indices = m.indexer(
        hidden_states, q_lora, positions, forward_batch, m.layer_id
    )

    return (
        q_pe,
        k_pe,
        q_nope_out,
        k_nope,
        topk_indices,
        forward_batch,
        zero_allocator,
        positions,
    )


def forward_dsa_core_npu(
    m: "DeepseekV2AttentionMLA",
    q_pe: torch.Tensor,
    k_pe: torch.Tensor,
    q_nope_out: torch.Tensor,
    k_nope: torch.Tensor,
    topk_indices: torch.Tensor,
    forward_batch: "ForwardBatch",
    zero_allocator: "BumpAllocator",
    positions: torch.Tensor,
) -> torch.Tensor:
    attn_output = m.attn_mqa(
        q_nope_out.contiguous(),
        k_nope.contiguous(),
        k_nope.contiguous(),
        forward_batch,
        save_kv_cache=True,  # False if forward_batch.forward_mode.is_extend() else True,
        q_rope=q_pe.contiguous(),
        k_rope=k_pe.contiguous(),
        topk_indices=topk_indices,
    )
    attn_output = attn_output.view(-1, m.num_local_heads, m.kv_lora_rank)

    attn_bmm_output = torch.empty(
        (attn_output.shape[0], m.num_local_heads, m.v_head_dim),
        dtype=attn_output.dtype,
        device=attn_output.device,
    )

    if (
        forward_batch.forward_mode.is_extend()
        and not forward_batch.forward_mode.is_draft_extend(include_v2=True)
        and not forward_batch.forward_mode.is_target_verify()
    ):
        attn_output = attn_output.transpose(0, 1)
        torch.bmm(
            attn_output,
            m.w_vc,
            out=attn_bmm_output.view(-1, m.num_local_heads, m.v_head_dim).transpose(
                0, 1
            ),
        )
    else:
        attn_output = attn_output.contiguous()
        torch.ops.npu.batch_matmul_transpose(attn_output, m.w_vc, attn_bmm_output)

    attn_bmm_output = attn_bmm_output.reshape(-1, m.num_local_heads * m.v_head_dim)

    output, _ = m.o_proj(attn_bmm_output)
    return output


# endregion
