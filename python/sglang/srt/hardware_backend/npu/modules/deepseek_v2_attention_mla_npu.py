from typing import TYPE_CHECKING

import torch

from sglang.srt.hardware_backend.npu.attention.mla_preprocess import (
    NPUFusedMLAPreprocess,
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

    kv_a = m.kv_a_layernorm(kv_a)
    kv = m.kv_b_proj(kv_a)[0]

    k_pe = latent_cache[:, :, m.kv_lora_rank :]
    if m.rotary_emb is not None:
        q_pe, k_pe = m.rotary_emb(positions, q_pe, k_pe)
    q[..., m.qk_nope_head_dim :] = q_pe

    m._set_mla_kv_buffer(latent_cache, kv_a, k_pe, forward_batch)
    if forward_batch.mha_one_shot and sum(forward_batch.extend_prefix_lens_cpu) != 0:
        if m.use_nsa and m.kv_cache_dtype == "fp8_e4m3":
            # FP8 path: dequantize NSA-specific FP8 format to BF16
            kv_a, k_pe = m._get_mla_kv_buffer_from_fp8(forward_batch)
        else:
            # BF16/FP16 path: directly fetch from cache
            kv_a, k_pe = m._get_mla_kv_buffer(
                forward_batch.fetch_mha_one_shot_kv_indices(),
                q.dtype,
                forward_batch,
            )
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
        (attn_output.shape[0], m.num_local_heads * m.v_head_dim),
        dtype=attn_output.dtype,
        device=attn_output.device,
    )
    torch.bmm(
        attn_output.transpose(0, 1),
        m.w_vc,
        out=attn_bmm_output.view(-1, m.num_local_heads, m.v_head_dim).transpose(0, 1),
    )
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
    else:
        fused_qkv_a_proj_out = m.fused_qkv_a_proj_with_mqa(hidden_states)[0]
        q, latent_cache = fused_qkv_a_proj_out.split(
            [m.q_lora_rank, m.kv_lora_rank + m.qk_rope_head_dim], dim=-1
        )
        k_nope = latent_cache[..., : m.kv_lora_rank]

        q = m.q_a_layernorm(q)
        k_nope = m.kv_a_layernorm(k_nope)

        q_lora = q.clone()  # required for topk_indices
        k_nope = k_nope.unsqueeze(1)
        q = m.q_b_proj(q)[0].view(-1, m.num_local_heads, m.qk_head_dim)

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

    if not forward_batch.forward_mode.is_decode():
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
