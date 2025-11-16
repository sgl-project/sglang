from typing import TYPE_CHECKING

import torch

from sglang.srt.layers.attention.npu_ops.mla_preprocess import NPUFusedMLAPreprocess

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.models.deepseek_v2 import DeepseekV2AttentionMLA
    from sglang.srt.utils import BumpAllocator


def forward_dsa_prepare(
    self: DeepseekV2AttentionMLA,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    forward_batch: ForwardBatch,
    zero_allocator: BumpAllocator,
):
    if self.is_mla_preprocess_enabled and forward_batch.forward_mode.is_decode():
        if self.mla_preprocess is None:
            self.mla_preprocess = NPUFusedMLAPreprocess(
                self.fused_qkv_a_proj_with_mqa,
                self.q_a_layernorm,
                self.kv_a_layernorm,
                self.q_b_proj,
                self.w_kc,
                self.rotary_emb,
                self.layer_id,
                self.num_local_heads,
                self.qk_nope_head_dim,
                self.qk_rope_head_dim,
            )
        (
            q_pe,
            k_pe,
            q_nope_out,
            k_nope,
            forward_batch,
            zero_allocator,
            positions,
        ) = self.mla_preprocess.forward(
            positions, hidden_states, forward_batch, zero_allocator
        )

        fused_qkv_a_proj_out = self.fused_qkv_a_proj_with_mqa(hidden_states)[0]
        q, _ = fused_qkv_a_proj_out.split(
            [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim], dim=-1
        )
        q_lora = self.q_a_layernorm(q)
    else:

        fused_qkv_a_proj_out = self.fused_qkv_a_proj_with_mqa(hidden_states)[0]
        q, latent_cache = fused_qkv_a_proj_out.split(
            [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim], dim=-1
        )
        k_nope = latent_cache[..., : self.kv_lora_rank]

        q = self.q_a_layernorm(q)
        k_nope = self.kv_a_layernorm(k_nope)

        q_lora = q.clone()  # required for topk_indices
        k_nope = k_nope.unsqueeze(1)
        q = self.q_b_proj(q)[0].view(-1, self.num_local_heads, self.qk_head_dim)

        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        k_pe = latent_cache[..., self.kv_lora_rank :].unsqueeze(1)

        q_nope_out = torch.bmm(q_nope.transpose(0, 1), self.w_kc)

        q_nope_out = q_nope_out.transpose(0, 1)

        q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe)

    topk_indices = self.indexer(
        hidden_states, q_lora, positions, forward_batch, self.layer_id
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


def forward_dsa_core(
    self: DeepseekV2AttentionMLA,
    q_pe,
    k_pe,
    q_nope_out,
    k_nope,
    topk_indices,
    forward_batch,
    zero_allocator,
    positions,
):
    attn_output = self.attn_mqa(
        q_nope_out.contiguous(),
        k_nope.contiguous(),
        k_nope.contiguous(),
        forward_batch,
        save_kv_cache=True,  # False if forward_batch.forward_mode.is_extend() else True,
        q_rope=q_pe.contiguous(),
        k_rope=k_pe.contiguous(),
        topk_indices=topk_indices,
    )
    attn_output = attn_output.view(-1, self.num_local_heads, self.kv_lora_rank)

    attn_bmm_output = torch.empty(
        (attn_output.shape[0], self.num_local_heads, self.v_head_dim),
        dtype=attn_output.dtype,
        device=attn_output.device,
    )

    if not forward_batch.forward_mode.is_decode():
        attn_output = attn_output.transpose(0, 1)
        torch.bmm(
            attn_output,
            self.w_vc,
            out=attn_bmm_output.view(
                -1, self.num_local_heads, self.v_head_dim
            ).transpose(0, 1),
        )
    else:
        attn_output = attn_output.contiguous()
        torch.ops.npu.batch_matmul_transpose(attn_output, self.w_vc, attn_bmm_output)

    attn_bmm_output = attn_bmm_output.reshape(
        -1, self.num_local_heads * self.v_head_dim
    )

    output, _ = self.o_proj(attn_bmm_output)
    return output
