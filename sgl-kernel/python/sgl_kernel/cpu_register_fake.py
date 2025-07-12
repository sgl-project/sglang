import sgl_kernel
import torch

@torch.library.register_fake("sgl_kernel::shm_allreduce")
def _(
    data: torch.Tensor,
    reduce_op: int,
) -> None:
    return

@torch.library.register_fake("sgl_kernel::qkv_proj_with_rope_fused_weight")
def _(
    hidden_states,
    q_a_proj_weight,
    q_b_proj_weight,
    w_kc,
    q_a_layernorm_weight,
    kv_a_layernorm_weight,
    positions,
    cos_sin_cache,
    eps,
    use_int8_w8a8,
    use_fp8_w8a16,
    qkv_a_proj_scale,
    q_b_proj_scale,
    is_vnni,
    block_size,
    q_lora_rank,
    kv_lora_rank,
    qk_rope_head_dim):
    num_seqs = hidden_states.shape[0]
    num_heads = w_kc.shape[0]
    kv_lora_rank = w_kc.shape[1]
    weight_chunks = torch.split(q_a_proj_weight, [q_lora_rank, kv_lora_rank + qk_rope_head_dim], dim=0)
    qk_rope_head_dim = weight_chunks[1].shape[0] - kv_lora_rank
    q_input = torch.empty(num_seqs, num_heads, kv_lora_rank + qk_rope_head_dim, dtype=hidden_states.dtype, device=hidden_states.device)
    k_input = torch.empty(num_seqs, 1, kv_lora_rank + qk_rope_head_dim, dtype=hidden_states.dtype, device=hidden_states.device)
    v_input = k_input.narrow(-1, 0, kv_lora_rank)
    return q_input, k_input, v_input

@torch.library.register_fake("sgl_kernel::bmm_cpu")
def _(out: torch.Tensor, mat1:torch.Tensor, mat2:torch.Tensor, is_vnni:bool, scale:torch.Tensor)-> None:
    return

@torch.library.register_fake("sgl_kernel::fused_add_rmsnorm_cpu")
def _(input: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float)-> None:
    return

@torch.library.register_fake("sgl_kernel::weight_packed_linear")
def _(x, weight, bias, is_vnni):
    return x.new_empty(x.shape[0], weight.shape[0])

@torch.library.register_fake("sgl_kernel::shared_expert_cpu")
def _(
    hidden_states,
    w1,
    w2,
    fused_experts_out,
    routed_scaling_factor,
    inplace,
    use_int8_w8a8,
    use_fp8_w8a16,
    w1_scale,
    w2_scale,
    block_size,
    a1_scale,
    a2_scale,
    is_vnni,
):
    return torch.empty_like(hidden_states)

@torch.library.register_fake("sgl_kernel::decode_attention_cpu")
def _(
    query: torch.Tensor,
    k_cache: torch.Tensor,
    v_cahce: torch.Tensor,
    output: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    loc: torch.Tensor,
    attn_logits: torch.Tensor,
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    sm_scale: float,
    logit_cap: float,)-> None:
    return

@torch.library.register_fake("sgl_kernel::fused_experts_cpu")
def _(
    x,
    w13_weight,
    w2_weight,
    topk_weights,
    topk_ids,
    inplace,
    use_int8_w8a8,
    use_fp8_w8a16,
    w1_scale,
    w2_scale,
    block_size,
    a1_scale,
    a2_scale,
    is_vnni,
):
    return torch.empty_like(x)



# # @torch.library.register_fake("sgl_kernel::forward_moe_fused")
# # def _(
# #     hidden_states,
# #     MoEGate_weight,
# #     MoEGate_bias,
# #     fused_experts_w13_weight,
# #     fused_experts_w2_weight,
# #     shared_expert_w1,
# #     shared_expert_w2,
# #     top_k,
# #     use_grouped_topk,
# #     renormalize,
# #     fused_experts_use_int8_w8a8,
# #     fused_experts_use_fp8_w8a16,
# #     fused_experts_inplace,
# #     routed_scaling_factor,
# #     shared_expert_inplace,
# #     shared_expert_use_int8_w8a8,
# #     shared_expert_use_fp8_w8a16,
# #     tp_size,
# #     topk_group,
# #     num_expert_group,
# #     correction_bias,
# #     fused_experts_w1_scale,
# #     fused_experts_w2_scale,
# #     fused_experts_a1_scale,
# #     fused_experts_a2_scale,
# #     fused_experts_block_size,
# #     shared_expert_w1_scale,
# #     shared_expert_w2_scale,
# #     shared_expert_block_size,
# #     shared_expert_a1_scale,
# #     shared_expert_a2_scale,
# #     process_group,
# #     op,
# #     is_vnni,
# # ):
# #     num_tokens = hidden_states.shape[0]
# #     hidden_dim = hidden_states.shape[1]
# #     return hidden_states.new_empty(num_tokens, hidden_dim)


# # @torch.library.register_fake("sgl_kernel::forward_absorb_decode_fused_cpu")
# # def _(
# #     hidden_states,
# #     q_a_proj_weight,
# #     q_b_proj_weight,
# #     kv_a_proj_weight,
# #     w_kc,
# #     q_a_layernorm_weight,
# #     kv_a_layernorm_weight,
# #     positions,
# #     cos_sin_cache,
# #     k_cache,
# #     v_cache,
# #     loc,
# #     attn_logits,
# #     req_to_token,
# #     req_pool_indices,
# #     seq_lens,
# #     w_vc,
# #     o_proj_weight,
# #     o_proj_bias,
# #     eps,
# #     use_int8_w8a8,
# #     use_fp8_w8a16,
# #     sm_scale,
# #     logit_cap,
# #     tp_k_head_num,
# #     qk_head_dim,
# #     tp_v_head_num,
# #     v_head_dim,
# #     tp_q_head_num,
# #     num_local_heads,
# #     kv_lora_rank,
# #     tp_size,
# #     tp_rank,
# #     o_proj_use_int8_w8a8,
# #     o_proj_use_fp8_w8a16,
# #     o_proj_out_dtype,
# #     q_a_proj_scale,
# #     q_b_proj_scale,
# #     kv_a_proj_scale,
# #     block_size,
# #     bmm_scale,
# #     process_group,
# #     op,
# #     o_proj_scale,
# #     o_proj_block_size,
# #     is_vnni):
# #     num_seqs = hidden_states.shape[0]
# #     num_heads = w_kc.shape[0]
# #     kv_lora_rank = w_kc.shape[1]
# #     qk_rope_head_dim = kv_a_proj_weight.shape[0] - kv_lora_rank
# #     query = hidden_states.new_empty(num_seqs, num_heads, kv_lora_rank + qk_rope_head_dim)
    
# #     query= query.reshape(-1, tp_q_head_num * qk_head_dim)
# #     if qk_head_dim != v_head_dim:
# #         attn_output = query.new_empty(query.shape[0], tp_q_head_num * v_head_dim)
# #     else:
# #         attn_output = query
# #     attn_output = attn_output.reshape(-1, num_local_heads, kv_lora_rank)
# #     M = attn_output.shape[0]
# #     N = o_proj_weight.shape[0]
# #     return attn_output.new_empty(M, N)


# @torch.library.register_fake("sgl_kernel::grouped_topk_cpu")
# def _(
#     hidden_states,
#     router_logits,
#     top_k,
#     renormalize,
#     num_expert_group,
#     topk_group,
# ):
#     shape = (hidden_states.shape[0], top_k)
#     device = hidden_states.device
#     topk_weights = torch.empty(shape, device=device, dtype=torch.float32)
#     topk_ids = torch.empty(shape, device=device, dtype=torch.int)
#     return topk_weights, topk_ids


@torch.library.register_fake("sgl_kernel::biased_grouped_topk_cpu")
def _(
    hidden_states,
    gating_output,
    correction_bias,
    topk,
    renormalize,
    num_expert_group,
    topk_group,
    num_fused_shared_experts,
    routed_scaling_factor,
    num_token_non_padded,
):
    num_tokens = hidden_states.shape[0]
    num_experts = gating_output.shape[1]
    shape = (num_tokens, num_experts)
    device = hidden_states.device
    topk_weights = torch.empty(shape, device=device, dtype=torch.float32)
    topk_ids = torch.empty(shape, device=device, dtype=torch.int)
    return topk_weights, topk_ids


@torch.library.register_fake("sgl_kernel::rmsnorm_cpu")
def _(input, weight, eps):
    return torch.empty_like(input)


# @torch.library.register_fake("sgl_kernel::int8_scaled_mm_with_quant")
# def _(
#     mat1,
#     mat2,
#     scales2,
#     bias,
#     out_dtype,
#     is_vnni):
#     M = mat1.shape[0]
#     N = mat2.shape[0]
#     return mat1.new_empty(M, N, dtype=out_dtype)


# # @torch.library.register_fake("sgl_kernel::rotary_position_embedding_cpu")
# # def _(t_pos, q_pe, k_pe, t_emb_pos):
# #     return torch.empty_like(q_pe), torch.empty_like(k_pe)

@torch.library.register_fake("sgl_kernel::silu_and_mul_cpu")
def _(input):
    return input.new_empty(input.shape[0], input.shape[1] // 2)

@torch.library.register_fake("sgl_kernel::int8_scaled_mm_with_quant")
def _(
    mat1,
    mat2,
    scales2,
    bias,
    out_dtype,
    is_vnni,
):
    M = mat1.shape[0]
    N = mat2.shape[0]
    return mat1.new_empty(M, N, dtype=out_dtype)
