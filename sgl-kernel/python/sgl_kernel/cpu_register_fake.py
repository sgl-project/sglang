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

@torch.library.register_fake("sgl_kernel::fp8_scaled_mm_cpu")
def _(
    mat1,
    mat2,
    scales2,
    block_size,
    bias,
    out_dtype,
    is_vnni,
):
    M = mat1.shape[0]
    N = mat2.shape[0]
    return mat1.new_empty(M, N, dtype=out_dtype)
