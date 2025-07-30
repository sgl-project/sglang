import torch


@torch.library.register_fake("sgl_kernel::shm_allreduce")
def _(
    data,
    reduce_op,
) -> None:
    return


@torch.library.register_fake("sgl_kernel::qkv_proj_with_rope")
def _(
    hidden_states,
    q_a_proj_weight,
    q_b_proj_weight,
    kv_a_proj_weight,
    w_kc,
    q_a_layernorm_weight,
    kv_a_layernorm_weight,
    positions,
    cos_sin_cache,
    eps,
    use_int8_w8a8,
    use_fp8_w8a16,
    q_a_proj_scale,
    q_b_proj_scale,
    kv_a_proj_scale,
    is_vnni,
    block_size,
):
    num_seqs = hidden_states.shape[0]
    num_heads = w_kc.shape[0]
    kv_lora_rank = w_kc.shape[1]
    qk_rope_head_dim = kv_a_proj_weight.shape[0] - kv_lora_rank
    q_input = torch.empty(
        num_seqs,
        num_heads,
        kv_lora_rank + qk_rope_head_dim,
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    k_input = torch.empty(
        num_seqs,
        1,
        kv_lora_rank + qk_rope_head_dim,
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    v_input = k_input.narrow(-1, 0, kv_lora_rank)
    return q_input, k_input, v_input


@torch.library.register_fake("sgl_kernel::rotary_embedding_cpu")
def _(positions, query, key, head_size, cos_sin_cache, is_neox):
    if query.ndim == 2:
        return query, key
    else:
        return torch.empty_like(query), torch.empty_like(key)


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
    qk_rope_head_dim,
):
    num_seqs = hidden_states.shape[0]
    num_heads = w_kc.shape[0]
    kv_lora_rank = w_kc.shape[1]
    weight_chunks = torch.split(
        q_a_proj_weight, [q_lora_rank, kv_lora_rank + qk_rope_head_dim], dim=0
    )
    qk_rope_head_dim = weight_chunks[1].shape[0] - kv_lora_rank
    q_input = torch.empty(
        num_seqs,
        num_heads,
        kv_lora_rank + qk_rope_head_dim,
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    k_input = torch.empty(
        num_seqs,
        1,
        kv_lora_rank + qk_rope_head_dim,
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    v_input = k_input.narrow(-1, 0, kv_lora_rank)
    return q_input, k_input, v_input


@torch.library.register_fake("sgl_kernel::bmm_cpu")
def _(out, mat1, mat2, is_vnni, scale) -> None:
    return


@torch.library.register_fake("sgl_kernel::fused_add_rmsnorm_cpu")
def _(input, residual, weight, eps) -> None:
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
    query,
    k_cache,
    v_cahce,
    output,
    key,
    value,
    loc,
    attn_logits,
    req_to_token,
    req_pool_indices,
    seq_lens,
    sm_scale,
    logit_cap,
) -> None:
    return


@torch.library.register_fake("sgl_kernel::extend_attention_cpu")
def _(
    q_extend,
    k_extend,
    v_extend,
    o_extend,
    k_buffer,
    v_buffer,
    req_to_token,
    req_pool_indices,
    seq_lens,
    extend_seq_lens,
    extend_start_loc,
    max_len_extend,
    sm_scale,
    logit_cap,
) -> None:
    return


@torch.library.register_fake("sgl_kernel::per_token_quant_int8_cpu")
def _(input):
    M = input.shape[0]
    K = input.shape[1]
    Aq = input.new_empty(M, K, dtype=torch.int8)
    As = input.new_empty(M, dtype=torch.float32)
    return Aq, As


@torch.library.register_fake("sgl_kernel::int8_scaled_mm_cpu")
def _(mat1, mat2, scales1, scales2, bias, out_dtype, is_vnni):
    M = mat1.shape[0]
    N = mat2.shape[0]
    k = mat1.shape[1]
    out = mat1.new_empty(M, N, dtype=out_dtype)
    return out


@torch.library.register_fake("sgl_kernel::fused_experts_cpu")
def _(
    hidden_states,
    w1,
    w2,
    topk_weights,
    topk_ids,
    inplace,
    use_int8_w8a8,
    use_fp8_w8a16,
    use_int4_w4a16,
    w1_scale,
    w2_scale,
    w1_zero,
    w2_zero,
    block_size,
    a1_scale,
    a2_scale,
    is_vnni,
):
    return torch.empty_like(hidden_states)


@torch.library.register_fake("sgl_kernel::grouped_topk_cpu")
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
    shape = (num_tokens, topk)
    device = hidden_states.device
    topk_weights = torch.empty(shape, device=device, dtype=torch.float32)
    topk_ids = torch.empty(shape, device=device, dtype=torch.int)
    return topk_weights, topk_ids


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
    shape = (num_tokens, topk)
    device = hidden_states.device
    topk_weights = torch.empty(shape, device=device, dtype=torch.float32)
    topk_ids = torch.empty(shape, device=device, dtype=torch.int)
    return topk_weights, topk_ids


@torch.library.register_fake("sgl_kernel::rmsnorm_cpu")
def _(input, weight, eps):
    return torch.empty_like(input)


@torch.library.register_fake("sgl_kernel::l2norm_cpu")
def _(input, eps):
    return torch.empty_like(input)


@torch.library.register_fake("sgl_kernel::topk_sigmoid_cpu")
def _(hidden_states, gating_output, topk, renormalize):
    num_tokens = hidden_states.shape[0]
    shape = (num_tokens, topk)
    return (
        torch.empty(shape, device=hidden_states.device, dtype=torch.float),
        torch.empty(shape, device=hidden_states.device, dtype=torch.int),
    )


@torch.library.register_fake("sgl_kernel::topk_softmax_cpu")
def _(
    hidden_states,
    gating_output,
    topk,
    renormalize,
):
    num_tokens = hidden_states.shape[0]
    shape = (num_tokens, topk)
    return (
        torch.empty(shape, device=hidden_states.device, dtype=torch.float),
        torch.empty(shape, device=hidden_states.device, dtype=torch.int),
    )


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


@torch.library.register_fake("sgl_kernel::int4_scaled_mm_cpu")
def _(
    x,
    w,
    w_zeros,
    w_scales,
    bias,
):
    M = x.shape[0]
    N = w.shape[0]
    return x.new_empty(M, N)


@torch.library.register_fake("sgl_kernel::int4_scaled_mm_cpu_with_quant")
def _(
    input,
    weight,
    weight_scales,
    weight_qzeros,
    compensation,
    bias,
    output_dtype,
):
    N = weight.shape[0] * weight.shape[-1] * 2
    shape = list(input.shape)
    shape[-1] = N
    return input.new_empty(shape, dtype=output_dtype)
