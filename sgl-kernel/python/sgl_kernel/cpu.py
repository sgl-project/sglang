import sgl_kernel
import torch


def fused_experts(
    x,
    w13_weight,
    w2_weight,
    topk_weights,
    topk_ids,
    inplace,
    use_int8_w8a8=False,
    w1_scale=None,
    w2_scale=None,
    a1_scale=None,
    a2_scale=None,
    is_vnni=True,
):
    return sgl_kernel.common_ops.fused_experts_cpu(
        x,
        w13_weight,
        w2_weight,
        topk_weights,
        topk_ids,
        inplace,
        use_int8_w8a8,
        w1_scale,
        w2_scale,
        a1_scale,
        a2_scale,
        is_vnni,
    )


def convert_weight_packed(weight):
    return sgl_kernel.common_ops.convert_weight_packed(weight)


def decode_attention(
    q,
    k_buffer,
    v_buffer,
    o,
    kv_indptr,
    kv_indices,
    seq_lens,
    attn_logits,
    sm_scale,
    logit_cap=0.0,
):
    sgl_kernel.common_ops.decode_attention_cpu(
        q,
        o,
        k_buffer,
        v_buffer,
        attn_logits,
        kv_indptr,
        kv_indices,
        seq_lens,
        sm_scale,
        logit_cap,
    )


def extend_attention(
    q_extend,
    k_extend,
    v_extend,
    o_extend,
    k_buffer,
    v_buffer,
    req_to_tokens,
    req_pool_indices,
    seq_lens,
    extend_seq_lens,
    extend_start_loc,
    max_len_extend,
    sm_scale,
    logit_cap=0.0,
):
    sgl_kernel.common_ops.extend_attention_cpu(
        q_extend,
        k_extend,
        v_extend,
        o_extend,
        k_buffer,
        v_buffer,
        req_to_tokens,
        req_pool_indices,
        seq_lens,
        extend_seq_lens,
        extend_start_loc,
        max_len_extend,
        sm_scale,
        logit_cap,
    )


def weight_packed_linear(
    x,
    weight,
    bias,
    is_vnni=True,
):
    return sgl_kernel.common_ops.weight_packed_linear(
        x,
        weight,
        bias,
        is_vnni,
    )


def grouped_topk(
    topk_weights,
    topk_ids,
    hidden_states,
    router_logits,
    top_k,
    renormalize,
    num_expert_group,
    topk_group,
):
    sgl_kernel.common_ops.grouped_topk_cpu(
        topk_weights,
        topk_ids,
        hidden_states,
        router_logits,
        top_k,
        renormalize,
        num_expert_group,
        topk_group,
    )


def fused_add_rmsnorm(
    input,
    residual,
    weight,
    eps,
):
    sgl_kernel.common_ops.fused_add_rmsnorm_cpu(
        input,
        residual,
        weight,
        eps,
    )


def rmsnorm(
    output,
    input,
    weight,
    eps,
):
    return sgl_kernel.common_ops.rmsnorm_cpu(
        output,
        input,
        weight,
        eps,
    )


def int8_scaled_mm(
    mat1,
    mat2,
    scales1,
    scales2,
    bias,
    out_dtype,
    is_vnni=True,
):
    return sgl_kernel.common_ops.int8_scaled_mm_cpu(
        mat1, mat2, scales1, scales2, bias, out_dtype, is_vnni
    )


def per_token_quant_int8(x):
    return sgl_kernel.common_ops.per_token_quant_int8_cpu(x)
