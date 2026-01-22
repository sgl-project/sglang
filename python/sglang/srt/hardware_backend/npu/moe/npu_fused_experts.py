def npu_fused_experts_wna16(
    hidden_states: torch.Tensor,
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w13_offset: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    w2_offset: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    top_k: int,
):
    original_dtype = hidden_states.dtype
    scale_dtype = original_dtype if original_dtype == torch.bfloat16 else torch.float32
    num_tokens = hidden_states.shape[0]
    num_experts = w13.shape[0]
    row_idx_len = num_tokens * top_k
    row_idx = (
        torch.arange(0, row_idx_len, dtype=torch.int32, device=topk_weights.device)
        .view(top_k, -1)
        .permute(1, 0)
        .contiguous()
    )
    hidden_states, expanded_row_idx, expanded_expert_idx = (
        torch.ops.npu.npu_moe_init_routing(
            hidden_states, row_idx=row_idx, expert_idx=topk_ids, active_num=num_tokens
        )
    )
    expert_tokens = torch.ops.npu.npu_moe_compute_expert_tokens(
        expanded_expert_idx, num_experts
    )
    expert_tokens = expert_tokens.to(torch.int64)
    # gmm1: gate_up_proj
    scale_args13 = {
        "antiquant_scale": [w13_scale],
        "antiquant_offset": [w13_offset],
    }
    hidden_states = torch.ops.npu.npu_grouped_matmul(
        x=[hidden_states],
        weight=[w13],
        **scale_args13,
        split_item=2,
        group_list_type=0,
        group_type=0,
        group_list=expert_tokens,
        output_dtype=original_dtype,
    )[0]
    # act_fn: swiglu
    hidden_states = torch.ops.npu.npu_swiglu(hidden_states)
    scale_args2 = {"antiquant_scale": [w2_scale], "antiquant_offset": [w2_offset]}
    # gmm2: down_proj
    hidden_states = torch.ops.npu.npu_grouped_matmul(
        x=[hidden_states],
        weight=[w2],
        **scale_args2,
        split_item=2,
        group_list_type=0,
        group_type=0,
        group_list=expert_tokens,
        output_dtype=original_dtype,
    )[0]
    final_hidden_states = torch.ops.npu.npu_moe_finalize_routing(
        hidden_states,
        skip1=None,
        skip2=None,
        bias=None,
        scales=topk_weights,
        expanded_src_to_dst_row=expanded_row_idx,
        export_for_source_row=topk_ids,
    )
    return final_hidden_states

def npu_fused_experts_w4a8(
    hidden_states: torch.Tensor,
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w13_scale_bias: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    w2_scale_bias: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    top_k: int,
):
    group_list_type = 1
    original_shape = hidden_states.shape
    topk_weights = topk_weights

    num_tokens = hidden_states.shape[:-1].numel()

    first_expert_idx = 0
    num_experts = w13.shape[0]
    last_expert_idx = num_experts
    global_num_experts = num_experts

    sorted_hidden_states, expanded_row_idx, expert_tokens, pertoken_scale = (
        torch.ops.npu.npu_moe_init_routing_v2(
            hidden_states,
            topk_ids,
            active_num=num_tokens * top_k,
            expert_num=global_num_experts,
            expert_tokens_num_type=1,
            expert_tokens_num_flag=True,
            active_expert_range=[first_expert_idx, last_expert_idx],
            quant_mode=1,
        )
    )

    expanded_row_idx = expanded_row_idx.view(-1, top_k).permute(1, 0).reshape(-1)

    expert_tokens = expert_tokens.to(torch.int64)
    _output_dtype = torch.bfloat16

    w1_scale = [w13_scale]
    w2_scale = [w2_scale]
    w1_scale[0] = w1_scale[0].to(w2_scale[0].dtype)

    hidden_states = torch.ops.npu.npu_grouped_matmul(
        x=[sorted_hidden_states],
        weight=[w13],
        scale=w1_scale,
        bias=[w13_scale_bias],
        per_token_scale=[pertoken_scale],
        group_list=expert_tokens,
        split_item=2,
        group_type=0,
        group_list_type=group_list_type,
        output_dtype=_output_dtype,
    )[0]

    # act_fn: swiglu
    hidden_states = torch.ops.npu.npu_swiglu(hidden_states)
    hidden_states, swiglu_out_scale = torch.ops.npu.npu_dynamic_quant(hidden_states)

    output = torch.ops.npu.npu_grouped_matmul(
        x=[hidden_states],
        weight=[w2],
        scale=w1_scale,
        bias=[w2_scale_bias],
        per_token_scale=[swiglu_out_scale],
        group_list=expert_tokens,
        split_item=2,
        group_type=0,
        group_list_type=group_list_type,
        output_dtype=_output_dtype,
    )[0]

    assert original_shape is not None
    final_hidden_states = torch.ops.npu.npu_moe_token_unpermute(
        permuted_tokens=output,
        sorted_indices=torch.abs(expanded_row_idx),
        probs=topk_weights,
    )

    return final_hidden_states

def npu_fused_experts_w8a8(
    hidden_states: torch.Tensor,
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w13_offset: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    w2_offset: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    top_k: int,
):
    original_dtype = hidden_states.dtype
    scale_dtype = original_dtype if original_dtype == torch.bfloat16 else torch.float32
    num_tokens = hidden_states.shape[0]
    num_experts = w13.shape[0]
    row_idx_len = num_tokens * top_k
    row_idx = (
        torch.arange(0, row_idx_len, dtype=torch.int32, device=topk_weights.device)
        .view(top_k, -1)
        .permute(1, 0)
        .contiguous()
    )
    hidden_states, expanded_row_idx, expanded_expert_idx = (
        torch.ops.npu.npu_moe_init_routing(
            hidden_states, row_idx=row_idx, expert_idx=topk_ids, active_num=num_tokens
        )
    )
    expert_tokens = torch.ops.npu.npu_moe_compute_expert_tokens(
        expanded_expert_idx, num_experts
    )
    expert_tokens = expert_tokens.to(torch.int64)
    # gmm1: gate_up_proj
    hidden_states, pertoken_scale = torch.ops.npu.npu_dynamic_quant(hidden_states)
    scale_args13 = {
        "scale": [w13_scale.to(scale_dtype)],
        "per_token_scale": [pertoken_scale],
    }
    hidden_states = torch.ops.npu.npu_grouped_matmul(
        x=[hidden_states],
        weight=[w13],
        **scale_args13,
        split_item=2,
        group_list_type=0,
        group_type=0,
        group_list=expert_tokens,
        output_dtype=original_dtype,
    )[0]
    # act_fn: swiglu
    hidden_states = torch.ops.npu.npu_swiglu(hidden_states)
    hidden_states, pertoken_scale = torch.ops.npu.npu_dynamic_quant(hidden_states)
    scale_args2 = {
        "scale": [w2_scale.to(scale_dtype)],
        "per_token_scale": [pertoken_scale],
    }
    # gmm2: down_proj
    hidden_states = torch.ops.npu.npu_grouped_matmul(
        x=[hidden_states],
        weight=[w2],
        **scale_args2,
        split_item=2,
        group_list_type=0,
        group_type=0,
        group_list=expert_tokens,
        output_dtype=original_dtype,
    )[0]
    final_hidden_states = torch.ops.npu.npu_moe_finalize_routing(
        hidden_states,
        skip1=None,
        skip2=None,
        bias=None,
        scales=topk_weights,
        expanded_src_to_dst_row=expanded_row_idx,
        export_for_source_row=topk_ids,
    )
    return final_hidden_states
