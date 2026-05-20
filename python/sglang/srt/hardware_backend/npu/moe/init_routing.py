import torch


class NPUMoEInitRouting_v1:
    """
    NPU MoE init routing (v1 API)
    """

    def _init_routing(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
    ):
        num_tokens = hidden_states.shape[0]
        row_idx_len = num_tokens * topk_ids.shape[1]
        row_idx = (
            torch.arange(0, row_idx_len, dtype=torch.int32, device=topk_ids.device)
            .view(topk_ids.shape[1], -1)
            .permute(1, 0)
            .contiguous()
        )

        hidden_states, expanded_row_idx, expanded_expert_idx = (
            torch.ops.npu.npu_moe_init_routing(
                hidden_states,
                row_idx=row_idx,
                expert_idx=topk_ids,
                active_num=num_tokens,
            )
        )
        expert_tokens = torch.ops.npu.npu_moe_compute_expert_tokens(
            expanded_expert_idx, num_experts
        )
        expert_tokens = expert_tokens.to(torch.int64)
        return hidden_states, expanded_row_idx, expert_tokens


class NPUMoEInitRouting_v2:
    """
    NPU MoE init routing (v2 API)
    """

    def _init_routing(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
    ):
        num_tokens = hidden_states.shape[0]

        hidden_states, expanded_row_idx, expert_tokens, _ = (
            torch.ops.npu.npu_moe_init_routing_v2(
                hidden_states,
                topk_ids,
                active_num=num_tokens * topk_ids.shape[1],
                expert_num=num_experts,
                expert_tokens_num_type=1,
                expert_tokens_num_flag=True,
                active_expert_range=[0, num_experts],
                quant_mode=-1,
            )
        )
        expert_tokens = expert_tokens.to(torch.int64)
        return hidden_states, expanded_row_idx, expert_tokens
