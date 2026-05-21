import torch


class NPUFinalizeRouting:
    """
    NPU MoE finalize routing (standard path).
    Reassemble the results of the expert calculations in the original token order.
    """

    def _finalize_routing(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        expanded_row_idx: torch.Tensor,
        topk_ids: torch.Tensor,
    ):
        final_hidden_states = torch.ops.npu.npu_moe_finalize_routing(
            hidden_states,
            skip1=None,
            skip2=None,
            bias=None,
            scales=topk_weights,
            expanded_src_to_dst_row=expanded_row_idx,
            export_for_source_row=topk_ids,
            drop_pad_mode=2,
        )
        return final_hidden_states


class NPUMoETokenUnpermute:
    """
    NPU MoE token unpermute (simplified path without topk_ids).
    Used for the sorted token sequence, restore the original order through 'sorted_indices'.
    """

    def _finalize_routing(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        expanded_row_idx: torch.Tensor,
        topk_ids: torch.Tensor,
    ):
        final_hidden_states = torch.ops.npu.npu_moe_token_unpermute(
            permuted_tokens=hidden_states,
            sorted_indices=expanded_row_idx.abs(),
            probs=topk_weights,
        )
        return final_hidden_states
