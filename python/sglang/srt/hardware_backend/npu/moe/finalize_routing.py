"""
NPU MoE finalize routing components.

These classes reassemble expert outputs into the original token order
after the expert computation. Two strategies are provided:
- Standard routing finalization with topk information.
- Simplified token unpermute for sorted token sequences.
"""

from abc import ABC, abstractmethod

import torch


class BaseFinalizeRouting(ABC):
    """
    Abstract base for NPU MoE finalize routing variants.
    All subclasses must implement ``_finalize_routing`` with the same signature.
    """

    @abstractmethod
    def _finalize_routing(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        expanded_row_idx: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor: ...


class NPUFinalizeRouting(BaseFinalizeRouting):
    """
    Standard NPU MoE finalize routing.
    Reassembles the results of the expert calculations in the original token order.
    """

    def __init__(self, drop_pad_mode: int = 0):
        self.drop_pad_mode = drop_pad_mode

    def _finalize_routing(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        expanded_row_idx: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        final_hidden_states = torch.ops.npu.npu_moe_finalize_routing(
            hidden_states,
            skip1=None,
            skip2=None,
            bias=None,
            scales=topk_weights,
            expanded_src_to_dst_row=expanded_row_idx,
            export_for_source_row=topk_ids,
            drop_pad_mode=self.drop_pad_mode,
        )
        return final_hidden_states


class NPUMoETokenUnpermute(BaseFinalizeRouting):
    """
    Simplified NPU MoE token unpermute (without topk_ids).
    Restores the original token order using sorted indices.
    Used when the token sequence is already sorted.
    """

    def _finalize_routing(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        expanded_row_idx: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        final_hidden_states = torch.ops.npu.npu_moe_token_unpermute(
            permuted_tokens=hidden_states,
            sorted_indices=expanded_row_idx.abs(),
            probs=topk_weights,
        )
        return final_hidden_states
