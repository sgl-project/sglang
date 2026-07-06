"""
NPU MoE finalize routing components.

These classes reassemble expert outputs into the original token order
after the expert computation. A generic TP‑all‑gather wrapper is provided
to transparently gather the hidden dimension when needed (e.g. GGUF with
full weights).
"""

from abc import ABC, abstractmethod

import torch

from sglang.srt.distributed.communication_op import (
    tensor_model_parallel_all_gather,
)
from sglang.srt.distributed.parallel_state import (
    get_tensor_model_parallel_world_size,
)


class BaseFinalizeRouting(ABC):
    @abstractmethod
    def _finalize_routing(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        expanded_row_idx: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor: ...


# ---------------------------------------------------------------------------
# Concrete implementations (unchanged)
# ---------------------------------------------------------------------------
class NPUFinalizeRouting(BaseFinalizeRouting):
    def __init__(self, drop_pad_mode: int = 0):
        self.drop_pad_mode = drop_pad_mode

    def _finalize_routing(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        expanded_row_idx: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        return torch.ops.npu.npu_moe_finalize_routing(
            hidden_states,
            skip1=None,
            skip2=None,
            bias=None,
            scales=topk_weights,
            expanded_src_to_dst_row=expanded_row_idx,
            export_for_source_row=topk_ids,
            drop_pad_mode=self.drop_pad_mode,
        )


class NPUMoETokenUnpermute(BaseFinalizeRouting):
    def _finalize_routing(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        expanded_row_idx: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        return torch.ops.npu.npu_moe_token_unpermute(
            permuted_tokens=hidden_states,
            sorted_indices=expanded_row_idx.abs(),
            probs=topk_weights,
        )


# ---------------------------------------------------------------------------
# Generic TP‑all‑gather wrapper – transparently adds communication
# ---------------------------------------------------------------------------
class AllGatherFinalizeRoutingWrapper(BaseFinalizeRouting):
    """
    Wraps any finalize routing and performs an all‑gather along `dim`
    after the routing if tensor‑parallelism is active.

    This keeps the runner / permute hooks free of TP logic.
    """

    def __init__(self, inner: BaseFinalizeRouting, dim: int = -1):
        self.inner = inner
        self.dim = dim

    def _finalize_routing(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        expanded_row_idx: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        out = self.inner._finalize_routing(
            hidden_states, topk_weights, expanded_row_idx, topk_ids
        )
        if get_tensor_model_parallel_world_size() > 1:
            out = tensor_model_parallel_all_gather(out, dim=self.dim)
        return out
