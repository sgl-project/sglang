"""Single-token bf16 draft MoE: routed top-k plus the gated shared expert as two
GEMV passes, replacing align + up GEMM + act + down GEMM + sum + shared expert +
gate combine. Rounding points match that path; only dot-product order differs."""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch

from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.layers.dp_attention import is_allocation_symmetric
from sglang.srt.layers.moe.topk import TopKOutputChecker
from sglang.srt.runtime_context import get_parallel

logger = logging.getLogger(__name__)


class SingleTokenDraftMoe:
    """Eligibility verdict and scratch buffers for one MoE block."""

    def __init__(self, layer_id: int) -> None:
        self.layer_id = layer_id
        self.active = True
        self._scratch: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

    def _unsupported_reason(self, block, hidden_states, topk_output) -> Optional[str]:
        experts, shared = block.experts, block.shared_expert
        if shared is None or block.shared_expert_gate is None:
            return "no separate shared expert"
        if get_parallel().moe_ep_size != 1:
            return "expert parallelism"
        if not TopKOutputChecker.format_is_standard(topk_output):
            return "non-standard top-k output"
        w13, w2 = experts.w13_weight, experts.w2_weight
        ws13, ws2 = shared.gate_up_proj.weight, shared.down_proj.weight
        hidden = hidden_states.shape[-1]
        inter = w2.shape[-1]
        tensors = (hidden_states, w13, w2, ws13, ws2, block.shared_expert_gate.weight)
        if any(t.dtype != torch.bfloat16 for t in tensors):
            return "non-bf16 operands"
        if w13.dim() != 3 or tuple(w13.shape[1:]) != (2 * inter, hidden):
            return f"w13 shape {tuple(w13.shape)}"
        if tuple(ws13.shape) != (2 * inter, hidden) or tuple(ws2.shape) != (hidden, inter):
            return f"shared expert shapes {tuple(ws13.shape)} {tuple(ws2.shape)}"
        if hidden % 1024 != 0 or inter % 32 != 0:
            return f"hidden {hidden} / intermediate {inter} not tileable"
        if topk_output.topk_ids.dtype != torch.int32:
            return f"topk ids dtype {topk_output.topk_ids.dtype}"
        if topk_output.topk_weights.dtype != torch.float32:
            return f"topk weights dtype {topk_output.topk_weights.dtype}"
        if not all(t.is_contiguous() for t in tensors):
            return "non-contiguous operands"
        return None

    def forward(self, block, hidden_states: torch.Tensor) -> Optional[torch.Tensor]:
        """Return the block output, or None when the caller must take the stock path."""
        from sglang.kernels.ops.speculative.draft_moe_gemv import draft_moe_gemv

        router_logits, _ = block.gate(hidden_states)
        topk_output = block.topk(hidden_states, router_logits)
        if self._scratch is None:
            reason = self._unsupported_reason(block, hidden_states, topk_output)
            if reason is None and torch.cuda.is_current_stream_capturing():
                reason = "first call inside graph capture"
            if reason is not None:
                logger.warning(
                    "Single-token draft MoE disabled for layer %d (%s); using the stock path.",
                    self.layer_id,
                    reason,
                )
                self.active = False
                return None
            inter = block.experts.w2_weight.shape[-1]
            topk = topk_output.topk_ids.shape[-1]
            self._scratch = (
                torch.empty(topk + 1, inter, dtype=torch.bfloat16, device=hidden_states.device),
                torch.empty(1, dtype=torch.float32, device=hidden_states.device),
            )
            logger.info("Single-token draft MoE active for layer %d.", self.layer_id)
        with use_symmetric_memory(
            get_parallel().tp_group, disabled=not is_allocation_symmetric()
        ):
            out = torch.empty_like(hidden_states)
        act, gate = self._scratch
        draft_moe_gemv(
            hidden_states,
            block.experts.w13_weight,
            block.experts.w2_weight,
            block.shared_expert.gate_up_proj.weight,
            block.shared_expert.down_proj.weight,
            block.shared_expert_gate.weight.view(-1),
            topk_output.topk_ids,
            topk_output.topk_weights,
            out,
            act,
            gate,
        )
        return out
