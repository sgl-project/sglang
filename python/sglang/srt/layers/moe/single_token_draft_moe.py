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
from sglang.srt.environ import envs
from sglang.srt.layers.dp_attention import is_allocation_symmetric
from sglang.srt.layers.moe.topk import StandardTopKOutput, TopKOutputChecker
from sglang.srt.runtime_context import get_parallel

logger = logging.getLogger(__name__)


class SingleTokenDraftMoe:
    """Eligibility verdict and scratch buffers for one MoE block."""

    MAX_TOKENS = 2

    def __init__(self, layer_id: int) -> None:
        self.layer_id = layer_id
        self.active = True
        self.fused_router = envs.SGLANG_Q35MF_DRAFT_MOE_ROUTER.get()
        self._scratch: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        self._router_scratch: Optional[Tuple[torch.Tensor, ...]] = None

    def _unsupported_reason(self, block, hidden_states, topk_output) -> Optional[str]:
        from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod

        experts, shared = block.experts, block.shared_expert
        if shared is None or block.shared_expert_gate is None:
            return "no separate shared expert"
        # The GEMV passes read W13 as [gate rows | up rows] in checkpoint order.
        method = experts.quant_method
        if not isinstance(method, UnquantizedFusedMoEMethod):
            return f"expert weights handled by {type(method).__name__}"
        if method.use_flashinfer_trtllm_moe or method.w13_swiglu_interleaved:
            return "expert weights in a kernel-specific layout"
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

    @staticmethod
    def _router_unsupported_reason(block) -> Optional[str]:
        cfg = block.topk.topk_config
        if (
            cfg.scoring_func != "softmax"
            or not cfg.renormalize
            or cfg.use_grouped_topk
            or cfg.correction_bias is not None
            or cfg.custom_routing_function is not None
            or cfg.num_fused_shared_experts != 0
            or cfg.routed_scaling_factor not in (None, 1.0)
            or block.topk.enable_waterfill
        ):
            return "router is not plain softmax top-k with renormalization"
        w = block.gate.weight
        if w.dtype != torch.bfloat16 or w.shape[0] % 32 != 0 or not w.is_contiguous():
            return f"router weight {w.dtype} {tuple(w.shape)}"
        return None

    def _route(self, block, hidden_states: torch.Tensor):
        if self._router_scratch is None:
            return block.topk(hidden_states, block.gate(hidden_states)[0])
        from sglang.kernels.ops.speculative.draft_moe_gemv import draft_router_topk

        logits, counter, weights, ids = self._router_scratch
        draft_router_topk(hidden_states, block.gate.weight, logits, counter, weights, ids)
        return StandardTopKOutput(topk_weights=weights, topk_ids=ids, router_logits=logits)

    def _setup(self, block, hidden_states: torch.Tensor) -> bool:
        topk_output = block.topk(hidden_states, block.gate(hidden_states)[0])
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
            return False
        inter = block.experts.w2_weight.shape[-1]
        topk = topk_output.topk_ids.shape[-1]
        dev = hidden_states.device
        self._scratch = (
            torch.empty(topk + 1, inter, dtype=torch.bfloat16, device=dev),
            torch.empty(1, dtype=torch.float32, device=dev),
        )
        router_reason = self._router_unsupported_reason(block) if self.fused_router else "off"
        if router_reason is None:
            num_experts = block.gate.weight.shape[0]
            self._router_scratch = (
                torch.empty(num_experts, dtype=torch.float32, device=dev),
                torch.zeros(1, dtype=torch.int32, device=dev),
                torch.empty(1, topk, dtype=torch.float32, device=dev),
                torch.empty(1, topk, dtype=torch.int32, device=dev),
            )
        logger.info(
            "Single-token draft MoE active for layer %d (fused router: %s).",
            self.layer_id,
            "yes" if router_reason is None else router_reason,
        )
        return True

    def forward(self, block, hidden_states: torch.Tensor) -> Optional[torch.Tensor]:
        """Return the block output, or None when the caller must take the stock path.
        Each of the (at most MAX_TOKENS) tokens runs its own GEMV passes in turn."""
        from sglang.kernels.ops.speculative.draft_moe_gemv import draft_moe_gemv

        if self._scratch is None and not self._setup(block, hidden_states[:1]):
            return None
        with use_symmetric_memory(
            get_parallel().tp_group, disabled=not is_allocation_symmetric()
        ):
            out = torch.empty_like(hidden_states)
        act, gate = self._scratch
        stock_route = None
        if self._router_scratch is None:
            stock_route = self._route(block, hidden_states)
        for m in range(hidden_states.shape[0]):
            x = hidden_states[m : m + 1]
            if stock_route is None:
                topk_output = self._route(block, x)
                ids, weights = topk_output.topk_ids, topk_output.topk_weights
            else:
                ids = stock_route.topk_ids[m : m + 1]
                weights = stock_route.topk_weights[m : m + 1]
            draft_moe_gemv(
                x,
                block.experts.w13_weight,
                block.experts.w2_weight,
                block.shared_expert.gate_up_proj.weight,
                block.shared_expert.down_proj.weight,
                block.shared_expert_gate.weight.view(-1),
                ids,
                weights,
                out[m : m + 1],
                act,
                gate,
            )
        return out
