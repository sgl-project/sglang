"""Single-token bf16 draft MoE: routed top-k plus the gated shared expert as two
GEMV passes, replacing align + up GEMM + act + down GEMM + sum + shared expert +
gate combine. Rounding points match that path; only dot-product order differs."""

from __future__ import annotations

import logging
from typing import Callable, Optional, Tuple

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
        self.single_launch = envs.SGLANG_Q35MF_MKD_MOE1.get()
        self._scratch: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        self._router_scratch: Optional[Tuple[torch.Tensor, ...]] = None
        self._single_launch_scratch: Optional[Tuple[torch.Tensor, ...]] = None
        # Called after the block's last kernel (single-token path only).
        self.after_hook: Optional[Callable[[], None]] = None
        # Host int64 site fields of the hop the single launch pushes into, built on first use.
        self._hop_site: Optional[torch.Tensor] = None

    def ready(self, num_tokens: int) -> bool:
        """Set up by an eager call and taking a batch of this many tokens."""
        return (
            self.active
            and self._scratch is not None
            and 0 < num_tokens <= self.MAX_TOKENS
        )

    @property
    def single_launch_ready(self) -> bool:
        return self._single_launch_scratch is not None

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
        if tuple(ws13.shape) != (2 * inter, hidden) or tuple(ws2.shape) != (
            hidden,
            inter,
        ):
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
        draft_router_topk(
            hidden_states, block.gate.weight, logits, counter, weights, ids
        )
        return StandardTopKOutput(
            topk_weights=weights, topk_ids=ids, router_logits=logits
        )

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
        router_on = self.fused_router or self.single_launch
        router_reason = self._router_unsupported_reason(block) if router_on else "off"
        if router_reason is None and self.single_launch:
            from sglang.kernels.ops.speculative.draft_moe_gemv import (
                draft_moe_fused_scratch,
            )

            num_experts = block.gate.weight.shape[0]
            self._single_launch_scratch = draft_moe_fused_scratch(
                num_experts, topk, inter, dev
            ) + (
                torch.empty(1, topk, dtype=torch.float32, device=dev),
                torch.empty(1, topk, dtype=torch.int32, device=dev),
            )
        elif router_reason is None:
            num_experts = block.gate.weight.shape[0]
            self._router_scratch = (
                torch.empty(num_experts, dtype=torch.float32, device=dev),
                torch.zeros(1, dtype=torch.int32, device=dev),
                torch.empty(1, topk, dtype=torch.float32, device=dev),
                torch.empty(1, topk, dtype=torch.int32, device=dev),
            )
        logger.info(
            "Single-token draft MoE active for layer %d (fused router: %s%s).",
            self.layer_id,
            "yes" if router_reason is None else router_reason,
            ", single launch" if self._single_launch_scratch is not None else "",
        )
        return True

    def _forward_single_launch(
        self, block, hidden_states, out, push_hop
    ) -> torch.Tensor:
        from sglang.kernels.ops.speculative.draft_moe_gemv import draft_moe_fused

        logit_buf, act_buf, epoch, weights, ids = self._single_launch_scratch
        hop_site = None
        if push_hop is not None:
            if self._hop_site is None:
                self._hop_site = torch.tensor(push_hop.site_fields(), dtype=torch.int64)
            hop_site = self._hop_site
        for m in range(hidden_states.shape[0]):
            draft_moe_fused(
                hidden_states[m : m + 1],
                block.gate.weight,
                block.experts.w13_weight,
                block.experts.w2_weight,
                block.shared_expert.gate_up_proj.weight,
                block.shared_expert.down_proj.weight,
                block.shared_expert_gate.weight.view(-1),
                (logit_buf, act_buf, epoch),
                weights,
                ids,
                out[m : m + 1],
                hop_site=hop_site,
                hop_token=m,
            )
        if push_hop is not None:
            push_hop.mark_pushed(out)
        return out

    def forward(
        self, block, hidden_states: torch.Tensor, push_hop=None
    ) -> Optional[torch.Tensor]:
        """Return the block output, or None when the caller must take the stock path.
        Each of the (at most MAX_TOKENS) tokens runs its own GEMV passes in turn.
        push_hop: a Q35Hop the single launch also pushes the output into (row = token)."""
        if self._scratch is None and not self._setup(block, hidden_states[:1]):
            return None
        with use_symmetric_memory(
            get_parallel().tp_group, disabled=not is_allocation_symmetric()
        ):
            out = torch.empty_like(hidden_states)
        if self._single_launch_scratch is not None:
            self._forward_single_launch(block, hidden_states, out, push_hop)
        else:
            self._forward_gemv(block, hidden_states, out)
        if self.after_hook is not None:
            self.after_hook()
        return out

    def _forward_gemv(self, block, hidden_states, out) -> None:
        from sglang.kernels.ops.speculative.draft_moe_gemv import draft_moe_gemv

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
