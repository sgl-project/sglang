from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
from torch import nn

from sglang.srt.environ import envs
from sglang.srt.eplb.expert_distribution import (
    get_global_expert_distribution_recorder,
)
from sglang.srt.eplb.expert_location_dispatch import (
    ExpertLocationDispatchInfo,
    topk_ids_logical_to_physical,
)
from sglang.srt.layers.moe.topk import (
    _RENORMALIZE_SUM_EPSILON,
    StandardTopKOutput,
    TopKConfig,
    _mask_topk_ids_padded_region,
    _zero_topk_weights_padded_region,
    remap_topk_for_per_rank_shared_slots,
    resolve_routing_inputs,
)
from sglang.srt.layers.moe.utils import has_per_rank_fused_shared_slots
from sglang.srt.runtime_context import get_flags, get_model
from sglang.srt.utils import is_hip, is_npu, is_xpu

logger = logging.getLogger(__name__)

_is_hip = is_hip()
_is_npu = is_npu()
_is_xpu = is_xpu()


class HashTopK(nn.Module):
    def __init__(
        self,
        topk,
        num_experts,
        *,
        vocab_size,
        scoring_func="sqrtsoftplus",
        routed_scaling_factor=1.5,
        apply_routed_scaling_factor_on_output=False,
        layer_id: Optional[int] = None,
        is_draft: Optional[bool] = None,
    ):
        super().__init__()
        self.layer_id = layer_id
        self.is_draft = (
            get_flags().moe.in_speculative_scope if is_draft is None else is_draft
        )
        # See TopK: the MoE load balancer routes after TopK on target layers.
        from sglang.srt.eplb.moe_load_balancer_glue import (
            get_configured_moe_load_balancer_pipeline,
        )

        pipeline = get_configured_moe_load_balancer_pipeline()
        self.mlb_requires_post_topk_routing = (
            pipeline is not None
            and pipeline.capabilities.requires_post_topk_routing
            and not self.is_draft
        )
        self.moe_load_balancer = None

        self.shared_experts = get_model().shared_experts
        self.enable_waterfill = self.shared_experts.use_waterfill or (
            self.is_draft and self.shared_experts.routed_by_load_balancer
        )
        self.waterfill_balancer = None

        self.num_experts = num_experts
        # topk counts routed experts; the trailing columns the hash router fills
        # with fused shared experts follow them.
        self.topk = topk + self.num_shared_topk_slots
        self.routed_scaling_factor = routed_scaling_factor
        self.score_func = scoring_func
        self.tid2eid = nn.Parameter(
            torch.empty(vocab_size, topk, dtype=torch.int32),
            requires_grad=False,
        )
        self._init_default_tid2eid()

        self.apply_routed_scaling_factor_on_output = (
            apply_routed_scaling_factor_on_output
        )
        if apply_routed_scaling_factor_on_output and self.num_shared_topk_slots > 0:
            raise NotImplementedError(
                "HashTopK + apply_routed_scaling_factor_on_output is not supported "
                "with fused shared experts; pass --disable-shared-experts-fusion."
            )

    def routing_inputs(self):
        return resolve_routing_inputs(layer_id=self.layer_id, is_draft=self.is_draft)

    @property
    def num_shared_topk_slots(self) -> int:
        return self.shared_experts.num_shared_topk_slots

    def _init_default_tid2eid(self) -> None:
        topk = self.tid2eid.shape[1]
        if topk == 0:
            return

        # DummyModelLoader only initializes floating tensors, so keep this int
        # lookup table valid until real checkpoints overwrite it.
        token_ids = torch.arange(
            self.tid2eid.shape[0], dtype=self.tid2eid.dtype, device=self.tid2eid.device
        ).unsqueeze(1)
        expert_offsets = torch.arange(
            topk, dtype=self.tid2eid.dtype, device=self.tid2eid.device
        ).unsqueeze(0)
        tid2eid = (token_ids + expert_offsets) % self.num_experts
        with torch.no_grad():
            self.tid2eid.copy_(tid2eid.to(self.tid2eid.dtype))

    def empty_topk_output(
        self, device: torch.device, *, layer_id: Optional[int] = None
    ):
        topk = self.topk - self.num_shared_topk_slots
        if not self.is_draft:
            layer_id = self.layer_id
            from sglang.srt.eplb.lplb_solver import get_global_lplb_solver

            lplb_solver = get_global_lplb_solver(layer_id)
            if lplb_solver is not None:
                lplb_solver.solve(
                    torch.empty((0, topk), dtype=torch.int32, device=device)
                )
        topk_weights = torch.empty((0, topk), dtype=torch.float32, device=device)
        topk_ids = torch.full((0, topk), -1, dtype=torch.int32, device=device)
        router_logits = torch.empty((0, topk), dtype=torch.float32, device=device)
        topk_output = StandardTopKOutput(topk_weights, topk_ids, router_logits)
        if has_per_rank_fused_shared_slots(self.num_shared_topk_slots):
            n = self.num_shared_topk_slots
            topk_output = topk_output._replace(
                topk_ids=topk_output.topk_ids.new_empty(
                    (0, topk_output.topk_ids.shape[-1] + n)
                ),
                topk_weights=topk_output.topk_weights.new_empty(
                    (0, topk_output.topk_weights.shape[-1] + n)
                ),
            )
        return self._apply_moe_load_balancer(topk_output, num_tokens=0)

    def _apply_moe_load_balancer(
        self,
        topk_output: StandardTopKOutput,
        num_tokens: int,
        *,
        num_token_non_padded: Optional[torch.Tensor] = None,
    ) -> StandardTopKOutput:
        if self.moe_load_balancer is None:
            if self.mlb_requires_post_topk_routing:
                raise RuntimeError(
                    "MoE load balancer routing is enabled but ModelRunner did not "
                    "attach MoELoadBalancer to this HashTopK."
                )
            return self._apply_waterfill(topk_output, num_tokens)

        from sglang.srt.eplb.moe_load_balancer_glue import route_topk_with_mlb

        return route_topk_with_mlb(
            moe_load_balancer=self.moe_load_balancer,
            layer_id=self.layer_id,
            topk_output=topk_output,
            num_tokens=num_tokens,
            num_token_non_padded=num_token_non_padded,
            routed_scaling_factor=self.routed_scaling_factor,
        )

    def _apply_waterfill(
        self, topk_output: StandardTopKOutput, num_tokens: int
    ) -> StandardTopKOutput:
        if self.enable_waterfill and self.waterfill_balancer is None:
            raise RuntimeError(
                "Waterfill HashTopK must be prepared by ModelRunner before forward."
            )
        if self.waterfill_balancer is None:
            return topk_output
        return self.waterfill_balancer.expand_topk(topk_output, num_tokens)

    def _forward_torch(
        self, router_logits: torch.Tensor, input_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.score_func == "softmax":
            scores = router_logits.softmax(dim=-1)
        elif self.score_func == "sigmoid":
            scores = router_logits.sigmoid()
        else:
            scores = torch.nn.functional.softplus(router_logits).sqrt()

        num_token = scores.shape[0]

        topk_ids = torch.zeros(
            (num_token, self.topk), dtype=torch.int32, device=scores.device
        )
        topk_weights = torch.zeros(
            (num_token, self.topk), dtype=scores.dtype, device=scores.device
        )

        if self.num_shared_topk_slots == 1:
            topk_ids[:, :-1] = self.tid2eid[input_ids]
            topk_weights[:, :-1] = scores.gather(1, topk_ids[:, :-1])

            if self.score_func != "softmax":
                topk_weights[:, :-1] /= (
                    topk_weights[:, :-1].sum(dim=-1, keepdim=True)
                    + _RENORMALIZE_SUM_EPSILON
                )

            topk_ids[:, -1] = torch.randint(
                low=self.num_experts,
                high=self.num_experts + self.num_shared_topk_slots,
                size=(num_token,),
                dtype=topk_ids.dtype,
                device=topk_ids.device,
            )

            topk_weights[:, -1] = (
                topk_weights[:, :-1].sum(dim=-1) / self.routed_scaling_factor
            )
        else:
            topk_ids[:, :] = self.tid2eid[input_ids]
            topk_weights[:, :] = scores.gather(1, topk_ids[:, :])
            if self.score_func != "softmax":
                topk_weights[:, :] /= (
                    topk_weights[:, :].sum(dim=-1, keepdim=True)
                    + _RENORMALIZE_SUM_EPSILON
                )

        return topk_weights, topk_ids

    def _forward_xpu(
        self, router_logits: torch.Tensor, input_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # The XPU 'hash_topk' kernel currently supports the 'sqrtsoftplus' score func only.
        # Other score funcs fall back to the torch implementation; more will be supported in the future.
        if self.score_func == "sqrtsoftplus":
            from sgl_kernel import hash_topk

            num_tokens = router_logits.size(0)
            topk_routed = self.tid2eid.size(1)
            topk_fused = topk_routed + self.num_shared_topk_slots
            topk_ids = torch.empty(
                (num_tokens, topk_fused), dtype=torch.int32, device=router_logits.device
            )
            topk_weights = torch.empty(
                (num_tokens, topk_fused),
                dtype=torch.float32,
                device=router_logits.device,
            )
            hash_topk(
                router_logits,
                input_ids,
                self.tid2eid,
                topk_weights,
                topk_ids,
                self.routed_scaling_factor,
                self.score_func,
            )
            return topk_weights, topk_ids
        else:
            return self._forward_torch(router_logits, input_ids)

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor,
        num_token_non_padded: Optional[torch.Tensor] = None,
        expert_location_dispatch_info: Optional[ExpertLocationDispatchInfo] = None,
    ):
        num_token_non_padded, expert_location_dispatch_info = resolve_routing_inputs(
            layer_id=self.layer_id,
            is_draft=self.is_draft,
            num_token_non_padded=num_token_non_padded,
            expert_location_dispatch_info=expert_location_dispatch_info,
        )
        assert input_ids.shape[0] == hidden_states.shape[0] == router_logits.shape[0], (
            f"{input_ids.shape=} {hidden_states.shape=} {router_logits.shape=}"
        )

        if _is_xpu:
            topk_weights, topk_ids = self._forward_xpu(router_logits, input_ids)
        elif envs.SGLANG_OPT_USE_FUSED_HASH_TOPK.get():
            from sglang.kernels.ops.moe.dsv4 import hash_topk

            topk_weights, topk_ids = hash_topk(
                router_logits=router_logits,
                input_ids=input_ids,
                tid2eid=self.tid2eid,
                num_fused_shared_experts=self.num_shared_topk_slots,
                routed_scaling_factor=self.routed_scaling_factor,
                scoring_func=self.score_func,
            )
        else:
            topk_weights, topk_ids = self._forward_torch(router_logits, input_ids)
        if _is_hip or _is_npu:
            topk_weights = topk_weights.to(torch.float32)

        if self.apply_routed_scaling_factor_on_output:
            topk_weights = topk_weights * self.routed_scaling_factor

        if self.moe_load_balancer is not None:
            # The load balancer maps logical ids to physical replicas and
            # records them itself (moe_load_balancer_glue).
            expert_location_dispatch_info = None
        num_shared_topk_slots = self.num_shared_topk_slots
        routed_ids = (
            topk_ids[:, :-num_shared_topk_slots] if num_shared_topk_slots else topk_ids
        )
        log2phy_prob = None
        if (
            expert_location_dispatch_info is not None
            and expert_location_dispatch_info.ep_dispatch_algorithm == "lp"
        ):
            if self.layer_id is None:
                raise RuntimeError("HashTopK LP dispatch requires layer_id.")
            from sglang.srt.eplb.lplb_solver import get_global_lplb_solver

            lplb_solver = get_global_lplb_solver(self.layer_id)
            if lplb_solver is not None:
                log2phy_prob = lplb_solver.solve(routed_ids)

        routed_cols = topk_ids_logical_to_physical(
            routed_ids, expert_location_dispatch_info, log2phy_prob
        )
        recorder_topk_ids = routed_cols if num_shared_topk_slots else None
        if has_per_rank_fused_shared_slots(num_shared_topk_slots):
            num_physical_routed_experts = (
                expert_location_dispatch_info.num_physical_experts
                if expert_location_dispatch_info is not None
                else self.num_experts
            )
            topk_ids, topk_weights = remap_topk_for_per_rank_shared_slots(
                routed_cols,
                topk_weights[:, :-num_shared_topk_slots],
                num_shared_topk_slots,
                num_physical_routed_experts,
                TopKConfig(
                    top_k=self.topk,
                    shared_experts=self.shared_experts,
                    routed_scaling_factor=self.routed_scaling_factor,
                    apply_routed_scaling_factor_on_output=self.apply_routed_scaling_factor_on_output,
                ),
            )
        elif num_shared_topk_slots:
            topk_ids = torch.cat(
                [routed_cols, topk_ids[:, -num_shared_topk_slots:]], dim=-1
            )
        else:
            topk_ids = routed_cols
        if _is_hip or _is_npu:
            _zero_topk_weights_padded_region(topk_weights, num_token_non_padded)
        else:
            _mask_topk_ids_padded_region(topk_ids, num_token_non_padded)
            if recorder_topk_ids is not None:
                _mask_topk_ids_padded_region(recorder_topk_ids, num_token_non_padded)
        if recorder_topk_ids is None:
            recorder_topk_ids = topk_ids
        if _is_npu and num_token_non_padded is not None:
            rows = torch.arange(topk_ids.shape[0], device=topk_ids.device)
            recorder_topk_ids = recorder_topk_ids.masked_fill(
                (rows >= num_token_non_padded)[:, None], -1
            )
        if self.moe_load_balancer is None:
            get_global_expert_distribution_recorder().on_select_experts(
                topk_ids=recorder_topk_ids
            )
        topk_output = StandardTopKOutput(
            topk_weights=topk_weights, topk_ids=topk_ids, router_logits=router_logits
        )
        topk_output = self._apply_moe_load_balancer(
            topk_output,
            hidden_states.shape[0],
            num_token_non_padded=num_token_non_padded,
        )
        if _is_hip or _is_npu:
            _zero_topk_weights_padded_region(
                topk_output.topk_weights, num_token_non_padded
            )
        return topk_output
