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
    StandardTopKOutput,
    TopKConfig,
    _mask_topk_ids_padded_region,
    _zero_topk_weights_padded_region,
    remap_topk_for_per_rank_shared_slots,
)
from sglang.srt.layers.moe.utils import has_per_rank_fused_shared_slots
from sglang.srt.runtime_context import get_exec
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
        num_fused_shared_experts,
        vocab_size,
        scoring_func="sqrtsoftplus",
        routed_scaling_factor=1.5,
        apply_routed_scaling_factor_on_output=False,
        layer_id: Optional[int] = None,
        correction_bias_vl: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.layer_id = layer_id
        # DeepSeek-V4-Vision: image tokens do not hash-route at all -- they take
        # the top-k of scores biased by this second routed-expert bias. Held as a
        # plain attribute (not a submodule parameter) the way TopK holds the text
        # bias: the owner is the gate.
        self.correction_bias_vl = correction_bias_vl

        self.enable_waterfill = (
            num_fused_shared_experts > 0 and get_exec().moe.enable_waterfill
        )
        self.waterfill_balancer = None

        if self.enable_waterfill:
            # Waterfill appends the shared expert after EPLB maps routed IDs.
            topk -= num_fused_shared_experts
            num_fused_shared_experts = 0

        self.num_experts = num_experts
        self.topk = topk
        self.routed_scaling_factor = routed_scaling_factor
        self.num_fused_shared_experts = num_fused_shared_experts
        self.score_func = scoring_func
        self.tid2eid = nn.Parameter(
            torch.empty(vocab_size, topk - num_fused_shared_experts, dtype=torch.int32),
            requires_grad=False,
        )
        self._init_default_tid2eid()

        self.apply_routed_scaling_factor_on_output = (
            apply_routed_scaling_factor_on_output
        )
        if apply_routed_scaling_factor_on_output and num_fused_shared_experts > 0:
            raise NotImplementedError(
                "HashTopK + apply_routed_scaling_factor_on_output is not supported "
                "with fused shared experts; pass --disable-shared-experts-fusion."
            )

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
        topk = self.topk - self.num_fused_shared_experts
        if layer_id is not None:
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
        if has_per_rank_fused_shared_slots(self.num_fused_shared_experts):
            n = self.num_fused_shared_experts
            topk_output = topk_output._replace(
                topk_ids=topk_output.topk_ids.new_empty(
                    (0, topk_output.topk_ids.shape[-1] + n)
                ),
                topk_weights=topk_output.topk_weights.new_empty(
                    (0, topk_output.topk_weights.shape[-1] + n)
                ),
            )
        return self._apply_waterfill(topk_output, num_tokens=0)

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
        scores = self._scores(router_logits)
        num_token = scores.shape[0]

        topk_ids = torch.zeros(
            (num_token, self.topk), dtype=torch.int32, device=scores.device
        )
        topk_weights = torch.zeros(
            (num_token, self.topk), dtype=scores.dtype, device=scores.device
        )

        if self.num_fused_shared_experts == 1:
            topk_ids[:, :-1] = self.tid2eid[input_ids]
            topk_weights[:, :-1] = scores.gather(1, topk_ids[:, :-1])

            if self.score_func != "softmax":
                topk_weights[:, :-1] /= topk_weights[:, :-1].sum(dim=-1, keepdim=True)

            topk_ids[:, -1] = torch.randint(
                low=self.num_experts,
                high=self.num_experts + self.num_fused_shared_experts,
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
                topk_weights[:, :] /= topk_weights[:, :].sum(dim=-1, keepdim=True)

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
            topk_fused = topk_routed + self.num_fused_shared_experts
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

    def _scores(self, router_logits: torch.Tensor) -> torch.Tensor:
        if self.score_func == "softmax":
            return router_logits.softmax(dim=-1)
        if self.score_func == "sigmoid":
            return router_logits.sigmoid()
        return torch.nn.functional.softplus(router_logits).sqrt()

    def _route_image_tokens(
        self,
        router_logits: torch.Tensor,
        image_mask: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ):
        """Replace the hash routing of image-token rows with vision-biased top-k.

        Only the routed columns are replaced. A fused shared-expert column, when
        present, holds ``1 / routed_scaling_factor`` for every row once the
        routed weights are renormalized, so it needs no per-row fixup.
        """
        assert self.correction_bias_vl is not None
        scores = self._scores(router_logits)
        num_routed = self.topk - self.num_fused_shared_experts
        vl_ids = (scores + self.correction_bias_vl.to(scores.dtype)).topk(
            num_routed, dim=-1, sorted=False
        )[1]
        vl_weights = scores.gather(1, vl_ids)
        if self.score_func != "softmax":
            vl_weights = vl_weights / vl_weights.sum(dim=-1, keepdim=True)

        mask = image_mask.unsqueeze(-1)
        routed = slice(None, num_routed)
        topk_ids[:, routed] = torch.where(
            mask, vl_ids.to(topk_ids.dtype), topk_ids[:, routed]
        )
        topk_weights[:, routed] = torch.where(
            mask, vl_weights.to(topk_weights.dtype), topk_weights[:, routed]
        )
        return topk_weights, topk_ids

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor,
        num_token_non_padded: Optional[torch.Tensor] = None,
        expert_location_dispatch_info: Optional[ExpertLocationDispatchInfo] = None,
        image_mask: Optional[torch.Tensor] = None,
    ):
        assert (
            input_ids.shape[0] == hidden_states.shape[0] == router_logits.shape[0]
        ), f"{input_ids.shape=} {hidden_states.shape=} {router_logits.shape=}"

        if image_mask is not None:
            # Image-token ids are multimodal pad sentinels far above the
            # vocabulary; the table lookup would read out of bounds. Their rows
            # are overwritten by _route_image_tokens below, so any in-range
            # index will do.
            input_ids = torch.where(image_mask, 0, input_ids)

        if _is_xpu:
            topk_weights, topk_ids = self._forward_xpu(router_logits, input_ids)
        elif envs.SGLANG_OPT_USE_FUSED_HASH_TOPK.get():
            from sglang.kernels.ops.attention.dsv4 import hash_topk

            topk_weights, topk_ids = hash_topk(
                router_logits=router_logits,
                input_ids=input_ids,
                tid2eid=self.tid2eid,
                num_fused_shared_experts=self.num_fused_shared_experts,
                routed_scaling_factor=self.routed_scaling_factor,
                scoring_func=self.score_func,
            )
        else:
            topk_weights, topk_ids = self._forward_torch(router_logits, input_ids)
        if image_mask is not None:
            topk_weights, topk_ids = self._route_image_tokens(
                router_logits, image_mask, topk_weights, topk_ids
            )
        if _is_hip or _is_npu:
            topk_weights = topk_weights.to(torch.float32)

        if self.apply_routed_scaling_factor_on_output:
            topk_weights = topk_weights * self.routed_scaling_factor

        num_fused_shared_experts = self.num_fused_shared_experts
        log2phy_prob = None
        if (
            expert_location_dispatch_info is not None
            and getattr(expert_location_dispatch_info, "ep_dispatch_algorithm", None)
            == "lp"
        ):
            if self.layer_id is None:
                raise RuntimeError("HashTopK LP dispatch requires layer_id.")
            from sglang.srt.eplb.lplb_solver import get_global_lplb_solver

            lplb_solver = get_global_lplb_solver(self.layer_id)
            if lplb_solver is not None:
                log2phy_prob = lplb_solver.solve(topk_ids)

        recorder_topk_ids = None
        if has_per_rank_fused_shared_slots(num_fused_shared_experts):
            shared_cols = topk_ids[:, -num_fused_shared_experts:]
            routed_cols = topk_ids[:, :-num_fused_shared_experts]
            routed_cols = topk_ids_logical_to_physical(
                routed_cols, expert_location_dispatch_info, log2phy_prob
            )
            topk_ids = torch.cat([routed_cols, shared_cols], dim=-1)
            recorder_topk_ids = routed_cols

            num_physical_routed_experts = (
                expert_location_dispatch_info.num_physical_experts
                if expert_location_dispatch_info is not None
                else self.num_experts
            )
            topk_ids, topk_weights = remap_topk_for_per_rank_shared_slots(
                topk_ids,
                topk_weights,
                num_fused_shared_experts,
                num_physical_routed_experts,
                TopKConfig(
                    top_k=self.topk,
                    num_fused_shared_experts=num_fused_shared_experts,
                    routed_scaling_factor=self.routed_scaling_factor,
                ),
            )
        else:
            topk_ids = topk_ids_logical_to_physical(
                topk_ids, expert_location_dispatch_info, log2phy_prob
            )
        if is_hip():
            _zero_topk_weights_padded_region(topk_weights, num_token_non_padded)
        else:
            _mask_topk_ids_padded_region(topk_ids, num_token_non_padded)
            if recorder_topk_ids is not None:
                _mask_topk_ids_padded_region(recorder_topk_ids, num_token_non_padded)
        if recorder_topk_ids is None:
            recorder_topk_ids = topk_ids
        get_global_expert_distribution_recorder().on_select_experts(
            topk_ids=recorder_topk_ids
        )
        topk_output = StandardTopKOutput(
            topk_weights=topk_weights, topk_ids=topk_ids, router_logits=router_logits
        )
        topk_output = self._apply_waterfill(topk_output, hidden_states.shape[0])
        if is_hip():
            _zero_topk_weights_padded_region(
                topk_output.topk_weights, num_token_non_padded
            )
        return topk_output
