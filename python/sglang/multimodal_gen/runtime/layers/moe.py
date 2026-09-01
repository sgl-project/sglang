# Adapted from LingBot-Video (https://github.com/Robbyant/lingbot-video).
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import msgspec
import torch
import torch.nn.functional as F
from torch import nn

from sglang.multimodal_gen.runtime.distributed import (
    get_sp_parallel_rank,
    get_sp_world_size,
)
from sglang.multimodal_gen.runtime.layers.moe_token_dispatcher import (
    FixedCapacityMoeTokenDispatcher,
    create_moe_token_dispatcher,
)


class MoeExpertParallelInfo(msgspec.Struct, frozen=True):
    ep_size: int
    ep_rank: int
    num_local_experts: int
    local_expert_start: int

    @property
    def enabled(self) -> bool:
        return self.ep_size > 1


def resolve_moe_expert_parallel(
    num_experts: int, *, ep_size: int
) -> MoeExpertParallelInfo:
    """Resolve this rank's contiguous expert shard."""
    if ep_size < 1:
        raise ValueError(f"ep_size ({ep_size}) must be >= 1")
    if num_experts % ep_size != 0:
        raise ValueError(
            f"num_experts ({num_experts}) must be divisible by ep_size ({ep_size})"
        )
    if ep_size > 1:
        sp_world_size = get_sp_world_size()
        if sp_world_size != ep_size:
            raise RuntimeError(
                f"configured ep_size ({ep_size}) does not match the SP world size "
                f"({sp_world_size})"
            )
    num_local_experts = num_experts // ep_size
    ep_rank = get_sp_parallel_rank() if ep_size > 1 else 0
    return MoeExpertParallelInfo(
        ep_size=ep_size,
        ep_rank=ep_rank,
        num_local_experts=num_local_experts,
        local_expert_start=ep_rank * num_local_experts,
    )


def _reduce_unweighted_expert_outputs(
    expert_outputs: torch.Tensor,
    top_scores: torch.Tensor,
    top_indices: torch.Tensor,
) -> torch.Tensor:
    """Reduce valid expert outputs in FP32 without reading non-local slots."""
    from sglang.kernels.ops.moe.moe_fused_mul_sum import moe_fused_mul_sum

    num_tokens, _, hidden = expert_outputs.shape
    out = torch.empty(
        num_tokens, hidden, dtype=torch.float32, device=expert_outputs.device
    )
    return moe_fused_mul_sum(
        expert_outputs.contiguous(),
        top_scores.contiguous(),
        outputs=out,
        topk_ids=top_indices.to(torch.int32).contiguous(),
        is_ep=True,
    )


class LingBotVideoMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class LingBotVideoRouter(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int,
        score_func: str,
        norm_topk_prob: bool,
        n_group: int | None,
        topk_group: int | None,
        route_scale: float,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.score_func = score_func
        self.norm_topk_prob = norm_topk_prob
        self.n_group = n_group
        self.topk_group = topk_group
        self.route_scale = route_scale
        self.weight = nn.Parameter(torch.empty(num_experts, hidden_size))
        self.register_buffer(
            "e_score_correction_bias", torch.zeros(num_experts), persistent=True
        )

    def _group_limited_topk(self, scores_for_choice: torch.Tensor) -> torch.Tensor:
        seq_len = scores_for_choice.shape[0]
        experts_per_group = self.num_experts // self.n_group
        grouped = scores_for_choice.view(seq_len, self.n_group, experts_per_group)
        group_scores = grouped.topk(2, dim=-1)[0].sum(dim=-1)
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(seq_len, self.n_group, experts_per_group)
            .reshape(seq_len, -1)
        )
        masked = scores_for_choice.masked_fill(~score_mask.bool(), float("-inf"))
        return torch.topk(masked, k=self.top_k, dim=-1, sorted=False)[1]

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.amp.autocast(tokens.device.type, enabled=False):
            logits = F.linear(tokens.float(), self.weight.float())
        if self.score_func == "softmax":
            scores = F.softmax(logits, dim=-1)
        else:
            scores = logits.sigmoid()
        scores_for_choice = scores + self.e_score_correction_bias.unsqueeze(0)
        if self.n_group is not None and self.n_group > 1:
            top_indices = self._group_limited_topk(scores_for_choice)
        else:
            top_indices = torch.topk(
                scores_for_choice, k=self.top_k, dim=-1, sorted=False
            )[1]
        top_scores = scores.gather(1, top_indices)
        if self.top_k > 1 and self.norm_topk_prob:
            top_scores = top_scores / (top_scores.sum(dim=-1, keepdim=True) + 1e-20)
        top_scores = top_scores * self.route_scale
        return top_indices, top_scores.to(tokens.dtype)


class LingBotVideoGroupedExperts(nn.Module):
    def __init__(
        self,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        num_local_experts: int | None = None,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.num_local_experts = (
            num_experts if num_local_experts is None else num_local_experts
        )
        self.w13_weight = nn.Parameter(
            torch.empty(self.num_local_experts, 2 * intermediate_size, hidden_size)
        )
        self.w2 = nn.Parameter(
            torch.empty(self.num_local_experts, hidden_size, intermediate_size)
        )


class LingBotVideoSparseMoeBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        top_k: int,
        score_func: str,
        norm_topk_prob: bool,
        n_group: int | None,
        topk_group: int | None,
        routed_scaling_factor: float,
        n_shared_experts: int | None,
        ep_info: MoeExpertParallelInfo | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.intermediate_size = intermediate_size
        self.ep_info = ep_info or MoeExpertParallelInfo(
            ep_size=1,
            ep_rank=0,
            num_local_experts=num_experts,
            local_expert_start=0,
        )
        # Dynamo cannot trace a msgspec.Struct property in the compiled expert call.
        self.num_local_experts = self.ep_info.num_local_experts
        self.token_dispatcher: FixedCapacityMoeTokenDispatcher | None = (
            create_moe_token_dispatcher(
                ep_size=self.ep_info.ep_size,
                num_local_experts=self.ep_info.num_local_experts,
            )
        )
        self.router = LingBotVideoRouter(
            hidden_size,
            num_experts,
            top_k,
            score_func,
            norm_topk_prob,
            n_group,
            topk_group,
            routed_scaling_factor,
        )
        self.experts = LingBotVideoGroupedExperts(
            num_experts,
            hidden_size,
            intermediate_size,
            num_local_experts=self.ep_info.num_local_experts,
        )
        self.shared_experts: LingBotVideoMLP | None = None
        if n_shared_experts is not None and n_shared_experts > 0:
            self.shared_experts = LingBotVideoMLP(
                hidden_size, intermediate_size * n_shared_experts
            )

    def _run_sglang_triton_experts(
        self,
        tokens: torch.Tensor,
        top_scores: torch.Tensor,
        top_indices: torch.Tensor,
        no_combine: bool = False,
    ) -> torch.Tensor:
        from sglang.srt.layers.moe.moe_runner import MoeRunnerConfig
        from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import (
            fused_experts,
        )
        from sglang.srt.layers.moe.topk import StandardTopKOutput

        topk_output = StandardTopKOutput(
            topk_weights=top_scores.float(),
            topk_ids=top_indices.to(torch.int32),
            router_logits=torch.empty(0, device=tokens.device),
        )
        # Router pre-scales the topk scores; fused_experts must not apply routed_scaling_factor.
        runner_config = MoeRunnerConfig(
            # Filtered routing needs the global expert count to accept -1 routes.
            num_experts=self.num_experts,
            num_local_experts=self.num_local_experts,
            hidden_size=self.hidden_size,
            intermediate_size_per_partition=self.intermediate_size,
            top_k=self.top_k,
            activation="silu",
            is_gated=True,
            inplace=False,
            apply_router_weight_on_input=False,
            routed_scaling_factor=None,
            gate_up_interleaved=False,
            no_combine=no_combine,
        )
        out = fused_experts(
            tokens.contiguous().bfloat16(),
            self.experts.w13_weight.bfloat16(),
            self.experts.w2.bfloat16(),
            topk_output,
            runner_config,
        )
        if not no_combine:
            return out.type_as(tokens)
        # no_combine leaves filtered slots uninitialized; the reducer masks them.
        return _reduce_unweighted_expert_outputs(out, top_scores, top_indices)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        b = hidden_states.shape[0]
        tokens = hidden_states.reshape(-1, self.hidden_size)
        top_indices, top_scores = self.router(tokens)
        if self.token_dispatcher is not None:
            dispatched = self.token_dispatcher.dispatch(
                hidden_states=tokens,
                topk_weights=top_scores,
                global_expert_ids=top_indices,
            )
            local_out = self._run_sglang_triton_experts(
                dispatched.hidden_states,
                dispatched.topk_weights,
                dispatched.local_expert_ids,
                no_combine=True,
            )
            out = self.token_dispatcher.combine(
                expert_output=local_out,
                dispatched=dispatched,
            )
            out = out.to(torch.bfloat16).type_as(tokens)
        else:
            out = self._run_sglang_triton_experts(tokens, top_scores, top_indices)
        out = out.reshape(b, -1, self.hidden_size)
        if self.shared_experts is not None:
            out = out + self.shared_experts(hidden_states)
        return out
