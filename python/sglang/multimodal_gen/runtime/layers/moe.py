# Adapted from LingBot-Video (https://github.com/Robbyant/lingbot-video).
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from sglang.multimodal_gen.runtime.layers.quantization.configs.base_config import (
    QuantizationConfig,
)
from sglang.srt.layers.moe.moe_runner import MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import fused_experts
from sglang.srt.layers.moe.topk import StandardTopKOutput

FP8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max


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


def _quantize_per_channel(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """One fp8 scale per output row of each expert, the layout ``per_channel_quant`` wants."""
    scale = weight.float().abs().amax(dim=2).clamp(min=1e-12) / FP8_E4M3_MAX
    quantized = (weight.float() / scale[:, :, None]).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
    return quantized.to(torch.float8_e4m3fn), scale


class LingBotVideoGroupedExperts(nn.Module):
    def __init__(
        self, num_experts: int, hidden_size: int, intermediate_size: int
    ) -> None:
        super().__init__()
        self.w13_weight = nn.Parameter(
            torch.empty(num_experts, 2 * intermediate_size, hidden_size)
        )
        self.w2 = nn.Parameter(torch.empty(num_experts, hidden_size, intermediate_size))

    def quantize_to_fp8(self) -> None:
        w13, w13_scale = _quantize_per_channel(self.w13_weight.data)
        w2, w2_scale = _quantize_per_channel(self.w2.data)
        self.w13_weight = nn.Parameter(w13, requires_grad=False)
        self.w2 = nn.Parameter(w2, requires_grad=False)
        self.register_parameter(
            "w13_weight_scale", nn.Parameter(w13_scale, requires_grad=False)
        )
        self.register_parameter(
            "w2_weight_scale", nn.Parameter(w2_scale, requires_grad=False)
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
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.intermediate_size = intermediate_size
        # The experts are bare Parameters, so get_quant_method never sees them and
        # the generic post-load walk cannot quantize them; post_load_weights does.
        self.quantize_experts_to_fp8 = (
            quant_config is not None and quant_config.get_name() == "fp8"
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
            num_experts, hidden_size, intermediate_size
        )
        self.shared_experts: LingBotVideoMLP | None = None
        if n_shared_experts is not None and n_shared_experts > 0:
            self.shared_experts = LingBotVideoMLP(
                hidden_size, intermediate_size * n_shared_experts
            )
        # Router pre-scales the topk scores, so routed_scaling_factor stays unset.
        self.runner_config = MoeRunnerConfig(
            num_experts=num_experts,
            num_local_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size_per_partition=intermediate_size,
            top_k=top_k,
            activation="silu",
            is_gated=True,
            inplace=False,
            apply_router_weight_on_input=False,
            routed_scaling_factor=None,
            gate_up_interleaved=False,
        )

    def _run_sglang_triton_experts(
        self,
        tokens: torch.Tensor,
        top_scores: torch.Tensor,
        top_indices: torch.Tensor,
    ) -> torch.Tensor:
        topk_output = StandardTopKOutput(
            topk_weights=top_scores.float(),
            topk_ids=top_indices.to(torch.int32),
            router_logits=torch.empty(0, device=tokens.device),
        )
        if self.experts.w13_weight.dtype == torch.float8_e4m3fn:
            return fused_experts(
                tokens.contiguous().bfloat16(),
                self.experts.w13_weight,
                self.experts.w2,
                topk_output,
                self.runner_config,
                use_fp8_w8a8=True,
                w1_scale=self.experts.w13_weight_scale,
                w2_scale=self.experts.w2_weight_scale,
                per_channel_quant=True,
            ).type_as(tokens)
        return fused_experts(
            tokens.contiguous().bfloat16(),
            self.experts.w13_weight.bfloat16(),
            self.experts.w2.bfloat16(),
            topk_output,
            self.runner_config,
        ).type_as(tokens)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        b = hidden_states.shape[0]
        tokens = hidden_states.reshape(-1, self.hidden_size)
        top_indices, top_scores = self.router(tokens)
        out = self._run_sglang_triton_experts(tokens, top_scores, top_indices)
        out = out.reshape(b, -1, self.hidden_size)
        if self.shared_experts is not None:
            out = out + self.shared_experts(hidden_states)
        return out
