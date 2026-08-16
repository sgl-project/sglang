# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

# gate * sigmoid(alpha * gate) * (up + 1), gate clamped above and up both ways;
# swiglu_no_interleaved_with_alpha_and_limit in the Triton runner.
SWIGLU7_ALPHA = 1.702
SWIGLU7_LIMIT = 7.0

# The reference router L1-normalizes with this epsilon. sglang's single-head
# router uses 1e-20, which is a visible difference at bf16 score magnitudes.
ROUTE_NORM_EPS = 1e-12


class Magi2MultiHeadRouter(nn.Module):
    """``gate`` is ``[num_heads * num_experts, head_dim]``; returned ids are flattened global ids."""

    def __init__(
        self,
        *,
        num_heads: int,
        num_experts: int,
        head_dim: int,
        top_k: int,
        route_scale: float,
        score_func: str = "sigmoid",
        route_norm: bool = True,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.num_experts = num_experts
        self.head_dim = head_dim
        self.top_k = top_k
        self.route_scale = route_scale
        self.score_func = score_func
        self.route_norm = route_norm

        self.gate = nn.Parameter(
            torch.empty(num_heads * num_experts, head_dim, dtype=torch.float32)
        )
        # Aux-loss-free balancing: steers selection only, never the returned weights.
        self.register_buffer(
            "expert_bias",
            torch.zeros(num_heads * num_experts, dtype=torch.float32),
            persistent=True,
        )

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Rows are token-major so a later view collapses the head axis back into the hidden axis."""
        num_tokens = tokens.shape[0]
        heads, experts = self.num_heads, self.num_experts

        with torch.amp.autocast(tokens.device.type, enabled=False):
            per_head = tokens.float().transpose(0, 1)
            gate = self.gate.view(heads, experts, self.head_dim).transpose(1, 2)
            logits = torch.bmm(per_head, gate.float())

        scores = (
            F.softmax(logits, dim=-1)
            if self.score_func == "softmax"
            else logits.sigmoid()
        )

        biased = scores + self.expert_bias.view(heads, 1, experts)
        topk_ids = torch.topk(biased, k=self.top_k, dim=-1, sorted=False)[1]
        topk_weights = scores.gather(-1, topk_ids)
        if self.route_norm and self.top_k > 1:
            topk_weights = F.normalize(topk_weights, p=1, dim=-1, eps=ROUTE_NORM_EPS)
        topk_weights = topk_weights * self.route_scale

        head_offset = (
            torch.arange(heads, device=tokens.device, dtype=topk_ids.dtype) * experts
        )
        topk_ids = topk_ids + head_offset.view(heads, 1, 1)

        topk_ids = topk_ids.transpose(0, 1).reshape(num_tokens * heads, self.top_k)
        topk_weights = topk_weights.transpose(0, 1).reshape(
            num_tokens * heads, self.top_k
        )
        return topk_ids, topk_weights


class Magi2MultiHeadExperts(nn.Module):
    """Weight layout follows sglang's grouped-expert convention, gate half first, so ``fused_experts`` runs unmodified."""

    def __init__(
        self,
        *,
        num_local_experts: int,
        head_dim: int,
        intermediate_size: int,
    ) -> None:
        super().__init__()
        self.num_local_experts = num_local_experts
        self.head_dim = head_dim
        self.intermediate_size = intermediate_size

        self.w13_weight = nn.Parameter(
            torch.empty(num_local_experts, 2 * intermediate_size, head_dim)
        )
        self.w2 = nn.Parameter(
            torch.empty(num_local_experts, head_dim, intermediate_size)
        )

    def forward(
        self,
        tokens: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        from sglang.srt.layers.moe.moe_runner import MoeRunnerConfig
        from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import (
            fused_experts,
        )
        from sglang.srt.layers.moe.topk import StandardTopKOutput

        topk_output = StandardTopKOutput(
            topk_weights=topk_weights.float(),
            topk_ids=topk_ids.to(torch.int32),
            router_logits=torch.empty(0, device=tokens.device),
        )
        runner_config = MoeRunnerConfig(
            # Equal counts: ids are already rank-local, so the kernel's expert
            # filter has nothing to drop.
            num_experts=self.num_local_experts,
            num_local_experts=self.num_local_experts,
            hidden_size=self.head_dim,
            intermediate_size_per_partition=self.intermediate_size,
            top_k=topk_ids.shape[-1],
            activation="silu",
            is_gated=True,
            inplace=False,
            apply_router_weight_on_input=False,
            # The router already applied route_scale; setting this double-applies it.
            routed_scaling_factor=None,
            # gate and up are separate accumulators, not interleaved columns.
            gate_up_interleaved=False,
            gemm1_alpha=SWIGLU7_ALPHA,
            gemm1_clamp_limit=SWIGLU7_LIMIT,
        )
        return fused_experts(
            tokens.contiguous().bfloat16(),
            self.w13_weight.bfloat16(),
            self.w2.bfloat16(),
            topk_output,
            runner_config,
        ).type_as(tokens)


class Magi2MultiHeadMoE(nn.Module):
    """A rank owns a contiguous run of whole heads, keeping top-k rank-local so no expert score crosses a rank."""

    def __init__(
        self,
        *,
        num_heads: int,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        top_k: int,
        route_scale: float,
        score_func: str = "sigmoid",
        route_norm: bool = True,
        ep_group: dist.ProcessGroup | None = None,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError(
                f"hidden_size {hidden_size} must be divisible by num_heads "
                f"{num_heads}"
            )

        self.ep_group = ep_group
        self.ep_size = 1 if ep_group is None else dist.get_world_size(ep_group)
        self.ep_rank = 0 if ep_group is None else dist.get_rank(ep_group)

        if num_heads % self.ep_size:
            raise ValueError(
                f"num_heads {num_heads} must be divisible by ep_size "
                f"{self.ep_size}; a rank owns whole heads"
            )

        self.num_heads = num_heads
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // num_heads
        self.local_num_heads = num_heads // self.ep_size
        self.head_start = self.ep_rank * self.local_num_heads

        self.router = Magi2MultiHeadRouter(
            num_heads=self.local_num_heads,
            num_experts=num_experts,
            head_dim=self.head_dim,
            top_k=top_k,
            route_scale=route_scale,
            score_func=score_func,
            route_norm=route_norm,
        )
        self.experts = Magi2MultiHeadExperts(
            num_local_experts=self.local_num_heads * num_experts,
            head_dim=self.head_dim,
            intermediate_size=intermediate_size,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens = hidden_states.shape[0]

        owned = self._gather_owned_heads(hidden_states)
        topk_ids, topk_weights = self.router(owned)
        flat = owned.reshape(-1, self.head_dim)
        out = self.experts(flat, topk_ids, topk_weights)
        out = out.view(owned.shape[0], self.local_num_heads, self.head_dim)
        return self._scatter_owned_heads(out, num_tokens=num_tokens)

    def _gather_owned_heads(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Trade the sequence shard for a head shard, giving ``[num_tokens * ep_size, local_num_heads, head_dim]``."""
        heads = hidden_states.view(-1, self.num_heads, self.head_dim)
        if self.ep_size == 1:
            return heads

        num_tokens = heads.shape[0]
        # Rank must lead: all_to_all_single splits on the leading axis.
        send = (
            heads.view(num_tokens, self.ep_size, self.local_num_heads, self.head_dim)
            .transpose(0, 1)
            .contiguous()
        )
        recv = torch.empty_like(send)
        dist.all_to_all_single(recv, send, group=self.ep_group)
        # recv[s] holds rank s's tokens for the heads this rank owns.
        return recv.reshape(
            self.ep_size * num_tokens, self.local_num_heads, self.head_dim
        )

    def _scatter_owned_heads(
        self, out: torch.Tensor, *, num_tokens: int
    ) -> torch.Tensor:
        if self.ep_size == 1:
            return out.reshape(num_tokens, self.hidden_size)

        send = out.view(
            self.ep_size, num_tokens, self.local_num_heads, self.head_dim
        ).contiguous()
        recv = torch.empty_like(send)
        dist.all_to_all_single(recv, send, group=self.ep_group)
        # recv[s] is rank s's heads for this rank's tokens; ranks own heads in
        # ascending order, so stacking restores head order 0..H-1.
        return recv.transpose(0, 1).contiguous().reshape(num_tokens, self.hidden_size)
