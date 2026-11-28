# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import functools

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from sglang.multimodal_gen import envs

# gate * sigmoid(alpha * gate) * (up + 1), gate clamped above and up both ways;
# swiglu_no_interleaved_with_alpha_and_limit in the Triton runner.
SWIGLU7_ALPHA = 1.702
SWIGLU7_LIMIT = 7.0

# The reference router L1-normalizes with this epsilon. sglang's single-head
# router uses 1e-20, which is a visible difference at bf16 score magnitudes.
ROUTE_NORM_EPS = 1e-12

# moe_align_block_size runs 1024 threads and writes prefix[num_experts], so more
# than this takes an illegal memory access. Experts run in head chunks that fit.
MAX_KERNEL_EXPERTS = 1023


@functools.lru_cache(maxsize=None)
def _exchange_group_is_nvlink(group: dist.ProcessGroup) -> bool:
    """Whether every pair in ``group`` is NVLink-connected; identical on every rank.

    Peers, not the node: a repeated device id means the group crosses hosts, so
    its exchange is not an NVLink one.
    """
    from sglang.multimodal_gen.runtime.platforms import current_platform

    if not current_platform.is_cuda():
        return False
    from sglang.multimodal_gen.runtime.platforms.cuda import (
        device_id_to_physical_device_id,
    )

    own = torch.tensor(
        [device_id_to_physical_device_id(torch.cuda.current_device())],
        dtype=torch.int64,
        device="cuda",
    )
    peers = [torch.zeros_like(own) for _ in range(dist.get_world_size(group=group))]
    dist.all_gather(peers, own, group=group)
    devices = [int(peer.item()) for peer in peers]
    if len(set(devices)) != len(devices):
        return False
    return current_platform.is_full_nvlink(sorted(devices))


def resolve_pipeline_groups(
    *, local_heads: int, preferred: int, group: dist.ProcessGroup | None
) -> int:
    """Head groups an exchange over ``local_heads`` is split into; 1 keeps it sequential.

    0 or 1 is off, N >= 2 is forced, and -1 picks ``preferred`` unless the
    exchange's own peers are fully NVLink-connected.
    """
    requested = envs.SGLANG_DIFFUSION_ULYSSES_PIPELINE_GROUPS
    # The exchanges are not traceable, so a compiling block keeps the
    # sequential path, as every other consumer of this knob does.
    if torch.compiler.is_compiling():
        return 1
    if requested in (0, 1) or local_heads < 2:
        return 1
    if requested < 0:
        # Measured on magi-2-preview: pipelining wins on a PCIe exchange and loses
        # on an NVLink one, where the exchange is too cheap to hide.
        if group is None or _exchange_group_is_nvlink(group):
            return 1
        groups = preferred
    else:
        groups = min(requested, local_heads)
    while groups > 1 and local_heads % groups:
        groups -= 1
    return groups


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
        num_heads: int,
        num_experts_per_head: int,
        head_dim: int,
        intermediate_size: int,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.num_experts_per_head = num_experts_per_head
        self.num_local_experts = num_heads * num_experts_per_head
        self.head_dim = head_dim
        self.intermediate_size = intermediate_size
        if num_experts_per_head > MAX_KERNEL_EXPERTS:
            raise ValueError(
                f"num_experts_per_head ({num_experts_per_head}) exceeds the "
                f"{MAX_KERNEL_EXPERTS}-expert kernel limit even for one head"
            )
        # Balanced, not greedily filled: a trailing chunk of one head returns
        # wrong values.
        max_heads = MAX_KERNEL_EXPERTS // num_experts_per_head
        num_calls = -(-num_heads // max_heads)
        self.heads_per_call = -(-num_heads // num_calls)

        self.w13_weight = nn.Parameter(
            torch.empty(self.num_local_experts, 2 * intermediate_size, head_dim)
        )
        self.w2 = nn.Parameter(
            torch.empty(self.num_local_experts, head_dim, intermediate_size)
        )

    # Opaque to dynamo: autotuning these GEMMs asks more shared memory than sm90
    # has, so a compiled region containing them fails to build.
    @torch._dynamo.disable()
    def _run_heads(
        self,
        tokens: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        w13: torch.Tensor,
        w2: torch.Tensor,
        num_experts: int,
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
            num_experts=num_experts,
            num_local_experts=num_experts,
            hidden_size=self.head_dim,
            intermediate_size_per_partition=self.intermediate_size,
            top_k=topk_ids.shape[-1],
            activation="silu",
            is_gated=True,
            inplace=False,
            apply_router_weight_on_input=False,
            # The router already applied route_scale; setting this double-applies it.
            routed_scaling_factor=None,
            # w13_weight interleaves gate and up rows for the fused epilogue.
            gate_up_interleaved=True,
            gemm1_alpha=SWIGLU7_ALPHA,
            gemm1_clamp_limit=SWIGLU7_LIMIT,
        )
        return fused_experts(
            tokens.contiguous().bfloat16(),
            w13.bfloat16(),
            w2.bfloat16(),
            topk_output,
            runner_config,
            fuse_swiglu_interleaved=True,
        ).type_as(tokens)

    def _run_head_range(
        self,
        tokens: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        *,
        start: int,
        stop: int,
    ) -> torch.Tensor:
        """Run heads ``[start, stop)``; ids carry the head offset, so rebase them."""
        per_head = self.num_experts_per_head
        return self._run_heads(
            tokens,
            topk_ids - start * per_head,
            topk_weights,
            self.w13_weight[start * per_head : stop * per_head],
            self.w2[start * per_head : stop * per_head],
            (stop - start) * per_head,
        )

    @torch._dynamo.disable()
    def forward(
        self,
        tokens: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        if self.num_heads <= self.heads_per_call:
            return self._run_heads(
                tokens,
                topk_ids,
                topk_weights,
                self.w13_weight,
                self.w2,
                self.num_local_experts,
            )

        # Rows are token-major, so a head slice is dim 1 of the unflattened view;
        # expert rows are head-major, so the same slice is a row range.
        num_tokens = tokens.shape[0] // self.num_heads
        tokens_3d = tokens.view(num_tokens, self.num_heads, self.head_dim)
        ids_3d = topk_ids.view(num_tokens, self.num_heads, -1)
        weights_3d = topk_weights.view(num_tokens, self.num_heads, -1)
        out = torch.empty_like(tokens_3d)

        for start in range(0, self.num_heads, self.heads_per_call):
            stop = min(start + self.heads_per_call, self.num_heads)
            chunk = self._run_head_range(
                tokens_3d[:, start:stop].reshape(-1, self.head_dim),
                ids_3d[:, start:stop].reshape(-1, ids_3d.shape[-1]),
                weights_3d[:, start:stop].reshape(-1, weights_3d.shape[-1]),
                start=start,
                stop=stop,
            )
            out[:, start:stop] = chunk.view(num_tokens, stop - start, self.head_dim)
        return out.reshape(-1, self.head_dim)


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
                f"hidden_size {hidden_size} must be divisible by num_heads {num_heads}"
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
            num_heads=self.local_num_heads,
            num_experts_per_head=num_experts,
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
