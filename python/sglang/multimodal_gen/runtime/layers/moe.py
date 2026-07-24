# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# Adapted from LingBot-Video (https://github.com/Robbyant/lingbot-video).

"""Mixture-of-Experts FFN for the LingBot-Video MoE DiT.

This is the first MoE layer in the diffusion runtime (``multimodal_gen``). It ports
the upstream DeepSeek-V3-style grouped MoE (128 routed experts, top-8, group-limited
routing, sigmoid + ``e_score_correction_bias``, 1 shared expert) and reuses SGLang's
LLM-runtime fused-MoE Triton kernel (``sglang.srt.layers.moe...fused_experts``) for
the expert GEMMs, exactly as the upstream ``sglang_moe_shim`` does.

The expert weights are plain ``nn.Parameter`` tensors with the upstream layout
(``w1/w3`` ``[E, I, H]``, ``w2`` ``[E, H, I]``) so they load from the diffusers
checkpoint by name match with an identity ``param_names_mapping``.

MVP: the ``sglang_triton`` backend only (no ``grouped_mm``/reorder/restore/fp8 paths).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


# Minimal server args for the SGLang MoE runtime: ``fused_experts``'s Triton
# config selection reads ``get_global_server_args()``. The multimodal_gen server
# doesn't always set the srt global server args, so we set a minimal one (mirrors
# the upstream ``sglang_moe_shim.ensure_sglang_moe_ready``).
class _MoeServerArgs:
    """Minimal server args for the SRT MoE runtime.

    Defaults any missing attr to ``False`` so the srt ``fused_experts`` path's
    server_args checks all read ``False`` (disabled paths) — robust against the
    srt runtime reading attrs (``enable_symm_mem``, ``enable_fused_moe_sum_all_reduce``,
    ...) that the multimodal_gen server doesn't set on the srt global server args.
    """

    enable_deterministic_inference = False
    enable_fused_moe_sum_all_reduce = False
    enable_symm_mem = False

    def __getattr__(self, name: str) -> bool:
        return False


_SGLANG_MOE_SERVER_ARGS = _MoeServerArgs()
_moe_server_args_ready = False


def _ensure_moe_server_args() -> None:
    global _moe_server_args_ready
    if _moe_server_args_ready:
        return
    from sglang.srt.server_args import (
        get_global_server_args,
        set_global_server_args_for_scheduler,
    )

    try:
        server_args = get_global_server_args()
    except Exception:  # noqa: BLE001
        server_args = None
    if server_args is None:
        server_args = _SGLANG_MOE_SERVER_ARGS
        set_global_server_args_for_scheduler(server_args)
    if not hasattr(server_args, "enable_deterministic_inference"):
        server_args.enable_deterministic_inference = False
    if not hasattr(server_args, "enable_fused_moe_sum_all_reduce"):
        server_args.enable_fused_moe_sum_all_reduce = False
    _moe_server_args_ready = True


def _ensure_srt_distributed() -> None:
    """Init the SRT tensor-model-parallel group for single-GPU.

    SGLang's ``fused_experts`` (srt) uses ``srt.distributed.get_tp_group`` for the
    outplace symmetric-memory allocation path. The multimodal_gen server has its
    own ``parallel_state`` but does not init the srt one, so we lazily init a
    single-process TP group here (a no-op group) on first use.
    """
    from sglang.srt.distributed import parallel_state

    try:
        parallel_state.get_tp_group()
        return
    except Exception:  # noqa: BLE001, S110
        pass
    import torch.distributed as dist

    if not dist.is_initialized():
        import os

        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29512")
        dist.init_process_group(backend="gloo", world_size=1, rank=0)
    parallel_state.init_distributed_environment(
        world_size=1, rank=0, local_rank=0, backend="gloo"
    )
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=1)


class LingBotVideoMLP(nn.Module):
    """SwiGLU MLP used for the shared expert (and the dense-MLP fallback)."""

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class LingBotVideoRouter(nn.Module):
    """Token-choice top-k router (inference path; no capacity/jitter/load stats).

    The asymmetry must be preserved for parity: expert *selection* uses the
    bias-added score, while the gating *weights* gather the bias-free score.
    The router runs in fp32 (autocast disabled) — this is parity-critical.
    """

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
    """Routed-expert weights in the GroupedExperts layout.

    ``w1``/``w3`` are ``[E, I, H]`` (gate/up), ``w2`` is ``[E, H, I]`` (down).
    Kept as plain ``nn.Parameter`` so the diffusers checkpoint loads by name.
    """

    def __init__(
        self, num_experts: int, hidden_size: int, intermediate_size: int
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.w1 = nn.Parameter(torch.empty(num_experts, intermediate_size, hidden_size))
        self.w2 = nn.Parameter(torch.empty(num_experts, hidden_size, intermediate_size))
        self.w3 = nn.Parameter(torch.empty(num_experts, intermediate_size, hidden_size))


class LingBotVideoSparseMoeBlock(nn.Module):
    """DeepSeek-V3-style sparse MoE FFN block (routed experts + shared expert).

    Expert GEMMs reuse SGLang's ``fused_experts`` Triton kernel (the same path the
    upstream ``sglang_moe_shim`` uses): ``w13 = cat(w1, w3, dim=1)`` ``[E, 2I, H]``
    (non-interleaved gate|up), ``w2`` ``[E, H, I]``. The router already applies
    ``routed_scaling_factor``, so it is passed as ``None`` to avoid double-scaling.
    """

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
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.intermediate_size = intermediate_size
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

    def _run_sglang_triton_experts(
        self,
        tokens: torch.Tensor,
        top_scores: torch.Tensor,
        top_indices: torch.Tensor,
    ) -> torch.Tensor:
        # Lazy import: keeps the diffusion runtime importable without pulling the
        # SGLang MoE runtime until the MoE is actually used.
        from sglang.srt.layers.moe.moe_runner import MoeRunnerConfig
        from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import (
            fused_experts,
        )
        from sglang.srt.layers.moe.topk import StandardTopKOutput

        # fused_experts' Triton config selection reads get_global_server_args();
        # the multimodal_gen server may not have set it, so ensure a minimal one.
        _ensure_moe_server_args()
        # fused_experts' outplace path calls srt get_tp_group (symmetric memory);
        # init a single-process SRT TP group (the multimodal_gen doesn't).
        _ensure_srt_distributed()

        topk_output = StandardTopKOutput(
            topk_weights=top_scores.float(),
            topk_ids=top_indices.to(torch.int32),
            router_logits=torch.empty(0, device=tokens.device),
        )
        # CRITICAL parity fields: inplace=False (outplace, the router already scaled
        # the weights) and gate_up_interleaved=False (w13 = cat(gate, up) is a
        # contiguous gate|up split, NOT interleaved — the MoeRunnerConfig default
        # True would misread the layout and produce garbage). routed_scaling_factor
        # is None here because the upstream router applies route_scale itself.
        runner_config = MoeRunnerConfig(
            num_experts=self.num_experts,
            num_local_experts=self.num_experts,
            hidden_size=self.hidden_size,
            intermediate_size_per_partition=self.intermediate_size,
            top_k=self.top_k,
            activation="silu",
            is_gated=True,
            inplace=False,
            apply_router_weight_on_input=False,
            routed_scaling_factor=None,
            gate_up_interleaved=False,
        )
        w13 = torch.cat((self.experts.w1, self.experts.w3), dim=1).contiguous()
        w2 = self.experts.w2.contiguous()
        return fused_experts(
            tokens.contiguous().bfloat16(),
            w13.bfloat16(),
            w2.bfloat16(),
            topk_output,
            runner_config,
        ).type_as(tokens)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states: (B, S, H); MVP is B=1 with an already-trimmed mask.
        b = hidden_states.shape[0]
        tokens = hidden_states.reshape(-1, self.hidden_size)
        top_indices, top_scores = self.router(tokens)
        out = self._run_sglang_triton_experts(tokens, top_scores, top_indices)
        out = out.reshape(b, -1, self.hidden_size)
        if self.shared_experts is not None:
            out = out + self.shared_experts(hidden_states)
        return out
