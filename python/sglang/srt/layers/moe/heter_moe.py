"""Heterogeneous-precision MoE layer.

Stores expert weights in multiple precisions (e.g., BF16 + INT4).
Classifies experts per-batch based on token load, runs separate
group-GEMMs per precision, sums outputs.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from sglang.srt.layers.moe.fused_moe_triton.fused_moe import outplace_fused_experts
from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import fused_marlin_moe
from sglang.srt.layers.moe.heter_policy import (
    BaseHeterPolicy,
    HeterDispatchPlan,
    TokenCountPolicy,
    create_policy,
)


logger = logging.getLogger(__name__)


def _parse_heter_config(config_path: str) -> Dict[str, Any]:
    with open(config_path) as f:
        cfg = json.load(f)
    groups = cfg["groups"]
    total_ratio = sum(g["size_ratio"] for g in groups)
    assert abs(total_ratio - 1.0) < 1e-3, (
        f"size_ratios must sum to 1.0, got {total_ratio}"
    )
    return cfg


def _subset_for_group(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    group_expert_ids: List[int],
    num_experts: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Remap topk_ids to local expert indices and zero non-group weights.

    Given global topk_ids and a list of expert IDs belonging to this group,
    returns (local_topk_weights, local_topk_ids, num_group_experts) where:
    - local_topk_ids maps global expert IDs → 0..len(group_expert_ids)-1
    - local_topk_weights is zeroed for tokens not routed to this group
    """
    num_group_experts = len(group_expert_ids)

    # Build global→local remap: non-group experts map to 0 (will be zeroed)
    global_to_local = torch.zeros(num_experts, dtype=torch.int64, device=device)
    for local_idx, global_idx in enumerate(group_expert_ids):
        global_to_local[global_idx] = local_idx

    local_topk_ids = global_to_local[topk_ids]

    # Zero weights for tokens routed to experts outside this group
    active_mask = torch.zeros(num_experts, dtype=torch.bool, device=device)
    active_mask[group_expert_ids] = True
    mask = active_mask[topk_ids]
    local_topk_weights = topk_weights * mask.to(topk_weights.dtype)

    return local_topk_weights, local_topk_ids, num_group_experts


class HeterFusedMoE(nn.Module):
    """Multi-precision MoE layer with per-batch dynamic expert assignment.

    Architecture: composition with shared routing, separate weight stores.
    Each precision group has its own weight tensors. At forward time, token
    routing weights are masked per-group so each kernel processes only its
    assigned experts.
    """

    def __init__(
        self,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        top_k: int,
        heter_config: Dict[str, Any],
        dtype: torch.dtype = torch.bfloat16,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.top_k = top_k
        self.dtype = dtype
        self.device = device or torch.device("cuda")

        self.group_cfgs = heter_config["groups"]
        self.num_groups = len(self.group_cfgs)
        self.group_ratios = [g["size_ratio"] for g in self.group_cfgs]

        policy_name = heter_config.get("policy", "token_count")
        policy_kwargs = heter_config.get("policy_params", {})
        self.policy: BaseHeterPolicy = create_policy(policy_name, **policy_kwargs)

        self._init_group_weights()

    def _init_group_weights(self) -> None:
        """Create weight containers per precision group.

        For BF16 groups: standard w13_weight [E, 2*I, H] + w2_weight [E, H, I]
        For INT4 groups: packed qweights + scales (Marlin format)
        """
        E = self.num_experts
        H = self.hidden_size
        I = self.intermediate_size

        for idx, gcfg in enumerate(self.group_cfgs):
            num_bits = gcfg.get("num_bits", 16)
            prefix = f"group{idx}"

            if num_bits == 16:
                self._init_bf16_weights(prefix, E, H, I)
            elif num_bits == 4:
                self._init_int4_weights(prefix, E, H, I, gcfg)
            elif num_bits == 8:
                # DEPRECATED: Triton INT8 on A100 achieves ~6% of peak tensor core
                # utilization due to layout constraints and missing pipelining.
                # See: https://github.com/triton-lang/triton/issues/2818
                #      https://github.com/triton-lang/triton/issues/1397
                # Use num_bits=4 (Marlin) instead. Kept for forward-compat with
                # Hopper FP8 path (future work).
                import warnings

                warnings.warn(
                    "INT8 (num_bits=8) is deprecated for HeterMoE on A100. "
                    "Triton INT8 achieves only ~6% of peak tensor core throughput "
                    "due to Ampere layout constraints (triton-lang/triton#2818). "
                    "Use num_bits=4 (Marlin INT4) instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                self._init_int8_weights(prefix, E, H, I)
            else:
                raise ValueError(f"Unsupported num_bits={num_bits} in group {idx}")

    def _init_bf16_weights(self, prefix: str, E: int, H: int, I: int) -> None:
        w13 = nn.Parameter(
            torch.empty(E, 2 * I, H, dtype=self.dtype, device=self.device),
            requires_grad=False,
        )
        w2 = nn.Parameter(
            torch.empty(E, H, I, dtype=self.dtype, device=self.device),
            requires_grad=False,
        )
        self.register_parameter(f"{prefix}_w13_weight", w13)
        self.register_parameter(f"{prefix}_w2_weight", w2)

    def _init_int4_weights(
        self, prefix: str, E: int, H: int, I: int, gcfg: Dict
    ) -> None:
        group_size = gcfg.get("group_size", 128)
        # Marlin packing: 8 INT4 values per int32 → hidden_size // 8
        pack_factor = 8
        w13_qw = nn.Parameter(
            torch.empty(
                E, 2 * I, H // pack_factor, dtype=torch.int32, device=self.device
            ),
            requires_grad=False,
        )
        w2_qw = nn.Parameter(
            torch.empty(E, H, I // pack_factor, dtype=torch.int32, device=self.device),
            requires_grad=False,
        )
        w13_scales = nn.Parameter(
            torch.empty(
                E, 2 * I, H // group_size, dtype=self.dtype, device=self.device
            ),
            requires_grad=False,
        )
        w2_scales = nn.Parameter(
            torch.empty(E, H, I // group_size, dtype=self.dtype, device=self.device),
            requires_grad=False,
        )
        self.register_parameter(f"{prefix}_w13_qweight", w13_qw)
        self.register_parameter(f"{prefix}_w2_qweight", w2_qw)
        self.register_parameter(f"{prefix}_w13_scales", w13_scales)
        self.register_parameter(f"{prefix}_w2_scales", w2_scales)

    def _init_int8_weights(self, prefix: str, E: int, H: int, I: int) -> None:
        w13 = nn.Parameter(
            torch.empty(E, 2 * I, H, dtype=torch.int8, device=self.device),
            requires_grad=False,
        )
        w2 = nn.Parameter(
            torch.empty(E, H, I, dtype=torch.int8, device=self.device),
            requires_grad=False,
        )
        w13_scale = nn.Parameter(
            torch.ones(E, 2 * I, 1, dtype=torch.float32, device=self.device),
            requires_grad=False,
        )
        w2_scale = nn.Parameter(
            torch.ones(E, H, 1, dtype=torch.float32, device=self.device),
            requires_grad=False,
        )
        self.register_parameter(f"{prefix}_w13_weight", w13)
        self.register_parameter(f"{prefix}_w2_weight", w2)
        self.register_parameter(f"{prefix}_w13_weight_scale", w13_scale)
        self.register_parameter(f"{prefix}_w2_weight_scale", w2_scale)

    def init_fake_weights(self, seed: int = 0) -> None:
        """Fill all weight tensors with random data for testing."""
        gen = torch.Generator(device=self.device)
        gen.manual_seed(seed)
        for name, param in self.named_parameters():
            if param.dtype in (torch.bfloat16, torch.float16, torch.float32):
                param.data.normal_(0, 0.02, generator=gen)
            elif param.dtype == torch.int8:
                param.data.copy_(
                    torch.randint(
                        -128, 127, param.shape, device=self.device, dtype=torch.int8
                    )
                )
            elif param.dtype == torch.int32:
                param.data.copy_(
                    torch.randint(
                        0, 2**31 - 1, param.shape, device=self.device, dtype=torch.int32
                    )
                )

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        router_logits: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with heterogeneous precision.

        Args:
            hidden_states: [num_tokens, hidden_size]
            topk_weights:  [num_tokens, top_k]
            topk_ids:      [num_tokens, top_k]
            router_logits: [num_tokens, num_experts] (needed for fused_marlin_moe)

        Returns:
            output: [num_tokens, hidden_size]
        """
        assignment = self.policy.assign(topk_ids, self.num_experts, self.group_ratios)

        output = torch.zeros_like(hidden_states)

        for group_idx, gcfg in enumerate(self.group_cfgs):
            expert_ids_list = assignment.group_assignments[group_idx]
            if len(expert_ids_list) == 0:
                continue

            local_weights, local_ids, n_group = _subset_for_group(
                topk_weights,
                topk_ids,
                expert_ids_list,
                self.num_experts,
                hidden_states.device,
            )
            eidx = torch.tensor(expert_ids_list, device=hidden_states.device)

            num_bits = gcfg.get("num_bits", 16)
            prefix = f"group{group_idx}"

            if num_bits == 16:
                group_out = self._run_bf16_group(
                    prefix, hidden_states, local_weights, local_ids, eidx
                )
            elif num_bits == 4:
                group_out = self._run_int4_group(
                    prefix,
                    hidden_states,
                    local_weights,
                    local_ids,
                    eidx,
                    router_logits,
                    gcfg,
                )
            elif num_bits == 8:
                # DEPRECATED: see _init_group_weights for rationale
                group_out = self._run_int8_group(
                    prefix, hidden_states, local_weights, local_ids, eidx
                )
            else:
                raise ValueError(f"Unsupported num_bits={num_bits}")

            output = output + group_out

        return output

    def _run_bf16_group(
        self,
        prefix: str,
        hidden_states: torch.Tensor,
        local_weights: torch.Tensor,
        local_ids: torch.Tensor,
        expert_idx: torch.Tensor,
    ) -> torch.Tensor:
        w13 = getattr(self, f"{prefix}_w13_weight")[expert_idx]
        w2 = getattr(self, f"{prefix}_w2_weight")[expert_idx]
        return outplace_fused_experts(
            hidden_states,
            w13,
            w2,
            local_weights,
            local_ids,
        )

    def _run_int4_group(
        self,
        prefix: str,
        hidden_states: torch.Tensor,
        local_weights: torch.Tensor,
        local_ids: torch.Tensor,
        expert_idx: torch.Tensor,
        router_logits: Optional[torch.Tensor],
        gcfg: Dict,
    ) -> torch.Tensor:
        w13_qw = getattr(self, f"{prefix}_w13_qweight")[expert_idx]
        w2_qw = getattr(self, f"{prefix}_w2_qweight")[expert_idx]
        w13_scales = getattr(self, f"{prefix}_w13_scales")[expert_idx]
        w2_scales = getattr(self, f"{prefix}_w2_scales")[expert_idx]

        if router_logits is None:
            raise ValueError("router_logits required for Marlin INT4 kernel")

        # Marlin takes gating_output [M, num_experts] — subset to group's experts
        local_gating = router_logits[:, expert_idx]

        return fused_marlin_moe(
            hidden_states=hidden_states,
            w1=w13_qw,
            w2=w2_qw,
            w1_scale=w13_scales,
            w2_scale=w2_scales,
            gating_output=local_gating,
            topk_weights=local_weights,
            topk_ids=local_ids,
            num_bits=4,
            is_k_full=True,
            inplace=False,
        )

    def _run_int8_group(
        self,
        prefix: str,
        hidden_states: torch.Tensor,
        local_weights: torch.Tensor,
        local_ids: torch.Tensor,
        expert_idx: torch.Tensor,
    ) -> torch.Tensor:
        w13 = getattr(self, f"{prefix}_w13_weight")[expert_idx]
        w2 = getattr(self, f"{prefix}_w2_weight")[expert_idx]
        w13_scale = getattr(self, f"{prefix}_w13_weight_scale")[expert_idx]
        w2_scale = getattr(self, f"{prefix}_w2_weight_scale")[expert_idx]
        return outplace_fused_experts(
            hidden_states,
            w13,
            w2,
            local_weights,
            local_ids,
            use_int8_w8a8=True,
            per_channel_quant=True,
            w1_scale=w13_scale,
            w2_scale=w2_scale,
        )
