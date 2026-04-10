"""Heterogeneous-precision MoE dispatch policies.

Classifies experts into precision groups (e.g., cold→INT4, hot→BF16)
based on per-batch routing signals. Assignment is dynamic per forward pass.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch


@dataclass
class HeterDispatchPlan:
    """Expert-to-group assignment for one forward pass.

    group_assignments[i] = sorted list of expert IDs in group i.
    Every expert appears in exactly one group.
    """

    group_assignments: List[List[int]]

    def validate(self, num_experts: int) -> None:
        all_experts = []
        for group in self.group_assignments:
            all_experts.extend(group)
        all_experts_sorted = sorted(all_experts)
        expected = list(range(num_experts))
        assert all_experts_sorted == expected, (
            f"Expert assignment mismatch: got {all_experts_sorted}, expected {expected}"
        )

    def get_expert_to_group(self, num_experts: int) -> torch.Tensor:
        """Return tensor [num_experts] mapping expert_id -> group_idx."""
        mapping = torch.zeros(num_experts, dtype=torch.long)
        for group_idx, expert_ids in enumerate(self.group_assignments):
            for eid in expert_ids:
                mapping[eid] = group_idx
        return mapping


class BaseHeterPolicy:
    """Abstract base for heterogeneous dispatch policies."""

    def assign(
        self,
        topk_ids: torch.Tensor,
        num_experts: int,
        group_ratios: List[float],
        topk_weights: Optional[torch.Tensor] = None,
    ) -> HeterDispatchPlan:
        """Assign experts to precision groups.

        Args:
            topk_ids: [num_tokens, top_k] selected expert IDs.
            num_experts: total expert count.
            group_ratios: fraction of experts per group (must sum to 1.0).
            topk_weights: [num_tokens, top_k] routing weights (optional).

        Returns:
            HeterDispatchPlan with expert-to-group mapping.
        """
        raise NotImplementedError


class TokenCountPolicy(BaseHeterPolicy):
    """Assign experts based on token activation frequency.

    Hot experts (high token count) go to the high-precision group (last group).
    Cold experts (low token count) go to the low-precision group (first group).
    """

    def assign(
        self,
        topk_ids: torch.Tensor,
        num_experts: int,
        group_ratios: List[float],
        topk_weights: Optional[torch.Tensor] = None,
    ) -> HeterDispatchPlan:
        # Count tokens per expert
        flat_ids = topk_ids.reshape(-1)
        counts = torch.bincount(flat_ids, minlength=num_experts)

        # Sort experts by count (ascending: cold first, hot last)
        sorted_indices = torch.argsort(counts, descending=False)

        # Split into groups by ratio
        group_assignments: List[List[int]] = []
        start = 0
        for i, ratio in enumerate(group_ratios):
            if i == len(group_ratios) - 1:
                # Last group gets remaining experts (avoids rounding issues)
                end = num_experts
            else:
                end = start + round(ratio * num_experts)
                end = min(end, num_experts)
            group_experts = sorted(sorted_indices[start:end].tolist())
            group_assignments.append(group_experts)
            start = end

        plan = HeterDispatchPlan(group_assignments=group_assignments)
        plan.validate(num_experts)
        return plan


class FixedPolicy(BaseHeterPolicy):
    """Fixed expert-to-group assignment (for testing / manual override)."""

    def __init__(self, fixed_assignments: List[List[int]]):
        self.fixed_assignments = fixed_assignments

    def assign(
        self,
        topk_ids: torch.Tensor,
        num_experts: int,
        group_ratios: List[float],
        topk_weights: Optional[torch.Tensor] = None,
    ) -> HeterDispatchPlan:
        plan = HeterDispatchPlan(group_assignments=self.fixed_assignments)
        plan.validate(num_experts)
        return plan


class RandomPolicy(BaseHeterPolicy):
    """Deterministic random assignment (for testing).

    Uses a fixed seed so assignment is reproducible given the same config.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed

    def assign(
        self,
        topk_ids: torch.Tensor,
        num_experts: int,
        group_ratios: List[float],
        topk_weights: Optional[torch.Tensor] = None,
    ) -> HeterDispatchPlan:
        gen = torch.Generator()
        gen.manual_seed(self.seed)
        perm = torch.randperm(num_experts, generator=gen)

        group_assignments: List[List[int]] = []
        start = 0
        for i, ratio in enumerate(group_ratios):
            if i == len(group_ratios) - 1:
                end = num_experts
            else:
                end = start + round(ratio * num_experts)
                end = min(end, num_experts)
            group_experts = sorted(perm[start:end].tolist())
            group_assignments.append(group_experts)
            start = end

        plan = HeterDispatchPlan(group_assignments=group_assignments)
        plan.validate(num_experts)
        return plan


_POLICY_REGISTRY: Dict[str, type] = {
    "token_count": TokenCountPolicy,
    "fixed": FixedPolicy,
    "random": RandomPolicy,
}


def create_policy(policy_name: str, **kwargs) -> BaseHeterPolicy:
    cls = _POLICY_REGISTRY.get(policy_name)
    if cls is None:
        raise ValueError(
            f"Unknown heter policy: {policy_name}. Available: {list(_POLICY_REGISTRY.keys())}"
        )
    return cls(**kwargs)
