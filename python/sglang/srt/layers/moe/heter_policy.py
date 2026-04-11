"""Heterogeneous dispatch policies for mixed-precision MoE.

Partitions experts into precision groups and transforms standard MoE routing
into per-group dispatches.  Each group's dispatch is ``(experts, scales)``
with shape ``[N, K]``.  Non-group expert slots use sentinel expert ID
(``num_experts``) and zero scale so the kernel skips them.

All operations use fixed-shape GPU tensors -- torch.compile and CUDA graph safe.
"""

import abc
import logging
from typing import List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

# (experts, scales) pair for one group, both [N, K].
GroupDispatchTuple = Tuple[torch.Tensor, torch.Tensor]


def _compute_group_sizes(
    num_experts: int,
    group_size_ratios: List[float],
) -> List[int]:
    """Convert fractional ratios to concrete group sizes. Last group absorbs rounding."""
    sizes: List[int] = []
    offset = 0
    for i, ratio in enumerate(group_size_ratios):
        if i == len(group_size_ratios) - 1:
            sizes.append(num_experts - offset)
        else:
            count = round(ratio * num_experts)
            sizes.append(count)
            offset += count
    return sizes


def _build_group_labels(
    num_experts: int,
    group_size_ratios: List[float],
    device: torch.device,
) -> torch.Tensor:
    """Pre-compute position-to-group labels for the N-group argsort path.

    Returns ``[num_experts]`` mapping sorted positions to group indices.
    Built once at init -- CPU->GPU transfer only happens here, not in hot path.
    """
    num_groups = len(group_size_ratios)
    group_sizes = _compute_group_sizes(num_experts, group_size_ratios)
    group_labels_list: List[int] = []
    for rev_idx, size in enumerate(reversed(group_sizes)):
        original_gidx = num_groups - 1 - rev_idx
        group_labels_list.extend([original_gidx] * size)
    return torch.tensor(group_labels_list, dtype=torch.long, device=device)


def _assign_by_score_gpu(
    scores: torch.Tensor,
    num_experts: int,
    group_size_ratios: List[float],
    expert_to_group: torch.Tensor,
    group_labels: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Split experts into groups by descending score (GPU-only, no sync).

    Last group gets highest-scoring experts (high-precision).
    G=2: ``torch.topk`` (O(E)).  G>2: ``torch.argsort + scatter_`` (O(E log E)).
    """
    num_groups = len(group_size_ratios)

    if num_groups <= 1:
        expert_to_group.zero_()
        return expert_to_group

    if num_groups == 2:
        k_high = round(num_experts * group_size_ratios[1])
        expert_to_group.zero_()
        _, top_indices = torch.topk(scores, k_high)
        expert_to_group[top_indices] = 1
        return expert_to_group
    else:
        assert group_labels is not None, (
            "group_labels required for N-group path (N>2)")
        sorted_ids = torch.argsort(scores, descending=True)
        expert_to_group.scatter_(0, sorted_ids, group_labels)
        return expert_to_group


class HeterDispatchPolicy(abc.ABC):
    """Base class for heterogeneous MoE dispatch policies.

    Subclasses implement ``_assign()`` -- the expert-to-group assignment
    strategy.  ``dispatch()`` calls ``_assign()``, then builds per-group
    ``(experts, scales)`` tuples with sentinel masking via ``torch.where``.
    """

    def __init__(
        self,
        num_experts: int,
        group_size_ratios: List[float],
        device: Optional[torch.device] = None,
    ):
        self._num_experts = num_experts
        self._group_size_ratios = group_size_ratios
        if device is None:
            device = torch.device("cuda")
        self._device = device
        self._expert_to_group_buf = torch.empty(
            num_experts, dtype=torch.long, device=device)
        self._group_labels: Optional[torch.Tensor] = None
        if len(group_size_ratios) > 2:
            self._group_labels = _build_group_labels(
                num_experts, group_size_ratios, device)

    @property
    def num_experts(self) -> int:
        return self._num_experts

    @property
    def num_groups(self) -> int:
        return len(self._group_size_ratios)

    @property
    def group_size_ratios(self) -> List[float]:
        return self._group_size_ratios

    @abc.abstractmethod
    def _assign(
        self,
        token_selected_experts: Optional[torch.Tensor],
        token_final_scales: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Return ``expert_to_group`` tensor ``[num_experts]`` on GPU."""
        ...

    def dispatch(
        self,
        token_selected_experts: torch.Tensor,
        token_final_scales: torch.Tensor,
    ) -> List[GroupDispatchTuple]:
        """Transform N-expert routing into per-group dispatches.

        Returns list of ``(experts, scales)`` with shape ``[N, K]`` per group.
        Non-group slots: expert=``num_experts``, scale=0.
        """
        expert_to_group = self._assign(
            token_selected_experts,
            token_final_scales,
        )
        return self._dispatch_from_expert_to_group(
            expert_to_group, token_selected_experts, token_final_scales)

    def _dispatch_from_expert_to_group(
        self,
        expert_to_group: torch.Tensor,
        token_selected_experts: torch.Tensor,
        token_final_scales: torch.Tensor,
    ) -> List[GroupDispatchTuple]:
        """Build per-group dispatch tuples using torch.where (fixed shapes)."""
        num_groups = self.num_groups
        slot_groups = expert_to_group[token_selected_experts.long()]

        results: List[GroupDispatchTuple] = []
        for gidx in range(num_groups):
            in_group = (slot_groups == gidx)
            experts_g = torch.where(
                in_group, token_selected_experts, self._num_experts)
            scales_g = torch.where(
                in_group, token_final_scales, 0.0)
            results.append((experts_g, scales_g))

        return results


class RandomHeterDispatch(HeterDispatchPolicy):
    """Random expert-to-group assignment. Deterministic when seed is set."""

    def __init__(
        self,
        num_experts: int,
        group_size_ratios: List[float],
        seed: int = 42,
        device: Optional[torch.device] = None,
    ):
        super().__init__(num_experts, group_size_ratios, device=device)
        gen = torch.Generator(device=self._device).manual_seed(seed)
        scores = torch.rand(
            num_experts, device=self._device, generator=gen)
        _assign_by_score_gpu(
            scores, num_experts, group_size_ratios,
            self._expert_to_group_buf, self._group_labels)

    def _assign(self, token_selected_experts, token_final_scales):
        return self._expert_to_group_buf


class ConfidenceThresholdHeterDispatch(HeterDispatchPolicy):
    """Assign by per-expert mean routing weight.

    High-weight experts -> last group (high-precision).
    Falls back to random when signals are unavailable.
    """

    def __init__(
        self,
        num_experts: int,
        group_size_ratios: List[float],
        confidence_threshold: float = 0.5,
        fallback_seed: int = 42,
        device: Optional[torch.device] = None,
    ):
        super().__init__(num_experts, group_size_ratios, device=device)
        self._confidence_threshold = confidence_threshold
        self._fallback = RandomHeterDispatch(
            num_experts, group_size_ratios, seed=fallback_seed, device=device)
        self._expert_weight_sum = torch.empty(
            num_experts, device=self._device, dtype=torch.float32)
        self._expert_count_buf = torch.zeros(
            num_experts, device=self._device, dtype=torch.float32)

    def _assign(self, token_selected_experts, token_final_scales):
        if token_selected_experts is None or token_final_scales is None:
            return self._fallback._assign(
                token_selected_experts, token_final_scales)

        buf = self._expert_weight_sum
        flat_experts = token_selected_experts.reshape(-1).long()
        flat_scales = token_final_scales.reshape(-1)

        buf.zero_()
        buf.scatter_add_(0, flat_experts, flat_scales)

        expert_count = self._expert_count_buf
        expert_count.zero_()
        expert_count.scatter_add_(
            0, flat_experts,
            torch.ones_like(flat_experts, dtype=torch.float32))
        expert_count.clamp_min_(1.0)
        buf.div_(expert_count)

        return _assign_by_score_gpu(
            buf, self._num_experts, self._group_size_ratios,
            self._expert_to_group_buf, self._group_labels)


class ExpertLoadHeterDispatch(HeterDispatchPolicy):
    """Assign by expert activation frequency.

    Hot experts -> last group (high-precision).
    Falls back to random when signals are unavailable.
    """

    def __init__(
        self,
        num_experts: int,
        group_size_ratios: List[float],
        fallback_seed: int = 42,
        device: Optional[torch.device] = None,
    ):
        super().__init__(num_experts, group_size_ratios, device=device)
        self._fallback = RandomHeterDispatch(
            num_experts, group_size_ratios, seed=fallback_seed, device=device)
        self._count_buf = torch.zeros(
            num_experts, device=self._device, dtype=torch.float32)

    def _assign(self, token_selected_experts, token_final_scales):
        if token_selected_experts is None:
            return self._fallback._assign(
                token_selected_experts, token_final_scales)

        flat_experts = token_selected_experts.reshape(-1).long()
        counts = self._count_buf
        counts.zero_()
        counts.scatter_add_(
            0, flat_experts,
            torch.ones_like(flat_experts, dtype=torch.float32))

        return _assign_by_score_gpu(
            counts, self._num_experts, self._group_size_ratios,
            self._expert_to_group_buf, self._group_labels)


_POLICY_REGISTRY = {
    "expert_load": ExpertLoadHeterDispatch,
    "confidence": ConfidenceThresholdHeterDispatch,
    "random": RandomHeterDispatch,
}


def create_policy(
    policy_name: str,
    num_experts: int,
    group_size_ratios: List[float],
    device: Optional[torch.device] = None,
    **kwargs,
) -> HeterDispatchPolicy:
    cls = _POLICY_REGISTRY.get(policy_name)
    if cls is None:
        raise ValueError(
            f"Unknown heter policy: {policy_name}. "
            f"Available: {list(_POLICY_REGISTRY.keys())}"
        )
    return cls(
        num_experts=num_experts,
        group_size_ratios=group_size_ratios,
        device=device,
        **kwargs,
    )
