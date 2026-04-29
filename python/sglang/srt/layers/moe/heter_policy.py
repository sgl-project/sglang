"""Heterogeneous dispatch policies for mixed-precision MoE.

Partitions experts into precision groups and transforms standard MoE routing
into per-group dispatches.  Each group's dispatch is ``(experts, scales)``
with shape ``[N, K]``.  Non-group expert slots use sentinel expert ID
(``num_experts``) and zero scale so the kernel skips them.

All operations use fixed-shape GPU tensors -- torch.compile and CUDA graph safe.

Layered design:
  - The ABC owns a per-expert routed-token-count tensor and a universal
    "promote any expert with count >= threshold to BF16" rule. The threshold
    is a required ctor arg sourced from ``heter_config.bf16_promotion_threshold``.
  - Subclasses implement ``_assign(counts, ...)`` -- the policy-specific
    expert-to-group scoring strategy. They may read the precomputed counts.
  - ``int4_only_mask`` is the final word: experts in that mask have no BF16
    weights loaded, so they stay INT4 even when the threshold rule would
    promote them. ``bf16_only_mask`` is also a final-word forced override.
"""

import abc
import logging
import os
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
        expert_to_group.scatter_(0, top_indices, 1)
        return expert_to_group
    else:
        assert group_labels is not None, (
            "group_labels required for N-group path (N>2)")
        sorted_ids = torch.argsort(scores, descending=True)
        expert_to_group.scatter_(0, sorted_ids, group_labels)
        return expert_to_group


class HeterDispatchPolicy(abc.ABC):
    """Base class for heterogeneous MoE dispatch policies.

    Layered responsibilities:

    * ABC owns ``_token_count_buf`` (per-expert routed-token counts), populated
      every dispatch by ``_compute_token_counts`` from ``token_selected_experts``.
    * Subclasses implement ``_assign(counts, experts, scales)`` -- the
      policy-specific scoring strategy.  They may read ``counts`` rather than
      recomputing.  Children also (optionally) hold a per-expert score buffer
      e.g.  ``_weight_sum_buf`` for total-weight ranking or static ``_importance``
      for Hessian weighting.
    * After ``_assign`` returns, ``dispatch`` applies the universal **BF16
      promotion rule**: any expert with ``count >= bf16_promotion_threshold``
      goes to the BF16 group regardless of policy choice.
    * ``_dispatch_from_expert_to_group`` then applies forced-precision masks
      (``int4_only`` first, ``bf16_only`` last), so an INT4-only expert stays
      INT4 even when the threshold rule would have promoted it.
    """

    def __init__(
        self,
        num_experts: int,
        group_size_ratios: List[float],
        bf16_promotion_threshold: int,
        device: Optional[torch.device] = None,
        int4_only_mask: Optional[torch.Tensor] = None,
        int4_group_idx: int = 0,
        bf16_only_mask: Optional[torch.Tensor] = None,
        bf16_group_idx: int = 1,
    ):
        self._num_experts = num_experts
        self._group_size_ratios = group_size_ratios
        if device is None:
            device = torch.device("cuda")
        self._device = device
        # Pre-zero the assignment buffer at construction time so that cold
        # downstream kernels (e.g. Marlin INT4) never read stale CUDA-allocator
        # memory before ``_assign`` populates it.
        self._expert_to_group_buf = torch.zeros(
            num_experts, dtype=torch.long, device=device)
        # Scratch tensor used by ``dispatch`` when applying the universal
        # BF16 promotion rule.  Decoupled from the buffer the policy's
        # ``_assign`` returns so that mutating the dispatch result never
        # corrupts the policy's internal state across calls.
        self._dispatch_scratch = torch.zeros(
            num_experts, dtype=torch.long, device=device)
        self._group_labels: Optional[torch.Tensor] = None
        if len(group_size_ratios) > 2:
            self._group_labels = _build_group_labels(
                num_experts, group_size_ratios, device)

        # INT4-only / BF16-only forced-precision masks (final word in dispatch).
        self._int4_only_mask = int4_only_mask
        self._int4_group_idx = int4_group_idx
        self._bf16_only_mask = bf16_only_mask
        self._bf16_group_idx = bf16_group_idx

        # Universal token-count tensor + ones buffer shared by all subclasses.
        # The counts buffer is sized ``num_experts + 1``: the extra slot at
        # index ``num_experts`` is a "trash" sink for out-of-range routing
        # indices (e.g. EP non-local sentinels < 0, or the Marlin
        # ``num_experts`` sentinel) so ``scatter_add_`` never trips its
        # bounds check. Subclasses see only the first ``num_experts`` entries
        # via the slice returned from ``_compute_token_counts``.
        self._token_count_buf = torch.zeros(
            num_experts + 1, device=self._device, dtype=torch.float32)
        self._ones_buf = torch.ones(
            num_experts, device=self._device, dtype=torch.float32)

        # Universal BF16 promotion threshold. Required: must be specified in
        # heter_config.json and threaded by the runtime.
        self._bf16_promotion_threshold = int(bf16_promotion_threshold)

    @property
    def num_experts(self) -> int:
        return self._num_experts

    @property
    def num_groups(self) -> int:
        return len(self._group_size_ratios)

    @property
    def group_size_ratios(self) -> List[float]:
        return self._group_size_ratios

    @property
    def bf16_promotion_threshold(self) -> int:
        return self._bf16_promotion_threshold

    def _compute_token_counts(
        self,
        token_selected_experts: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Populate ``self._token_count_buf`` with per-expert routed counts.

        Returns the buffer.  When ``token_selected_experts is None``, the
        buffer is zeroed (no expert can cross threshold) -- subclasses can
        treat that case as "no routing info".

        EP layers remap non-local experts to negative sentinels (and Marlin
        uses ``num_experts`` as its sentinel); both must be filtered out of
        the scatter_add_ to keep CUDA's bounds-checker happy without forcing
        a host sync. We do this by routing all out-of-range indices to a
        dedicated trash slot at the end of an oversized counts buffer, then
        slicing it off.
        """
        counts = self._token_count_buf
        counts.zero_()
        if token_selected_experts is None:
            return counts[:self._num_experts]
        flat = token_selected_experts.reshape(-1).long()
        # Map any out-of-range (negative or >= num_experts) index to the
        # trash slot at index ``num_experts``.  This is a single in-place
        # ``where`` against a host-known scalar -- no allocations.
        valid = (flat >= 0) & (flat < self._num_experts)
        safe = torch.where(valid, flat, self._num_experts)
        n = safe.shape[0]
        if self._ones_buf.shape[0] < n:
            self._ones_buf = torch.ones(
                n, device=self._device, dtype=torch.float32)
        counts.scatter_add_(0, safe, self._ones_buf[:n])
        return counts[:self._num_experts]

    @abc.abstractmethod
    def _assign(
        self,
        token_counts: torch.Tensor,
        token_selected_experts: Optional[torch.Tensor],
        token_final_scales: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Return ``expert_to_group`` tensor ``[num_experts]`` on GPU.

        Children may read ``token_counts`` (precomputed by the ABC) instead
        of recomputing.  The ABC's ``dispatch`` will then apply the universal
        threshold-promotion rule on top of whatever assignment is returned.
        """
        ...

    def dispatch(
        self,
        token_selected_experts: torch.Tensor,
        token_final_scales: torch.Tensor,
        sentinel: int = -1,
    ) -> List[GroupDispatchTuple]:
        """Transform N-expert routing into per-group dispatches.

        Returns list of ``(experts, scales)`` with shape ``[N, K]`` per group.
        Non-group slots: expert=``sentinel``, scale=0.

        Args:
            sentinel: expert ID for non-group slots. Default -1 (Triton
                kernels skip -1). Marlin INT4 needs ``num_experts``.
        """
        token_counts = self._compute_token_counts(token_selected_experts)
        policy_assignment = self._assign(
            token_counts, token_selected_experts, token_final_scales,
        )
        # Copy into our dedicated dispatch scratch before applying the
        # threshold rule so we never mutate whatever buffer the policy
        # returned (some children share that buffer with their own state).
        scratch = self._dispatch_scratch
        scratch.copy_(policy_assignment)
        # Universal BF16 promotion: any expert with count >= threshold goes
        # to BF16, regardless of policy assignment. ``int4_only_mask`` in
        # ``_dispatch_from_expert_to_group`` then forces ``int4_only`` experts
        # back to INT4 (no BF16 weights loaded), so the threshold rule is
        # safe even for forced-INT4 experts.
        is_promoted = token_counts >= self._bf16_promotion_threshold
        scratch[is_promoted] = self._bf16_group_idx
        return self._dispatch_from_expert_to_group(
            scratch, token_selected_experts, token_final_scales,
            sentinel=sentinel)

    def should_skip_group(self, group_idx: int, num_tokens: int) -> bool:
        """Host-side skip check: must be derivable from host state only
        (no tensor reads) so the branch resolves at CUDA-graph capture
        time. Default: never skip.
        """
        return False

    def _dispatch_from_expert_to_group(
        self,
        expert_to_group: torch.Tensor,
        token_selected_experts: torch.Tensor,
        token_final_scales: torch.Tensor,
        sentinel: int = -1,
    ) -> List[GroupDispatchTuple]:
        """Build per-group dispatch tuples using torch.where (fixed shapes)."""
        # INT4-only experts have no BF16 weights loaded -- force back to INT4
        # even if the threshold rule promoted them.
        if self._int4_only_mask is not None:
            expert_to_group[self._int4_only_mask] = self._int4_group_idx
        # BF16-only experts have no INT4 weights -- force to BF16.
        if self._bf16_only_mask is not None:
            expert_to_group[self._bf16_only_mask] = self._bf16_group_idx

        num_groups = self.num_groups
        slot_groups = expert_to_group[token_selected_experts.long()]

        results: List[GroupDispatchTuple] = []
        for gidx in range(num_groups):
            in_group = (slot_groups == gidx)
            experts_g = torch.where(
                in_group, token_selected_experts, sentinel)
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
        bf16_promotion_threshold: int,
        seed: int = 42,
        device: Optional[torch.device] = None,
        int4_only_mask: Optional[torch.Tensor] = None,
        int4_group_idx: int = 0,
        bf16_only_mask: Optional[torch.Tensor] = None,
        bf16_group_idx: int = 1,
    ):
        super().__init__(
            num_experts, group_size_ratios,
            bf16_promotion_threshold=bf16_promotion_threshold,
            device=device,
            int4_only_mask=int4_only_mask, int4_group_idx=int4_group_idx,
            bf16_only_mask=bf16_only_mask, bf16_group_idx=bf16_group_idx,
        )
        gen = torch.Generator(device=self._device).manual_seed(seed)
        scores = torch.rand(
            num_experts, device=self._device, generator=gen)
        # Write the random assignment directly into ``_expert_to_group_buf``
        # at construction time. The ABC's ``dispatch`` does NOT mutate this
        # buffer -- it copies into ``_dispatch_scratch`` first -- so a single
        # init-time write is sufficient for all subsequent calls.
        _assign_by_score_gpu(
            scores, num_experts, group_size_ratios,
            self._expert_to_group_buf, self._group_labels)

    def _assign(self, token_counts, token_selected_experts, token_final_scales):
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
        bf16_promotion_threshold: int,
        confidence_threshold: float = 0.5,
        fallback_seed: int = 42,
        device: Optional[torch.device] = None,
        int4_only_mask: Optional[torch.Tensor] = None,
        int4_group_idx: int = 0,
        bf16_only_mask: Optional[torch.Tensor] = None,
        bf16_group_idx: int = 1,
    ):
        super().__init__(
            num_experts, group_size_ratios,
            bf16_promotion_threshold=bf16_promotion_threshold,
            device=device,
            int4_only_mask=int4_only_mask, int4_group_idx=int4_group_idx,
            bf16_only_mask=bf16_only_mask, bf16_group_idx=bf16_group_idx,
        )
        self._confidence_threshold = confidence_threshold
        self._fallback = RandomHeterDispatch(
            num_experts, group_size_ratios,
            bf16_promotion_threshold=bf16_promotion_threshold,
            seed=fallback_seed, device=device,
            int4_only_mask=int4_only_mask, int4_group_idx=int4_group_idx,
            bf16_only_mask=bf16_only_mask, bf16_group_idx=bf16_group_idx)
        self._expert_weight_sum = torch.empty(
            num_experts, device=self._device, dtype=torch.float32)
        # Reciprocal-of-counts buffer used as the divisor for mean weight.
        # Pre-allocated to avoid per-dispatch alloc; populated each call.
        self._inv_count_buf = torch.empty(
            num_experts, device=self._device, dtype=torch.float32)

    def _assign(self, token_counts, token_selected_experts, token_final_scales):
        if token_selected_experts is None or token_final_scales is None:
            return self._fallback._assign(
                token_counts, token_selected_experts, token_final_scales)

        buf = self._expert_weight_sum
        flat_experts = token_selected_experts.reshape(-1).long()
        flat_scales = token_final_scales.reshape(-1)

        buf.zero_()
        buf.scatter_add_(0, flat_experts, flat_scales)

        # Mean weight = sum(weight) / count, with count clamped to >=1 for
        # zero-routed experts. Do not mutate the ABC's ``token_counts`` buffer
        # in place -- the ABC re-reads it for the threshold rule after we return.
        inv = self._inv_count_buf
        inv.copy_(token_counts).clamp_(min=1.0).reciprocal_()
        buf.mul_(inv)

        return _assign_by_score_gpu(
            buf, self._num_experts, self._group_size_ratios,
            self._expert_to_group_buf, self._group_labels)


class TotalWeightHeterDispatch(HeterDispatchPolicy):
    """Assign by per-expert total routing weight (sum, not mean).

    High-total-weight experts -> last group (high-precision).
    Falls back to random when signals are unavailable.
    """

    def __init__(
        self,
        num_experts: int,
        group_size_ratios: List[float],
        bf16_promotion_threshold: int,
        fallback_seed: int = 42,
        device: Optional[torch.device] = None,
        int4_only_mask: Optional[torch.Tensor] = None,
        int4_group_idx: int = 0,
        bf16_only_mask: Optional[torch.Tensor] = None,
        bf16_group_idx: int = 1,
    ):
        super().__init__(
            num_experts, group_size_ratios,
            bf16_promotion_threshold=bf16_promotion_threshold,
            device=device,
            int4_only_mask=int4_only_mask, int4_group_idx=int4_group_idx,
            bf16_only_mask=bf16_only_mask, bf16_group_idx=bf16_group_idx,
        )
        self._fallback = RandomHeterDispatch(
            num_experts, group_size_ratios,
            bf16_promotion_threshold=bf16_promotion_threshold,
            seed=fallback_seed, device=device,
            int4_only_mask=int4_only_mask, int4_group_idx=int4_group_idx,
            bf16_only_mask=bf16_only_mask, bf16_group_idx=bf16_group_idx)
        self._weight_sum_buf = torch.empty(
            num_experts, device=self._device, dtype=torch.float32)

    def _assign(self, token_counts, token_selected_experts, token_final_scales):
        if token_selected_experts is None or token_final_scales is None:
            return self._fallback._assign(
                token_counts, token_selected_experts, token_final_scales)

        buf = self._weight_sum_buf
        flat_experts = token_selected_experts.reshape(-1).long()
        flat_scales = token_final_scales.reshape(-1)

        buf.zero_()
        buf.scatter_add_(0, flat_experts, flat_scales)

        return _assign_by_score_gpu(
            buf, self._num_experts, self._group_size_ratios,
            self._expert_to_group_buf, self._group_labels)


class HessianWeightedRoutingWeightsDispatch(HeterDispatchPolicy):
    """Assign by importance(E) × per-expert total routing weight.

    Importance is a static, non-negative per-expert vector (typically
    derived offline from Hessian sensitivity, zeroed below the first-order
    noise floor, or all-ones when a layer has no statistically meaningful
    hessian signal). High product -> last group (high-precision).
    Falls back to random when signals are unavailable.

    Threshold-promotion on top: any expert whose routed-token count crosses
    ``bf16_promotion_threshold`` is forced to BF16 by the ABC after this
    policy's score-based assignment, i.e. policy_hot ∪ count_hot.
    ``int4_only_mask`` still wins as a final override.
    """

    def __init__(
        self,
        num_experts: int,
        group_size_ratios: List[float],
        importance: torch.Tensor,
        bf16_promotion_threshold: int,
        fallback_seed: int = 42,
        device: Optional[torch.device] = None,
        int4_only_mask: Optional[torch.Tensor] = None,
        int4_group_idx: int = 0,
        bf16_only_mask: Optional[torch.Tensor] = None,
        bf16_group_idx: int = 1,
    ):
        super().__init__(
            num_experts, group_size_ratios,
            bf16_promotion_threshold=bf16_promotion_threshold,
            device=device,
            int4_only_mask=int4_only_mask, int4_group_idx=int4_group_idx,
            bf16_only_mask=bf16_only_mask, bf16_group_idx=bf16_group_idx,
        )
        assert importance.shape == (num_experts,), (
            f"importance shape {tuple(importance.shape)} "
            f"!= (num_experts={num_experts},)"
        )
        self._importance = importance.to(
            device=self._device, dtype=torch.float32)
        self._fallback = RandomHeterDispatch(
            num_experts, group_size_ratios,
            bf16_promotion_threshold=bf16_promotion_threshold,
            seed=fallback_seed, device=device,
            int4_only_mask=int4_only_mask, int4_group_idx=int4_group_idx,
            bf16_only_mask=bf16_only_mask, bf16_group_idx=bf16_group_idx)
        self._weight_sum_buf = torch.empty(
            num_experts, device=self._device, dtype=torch.float32)

    def _assign(self, token_counts, token_selected_experts, token_final_scales):
        if token_selected_experts is None or token_final_scales is None:
            return self._fallback._assign(
                token_counts, token_selected_experts, token_final_scales)

        buf = self._weight_sum_buf
        flat_experts = token_selected_experts.reshape(-1).long()
        flat_scales = token_final_scales.reshape(-1)

        buf.zero_()
        buf.scatter_add_(0, flat_experts, flat_scales)
        buf.mul_(self._importance)

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
        bf16_promotion_threshold: int,
        fallback_seed: int = 42,
        device: Optional[torch.device] = None,
        int4_only_mask: Optional[torch.Tensor] = None,
        int4_group_idx: int = 0,
        bf16_only_mask: Optional[torch.Tensor] = None,
        bf16_group_idx: int = 1,
    ):
        super().__init__(
            num_experts, group_size_ratios,
            bf16_promotion_threshold=bf16_promotion_threshold,
            device=device,
            int4_only_mask=int4_only_mask, int4_group_idx=int4_group_idx,
            bf16_only_mask=bf16_only_mask, bf16_group_idx=bf16_group_idx,
        )
        self._fallback = RandomHeterDispatch(
            num_experts, group_size_ratios,
            bf16_promotion_threshold=bf16_promotion_threshold,
            seed=fallback_seed, device=device,
            int4_only_mask=int4_only_mask, int4_group_idx=int4_group_idx,
            bf16_only_mask=bf16_only_mask, bf16_group_idx=bf16_group_idx)

    def _assign(self, token_counts, token_selected_experts, token_final_scales):
        if token_selected_experts is None:
            return self._fallback._assign(
                token_counts, token_selected_experts, token_final_scales)
        # ABC's token_counts is the load -- use directly as the score.
        return _assign_by_score_gpu(
            token_counts, self._num_experts, self._group_size_ratios,
            self._expert_to_group_buf, self._group_labels)


class ExpertBatchGatedHeterDispatch(HeterDispatchPolicy):
    """No-scoring dispatch: every expert defaults to INT4.

    The ABC's universal BF16 promotion rule then promotes any expert whose
    routed-token count crosses ``bf16_promotion_threshold`` to BF16.
    ``int4_only`` experts stay INT4 by mask (no BF16 weights loaded).

    Useful as a baseline policy when there is no scoring signal -- the
    behavior reduces to pure count-gated promotion.
    """

    def __init__(
        self,
        num_experts: int,
        group_size_ratios: List[float],
        bf16_promotion_threshold: int,
        device: Optional[torch.device] = None,
        int4_only_mask: Optional[torch.Tensor] = None,
        int4_group_idx: int = 0,
        bf16_only_mask: Optional[torch.Tensor] = None,
        bf16_group_idx: Optional[int] = None,
    ):
        _bf16_gidx = (
            bf16_group_idx if bf16_group_idx is not None
            else (1 - int4_group_idx)
        )
        super().__init__(
            num_experts, group_size_ratios,
            bf16_promotion_threshold=bf16_promotion_threshold,
            device=device,
            int4_only_mask=int4_only_mask, int4_group_idx=int4_group_idx,
            bf16_only_mask=bf16_only_mask, bf16_group_idx=_bf16_gidx,
        )

    def _assign(self, token_counts, token_selected_experts, token_final_scales):
        # Default everyone to INT4; the ABC's threshold rule then promotes
        # >=threshold experts to BF16, and forced masks have the final word.
        self._expert_to_group_buf.fill_(self._int4_group_idx)
        return self._expert_to_group_buf

    def _global_batch_below_threshold(self, num_tokens: int) -> bool:
        return num_tokens < self._bf16_promotion_threshold

    def should_skip_group(self, group_idx: int, num_tokens: int) -> bool:
        # Base assignment is all-INT4, so the BF16 group can only be populated
        # by the threshold rule. When the global batch has fewer than threshold
        # tokens, no expert can cross threshold, so the BF16 group is provably
        # empty and safe to skip.
        if (self._global_batch_below_threshold(num_tokens)
                and group_idx == self._bf16_group_idx):
            return True
        return False


class EfficiencyPromotionPolicy(HeterDispatchPolicy):
    """Curve-driven dynamic-threshold dispatch.

    Reads ``x_star_curve.csv`` (the ground-truth measurement-based optimum
    per global batch size from kernel_profile/) at construction time. On
    every dispatch:

      1. Read M_global = ``token_selected_experts.shape[0]`` (host-known
         under CUDA graph capture).
      2. Look up ``x_runtime`` for this M_global from the curve.
      3. Sort per-expert token counts descending; pick threshold
         ``T = midpoint(counts[x_runtime - 1], counts[x_runtime])`` as a
         GPU 0-d scalar.
      4. ``is_promoted = counts >= T`` → those experts go to BF16.

    When ``x_runtime == 0`` (small-M regime where pure-INT4 wins per
    the curve), no promotion fires; the BF16 group is empty and the
    runtime can skip the BF16 kernel via ``should_skip_group``.

    All hot-path work is GPU-side after the host-known M-keyed branch,
    so the policy is CUDA-graph capturable.
    """

    def __init__(
        self,
        num_experts: int,
        group_size_ratios: List[float],
        bf16_promotion_threshold: int,
        device: Optional[torch.device] = None,
        int4_only_mask: Optional[torch.Tensor] = None,
        int4_group_idx: int = 0,
        bf16_only_mask: Optional[torch.Tensor] = None,
        bf16_group_idx: Optional[int] = None,
        curve_file: Optional[str] = None,
    ):
        _bf16_gidx = (
            bf16_group_idx if bf16_group_idx is not None
            else (1 - int4_group_idx)
        )
        super().__init__(
            num_experts, group_size_ratios,
            bf16_promotion_threshold=bf16_promotion_threshold,
            device=device,
            int4_only_mask=int4_only_mask, int4_group_idx=int4_group_idx,
            bf16_only_mask=bf16_only_mask, bf16_group_idx=_bf16_gidx,
        )
        # M_global → x_runtime lookup table, sorted by M ascending.
        self._curve_M: List[int] = []
        self._curve_x: List[int] = []
        if curve_file is not None:
            self._load_curve(curve_file)
        # Cache of M_global → x_runtime (Python int) so repeated lookups
        # at the same batch shape are O(1).
        self._x_cache: dict = {}

    def _load_curve(self, path: str) -> None:
        """Parse x_star_curve.csv. Expected columns include ``M_global``
        and ``winner_x`` (or ``x_star_meas``/``x_star``)."""
        import csv
        if not os.path.exists(path):
            logger.warning(
                "EfficiencyPromotionPolicy: curve file %s missing; "
                "x_runtime will always be 0 (no promotion).", path)
            return
        with open(path) as f:
            r = csv.reader(f)
            header = next(r)
            # Find x column: prefer winner_x, fall back to x_star_meas / x_star
            x_col = None
            for cand in ("winner_x", "x_star_meas", "x_star"):
                if cand in header:
                    x_col = header.index(cand)
                    break
            if x_col is None:
                raise ValueError(
                    f"curve {path} has no column named winner_x / "
                    f"x_star_meas / x_star (header={header})")
            m_col = header.index("M_global")
            rows = []
            for row in r:
                M = int(row[m_col]); x = int(row[x_col])
                rows.append((M, x))
        rows.sort(key=lambda p: p[0])
        self._curve_M = [p[0] for p in rows]
        self._curve_x = [p[1] for p in rows]

    def _lookup_x_runtime(self, M_global: int) -> int:
        """Return x_runtime via nearest-M lookup. Cached per M."""
        if M_global in self._x_cache:
            return self._x_cache[M_global]
        if not self._curve_M:
            x = 0
        else:
            x = min(
                range(len(self._curve_M)),
                key=lambda i: abs(self._curve_M[i] - M_global),
            )
            x = self._curve_x[x]
            x = max(0, min(x, self._num_experts))
        self._x_cache[M_global] = x
        return x

    def _assign(self, token_counts, token_selected_experts, token_final_scales):
        # Default everyone to INT4 (the policy decides nothing here; the
        # dynamic-threshold step in dispatch promotes to BF16).
        self._expert_to_group_buf.fill_(self._int4_group_idx)
        return self._expert_to_group_buf

    def dispatch(
        self,
        token_selected_experts: torch.Tensor,
        token_final_scales: torch.Tensor,
        sentinel: int = -1,
    ) -> List[GroupDispatchTuple]:
        token_counts = self._compute_token_counts(token_selected_experts)
        policy_assignment = self._assign(
            token_counts, token_selected_experts, token_final_scales,
        )
        scratch = self._dispatch_scratch
        scratch.copy_(policy_assignment)

        # Host-known M_global → host-known x_runtime → host-known branch.
        # The Python ``if`` resolves at CUDA-graph capture time so the
        # captured graph is shape-specific; replays at the same shape skip
        # this branch entirely.
        M_global = token_selected_experts.shape[0]
        x_runtime = self._lookup_x_runtime(M_global)

        if x_runtime > 0 and x_runtime < self._num_experts:
            sorted_counts, _ = token_counts.sort(descending=True)
            # T = midpoint as a GPU 0-d tensor; counts >= T then promotes
            # exactly the top-x_runtime experts (modulo ties).
            T = (sorted_counts[x_runtime - 1] + sorted_counts[x_runtime]) / 2
            is_promoted = token_counts >= T
            scratch[is_promoted] = self._bf16_group_idx
        elif x_runtime >= self._num_experts:
            scratch.fill_(self._bf16_group_idx)
        # else x_runtime == 0: keep all-INT4

        return self._dispatch_from_expert_to_group(
            scratch, token_selected_experts, token_final_scales,
            sentinel=sentinel)

    def should_skip_group(self, group_idx: int, num_tokens: int) -> bool:
        # When the curve says x*=0 at this M, no expert can be promoted —
        # safe to skip the BF16 group entirely (also skips its kernel call).
        if (group_idx == self._bf16_group_idx
                and self._lookup_x_runtime(num_tokens) == 0):
            return True
        return False


_POLICY_REGISTRY = {
    "expert_load": ExpertLoadHeterDispatch,
    "confidence": ConfidenceThresholdHeterDispatch,
    "total_weight": TotalWeightHeterDispatch,
    "hessian_weighted_routing_weights": HessianWeightedRoutingWeightsDispatch,
    "random": RandomHeterDispatch,
    "expert_batch": ExpertBatchGatedHeterDispatch,
    "efficiency_promotion": EfficiencyPromotionPolicy,
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
