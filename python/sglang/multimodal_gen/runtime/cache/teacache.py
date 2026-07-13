# SPDX-License-Identifier: Apache-2.0
"""
TeaCache: Temporal similarity-based caching for diffusion models.

TeaCache accelerates diffusion inference by selectively skipping redundant
computation when consecutive diffusion steps are similar enough. This is
achieved by tracking the L1 distance between modulated inputs across timesteps.

Key concepts:
- Modulated input: The input to transformer blocks after timestep conditioning
- L1 distance: Measures how different consecutive timesteps are
- Threshold: When accumulated L1 distance exceeds threshold, force computation
- CFG support: Separate caches for positive and negative branches

References:
- TeaCache: Accelerating Diffusion Models with Temporal Similarity
  https://arxiv.org/abs/2411.14324
"""

from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from sglang.multimodal_gen.configs.models import DiTConfig

if TYPE_CHECKING:
    from sglang.multimodal_gen.configs.sample.teacache import TeaCacheParams


# Scalar attributes shared by every TeaCache-capable model.
TEACACHE_SCALAR_FIELDS: tuple[str, ...] = (
    "cnt",
    "enable_teacache",
    "is_cfg_negative",
    "accumulated_rel_l1_distance",
    "accumulated_rel_l1_distance_negative",
)
# Tensor attributes (positive and CFG-negative caches).
TEACACHE_TENSOR_FIELDS: tuple[str, ...] = (
    "previous_modulated_input",
    "previous_residual",
    "previous_modulated_input_negative",
    "previous_residual_negative",
)
# Negative-branch fields exist on the model only when it supports CFG caches.
_TEACACHE_NEGATIVE_FIELDS: frozenset[str] = frozenset(
    name
    for name in TEACACHE_SCALAR_FIELDS + TEACACHE_TENSOR_FIELDS
    if name.endswith("_negative")
)


def compute_l1_decision(
    modulated_inp: torch.Tensor,
    previous_modulated_input: torch.Tensor | None,
    accumulated_rel_l1_distance: float,
    coefficients: list[float],
    teacache_thresh: float,
) -> tuple[float, bool]:
    """Pure L1-distance cache decision shared by attr- and state-based callers.

    Returns ``(new_accumulated_distance, should_calc)``.
    """
    if previous_modulated_input is None:
        return 0.0, True

    diff = modulated_inp - previous_modulated_input
    rel_l1 = (diff.abs().mean() / previous_modulated_input.abs().mean()).cpu().item()
    rescale_func = np.poly1d(coefficients)
    accumulated = accumulated_rel_l1_distance + rescale_func(rel_l1)

    if accumulated >= teacache_thresh:
        # Threshold exceeded: force compute and reset accumulator.
        return 0.0, True
    # Cache hit: keep accumulated distance.
    return accumulated, False


@dataclass
class TeaCacheRequestState:
    """Explicit, per-request snapshot of TeaCache state.

    This is the canonical interchange format between a shared DiT model and
    per-request bookkeeping (continuous batching, drain/resume, packed
    row plans). Field names intentionally mirror the model attributes managed
    by :class:`TeaCacheMixin`.
    """

    cnt: int = 0
    enable_teacache: bool = True
    is_cfg_negative: bool = False
    accumulated_rel_l1_distance: float = 0.0
    accumulated_rel_l1_distance_negative: float = 0.0
    previous_modulated_input: torch.Tensor | None = None
    previous_residual: torch.Tensor | None = None
    previous_modulated_input_negative: torch.Tensor | None = None
    previous_residual_negative: torch.Tensor | None = None

    @classmethod
    def capture_from(cls, model: Any) -> "TeaCacheRequestState":
        """Capture the model's current TeaCache attributes into a snapshot."""
        state = cls()
        for name in TEACACHE_SCALAR_FIELDS + TEACACHE_TENSOR_FIELDS:
            if hasattr(model, name):
                setattr(state, name, getattr(model, name))
        return state

    def install_to(self, model: Any) -> None:
        """Install this snapshot onto the model's TeaCache attributes."""
        supports_cfg_cache = bool(getattr(model, "_supports_cfg_cache", False))
        for name in TEACACHE_SCALAR_FIELDS + TEACACHE_TENSOR_FIELDS:
            if name in _TEACACHE_NEGATIVE_FIELDS and not supports_cfg_cache:
                continue
            setattr(model, name, getattr(self, name))

    def reset(self) -> None:
        """Reset to a fresh generation-start state (mirrors the mixin reset)."""
        self.cnt = 0
        self.enable_teacache = True
        self.is_cfg_negative = False
        self.accumulated_rel_l1_distance = 0.0
        self.accumulated_rel_l1_distance_negative = 0.0
        self.previous_modulated_input = None
        self.previous_residual = None
        self.previous_modulated_input_negative = None
        self.previous_residual_negative = None

    def for_next_phase(self) -> "TeaCacheRequestState":
        """Seed the state for a new transformer phase (e.g. Wan2.2 experts).

        The forward-pass counter carries across the boundary so skip windows
        keep their meaning over the whole schedule, but cached tensors never
        cross models: each expert must rebuild its own baselines.
        """
        return TeaCacheRequestState(
            cnt=self.cnt,
            enable_teacache=self.enable_teacache,
            is_cfg_negative=self.is_cfg_negative,
        )

    def clone(self) -> "TeaCacheRequestState":
        return TeaCacheRequestState(
            **{item.name: getattr(self, item.name) for item in fields(self)}
        )

    def to_payload(self) -> dict[str, Any]:
        """Serialize for drain/resume; tensors move to CPU."""
        payload: dict[str, Any] = {}
        for item in fields(self):
            value = getattr(self, item.name)
            if isinstance(value, torch.Tensor):
                value = value.detach().to("cpu")
            payload[item.name] = value
        return payload

    @classmethod
    def from_payload(
        cls,
        payload: dict[str, Any],
        device: torch.device | str | None = None,
    ) -> "TeaCacheRequestState":
        state = cls()
        valid = {item.name for item in fields(cls)}
        for name, value in payload.items():
            if name not in valid:
                continue
            if isinstance(value, torch.Tensor) and device is not None:
                value = value.to(device=device)
            setattr(state, name, value)
        return state


def resolve_teacache_phase_state(
    phase_states: dict[str, "TeaCacheRequestState | None"],
    phase: str,
    *,
    create: bool = False,
) -> "TeaCacheRequestState | None":
    """Return (and lazily seed) the per-phase snapshot for one request.

    Entering a phase for the first time seeds from the most recent other
    phase via :meth:`TeaCacheRequestState.for_next_phase`, carrying the
    forward counter but never cached tensors across experts. When ``create``
    is True a missing snapshot is materialized as a fresh state so callers
    can mutate it in place (packed row plans).
    """
    if phase in phase_states:
        state = phase_states[phase]
        if state is None and create:
            state = TeaCacheRequestState()
            phase_states[phase] = state
        return state
    seed: TeaCacheRequestState | None = None
    for previous_phase, previous_state in phase_states.items():
        if previous_phase == phase or previous_state is None:
            continue
        for_next_phase = getattr(previous_state, "for_next_phase", None)
        if callable(for_next_phase):
            seed = for_next_phase()
    if seed is None and create:
        seed = TeaCacheRequestState()
    phase_states[phase] = seed
    return seed


@dataclass
class TeaCachePackedMember:
    """Per-request TeaCache bookkeeping for one member of a packed step batch.

    ``state is None`` means TeaCache is disabled for this member: its rows are
    always computed and no state is updated.
    """

    row_slice: slice
    state: TeaCacheRequestState | None = None
    step_index: int = 0
    num_inference_steps: int = 0
    do_cfg: bool = False
    teacache_params: Any = None

    @property
    def enabled(self) -> bool:
        return (
            self.state is not None
            and self.state.enable_teacache
            and self.teacache_params is not None
        )

    def maybe_reset_for_first_step(self, is_cfg_negative: bool) -> None:
        """Mirror the mixin reset at the first positive-branch step."""
        if self.state is None:
            return
        if self.step_index == 0 and not self.state.is_cfg_negative:
            enable = self.state.enable_teacache
            self.state.reset()
            self.state.enable_teacache = enable
        self.state.is_cfg_negative = is_cfg_negative

    def decide(self, modulated_inp: torch.Tensor, is_cfg_negative: bool) -> bool:
        """Return True when this member's rows must run the model forward.

        Mirrors ``TeaCacheMixin._compute_teacache_decision`` semantics on the
        member's own state instead of shared model attributes.
        """
        if not self.enabled:
            return True
        state = self.state
        self.maybe_reset_for_first_step(is_cfg_negative)

        params = self.teacache_params
        start_skipping, end_skipping = params.get_skip_boundaries(
            self.num_inference_steps, self.do_cfg
        )
        is_boundary_step = state.cnt < start_skipping or state.cnt >= end_skipping

        if is_boundary_step:
            new_accum, should_calc = 0.0, True
        else:
            if is_cfg_negative:
                prev_inp = state.previous_modulated_input_negative
                accumulated = state.accumulated_rel_l1_distance_negative
            else:
                prev_inp = state.previous_modulated_input
                accumulated = state.accumulated_rel_l1_distance
            new_accum, should_calc = compute_l1_decision(
                modulated_inp=modulated_inp,
                previous_modulated_input=prev_inp,
                accumulated_rel_l1_distance=accumulated,
                coefficients=params.get_coefficients(),
                teacache_thresh=params.teacache_thresh,
            )

        if is_cfg_negative:
            state.previous_modulated_input_negative = modulated_inp.clone()
            state.accumulated_rel_l1_distance_negative = new_accum
        else:
            state.previous_modulated_input = modulated_inp.clone()
            state.accumulated_rel_l1_distance = new_accum
        return should_calc

    def stash_residual(self, residual: torch.Tensor, is_cfg_negative: bool) -> None:
        if self.state is None:
            return
        if is_cfg_negative:
            self.state.previous_residual_negative = residual
        else:
            self.state.previous_residual = residual

    def cached_residual(self, is_cfg_negative: bool) -> torch.Tensor | None:
        if self.state is None:
            return None
        if is_cfg_negative:
            return self.state.previous_residual_negative
        return self.state.previous_residual

    def advance(self) -> None:
        """Count one model forward for this member (mirrors ``self.cnt += 1``)."""
        if self.state is not None:
            self.state.cnt += 1


@dataclass
class TeaCachePackedPlan:
    """Row-level TeaCache plan for one packed forward.

    Models that set ``supports_packed_teacache = True`` read this plan from
    the forward context and gather/scatter compute rows around their block
    loop so cache-hit rows skip the transformer blocks entirely.
    """

    members: list[TeaCachePackedMember] = field(default_factory=list)

    @property
    def any_enabled(self) -> bool:
        return any(member.enabled for member in self.members)

    def partition(
        self,
        modulated_inp: torch.Tensor,
        is_cfg_negative: bool,
    ) -> tuple[list[TeaCachePackedMember], list[TeaCachePackedMember]]:
        """Split members into (compute, skip) using per-member decisions.

        ``modulated_inp`` is indexed with each member's ``row_slice``; rows
        beyond the members' coverage (e.g. bucket padding) are always computed
        by the caller.
        """
        compute: list[TeaCachePackedMember] = []
        skip: list[TeaCachePackedMember] = []
        for member in self.members:
            member_inp = modulated_inp[member.row_slice]
            if member.decide(member_inp, is_cfg_negative):
                compute.append(member)
            else:
                skip.append(member)
        return compute, skip


@dataclass
class TeaCacheContext:
    """Common context extracted for TeaCache skip decision.

    This context is populated from the forward_batch and forward_context
    during each denoising step, providing all information needed to make
    cache decisions.

    Attributes:
        current_timestep: Current denoising timestep index (0-indexed).
        num_inference_steps: Total number of inference steps.
        do_cfg: Whether classifier-free guidance is enabled.
        is_cfg_negative: True if currently processing negative CFG branch.
        teacache_thresh: Threshold for accumulated L1 distance.
        coefficients: Polynomial coefficients for L1 rescaling.
        teacache_params: Full TeaCacheParams for model-specific access.
    """

    current_timestep: int
    num_inference_steps: int
    do_cfg: bool
    is_cfg_negative: bool  # For CFG branch selection
    teacache_thresh: float
    coefficients: list[float]
    teacache_params: "TeaCacheParams"  # Full params for model-specific access


class TeaCacheMixin:
    """
    Mixin class providing TeaCache optimization functionality.

    TeaCache accelerates diffusion inference by selectively skipping redundant
    computation when consecutive diffusion steps are similar enough.

    This mixin should be inherited by DiT model classes that want to support
    TeaCache optimization. It provides:
    - State management for tracking L1 distances
    - CFG-aware caching (separate caches for positive/negative branches)
    - Decision logic for when to compute vs. use cache

    Example usage in a DiT model:
        class MyDiT(TeaCacheMixin, BaseDiT):
            def __init__(self, config, **kwargs):
                super().__init__(config, **kwargs)
                self._init_teacache_state()

            def forward(self, hidden_states, timestep, ...):
                ctx = self._get_teacache_context()
                if ctx is not None:
                    # Compute modulated input (model-specific, e.g., after timestep embedding)
                    modulated_input = self._compute_modulated_input(hidden_states, timestep)
                    is_boundary = (ctx.current_timestep == 0 or
                                   ctx.current_timestep >= ctx.num_inference_steps - 1)

                    should_calc = self._compute_teacache_decision(
                        modulated_inp=modulated_input,
                        is_boundary_step=is_boundary,
                        coefficients=ctx.coefficients,
                        teacache_thresh=ctx.teacache_thresh,
                    )

                    if not should_calc:
                        # Use cached residual (must implement retrieve_cached_states)
                        return self.retrieve_cached_states(hidden_states)

                # Normal forward pass...
                output = self._transformer_forward(hidden_states, timestep, ...)

                # Cache states for next step
                if ctx is not None:
                    self.maybe_cache_states(output, hidden_states)

                return output

    Subclass implementation notes:
        - `_compute_modulated_input()`: Model-specific method to compute the input
          after timestep conditioning (used for L1 distance calculation)
        - `retrieve_cached_states()`: Must be overridden to return cached output
        - `maybe_cache_states()`: Override to store states for cache retrieval

    Attributes:
        cnt: Counter for tracking steps.
        enable_teacache: Whether TeaCache is enabled.
        previous_modulated_input: Cached modulated input for positive branch.
        previous_residual: Cached residual for positive branch.
        accumulated_rel_l1_distance: Accumulated L1 distance for positive branch.
        is_cfg_negative: Whether currently processing negative CFG branch.
        _supports_cfg_cache: Whether this model supports CFG cache separation.

    CFG-specific attributes (only when _supports_cfg_cache is True):
        previous_modulated_input_negative: Cached input for negative branch.
        previous_residual_negative: Cached residual for negative branch.
        accumulated_rel_l1_distance_negative: L1 distance for negative branch.
    """

    # Models that support CFG cache separation (wan/hunyuan/zimage)
    # Models not in this set (flux/qwen) auto-disable TeaCache when CFG is enabled
    _CFG_SUPPORTED_PREFIXES: set[str] = {"wan", "hunyuan", "zimage"}
    config: DiTConfig

    def _init_teacache_state(self) -> None:
        """Initialize TeaCache state. Call this in subclass __init__."""
        # Common TeaCache state
        self.cnt = 0
        self.enable_teacache = True
        # Flag indicating if this model supports CFG cache separation
        self._supports_cfg_cache = (
            self.config.prefix.lower() in self._CFG_SUPPORTED_PREFIXES
        )

        # Always initialize positive cache fields (used in all modes)
        self.previous_modulated_input: torch.Tensor | None = None
        self.previous_residual: torch.Tensor | None = None
        self.accumulated_rel_l1_distance: float = 0.0

        self.is_cfg_negative = False
        # CFG-specific fields initialized to None (created when CFG is used)
        # These are only used when _supports_cfg_cache is True AND do_cfg is True
        if self._supports_cfg_cache:
            self.previous_modulated_input_negative: torch.Tensor | None = None
            self.previous_residual_negative: torch.Tensor | None = None
            self.accumulated_rel_l1_distance_negative: float = 0.0

    def reset_teacache_state(self) -> None:
        """Reset all TeaCache state at the start of each generation task."""
        self.cnt = 0

        # Primary cache fields (always present)
        self.previous_modulated_input = None
        self.previous_residual = None
        self.accumulated_rel_l1_distance = 0.0
        self.is_cfg_negative = False
        self.enable_teacache = True
        # CFG negative cache fields (always reset, may be unused)
        if self._supports_cfg_cache:
            self.previous_modulated_input_negative = None
            self.previous_residual_negative = None
            self.accumulated_rel_l1_distance_negative = 0.0

    def capture_teacache_state(self) -> TeaCacheRequestState:
        """Explicit cache-state interface: snapshot the model's TeaCache state."""
        return TeaCacheRequestState.capture_from(self)

    def install_teacache_state(self, state: TeaCacheRequestState | None) -> None:
        """Explicit cache-state interface: install (or reset when None)."""
        if state is None:
            self.reset_teacache_state()
        else:
            state.install_to(self)

    def _compute_l1_and_decide(
        self,
        modulated_inp: torch.Tensor,
        coefficients: list[float],
        teacache_thresh: float,
    ) -> tuple[float, bool]:
        """
        Compute L1 distance and decide whether to calculate or use cache.

        Args:
            modulated_inp: Current timestep's modulated input.
            coefficients: Polynomial coefficients for L1 rescaling.
            teacache_thresh: Threshold for cache decision.

        Returns:
            Tuple of (new_accumulated_distance, should_calc).
        """
        prev_modulated_inp = (
            self.previous_modulated_input_negative
            if self.is_cfg_negative
            else self.previous_modulated_input
        )
        accumulated_rel_l1_distance = (
            self.accumulated_rel_l1_distance_negative
            if self.is_cfg_negative
            else self.accumulated_rel_l1_distance
        )
        return compute_l1_decision(
            modulated_inp=modulated_inp,
            previous_modulated_input=prev_modulated_inp,
            accumulated_rel_l1_distance=accumulated_rel_l1_distance,
            coefficients=coefficients,
            teacache_thresh=teacache_thresh,
        )

    def _compute_teacache_decision(
        self,
        modulated_inp: torch.Tensor,
        is_boundary_step: bool,
        coefficients: list[float],
        teacache_thresh: float,
    ) -> bool:
        """
        Compute cache decision for TeaCache.

        Args:
            modulated_inp: Current timestep's modulated input.
            is_boundary_step: True for boundary timesteps that always compute.
            coefficients: Polynomial coefficients for L1 rescaling.
            teacache_thresh: Threshold for cache decision.

        Returns:
            True if forward computation is needed, False to use cache.
        """
        if not self.enable_teacache:
            return True

        if is_boundary_step:
            new_accum, should_calc = 0.0, True
        else:
            new_accum, should_calc = self._compute_l1_and_decide(
                modulated_inp=modulated_inp,
                coefficients=coefficients,
                teacache_thresh=teacache_thresh,
            )

        # Advance baseline and accumulator for the active branch
        if not self.is_cfg_negative:
            self.previous_modulated_input = modulated_inp.clone()
            self.accumulated_rel_l1_distance = new_accum
        elif self._supports_cfg_cache:
            self.previous_modulated_input_negative = modulated_inp.clone()
            self.accumulated_rel_l1_distance_negative = new_accum

        return should_calc

    def _get_teacache_context(self) -> TeaCacheContext | None:
        """
        Check TeaCache preconditions and extract common context.

        Returns:
            TeaCacheContext if TeaCache is enabled and properly configured,
            None if should skip TeaCache logic entirely.
        """
        from sglang.multimodal_gen.runtime.managers.forward_context import (
            get_forward_context,
        )

        forward_context = get_forward_context()
        forward_batch = forward_context.forward_batch

        # Early return checks
        if (
            forward_batch is None
            or not forward_batch.enable_teacache
            or forward_batch.teacache_params is None
        ):
            return None

        teacache_params = forward_batch.teacache_params

        # Extract common values
        current_timestep = forward_context.current_timestep
        num_inference_steps = forward_batch.num_inference_steps
        do_cfg = forward_batch.do_classifier_free_guidance
        is_cfg_negative = forward_batch.is_cfg_negative

        # Reset at first timestep
        if current_timestep == 0 and not self.is_cfg_negative:
            self.reset_teacache_state()

        return TeaCacheContext(
            current_timestep=current_timestep,
            num_inference_steps=num_inference_steps,
            do_cfg=do_cfg,
            is_cfg_negative=is_cfg_negative,
            teacache_thresh=teacache_params.teacache_thresh,
            coefficients=teacache_params.get_coefficients(),
            teacache_params=teacache_params,
        )

    def maybe_cache_states(
        self, hidden_states: torch.Tensor, original_hidden_states: torch.Tensor
    ) -> None:
        """Cache states for later retrieval. Override in subclass if needed."""
        pass

    def should_skip_forward_for_cached_states(self, **kwargs: dict[str, Any]) -> bool:
        """Check if forward can be skipped using cached states."""
        return False

    def retrieve_cached_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Retrieve cached states. Must be implemented by subclass."""
        raise NotImplementedError("retrieve_cached_states is not implemented")
