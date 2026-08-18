# SPDX-License-Identifier: Apache-2.0
"""
Step-reuse contract for stateful iterative (denoising) models.

This module gives model adapters a way to say "this real prediction is
similar enough that following steps may reuse it" without hard-coding one
universal similarity metric into the runtime, and without conflating a
reusable *prediction* with persistent *session/KV state*.

The runtime (this module) owns:
  - one independent lifecycle (:class:`StepReuseState`) per declared
    ``state_scope`` key (e.g. per request, per modality, per CFG branch);
  - enforcing ``max_skip_steps`` and the declared force-real points;
  - forcing a real forward whenever a :class:`StepSideEffectContract`
    declares a required terminal write;
  - never feeding a *reused* prediction back into the similarity history.

The model adapter owns:
  - the ``decide_reuse`` callable, which inspects an ``observation`` taken at
    ``policy.observation_point`` (e.g. "post_cfg_velocity") and decides
    whether the just-computed real prediction is similar enough to the
    recent history to be worth reusing.

Nothing here is GPU- or torch-specific: predictions and observations are
opaque, adapter-supplied objects, so the whole contract can be exercised
with plain Python values in unit tests.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Deque,
    Dict,
    FrozenSet,
    Hashable,
    Optional,
    Tuple,
)

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

if TYPE_CHECKING:
    import torch

    from sglang.multimodal_gen.configs.models import DiTConfig

logger = init_logger(__name__)

ScopeKey = Tuple[Hashable, ...]

FORCE_FIRST_ONE = "first_one"
FORCE_FIRST_TWO = "first_two"
FORCE_TERMINAL = "terminal"
_KNOWN_FORCE_POINTS = frozenset({FORCE_FIRST_ONE, FORCE_FIRST_TWO, FORCE_TERMINAL})


class StepReuseError(RuntimeError):
    """Raised when the step-reuse contract is violated by the caller."""


@dataclass(frozen=True)
class StepReusePolicy:
    """Declares when a model may reuse a previous real prediction.

    Attributes:
        policy_name: Human-readable identifier for logging/metrics.
        observation_point: Where in the forward the adapter captured the
            observation used for the similarity decision (e.g.
            "post_cfg_velocity"). Informational for the runtime; the
            adapter is the only party that interprets it.
        history_size: Max number of past observations kept per scope.
        max_skip_steps: Max consecutive steps that may reuse one real
            prediction before a real forward is forced again.
        force_real_steps: Named force points that always require a real
            forward, regardless of the similarity decision. Supported
            values: "first_two", "terminal".
        state_scope: Names of the axes that partition lifecycle state
            (e.g. ("request", "modality", "cfg_role")). Purely
            documentation for the policy; callers key state explicitly via
            ``scope_key`` on every call.
    """

    policy_name: str
    observation_point: str
    history_size: int
    max_skip_steps: int
    force_real_steps: FrozenSet[str] = frozenset()
    state_scope: Tuple[str, ...] = ("request",)

    def __post_init__(self) -> None:
        if self.history_size < 0:
            raise ValueError("history_size must be >= 0")
        if self.max_skip_steps < 0:
            raise ValueError("max_skip_steps must be >= 0")
        unknown = self.force_real_steps - _KNOWN_FORCE_POINTS
        if unknown:
            raise ValueError(f"unknown force_real_steps entries: {sorted(unknown)}")


@dataclass
class StepReuseState:
    """Mutable lifecycle state for one scope (e.g. one request).

    ``last_real_prediction`` is intentionally separate from any persistent
    session/KV state the model may hold elsewhere: this state tracks only
    what the runtime needs to make and audit reuse decisions.
    """

    last_real_prediction: Any = None
    real_history: Deque[Any] = field(default_factory=deque)
    skip_remaining: int = 0
    real_forwards: int = 0
    reused_steps: int = 0
    total_steps_seen: int = 0

    def reset(self) -> None:
        self.last_real_prediction = None
        self.real_history.clear()
        self.skip_remaining = 0
        self.real_forwards = 0
        self.reused_steps = 0
        self.total_steps_seen = 0


@dataclass(frozen=True)
class StepSideEffectContract:
    """Declares side effects a real forward commits that reuse cannot skip.

    Attributes:
        terminal_write_required: If True, the final step of the rollout
            (``step_index == total_steps - 1``) always forces a real
            forward, even if the reuse policy would otherwise allow a skip.
        write_tags: Free-form labels for the writes being protected, for
            logging/observability only.
    """

    terminal_write_required: bool = False
    write_tags: FrozenSet[str] = frozenset()


# decide_reuse(state, observation) -> True if the just-recorded real
# prediction is similar enough to warrant opening a reuse window.
#
# Under tensor/sequence parallelism this callable is also where
# distributed agreement belongs: every rank must reach the same skip/real
# decision for a shared step, so a parallel-aware adapter should
# all_reduce its local similarity signal *inside* decide_reuse before
# returning -- the same pattern cache_dit_integration.py's
# ``patched_similarity`` already uses for its own similarity check. The
# controller itself stays parallelism-agnostic; it just trusts whatever
# boolean decide_reuse returns.
DecideReuseFn = Callable[[StepReuseState, Any], bool]


class StepReuseController:
    """Runtime-owned controller implementing the step-reuse contract.

    One controller instance is expected per policy (i.e. per model/pipeline
    that declares a :class:`StepReusePolicy`). It multiplexes independent
    :class:`StepReuseState` lifecycles across arbitrary ``scope_key`` tuples,
    so unrelated modalities, CFG branches, or requests never share reuse
    decisions.
    """

    def __init__(self, policy: StepReusePolicy, decide_reuse: DecideReuseFn):
        self._policy = policy
        self._decide_reuse = decide_reuse
        self._states: Dict[ScopeKey, StepReuseState] = {}

    @property
    def policy(self) -> StepReusePolicy:
        return self._policy

    def _state(self, scope_key: ScopeKey) -> StepReuseState:
        return self._states.setdefault(scope_key, StepReuseState())

    def _peek_state(self, scope_key: ScopeKey) -> StepReuseState:
        """Read-only state lookup.

        Unlike ``_state``, this never creates or persists a new entry in
        ``self._states`` for a scope that hasn't recorded anything yet, so
        purely observational calls (``should_reuse``, ``metrics``,
        ``get_reused_prediction``) can't silently leak empty scope entries.
        """
        return self._states.get(scope_key, StepReuseState())

    def reset(self, scope_key: ScopeKey) -> None:
        """Drop all lifecycle state for one scope (e.g. on request end)."""
        self._states.pop(scope_key, None)

    def reset_all(self) -> None:
        self._states.clear()

    def metrics(self, scope_key: ScopeKey) -> Dict[str, int]:
        state = self._peek_state(scope_key)
        return {
            "real_forwards": state.real_forwards,
            "reused_steps": state.reused_steps,
            "total_steps_seen": state.total_steps_seen,
        }

    def _is_forced(
        self,
        step_index: int,
        total_steps: int,
        side_effects: Optional[StepSideEffectContract],
    ) -> bool:
        force = self._policy.force_real_steps
        if FORCE_FIRST_ONE in force and step_index < 1:
            return True
        if FORCE_FIRST_TWO in force and step_index < 2:
            return True
        if FORCE_TERMINAL in force and step_index == total_steps - 1:
            return True
        if (
            side_effects is not None
            and side_effects.terminal_write_required
            and step_index == total_steps - 1
        ):
            return True
        return False

    def should_reuse(
        self,
        scope_key: ScopeKey,
        step_index: int,
        total_steps: int,
        side_effects: Optional[StepSideEffectContract] = None,
    ) -> bool:
        """Decide whether ``step_index`` may reuse the last real prediction.

        This never mutates state, so it is safe to call more than once per
        step (e.g. for logging) before committing to ``record_reuse`` or
        ``record_real``. The scheduler update for this step must still run
        regardless of the outcome; only the transformer forward is skipped
        when this returns True.
        """
        if total_steps <= 0:
            raise StepReuseError("total_steps must be positive")
        if step_index < 0 or step_index >= total_steps:
            raise StepReuseError(
                f"step_index {step_index} out of range for total_steps {total_steps}"
            )

        if self._is_forced(step_index, total_steps, side_effects):
            return False

        state = self._peek_state(scope_key)
        if state.last_real_prediction is None:
            return False
        return state.skip_remaining > 0

    def get_reused_prediction(self, scope_key: ScopeKey) -> Any:
        state = self._peek_state(scope_key)
        if state.last_real_prediction is None:
            raise StepReuseError(
                f"no real prediction recorded yet for scope {scope_key!r}"
            )
        return state.last_real_prediction

    def record_reuse(self, scope_key: ScopeKey) -> None:
        """Record that a step actually reused the cached prediction.

        Intentionally does not touch ``real_history``: a reused value must
        never be treated as a new similarity observation, otherwise error
        could accumulate silently across a skip run.
        """
        state = self._state(scope_key)
        if state.skip_remaining <= 0:
            raise StepReuseError(
                f"record_reuse called for scope {scope_key!r} with no skip budget remaining"
            )
        state.skip_remaining -= 1
        state.reused_steps += 1
        state.total_steps_seen += 1

    def record_real(
        self, scope_key: ScopeKey, prediction: Any, observation: Any
    ) -> bool:
        """Record that a step ran a real forward, and decide the next skip window.

        ``observation`` is the adapter-captured quantity at
        ``policy.observation_point`` for *this* real forward. It is compared
        against prior history by ``decide_reuse`` before being appended, so
        the decision never sees itself as history. Returns whether a skip
        window was opened for subsequent steps.
        """
        state = self._state(scope_key)

        can_skip = False
        if state.real_history and self._policy.max_skip_steps > 0:
            can_skip = bool(self._decide_reuse(state, observation))

        state.last_real_prediction = prediction
        if self._policy.history_size > 0:
            state.real_history.append(observation)
            while len(state.real_history) > self._policy.history_size:
                state.real_history.popleft()

        state.skip_remaining = self._policy.max_skip_steps if can_skip else 0
        state.real_forwards += 1
        state.total_steps_seen += 1
        return can_skip


class StepReuseMixin:
    """
    Mixin providing the step-reuse skip-forward strategy for DiT models.

    Mirrors ``TeaCacheMixin``'s integration pattern (see
    ``sglang.multimodal_gen.runtime.cache.teacache``) for models that opt
    into the generic step-reuse contract above instead of TeaCache's
    model-tuned polynomial rescaling: a plain relative-L1 threshold on the
    modulated input, with an explicit cap (``max_skip_steps``) on how many
    consecutive steps may reuse one real prediction, and the first/terminal
    step always forced real.

    Example usage in a DiT model (matching ``WanTransformer3DModel``'s
    TeaCache call sites):

        class MyDiT(StepReuseMixin, BaseDiT):
            def __init__(self, config, **kwargs):
                super().__init__(config, **kwargs)
                self._init_step_reuse_state()

            def forward(self, hidden_states, timestep, ...):
                # Unlike TeaCache, the skip decision is made once per real
                # forward (opening a skip *window* of up to max_skip_steps),
                # not re-evaluated against modulated_inp on every step -- so
                # checking whether to skip needs no observation at all.
                if self.should_skip_forward_for_step_reuse():
                    return self.retrieve_step_reuse_prediction(hidden_states)

                modulated_inp = temb  # or timestep_proj, model-specific
                original_hidden_states = hidden_states.clone()
                for block in self.blocks:
                    hidden_states = block(hidden_states, ...)
                self.maybe_record_step_reuse(hidden_states, original_hidden_states, modulated_inp=modulated_inp)
                return hidden_states

    Attributes:
        is_cfg_negative: Whether currently processing the negative CFG
            branch (mirrors ``TeaCacheMixin``, used as the scope key so
            positive/negative branches never share a reuse decision).
    """

    config: DiTConfig

    def _init_step_reuse_state(self) -> None:
        """Initialize step-reuse state. Call this in subclass ``__init__``."""
        self._step_reuse_controller: Optional[StepReuseController] = None
        self._step_reuse_params_snapshot: Optional[Tuple[float, int, int]] = None
        self.is_cfg_negative = getattr(self, "is_cfg_negative", False)
        # Armed whenever we're not at timestep 0, so the *first* timestep-0
        # call of a new generation task resets state exactly once,
        # regardless of whether the positive or negative CFG branch happens
        # to be dispatched first. (Guarding purely on
        # "not self.is_cfg_negative", as TeaCacheMixin does, is only correct
        # if the negative branch is always dispatched first at each
        # timestep; this stays correct either way.)
        self._step_reuse_reset_armed = True

    def _maybe_reset_step_reuse_for_new_task(self, current_timestep: int) -> None:
        if current_timestep == 0:
            if self._step_reuse_reset_armed:
                self.reset_step_reuse_state()
                self._step_reuse_reset_armed = False
        else:
            self._step_reuse_reset_armed = True

    def reset_step_reuse_state(self) -> None:
        """Reset all step-reuse state at the start of each generation task."""
        if self._step_reuse_controller is not None:
            self._step_reuse_controller.reset_all()

    def _decide_reuse_by_relative_l1(self, threshold: float) -> DecideReuseFn:
        def _decide(state: StepReuseState, observation: torch.Tensor) -> bool:
            previous = state.real_history[-1]
            diff = observation - previous
            rel_l1 = (diff.abs().mean() / previous.abs().mean()).cpu().item()
            return rel_l1 < threshold

        return _decide

    def _get_step_reuse_controller(self) -> Optional[StepReuseController]:
        """
        Check step-reuse preconditions and return the (possibly freshly
        built) controller, or None if step-reuse is disabled for this
        forward pass.
        """
        from sglang.multimodal_gen.configs.sample.step_reuse import StepReuseParams
        from sglang.multimodal_gen.runtime.managers.forward_context import (
            get_forward_context,
        )

        forward_context = get_forward_context()
        forward_batch = forward_context.forward_batch

        if forward_batch is None or not getattr(
            forward_batch, "enable_step_reuse", False
        ):
            return None

        params = getattr(forward_batch, "step_reuse_params", None) or StepReuseParams()
        snapshot = (params.threshold, params.max_skip_steps, params.history_size)

        if (
            self._step_reuse_controller is None
            or self._step_reuse_params_snapshot != snapshot
        ):
            policy = StepReusePolicy(
                policy_name=f"step_reuse:{type(self).__name__}",
                observation_point="modulated_input",
                history_size=params.history_size,
                max_skip_steps=params.max_skip_steps,
                force_real_steps=frozenset({FORCE_FIRST_ONE, FORCE_TERMINAL}),
            )
            self._step_reuse_controller = StepReuseController(
                policy, self._decide_reuse_by_relative_l1(params.threshold)
            )
            self._step_reuse_params_snapshot = snapshot

        self._maybe_reset_step_reuse_for_new_task(forward_context.current_timestep)

        return self._step_reuse_controller

    def _step_reuse_scope(self) -> ScopeKey:
        return ("cfg_negative" if self.is_cfg_negative else "cfg_positive",)

    def should_skip_forward_for_step_reuse(self) -> bool:
        """Check whether the current step may reuse the last real prediction.

        Takes no observation: the reuse decision was already made when the
        skip window was opened (inside ``maybe_record_step_reuse`` on the
        last real forward); this only checks step-index forcing and
        remaining skip budget.
        """
        # _get_step_reuse_controller's reset-on-new-request check must see
        # the *previous* call's is_cfg_negative, so it runs before we
        # update it below for the current call (mirrors TeaCacheMixin's
        # _get_teacache_context / should_skip_forward_for_cached_states
        # split).
        controller = self._get_step_reuse_controller()
        if controller is None:
            return False

        from sglang.multimodal_gen.runtime.managers.forward_context import (
            get_forward_context,
        )

        forward_context = get_forward_context()
        forward_batch = forward_context.forward_batch
        self.is_cfg_negative = forward_batch.is_cfg_negative

        total_steps = forward_batch.num_inference_steps
        step_index = forward_context.current_timestep
        if forward_batch.do_classifier_free_guidance:
            total_steps *= 2
            step_index = step_index * 2 + (1 if self.is_cfg_negative else 0)

        scope = self._step_reuse_scope()
        if controller.should_reuse(scope, step_index, total_steps):
            controller.record_reuse(scope)
            return True

        return False

    def maybe_record_step_reuse(
        self,
        hidden_states: torch.Tensor,
        original_hidden_states: torch.Tensor,
        modulated_inp: torch.Tensor,
    ) -> None:
        """Record a real forward's prediction and residual for later reuse."""
        controller = self._step_reuse_controller
        if controller is None:
            return
        residual = hidden_states.squeeze(0) - original_hidden_states
        controller.record_real(
            self._step_reuse_scope(), prediction=residual, observation=modulated_inp
        )

    def retrieve_step_reuse_prediction(
        self, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """Apply the reused residual to the current hidden states."""
        controller = self._step_reuse_controller
        residual = controller.get_reused_prediction(self._step_reuse_scope())
        return hidden_states + residual
