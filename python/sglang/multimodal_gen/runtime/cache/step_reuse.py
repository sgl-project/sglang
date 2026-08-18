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
from typing import Any, Callable, Deque, Dict, FrozenSet, Hashable, Optional, Tuple

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

ScopeKey = Tuple[Hashable, ...]

FORCE_FIRST_TWO = "first_two"
FORCE_TERMINAL = "terminal"
_KNOWN_FORCE_POINTS = frozenset({FORCE_FIRST_TWO, FORCE_TERMINAL})


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
