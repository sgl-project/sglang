from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Optional, Protocol, Union


class FlipState(Enum):
    SAFE = "safe"
    PREPARING = "preparing"
    FLIPPING = "flipping"


class FlipDirection(Enum):
    NONE = "none"
    D_TO_P = "d_to_p"
    P_TO_D = "p_to_d"


class FlipTransition(Enum):
    NONE = "none"
    START_PREPARING = "start_preparing"
    START_FLIPPING = "start_flipping"
    FINISH_FLIPPING = "finish_flipping"
    RECOVER_SAFE = "recover_safe"
    PREPARING_NOT_READY = "preparing_not_ready"
    FLIPPING_NOT_READY = "flipping_not_ready"
    ABORT = "abort"


@dataclass(frozen=True)
class ClusterSnapshot:
    timestamp: float
    role: str
    prefill_nodes: int = 0
    decode_nodes: int = 0
    waiting_reqs: int = 0
    running_reqs: int = 0
    prefill_bootstrap_reqs: int = 0
    prefill_inflight_reqs: int = 0
    decode_prealloc_reqs: int = 0
    decode_transfer_reqs: int = 0
    kv_used_tokens: Optional[int] = None
    kv_total_tokens: Optional[int] = None
    prefill_slo_attainment: Optional[float] = None
    decode_slo_attainment: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FlipDecision:
    should_flip: bool
    direction: FlipDirection = FlipDirection.NONE
    reason: str = ""
    target_prefill_nodes: Optional[int] = None
    target_decode_nodes: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def stay_safe(cls, reason: str = "") -> "FlipDecision":
        return cls(should_flip=False, direction=FlipDirection.NONE, reason=reason)


@dataclass(frozen=True)
class FlipEvent:
    transition: FlipTransition
    from_state: FlipState
    to_state: FlipState
    direction: FlipDirection
    reason: str = ""
    decision: Optional[FlipDecision] = None


class FlipEvaluator(Protocol):
    def evaluate(self, snapshot: ClusterSnapshot) -> FlipDecision:
        ...


FlipEvaluatorLike = Union[
    FlipEvaluator, Callable[[ClusterSnapshot], FlipDecision]
]
FlipCallback = Callable[[ClusterSnapshot, FlipDecision], bool]


class SLOThresholdFlipEvaluator:
    """A small what-if evaluator scaffold based on SLO attainment.

    A real Janus evaluator can replace this class with one that simulates nearby
    P/D ratios from a cluster-wide snapshot. This implementation keeps the first
    code path deterministic and testable: if prefill SLO is below threshold, a
    decode node is eligible to flip D->P; if decode SLO is below threshold, a
    prefill node is eligible to flip P->D.
    """

    def __init__(self, slo_threshold: float = 0.9):
        self.slo_threshold = slo_threshold

    def evaluate(self, snapshot: ClusterSnapshot) -> FlipDecision:
        prefill_deficit = self._deficit(snapshot.prefill_slo_attainment)
        decode_deficit = self._deficit(snapshot.decode_slo_attainment)

        if prefill_deficit <= 0 and decode_deficit <= 0:
            return FlipDecision.stay_safe("both SLO attainments are inside safe region")

        candidates = []
        if prefill_deficit > 0 and snapshot.decode_nodes > 0:
            candidates.append((prefill_deficit, FlipDirection.D_TO_P))
        if decode_deficit > 0 and snapshot.prefill_nodes > 0:
            candidates.append((decode_deficit, FlipDirection.P_TO_D))

        if not candidates:
            return FlipDecision.stay_safe("no eligible node exists for the risky SLO")

        _, direction = max(candidates, key=lambda item: item[0])
        if direction == FlipDirection.D_TO_P:
            if snapshot.role != "decode":
                return FlipDecision.stay_safe("local node is not decode; skip D->P")
            return FlipDecision(
                should_flip=True,
                direction=direction,
                reason="prefill SLO below threshold",
                target_prefill_nodes=snapshot.prefill_nodes + 1,
                target_decode_nodes=max(0, snapshot.decode_nodes - 1),
            )

        if snapshot.role != "prefill":
            return FlipDecision.stay_safe("local node is not prefill; skip P->D")
        return FlipDecision(
            should_flip=True,
            direction=direction,
            reason="decode SLO below threshold",
            target_prefill_nodes=max(0, snapshot.prefill_nodes - 1),
            target_decode_nodes=snapshot.decode_nodes + 1,
        )

    def _deficit(self, attainment: Optional[float]) -> float:
        if attainment is None:
            return 0.0
        return max(0.0, self.slo_threshold - attainment)


class PDRatioSLOFlipEvaluator:
    """P/D ratio policy with simple enter/exit hysteresis.

    The policy still consumes the same ``ClusterSnapshot`` shape as the
    threshold evaluator. It adds consecutive-window gating and explicit metadata
    so the monitor/controller can explain why a nearby P/D ratio was selected.
    """

    def __init__(
        self,
        enter_threshold: float = 0.9,
        exit_threshold: Optional[float] = None,
        commit_threshold: Optional[float] = None,
        min_enter_windows: int = 1,
        min_exit_windows: int = 1,
    ):
        self.enter_threshold = enter_threshold
        self.exit_threshold = exit_threshold if exit_threshold is not None else enter_threshold
        self.commit_threshold = (
            commit_threshold if commit_threshold is not None else enter_threshold
        )
        self.min_enter_windows = max(1, int(min_enter_windows))
        self.min_exit_windows = max(1, int(min_exit_windows))
        self._last_enter_direction = FlipDirection.NONE
        self._enter_windows = 0
        self._exit_windows = 0

    def evaluate(self, snapshot: ClusterSnapshot) -> FlipDecision:
        decision = self._evaluate_once(snapshot, self.enter_threshold)
        if not decision.should_flip:
            self._last_enter_direction = FlipDirection.NONE
            self._enter_windows = 0
            return decision

        if decision.direction == self._last_enter_direction:
            self._enter_windows += 1
        else:
            self._last_enter_direction = decision.direction
            self._enter_windows = 1

        if self._enter_windows < self.min_enter_windows:
            return FlipDecision.stay_safe(
                f"waiting for enter hysteresis {self._enter_windows}/{self.min_enter_windows}"
            )
        return decision

    def evaluate_recovery(
        self, snapshot: ClusterSnapshot, direction: FlipDirection
    ) -> Optional[FlipDecision]:
        if direction == FlipDirection.D_TO_P:
            if snapshot.prefill_slo_attainment is None:
                return None
            recovered = self._attainment_at_or_above(
                snapshot.prefill_slo_attainment, self.exit_threshold
            )
            reason = "prefill SLO recovered above exit threshold"
        elif direction == FlipDirection.P_TO_D:
            if snapshot.decode_slo_attainment is None:
                return None
            recovered = self._attainment_at_or_above(
                snapshot.decode_slo_attainment, self.exit_threshold
            )
            reason = "decode SLO recovered above exit threshold"
        else:
            return None

        if recovered:
            self._exit_windows += 1
            if self._exit_windows >= self.min_exit_windows:
                return FlipDecision.stay_safe(reason)
            return self._keep_direction_decision(
                snapshot,
                direction,
                f"waiting for exit hysteresis {self._exit_windows}/{self.min_exit_windows}",
            )
        else:
            self._exit_windows = 0
        return self._keep_direction_decision(
            snapshot, direction, "SLO still inside preparing hysteresis band"
        )

    def _evaluate_once(
        self, snapshot: ClusterSnapshot, threshold: float
    ) -> FlipDecision:
        prefill_deficit = self._deficit(snapshot.prefill_slo_attainment, threshold)
        decode_deficit = self._deficit(snapshot.decode_slo_attainment, threshold)

        if prefill_deficit <= 0 and decode_deficit <= 0:
            return FlipDecision.stay_safe("both SLO attainments are inside safe region")

        candidates = []
        if prefill_deficit > 0 and snapshot.decode_nodes > 0:
            candidates.append((prefill_deficit, FlipDirection.D_TO_P, "prefill"))
        if decode_deficit > 0 and snapshot.prefill_nodes > 0:
            candidates.append((decode_deficit, FlipDirection.P_TO_D, "decode"))
        if not candidates:
            return FlipDecision.stay_safe("no eligible node exists for the risky SLO")

        deficit, direction, risky_role = max(candidates, key=lambda item: item[0])
        if direction == FlipDirection.D_TO_P:
            if snapshot.role != "decode":
                return FlipDecision.stay_safe("local node is not decode; skip D->P")
            target_prefill_nodes = snapshot.prefill_nodes + 1
            target_decode_nodes = max(0, snapshot.decode_nodes - 1)
        else:
            if snapshot.role != "prefill":
                return FlipDecision.stay_safe("local node is not prefill; skip P->D")
            target_prefill_nodes = max(0, snapshot.prefill_nodes - 1)
            target_decode_nodes = snapshot.decode_nodes + 1

        return FlipDecision(
            should_flip=True,
            direction=direction,
            reason=f"{risky_role} SLO below P/D policy threshold",
            target_prefill_nodes=target_prefill_nodes,
            target_decode_nodes=target_decode_nodes,
            metadata={
                "policy": "pd_ratio_slo",
                "risky_role": risky_role,
                "deficit": deficit,
                "enter_threshold": self.enter_threshold,
                "exit_threshold": self.exit_threshold,
                "commit_threshold": self.commit_threshold,
                "current_pd_ratio": self._ratio(snapshot.prefill_nodes, snapshot.decode_nodes),
                "target_pd_ratio": self._ratio(target_prefill_nodes, target_decode_nodes),
            },
        )

    def _keep_direction_decision(
        self, snapshot: ClusterSnapshot, direction: FlipDirection, reason: str
    ) -> FlipDecision:
        if direction == FlipDirection.D_TO_P:
            target_prefill_nodes = snapshot.prefill_nodes + 1
            target_decode_nodes = max(0, snapshot.decode_nodes - 1)
        elif direction == FlipDirection.P_TO_D:
            target_prefill_nodes = max(0, snapshot.prefill_nodes - 1)
            target_decode_nodes = snapshot.decode_nodes + 1
        else:
            return FlipDecision.stay_safe(reason)

        return FlipDecision(
            should_flip=True,
            direction=direction,
            reason=reason,
            target_prefill_nodes=target_prefill_nodes,
            target_decode_nodes=target_decode_nodes,
            metadata={
                "policy": "pd_ratio_slo",
                "enter_threshold": self.enter_threshold,
                "exit_threshold": self.exit_threshold,
                "commit_threshold": self.commit_threshold,
                "current_pd_ratio": self._ratio(snapshot.prefill_nodes, snapshot.decode_nodes),
                "target_pd_ratio": self._ratio(target_prefill_nodes, target_decode_nodes),
            },
        )

    @staticmethod
    def _deficit(attainment: Optional[float], threshold: float) -> float:
        if attainment is None:
            return 0.0
        return max(0.0, threshold - attainment)

    @staticmethod
    def _attainment_at_or_above(attainment: Optional[float], threshold: float) -> bool:
        return attainment is not None and attainment >= threshold

    @staticmethod
    def _ratio(prefill_nodes: int, decode_nodes: int) -> str:
        return f"{prefill_nodes}:{decode_nodes}"


class FlipStateMachine:
    def __init__(
        self,
        evaluator: FlipEvaluatorLike,
        prepare_flip: Optional[FlipCallback] = None,
        commit_flip: Optional[FlipCallback] = None,
        min_window_seconds: float = 1.0,
        time_fn: Callable[[], float] = time.monotonic,
    ):
        self.evaluator = evaluator
        self.prepare_flip = prepare_flip or self._ready
        self.commit_flip = commit_flip or self._ready
        self.min_window_seconds = min_window_seconds
        self.time_fn = time_fn

        self.state = FlipState.SAFE
        self.direction = FlipDirection.NONE
        self.pending_decision: Optional[FlipDecision] = None
        self.last_completed_decision: Optional[FlipDecision] = None
        self.last_completed_direction = FlipDirection.NONE
        self.last_abort_reason = ""
        self.last_recovery_reason = ""
        self.last_prepare_not_ready_reason = ""
        self.prepare_started_at: Optional[float] = None
        self.last_eval_time: Optional[float] = None
        self.last_snapshot: Optional[ClusterSnapshot] = None
        self.last_event = FlipEvent(
            transition=FlipTransition.NONE,
            from_state=FlipState.SAFE,
            to_state=FlipState.SAFE,
            direction=FlipDirection.NONE,
        )

    def tick(self, snapshot: ClusterSnapshot) -> FlipEvent:
        self.last_snapshot = snapshot
        if self.state == FlipState.SAFE:
            event = self._tick_safe(snapshot)
        elif self.state == FlipState.PREPARING:
            event = self._tick_preparing(snapshot)
        else:
            event = self._tick_flipping(snapshot)

        self.last_event = event
        return event

    def status(self) -> Dict[str, Any]:
        decision = self.pending_decision
        source_role = self._source_role(self.direction)
        target_role = self._target_role(self.direction)
        needs_migration = self.direction == FlipDirection.D_TO_P
        return {
            "state": self.state.value,
            "direction": self.direction.value,
            "source_role": source_role,
            "requested_role": target_role,
            "target_role": target_role,
            "target_prefill_nodes": decision.target_prefill_nodes
            if decision is not None
            else None,
            "target_decode_nodes": decision.target_decode_nodes
            if decision is not None
            else None,
            "pending_reason": decision.reason if decision else "",
            "decision_metadata": dict(decision.metadata) if decision else {},
            "last_transition": self.last_event.transition.value,
            "requires_active_request_migration": needs_migration,
            "requires_kv_migration": needs_migration,
            "requires_drain_to_idle": self.direction != FlipDirection.NONE,
            "active_request_migration_strategy": "drain_to_idle"
            if needs_migration
            else "none",
            "requires_external_orchestrator": self.direction != FlipDirection.NONE,
            "can_hot_switch_in_process": True,
            "requires_process_restart": False,
            "router_action": self._router_action(),
            "last_completed_direction": self.last_completed_direction.value,
            "last_completed_target_role": self._target_role(
                self.last_completed_direction
            ),
            "last_completed_reason": self.last_completed_decision.reason
            if self.last_completed_decision
            else "",
            "last_abort_reason": self.last_abort_reason,
            "last_recovery_reason": self.last_recovery_reason,
            "last_prepare_not_ready_reason": self.last_prepare_not_ready_reason,
            "prepare_started_at": self.prepare_started_at,
            "prepare_elapsed_seconds": self._prepare_elapsed_seconds(),
            "commit_ready": self.last_event.transition == FlipTransition.START_FLIPPING,
            "abort_ready": self.last_event.transition
            in (FlipTransition.ABORT, FlipTransition.RECOVER_SAFE),
            "migration_commit_mode": self._migration_commit_mode(decision),
            "snapshot": asdict(self.last_snapshot) if self.last_snapshot else None,
        }

    def abort(self, reason: str = "") -> FlipEvent:
        if self.state == FlipState.SAFE:
            event = self._event(FlipTransition.NONE, FlipState.SAFE, reason)
            self.last_event = event
            return event

        old_state = self.state
        old_direction = self.direction
        decision = self.pending_decision
        self.state = FlipState.SAFE
        self.direction = FlipDirection.NONE
        self.pending_decision = None
        self.prepare_started_at = None
        self.last_prepare_not_ready_reason = ""
        self.last_abort_reason = reason
        event = FlipEvent(
            transition=FlipTransition.ABORT,
            from_state=old_state,
            to_state=self.state,
            direction=old_direction,
            reason=reason,
            decision=decision,
        )
        self.last_event = event
        return event

    def _tick_safe(self, snapshot: ClusterSnapshot) -> FlipEvent:
        now = snapshot.timestamp if snapshot.timestamp is not None else self.time_fn()
        if (
            self.last_eval_time is not None
            and now - self.last_eval_time < self.min_window_seconds
        ):
            return self._event(FlipTransition.NONE, FlipState.SAFE, "")

        self.last_eval_time = now
        decision = self._evaluate(snapshot)
        if not decision.should_flip or decision.direction == FlipDirection.NONE:
            return self._event(
                FlipTransition.NONE,
                FlipState.SAFE,
                decision.reason,
                decision,
            )

        old_state = self.state
        self.state = FlipState.PREPARING
        self.direction = decision.direction
        self.pending_decision = decision
        self.prepare_started_at = now
        self.last_recovery_reason = ""
        self.last_prepare_not_ready_reason = ""
        return FlipEvent(
            transition=FlipTransition.START_PREPARING,
            from_state=old_state,
            to_state=self.state,
            direction=self.direction,
            reason=decision.reason,
            decision=decision,
        )

    def _tick_preparing(self, snapshot: ClusterSnapshot) -> FlipEvent:
        decision = self._require_decision()
        recovery_decision = self._evaluate_recovery(snapshot)
        if recovery_decision is not None and not self._same_direction(
            recovery_decision
        ):
            return self._recover_to_safe(
                recovery_decision.reason or "SLO recovered",
                recovery_decision,
            )

        if snapshot.metadata.get("commit_decision") is False:
            self.last_prepare_not_ready_reason = "commit_decision_not_ready"
            return self._event(
                FlipTransition.PREPARING_NOT_READY,
                FlipState.PREPARING,
                self.last_prepare_not_ready_reason,
                decision,
            )

        if snapshot.metadata.get("kv_pretransfer_complete") is False:
            self.last_prepare_not_ready_reason = "kv_pretransfer_not_ready"
            return self._event(
                FlipTransition.PREPARING_NOT_READY,
                FlipState.PREPARING,
                self.last_prepare_not_ready_reason,
                decision,
            )

        if not self.prepare_flip(snapshot, decision):
            self.last_prepare_not_ready_reason = (
                snapshot.metadata.get("prepare_not_ready_reason")
                or "prepare_callback_not_ready"
            )
            return self._event(
                FlipTransition.PREPARING_NOT_READY,
                FlipState.PREPARING,
                self.last_prepare_not_ready_reason,
                decision,
            )

        old_state = self.state
        self.state = FlipState.FLIPPING
        self.last_prepare_not_ready_reason = ""
        return FlipEvent(
            transition=FlipTransition.START_FLIPPING,
            from_state=old_state,
            to_state=self.state,
            direction=self.direction,
            reason=decision.reason,
            decision=decision,
        )

    def _tick_flipping(self, snapshot: ClusterSnapshot) -> FlipEvent:
        decision = self._require_decision()
        if not self.commit_flip(snapshot, decision):
            return self._event(
                FlipTransition.FLIPPING_NOT_READY,
                FlipState.FLIPPING,
                decision.reason,
                decision,
            )

        old_state = self.state
        direction = self.direction
        self.state = FlipState.SAFE
        self.direction = FlipDirection.NONE
        self.pending_decision = None
        self.prepare_started_at = None
        self.last_prepare_not_ready_reason = ""
        self.last_completed_decision = decision
        self.last_completed_direction = direction
        self.last_abort_reason = ""
        return FlipEvent(
            transition=FlipTransition.FINISH_FLIPPING,
            from_state=old_state,
            to_state=self.state,
            direction=direction,
            reason=decision.reason,
            decision=decision,
        )

    def _evaluate(self, snapshot: ClusterSnapshot) -> FlipDecision:
        evaluate = getattr(self.evaluator, "evaluate", None)
        if evaluate is not None:
            return evaluate(snapshot)
        return self.evaluator(snapshot)

    def _evaluate_recovery(self, snapshot: ClusterSnapshot) -> Optional[FlipDecision]:
        if snapshot.metadata.get("slo_recovered") is True:
            return FlipDecision.stay_safe(
                str(snapshot.metadata.get("recovery_reason") or "SLO recovered")
            )
        if not self._has_slo_attainment(snapshot):
            return None
        evaluate_recovery = getattr(self.evaluator, "evaluate_recovery", None)
        if evaluate_recovery is not None:
            return evaluate_recovery(snapshot, self.direction)
        return self._evaluate(snapshot)

    @staticmethod
    def _has_slo_attainment(snapshot: ClusterSnapshot) -> bool:
        return (
            snapshot.prefill_slo_attainment is not None
            or snapshot.decode_slo_attainment is not None
        )

    def _same_direction(self, decision: FlipDecision) -> bool:
        return decision.should_flip and decision.direction == self.direction

    def _recover_to_safe(
        self, reason: str, decision: Optional[FlipDecision] = None
    ) -> FlipEvent:
        old_state = self.state
        old_direction = self.direction
        old_decision = self.pending_decision
        self.state = FlipState.SAFE
        self.direction = FlipDirection.NONE
        self.pending_decision = None
        self.prepare_started_at = None
        self.last_recovery_reason = reason
        self.last_prepare_not_ready_reason = ""
        return FlipEvent(
            transition=FlipTransition.RECOVER_SAFE,
            from_state=old_state,
            to_state=self.state,
            direction=old_direction,
            reason=reason,
            decision=decision or old_decision,
        )

    def _require_decision(self) -> FlipDecision:
        if self.pending_decision is None:
            raise RuntimeError("flip state machine entered transition without a decision")
        return self.pending_decision

    def _event(
        self,
        transition: FlipTransition,
        to_state: FlipState,
        reason: str,
        decision: Optional[FlipDecision] = None,
    ) -> FlipEvent:
        return FlipEvent(
            transition=transition,
            from_state=self.state,
            to_state=to_state,
            direction=self.direction,
            reason=reason,
            decision=decision,
        )

    @staticmethod
    def _ready(snapshot: ClusterSnapshot, decision: FlipDecision) -> bool:
        return True

    def _prepare_elapsed_seconds(self) -> Optional[float]:
        if self.prepare_started_at is None or self.last_snapshot is None:
            return None
        return max(0.0, self.last_snapshot.timestamp - self.prepare_started_at)

    @staticmethod
    def _migration_commit_mode(decision: Optional[FlipDecision]) -> str:
        if decision is None:
            return "none"
        value = decision.metadata.get("migration_commit_mode")
        return str(value) if value else "two_phase"

    @staticmethod
    def _source_role(direction: FlipDirection) -> Optional[str]:
        if direction == FlipDirection.D_TO_P:
            return "decode"
        if direction == FlipDirection.P_TO_D:
            return "prefill"
        return None

    @staticmethod
    def _target_role(direction: FlipDirection) -> Optional[str]:
        if direction == FlipDirection.D_TO_P:
            return "prefill"
        if direction == FlipDirection.P_TO_D:
            return "decode"
        return None

    def _router_action(self) -> str:
        if self.direction == FlipDirection.NONE:
            return "serve_current_role"
        if self.direction == FlipDirection.D_TO_P:
            if self.state == FlipState.PREPARING:
                return "redirect_decode_and_drain_to_idle"
            if self.state == FlipState.FLIPPING:
                return "commit_route_as_prefill"
        if self.direction == FlipDirection.P_TO_D:
            if self.state == FlipState.PREPARING:
                return "stop_prefill_admission_and_drain_inflight"
            if self.state == FlipState.FLIPPING:
                return "commit_route_as_decode"
        return "serve_current_role"
