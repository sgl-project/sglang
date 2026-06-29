import unittest

from sglang.srt.disaggregation.flip_state_machine import (
    ClusterSnapshot,
    FlipDecision,
    FlipDirection,
    FlipState,
    FlipStateMachine,
    FlipTransition,
    PDRatioSLOFlipEvaluator,
    SLOThresholdFlipEvaluator,
)


class TestPDFlipStateMachine(unittest.TestCase):
    def test_slo_evaluator_selects_d_to_p_when_prefill_slo_is_at_risk(self):
        evaluator = SLOThresholdFlipEvaluator(slo_threshold=0.9)
        snapshot = ClusterSnapshot(
            timestamp=1.0,
            role="decode",
            prefill_nodes=1,
            decode_nodes=3,
            prefill_slo_attainment=0.82,
            decode_slo_attainment=0.96,
        )

        decision = evaluator.evaluate(snapshot)

        self.assertTrue(decision.should_flip)
        self.assertEqual(decision.direction, FlipDirection.D_TO_P)
        self.assertEqual(decision.target_prefill_nodes, 2)
        self.assertEqual(decision.target_decode_nodes, 2)

    def test_slo_evaluator_selects_p_to_d_when_decode_slo_is_at_risk(self):
        evaluator = SLOThresholdFlipEvaluator(slo_threshold=0.9)
        snapshot = ClusterSnapshot(
            timestamp=1.0,
            role="prefill",
            prefill_nodes=3,
            decode_nodes=1,
            prefill_slo_attainment=0.95,
            decode_slo_attainment=0.71,
        )

        decision = evaluator.evaluate(snapshot)

        self.assertTrue(decision.should_flip)
        self.assertEqual(decision.direction, FlipDirection.P_TO_D)
        self.assertEqual(decision.target_prefill_nodes, 2)
        self.assertEqual(decision.target_decode_nodes, 2)

    def test_pd_ratio_evaluator_waits_for_enter_hysteresis(self):
        evaluator = PDRatioSLOFlipEvaluator(
            enter_threshold=0.9,
            min_enter_windows=2,
        )
        snapshot = ClusterSnapshot(
            timestamp=1.0,
            role="decode",
            prefill_nodes=1,
            decode_nodes=3,
            prefill_slo_attainment=0.72,
            decode_slo_attainment=0.96,
        )

        first = evaluator.evaluate(snapshot)
        second = evaluator.evaluate(snapshot)

        self.assertFalse(first.should_flip)
        self.assertIn("waiting for enter hysteresis", first.reason)
        self.assertTrue(second.should_flip)
        self.assertEqual(second.direction, FlipDirection.D_TO_P)
        self.assertEqual(second.metadata["policy"], "pd_ratio_slo")
        self.assertEqual(second.metadata["current_pd_ratio"], "1:3")
        self.assertEqual(second.metadata["target_pd_ratio"], "2:2")

    def test_pd_ratio_recovery_uses_exit_hysteresis(self):
        evaluator = PDRatioSLOFlipEvaluator(
            enter_threshold=0.9,
            exit_threshold=0.93,
            min_enter_windows=1,
            min_exit_windows=2,
        )
        machine = FlipStateMachine(
            evaluator=evaluator,
            prepare_flip=lambda snapshot, decision: bool(
                snapshot.metadata.get("kv_pretransfer_complete", False)
            ),
            min_window_seconds=0.0,
        )

        machine.tick(
            ClusterSnapshot(
                timestamp=1.0,
                role="decode",
                prefill_nodes=1,
                decode_nodes=3,
                prefill_slo_attainment=0.72,
                decode_slo_attainment=0.96,
            )
        )
        first_recovery_window = machine.tick(
            ClusterSnapshot(
                timestamp=2.0,
                role="decode",
                prefill_nodes=1,
                decode_nodes=3,
                prefill_slo_attainment=0.94,
                decode_slo_attainment=0.96,
                metadata={"kv_pretransfer_complete": False},
            )
        )
        second_recovery_window = machine.tick(
            ClusterSnapshot(
                timestamp=3.0,
                role="decode",
                prefill_nodes=1,
                decode_nodes=3,
                prefill_slo_attainment=0.95,
                decode_slo_attainment=0.96,
                metadata={"kv_pretransfer_complete": False},
            )
        )

        self.assertEqual(
            first_recovery_window.transition, FlipTransition.PREPARING_NOT_READY
        )
        self.assertEqual(second_recovery_window.transition, FlipTransition.RECOVER_SAFE)
        self.assertEqual(machine.state, FlipState.SAFE)

    def test_state_machine_progresses_through_prepare_and_flip(self):
        snapshots = [
            ClusterSnapshot(
                timestamp=1.0,
                role="decode",
                prefill_nodes=1,
                decode_nodes=3,
                prefill_slo_attainment=0.82,
                decode_slo_attainment=0.96,
            ),
            ClusterSnapshot(timestamp=2.0, role="decode"),
            ClusterSnapshot(timestamp=3.0, role="decode"),
        ]
        prepared = []
        committed = []

        machine = FlipStateMachine(
            evaluator=SLOThresholdFlipEvaluator(slo_threshold=0.9),
            prepare_flip=lambda snapshot, decision: prepared.append(
                (snapshot.timestamp, decision.direction)
            )
            or True,
            commit_flip=lambda snapshot, decision: committed.append(
                (snapshot.timestamp, decision.direction)
            )
            or True,
            min_window_seconds=0.0,
        )

        first = machine.tick(snapshots[0])
        second = machine.tick(snapshots[1])
        third = machine.tick(snapshots[2])

        self.assertEqual(first.transition, FlipTransition.START_PREPARING)
        self.assertEqual(first.to_state, FlipState.PREPARING)
        self.assertEqual(second.transition, FlipTransition.START_FLIPPING)
        self.assertEqual(second.to_state, FlipState.FLIPPING)
        self.assertEqual(third.transition, FlipTransition.FINISH_FLIPPING)
        self.assertEqual(third.to_state, FlipState.SAFE)
        self.assertEqual(machine.state, FlipState.SAFE)
        self.assertEqual(machine.direction, FlipDirection.NONE)
        self.assertEqual(prepared, [(2.0, FlipDirection.D_TO_P)])
        self.assertEqual(committed, [(3.0, FlipDirection.D_TO_P)])
        self.assertEqual(machine.status()["last_completed_target_role"], "prefill")
        self.assertEqual(machine.status()["last_completed_direction"], "d_to_p")

    def test_preparing_recovers_to_safe_when_slo_returns_inside_threshold(self):
        machine = FlipStateMachine(
            evaluator=SLOThresholdFlipEvaluator(slo_threshold=0.9),
            prepare_flip=lambda snapshot, decision: bool(
                snapshot.metadata.get("kv_pretransfer_complete", False)
            ),
            min_window_seconds=0.0,
        )

        first = machine.tick(
            ClusterSnapshot(
                timestamp=1.0,
                role="decode",
                prefill_nodes=1,
                decode_nodes=3,
                prefill_slo_attainment=0.72,
                decode_slo_attainment=0.96,
            )
        )
        recovered = machine.tick(
            ClusterSnapshot(
                timestamp=2.0,
                role="decode",
                prefill_nodes=1,
                decode_nodes=3,
                prefill_slo_attainment=0.94,
                decode_slo_attainment=0.96,
                metadata={"kv_pretransfer_complete": False},
            )
        )

        self.assertEqual(first.transition, FlipTransition.START_PREPARING)
        self.assertEqual(recovered.transition, FlipTransition.RECOVER_SAFE)
        self.assertEqual(recovered.from_state, FlipState.PREPARING)
        self.assertEqual(recovered.to_state, FlipState.SAFE)
        self.assertEqual(machine.state, FlipState.SAFE)
        self.assertEqual(machine.direction, FlipDirection.NONE)
        self.assertIsNone(machine.pending_decision)
        self.assertEqual(machine.status()["last_recovery_reason"], "both SLO attainments are inside safe region")

    def test_preparing_waits_for_kv_pretransfer_before_flipping(self):
        machine = FlipStateMachine(
            evaluator=SLOThresholdFlipEvaluator(slo_threshold=0.9),
            prepare_flip=lambda snapshot, decision: bool(
                snapshot.metadata.get("kv_pretransfer_complete", False)
            ),
            min_window_seconds=0.0,
        )

        machine.tick(
            ClusterSnapshot(
                timestamp=1.0,
                role="decode",
                prefill_nodes=1,
                decode_nodes=3,
                prefill_slo_attainment=0.72,
                decode_slo_attainment=0.96,
            )
        )
        waiting = machine.tick(
            ClusterSnapshot(
                timestamp=2.0,
                role="decode",
                prefill_nodes=1,
                decode_nodes=3,
                prefill_slo_attainment=0.73,
                decode_slo_attainment=0.96,
                metadata={"kv_pretransfer_complete": False},
            )
        )
        ready = machine.tick(
            ClusterSnapshot(
                timestamp=3.0,
                role="decode",
                prefill_nodes=1,
                decode_nodes=3,
                prefill_slo_attainment=0.74,
                decode_slo_attainment=0.96,
                metadata={"kv_pretransfer_complete": True},
            )
        )

        self.assertEqual(waiting.transition, FlipTransition.PREPARING_NOT_READY)
        self.assertEqual(waiting.to_state, FlipState.PREPARING)
        self.assertEqual(ready.transition, FlipTransition.START_FLIPPING)
        self.assertEqual(ready.to_state, FlipState.FLIPPING)

    def test_preparing_can_be_held_by_commit_decision(self):
        machine = FlipStateMachine(
            evaluator=SLOThresholdFlipEvaluator(slo_threshold=0.9),
            min_window_seconds=0.0,
        )

        machine.tick(
            ClusterSnapshot(
                timestamp=1.0,
                role="decode",
                prefill_nodes=1,
                decode_nodes=3,
                prefill_slo_attainment=0.72,
                decode_slo_attainment=0.96,
            )
        )
        event = machine.tick(
            ClusterSnapshot(
                timestamp=2.0,
                role="decode",
                prefill_nodes=1,
                decode_nodes=3,
                prefill_slo_attainment=0.73,
                decode_slo_attainment=0.96,
                metadata={
                    "kv_pretransfer_complete": True,
                    "commit_decision": False,
                },
            )
        )

        self.assertEqual(event.transition, FlipTransition.PREPARING_NOT_READY)
        self.assertEqual(event.reason, "commit_decision_not_ready")
        self.assertEqual(machine.state, FlipState.PREPARING)

    def test_state_machine_stays_safe_without_flip_decision(self):
        machine = FlipStateMachine(
            evaluator=lambda snapshot: FlipDecision.stay_safe("inside safe region"),
            min_window_seconds=0.0,
        )

        event = machine.tick(ClusterSnapshot(timestamp=1.0, role="prefill"))

        self.assertEqual(event.transition, FlipTransition.NONE)
        self.assertEqual(event.from_state, FlipState.SAFE)
        self.assertEqual(event.to_state, FlipState.SAFE)

    def test_state_machine_can_abort_pending_flip(self):
        machine = FlipStateMachine(
            evaluator=SLOThresholdFlipEvaluator(slo_threshold=0.9),
            min_window_seconds=0.0,
        )

        machine.tick(
            ClusterSnapshot(
                timestamp=1.0,
                role="decode",
                prefill_nodes=1,
                decode_nodes=2,
                prefill_slo_attainment=0.7,
                decode_slo_attainment=0.99,
            )
        )
        event = machine.abort("external orchestrator rejected migration")

        self.assertEqual(event.transition, FlipTransition.ABORT)
        self.assertEqual(event.from_state, FlipState.PREPARING)
        self.assertEqual(event.to_state, FlipState.SAFE)
        self.assertEqual(machine.state, FlipState.SAFE)
        self.assertEqual(machine.direction, FlipDirection.NONE)
        self.assertIsNone(machine.pending_decision)
        self.assertEqual(machine.status()["router_action"], "serve_current_role")
        self.assertEqual(machine.status()["last_abort_reason"], "external orchestrator rejected migration")

    def test_status_exposes_role_request_and_migration_requirements(self):
        machine = FlipStateMachine(
            evaluator=SLOThresholdFlipEvaluator(slo_threshold=0.9),
            min_window_seconds=0.0,
        )

        machine.tick(
            ClusterSnapshot(
                timestamp=1.0,
                role="decode",
                prefill_nodes=1,
                decode_nodes=3,
                waiting_reqs=2,
                running_reqs=4,
                kv_used_tokens=128,
                kv_total_tokens=512,
                prefill_slo_attainment=0.82,
                decode_slo_attainment=0.96,
            )
        )

        status = machine.status()

        self.assertEqual(status["state"], "preparing")
        self.assertEqual(status["source_role"], "decode")
        self.assertEqual(status["requested_role"], "prefill")
        self.assertEqual(status["target_role"], "prefill")
        self.assertEqual(status["target_prefill_nodes"], 2)
        self.assertEqual(status["target_decode_nodes"], 2)
        self.assertTrue(status["requires_active_request_migration"])
        self.assertTrue(status["requires_kv_migration"])
        self.assertTrue(status["requires_drain_to_idle"])
        self.assertEqual(status["active_request_migration_strategy"], "drain_to_idle")
        self.assertTrue(status["can_hot_switch_in_process"])
        self.assertFalse(status["requires_process_restart"])
        self.assertTrue(status["requires_external_orchestrator"])
        self.assertEqual(status["router_action"], "redirect_decode_and_drain_to_idle")
        self.assertEqual(status["snapshot"]["running_reqs"], 4)
        self.assertEqual(status["snapshot"]["kv_used_tokens"], 128)


if __name__ == "__main__":
    unittest.main()
