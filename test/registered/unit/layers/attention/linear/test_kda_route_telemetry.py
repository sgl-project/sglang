"""Pure CPU tests for Phase-A KDA terminal route telemetry."""

import unittest
from dataclasses import FrozenInstanceError, replace
from unittest.mock import Mock

from sglang.srt.layers.attention.linear.kda_route_telemetry import (
    CAKE_PACKED_EXCEPTION,
    CAKE_PREFILL_EXCEPTION,
    CUDA_GRAPH_CAPTURE_EXCEPTION,
    CUDA_GRAPH_PLAN_BIND_EXCEPTION,
    CUDA_GRAPH_REPLAY_EXCEPTION,
    CUDA_GRAPH_REPLAY_PLAN_MISMATCH,
    KDA_ROUTE_EVENT_PREFIX,
    KDA_ROUTE_SCHEMA_VERSION,
    PACKED_SELECTOR_EXCEPTION,
    PREFILL_SELECTOR_EXCEPTION,
    TRITON_FALLBACK_EXCEPTION,
    CakePackedDecodeReason,
    CakePrefillReason,
    KDACudaGraphPlanError,
    KDACudaGraphRoutePlans,
    KDATerminalRouteEvent,
    KDATerminalRouteTelemetry,
    capture_kda_route_plan,
    record_kda_terminal_route,
    replay_kda_route_plan,
    suppress_kda_route_recording,
)
from sglang.srt.model_executor.runner.shape_key import ShapeKey
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _recorder():
    return KDATerminalRouteTelemetry(raw_event_capacity=128, emit_log=False)


def _success(
    telemetry,
    *,
    mode="decode",
    layer_id=3,
    copy_count=0,
    copy_count_source="static_zero_copy_row_view",
):
    return record_kda_terminal_route(
        mode=mode,
        layer_id=layer_id,
        eligible=True,
        attempted_cake=True,
        cake_success=True,
        triton_fallback=False,
        fatal=False,
        reason="eligible",
        copy_count=copy_count,
        copy_count_source=copy_count_source,
        telemetry=telemetry,
    )


def _event(
    *,
    mode="decode",
    outcome="cake_success",
    reason="eligible",
    eligible=None,
    graph_phase="direct",
    copy_count=None,
    copy_count_source=None,
):
    if eligible is None:
        eligible = outcome == "cake_success"
    packed_zero_copy = mode == "decode" and outcome == "cake_success"
    if copy_count_source is None:
        copy_count = 0 if packed_zero_copy else None
        copy_count_source = (
            "static_zero_copy_row_view"
            if packed_zero_copy
            else "unknown_requires_cupti"
        )
    return KDATerminalRouteEvent(
        schema_version=KDA_ROUTE_SCHEMA_VERSION,
        mode=mode,
        layer_id=3,
        considered=1,
        eligible=int(eligible),
        attempted_cake=int(eligible),
        cake_success=int(outcome == "cake_success"),
        triton_fallback=int(outcome == "triton_fallback"),
        fatal=int(outcome == "fatal"),
        reason=reason,
        detail="",
        graph_phase=graph_phase,
        copy_count=copy_count,
        copy_count_source=copy_count_source,
    )


class TestKDATerminalRouteTelemetry(unittest.TestCase):
    def test_direct_success_fallback_fatal_and_counter_closure(self):
        telemetry = _recorder()
        _success(telemetry, layer_id=3)
        record_kda_terminal_route(
            mode="decode",
            layer_id=4,
            eligible=False,
            attempted_cake=False,
            cake_success=False,
            triton_fallback=True,
            fatal=False,
            reason="inner_stride",
            detail="raw_beta",
            telemetry=telemetry,
        )
        record_kda_terminal_route(
            mode="prefill",
            layer_id=5,
            eligible=True,
            attempted_cake=True,
            cake_success=False,
            triton_fallback=False,
            fatal=True,
            reason="cake_prefill_exception",
            detail="builtins.RuntimeError",
            telemetry=telemetry,
        )

        events = telemetry.raw_events_snapshot()
        self.assertEqual(len(events), 3)
        self.assertEqual([event.graph_phase for event in events], ["direct"] * 3)
        for event in events:
            self.assertEqual(
                event.considered,
                event.cake_success + event.triton_fallback + event.fatal,
            )

        snapshot = telemetry.counter_snapshot()
        self.assertEqual(snapshot["events_total"], 3)
        self.assertEqual(
            snapshot["key_fields"],
            ["mode", "layer_id", "graph_phase", "reason"],
        )
        self.assertEqual(len(snapshot["counters"]), 3)
        for row in snapshot["counters"]:
            self.assertEqual(
                row["considered"],
                row["cake_success"] + row["triton_fallback"] + row["fatal"],
            )

    def test_terminal_event_is_immutable_and_rejects_nonterminal_outcome(self):
        telemetry = _recorder()
        event = _success(telemetry)
        with self.assertRaises(FrozenInstanceError):
            event.reason = "changed"
        with self.assertRaisesRegex(ValueError, "terminal invariant"):
            record_kda_terminal_route(
                mode="decode",
                layer_id=3,
                eligible=True,
                attempted_cake=True,
                cake_success=False,
                triton_fallback=False,
                fatal=False,
                reason="incomplete",
                telemetry=telemetry,
            )

    def test_schema_accepts_only_the_v1_reason_matrix(self):
        decode_fallback_reasons = (
            CakePackedDecodeReason.KERNEL_UNAVAILABLE,
            CakePackedDecodeReason.REPLAYSSM_REQUESTED,
            CakePackedDecodeReason.UNSUPPORTED_CONTRACT,
            CakePackedDecodeReason.INNER_STRIDE,
            CakePackedDecodeReason.ZERO_ROW_STRIDE,
            CakePackedDecodeReason.NEGATIVE_ROW_STRIDE,
            CakePackedDecodeReason.OVERLAPPING_ROW_STRIDE,
            CakePackedDecodeReason.STORAGE_ALIAS,
            CakePackedDecodeReason.CACHE_INDEX_OOB,
            CakePackedDecodeReason.CACHE_INDEX_DUPLICATE,
        )
        prefill_fallback_reasons = (
            CakePrefillReason.SPEC_DECODE,
            CakePrefillReason.INTERIOR_CHECKPOINT,
            CakePrefillReason.INVALID_LOWER_BOUND,
            CakePrefillReason.MISSING_GATE_PARAMS,
            CakePrefillReason.UNSUPPORTED_Q_CONTRACT,
            CakePrefillReason.CUDA_GRAPH_ALLOCATION,
            CakePrefillReason.T1_DECODE_SHAPE,
            CakePrefillReason.UNSUPPORTED_HEAD_DIM,
            CakePrefillReason.SHAPE_MISMATCH,
            CakePrefillReason.UNSUPPORTED_ARCH,
            CakePrefillReason.UNSUPPORTED_CONTRACT,
        )
        cases = [
            dict(mode="decode", outcome="cake_success", reason="eligible"),
            dict(mode="prefill", outcome="cake_success", reason="eligible"),
            *(
                dict(mode="decode", outcome="triton_fallback", reason=reason)
                for reason in decode_fallback_reasons
            ),
            *(
                dict(mode="prefill", outcome="triton_fallback", reason=reason)
                for reason in prefill_fallback_reasons
            ),
            dict(
                mode="decode",
                outcome="fatal",
                reason=PACKED_SELECTOR_EXCEPTION,
                eligible=False,
            ),
            dict(
                mode="decode",
                outcome="fatal",
                reason=TRITON_FALLBACK_EXCEPTION,
                eligible=False,
            ),
            dict(
                mode="decode",
                outcome="fatal",
                reason=CAKE_PACKED_EXCEPTION,
                eligible=True,
            ),
            dict(
                mode="prefill",
                outcome="fatal",
                reason=PREFILL_SELECTOR_EXCEPTION,
                eligible=False,
            ),
            dict(
                mode="prefill",
                outcome="fatal",
                reason=TRITON_FALLBACK_EXCEPTION,
                eligible=False,
            ),
            dict(
                mode="prefill",
                outcome="fatal",
                reason=CAKE_PREFILL_EXCEPTION,
                eligible=True,
                graph_phase="replay",
            ),
            dict(
                mode="decode",
                outcome="fatal",
                reason=CUDA_GRAPH_CAPTURE_EXCEPTION,
                eligible=True,
                graph_phase="capture",
            ),
            dict(
                mode="prefill",
                outcome="fatal",
                reason=CUDA_GRAPH_PLAN_BIND_EXCEPTION,
                eligible=False,
                graph_phase="capture",
            ),
            dict(
                mode="decode",
                outcome="fatal",
                reason=CUDA_GRAPH_REPLAY_EXCEPTION,
                eligible=False,
                graph_phase="replay",
            ),
            dict(
                mode="prefill",
                outcome="fatal",
                reason=CUDA_GRAPH_REPLAY_PLAN_MISMATCH,
                eligible=True,
                graph_phase="replay",
            ),
        ]

        for case in cases:
            with self.subTest(case=case):
                self.assertEqual(_event(**case).considered, 1)

    def test_schema_rejects_invalid_funnel_reason_phase_and_copy_states(self):
        fallback = _event(
            outcome="triton_fallback", reason=CakePackedDecodeReason.INNER_STRIDE
        )
        fatal = _event(
            outcome="fatal", reason=PACKED_SELECTOR_EXCEPTION, eligible=False
        )
        with self.assertRaisesRegex(ValueError, "Triton fallback"):
            replace(fallback, eligible=1, attempted_cake=1)
        with self.assertRaisesRegex(ValueError, "eligible must equal"):
            replace(fatal, eligible=1, attempted_cake=0)
        with self.assertRaisesRegex(ValueError, "eligible must equal"):
            replace(fatal, eligible=0, attempted_cake=1)
        with self.assertRaisesRegex(ValueError, "invalid KDA route reason"):
            replace(fallback, reason="arbitrary_reason")
        with self.assertRaisesRegex(ValueError, "invalid KDA route reason"):
            replace(fallback, reason=CakePrefillReason.SPEC_DECODE)
        with self.assertRaisesRegex(ValueError, "invalid KDA route reason"):
            replace(fallback, reason=CakePackedDecodeReason.ELIGIBLE)
        with self.assertRaisesRegex(ValueError, "requires graph_phase='capture'"):
            _event(
                outcome="fatal",
                reason=CUDA_GRAPH_CAPTURE_EXCEPTION,
                eligible=False,
                graph_phase="replay",
            )
        with self.assertRaisesRegex(ValueError, "requires graph_phase='replay'"):
            _event(
                outcome="fatal",
                reason=CUDA_GRAPH_REPLAY_EXCEPTION,
                eligible=False,
                graph_phase="capture",
            )
        with self.assertRaisesRegex(ValueError, "invalid KDA copy-count state"):
            _event(
                copy_count=7,
                copy_count_source="static_zero_copy_row_view",
            )
        with self.assertRaisesRegex(ValueError, "invalid KDA copy-count state"):
            _event(
                mode="prefill",
                copy_count=0,
                copy_count_source="static_zero_copy_row_view",
            )
        with self.assertRaisesRegex(ValueError, "invalid KDA copy-count state"):
            replace(
                fallback,
                copy_count=0,
                copy_count_source="static_zero_copy_row_view",
            )
        with self.assertRaisesRegex(ValueError, "invalid KDA copy-count state"):
            _event(copy_count=0, copy_count_source="unknown_requires_cupti")

    def test_capture_suppresses_warmups_and_emits_actual_capture_once(self):
        telemetry = _recorder()
        plans = KDACudaGraphRoutePlans()
        shape_key = ShapeKey(size=8)
        capture_state = {"active": False}

        with capture_kda_route_plan(
            shape_key,
            "decode",
            capture_probe=lambda: capture_state["active"],
            physical_capture_probe=lambda: True,
            telemetry=telemetry,
            plans=plans,
        ):
            warmup = _success(telemetry, layer_id=3)
            capture_state["active"] = True
            captured = _success(telemetry, layer_id=3)

        self.assertIsNone(warmup)
        self.assertIsNotNone(captured)
        events = telemetry.raw_events_snapshot()
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].graph_phase, "capture")
        self.assertEqual(len(plans.snapshot("decode", shape_key)), 1)

        # Recapturing the same immutable plan does not emit capture twice.
        with capture_kda_route_plan(
            shape_key,
            "decode",
            capture_probe=lambda: True,
            physical_capture_probe=lambda: True,
            telemetry=telemetry,
            plans=plans,
        ):
            _success(telemetry, layer_id=3)
        self.assertEqual(len(telemetry.raw_events_snapshot()), 1)

    def test_capture_and_changed_input_replay_exact_one_event_per_layer(self):
        telemetry = _recorder()
        plans = KDACudaGraphRoutePlans()
        shape_key = ShapeKey(size=31, variant_label="nolora")

        with capture_kda_route_plan(
            shape_key,
            "decode",
            capture_probe=lambda: True,
            physical_capture_probe=lambda: True,
            telemetry=telemetry,
            plans=plans,
        ):
            _success(telemetry, layer_id=3)
            _success(telemetry, layer_id=9)

        live_input = {"value": 11}

        def replay():
            return live_input["value"]

        self.assertEqual(
            replay_kda_route_plan(
                shape_key,
                "decode",
                replay,
                telemetry=telemetry,
                plans=plans,
            ),
            11,
        )
        live_input["value"] = 29
        self.assertEqual(
            replay_kda_route_plan(
                shape_key,
                "decode",
                replay,
                telemetry=telemetry,
                plans=plans,
            ),
            29,
        )

        events = telemetry.raw_events_snapshot()
        self.assertEqual(len(events), 6)
        for phase_offset in (0, 2, 4):
            phase_events = events[phase_offset : phase_offset + 2]
            self.assertEqual([event.layer_id for event in phase_events], [3, 9])
        self.assertEqual(
            [event.graph_phase for event in events],
            ["capture", "capture", "replay", "replay", "replay", "replay"],
        )
        self.assertEqual(
            [(event.reason, event.copy_count) for event in events[:2]],
            [(event.reason, event.copy_count) for event in events[2:4]],
        )

    def test_capture_exception_is_fatal_once_and_does_not_bind_plan(self):
        telemetry = _recorder()
        plans = KDACudaGraphRoutePlans()
        shape_key = ShapeKey(size=64)

        with self.assertRaisesRegex(RuntimeError, "capture failed"):
            with capture_kda_route_plan(
                shape_key,
                "decode",
                capture_probe=lambda: True,
                physical_capture_probe=lambda: True,
                telemetry=telemetry,
                plans=plans,
            ):
                _success(telemetry, layer_id=3)
                raise RuntimeError("capture failed")

        events = telemetry.raw_events_snapshot()
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].fatal, 1)
        self.assertEqual(events[0].reason, "cuda_graph_capture_exception")
        with self.assertRaises(KDACudaGraphPlanError):
            plans.require("decode", shape_key)

    def test_replay_ownership_change_rejects_immutable_plan_binding(self):
        telemetry = _recorder()
        plans = KDACudaGraphRoutePlans()
        shape_key = ShapeKey(size=65)
        with capture_kda_route_plan(
            shape_key,
            "decode",
            capture_probe=lambda: True,
            physical_capture_probe=lambda: True,
            telemetry=telemetry,
            plans=plans,
        ):
            _success(telemetry)

        with self.assertRaisesRegex(KDACudaGraphPlanError, "plan changed"):
            with capture_kda_route_plan(
                shape_key,
                "decode",
                capture_probe=lambda: True,
                physical_capture_probe=lambda: False,
                telemetry=telemetry,
                plans=plans,
            ):
                _success(telemetry)

        events = telemetry.raw_events_snapshot()
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0].cake_success, 1)
        self.assertEqual(events[1].fatal, 1)
        self.assertEqual(events[1].reason, CUDA_GRAPH_PLAN_BIND_EXCEPTION)

    def test_replay_exception_is_fatal_once_per_planned_layer(self):
        telemetry = _recorder()
        plans = KDACudaGraphRoutePlans()
        shape_key = ShapeKey(size=128)
        with capture_kda_route_plan(
            shape_key,
            "prefill",
            capture_probe=lambda: True,
            physical_capture_probe=lambda: True,
            telemetry=telemetry,
            plans=plans,
        ):
            _success(
                telemetry,
                mode="prefill",
                layer_id=2,
                copy_count=None,
                copy_count_source="unknown_requires_cupti",
            )
            _success(
                telemetry,
                mode="prefill",
                layer_id=6,
                copy_count=None,
                copy_count_source="unknown_requires_cupti",
            )

        def fail_replay():
            raise KeyError("synchronous replay failure")

        with self.assertRaises(KeyError):
            replay_kda_route_plan(
                shape_key,
                "prefill",
                fail_replay,
                telemetry=telemetry,
                plans=plans,
            )

        events = telemetry.raw_events_snapshot()
        self.assertEqual(len(events), 4)
        replay_events = events[2:]
        self.assertEqual([event.layer_id for event in replay_events], [2, 6])
        self.assertTrue(all(event.fatal == 1 for event in replay_events))
        self.assertTrue(
            all(
                event.reason == "cuda_graph_replay_exception" for event in replay_events
            )
        )

    def test_live_python_replay_commits_only_after_backend_returns(self):
        telemetry = _recorder()
        plans = KDACudaGraphRoutePlans()
        shape_key = ShapeKey(size=7)
        with capture_kda_route_plan(
            shape_key,
            "prefill",
            capture_probe=lambda: True,
            physical_capture_probe=lambda: False,
            telemetry=telemetry,
            plans=plans,
        ):
            _success(
                telemetry,
                mode="prefill",
                layer_id=2,
                copy_count=None,
                copy_count_source="unknown_requires_cupti",
            )

        def replay():
            _success(
                telemetry,
                mode="prefill",
                layer_id=2,
                copy_count=None,
                copy_count_source="unknown_requires_cupti",
            )
            self.assertEqual(len(telemetry.raw_events_snapshot()), 1)
            return "ok"

        result = replay_kda_route_plan(
            shape_key,
            "prefill",
            replay,
            telemetry=telemetry,
            plans=plans,
        )
        self.assertEqual(result, "ok")
        events = telemetry.raw_events_snapshot()
        self.assertEqual([event.graph_phase for event in events], ["capture", "replay"])
        self.assertTrue(all(event.cake_success for event in events))

    def test_live_python_success_is_fatalized_by_later_replay_failure(self):
        telemetry = _recorder()
        plans = KDACudaGraphRoutePlans()
        shape_key = ShapeKey(size=9)
        with capture_kda_route_plan(
            shape_key,
            "prefill",
            capture_probe=lambda: True,
            physical_capture_probe=lambda: False,
            telemetry=telemetry,
            plans=plans,
        ):
            _success(
                telemetry,
                mode="prefill",
                copy_count=None,
                copy_count_source="unknown_requires_cupti",
            )

        def replay_then_fail():
            _success(
                telemetry,
                mode="prefill",
                copy_count=None,
                copy_count_source="unknown_requires_cupti",
            )
            raise RuntimeError("bridge failed after eager KDA seam")

        with self.assertRaisesRegex(RuntimeError, "bridge failed"):
            replay_kda_route_plan(
                shape_key,
                "prefill",
                replay_then_fail,
                telemetry=telemetry,
                plans=plans,
            )
        replay_events = telemetry.raw_events_snapshot()[1:]
        self.assertEqual(len(replay_events), 1)
        self.assertEqual(replay_events[0].cake_success, 0)
        self.assertEqual(replay_events[0].fatal, 1)
        self.assertEqual(replay_events[0].reason, CUDA_GRAPH_REPLAY_EXCEPTION)

    def test_live_python_fatal_event_is_not_duplicated(self):
        telemetry = _recorder()
        plans = KDACudaGraphRoutePlans()
        shape_key = ShapeKey(size=10)
        with capture_kda_route_plan(
            shape_key,
            "prefill",
            capture_probe=lambda: True,
            physical_capture_probe=lambda: False,
            telemetry=telemetry,
            plans=plans,
        ):
            _success(
                telemetry,
                mode="prefill",
                copy_count=None,
                copy_count_source="unknown_requires_cupti",
            )

        def fatal_replay():
            record_kda_terminal_route(
                mode="prefill",
                layer_id=3,
                eligible=True,
                attempted_cake=True,
                cake_success=False,
                triton_fallback=False,
                fatal=True,
                reason=CAKE_PREFILL_EXCEPTION,
                detail="builtins.RuntimeError",
                telemetry=telemetry,
            )
            raise RuntimeError("CAKE failed")

        with self.assertRaisesRegex(RuntimeError, "CAKE failed"):
            replay_kda_route_plan(
                shape_key,
                "prefill",
                fatal_replay,
                telemetry=telemetry,
                plans=plans,
            )
        replay_events = telemetry.raw_events_snapshot()[1:]
        self.assertEqual(len(replay_events), 1)
        self.assertEqual(replay_events[0].reason, CAKE_PREFILL_EXCEPTION)
        self.assertEqual(replay_events[0].fatal, 1)

    def test_mixed_captured_and_live_replay_ownership_preserves_layer_order(self):
        telemetry = _recorder()
        plans = KDACudaGraphRoutePlans()
        shape_key = ShapeKey(size=11)
        physical = {"active": True}
        with capture_kda_route_plan(
            shape_key,
            "decode",
            capture_probe=lambda: True,
            physical_capture_probe=lambda: physical["active"],
            telemetry=telemetry,
            plans=plans,
        ):
            _success(telemetry, layer_id=1)
            physical["active"] = False
            _success(telemetry, layer_id=2)
            physical["active"] = True
            _success(telemetry, layer_id=3)

        replay_kda_route_plan(
            shape_key,
            "decode",
            lambda: _success(telemetry, layer_id=2),
            telemetry=telemetry,
            plans=plans,
        )
        events = telemetry.raw_events_snapshot()
        self.assertEqual([event.layer_id for event in events], [1, 2, 3, 1, 2, 3])
        self.assertEqual([event.graph_phase for event in events[3:]], ["replay"] * 3)

    def test_live_replay_plan_mismatch_fails_closed(self):
        telemetry = _recorder()
        plans = KDACudaGraphRoutePlans()
        shape_key = ShapeKey(size=12)
        with capture_kda_route_plan(
            shape_key,
            "decode",
            capture_probe=lambda: True,
            physical_capture_probe=lambda: False,
            telemetry=telemetry,
            plans=plans,
        ):
            _success(telemetry)

        def changed_route():
            return record_kda_terminal_route(
                mode="decode",
                layer_id=3,
                eligible=False,
                attempted_cake=False,
                cake_success=False,
                triton_fallback=True,
                fatal=False,
                reason=CakePackedDecodeReason.INNER_STRIDE,
                telemetry=telemetry,
            )

        with self.assertRaisesRegex(KDACudaGraphPlanError, "route changed"):
            replay_kda_route_plan(
                shape_key,
                "decode",
                changed_route,
                telemetry=telemetry,
                plans=plans,
            )
        replay_events = telemetry.raw_events_snapshot()[1:]
        self.assertEqual(len(replay_events), 1)
        self.assertEqual(replay_events[0].reason, CUDA_GRAPH_REPLAY_PLAN_MISMATCH)
        self.assertEqual(replay_events[0].fatal, 1)

    def test_explicit_warmup_suppression_is_host_only_and_scoped(self):
        telemetry = _recorder()
        with suppress_kda_route_recording():
            self.assertIsNone(_success(telemetry))
        event = _success(telemetry)
        self.assertIsNotNone(event)
        self.assertEqual(telemetry.raw_events_snapshot(), (event,))

    def test_missing_plan_fails_before_backend_replay(self):
        telemetry = _recorder()
        plans = KDACudaGraphRoutePlans()
        replay_calls = []

        with self.assertRaisesRegex(KDACudaGraphPlanError, "missing"):
            replay_kda_route_plan(
                ShapeKey(size=1),
                "decode",
                lambda: replay_calls.append(True),
                telemetry=telemetry,
                plans=plans,
            )
        self.assertEqual(replay_calls, [])
        self.assertEqual(telemetry.raw_events_snapshot(), ())

    def test_stable_event_and_counter_serialization(self):
        telemetry = _recorder()
        event = _success(telemetry, layer_id=17)
        expected = (
            '{"attempted_cake":1,"cake_success":1,"considered":1,'
            '"copy_count":0,"copy_count_source":"static_zero_copy_row_view",'
            '"detail":"","eligible":1,"fatal":0,"graph_phase":"direct",'
            '"layer_id":17,"mode":"decode","reason":"eligible",'
            '"schema_version":1,"triton_fallback":0}'
        )
        self.assertEqual(event.to_json(), expected)
        self.assertEqual(telemetry.raw_events_jsonl(), expected)
        expected_counters = (
            '{"counters":[{"attempted_cake":1,"cake_success":1,'
            '"considered":1,"eligible":1,"fatal":0,"graph_phase":"direct",'
            '"known_copy_calls":1,"known_copy_count":0,"layer_id":17,'
            '"mode":"decode","reason":"eligible","triton_fallback":0,'
            '"unknown_copy_calls":0}],"events_dropped_from_memory":0,'
            '"events_retained":1,"events_total":1,"key_fields":["mode",'
            '"layer_id","graph_phase","reason"],"schema_version":1}'
        )
        self.assertEqual(telemetry.counter_snapshot_json(), expected_counters)

    def test_machine_readable_json_log_emission(self):
        event_logger = Mock()
        telemetry = KDATerminalRouteTelemetry(
            raw_event_capacity=4,
            emit_log=True,
            event_logger=event_logger,
        )
        event = _success(telemetry, layer_id=1)
        event_logger.info.assert_called_once_with(
            "%s%s", KDA_ROUTE_EVENT_PREFIX, event.to_json()
        )


if __name__ == "__main__":
    unittest.main()
