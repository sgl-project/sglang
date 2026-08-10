"""Pure CPU tests for Phase-A KDA terminal route telemetry."""

import unittest
from dataclasses import FrozenInstanceError
from unittest.mock import Mock

from sglang.srt.layers.attention.linear.kda_route_telemetry import (
    KDA_ROUTE_EVENT_PREFIX,
    KDACudaGraphPlanError,
    KDACudaGraphRoutePlans,
    KDATerminalRouteTelemetry,
    capture_kda_route_plan,
    record_kda_terminal_route,
    replay_kda_route_plan,
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

    def test_capture_suppresses_warmups_and_emits_actual_capture_once(self):
        telemetry = _recorder()
        plans = KDACudaGraphRoutePlans()
        shape_key = ShapeKey(size=8)
        capture_state = {"active": False}

        with capture_kda_route_plan(
            shape_key,
            "decode",
            capture_probe=lambda: capture_state["active"],
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

    def test_replay_exception_is_fatal_once_per_planned_layer(self):
        telemetry = _recorder()
        plans = KDACudaGraphRoutePlans()
        shape_key = ShapeKey(size=128)
        with capture_kda_route_plan(
            shape_key,
            "prefill",
            capture_probe=lambda: True,
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
        self.assertEqual(
            telemetry.counter_snapshot_json(), telemetry.counter_snapshot_json()
        )

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
