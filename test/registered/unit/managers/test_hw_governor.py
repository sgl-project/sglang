"""Hardware governor: policy semantics, knob resolution, and the leader-to-rank advice handoff."""

import unittest
from typing import List

from sglang.srt.managers.hw_governor.governor import HardwareGovernor
from sglang.srt.managers.hw_governor.knobs import KnobOverrides
from sglang.srt.managers.hw_governor.policy import (
    GovernorDecision,
    GovernorPolicy,
    StepwiseConfig,
    StepwisePolicy,
    StepwiseState,
    TripRule,
    Verdict,
    combined_verdict,
    next_stepwise_state,
    resolve_policy_cls,
    rule_verdict,
)
from sglang.srt.managers.hw_governor.telemetry import (
    ClockEventReason,
    DeviceTelemetry,
    TelemetrySource,
    resolve_telemetry_source_cls,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _sample(**fields) -> DeviceTelemetry:
    return DeviceTelemetry(device_uuid="GPU-0", timestamp=0.0, **fields)


class _FakeSource(TelemetrySource):
    def __init__(self, *, device_uuids, samples=None, error=None):
        super().__init__(device_uuids=device_uuids)
        self.samples: List[DeviceTelemetry] = samples or []
        self.error = error

    def sample(self) -> List[DeviceTelemetry]:
        if self.error is not None:
            raise self.error
        return self.samples


class _AlwaysLevelOne(GovernorPolicy):
    def update(self, samples, *, now):
        return GovernorDecision(
            level=1,
            overrides=KnobOverrides(
                chunked_prefill_size=0.5, prefill_decode_interval=3
            ),
        )


def _leader(source, policy) -> HardwareGovernor:
    # A long interval keeps the background thread idle; tests drive sample_once().
    return HardwareGovernor(
        page_size=1, source=source, policy=policy, interval_s=3600.0
    )


class TestTripRules(CustomTestCase):
    def test_rule_direction_follows_trip_clear_order(self):
        rising = TripRule(metric="gpu_temp_c", trip=80.0, clear=75.0)
        falling = TripRule(metric="temp_margin_c", trip=5.0, clear=10.0)
        cases = [
            (rising, _sample(gpu_temp_c=80.0), Verdict.HOT),
            (rising, _sample(gpu_temp_c=77.0), Verdict.HOLD),
            (rising, _sample(gpu_temp_c=75.0), Verdict.COOL),
            (falling, _sample(temp_margin_c=5.0), Verdict.HOT),
            (falling, _sample(temp_margin_c=7.0), Verdict.HOLD),
            (falling, _sample(temp_margin_c=10.0), Verdict.COOL),
        ]
        for rule, sample, expected in cases:
            with self.subTest(metric=rule.metric, sample=sample):
                self.assertEqual(rule_verdict(rule, sample), expected)

    def test_missing_telemetry_releases_the_throttle(self):
        rule = TripRule(metric="gpu_temp_c", trip=80.0, clear=75.0)
        self.assertEqual(rule_verdict(rule, _sample()), Verdict.COOL)
        self.assertEqual(combined_verdict([rule], []), Verdict.COOL)

    def test_worst_device_wins(self):
        rule = TripRule(metric="gpu_temp_c", trip=80.0, clear=75.0)
        samples = [_sample(gpu_temp_c=60.0), _sample(gpu_temp_c=85.0)]
        self.assertEqual(combined_verdict([rule], samples), Verdict.HOT)

    def test_default_rules_ignore_routine_power_capping(self):
        rules = StepwiseConfig().rules
        power_capped = _sample(clock_event_reasons=int(ClockEventReason.SW_POWER_CAP))
        hw_slowdown = _sample(clock_event_reasons=int(ClockEventReason.HW_SLOWDOWN))
        self.assertEqual(combined_verdict(rules, [power_capped]), Verdict.COOL)
        self.assertEqual(combined_verdict(rules, [hw_slowdown]), Verdict.HOT)

    def test_clock_event_bits_match_nvml(self):
        import pynvml

        for reason, nvml_name in [
            (ClockEventReason.GPU_IDLE, "nvmlClocksEventReasonGpuIdle"),
            (ClockEventReason.SW_POWER_CAP, "nvmlClocksEventReasonSwPowerCap"),
            (ClockEventReason.HW_SLOWDOWN, "nvmlClocksEventReasonHwSlowdown"),
            (ClockEventReason.SYNC_BOOST, "nvmlClocksEventReasonSyncBoost"),
            (
                ClockEventReason.SW_THERMAL_SLOWDOWN,
                "nvmlClocksEventReasonSwThermalSlowdown",
            ),
            (
                ClockEventReason.HW_THERMAL_SLOWDOWN,
                "nvmlClocksEventReasonHwThermalSlowdown",
            ),
            (
                ClockEventReason.HW_POWER_BRAKE_SLOWDOWN,
                "nvmlClocksEventReasonHwPowerBrakeSlowdown",
            ),
        ]:
            with self.subTest(reason=reason.name):
                self.assertEqual(int(reason), getattr(pynvml, nvml_name))


class TestStepwiseState(CustomTestCase):
    def setUp(self):
        self.config = StepwiseConfig(
            rules=(TripRule(metric="gpu_temp_c", trip=80.0, clear=75.0),),
            levels=(
                KnobOverrides(max_prefill_tokens=0.5),
                KnobOverrides(max_prefill_tokens=0.25),
            ),
            escalate_after_s=3.0,
            recover_after_s=10.0,
        )

    def _run(self, state, steps):
        levels = []
        for now, verdict in steps:
            state = next_stepwise_state(state, verdict, now=now, config=self.config)
            levels.append(state.level)
        return state, levels

    def test_escalates_one_level_per_sustained_hot_streak(self):
        hot = Verdict.HOT
        _, levels = self._run(
            StepwiseState(),
            [(0.0, hot), (2.9, hot), (3.0, hot), (5.9, hot), (6.0, hot), (20.0, hot)],
        )
        self.assertEqual(levels, [0, 0, 1, 1, 2, 2])

    def test_reading_inside_band_restarts_the_streak(self):
        _, levels = self._run(
            StepwiseState(),
            [
                (0.0, Verdict.HOT),
                (2.0, Verdict.HOLD),
                (2.5, Verdict.HOT),
                (5.0, Verdict.HOT),
                (5.5, Verdict.HOT),
            ],
        )
        self.assertEqual(levels, [0, 0, 0, 0, 1])

    def test_recovers_one_level_per_cool_streak(self):
        cool = Verdict.COOL
        _, levels = self._run(
            StepwiseState(level=2),
            [(0.0, cool), (9.0, cool), (10.0, cool), (20.0, cool), (40.0, cool)],
        )
        self.assertEqual(levels, [2, 2, 1, 0, 0])


class TestStepwiseConfig(CustomTestCase):
    def test_invalid_configs_are_rejected(self):
        bad_configs = {
            "fraction above 1": {"levels": [{"max_prefill_tokens": 2048}]},
            "no hysteresis band": {
                "rules": [{"metric": "gpu_temp_c", "trip": 80, "clear": 80}]
            },
            "unknown metric": {"rules": [{"metric": "fan_rpm", "trip": 1, "clear": 0}]},
            "unknown reason": {
                "rules": [{"metric": "clock_event_reasons", "any_of": ["sw_power"]}]
            },
            "misspelled key": {"level": [{"max_prefill_tokens": 0.5}]},
            "misspelled knob": {"levels": [{"chunk_prefill_size": 0.5}]},
        }
        for name, config in bad_configs.items():
            with self.subTest(name), self.assertRaises(ValueError):
                StepwisePolicy(config)

    def test_partial_config_keeps_default_levels(self):
        policy = StepwisePolicy(
            {
                "rules": [{"metric": "power_ratio", "trip": 0.9, "clear": 0.8}],
                "escalate_after_s": 0,
            }
        )
        decision = policy.update([_sample(power_w=300.0, power_limit_w=320.0)], now=0.0)
        self.assertEqual(decision.level, 1)
        self.assertEqual(decision.overrides, StepwiseConfig().levels[0])


class TestKnobOverrides(CustomTestCase):
    def test_token_budget_aligns_down_and_floors_at_one_unit(self):
        cases = [
            # (configured, fraction, page_size, expected)
            (8192, 0.3, 1, 2432),  # 2457 rounded down to a multiple of 64
            (8192, 0.3, 256, 2304),  # page_size above 64 sets the alignment
            (8192, 0.001, 1, 64),  # never below one alignment unit
            (32, 0.5, 1, 32),  # a budget below one unit is left alone
        ]
        for configured, fraction, page_size, expected in cases:
            with self.subTest(
                configured=configured, fraction=fraction, page_size=page_size
            ):
                overrides = KnobOverrides(
                    max_prefill_tokens=fraction, chunked_prefill_size=fraction
                )
                self.assertEqual(
                    overrides.resolve_max_prefill_tokens(
                        configured, page_size=page_size
                    ),
                    expected,
                )
                self.assertEqual(
                    overrides.resolve_chunked_prefill_size(
                        configured, page_size=page_size
                    ),
                    expected,
                )

    def test_overrides_never_loosen_a_knob(self):
        overrides = KnobOverrides(
            chunked_prefill_size=0.5,
            max_running_requests=0.01,
            prefill_decode_interval=2,
        )
        self.assertIsNone(overrides.resolve_chunked_prefill_size(None, page_size=1))
        self.assertEqual(overrides.resolve_max_running_requests(8), 1)
        self.assertEqual(overrides.resolve_prefill_decode_interval(4), 4)
        self.assertEqual(overrides.resolve_prefill_decode_interval(0), 2)


class TestHardwareGovernor(CustomTestCase):
    def test_knobs_change_only_when_advice_is_applied(self):
        leader = _leader(_FakeSource(device_uuids=["GPU-0"]), _AlwaysLevelOne(None))
        follower = HardwareGovernor(
            page_size=1, source=None, policy=None, interval_s=3600.0
        )
        try:
            leader.sample_once()
            # The leader has a new decision, but no rank resolves knobs from it yet.
            self.assertEqual(leader.chunked_prefill_size(8192), 8192)
            [req] = leader.poll_advice()
            # Unapplied advice is resent: a receive pass may have dropped it.
            self.assertEqual(leader.poll_advice(), [req])
            for governor in (leader, follower):
                governor.apply_advice(req)
            self.assertEqual(leader.poll_advice(), [])
            self.assertEqual(follower.poll_advice(), [])
            for governor in (leader, follower):
                self.assertEqual(governor.chunked_prefill_size(8192), 4096)
                self.assertEqual(governor.prefill_decode_interval(0), 3)
        finally:
            leader.shutdown()

    def test_failed_telemetry_read_falls_back_to_cool(self):
        policy = StepwisePolicy({"escalate_after_s": 0, "recover_after_s": 0})
        source = _FakeSource(
            device_uuids=["GPU-0"],
            samples=[_sample(clock_event_reasons=int(ClockEventReason.HW_SLOWDOWN))],
        )
        governor = _leader(source, policy)
        try:
            governor.sample_once()
            [req] = governor.poll_advice()
            governor.apply_advice(req)
            self.assertEqual(governor.level, 1)
            source.error = RuntimeError("NVML went away")
            governor.sample_once()
            [req] = governor.poll_advice()
            self.assertEqual(req.decision.level, 0)
        finally:
            governor.shutdown()


class TestPluginResolution(CustomTestCase):
    def test_builtin_names_and_import_paths(self):
        from sglang.srt.managers.hw_governor.nvml_source import NvmlTelemetrySource

        self.assertIs(resolve_telemetry_source_cls("nvml"), NvmlTelemetrySource)
        self.assertIs(resolve_policy_cls("stepwise"), StepwisePolicy)
        self.assertIs(
            resolve_policy_cls("sglang.srt.managers.hw_governor.policy:StepwisePolicy"),
            StepwisePolicy,
        )
        with self.assertRaises(ValueError):
            resolve_telemetry_source_cls("dcgm")
        with self.assertRaises(TypeError):
            resolve_policy_cls("sglang.srt.managers.hw_governor.knobs:KnobOverrides")


if __name__ == "__main__":
    unittest.main()
