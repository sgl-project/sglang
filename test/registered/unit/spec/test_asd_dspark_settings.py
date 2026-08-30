"""Unit tests for the DSpark ASD adapter settings and env parsing.

Covers the environment-variable contract of DSparkASDSettings: the
disabled-by-default arm switch, the calibration trace arm, and the
validation rules that keep accidental misconfiguration loud. The
optional-ASD-package degradation path is exercised only when the
research package is not installed (the default CI environment).
"""

import unittest

from sglang.srt.speculative.dspark_components.asd_dspark import (
    _DEFAULT_TRACE_CAPACITY,
    ASD_MODE_CALIBRATION,
    ASD_MODE_DISABLED,
    DSparkASDSettings,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestDisabledByDefault(CustomTestCase):
    def test_empty_env_is_disabled(self):
        settings = DSparkASDSettings.from_environ(environ={})
        self.assertEqual(settings.mode, ASD_MODE_DISABLED)
        self.assertFalse(settings.active)
        self.assertIsNone(settings.config)

    def test_enabled_zero_is_disabled(self):
        settings = DSparkASDSettings.from_environ(environ={"ASD_ENABLED": "0"})
        self.assertEqual(settings.mode, ASD_MODE_DISABLED)
        self.assertFalse(settings.active)


class TestCalibrationArm(CustomTestCase):
    def test_trace_flag_selects_calibration(self):
        settings = DSparkASDSettings.from_environ(
            environ={"SGLANG_DSPARK_ASD_CALIBRATION_TRACE": "1"}
        )
        self.assertEqual(settings.mode, ASD_MODE_CALIBRATION)
        self.assertTrue(settings.active)
        self.assertEqual(settings.trace_capacity, _DEFAULT_TRACE_CAPACITY)

    def test_trace_capacity_is_parsed(self):
        settings = DSparkASDSettings.from_environ(
            environ={
                "SGLANG_DSPARK_ASD_CALIBRATION_TRACE": "1",
                "SGLANG_DSPARK_ASD_TRACE_CAPACITY": "128",
            }
        )
        self.assertEqual(settings.trace_capacity, 128)

    def test_non_integer_capacity_raises(self):
        with self.assertRaises(ValueError):
            DSparkASDSettings.from_environ(
                environ={
                    "SGLANG_DSPARK_ASD_CALIBRATION_TRACE": "1",
                    "SGLANG_DSPARK_ASD_TRACE_CAPACITY": "not-a-number",
                }
            )

    def test_trace_conflicts_with_enabled(self):
        with self.assertRaises(ValueError):
            DSparkASDSettings.from_environ(
                environ={
                    "SGLANG_DSPARK_ASD_CALIBRATION_TRACE": "1",
                    "ASD_ENABLED": "1",
                }
            )


class TestEnabledArmValidation(CustomTestCase):
    def test_enabled_requires_exactly_one_config_source(self):
        # No config source at all.
        with self.assertRaises(ValueError):
            DSparkASDSettings.from_environ(environ={"ASD_ENABLED": "1"})

        # Both sources at once.
        with self.assertRaises(ValueError):
            DSparkASDSettings.from_environ(
                environ={
                    "ASD_ENABLED": "1",
                    "SGLANG_DSPARK_ASD_CONFIG_JSON": "{}",
                    "SGLANG_DSPARK_ASD_CONFIG_PATH": "config.json",
                }
            )

    def test_config_is_invalid_when_disabled(self):
        with self.assertRaises(ValueError):
            DSparkASDSettings.from_environ(
                environ={
                    "ASD_ENABLED": "0",
                    "SGLANG_DSPARK_ASD_CONFIG_JSON": "{}",
                }
            )

    def test_enabled_must_be_binary(self):
        with self.assertRaises(ValueError):
            DSparkASDSettings.from_environ(environ={"ASD_ENABLED": "true"})

    def test_trace_flag_must_be_binary(self):
        with self.assertRaises(ValueError):
            DSparkASDSettings.from_environ(
                environ={"SGLANG_DSPARK_ASD_CALIBRATION_TRACE": "yes"}
            )

    def test_legacy_mode_env_is_rejected(self):
        with self.assertRaises(ValueError):
            DSparkASDSettings.from_environ(
                environ={"SGLANG_DSPARK_ASD_MODE": "enabled"}
            )


class TestSettingsInvariants(CustomTestCase):
    def test_unknown_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            DSparkASDSettings(mode="bogus")

    def test_enabled_mode_requires_a_config(self):
        with self.assertRaises(ValueError):
            DSparkASDSettings(mode="enabled")

    def test_config_is_invalid_outside_enabled_mode(self):
        with self.assertRaises(ValueError):
            DSparkASDSettings(mode=ASD_MODE_DISABLED, config=object())

    def test_trace_capacity_must_be_a_positive_integer(self):
        with self.assertRaises(ValueError):
            DSparkASDSettings(trace_capacity=0)
        with self.assertRaises(ValueError):
            DSparkASDSettings(trace_capacity=True)


class TestOptionalPackageDegradation(CustomTestCase):
    """ASD_ENABLED=1 without the research package must fail loudly, not
    silently fall back or crash with an ImportError in the hot path."""

    def test_missing_package_raises_clear_runtime_error(self):
        from sglang.srt.speculative.dspark_components import asd_dspark

        if asd_dspark.DSparkASDConfig is not None:
            self.skipTest("ASD package installed; degradation path not exercised")

        with self.assertRaises(RuntimeError) as ctx:
            DSparkASDSettings.from_environ(
                environ={
                    "ASD_ENABLED": "1",
                    "SGLANG_DSPARK_ASD_CONFIG_JSON": '{"budget": 8}',
                }
            )
        self.assertIn("ASD package", str(ctx.exception))


if __name__ == "__main__":
    unittest.main(verbosity=3)
