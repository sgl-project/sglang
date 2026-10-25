"""Unit tests for sglang.srt.environ: EnvField semantics and the deprecated-env registry."""

import os
import re
import subprocess
import sys
import unittest
import warnings
from contextlib import ExitStack

from sglang.srt.environ import _DeprecatedEnv, envs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")


class TestEnvField(unittest.TestCase):
    def setUp(self):
        envs.SGLANG_TEST_RETRACT.clear()
        self.addCleanup(envs.SGLANG_TEST_RETRACT.clear)

    def test_set_get_clear_is_set(self):
        self.assertFalse(envs.SGLANG_TEST_RETRACT.is_set())
        self.assertIs(envs.SGLANG_TEST_RETRACT.get(), False)

        envs.SGLANG_TEST_RETRACT.set(True)
        self.assertTrue(envs.SGLANG_TEST_RETRACT.is_set())
        self.assertIs(envs.SGLANG_TEST_RETRACT.get(), True)

        envs.SGLANG_TEST_RETRACT.clear()
        self.assertFalse(envs.SGLANG_TEST_RETRACT.is_set())
        self.assertIs(envs.SGLANG_TEST_RETRACT.get(), False)

    def test_set_to_none_is_distinct_from_clear(self):
        envs.SGLANG_TEST_RETRACT.set(None)
        self.assertTrue(envs.SGLANG_TEST_RETRACT.is_set())
        self.assertIsNone(envs.SGLANG_TEST_RETRACT.get())

        envs.SGLANG_TEST_RETRACT.clear()
        self.assertFalse(envs.SGLANG_TEST_RETRACT.is_set())
        self.assertIs(envs.SGLANG_TEST_RETRACT.get(), False)

    def test_override_restores_previous_state(self):
        envs.SGLANG_TEST_RETRACT.set(True)
        with envs.SGLANG_TEST_RETRACT.override(None):
            self.assertTrue(envs.SGLANG_TEST_RETRACT.is_set())
            self.assertIsNone(envs.SGLANG_TEST_RETRACT.get())
        self.assertIs(envs.SGLANG_TEST_RETRACT.get(), True)

        envs.SGLANG_TEST_RETRACT.set(None)
        with envs.SGLANG_TEST_RETRACT.override(True):
            self.assertIs(envs.SGLANG_TEST_RETRACT.get(), True)
        self.assertTrue(envs.SGLANG_TEST_RETRACT.is_set())
        self.assertIsNone(envs.SGLANG_TEST_RETRACT.get())

    def test_override_with_exit_stack(self):
        envs.SGLANG_TEST_RETRACT.set(None)
        exit_stack = ExitStack()
        exit_stack.enter_context(envs.SGLANG_TEST_RETRACT.override(False))
        self.assertIs(envs.SGLANG_TEST_RETRACT.get(), False)
        exit_stack.close()
        self.assertIsNone(envs.SGLANG_TEST_RETRACT.get())

    def test_override_is_inherited_by_subprocess(self):
        command = [
            sys.executable,
            "-c",
            "import os; print(os.getenv('SGLANG_TEST_RETRACT'))",
        ]
        with envs.SGLANG_TEST_RETRACT.override(True):
            output = subprocess.check_output(command).decode().strip()
            self.assertEqual(output, "True")

        output = subprocess.check_output(command).decode().strip()
        self.assertEqual(output, "None")

    def test_implicit_bool_raises(self):
        message = re.escape(
            "Please use `envs.YOUR_FLAG.get()` instead of `envs.YOUR_FLAG`"
        )

        with self.assertRaisesRegex(RuntimeError, message):
            if envs.SGLANG_TEST_RETRACT:
                pass

        with self.assertRaisesRegex(RuntimeError, message):
            if (1 != 1) or envs.SGLANG_TEST_RETRACT:
                pass

        with self.assertRaisesRegex(RuntimeError, message):
            if envs.SGLANG_TEST_RETRACT or (1 == 1):
                pass

    def test_invalid_value_warns_and_returns_default(self):
        os.environ["SGLANG_TEST_RETRACT"] = "not-a-bool"
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.assertIs(envs.SGLANG_TEST_RETRACT.get(), False)
        self.assertIn("Invalid value", str(caught[0].message))


class TestEnvBoolOrPhase(unittest.TestCase):
    def test_per_token_flag_uses_the_same_phase_contract(self):
        field = envs.SGLANG_FLASHINFER_NVFP4_PER_TOKEN_ACTIVATION
        for value in ("true", "1", "prefill", "decode", "false", "0"):
            with self.subTest(value=value), field.override(value):
                for phase in ("prefill", "decode"):
                    self.assertEqual(
                        field.enabled_for(phase), value in ("true", "1", phase)
                    )

    def test_boolean_spellings_keep_global_semantics(self):
        field = envs.SGLANG_FLASHINFER_CUTEDSL_NVFP4_W4A16
        for value in ("true", "1", "yes", "y", "all", "TRUE", True):
            with self.subTest(value=value), field.override(value):
                self.assertTrue(field.enabled_for_all())
                self.assertTrue(field.enabled_for("prefill"))
                self.assertTrue(field.enabled_for("decode"))
        for value in ("false", "0", "no", "n", False):
            with self.subTest(value=value), field.override(value):
                self.assertFalse(field.enabled_for_any())
                self.assertFalse(field.enabled_for("prefill"))
                self.assertFalse(field.enabled_for("decode"))

    def test_phase_selection(self):
        field = envs.SGLANG_FLASHINFER_CUTEDSL_NVFP4_W4A16
        for phase in ("prefill", "decode"):
            with self.subTest(phase=phase), field.override(phase.upper()):
                self.assertTrue(field.enabled_for_any())
                self.assertFalse(field.enabled_for_all())
                self.assertTrue(field.enabled_for(phase))
                other = "decode" if phase == "prefill" else "prefill"
                self.assertFalse(field.enabled_for(other))

    def test_invalid_values_are_not_truthy_opt_ins(self):
        field = envs.SGLANG_FLASHINFER_CUTEDSL_NVFP4_W4A16
        with (
            field.override("invalid-phase"),
            self.assertWarnsRegex(
                UserWarning, 'must be a boolean, "all", "prefill", or "decode"'
            ),
        ):
            self.assertFalse(field.enabled_for_any())
        with self.assertRaisesRegex(ValueError, "Unknown inference phase"):
            field.enabled_for("extend")


class TestDeprecatedEnvRegistry(unittest.TestCase):
    def _apply(self, old_name, deprecation):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            deprecation.apply(old_name)
        return caught

    def test_removed_env_warns_without_forwarding(self):
        old_name = "SGLANG_TEST_REMOVED_ENV"
        os.environ[old_name] = "1"
        self.addCleanup(os.environ.pop, old_name, None)

        caught = self._apply(old_name, _DeprecatedEnv())
        self.assertIn(f"{old_name} is deprecated", str(caught[0].message))

    def test_renamed_env_forwards_value(self):
        old_name, new_name = "SGLANG_TEST_OLD_ENV", "SGLANG_TEST_NEW_ENV"
        os.environ[old_name] = "abc"
        self.addCleanup(os.environ.pop, old_name, None)
        self.addCleanup(os.environ.pop, new_name, None)

        caught = self._apply(old_name, _DeprecatedEnv(replacement=new_name))
        self.assertIn(new_name, str(caught[0].message))
        self.assertEqual(os.environ[new_name], "abc")

    def test_unset_env_is_a_no_op(self):
        caught = self._apply("SGLANG_TEST_UNSET_ENV", _DeprecatedEnv())
        self.assertEqual(len(caught), 0)

    def test_renamed_env_applies_transform(self):
        old_name, new_name = "SGLANG_TEST_OLD_ENV", "SGLANG_TEST_NEW_ENV"
        os.environ[old_name] = "abc"
        self.addCleanup(os.environ.pop, old_name, None)
        self.addCleanup(os.environ.pop, new_name, None)

        self._apply(old_name, _DeprecatedEnv(replacement=new_name, transform=str.upper))
        self.assertEqual(os.environ[new_name], "ABC")


if __name__ == "__main__":
    unittest.main()
