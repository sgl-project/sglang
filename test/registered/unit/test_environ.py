"""Unit tests for sglang.srt.environ: EnvField semantics and the deprecated-env registry."""

import os
import re
import subprocess
import sys
import unittest
import warnings
from contextlib import ExitStack

from sglang.srt.environ import _DEPRECATED_ENVS, _DeprecatedEnv, envs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


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

    def test_w4a4_mxfp4_megamoe_envs_warn_to_use_cli_flag(self):
        old_names = (
            "SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_FP4_ACTS",
            "SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_MXF4_KIND",
        )
        for old_name in old_names:
            with self.subTest(old_name=old_name):
                os.environ[old_name] = "1"
                self.addCleanup(os.environ.pop, old_name, None)

                caught = self._apply(old_name, _DEPRECATED_ENVS[old_name])

                self.assertIn("--enable-w4a4-mxfp4-megamoe", str(caught[0].message))
                self.assertIsNone(_DEPRECATED_ENVS[old_name].replacement)

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

    def test_disable_tp_imbalance_check_polarity_is_inverted(self):
        old_name = "SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK"
        new_name = "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK"
        os.environ[old_name] = "1"
        self.addCleanup(os.environ.pop, old_name, None)
        self.addCleanup(os.environ.pop, new_name, None)

        self._apply(old_name, _DEPRECATED_ENVS[old_name])
        self.assertIs(envs.SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK.get(), False)

    def test_ms_to_s_transform(self):
        old_name = "SGLANG_QUEUED_TIMEOUT_MS"
        os.environ[old_name] = "1500"
        self.addCleanup(os.environ.pop, old_name, None)
        self.addCleanup(os.environ.pop, "SGLANG_REQ_WAITING_TIMEOUT", None)

        self._apply(old_name, _DEPRECATED_ENVS[old_name])
        self.assertEqual(envs.SGLANG_REQ_WAITING_TIMEOUT.get(), 1.5)


if __name__ == "__main__":
    unittest.main()
