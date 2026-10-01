import json
import unittest
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.environ import envs, exportable_env_vars

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestSchedulerInternalStateEnvVars(unittest.TestCase):
    def test_the_gate_is_declared_off(self):
        """Nothing is exposed unless an operator opts in, so the declared default is the safety net."""
        self.assertIs(envs.SGLANG_EXPOSE_OWN_ENV_VARS.default, False)

    def test_env_vars_absent_when_disabled(self):
        """The env_vars key must not exist at all when the gate is off."""
        with envs.SGLANG_EXPOSE_OWN_ENV_VARS.override(False):
            self.assertFalse(envs.SGLANG_EXPOSE_OWN_ENV_VARS.get())

    def test_declared_env_vars_exposed_when_enabled(self):
        """Enabling the gate exposes the declared, non secret environment of the scheduler."""
        with envs.SGLANG_EXPOSE_OWN_ENV_VARS.override(True):
            with patch.dict(
                "os.environ",
                {"SGLANG_LOG_SCHEDULER_STATUS_TARGET": "some-value"},
            ):
                exported = exportable_env_vars()

        self.assertEqual(exported["SGLANG_LOG_SCHEDULER_STATUS_TARGET"], "some-value")

    def test_undeclared_env_vars_never_exposed(self):
        """Only what Envs declares is auditable; the SGLANG_ namespace also holds real credentials."""
        with envs.SGLANG_EXPOSE_OWN_ENV_VARS.override(True):
            with patch.dict(
                "os.environ",
                {
                    "SGLANG_LOG_SCHEDULER_STATUS_TARGET": "some-value",
                    "SGLANG_S3_SECRET_ACCESS_KEY": "a-cloud-credential",
                    "SGLANG_DIFFUSION_SLACK_TOKEN": "a-slack-token",
                },
            ):
                exported = exportable_env_vars()

        self.assertNotIn("SGLANG_S3_SECRET_ACCESS_KEY", exported)
        self.assertNotIn("SGLANG_DIFFUSION_SLACK_TOKEN", exported)
        self.assertNotIn("a-cloud-credential", json.dumps(exported))

    def test_a_field_marked_secret_is_never_exposed(self):
        """A declared credential is still a credential, so the declaration has to be able to say so."""
        with envs.SGLANG_EXPOSE_OWN_ENV_VARS.override(True):
            with patch.dict("os.environ", {"EXA_API_KEY": "a-credential"}):
                exported = exportable_env_vars()

        self.assertNotIn("EXA_API_KEY", exported)

    def test_a_non_utf8_value_is_encoded_rather_than_dropped(self):
        """An undecodable byte in one variable must not cost the whole response its json encoding."""
        with envs.SGLANG_EXPOSE_OWN_ENV_VARS.override(True):
            with patch.dict(
                "os.environ", {"SGLANG_LOG_SCHEDULER_STATUS_TARGET": "bad-\udcff"}
            ):
                exported = exportable_env_vars()

        value = exported["SGLANG_LOG_SCHEDULER_STATUS_TARGET"]
        self.assertTrue(value.startswith("base64:"))
        json.dumps(exported).encode()


if __name__ == "__main__":
    unittest.main()
