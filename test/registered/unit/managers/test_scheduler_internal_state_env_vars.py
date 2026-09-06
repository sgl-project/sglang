import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.environ import envs
from sglang.srt.managers.io_struct import GetInternalStateReq
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.runtime_context import get_context

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestSchedulerInternalStateEnvVars(unittest.TestCase):
    def _get_internal_state(self) -> dict:
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.metrics_reporter = SimpleNamespace(
            last_gen_throughput=1.0,
            spec_total_num_forward_ct=0,
            spec_total_num_accept_tokens=0,
            step_time_dict={},
        )
        scheduler.tp_worker = SimpleNamespace(
            model_runner=SimpleNamespace(weight_load_mem_usage=1.0),
            graph_memory_usage=None,
        )
        scheduler.token_to_kv_pool_allocator = SimpleNamespace(
            get_kvcache=lambda: SimpleNamespace(mem_usage=3.0)
        )
        scheduler.startup_available_gpu_memory_gb = 4.0
        scheduler.startup_time = 1.0
        scheduler.max_total_num_tokens = 100
        scheduler.swa_tokens_per_layer = None
        scheduler.max_running_requests = 8
        scheduler.spec_algorithm = SimpleNamespace(
            is_none=lambda: True,
            is_dspark=lambda: False,
        )
        scheduler.draft_worker = None

        with get_context().override_server_args():
            output = scheduler.get_internal_state(recv_req=GetInternalStateReq())

        return output.internal_state

    def test_the_gate_is_declared_off(self):
        """Nothing is exposed unless an operator opts in, so the declared default is the safety net."""
        self.assertIs(envs.SGLANG_EXPOSE_OWN_ENV_VARS.default, False)

    def test_env_vars_absent_when_disabled(self):
        """The env_vars key must not exist at all when the gate is off."""
        with envs.SGLANG_EXPOSE_OWN_ENV_VARS.override(False):
            internal_state = self._get_internal_state()

        self.assertNotIn("env_vars", internal_state)

    def test_declared_env_vars_exposed_when_enabled(self):
        """Enabling the gate exposes the declared, non secret environment of the scheduler."""
        with envs.SGLANG_EXPOSE_OWN_ENV_VARS.override(True):
            with patch.dict(
                "os.environ",
                {"SGLANG_LOG_SCHEDULER_STATUS_TARGET": "some-value"},
            ):
                internal_state = self._get_internal_state()

        self.assertIn("env_vars", internal_state)
        self.assertEqual(
            internal_state["env_vars"]["SGLANG_LOG_SCHEDULER_STATUS_TARGET"],
            "some-value",
        )

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
                internal_state = self._get_internal_state()

        exported = internal_state["env_vars"]
        self.assertNotIn("SGLANG_S3_SECRET_ACCESS_KEY", exported)
        self.assertNotIn("SGLANG_DIFFUSION_SLACK_TOKEN", exported)
        self.assertNotIn("a-cloud-credential", json.dumps(exported))

    def test_a_field_marked_secret_is_never_exposed(self):
        """A declared credential is still a credential, so the declaration has to be able to say so."""
        with envs.SGLANG_EXPOSE_OWN_ENV_VARS.override(True):
            with patch.dict("os.environ", {"EXA_API_KEY": "a-credential"}):
                internal_state = self._get_internal_state()

        self.assertNotIn("EXA_API_KEY", internal_state["env_vars"])

    def test_a_non_utf8_value_is_encoded_rather_than_dropped(self):
        """An undecodable byte in one variable must not cost the whole response its json encoding."""
        with envs.SGLANG_EXPOSE_OWN_ENV_VARS.override(True):
            with patch.dict(
                "os.environ", {"SGLANG_LOG_SCHEDULER_STATUS_TARGET": "bad-\udcff"}
            ):
                internal_state = self._get_internal_state()

        exported = internal_state["env_vars"]["SGLANG_LOG_SCHEDULER_STATUS_TARGET"]
        self.assertTrue(exported.startswith("base64:"))
        json.dumps(internal_state["env_vars"]).encode()


if __name__ == "__main__":
    unittest.main()
