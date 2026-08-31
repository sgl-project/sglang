import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.model_executor.model_runner_components import (
    remote_instance_weight_transporter as transporter_module,
)
from sglang.srt.model_executor.model_runner_components.remote_instance_weight_transporter import (
    RemoteInstanceWeightTransporter,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestRemoteInstanceWeightTransporter(unittest.TestCase):
    def test_registration_publishes_global_rank(self):
        response = SimpleNamespace(status_code=200, text="OK")
        with (
            patch.object(
                transporter_module,
                "get_parallel",
                return_value=SimpleNamespace(dist_init_addr=None),
            ),
            patch.object(
                transporter_module,
                "get_model",
                return_value=SimpleNamespace(engine_info_bootstrap_port=12345),
            ),
            patch("requests.put", return_value=response) as put,
        ):
            transporter = RemoteInstanceWeightTransporter(
                get_model=lambda: None,
                rank=3,
                gpu_id=0,
                session_id="pp1-tp1-session",
                weight_info={"model.weight": (1, 2, 4)},
            )

            transporter._register_to_engine_info_bootstrap()

        payload = put.call_args.kwargs["json"]
        self.assertEqual(payload["rank"], 3)
        self.assertNotIn("tp_rank", payload)
        self.assertEqual(
            payload["transfer_engine_info"]["session_id"],
            "pp1-tp1-session",
        )


if __name__ == "__main__":
    unittest.main()
