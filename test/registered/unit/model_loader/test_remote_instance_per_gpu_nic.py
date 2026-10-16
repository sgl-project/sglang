import sys
import types
import unittest
from unittest.mock import MagicMock, patch

import sglang.srt.model_executor.model_runner_components.remote_instance_weight_transporter as transporter_mod
from sglang.srt.environ import envs
from sglang.srt.model_executor.model_runner_components.remote_instance_weight_transporter import (
    RemoteInstanceWeightTransporter,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestRemoteInstancePerGpuNic(CustomTestCase):
    """`init_engine()` should resolve the per-GPU NIC from MOONCAKE_DEVICE and
    pass it to `TransferEngine.initialize()` so each rank binds to its own NIC.
    """

    def _make_transporter(self, gpu_id: int) -> RemoteInstanceWeightTransporter:
        return RemoteInstanceWeightTransporter(
            server_args=object(),
            get_model=lambda: None,
            tp_rank=gpu_id,
            gpu_id=gpu_id,
        )

    def _run_init_engine(self, transporter: RemoteInstanceWeightTransporter):
        fake_engine = MagicMock()
        fake_engine.get_rpc_port.return_value = 12345
        fake_engine_cls = MagicMock(return_value=fake_engine)

        fake_mooncake = types.ModuleType("mooncake")
        fake_mooncake_engine = types.ModuleType("mooncake.engine")
        fake_mooncake_engine.TransferEngine = fake_engine_cls
        fake_mooncake.engine = fake_mooncake_engine

        with patch.dict(
            sys.modules,
            {"mooncake": fake_mooncake, "mooncake.engine": fake_mooncake_engine},
        ), patch.object(transporter_mod, "get_local_ip_auto", return_value="10.0.0.1"):
            transporter.init_engine()

        fake_engine.initialize.assert_called_once()
        # initialize(local_ip, "P2PHANDSHAKE", protocol, ib_device)
        return fake_engine.initialize.call_args.args[3]

    def test_per_gpu_mapping_resolves_rank_device(self):
        transporter = self._make_transporter(gpu_id=1)
        with envs.MOONCAKE_DEVICE.override('{"0": "mlx5_0", "1": "mlx5_1"}'):
            passed_device = self._run_init_engine(transporter)
        self.assertEqual(passed_device, "mlx5_1")

    def test_shared_device_passed_through(self):
        transporter = self._make_transporter(gpu_id=3)
        with envs.MOONCAKE_DEVICE.override("mlx5_0"):
            passed_device = self._run_init_engine(transporter)
        self.assertEqual(passed_device, "mlx5_0")


if __name__ == "__main__":
    unittest.main()
