import json
import os
import tempfile
import unittest
from unittest.mock import patch

from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-b-test-cpu")


class TestServerArgsCPUBackend(unittest.TestCase):
    def _make_server_args(self, attention_backend=None):
        server_args = ServerArgs.__new__(ServerArgs)
        server_args.device = "cpu"
        server_args.attention_backend = attention_backend
        server_args.sampling_backend = None
        return server_args

    @patch("sglang.srt.server_args.is_host_cpu_arm64", return_value=True)
    def test_arm_cpu_defaults_to_torch_native(self, _mock_is_arm64):
        server_args = self._make_server_args()

        ServerArgs._handle_cpu_backends(server_args)

        self.assertEqual(server_args.attention_backend, "torch_native")
        self.assertEqual(server_args.sampling_backend, "pytorch")

    @patch("sglang.srt.server_args.is_host_cpu_arm64", return_value=False)
    def test_x86_cpu_defaults_to_intel_amx(self, _mock_is_arm64):
        server_args = self._make_server_args()

        ServerArgs._handle_cpu_backends(server_args)

        self.assertEqual(server_args.attention_backend, "intel_amx")
        self.assertEqual(server_args.sampling_backend, "pytorch")


class TestServerArgsIBDeviceValidation(unittest.TestCase):
    def _validate_ib_devices(self, device_str, available_devices=None):
        server_args = ServerArgs.__new__(ServerArgs)
        available_devices = available_devices or [
            "mlx5_0",
            "mlx5_1",
            "mlx5_2",
            "mlx5_3",
        ]
        real_isdir = os.path.isdir
        real_listdir = os.listdir

        with patch(
            "sglang.srt.server_args.os.path.isdir",
            side_effect=lambda path: (
                True if path == "/sys/class/infiniband" else real_isdir(path)
            ),
        ), patch(
            "sglang.srt.server_args.os.listdir",
            side_effect=lambda path: (
                available_devices
                if path == "/sys/class/infiniband"
                else real_listdir(path)
            ),
        ):
            return ServerArgs._validate_ib_devices(server_args, device_str)

    def test_validate_ib_devices_accepts_comma_separated(self):
        self.assertEqual(
            self._validate_ib_devices("mlx5_0, mlx5_1"),
            "mlx5_0,mlx5_1",
        )

    def test_validate_ib_devices_accepts_json_object(self):
        result = self._validate_ib_devices(
            '{"0": "mlx5_0, mlx5_1", "1": "mlx5_2, mlx5_3"}'
        )
        self.assertEqual(
            json.loads(result),
            {"0": "mlx5_0,mlx5_1", "1": "mlx5_2,mlx5_3"},
        )

    def test_validate_ib_devices_accepts_json_file(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as file:
            json.dump({"0": "mlx5_0, mlx5_1", "1": "mlx5_2"}, file)
            json_file = file.name

        try:
            result = self._validate_ib_devices(json_file)
        finally:
            os.unlink(json_file)

        self.assertEqual(
            json.loads(result),
            {"0": "mlx5_0,mlx5_1", "1": "mlx5_2"},
        )


if __name__ == "__main__":
    unittest.main()
