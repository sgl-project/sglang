import threading
import unittest
from unittest.mock import MagicMock, patch

from sglang.srt import platforms
from sglang.srt.distributed.device_communicators.mooncake_transfer_engine import (
    MooncakeTransferEngine,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestMooncakeTransferEngineDevice(CustomTestCase):
    def test_initialize_selects_device_in_deadline_thread(self):
        calls = []
        platform = MagicMock()
        platform.get_device.side_effect = lambda gpu_id: f"cuda:{gpu_id}"
        platform.set_device.side_effect = lambda device: calls.append(
            ("set_device", device, threading.get_ident())
        )

        def initialize(*args):
            calls.append(("initialize", args[2], threading.get_ident()))
            return 0

        engine = object.__new__(MooncakeTransferEngine)
        engine.gpu_id = 3
        engine.engine = MagicMock(initialize=initialize)

        with patch.object(platforms, "_current_platform", platform):
            engine.initialize(hostname="127.0.0.1", device_name="")

        init_thread = calls[-1][2]
        self.assertNotEqual(init_thread, threading.get_ident())
        self.assertEqual(
            calls,
            [
                ("set_device", "cuda:3", init_thread),
                ("initialize", "rdma", init_thread),
            ],
        )


if __name__ == "__main__":
    unittest.main()
