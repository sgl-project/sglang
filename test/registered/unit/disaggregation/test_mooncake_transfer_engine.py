import sys
import types
import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.srt.distributed.device_communicators import mooncake_transfer_engine as mte

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")


class _FakeTransferEngine:
    def __init__(self):
        self.initialize_calls = []

    def initialize(self, hostname, handshake_type, transport, device_name):
        self.initialize_calls.append(
            (hostname, handshake_type, transport, device_name)
        )
        return 0

    def get_rpc_port(self):
        return 12345


class TestMooncakeTransferEngine(unittest.TestCase):
    def setUp(self):
        self._old_engine = mte._mooncake_transfer_engine
        mte._mooncake_transfer_engine = None

    def tearDown(self):
        mte._mooncake_transfer_engine = self._old_engine

    def _patch_mooncake_engine(self):
        fake_engine_module = types.ModuleType("mooncake.engine")
        fake_engine_module.TransferEngine = _FakeTransferEngine
        fake_mooncake_module = types.ModuleType("mooncake")
        fake_mooncake_module.engine = fake_engine_module
        return unittest.mock.patch.dict(
            sys.modules,
            {
                "mooncake": fake_mooncake_module,
                "mooncake.engine": fake_engine_module,
            },
        )

    def test_transport_defaults_to_rdma(self):
        with self._patch_mooncake_engine():
            engine = mte.MooncakeTransferEngine(hostname="127.0.0.1")

        self.assertEqual(engine.transport, "rdma")
        self.assertEqual(
            engine.engine.initialize_calls,
            [("127.0.0.1", "P2PHANDSHAKE", "rdma", "")],
        )

    def test_transport_is_passed_to_initialize(self):
        with self._patch_mooncake_engine():
            engine = mte.MooncakeTransferEngine(
                hostname="127.0.0.1",
                transport="efa",
            )

        self.assertEqual(engine.transport, "efa")
        self.assertEqual(
            engine.engine.initialize_calls,
            [("127.0.0.1", "P2PHANDSHAKE", "efa", "")],
        )


if __name__ == "__main__":
    unittest.main()