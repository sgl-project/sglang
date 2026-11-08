"""A Mooncake KV manager fails loudly when the engine rejects its buffers."""

import os
import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager
from sglang.srt.distributed.device_communicators.mooncake_transfer_engine import (
    MooncakeTransferEngine,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

DEVICE = "mlx5_1"
# Mooncake's ERR_CONTEXT, returned when ibv_reg_mr fails on a device.
ERR_CONTEXT = -202
# Mooncake's ERR_ADDRESS_OVERLAPPED, returned for an already registered region.
ERR_ADDRESS_OVERLAPPED = -7


class _FakeTransferEngine:
    """Stands in for mooncake.engine.TransferEngine."""

    def __init__(self):
        self.batch_register_result = 0
        self.registered = []

    def initialize(self, hostname, metadata_server, protocol, device_name):
        return 0

    def get_rpc_port(self):
        return 15000

    def batch_register_memory(self, ptrs, lengths):
        if isinstance(self.batch_register_result, Exception):
            raise self.batch_register_result
        if self.batch_register_result == 0:
            self.registered.extend(zip(ptrs, lengths))
        return self.batch_register_result


def _engine() -> MooncakeTransferEngine:
    mooncake_engine = ModuleType("mooncake.engine")
    mooncake_engine.TransferEngine = _FakeTransferEngine
    modules = {"mooncake": ModuleType("mooncake"), "mooncake.engine": mooncake_engine}
    with patch.dict(sys.modules, modules), patch.dict(os.environ):
        os.environ.pop("MC_FORCE_TCP", None)
        return MooncakeTransferEngine(hostname="127.0.0.1", gpu_id=0, ib_device=DEVICE)


def _manager(engine: MooncakeTransferEngine) -> MooncakeKVManager:
    manager = MooncakeKVManager.__new__(MooncakeKVManager)
    manager.engine = engine
    manager.kv_args = SimpleNamespace(
        kv_data_ptrs=[0x10000, 0x20000],
        kv_data_lens=[0x1000, 0x1000],
        aux_data_ptrs=[0x40000],
        aux_data_lens=[0x100],
        state_data_ptrs=[],
        state_data_lens=[],
    )
    return manager


class TestMooncakeRegisterBuffers(CustomTestCase):
    def test_rejected_registration_raises(self):
        engine = _engine()
        engine.engine.batch_register_result = ERR_CONTEXT

        with self.assertRaises(RuntimeError) as ctx:
            _manager(engine).register_buffer_to_engine()

        message = str(ctx.exception)
        self.assertIn(f"ret={ERR_CONTEXT}", message)
        self.assertIn(repr(DEVICE), message)
        self.assertIn("3 memory regions", message)
        self.assertIn("[0x10000, 0x40100)", message)

    def test_engine_exception_raises(self):
        engine = _engine()
        engine.engine.batch_register_result = OSError("ibv_reg_mr")

        with self.assertRaises(RuntimeError) as ctx:
            _manager(engine).register_buffer_to_engine()

        self.assertIn(repr(DEVICE), str(ctx.exception))
        self.assertIsInstance(ctx.exception.__cause__, OSError)

    def test_already_registered_is_not_fatal(self):
        # A resume after a failed deregister re-registers live regions.
        engine = _engine()
        engine.engine.batch_register_result = ERR_ADDRESS_OVERLAPPED

        with self.assertLogs(level="ERROR"):
            _manager(engine).register_buffer_to_engine()

    def test_accepted_registration_registers_every_buffer(self):
        engine = _engine()

        _manager(engine).register_buffer_to_engine()

        self.assertEqual(
            engine.engine.registered,
            [(0x10000, 0x1000), (0x20000, 0x1000), (0x40000, 0x100)],
        )


if __name__ == "__main__":
    unittest.main()
