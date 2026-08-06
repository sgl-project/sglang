import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _make_manager(register_ret=0, deregister_ret=0):
    """A MooncakeKVManager with only the pieces these methods touch."""
    manager = MooncakeKVManager.__new__(MooncakeKVManager)
    manager.engine = MagicMock()
    manager.engine.batch_register.return_value = register_ret
    manager.engine.batch_deregister.return_value = deregister_ret
    manager.kv_args = SimpleNamespace(
        kv_data_ptrs=[0x1000, 0x2000],
        kv_data_lens=[4096, 8192],
        aux_data_ptrs=[0x3000],
        aux_data_lens=[256],
        state_data_ptrs=[[0x4000]],
        state_data_lens=[[512]],
    )
    return manager


class TestMooncakeRegisterStatus(unittest.TestCase):
    def test_successful_registration_registers_every_buffer_class(self):
        manager = _make_manager()
        manager.register_buffer_to_engine()
        self.assertEqual(manager.engine.batch_register.call_count, 3)

    def test_failed_kv_registration_raises(self):
        manager = _make_manager(register_ret=-1)
        with self.assertRaises(RuntimeError) as ctx:
            manager.register_buffer_to_engine()
        message = str(ctx.exception)
        self.assertIn("KV data buffers", message)
        self.assertIn("2 regions", message)
        self.assertIn("12288 bytes", message)
        self.assertIn("-1", message)

    def test_failed_registration_stops_before_the_next_buffer_class(self):
        manager = _make_manager(register_ret=-1)
        with self.assertRaises(RuntimeError):
            manager.register_buffer_to_engine()
        self.assertEqual(manager.engine.batch_register.call_count, 1)

    def test_failed_staging_registration_raises(self):
        manager = _make_manager(register_ret=1)
        with self.assertRaises(RuntimeError) as ctx:
            manager._register_staging_memory(0x5000, 1024, "prefill staging buffer")
        self.assertIn("prefill staging buffer", str(ctx.exception))

    def test_failed_deregistration_only_warns(self):
        manager = _make_manager(deregister_ret=-1)
        with self.assertLogs(
            "sglang.srt.disaggregation.mooncake.conn", level="WARNING"
        ) as logs:
            manager._batch_deregister_logged([0x1000], "KV data buffers")
        self.assertTrue(
            any("deregistration failed" in line for line in logs.output), logs.output
        )


if __name__ == "__main__":
    unittest.main()
